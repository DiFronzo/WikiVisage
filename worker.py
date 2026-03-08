import io
import json
import logging
import multiprocessing
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
warnings.filterwarnings("ignore", message=".*pkg_resources.*", category=UserWarning)

import face_recognition
import numpy as np
import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from database import close_pool, execute_query, init_db, DatabaseError

# Configure Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("worker")

# Constants
COMMONS_API_URL = "https://commons.wikimedia.org/w/api.php"
SPARQL_URL = "https://commons-query.wikimedia.org/sparql"  # Requires OAuth; unused, kept for reference
FILE_PATH_URL = (
    "https://commons.wikimedia.org/wiki/Special:FilePath/{file_title}?width=1024"
)

USER_AGENT = "WikiVisage/1.0 (Wikimedia Toolforge; https://toolsadmin.wikimedia.org)"

POLL_INTERVAL = int(os.environ.get("WIKIVISAGE_WORKER_POLL_INTERVAL", 60))
BATCH_SIZE = 10
WAKE_UP_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".worker-wake-up"
)
MAX_CONCURRENT_PROJECTS = int(os.environ.get("WIKIVISAGE_WORKER_MAX_PROJECTS", 3))
IMAGE_THREADS = int(os.environ.get("WIKIVISAGE_WORKER_IMAGE_THREADS", 4))

# Maximum image download size (50 MB) — prevents OOM on abnormally large files
MAX_IMAGE_DOWNLOAD_BYTES = 50 * 1024 * 1024

# Maximum image pixel area (100 megapixels) — prevents OOM in face detection subprocess
MAX_IMAGE_PIXELS = 100_000_000

# Maximum images per project — stops category traversal once this limit is reached
MAX_IMAGES_PER_PROJECT = 9000

# Global Shutdown Flag
shutdown_requested = False


def _get_session() -> requests.Session:
    """Create a requests session with exponential backoff retries."""
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    retries = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS", "POST"],
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _api_request(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    method: str = "get",
    data: Optional[Dict[str, Any]] = None,
    timeout: int = 30,
) -> requests.Response:
    """Central HTTP request function with maxlag handling."""
    session = _get_session()

    headers = headers or {}
    if "User-Agent" not in headers:
        headers["User-Agent"] = USER_AGENT

    for _ in range(3):
        if shutdown_requested:
            raise InterruptedError("Worker shutting down")

        try:
            if method.lower() == "get":
                resp = session.get(url, params=params, headers=headers, timeout=timeout)
            else:
                resp = session.post(
                    url, data=data, params=params, headers=headers, timeout=timeout
                )

            resp.raise_for_status()

            # Check for MediaWiki Maxlag
            if "Retry-After" in resp.headers and resp.status_code == 200:
                try:
                    data_json = resp.json()
                    if (
                        "error" in data_json
                        and data_json["error"].get("code") == "maxlag"
                    ):
                        retry_after = int(resp.headers.get("Retry-After", 5))
                        logger.warning(
                            f"Maxlag encountered. Sleeping for {retry_after} seconds."
                        )
                        time.sleep(retry_after)
                        continue
                except ValueError:
                    pass

            return resp
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP Request failed: {e}")
            time.sleep(5)

    raise Exception(f"Failed to execute API request after 3 attempts: {url}")


def _download_image(url: str, max_bytes: int = MAX_IMAGE_DOWNLOAD_BYTES) -> bytes:
    """Download an image with streaming size cap to prevent OOM.

    Uses _get_session() for connection pooling / retry. Raises ValueError
    if the response exceeds max_bytes.
    """
    session = _get_session()
    resp = session.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=30,
        stream=True,
    )
    resp.raise_for_status()

    # Check Content-Length header first (fast reject)
    content_length = resp.headers.get("Content-Length")
    if content_length and int(content_length) > max_bytes:
        resp.close()
        raise ValueError(
            f"Image too large: {int(content_length)} bytes (limit {max_bytes})"
        )

    # Stream with enforced cap
    chunks: list[bytes] = []
    downloaded = 0
    for chunk in resp.iter_content(chunk_size=65536):
        downloaded += len(chunk)
        if downloaded > max_bytes:
            resp.close()
            raise ValueError(f"Image download exceeded {max_bytes} bytes limit")
        chunks.append(chunk)

    return b"".join(chunks)


def _validate_image_dimensions(image_bytes: bytes) -> None:
    """Check image dimensions via PIL header-only read. Raises ValueError if too large."""
    img = Image.open(io.BytesIO(image_bytes))
    w, h = img.size
    if w * h > MAX_IMAGE_PIXELS:
        raise ValueError(
            f"Image too large for face detection: {w}x{h} "
            f"({w * h:,} pixels, limit {MAX_IMAGE_PIXELS:,})"
        )


def _get_csrf_token(access_token: str) -> str:
    """Fetch CSRF token from MediaWiki API."""
    params = {"action": "query", "meta": "tokens", "type": "csrf", "format": "json"}
    headers = {"Authorization": f"Bearer {access_token}"}

    try:
        resp = _api_request(COMMONS_API_URL, params=params, headers=headers)
        data = resp.json()
        return data["query"]["tokens"]["csrftoken"]
    except Exception as e:
        logger.error(f"Failed to get CSRF token: {e}")
        raise


def traverse_category(project: Dict[str, Any]) -> int:
    """Fetch all files in the project's Commons category (including subcategories) and insert them as pending."""
    logger.info(f"Traversing category for project {project['id']}")

    # Check how many images this project already has
    existing_count_rows = execute_query(
        "SELECT COUNT(*) AS cnt FROM images WHERE project_id = %s",
        (project["id"],),
        fetch=True,
    )
    existing_count = existing_count_rows[0]["cnt"] if existing_count_rows else 0
    if existing_count >= MAX_IMAGES_PER_PROJECT:
        logger.info(
            f"Project {project['id']} already has {existing_count} images "
            f"(limit {MAX_IMAGES_PER_PROJECT}), skipping traversal."
        )
        return 0

    remaining = MAX_IMAGES_PER_PROJECT - existing_count

    root_category = project["commons_category"]
    if not root_category.startswith("Category:"):
        root_category = f"Category:{root_category}"

    added_count = 0
    visited_cats = set()
    cat_queue = [root_category]
    max_categories = 200

    while cat_queue and not shutdown_requested:
        if added_count >= remaining:
            logger.info(
                f"Reached image limit ({MAX_IMAGES_PER_PROJECT}) for project "
                f"{project['id']}, stopping traversal."
            )
            break
        if len(visited_cats) >= max_categories:
            logger.info(
                f"Reached category limit ({max_categories}), stopping traversal. "
                f"{len(cat_queue)} subcategories skipped."
            )
            break

        category = cat_queue.pop(0)
        if category in visited_cats:
            continue
        visited_cats.add(category)

        # Fetch both files and subcategories
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category,
            "cmtype": "file|subcat",
            "cmlimit": "500",
            "format": "json",
        }

        cmcontinue = None

        while not shutdown_requested:
            if cmcontinue:
                params["cmcontinue"] = cmcontinue

            try:
                resp = _api_request(COMMONS_API_URL, params=params)
                data = resp.json()

                members = data.get("query", {}).get("categorymembers", [])
                new_files = []  # Collect file tuples for batch insert
                for member in members:
                    ns = member.get("ns", 0)

                    if ns == 14:
                        # Subcategory — queue for traversal
                        subcat_title = member["title"]
                        if subcat_title not in visited_cats:
                            cat_queue.append(subcat_title)
                    elif ns == 6:
                        # File — collect for batch insert
                        new_files.append(
                            (project["id"], member["pageid"], member["title"])
                        )

                # Trim batch to stay within per-project image limit
                space_left = remaining - added_count
                if len(new_files) > space_left:
                    new_files = new_files[:space_left]

                # Batch insert all files from this API page (UNIQUE index handles duplicates)
                if new_files:
                    placeholders = ", ".join(
                        ["(%s, %s, %s, 'pending')"] * len(new_files)
                    )
                    flat_params = [val for tup in new_files for val in tup]
                    result = execute_query(
                        f"INSERT IGNORE INTO images (project_id, commons_page_id, file_title, status) "
                        f"VALUES {placeholders}",
                        tuple(flat_params),
                        fetch=False,
                    )
                    added_count += result if isinstance(result, int) else 0

                if added_count >= remaining:
                    break

                if "continue" in data and "cmcontinue" in data["continue"]:
                    cmcontinue = data["continue"]["cmcontinue"]
                    time.sleep(1)  # Respect rate limits
                else:
                    break

            except Exception as e:
                logger.error(f"Category traversal failed for {category}: {e}")
                break

        logger.info(
            f"Traversed {category}: {added_count} new files so far, "
            f"{len(visited_cats)} categories visited, {len(cat_queue)} queued"
        )

    # Update total image count
    execute_query(
        "UPDATE projects SET images_total = (SELECT COUNT(*) FROM images WHERE project_id = %s) WHERE id = %s",
        (project["id"], project["id"]),
        fetch=False,
    )

    return added_count


FACE_DETECT_TIMEOUT = 120  # seconds — kill subprocess if face detection hangs


def _detect_faces_in_subprocess(
    image_bytes: bytes,
    conn: Any,
) -> None:
    """Run face detection in an isolated subprocess. Sends results back via pipe.

    This isolates dlib's C++ code so a segfault kills only this subprocess,
    not the parent worker. Results are sent as (locations, encodings_bytes) or
    an error string.
    """
    try:
        image_data = face_recognition.load_image_file(io.BytesIO(image_bytes))
        face_locations = face_recognition.face_locations(image_data, model="hog")
        face_encodings = face_recognition.face_encodings(image_data, face_locations)

        encodings_as_bytes = [enc.tobytes() for enc in face_encodings]
        conn.send(("ok", face_locations, encodings_as_bytes))
    except Exception as e:
        conn.send(("error", str(e)))
    finally:
        conn.close()


def _run_face_detection(image_bytes: bytes) -> Tuple[List, List[bytes]]:
    """Run face detection in a subprocess with timeout. Returns (locations, encoding_bytes_list).

    Raises RuntimeError if subprocess crashes (segfault) or times out.
    """
    parent_conn, child_conn = multiprocessing.Pipe(duplex=False)
    proc = multiprocessing.Process(
        target=_detect_faces_in_subprocess,
        args=(image_bytes, child_conn),
        daemon=True,
    )
    proc.start()
    child_conn.close()

    if parent_conn.poll(FACE_DETECT_TIMEOUT):
        result = parent_conn.recv()
        parent_conn.close()
        proc.join(timeout=5)
    else:
        parent_conn.close()
        proc.kill()
        proc.join(timeout=5)
        raise RuntimeError(f"Face detection timed out after {FACE_DETECT_TIMEOUT}s")

    if result[0] == "error":
        raise RuntimeError(f"Face detection failed: {result[1]}")
    if result[0] != "ok":
        raise RuntimeError(f"Unexpected subprocess result: {result[0]}")

    exitcode = proc.exitcode
    if exitcode is not None and exitcode < 0:
        raise RuntimeError(f"Face detection subprocess crashed with signal {-exitcode}")

    return result[1], result[2]


def _process_single_image(img_id: int, title: str) -> bool:
    """Download one image, detect faces in subprocess, store encodings. Returns True on success."""
    clean_title = title[5:] if title.startswith("File:") else title
    url = FILE_PATH_URL.format(file_title=clean_title)

    try:
        logger.debug(f"Downloading image {title}")
        image_bytes = _download_image(url)
        _validate_image_dimensions(image_bytes)

        face_locations, encodings_bytes = _run_face_detection(image_bytes)
        del image_bytes

        face_count = len(face_locations)

        for location, encoding_bytes in zip(face_locations, encodings_bytes):
            top, right, bottom, left = location
            execute_query(
                """
                INSERT INTO faces (image_id, encoding, bbox_top, bbox_right, bbox_bottom, bbox_left)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (img_id, encoding_bytes, top, right, bottom, left),
                fetch=False,
            )

        execute_query(
            "UPDATE images SET status = 'processed', face_count = %s WHERE id = %s",
            (face_count, img_id),
            fetch=False,
        )

        return True

    except Exception as e:
        logger.error(f"Error processing image {title}: {e}")
        error_msg = str(e)[:1000]
        execute_query(
            "UPDATE images SET status = 'error', error_message = %s WHERE id = %s",
            (error_msg, img_id),
            fetch=False,
        )
        return False


def process_images(project: Dict[str, Any]) -> int:
    """Download images, detect faces, and extract embeddings using parallel threads."""
    logger.info(f"Processing images for project {project['id']}")

    pending_images = execute_query(
        "SELECT id, file_title FROM images WHERE project_id = %s AND status = 'pending' LIMIT %s",
        (project["id"], BATCH_SIZE),
        fetch=True,
    )

    if not pending_images:
        return 0

    processed_count = 0

    with ThreadPoolExecutor(max_workers=IMAGE_THREADS) as executor:
        futures = {}
        for img_row in pending_images:
            if shutdown_requested:
                break
            future = executor.submit(
                _process_single_image, img_row["id"], img_row["file_title"]
            )
            futures[future] = img_row["file_title"]

        for future in as_completed(futures):
            if shutdown_requested:
                executor.shutdown(wait=False, cancel_futures=True)
                break
            try:
                if future.result():
                    processed_count += 1
            except Exception as e:
                logger.error(f"Thread error processing {futures[future]}: {e}")

    # Update project stats
    execute_query(
        "UPDATE projects SET images_processed = (SELECT COUNT(*) FROM images WHERE project_id = %s AND status IN ('processed', 'error', 'enriched')) WHERE id = %s",
        (project["id"], project["id"]),
        fetch=False,
    )

    return processed_count


def bootstrap_from_sparql(project: Dict[str, Any]) -> int:
    """Bootstrap a project using images already linked to the QID via P180.

    Uses the Wikibase haswbstatement search API on Commons instead of SPARQL,
    since commons-query.wikimedia.org requires OAuth cookie authentication
    that is impractical for automated tools.
    """
    logger.info(
        f"Attempting bootstrap for project {project['id']} with QID {project['wikidata_qid']}"
    )

    # Check if project already has confirmed target faces (use actual count, not
    # the unreliable faces_confirmed counter)
    count_row = execute_query(
        "SELECT COUNT(*) AS cnt FROM faces f "
        "JOIN images i ON f.image_id = i.id "
        "WHERE i.project_id = %s AND f.is_target = 1",
        (project["id"],),
    )
    if count_row and count_row[0]["cnt"] > 0:
        return 0

    qid = project["wikidata_qid"]
    search_params: Dict[str, Any] = {
        "action": "query",
        "list": "search",
        "srsearch": f"haswbstatement:P180={qid}",
        "srnamespace": "6",
        "srlimit": "50",
        "format": "json",
    }

    try:
        bootstrapped_count = 0

        while True:
            if shutdown_requested:
                break

            resp = _api_request(COMMONS_API_URL, params=search_params)
            data = resp.json()
            results = data.get("query", {}).get("search", [])

            for result in results:
                if shutdown_requested:
                    break

                page_id = result.get("pageid")
                title = result.get("title")

                if not page_id or not title:
                    continue

                exists = execute_query(
                    "SELECT id, status FROM images WHERE project_id = %s AND commons_page_id = %s",
                    (project["id"], page_id),
                    fetch=True,
                )

                img_id = None
                if exists:
                    img_id = exists[0]["id"]
                    if exists[0]["status"] != "pending":
                        continue
                else:
                    execute_query(
                        """
                        INSERT IGNORE INTO images (project_id, commons_page_id, file_title, status)
                        VALUES (%s, %s, %s, 'pending')
                        """,
                        (project["id"], page_id, title),
                        fetch=False,
                    )
                    img_id_res = execute_query(
                        "SELECT id FROM images WHERE project_id = %s AND commons_page_id = %s",
                        (project["id"], page_id),
                        fetch=True,
                    )
                    if img_id_res:
                        img_id = img_id_res[0]["id"]

                if not img_id:
                    continue

                clean_title = title[5:] if title.startswith("File:") else title
                url = FILE_PATH_URL.format(file_title=clean_title)

                try:
                    image_bytes = _download_image(url)
                    _validate_image_dimensions(image_bytes)
                    face_locations, encodings_bytes = _run_face_detection(image_bytes)
                    del image_bytes

                    face_count = len(face_locations)

                    if face_count == 1:
                        top, right, bottom, left = face_locations[0]

                        execute_query(
                            """
                            INSERT INTO faces (image_id, encoding, bbox_top, bbox_right, bbox_bottom, bbox_left, is_target, classified_by)
                            VALUES (%s, %s, %s, %s, %s, %s, 1, 'bootstrap')
                            """,
                            (img_id, encodings_bytes[0], top, right, bottom, left),
                            fetch=False,
                        )
                        bootstrapped_count += 1
                    else:
                        for location, enc_bytes in zip(face_locations, encodings_bytes):
                            top, right, bottom, left = location
                            execute_query(
                                """
                                INSERT INTO faces (image_id, encoding, bbox_top, bbox_right, bbox_bottom, bbox_left)
                                VALUES (%s, %s, %s, %s, %s, %s)
                                """,
                                (img_id, enc_bytes, top, right, bottom, left),
                                fetch=False,
                            )

                    execute_query(
                        "UPDATE images SET status = 'processed', face_count = %s WHERE id = %s",
                        (face_count, img_id),
                        fetch=False,
                    )

                except Exception as e:
                    logger.error(f"Bootstrap image processing error for {title}: {e}")
                    execute_query(
                        "UPDATE images SET status = 'error', error_message = %s WHERE id = %s",
                        (str(e)[:1000], img_id),
                        fetch=False,
                    )

            # Paginate: check for continuation token
            continuation = data.get("continue", {})
            sr_offset = continuation.get("sroffset")
            if sr_offset is None:
                break
            search_params["sroffset"] = sr_offset
            logger.info(f"Bootstrap pagination: fetching from offset {sr_offset}")

        if bootstrapped_count > 0:
            execute_query(
                "UPDATE projects SET faces_confirmed = faces_confirmed + %s, images_processed = (SELECT COUNT(*) FROM images WHERE project_id = %s AND status IN ('processed', 'error', 'enriched')) WHERE id = %s",
                (bootstrapped_count, project["id"], project["id"]),
                fetch=False,
            )
            return bootstrapped_count

    except Exception as e:
        logger.error(f"SPARQL Bootstrap failed: {e}")

    return 0


def run_autonomous_inference(project: Dict[str, Any]) -> int:
    """Run model inference on unclassified faces based on confirmed faces."""
    min_confirmed = project.get("min_confirmed", 5)

    # Count actual is_target=1 faces instead of trusting faces_confirmed counter
    count_row = execute_query(
        "SELECT COUNT(*) AS cnt FROM faces f "
        "JOIN images i ON f.image_id = i.id "
        "WHERE i.project_id = %s AND f.is_target = 1",
        (project["id"],),
        fetch=True,
    )
    actual_confirmed = count_row[0]["cnt"] if count_row else 0

    if actual_confirmed < min_confirmed:
        logger.info(
            f"Project {project['id']}: {actual_confirmed} confirmed target faces < {min_confirmed} required, skipping inference"
        )
        return 0

    logger.info(
        f"Running autonomous inference for project {project['id']} ({actual_confirmed} confirmed faces)"
    )

    # Get confirmed faces
    confirmed_rows = execute_query(
        """
        SELECT f.encoding FROM faces f
        JOIN images i ON f.image_id = i.id
        WHERE i.project_id = %s AND f.is_target = 1
        """,
        (project["id"],),
        fetch=True,
    )

    if not confirmed_rows:
        return 0

    # Load all confirmed encodings to compute centroid. Each encoding is 128 float64
    # values (1024 bytes). Even 10K confirmed faces = ~10 MB — well within Toolforge
    # memory limits. Batching would add complexity with no practical benefit.
    confirmed_encodings = [
        np.frombuffer(row["encoding"], dtype=np.float64) for row in confirmed_rows
    ]
    centroid = np.mean(confirmed_encodings, axis=0)

    # Process all unclassified faces (inference is fast numpy math — single distance
    # computation per face). Same memory analysis applies: 10K faces ≈ 10 MB.
    unclassified_rows = execute_query(
        """
        SELECT f.id, f.encoding FROM faces f
        JOIN images i ON f.image_id = i.id
        WHERE i.project_id = %s AND f.is_target IS NULL
        """,
        (project["id"],),
        fetch=True,
    )

    classified_count = 0
    threshold = project.get("distance_threshold", 0.6)

    for row in unclassified_rows:
        if shutdown_requested:
            break

        encoding = np.frombuffer(row["encoding"], dtype=np.float64)
        distance = face_recognition.face_distance([centroid], encoding)[0]

        is_target = 1 if distance < threshold else 0

        execute_query(
            "UPDATE faces SET is_target = %s, confidence = %s, classified_by = 'model' WHERE id = %s",
            (is_target, float(distance), row["id"]),
            fetch=False,
        )
        classified_count += 1

    return classified_count


def write_sdc_claims(project: Dict[str, Any]) -> int:
    """Write SDC P180 claims for all faces marked as target.

    Called when sdc_write_requested=1. Processes all pending faces in batches,
    clears the flag when done (or sets sdc_write_error on failure).
    """
    project_id = project["id"]
    logger.info(f"Starting SDC writes for project {project_id}")

    # Get user token
    user_row = execute_query(
        "SELECT access_token FROM users WHERE id = %s",
        (project["user_id"],),
        fetch=True,
    )

    if not user_row:
        logger.error(f"User {project['user_id']} not found for SDC writes")
        execute_query(
            "UPDATE projects SET sdc_write_requested = 0, "
            "sdc_write_error = 'User not found' WHERE id = %s",
            (project_id,),
            fetch=False,
        )
        return 0

    access_token = user_row[0]["access_token"]
    if isinstance(access_token, bytes):
        access_token = access_token.decode("utf-8")

    try:
        csrf_token = _get_csrf_token(access_token)
    except Exception as e:
        logger.error(f"Failed to get CSRF token for project {project_id}: {e}")
        execute_query(
            "UPDATE projects SET sdc_write_requested = 0, "
            "sdc_write_error = 'Failed to get CSRF token' WHERE id = %s",
            (project_id,),
            fetch=False,
        )
        return 0

    qid = project["wikidata_qid"]
    try:
        numeric_id = int(qid[1:])
    except ValueError:
        logger.error(f"Invalid QID format: {qid}")
        execute_query(
            "UPDATE projects SET sdc_write_requested = 0, "
            "sdc_write_error = 'Invalid QID format' WHERE id = %s",
            (project_id,),
            fetch=False,
        )
        return 0

    total_written = 0
    SDC_BATCH = 50  # Faces per DB fetch batch

    while not shutdown_requested:
        # Fetch next batch of unwritten faces
        faces_to_write = execute_query(
            """
            SELECT f.id as face_id, i.commons_page_id
            FROM faces f
            JOIN images i ON f.image_id = i.id
            WHERE i.project_id = %s AND f.is_target = 1 AND f.sdc_written = 0
            LIMIT %s
            """,
            (project_id, SDC_BATCH),
            fetch=True,
        )

        if (
            not faces_to_write
            or not isinstance(faces_to_write, list)
            or len(faces_to_write) == 0
        ):
            break

        for row in faces_to_write:
            if shutdown_requested:
                break

            page_id = row["commons_page_id"]
            face_id = row["face_id"]
            mid = f"M{page_id}"

            try:
                # Idempotency check
                claim_params = {
                    "action": "wbgetclaims",
                    "entity": mid,
                    "property": "P180",
                    "format": "json",
                }
                claim_headers = {"Authorization": f"Bearer {access_token}"}

                claim_resp = _api_request(
                    COMMONS_API_URL, params=claim_params, headers=claim_headers
                )
                claim_data = claim_resp.json()

                already_exists = False
                claims = claim_data.get("claims", {}).get("P180", [])
                for claim in claims:
                    snak = claim.get("mainsnak", {})
                    if snak.get("datavalue", {}).get("value", {}).get("id") == qid:
                        already_exists = True
                        break

                if already_exists:
                    execute_query(
                        "UPDATE faces SET sdc_written = 1 WHERE id = %s",
                        (face_id,),
                        fetch=False,
                    )
                    total_written += 1
                    continue

                # Write claim
                edit_data = {
                    "action": "wbeditentity",
                    "id": mid,
                    "data": json.dumps(
                        {
                            "claims": [
                                {
                                    "mainsnak": {
                                        "snaktype": "value",
                                        "property": "P180",
                                        "datavalue": {
                                            "type": "wikibase-entityid",
                                            "value": {
                                                "numeric-id": numeric_id,
                                                "id": qid,
                                            },
                                        },
                                    },
                                    "type": "statement",
                                    "rank": "normal",
                                }
                            ]
                        }
                    ),
                    "token": csrf_token,
                    "summary": f"WikiVisage: Adding depicts (P180) claim for {qid}",
                    "format": "json",
                    "bot": "1",
                    "maxlag": "5",
                }

                edit_resp = _api_request(
                    COMMONS_API_URL,
                    data=edit_data,
                    headers=claim_headers,
                    method="post",
                )
                edit_json = edit_resp.json()

                if "error" in edit_json:
                    error_code = edit_json["error"].get("code")
                    error_info = edit_json["error"].get("info", "Unknown error")
                    if error_code == "badtoken":
                        # Refresh token and retry this face
                        try:
                            csrf_token = _get_csrf_token(access_token)
                        except Exception:
                            logger.error(
                                f"Failed to refresh CSRF token for project {project_id}"
                            )
                            break
                        continue
                    else:
                        # Any other API error — abort entire write
                        logger.error(f"SDC Write error for {mid}: {edit_json['error']}")
                        execute_query(
                            "UPDATE projects SET sdc_write_requested = 0, "
                            "sdc_write_error = %s WHERE id = %s",
                            (f"{error_code}: {error_info}", project_id),
                            fetch=False,
                        )
                        logger.info(
                            f"SDC writes aborted for project {project_id}: "
                            f"{total_written} written before error"
                        )
                        return total_written

                execute_query(
                    "UPDATE faces SET sdc_written = 1 WHERE id = %s",
                    (face_id,),
                    fetch=False,
                )
                total_written += 1

                # Rate limiting for MediaWiki API
                time.sleep(1)

            except Exception as e:
                logger.error(f"Error writing SDC for face {face_id} on {mid}: {e}")
                execute_query(
                    "UPDATE projects SET sdc_write_requested = 0, "
                    "sdc_write_error = %s WHERE id = %s",
                    (str(e)[:1024], project_id),
                    fetch=False,
                )
                logger.info(
                    f"SDC writes aborted for project {project_id}: "
                    f"{total_written} written before error"
                )
                return total_written

    # Clear the flag — all done successfully (or shutdown requested)
    error_msg = None
    if shutdown_requested:
        error_msg = "Worker shutdown interrupted SDC writes"

    execute_query(
        "UPDATE projects SET sdc_write_requested = 0, sdc_write_error = %s WHERE id = %s",
        (error_msg, project_id),
        fetch=False,
    )

    logger.info(f"SDC writes for project {project_id}: {total_written} written")
    return total_written


def process_project(project: Dict[str, Any]) -> None:
    """Orchestrate all phases for a single project."""
    logger.info(
        f"--- Processing project {project['id']} ({project['wikidata_qid']}) ---"
    )

    try:
        # 1. Traversal
        traverse_category(project)

        # 2. Bootstrap (if needed)
        bootstrap_from_sparql(project)

        # 3. Image Download & Face Detection — process all pending images
        #    Interleave inference every 5 batches so results appear progressively
        total_processed = 0
        batches_since_inference = 0
        while not shutdown_requested:
            batch_count = process_images(project)
            if batch_count == 0:
                break
            total_processed += batch_count
            batches_since_inference += 1
            logger.info(
                f"Processed batch of {batch_count} images ({total_processed} total so far)"
            )

            # Run inference periodically during image processing
            if batches_since_inference >= 5:
                batches_since_inference = 0
                fresh = execute_query(
                    "SELECT * FROM projects WHERE id = %s",
                    (project["id"],),
                    fetch=True,
                )
                proj = fresh[0] if fresh and isinstance(fresh, list) else project
                classified = run_autonomous_inference(proj)
                if classified:
                    logger.info(
                        f"Mid-processing inference classified {classified} faces"
                    )

        # 4. Final Autonomous Inference (catch any remaining unclassified faces)
        fresh = execute_query(
            "SELECT * FROM projects WHERE id = %s", (project["id"],), fetch=True
        )
        if fresh and isinstance(fresh, list):
            run_autonomous_inference(fresh[0])
        else:
            run_autonomous_inference(project)

        # 5. SDC writes are handled separately — triggered via web UI,
        #    processed in the main loop when sdc_write_requested=1.

    except Exception as e:
        logger.error(f"Failed to process project {project['id']}: {e}")


def signal_handler(signum, frame):
    """Graceful shutdown on SIGINT/SIGTERM."""
    global shutdown_requested
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    shutdown_requested = True


def main():
    """Main worker loop with concurrent project processing."""
    logger.info(
        f"Starting WikiVisage Worker "
        f"(max_projects={MAX_CONCURRENT_PROJECTS}, image_threads={IMAGE_THREADS})"
    )

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        init_db(pool_size=15)
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        sys.exit(1)

    try:
        diag = execute_query(
            "SELECT p.id, p.wikidata_qid, p.status, p.min_confirmed, p.faces_confirmed, "
            "  (SELECT COUNT(*) FROM faces f JOIN images i ON f.image_id = i.id "
            "   WHERE i.project_id = p.id AND f.is_target = 1) AS actual_target_faces, "
            "  (SELECT COUNT(*) FROM faces f JOIN images i ON f.image_id = i.id "
            "   WHERE i.project_id = p.id AND f.is_target = 0) AS actual_non_target, "
            "  (SELECT COUNT(*) FROM faces f JOIN images i ON f.image_id = i.id "
            "   WHERE i.project_id = p.id AND f.is_target IS NULL) AS unclassified_faces "
            "FROM projects p",
            fetch=True,
        )
        if diag and isinstance(diag, list):
            for row in diag:
                logger.info(
                    f"Project {row['id']} ({row['wikidata_qid']}): "
                    f"status={row['status']}, min_confirmed={row['min_confirmed']}, "
                    f"faces_confirmed_counter={row['faces_confirmed']}, "
                    f"actual_target={row['actual_target_faces']}, "
                    f"actual_non_target={row['actual_non_target']}, "
                    f"unclassified={row['unclassified_faces']}"
                )
    except Exception as e:
        logger.warning(f"Diagnostic query failed: {e}")

    try:
        while not shutdown_requested:
            try:
                execute_query(
                    "REPLACE INTO worker_heartbeat (id, last_seen) VALUES (1, NOW())",
                    fetch=False,
                )

                active_projects = execute_query(
                    "SELECT p.*, "
                    "  (SELECT COUNT(*) FROM images i WHERE i.project_id = p.id) AS image_count "
                    "FROM projects p WHERE p.status = 'active' "
                    "ORDER BY image_count ASC, p.id DESC",
                    fetch=True,
                )

                inference_projects = execute_query(
                    "SELECT p.* FROM projects p "
                    "WHERE p.status IN ('active', 'completed') "
                    "AND ("
                    "  SELECT COUNT(*) FROM faces f "
                    "  JOIN images i ON f.image_id = i.id "
                    "  WHERE i.project_id = p.id AND f.is_target = 1"
                    ") >= p.min_confirmed "
                    "AND EXISTS ("
                    "  SELECT 1 FROM faces f "
                    "  JOIN images i ON f.image_id = i.id "
                    "  WHERE i.project_id = p.id AND f.is_target IS NULL"
                    ")",
                    fetch=True,
                )

                active_count = (
                    len(active_projects) if isinstance(active_projects, list) else 0
                )
                inference_count = (
                    len(inference_projects)
                    if isinstance(inference_projects, list)
                    else 0
                )

                # Check for SDC write requests
                sdc_projects = execute_query(
                    "SELECT p.* FROM projects p WHERE p.sdc_write_requested = 1",
                    fetch=True,
                )
                sdc_count = len(sdc_projects) if isinstance(sdc_projects, list) else 0

                logger.info(
                    f"Poll: {active_count} active project(s), "
                    f"{inference_count} inference-eligible project(s), "
                    f"{sdc_count} SDC write request(s)"
                )

                # Process active projects concurrently
                seen_ids = set()
                if active_projects and isinstance(active_projects, list):
                    with ThreadPoolExecutor(
                        max_workers=MAX_CONCURRENT_PROJECTS
                    ) as executor:
                        futures = {}
                        for project in active_projects:
                            if shutdown_requested:
                                break
                            seen_ids.add(project["id"])
                            future = executor.submit(process_project, project)
                            futures[future] = project["id"]

                        last_heartbeat = time.time()
                        while futures:
                            if shutdown_requested:
                                break

                            # Refresh heartbeat during long processing cycles
                            if time.time() - last_heartbeat >= 60:
                                try:
                                    execute_query(
                                        "REPLACE INTO worker_heartbeat (id, last_seen) VALUES (1, NOW())",
                                        fetch=False,
                                    )
                                except Exception:
                                    pass
                                last_heartbeat = time.time()

                            # Check for wake-up signal every iteration (even if no futures done)
                            if os.path.exists(WAKE_UP_FILE):
                                try:
                                    os.remove(WAKE_UP_FILE)
                                except OSError:
                                    pass
                                logger.info(
                                    "Wake-up signal received mid-cycle, checking for new projects"
                                )
                                new_projects = execute_query(
                                    "SELECT p.*, "
                                    "  (SELECT COUNT(*) FROM images i WHERE i.project_id = p.id) AS image_count "
                                    "FROM projects p WHERE p.status = 'active' "
                                    "ORDER BY image_count ASC, p.id DESC",
                                    fetch=True,
                                )
                                if new_projects and isinstance(new_projects, list):
                                    for project in new_projects:
                                        if project["id"] not in seen_ids:
                                            seen_ids.add(project["id"])
                                            logger.info(
                                                f"Adding new project {project['id']} to current cycle"
                                            )
                                            future = executor.submit(
                                                process_project, project
                                            )
                                            futures[future] = project["id"]

                            done = {f for f in futures if f.done()}
                            if not done:
                                time.sleep(1)
                                continue

                            for future in done:
                                project_id = futures.pop(future)
                                try:
                                    future.result()
                                except Exception as e:
                                    logger.error(
                                        f"Project {project_id} processing failed: {e}"
                                    )

                # Run inference for eligible projects not already processed
                if inference_projects and isinstance(inference_projects, list):
                    inference_only = [
                        p for p in inference_projects if p["id"] not in seen_ids
                    ]
                    if inference_only:
                        with ThreadPoolExecutor(
                            max_workers=MAX_CONCURRENT_PROJECTS
                        ) as executor:
                            futures = {}
                            for project in inference_only:
                                if shutdown_requested:
                                    break
                                logger.info(
                                    f"Running inference-only for project {project['id']} "
                                    f"(status={project.get('status')})"
                                )
                                future = executor.submit(
                                    run_autonomous_inference, project
                                )
                                futures[future] = project["id"]

                            last_heartbeat_inf = time.time()
                            for future in as_completed(futures):
                                if shutdown_requested:
                                    break
                                # Refresh heartbeat during inference
                                if time.time() - last_heartbeat_inf >= 60:
                                    try:
                                        execute_query(
                                            "REPLACE INTO worker_heartbeat (id, last_seen) VALUES (1, NOW())",
                                            fetch=False,
                                        )
                                    except Exception:
                                        pass
                                    last_heartbeat_inf = time.time()
                                project_id = futures[future]
                                try:
                                    classified = future.result()
                                    logger.info(
                                        f"Inference classified {classified} faces "
                                        f"for project {project_id}"
                                    )
                                except Exception as e:
                                    logger.error(
                                        f"Inference failed for project {project_id}: {e}"
                                    )

                # Process SDC write requests (sequential — one project at a time)
                if sdc_projects and isinstance(sdc_projects, list):
                    for sdc_project in sdc_projects:
                        if shutdown_requested:
                            break
                        logger.info(
                            f"Processing SDC write request for project {sdc_project['id']}"
                        )
                        try:
                            written = write_sdc_claims(sdc_project)
                            logger.info(
                                f"SDC write complete for project {sdc_project['id']}: "
                                f"{written} claims written"
                            )
                        except Exception as e:
                            logger.error(
                                f"SDC write failed for project {sdc_project['id']}: {e}"
                            )
                            # Clear the flag so it doesn't retry endlessly
                            try:
                                execute_query(
                                    "UPDATE projects SET sdc_write_requested = 0, "
                                    "sdc_write_error = %s WHERE id = %s",
                                    (str(e)[:1000], sdc_project["id"]),
                                    fetch=False,
                                )
                            except DatabaseError:
                                pass

                for _ in range(POLL_INTERVAL):
                    if shutdown_requested:
                        break
                    if os.path.exists(WAKE_UP_FILE):
                        try:
                            os.remove(WAKE_UP_FILE)
                        except OSError:
                            pass
                        logger.info("Wake-up signal received, starting next cycle")
                        break
                    time.sleep(1)

            except DatabaseError as e:
                logger.error(f"Database error in main loop: {e}")
                time.sleep(10)
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}")
                time.sleep(10)

    finally:
        logger.info("Worker shutting down, closing database pool.")
        close_pool()


if __name__ == "__main__":
    # spawn avoids fork-safety issues with threads + dlib's C++ code
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
    main()
