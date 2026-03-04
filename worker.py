import io
import json
import logging
import os
import signal
import sys
import time
from typing import Any, Dict, List, Optional

import face_recognition
import numpy as np
import requests
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
    """Fetch all files in the project's Commons category and insert them as pending."""
    logger.info(f"Traversing category for project {project['id']}")

    category = project["commons_category"]
    if not category.startswith("Category:"):
        category = f"Category:{category}"

    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": category,
        "cmtype": "file",
        "cmlimit": "500",
        "format": "json",
    }

    added_count = 0
    cmcontinue = None

    while not shutdown_requested:
        if cmcontinue:
            params["cmcontinue"] = cmcontinue

        try:
            resp = _api_request(COMMONS_API_URL, params=params)
            data = resp.json()

            members = data.get("query", {}).get("categorymembers", [])
            for member in members:
                page_id = member["pageid"]
                title = member["title"]

                # Check if it already exists
                exists = execute_query(
                    "SELECT 1 FROM images WHERE project_id = %s AND commons_page_id = %s",
                    (project["id"], page_id),
                    fetch=True,
                )

                if not exists:
                    execute_query(
                        """
                        INSERT IGNORE INTO images (project_id, commons_page_id, file_title, status)
                        VALUES (%s, %s, %s, 'pending')
                        """,
                        (project["id"], page_id, title),
                        fetch=False,
                    )
                    added_count += 1

            if "continue" in data and "cmcontinue" in data["continue"]:
                cmcontinue = data["continue"]["cmcontinue"]
                time.sleep(1)  # Respect rate limits
            else:
                break

        except Exception as e:
            logger.error(f"Category traversal failed: {e}")
            break

    # Update total image count
    execute_query(
        "UPDATE projects SET images_total = (SELECT COUNT(*) FROM images WHERE project_id = %s) WHERE id = %s",
        (project["id"], project["id"]),
        fetch=False,
    )

    return added_count


def process_images(project: Dict[str, Any]) -> int:
    """Download images, detect faces, and extract embeddings."""
    logger.info(f"Processing images for project {project['id']}")

    pending_images = execute_query(
        "SELECT id, file_title FROM images WHERE project_id = %s AND status = 'pending' LIMIT %s",
        (project["id"], BATCH_SIZE),
        fetch=True,
    )

    if not pending_images:
        return 0

    processed_count = 0

    for img_row in pending_images:
        if shutdown_requested:
            break

        img_id = img_row["id"]
        title = img_row["file_title"]

        # Strip File: prefix if exists for URL
        clean_title = title[5:] if title.startswith("File:") else title
        url = FILE_PATH_URL.format(file_title=clean_title)

        try:
            logger.debug(f"Downloading image {title}")
            resp = _api_request(url)

            image_data = face_recognition.load_image_file(io.BytesIO(resp.content))
            del resp  # Free memory

            face_locations = face_recognition.face_locations(image_data, model="hog")
            face_encodings = face_recognition.face_encodings(image_data, face_locations)

            face_count = len(face_locations)

            # Insert faces
            for location, encoding in zip(face_locations, face_encodings):
                top, right, bottom, left = location
                encoding_bytes = encoding.tobytes()

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
            processed_count += 1

            del image_data, face_locations, face_encodings

        except Exception as e:
            logger.error(f"Error processing image {title}: {e}")
            error_msg = str(e)[:1000]
            execute_query(
                "UPDATE images SET status = 'error', error_message = %s WHERE id = %s",
                (error_msg, img_id),
                fetch=False,
            )

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

    if project.get("faces_confirmed", 0) > 0:
        return 0

    qid = project["wikidata_qid"]
    search_params = {
        "action": "query",
        "list": "search",
        "srsearch": f"haswbstatement:P180={qid}",
        "srnamespace": "6",
        "srlimit": "50",
        "format": "json",
    }

    try:
        resp = _api_request(COMMONS_API_URL, params=search_params)
        results = resp.json().get("query", {}).get("search", [])

        bootstrapped_count = 0

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
                img_resp = _api_request(url)
                image_data = face_recognition.load_image_file(
                    io.BytesIO(img_resp.content)
                )
                face_locations = face_recognition.face_locations(
                    image_data, model="hog"
                )
                face_encodings = face_recognition.face_encodings(
                    image_data, face_locations
                )

                face_count = len(face_locations)

                if face_count == 1:
                    top, right, bottom, left = face_locations[0]
                    encoding_bytes = face_encodings[0].tobytes()

                    execute_query(
                        """
                        INSERT INTO faces (image_id, encoding, bbox_top, bbox_right, bbox_bottom, bbox_left, is_target, classified_by)
                        VALUES (%s, %s, %s, %s, %s, %s, 1, 'bootstrap')
                        """,
                        (img_id, encoding_bytes, top, right, bottom, left),
                        fetch=False,
                    )
                    bootstrapped_count += 1
                else:
                    for location, encoding in zip(face_locations, face_encodings):
                        top, right, bottom, left = location
                        encoding_bytes = encoding.tobytes()
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

            except Exception as e:
                logger.error(f"Bootstrap image processing error for {title}: {e}")
                execute_query(
                    "UPDATE images SET status = 'error', error_message = %s WHERE id = %s",
                    (str(e)[:1000], img_id),
                    fetch=False,
                )

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
    if project.get("faces_confirmed", 0) < project.get("min_confirmed", 5):
        return 0

    logger.info(f"Running autonomous inference for project {project['id']}")

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

    confirmed_encodings = [
        np.frombuffer(row["encoding"], dtype=np.float64) for row in confirmed_rows
    ]
    centroid = np.mean(confirmed_encodings, axis=0)

    # Process unclassified faces in batches
    unclassified_rows = execute_query(
        """
        SELECT f.id, f.encoding FROM faces f
        JOIN images i ON f.image_id = i.id
        WHERE i.project_id = %s AND f.is_target IS NULL
        LIMIT %s
        """,
        (project["id"], BATCH_SIZE),
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
    """Write SDC P180 claims for faces marked as target."""
    logger.info(f"Writing SDC claims for project {project['id']}")

    # Get faces to write
    faces_to_write = execute_query(
        """
        SELECT f.id as face_id, i.commons_page_id, p.wikidata_qid, p.user_id 
        FROM faces f
        JOIN images i ON f.image_id = i.id
        JOIN projects p ON i.project_id = p.id
        WHERE i.project_id = %s AND f.is_target = 1 AND f.sdc_written = 0
        LIMIT %s
        """,
        (project["id"], BATCH_SIZE),
        fetch=True,
    )

    if not faces_to_write:
        return 0

    # Get user token
    user_row = execute_query(
        "SELECT access_token FROM users WHERE id = %s",
        (project["user_id"],),
        fetch=True,
    )

    if not user_row:
        logger.error(f"User {project['user_id']} not found for SDC writes")
        return 0

    access_token = user_row[0]["access_token"].decode("utf-8")

    try:
        csrf_token = _get_csrf_token(access_token)
    except Exception as e:
        logger.error(f"Failed to get CSRF token for project {project['id']}: {e}")
        return 0

    written_count = 0
    qid = project["wikidata_qid"]

    # Parse numeric id (e.g. Q42 -> 42)
    try:
        numeric_id = int(qid[1:])
    except ValueError:
        logger.error(f"Invalid QID format: {qid}")
        return 0

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
                                        "value": {"numeric-id": numeric_id, "id": qid},
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
            }

            edit_resp = _api_request(
                COMMONS_API_URL, data=edit_data, headers=claim_headers, method="post"
            )
            edit_json = edit_resp.json()

            if "error" in edit_json:
                error_code = edit_json["error"].get("code")
                if error_code == "badtoken":
                    # Refresh token and retry next loop
                    csrf_token = _get_csrf_token(access_token)
                    continue
                else:
                    logger.error(f"SDC Write error for {mid}: {edit_json['error']}")
                    continue

            execute_query(
                "UPDATE faces SET sdc_written = 1 WHERE id = %s",
                (face_id,),
                fetch=False,
            )
            written_count += 1

            # Rate limiting
            time.sleep(1)

        except Exception as e:
            logger.error(f"Error writing SDC for face {face_id} on {mid}: {e}")

    return written_count


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

        # 3. Image Download & Face Detection
        process_images(project)

        # 4. Autonomous Inference
        run_autonomous_inference(project)

        # 5. SDC Mutation
        write_sdc_claims(project)

    except Exception as e:
        logger.error(f"Failed to process project {project['id']}: {e}")


def signal_handler(signum, frame):
    """Graceful shutdown on SIGINT/SIGTERM."""
    global shutdown_requested
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    shutdown_requested = True


def main():
    """Main worker loop."""
    logger.info("Starting WikiVisage Worker")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        init_db()
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        sys.exit(1)

    try:
        while not shutdown_requested:
            try:
                active_projects = execute_query(
                    "SELECT * FROM projects WHERE status = 'active'", fetch=True
                )

                if active_projects:
                    for project in active_projects:
                        if shutdown_requested:
                            break
                        process_project(project)
                else:
                    logger.debug("No active projects found")

                # Sleep before next cycle, break if shutdown
                for _ in range(POLL_INTERVAL):
                    if shutdown_requested:
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
    main()
