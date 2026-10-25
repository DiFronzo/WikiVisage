import enum
import html
import io
import json
import logging
import os
import random
import re
import signal
import socket
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Any, Self
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()

# Optional throttle (in seconds) between image downloads and processing to
# avoid overloading Commons or downstream services. Defaults to 0 (no delay).
COMMONS_DOWNLOAD_THROTTLE_SECONDS = float(os.getenv("COMMONS_DOWNLOAD_THROTTLE_SECONDS", "0"))

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
warnings.filterwarnings("ignore", message=".*pkg_resources.*", category=UserWarning)

import numpy as np
import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

import face_client
import liftwing_client
import oauth_refresh
from config import DETECTION_WIDTH, HEARTBEAT_FILE_DIR, LIFTWING_PROBE_INTERVAL, USER_AGENT, WAKE_FILE_PATH
from database import DatabaseError, close_pool, execute_query, execute_transaction, init_db
from face_client import FaceServiceError, FaceServiceModelMismatch, FaceServiceRejected

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("worker")

# Constants
COMMONS_API_URL = "https://commons.wikimedia.org/w/api.php"
# Detection runs on this Commons thumbnail width (WIKIVISAGE_DETECTION_WIDTH, default 960).
FILE_PATH_URL = f"https://commons.wikimedia.org/wiki/Special:FilePath/{{file_title}}?width={DETECTION_WIDTH}"

POLL_INTERVAL = int(os.environ.get("WIKIVISAGE_WORKER_POLL_INTERVAL", 60))
BATCH_SIZE = int(os.environ.get("WIKIVISAGE_WORKER_BATCH_SIZE", 50))
MAX_CONCURRENT_PROJECTS = int(os.environ.get("WIKIVISAGE_WORKER_MAX_PROJECTS", 4))
IMAGE_THREADS = int(os.environ.get("WIKIVISAGE_WORKER_IMAGE_THREADS", 4))

# Images downloaded and held in memory at once. This — not BATCH_SIZE — bounds
# the worker's peak memory: at the 50 MB download cap, holding a whole batch of
# 50 would be 2.5 GB worst case. Sized to cover the download fan-out and one
# full detection batch so neither stage starves the other.
DETECTION_WAVE_SIZE = max(IMAGE_THREADS, face_client.FACE_SERVICE_BATCH_SIZE)

# Cosine-distance threshold used when a project row carries none. Provisional
# until calibrated on real data; keep in step with schema.sql / migrate.py.
DEFAULT_DISTANCE_THRESHOLD = 0.5

# A stored embedding is L2-normalised, so its norm is 1. Anything this far off
# is corrupt (or from another model) and is skipped; values inside the band are
# float noise and are renormalised.
EMBEDDING_NORM_MIN = 0.9
EMBEDDING_NORM_MAX = 1.1

# Distributed locking: stale claims from crashed workers expire after this many minutes
CLAIM_EXPIRY_MINUTES = 15

# Maximum image download size (50 MB) — prevents OOM on abnormally large files
MAX_IMAGE_DOWNLOAD_BYTES = 50 * 1024 * 1024

# Maximum image pixel area (100 megapixels) — keeps a decompression bomb from
# reaching the face service, which enforces its own independent ceiling.
MAX_IMAGE_PIXELS = 100_000_000

# Maximum images per project — stops category traversal once this limit is reached
MAX_IMAGES_PER_PROJECT = 9000

# --- Experimental LiftWing metadata pre-filter (Phase 2) ---------------------
# Files sent per LiftWing request (the model returns one verdict per file).
# Measured 2026-10-17: ~1 s per file, so 20 keeps a request ~20 s, well inside
# the client's 90 s timeout even under load.
LIFTWING_BATCH_SIZE = int(os.environ.get("WIKIVISAGE_LIFTWING_BATCH_SIZE", "20"))
# Images the pre-filter records per call — a project cycle or a sweep pass.
# Because the pre-filter runs after detection, a large backlog is worked off
# over several cycles without delaying images; verdicts are only shadow data.
LIFTWING_MAX_IMAGES = int(os.environ.get("WIKIVISAGE_LIFTWING_MAX_IMAGES", "250"))
# Consecutive failed probes before service_status flips to 'down' (banner).
LIFTWING_FAIL_THRESHOLD = int(os.environ.get("WIKIVISAGE_LIFTWING_FAIL_THRESHOLD", "3"))
# Pause between background shadow-sweep passes, in seconds. The sweep records
# verdicts for projects the processing pipelines never claim (completed, paused
# or fully processed); each pass covers up to LIFTWING_MAX_IMAGES files of one
# project. 0 disables the sweep.
LIFTWING_SWEEP_INTERVAL = int(os.environ.get("WIKIVISAGE_LIFTWING_SWEEP_INTERVAL", "60"))
# How long shutdown waits for the sweep thread to leave its current batch
# before claims are released and the DB pool / HTTP session are closed. Covers
# an in-flight LiftWing request (liftwing_client.REQUEST_TIMEOUT) plus the
# metadata fetch, verdict storage and claim release around it.
SWEEP_JOIN_TIMEOUT = int(liftwing_client.REQUEST_TIMEOUT + 15)

# Bootstrap processing cap — limits how many bootstrap images process_images() will
# download/detect per project.  All P180 images are inserted into the DB by
# bootstrap_from_sparql (no insertion cap), but processing is throttled here so
# the worker doesn't spend all its time on bootstrap downloads.
BOOTSTRAP_TARGET_RATIO = 0.10  # Base: ~10% of MAX_IMAGES_PER_PROJECT (900 for 9000 limit)
BOOTSTRAP_MAX_RATIO = 0.40  # Ceiling: never exceed 40% even for heavily-tagged categories
MIN_BOOTSTRAP_IMAGES = 100  # Floor: always process at least 100 bootstrap images

OAUTH_CLIENT_ID = os.environ.get("OAUTH_CLIENT_ID", "")
OAUTH_CLIENT_SECRET = os.environ.get("OAUTH_CLIENT_SECRET", "")
OAUTH_TOKEN_URL = "https://meta.wikimedia.org/w/rest.php/oauth2/access_token"

# Maximum number of token refresh retries per SDC write session
MAX_TOKEN_RETRIES = 2


# Lazy redis client for the OAuth single-flight lock. None means we skipped/failed
# to connect; the lock degrades to a no-op (DB CAS still protects correctness).
_REDIS_URL = os.environ.get("WIKIVISAGE_REDIS_URL", "redis://redis.svc.tools.eqiad1.wikimedia.cloud:6379")
_redis_client: Any = None
_redis_init_attempted = False


def _redact_redis_url(url: str) -> str:
    """Return the Redis URL with any password replaced by *** for safe logging."""
    try:
        parsed = urlparse(url)
        if parsed.password:
            # urlparse exposes the password separately; rebuild netloc with it masked.
            safe_netloc = parsed.netloc.replace(f":{parsed.password}@", ":***@")
            return parsed._replace(netloc=safe_netloc).geturl()
    except Exception:
        pass
    return url


def _get_redis_client() -> Any:
    """Lazily connect to Redis for cross-process locks. Cached after first call."""
    global _redis_client, _redis_init_attempted
    if _redis_init_attempted:
        return _redis_client
    _redis_init_attempted = True
    try:
        import redis as _redis_mod

        client = _redis_mod.from_url(_REDIS_URL, socket_connect_timeout=2)
        client.ping()
        _redis_client = client
    except Exception:
        logging.getLogger(__name__).warning(
            "Worker: Redis unavailable at %s — OAuth refresh single-flight lock disabled (DB CAS still active)",
            _redact_redis_url(_REDIS_URL),
        )
        _redis_client = None
    return _redis_client


# Non-image file extensions to skip during category traversal (video, audio)
_SKIP_EXTENSIONS = {".webm", ".ogv", ".ogg", ".mp3", ".wav", ".flac", ".opus", ".mid", ".oga"}


def _load_embedding(
    blob: bytes | None, dim: int = face_client.EMBEDDING_DIM, face_id: int | str = "unknown"
) -> np.ndarray | None:
    """Decode and validate a face embedding from a DB BLOB.

    The BLOB must be exactly ``dim`` little-endian float32 values, all finite,
    with an L2 norm in ``[EMBEDDING_NORM_MIN, EMBEDDING_NORM_MAX]``. Returns a
    writable, renormalised unit-length ``float32`` array, or None (with a
    warning) otherwise. This keeps DB-level tampering, truncation or a vector
    from another model (e.g. a leftover 1024-byte dlib encoding) from corrupting
    the centroid and silently mass-misclassifying faces.
    """
    expected = dim * 4
    if blob is None or len(blob) != expected:
        logger.warning(
            f"Face {face_id}: invalid embedding length ({len(blob) if blob else 'None'} bytes), expected {expected}"
        )
        return None
    arr = np.frombuffer(blob, dtype="<f4").astype(np.float32)
    if not np.isfinite(arr).all():
        logger.warning(f"Face {face_id}: embedding contains NaN/Inf — skipping")
        return None
    norm = float(np.linalg.norm(arr))
    if not EMBEDDING_NORM_MIN <= norm <= EMBEDDING_NORM_MAX:
        logger.warning(f"Face {face_id}: embedding norm {norm:.3f} outside expected range — skipping")
        return None
    return arr / norm


def _face_distance(centroid: np.ndarray, embedding: np.ndarray) -> float:
    """Cosine distance ``1 - cos`` between a candidate embedding and the project centroid.

    Both arguments must be unit vectors (the centroid is the *renormalised* mean
    of unit embeddings), so the cosine similarity is just their dot product. The
    result lies in ``[0, 2]``; 0 is identical. Float rounding can push a
    near-identical pair a hair below zero, so it is clamped.
    """
    return max(0.0, 1.0 - float(np.dot(centroid, embedding)))


def _convert_box_to_stored_space(
    box: tuple[int, int, int, int],
    from_size: tuple[int, int],
    stored_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Re-express a ``(top, right, bottom, left)`` box from one pixel space in another.

    ``from_size`` and ``stored_size`` are ``(width, height)``. Boxes are always
    stored in the space recorded in ``images.detection_width/height``. If a face
    is found later on an image at a different size (for example a re-detection
    at 960 px on an image whose stored boxes are in a 500 px space), the new box
    must be converted into the stored space before it is inserted, or it would
    sit beside the old boxes in a different coordinate system. Result is clipped
    to ``stored_size`` and kept at least 1 px wide and tall.
    """
    from_w, from_h = from_size
    stored_w, stored_h = stored_size
    if min(from_w, from_h, stored_w, stored_h) <= 0:
        raise ValueError(f"invalid image size: from={from_size} stored={stored_size}")
    top, right, bottom, left = box
    scale_x, scale_y = stored_w / from_w, stored_h / from_h
    left_s = min(max(0, round(left * scale_x)), stored_w - 1)
    top_s = min(max(0, round(top * scale_y)), stored_h - 1)
    right_s = min(max(left_s + 1, round(right * scale_x)), stored_w)
    bottom_s = min(max(top_s + 1, round(bottom * scale_y)), stored_h)
    return top_s, right_s, bottom_s, left_s


# Global Shutdown Flag
shutdown_requested = False

# Unique identifier for this worker instance (set in main())
_worker_id: str = ""


def _touch_heartbeat_file() -> None:
    """Touch a per-worker heartbeat file so the Toolforge health-check-script can verify liveness."""
    try:
        path = os.path.join(HEARTBEAT_FILE_DIR, f".wikivisage-worker-alive-{_worker_id}")
        with open(path, "w") as f:
            f.write(str(time.time()))
    except OSError:
        pass


def _create_session() -> requests.Session:
    """Create a requests session with exponential backoff retries and connection pooling."""
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    retries = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS", "POST"],
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=20)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


# Module-level singleton session — reuses TCP connections and TLS state across all
# HTTP requests, avoiding the ~100ms+ overhead of per-request Session creation.
_http_session: requests.Session | None = None
_http_session_lock = threading.Lock()


def _get_session() -> requests.Session:
    """Return the singleton HTTP session, creating it on first use."""
    global _http_session
    if _http_session is None:
        with _http_session_lock:
            if _http_session is None:
                _http_session = _create_session()
    return _http_session


def _api_request(
    url: str,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    method: str = "get",
    data: dict[str, Any] | None = None,
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
                resp = session.get(url, params=params, headers=headers, timeout=timeout, allow_redirects=False)
            else:
                resp = session.post(
                    url, data=data, params=params, headers=headers, timeout=timeout, allow_redirects=False
                )

            # Check for MediaWiki maxlag before raise_for_status().
            # MediaWiki returns HTTP 503 with Retry-After header for maxlag,
            # so raise_for_status() would throw before we can handle it.
            if resp.status_code == 503 and "Retry-After" in resp.headers:
                retry_after_header = resp.headers.get("Retry-After", "5")
                try:
                    retry_after = float(retry_after_header)
                except (TypeError, ValueError):
                    retry_after = 5.0
                logger.warning(f"Maxlag encountered (503). Sleeping for {retry_after} seconds.")
                resp.close()
                time.sleep(retry_after)
                continue

            resp.raise_for_status()

            # Also check for maxlag reported inside a 200 JSON response
            if "Retry-After" in resp.headers:
                try:
                    data_json = resp.json()
                    if "error" in data_json and data_json["error"].get("code") == "maxlag":
                        retry_after_header = resp.headers.get("Retry-After", "5")
                        try:
                            retry_after = float(retry_after_header)
                        except (TypeError, ValueError):
                            retry_after = 5.0
                        logger.warning(f"Maxlag encountered (200). Sleeping for {retry_after} seconds.")
                        resp.close()
                        time.sleep(retry_after)
                        continue
                except ValueError:
                    pass

            return resp
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP Request failed: {e}")
            time.sleep(5)

    raise Exception(f"Failed to execute API request after 3 attempts: {url}")


# Hosts a Commons media download may legitimately resolve to. Wikimedia splits
# these: Special:FilePath?width=N lands on thumb.wikimedia.org, while the
# original file (no width) still lands on upload.wikimedia.org. Dropping
# thumb.wikimedia.org silently fails every thumbnail download — see
# "Redirect to untrusted host" in Known Issues.
_ALLOWED_DOWNLOAD_HOSTS = frozenset(
    {
        "commons.wikimedia.org",
        "upload.wikimedia.org",
        "thumb.wikimedia.org",
    }
)


def _download_image(url: str, max_bytes: int = MAX_IMAGE_DOWNLOAD_BYTES) -> bytes:
    """Download an image with streaming size cap to prevent OOM.

    Uses _get_session() for connection pooling / retry. Raises ValueError
    if the URL points to an untrusted host, uses a non-HTTPS scheme, or if
    the response exceeds max_bytes.
    """
    parsed_url = urlparse(url)
    if parsed_url.scheme != "https" or parsed_url.hostname not in _ALLOWED_DOWNLOAD_HOSTS:
        raise ValueError(f"Blocked download from untrusted host: {parsed_url.hostname}")
    session = _get_session()
    resp = None
    current_url = url
    max_redirects = 5
    try:
        for _hop in range(max_redirects + 1):
            resp = session.get(
                current_url,
                headers={"User-Agent": USER_AGENT},
                timeout=30,
                stream=True,
                allow_redirects=False,
            )
            if not resp.is_redirect:
                break
            # Validate redirect target before following
            resp.close()
            location = resp.headers.get("Location", "")
            parsed_redirect = urlparse(location)
            if parsed_redirect.scheme != "https" or parsed_redirect.hostname not in _ALLOWED_DOWNLOAD_HOSTS:
                raise ValueError(f"Redirect to untrusted host: {parsed_redirect.hostname}")
            current_url = location
        else:
            raise ValueError(f"Too many redirects ({max_redirects}) from allowed download hosts")

        resp.raise_for_status()

        # Check Content-Length header first (fast reject)
        content_length = resp.headers.get("Content-Length")
        if content_length and int(content_length) > max_bytes:
            raise ValueError(f"Image too large: {int(content_length)} bytes (limit {max_bytes})")

        # Stream with enforced cap
        chunks: list[bytes] = []
        downloaded = 0
        for chunk in resp.iter_content(chunk_size=65536):
            downloaded += len(chunk)
            if downloaded > max_bytes:
                raise ValueError(f"Image download exceeded {max_bytes} bytes limit")
            chunks.append(chunk)
    finally:
        if resp is not None:
            resp.close()

    return b"".join(chunks)


def _validate_image_dimensions(image_bytes: bytes) -> None:
    """Check image dimensions via PIL header-only read. Raises ValueError if too large.

    Catches PIL.Image.DecompressionBombError explicitly so callers see a uniform
    ValueError instead of an unhandled exception (defense-in-depth — the face
    service also caps its own decoded pixel area).
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        w, h = img.size
    except Image.DecompressionBombError as e:
        raise ValueError(f"Image rejected as decompression bomb: {e}") from e
    if w * h > MAX_IMAGE_PIXELS:
        raise ValueError(f"Image too large for face detection: {w}x{h} ({w * h:,} pixels, limit {MAX_IMAGE_PIXELS:,})")


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


def _oauth_exchange(refresh_token: str) -> dict[str, Any]:
    """Trade a refresh token for new tokens at the Wikimedia OAuth endpoint."""
    resp = _get_session().post(
        OAUTH_TOKEN_URL,
        data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": OAUTH_CLIENT_ID,
            "client_secret": OAUTH_CLIENT_SECRET,
        },
        headers={"User-Agent": USER_AGENT},
        timeout=10,  # Must complete well within the single-flight lock's TTL.
    )
    try:
        resp.raise_for_status()
        return resp.json()
    finally:
        resp.close()


def _refresh_worker_token(user_id: int) -> str | None:
    """Return a valid access token for SDC writes, refreshing it first if it is about to expire.

    The refresh protocol (Redis single-flight lock, DB compare-and-swap, re-read after a
    lost race) lives in ``oauth_refresh`` and is shared with the web app. The worker
    keeps a refreshed token it could not save: the batch in hand can still finish with it.
    """
    tokens = oauth_refresh.refresh_tokens(
        user_id,
        label=f"user {user_id}",
        query=execute_query,
        get_redis=_get_redis_client,
        exchange=_oauth_exchange,
        keep_on_store_failure=True,
    )
    return tokens.access_token if tokens else None


def traverse_category(project: dict[str, Any]) -> int:
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
                f"Reached image limit ({MAX_IMAGES_PER_PROJECT}) for project {project['id']}, stopping traversal."
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
                        # File — collect for batch insert (skip video/audio)
                        file_title = member["title"]
                        ext = os.path.splitext(file_title)[1].lower()
                        if ext in _SKIP_EXTENSIONS:
                            continue
                        new_files.append((project["id"], member["pageid"], file_title))

                # Trim batch to stay within per-project image limit
                space_left = remaining - added_count
                if len(new_files) > space_left:
                    new_files = new_files[:space_left]

                # Batch insert all files from this API page (UNIQUE index handles duplicates)
                if new_files:
                    placeholders = ", ".join(["(%s, %s, %s, 'pending')"] * len(new_files))
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


# ---------------------------------------------------------------------------
# Image processing (detection happens in the face service — see face_client.py)
# ---------------------------------------------------------------------------


def _mark_image_error(img_id: int, message: str) -> None:
    """Record a terminal processing failure for an image."""
    execute_query(
        "UPDATE images SET status = 'error', error_message = %s WHERE id = %s",
        (message[:1000], img_id),
        fetch=False,
    )


def _download_for_detection(img_id: int, title: str) -> bytes | None:
    """Download and dimension-check one image, ready for detection.

    Returns the image bytes, or None if the image cannot be used now. A
    transient failure (network error, timeout, HTTP 429/5xx) leaves the row
    'pending' for a later cycle; anything else marks it 'error'. Keeping
    transient failures pending matters: 'error' counts as processed, so a CDN
    outage would otherwise let the project auto-complete as "no faces".
    Dimension validation stays on this side so oversized images are rejected
    before they are base64-inflated and pushed over the wire.
    """
    clean_title = title[5:] if title.startswith("File:") else title
    url = FILE_PATH_URL.format(file_title=clean_title)
    try:
        logger.debug(f"Downloading image {title}")
        image_bytes = _download_image(url)
        _validate_image_dimensions(image_bytes)
    except requests.RequestException as e:
        status = getattr(getattr(e, "response", None), "status_code", None)
        if status is None or status == 429 or status >= 500:
            logger.warning(f"Transient error downloading image {title}, leaving pending: {e}")
            return None
        logger.error(f"Error downloading image {title}: {e}")
        _mark_image_error(img_id, str(e))
        return None
    except Exception as e:
        logger.error(f"Error downloading image {title}: {e}")
        _mark_image_error(img_id, str(e))
        return None

    if COMMONS_DOWNLOAD_THROTTLE_SECONDS > 0:
        time.sleep(COMMONS_DOWNLOAD_THROTTLE_SECONDS)
    return image_bytes


def _persist_detection(
    img_id: int,
    result: face_client.DetectionResult,
    *,
    bootstrapped: bool = False,
    project_id: int | None = None,
    stored_size: tuple[int, int] | None = None,
) -> bool:
    """Store detected faces and mark the image processed.

    ``stored_size`` is ``(width, height)`` of an existing detection space the
    boxes must stay in (see ``_convert_box_to_stored_space``). Normal processing
    handles pending images that have no stored space, so it is None: boxes are
    stored as detected and the result's size becomes ``detection_width/height``.

    If the image is bootstrapped (already has a P180 depicts claim on Commons)
    and exactly one face was detected, auto-classify it as a target match.
    Multi-face bootstrapped images are left for manual classification.
    """
    face_locations = result.locations
    encodings_bytes = result.encodings
    scores = [face.score for face in result.faces]
    face_count = len(face_locations)
    detection_size = (result.width, result.height)
    if stored_size is not None and stored_size != detection_size:
        face_locations = [_convert_box_to_stored_space(loc, detection_size, stored_size) for loc in face_locations]
        detection_size = stored_size

    try:
        # Batch-insert detected faces in chunks to stay within max_allowed_packet.
        # All faces start unclassified (is_target=NULL). Classification
        # happens via human confirmation or autonomous inference.
        if face_locations:
            FACE_INSERT_CHUNK = 100  # ~200KB per chunk (2048-byte embedding + ints)
            for chunk_start in range(0, face_count, FACE_INSERT_CHUNK):
                chunk_locs = face_locations[chunk_start : chunk_start + FACE_INSERT_CHUNK]
                chunk_encs = encodings_bytes[chunk_start : chunk_start + FACE_INSERT_CHUNK]
                chunk_scores = scores[chunk_start : chunk_start + FACE_INSERT_CHUNK]
                chunk_size = len(chunk_locs)
                placeholders = ", ".join(["(%s, %s, %s, %s, %s, %s, %s, %s)"] * chunk_size)
                flat_params: list = []
                for location, enc_bytes, score in zip(chunk_locs, chunk_encs, chunk_scores):
                    top, right, bottom, left = location
                    flat_params.extend([img_id, enc_bytes, result.model_id, score, top, right, bottom, left])

                execute_query(
                    f"INSERT INTO faces "
                    f"(image_id, encoding, embedding_model, det_score, bbox_top, bbox_right, bbox_bottom, bbox_left) "
                    f"VALUES {placeholders}",
                    tuple(flat_params),
                    fetch=False,
                )

        execute_query(
            "UPDATE images SET status = 'processed', face_count = %s, "
            "detection_width = %s, detection_height = %s WHERE id = %s",
            (face_count, detection_size[0], detection_size[1], img_id),
            fetch=False,
        )

        # Auto-classify single-face bootstrapped images as target matches.
        # These images already have a P180 depicts claim on Commons, so the
        # single detected face is overwhelmingly likely to be the target person.
        # Multi-face bootstrapped images are left for manual classification.
        if bootstrapped and face_count == 1 and project_id is not None:
            auto_classified = execute_query(
                "UPDATE faces SET is_target = 1, classified_by = 'bootstrap', "
                "sdc_removal_pending = 0, classified_at = NOW() "
                "WHERE image_id = %s AND is_target IS NULL AND superseded_by IS NULL",
                (img_id,),
                fetch=False,
            )
            if auto_classified:
                execute_query(
                    "UPDATE projects SET faces_confirmed = faces_confirmed + %s WHERE id = %s",
                    (auto_classified, project_id),
                    fetch=False,
                )
                # Mark as sdc_written — this image already has P180 on Commons,
                # so there's nothing to write.  Prevents showing as SDC pending.
                execute_query(
                    "UPDATE faces SET sdc_written = 1 "
                    "WHERE image_id = %s AND is_target = 1 AND classified_by = 'bootstrap' "
                    "AND sdc_written = 0 AND superseded_by IS NULL",
                    (img_id,),
                    fetch=False,
                )
                logger.info(f"Auto-classified {auto_classified} face(s) on bootstrapped image {img_id} as target")

        return True
    except Exception as e:
        logger.error(f"Error storing faces for image {img_id}: {e}")
        _mark_image_error(img_id, str(e))
        return False


def _handle_detection_failure(img_id: int, title: str, error: FaceServiceError) -> None:
    """Apply the right retry policy to a failed detection.

    The distinction matters: a rejected image will never succeed, so it is
    marked 'error' and stops consuming batch slots. A service outage is
    transient, so the row is left 'pending' and picked up on a later poll cycle
    rather than being permanently written off because the model server happened
    to be restarting.
    """
    if isinstance(error, FaceServiceRejected):
        logger.error(f"Face service rejected image {title}: {error}")
        _mark_image_error(img_id, f"face service rejected image: {error}")
    else:
        logger.warning(f"Face service unavailable for image {title}, leaving pending: {error}")


def _process_image_wave(wave: list[dict[str, Any]], project_id: int) -> tuple[int, tuple[float, float, float]]:
    """Download, detect, and persist one wave of images.

    Three phases:

    1. Download in parallel — pure network I/O, and the slow half, so it keeps
       the same thread fan-out the per-image pipeline used.
    2. Detect in batches — one round trip per batch instead of per image. The
       client splits further on whichever of its count/byte budgets is hit
       first.
    3. Persist serially — the DB pool is small (<= 6 connections per worker
       against Toolforge's shared 20-connection ceiling) and these are short
       writes, so fanning them out would only contend for connections.

    Returns ``(processed_count, (download_seconds, detect_seconds, db_seconds))``.
    """
    t_start = time.monotonic()

    rows_by_key: dict[str, dict[str, Any]] = {}
    downloaded: list[tuple[str, bytes]] = []
    with ThreadPoolExecutor(max_workers=IMAGE_THREADS) as executor:
        futures = {}
        for img_row in wave:
            if shutdown_requested:
                break
            key = str(img_row["id"])
            rows_by_key[key] = img_row
            futures[executor.submit(_download_for_detection, img_row["id"], img_row["file_title"])] = key

        for future in as_completed(futures):
            if shutdown_requested:
                executor.shutdown(wait=False, cancel_futures=True)
                break
            key = futures[future]
            try:
                image_bytes = future.result()
            except Exception as e:
                logger.error(f"Thread error downloading {rows_by_key[key]['file_title']}: {e}")
                continue
            if image_bytes is not None:
                downloaded.append((key, image_bytes))

    t_download = time.monotonic()

    detections: dict[str, face_client.DetectionResult | FaceServiceError] = {}
    if downloaded and not shutdown_requested:
        detections = face_client.detect_faces_batch(downloaded)
    # Release the wave's image bytes before touching the DB — this is the peak
    # memory point of the whole pipeline.
    downloaded.clear()

    t_detect = time.monotonic()

    processed_count = 0
    unavailable = 0
    for key, outcome in detections.items():
        if shutdown_requested:
            break
        img_row = rows_by_key[key]
        if isinstance(outcome, FaceServiceError):
            if not isinstance(outcome, FaceServiceRejected):
                unavailable += 1
            _handle_detection_failure(img_row["id"], img_row["file_title"], outcome)
            continue
        if _persist_detection(
            img_row["id"],
            outcome,
            bootstrapped=bool(img_row.get("bootstrapped")),
            project_id=project_id,
        ):
            processed_count += 1

    # A whole wave failing as "unavailable" means the service is down, not that
    # the images were bad. Those rows stay pending by design, so without an
    # explicit error here the pipeline would stall in total silence.
    if detections and unavailable == len(detections):
        logger.error(
            "Face service unreachable at %s — %d image(s) left pending and NOT processed. "
            "Check WIKIVISAGE_FACE_SERVICE_URL and that the service is running.",
            face_client.FACE_SERVICE_URL,
            unavailable,
        )

    t_db = time.monotonic()
    return processed_count, (t_download - t_start, t_detect - t_download, t_db - t_detect)


def process_images(project: dict[str, Any]) -> int:
    """Download images, detect faces, and extract embeddings using parallel threads.

    Bootstrap images are processed first (up to a dynamic cap) so that faces
    from images already known to depict the target appear in the gallery
    quickly.  Non-bootstrap images fill the remaining batch slots.  The
    bootstrap cap uses BOOTSTRAP_TARGET_RATIO / BOOTSTRAP_MAX_RATIO to avoid
    the worker spending all its time on bootstrap downloads while
    category-discovered images wait.  The cap is enforced here instead of in
    bootstrap_from_sparql so that eligible P180 images are inserted into the
    database (subject to the global MAX_IMAGES_PER_PROJECT cap and file-type
    filtering) for correct SDC-pending bookkeeping.
    """
    project_id = project["id"]
    logger.info(f"Processing images for project {project_id}")

    # --- Compute how many bootstrap images we're allowed to process ----------
    stats = execute_query(
        "SELECT "
        "  SUM(CASE WHEN bootstrapped = 1 AND status != 'pending' THEN 1 ELSE 0 END) AS bs_done, "
        "  COUNT(*) AS total "
        "FROM images WHERE project_id = %s",
        (project_id,),
        fetch=True,
    )
    bs_already_processed = int(stats[0]["bs_done"] or 0) if stats else 0
    total_images = int(stats[0]["total"] or 0) if stats else 0

    # Dynamic cap mirrors the old bootstrap_from_sparql sizing logic
    base_cap = int(MAX_IMAGES_PER_PROJECT * BOOTSTRAP_TARGET_RATIO)
    max_cap = int(MAX_IMAGES_PER_PROJECT * BOOTSTRAP_MAX_RATIO)
    # Scale linearly based on how bootstrap-heavy the project is
    if total_images > 0:
        bs_total = execute_query(
            "SELECT COUNT(*) AS cnt FROM images WHERE project_id = %s AND bootstrapped = 1",
            (project_id,),
            fetch=True,
        )
        bs_count = int(bs_total[0]["cnt"] or 0) if bs_total else 0
        bs_ratio = bs_count / total_images
        if bs_ratio > 0.5:
            scale = min(bs_ratio, 1.0)
            bootstrap_cap = int(base_cap + (max_cap - base_cap) * (scale - 0.5) * 2)
        else:
            bootstrap_cap = base_cap
    else:
        bootstrap_cap = base_cap
    bootstrap_cap = max(MIN_BOOTSTRAP_IMAGES, bootstrap_cap)
    bootstrap_remaining = max(0, bootstrap_cap - bs_already_processed)

    # --- Build the batch: bootstrap first, then non-bootstrap ----------------
    # Bootstrap images are processed first so that faces from images already
    # known to depict the target (via existing P180 claims on Commons) appear
    # in the gallery quickly — giving the user early visual feedback and
    # providing training data for autonomous inference sooner.
    pending_images: list[dict[str, Any]] = []
    bs_to_fetch = min(BATCH_SIZE, bootstrap_remaining)
    if bs_to_fetch > 0:
        bs = execute_query(
            "SELECT id, file_title, bootstrapped FROM images "
            "WHERE project_id = %s AND status = 'pending' AND bootstrapped = 1 "
            "LIMIT %s",
            (project_id, bs_to_fetch),
            fetch=True,
        )
        if bs:
            pending_images.extend(bs)

    remaining_slots = BATCH_SIZE - len(pending_images)
    if remaining_slots > 0:
        non_bs = execute_query(
            "SELECT id, file_title, bootstrapped FROM images "
            "WHERE project_id = %s AND status = 'pending' AND bootstrapped = 0 "
            "LIMIT %s",
            (project_id, remaining_slots),
            fetch=True,
        )
        if non_bs:
            pending_images.extend(non_bs)

    if not pending_images:
        # If no non-bootstrap images remain and bootstrap cap is reached,
        # mark any leftover pending bootstrap images as 'skipped' so the
        # progress bar can reach 100%.
        if bootstrap_remaining == 0:
            skipped = execute_query(
                "UPDATE images SET status = 'skipped' "
                "WHERE project_id = %s AND status = 'pending' AND bootstrapped = 1",
                (project_id,),
                fetch=False,
            )
            if skipped:
                logger.info(f"Marked {skipped} bootstrap images as skipped (cap reached)")
                # Recount images_processed to include skipped
                execute_query(
                    "UPDATE projects SET images_processed = "
                    "(SELECT COUNT(*) FROM images WHERE project_id = %s "
                    "AND status IN ('processed', 'error', 'enriched', 'skipped')) "
                    "WHERE id = %s",
                    (project_id, project_id),
                    fetch=False,
                )
        return 0

    batch_start = time.monotonic()
    processed_count = 0
    t_download = t_detect = t_db = 0.0

    # Waves, not one big download pass — see DETECTION_WAVE_SIZE.
    for wave_start in range(0, len(pending_images), DETECTION_WAVE_SIZE):
        if shutdown_requested:
            break
        wave = pending_images[wave_start : wave_start + DETECTION_WAVE_SIZE]
        processed, timings = _process_image_wave(wave, project_id)
        processed_count += processed
        t_download += timings[0]
        t_detect += timings[1]
        t_db += timings[2]

    batch_elapsed = time.monotonic() - batch_start
    batch_size = len(pending_images) if pending_images else 0
    per_image = batch_elapsed / batch_size if batch_size > 0 else 0
    logger.info(
        f"Batch complete: {processed_count}/{batch_size} images in {batch_elapsed:.1f}s "
        f"({per_image:.2f}s/image avg) | "
        f"download={t_download:.1f}s ({IMAGE_THREADS} threads), "
        f"detect={t_detect:.1f}s (waves of {DETECTION_WAVE_SIZE}), "
        f"db={t_db:.1f}s"
    )

    # Update project stats
    execute_query(
        "UPDATE projects SET images_processed = (SELECT COUNT(*) FROM images WHERE project_id = %s AND status IN ('processed', 'error', 'enriched', 'skipped')) WHERE id = %s",
        (project_id, project_id),
        fetch=False,
    )

    return processed_count


def bootstrap_from_sparql(project: dict[str, Any]) -> int:
    """Bootstrap a project using images already linked to the QID via P180.

    Uses the Wikibase haswbstatement search API on Commons instead of SPARQL,
    since commons-query.wikimedia.org requires OAuth cookie authentication
    that is impractical for automated tools.

    Runs on every poll cycle (not just the first time). Newly discovered images
    that already have P180 on Commons are flagged with bootstrapped=1.
    Already-processed single-face images are auto-classified as target matches.
    Multi-face images (pending or processed) are left for manual classification.
    """
    logger.info(f"Attempting bootstrap for project {project['id']} with QID {project['wikidata_qid']}")

    # Enforce global image cap (same as traverse_category)
    existing_count_rows = execute_query(
        "SELECT COUNT(*) AS cnt FROM images WHERE project_id = %s",
        (project["id"],),
        fetch=True,
    )
    existing_count = existing_count_rows[0]["cnt"] if existing_count_rows else 0
    remaining_global = max(0, MAX_IMAGES_PER_PROJECT - existing_count)

    qid = project["wikidata_qid"]
    category = project["commons_category"]
    # deepcat: search keyword expects bare category name without "Category:" prefix
    if category.startswith("Category:"):
        category = category[len("Category:") :]
    search_params: dict[str, Any] = {
        "action": "query",
        "list": "search",
        "srsearch": f'haswbstatement:P180={qid} deepcat:"{category}"',
        "srnamespace": "6",
        "srlimit": "max",
        "format": "json",
    }

    try:
        flagged_count = 0
        inserted_count = 0
        results_seen = 0

        while True:
            if shutdown_requested:
                break

            resp = _api_request(COMMONS_API_URL, params=search_params)
            data = resp.json()
            results = data.get("query", {}).get("search", [])

            if not results:
                break

            for result in results:
                if shutdown_requested:
                    break

                results_seen += 1
                page_id = result.get("pageid")
                title = result.get("title")

                if not page_id or not title:
                    continue

                # Skip non-image files (video, audio) — same filter as traverse_category
                ext = os.path.splitext(title)[1].lower()
                if ext in _SKIP_EXTENSIONS:
                    continue

                exists = execute_query(
                    "SELECT id, status FROM images WHERE project_id = %s AND commons_page_id = %s",
                    (project["id"], page_id),
                    fetch=True,
                )

                if exists:
                    image_id = exists[0]["id"]
                    img_status = exists[0]["status"]

                    # Flag as bootstrapped regardless of current status —
                    # ensures all P180 images are marked so they don't show
                    # as SDC pending in the UI.
                    affected = execute_query(
                        "UPDATE images SET bootstrapped = 1 WHERE id = %s AND bootstrapped != 1",
                        (image_id,),
                        fetch=False,
                    )
                    if affected:
                        flagged_count += 1

                    # For already-processed images with faces, always run the
                    # reconciliation logic when we find the image via P180,
                    # regardless of whether the bootstrapped flag changed.
                    if img_status == "processed":
                        # Mark any human/model-classified target faces as
                        # sdc_written since P180 already exists on Commons.
                        marked = execute_query(
                            "UPDATE faces SET sdc_written = 1 "
                            "WHERE image_id = %s AND is_target = 1 AND sdc_written = 0 "
                            "AND superseded_by IS NULL",
                            (image_id,),
                            fetch=False,
                        )
                        if marked:
                            logger.info(
                                f"Bootstrap marked {marked} already-classified face(s) "
                                f"as sdc_written on image {image_id} (P180 exists on Commons)"
                            )

                        # Auto-classify single unclassified faces as target
                        # matches (same logic as _persist_detection).
                        # Perform this in a single conditional UPDATE to avoid a per-image COUNT query.
                        auto = execute_query(
                            "UPDATE faces "
                            "SET is_target = 1, classified_by = 'bootstrap', "
                            "sdc_removal_pending = 0, classified_at = NOW() "
                            "WHERE image_id = %s "
                            "AND is_target IS NULL "
                            "AND superseded_by IS NULL "
                            "AND (SELECT COUNT(*) FROM faces "
                            "     WHERE image_id = %s AND superseded_by IS NULL) = 1",
                            (image_id, image_id),
                            fetch=False,
                        )
                        if auto:
                            execute_query(
                                "UPDATE projects SET faces_confirmed = faces_confirmed + %s WHERE id = %s",
                                (auto, project["id"]),
                                fetch=False,
                            )
                            logger.info(
                                f"Bootstrap auto-classified {auto} face(s) on "
                                f"already-processed image {image_id} as target"
                            )
                else:
                    # New image — insert as pending + bootstrapped.
                    # Respects MAX_IMAGES_PER_PROJECT cap for new inserts, but the
                    # flagging loop above always runs for existing images regardless.
                    if inserted_count >= remaining_global:
                        continue

                    affected = execute_query(
                        """
                        INSERT IGNORE INTO images (project_id, commons_page_id, file_title, status, bootstrapped)
                        VALUES (%s, %s, %s, 'pending', 1)
                        """,
                        (project["id"], page_id, title),
                        fetch=False,
                    )
                    if affected:
                        flagged_count += 1
                        inserted_count += 1
                    else:
                        # Race: another process inserted this image between our
                        # SELECT and INSERT.  Ensure it gets the bootstrap flag.
                        execute_query(
                            "UPDATE images SET bootstrapped = 1 "
                            "WHERE project_id = %s AND commons_page_id = %s AND bootstrapped != 1",
                            (project["id"], page_id),
                            fetch=False,
                        )

            # Paginate: check for continuation token
            continuation = data.get("continue", {})
            sr_offset = continuation.get("sroffset")
            if sr_offset is None:
                break
            search_params["sroffset"] = sr_offset
            logger.info(
                f"Bootstrap pagination: offset {sr_offset} (flagged={flagged_count}, inserted={inserted_count})"
            )

        if flagged_count > 0:
            # Update images_total to include any newly inserted bootstrap images
            execute_query(
                "UPDATE projects SET images_total = (SELECT COUNT(*) FROM images WHERE project_id = %s) WHERE id = %s",
                (project["id"], project["id"]),
                fetch=False,
            )
            logger.info(
                f"Bootstrap flagged {flagged_count} images for project {project['id']}"
                f" ({inserted_count} new inserts, {flagged_count - inserted_count} existing flagged,"
                f" scanned={results_seen})"
            )

        return flagged_count

    except Exception as e:
        logger.error(f"Bootstrap search failed: {e}")

    return 0


# ---------------------------------------------------------------------------
# Experimental LiftWing metadata pre-filter (Phase 2, shadow mode)
# ---------------------------------------------------------------------------

_HTML_TAG_RE = re.compile(r"<[^>]+>")
# Descriptions regularly embed wikitext templates ({{…}}) that are pure noise
# to the pre-filter; they are removed rather than forwarded as tokens.
_WIKITEXT_TEMPLATE_RE = re.compile(r"\{\{.*?\}\}", re.DOTALL)


def _plain_text(value: Any, limit: int = 250) -> str:
    """Flatten the HTML/wikitext fragment extmetadata returns into plain text."""
    text = html.unescape(str(value or ""))
    text = _HTML_TAG_RE.sub(" ", text)
    text = _WIKITEXT_TEMPLATE_RE.sub(" ", text)
    return " ".join(text.split())[:limit]


def _fetch_metadata(titles: list[str]) -> dict[str, dict[str, Any]]:
    """Batched Commons metadata for the pre-filter, keyed by file title.

    One POST for the whole list (50 titles fit comfortably in a form body,
    unlike a URL). Returns ``{}`` on any failure so the caller leaves those
    images unchecked for a later cycle instead of dropping them.
    """
    if not titles:
        return {}
    try:
        response = _api_request(
            COMMONS_API_URL,
            method="post",
            data={
                "action": "query",
                "format": "json",
                "formatversion": "2",
                "prop": "imageinfo|categories",
                "iiprop": "extmetadata|mime",
                "cllimit": "50",
                "maxlag": "5",
                "titles": "|".join(titles),
            },
            timeout=30,
        )
        data = response.json()
    except Exception as exc:
        logger.warning(f"Pre-filter: Commons metadata request failed: {exc}")
        return {}

    pages = data.get("query", {}).get("pages", [])
    if isinstance(pages, dict):
        pages = list(pages.values())
    by_title: dict[str, dict[str, Any]] = {page["title"]: page for page in pages if page.get("title")}

    # The API answers under canonical titles (e.g. spaces instead of
    # underscores, first letter capitalised) and lists the corrections in
    # ``query.normalized``. Look each requested title up through that map so a
    # non-canonical title does not silently get an empty description.
    normalized = {
        entry["from"]: entry["to"]
        for entry in (data.get("query", {}).get("normalized") or [])
        if isinstance(entry, dict) and entry.get("from") and entry.get("to")
    }

    result: dict[str, dict[str, Any]] = {}
    for requested in titles:
        page = by_title.get(requested)
        if page is None:
            page = by_title.get(normalized.get(requested, ""))
        if page is None:
            continue
        info = (page.get("imageinfo") or [{}])[0]
        extmetadata = info.get("extmetadata") or {}
        description = _plain_text((extmetadata.get("ImageDescription") or {}).get("value", ""))
        if not description:
            description = _plain_text((extmetadata.get("ObjectName") or {}).get("value", ""))
        categories = tuple(
            category.get("title", "") for category in (page.get("categories") or []) if category.get("title")
        )
        result[requested] = {
            "description": description,
            "categories": categories,
            "mime": info.get("mime", ""),
        }
    return result


def _liftwing_service_down() -> bool:
    """True when the last probe marked LiftWing down (pre-filter backs off)."""
    try:
        rows = execute_query("SELECT status FROM service_status WHERE name = 'liftwing'", fetch=True)
    except DatabaseError:
        return False
    return bool(rows) and rows[0]["status"] == "down"


def _store_prefilter_verdicts(updates: list[tuple[int, str | None, int]]) -> None:
    """Apply a batch of ``(keep, reason, image_id)`` verdicts atomically."""

    def _txn(conn: Any, cursor: Any) -> int:
        for keep, reason, image_id in updates:
            cursor.execute(
                "UPDATE images SET prefilter_keep = %s, prefilter_reason = %s, prefilter_checked_at = NOW() "
                "WHERE id = %s AND prefilter_checked_at IS NULL",
                (keep, reason, image_id),
            )
        return len(updates)

    execute_transaction(_txn)


def prefilter_images(project: dict[str, Any]) -> int:
    """Record shadow-mode LiftWing verdicts for crawled images.

    Experimental (Phase 2). Changes no behaviour: whatever the verdict, the
    image is still processed — the verdicts land on the image row for the
    offline readout (``scripts/eval_prefilter.py``). Bootstrapped images always
    bypass the filter (they carry a P180 claim already), and ``enforce`` is not
    built yet, so any mode other than ``off`` behaves as shadow.

    Shadow verdicts are metadata-only, so an image's processing status is
    irrelevant: this runs *after* ``process_images`` and selects any not-yet
    checked image, not just pending ones, so a backlog larger than one cycle's
    cap is fully covered over time. (A future ``enforce`` mode is different:
    it must select pending images and run *before* detection, because dropping
    has to happen before a download.)

    Fail open: an outage, a rate limit or an unparseable answer leaves the batch
    unchecked for a later cycle; the circuit breaker in ``liftwing_client``
    backs off, and a service already marked down is skipped entirely. Files the
    model omitted are kept and recorded with a NULL reason. Real request
    outcomes are mirrored into ``service_status`` so the banner reflects what
    the pre-filter actually experiences, not just what the probe sees. While a
    pass runs, the caller's claim is refreshed after every stored batch, so a
    pass longer than the claim expiry (a sweep can hold a project for many
    minutes with nothing else refreshing it) cannot be claimed by the other
    worker too; a failed refresh stops the pass, leaving the remaining files
    for a later pass. Returns the number of images whose verdicts were stored.
    """
    if project.get("prefilter_mode", "off") == "off":
        return 0
    if _liftwing_service_down():
        logger.info(f"Pre-filter: LiftWing is marked down, skipping project {project['id']}")
        return 0

    project_id = project["id"]
    try:
        rows = execute_query(
            "SELECT id, commons_page_id, file_title FROM images "
            "WHERE project_id = %s AND bootstrapped = 0 AND prefilter_checked_at IS NULL "
            "ORDER BY id LIMIT %s",
            (project_id, LIFTWING_MAX_IMAGES),
            fetch=True,
        )
    except DatabaseError:
        logger.exception(f"Pre-filter: could not load unchecked images for project {project_id}")
        return 0
    rows = list(rows or [])
    if not rows:
        return 0

    client = liftwing_client.get_client()
    checked = 0
    failed = False
    for start in range(0, len(rows), LIFTWING_BATCH_SIZE):
        if shutdown_requested:
            break
        batch = rows[start : start + LIFTWING_BATCH_SIZE]
        metadata = _fetch_metadata([row["file_title"] for row in batch])
        if not metadata:
            logger.warning(f"Pre-filter: no metadata for {len(batch)} image(s); leaving them unchecked")
            break

        items = []
        for row in batch:
            info = metadata.get(row["file_title"], {})
            items.append(
                liftwing_client.MetadataItem(
                    page_id=int(row["commons_page_id"]),
                    title=row["file_title"],
                    description=info.get("description", ""),
                    categories=info.get("categories", ()),
                    mime=info.get("mime", ""),
                )
            )

        t0 = time.monotonic()
        verdicts = client.filter_metadata(items)
        elapsed = time.monotonic() - t0
        if verdicts is None:
            failed = True
            logger.warning(f"Pre-filter: LiftWing returned no verdicts for {len(batch)} image(s) in {elapsed:.1f}s")
            _record_liftwing_failure("pre-filter request failed (see worker log)")
            break  # fail open: keep everything and retry on a later cycle
        logger.info(f"Pre-filter: LiftWing batch of {len(batch)} in {elapsed:.1f}s")

        updates: list[tuple[int, str | None, int]] = []
        for row in batch:
            verdict = verdicts.get(int(row["commons_page_id"]))
            keep = True if verdict is None else verdict.keep
            reason = None if verdict is None or not verdict.reason else verdict.reason
            updates.append((1 if keep else 0, reason, row["id"]))
        try:
            _store_prefilter_verdicts(updates)
        except DatabaseError:
            logger.exception(f"Pre-filter: could not store verdicts for project {project_id}")
            break
        checked += len(batch)
        # Keep our claim fresh for the whole pass: a sweep runs in its own
        # thread while the main loop is idle, so nothing else refreshes it.
        # A failed refresh stops the pass instead of processing on without a
        # live claim; the remaining files are retried on a later pass.
        try:
            execute_query(
                "UPDATE projects SET worker_claimed_at = NOW() WHERE id = %s AND worker_claimed_by = %s",
                (project_id, _worker_id),
                fetch=False,
            )
        except DatabaseError:
            logger.warning(f"Pre-filter: could not refresh the claim on project {project_id}; stopping the pass")
            break

    if checked and not failed:
        _record_liftwing_success()
    if checked:
        logger.info(f"Pre-filter (shadow): recorded LiftWing verdicts for {checked} image(s) in project {project_id}")
    return checked


def _claim_shadow_project() -> dict[str, Any] | None:
    """Atomically claim one project the processing pipelines will not take.

    ``_claim_active_projects`` only takes active projects with pending images
    or unclassified faces, and ``_claim_inference_projects`` the ones that
    crossed ``min_confirmed`` — so a completed, paused or fully processed
    project would never receive pre-filter verdicts even with the mode on.
    This claim picks exactly those projects (non-deleted, mode not ``off``,
    unchecked non-bootstrapped images still present), so a sweep can never
    delay work the pipelines would otherwise do; projects with a pending SDC
    write request are left to the SDC path. The work predicates below mirror
    those of ``_claim_active_projects`` / ``_claim_inference_projects``; keep
    them in sync. Ordering by last claim time rotates sweeps across projects:
    a just-swept project sorts last, so it is only picked again once every
    other candidate has been swept.
    """

    def _txn(conn: Any, cursor: Any) -> dict[str, Any] | None:
        cursor.execute(
            "SELECT p.id FROM projects p "
            "WHERE p.status != 'deleted' "
            "AND p.prefilter_mode != 'off' "
            "AND p.sdc_write_requested = 0 "
            "AND (p.worker_claimed_by IS NULL "
            "     OR p.worker_claimed_at < NOW() - INTERVAL %s MINUTE) "
            "AND EXISTS ("
            "  SELECT 1 FROM images i "
            "  WHERE i.project_id = p.id AND i.bootstrapped = 0 AND i.prefilter_checked_at IS NULL"
            ") "
            "AND NOT ("
            "  (p.status = 'active' AND ("
            "    NOT EXISTS (SELECT 1 FROM images i WHERE i.project_id = p.id) "
            "    OR EXISTS (SELECT 1 FROM images i WHERE i.project_id = p.id AND i.status = 'pending') "
            "    OR EXISTS ("
            "      SELECT 1 FROM faces f JOIN images i ON f.image_id = i.id "
            "      WHERE i.project_id = p.id AND f.is_target IS NULL AND f.superseded_by IS NULL"
            "    )"
            "  )) "
            "  OR (p.status = 'completed' AND EXISTS ("
            "    SELECT 1 FROM faces f JOIN images i ON f.image_id = i.id "
            "    WHERE i.project_id = p.id AND f.is_target IS NULL AND f.superseded_by IS NULL "
            "    AND f.sdc_removal_pending = 0"
            "  ) AND ("
            "    SELECT COUNT(*) FROM faces f JOIN images i ON f.image_id = i.id "
            "    WHERE i.project_id = p.id AND f.is_target = 1 AND f.superseded_by IS NULL "
            "    AND f.classified_by_user_id IS NOT NULL"
            "  ) >= p.min_confirmed)"
            ") "
            "ORDER BY COALESCE(p.worker_claimed_at, '1970-01-01') ASC, p.id DESC "
            "LIMIT 1 "
            "FOR UPDATE SKIP LOCKED",
            (CLAIM_EXPIRY_MINUTES,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        project_id = row["id"]
        cursor.execute(
            "UPDATE projects SET worker_claimed_by = %s, worker_claimed_at = NOW() WHERE id = %s",
            (_worker_id, project_id),
        )
        cursor.execute("SELECT p.* FROM projects p WHERE p.id = %s", (project_id,))
        return dict(cursor.fetchone())

    try:
        project = execute_transaction(_txn)
    except DatabaseError:
        logger.exception("Shadow pre-filter: could not claim a project")
        return None
    if project:
        logger.info(f"Shadow pre-filter: claimed project {project['id']} (status={project.get('status')})")
    return project


def _sweep_shadow_prefilter() -> int:
    """Record one project's worth of shadow verdicts (up to LIFTWING_MAX_IMAGES).

    Runs from the background sweep thread so projects the processing pipelines
    never claim are covered too. The claim is released when the pass ends, so
    other cycles and the other worker instance can continue the backlog.
    Fail-open and paced exactly like the in-pipeline call — only the trigger
    differs; verdicts still change nothing.
    """
    if _liftwing_service_down():
        return 0
    project = _claim_shadow_project()
    if project is None:
        return 0
    try:
        return prefilter_images(project)
    except Exception:
        logger.exception(f"Shadow pre-filter: sweep failed for project {project.get('id')}")
        return 0
    finally:
        _release_project(project["id"])


def _shadow_prefilter_loop() -> None:
    """Daemon-thread loop (started by ``main()``) recording shadow verdicts.

    Runs independently of the main loop because one pass can take minutes,
    which would otherwise stall the heartbeat refreshes the Toolforge health
    check watches. Sleeps in one-second slices so shutdown stays prompt.
    """
    if LIFTWING_SWEEP_INTERVAL <= 0:
        return
    logger.info(
        f"Shadow pre-filter sweep started (pause {LIFTWING_SWEEP_INTERVAL}s between "
        f"passes, up to {LIFTWING_MAX_IMAGES} files per project per pass)"
    )
    while not shutdown_requested:
        try:
            _sweep_shadow_prefilter()
        except Exception:
            logger.exception("Shadow pre-filter: unexpected sweep error")
        for _ in range(LIFTWING_SWEEP_INTERVAL):
            if shutdown_requested:
                break
            time.sleep(1)


def _persist_inference_batch(project_id: int, batch: list[tuple[int, int, float]]) -> tuple[int, int]:
    """Apply a batch under the project lock shared with human classifications.

    The candidate snapshot can predate an approval, undo or queued P180 removal.
    Re-read eligibility and sibling targets before writing any model decisions.
    """
    if not batch:
        return 0, 0

    def _apply(conn, cursor):
        cursor.execute(
            "SELECT id FROM projects WHERE id = %s AND status IN ('active', 'completed') FOR UPDATE",
            (project_id,),
        )
        if not cursor.fetchone():
            return 0, 0

        placeholders = ",".join(["%s"] * len(batch))
        cursor.execute(
            "SELECT f.id, f.image_id, EXISTS ("
            "  SELECT 1 FROM faces t WHERE t.image_id = f.image_id "
            "  AND t.is_target = 1 AND t.superseded_by IS NULL"
            ") AS image_has_target "
            "FROM faces f JOIN images i ON f.image_id = i.id "
            f"WHERE f.id IN ({placeholders}) AND i.project_id = %s "
            "AND f.is_target IS NULL AND f.classified_by_user_id IS NULL "
            "AND f.superseded_by IS NULL AND f.sdc_removal_pending = 0 "
            "AND f.embedding_model = %s FOR UPDATE",
            (*[face_id for face_id, _, _ in batch], project_id, face_client.EXPECTED_MODEL_ID),
        )
        live = {row["id"]: row for row in cursor.fetchall()}
        updates = [
            (face_id, 0 if live[face_id]["image_has_target"] else is_target, distance)
            for face_id, is_target, distance in batch
            if face_id in live
        ]
        if not updates:
            return 0, 0

        target_cases = " ".join(["WHEN %s THEN %s" for _ in updates])
        conf_cases = " ".join(["WHEN %s THEN %s" for _ in updates])
        id_placeholders = ",".join(["%s"] * len(updates))
        params: list = []
        for face_id, is_target, _ in updates:
            params.extend([face_id, is_target])
        for face_id, _, distance in updates:
            params.extend([face_id, distance])
        params.extend(face_id for face_id, _, _ in updates)
        cursor.execute(
            "UPDATE faces SET "
            f"is_target = CASE id {target_cases} END, "
            f"confidence = CASE id {conf_cases} END, "
            "classified_by = 'model', classified_at = NOW() "
            f"WHERE id IN ({id_placeholders}) AND is_target IS NULL "
            "AND classified_by_user_id IS NULL AND superseded_by IS NULL "
            "AND sdc_removal_pending = 0",
            tuple(params),
        )
        return cursor.rowcount, sum(is_target for _, is_target, _ in updates)

    return execute_transaction(_apply)


def run_autonomous_inference(project: dict[str, Any]) -> int:
    """
    Run model inference on unclassified faces based on confirmed faces.

    The inference gate counts only faces with ``classified_by_user_id IS NOT NULL``
    (i.e., human-confirmed faces). Projects that have enough bootstrap/model
    ``is_target = 1`` faces but *zero* human-confirmed faces must still skip
    autonomous inference, even if encodings exist.

    The following doctest exercises this behavior by simulating a project where
    the COUNT query for human-confirmed faces returns 0 while the confirmed-
    encodings query returns rows. In this case, inference must be skipped and
    the function should return 0.

    >>> project = {"id": 1, "min_confirmed": 5}
    >>> # Save the real execute_query so we can restore it after the test.
    >>> _real_execute_query = execute_query
    >>> def _fake_execute_query(sql, params, fetch=False):
    ...     # Simulate 0 human-confirmed faces for the COUNT(*) query.
    ...     if "COUNT(*) AS cnt" in sql:
    ...         return [{"cnt": 0}]
    ...     # Simulate existing confirmed encodings for the second query.
    ...     if "f.encoding FROM faces f" in sql:
    ...         return [{"encoding": b"fake-encoding"}]
    ...     return []
    >>> try:
    ...     execute_query = _fake_execute_query
    ...     run_autonomous_inference(project)
    ... finally:
    ...     execute_query = _real_execute_query
    0
    """
    t_start = time.monotonic()
    min_confirmed = project.get("min_confirmed", 5)

    # Count human-confirmed target faces only.  Bootstrap auto-classifications
    # do NOT count toward the gate — a human must review at least min_confirmed
    # faces before the model is allowed to classify autonomously.  This prevents
    # the model from running before the user has seen any results.
    count_row = execute_query(
        "SELECT COUNT(*) AS cnt FROM faces f "
        "JOIN images i ON f.image_id = i.id "
        "WHERE i.project_id = %s AND f.is_target = 1 AND f.superseded_by IS NULL "
        "AND f.classified_by_user_id IS NOT NULL",
        (project["id"],),
        fetch=True,
    )
    human_confirmed = count_row[0]["cnt"] if count_row else 0

    if human_confirmed < min_confirmed:
        logger.info(
            f"Project {project['id']}: {human_confirmed} human-confirmed target faces < {min_confirmed} required, skipping inference"
        )
        return 0

    logger.info(f"Running autonomous inference for project {project['id']} ({human_confirmed} human-confirmed faces)")

    # Get confirmed faces
    # Only embeddings from the current model are comparable with each other.
    confirmed_rows = execute_query(
        """
        SELECT f.id, f.encoding FROM faces f
        JOIN images i ON f.image_id = i.id
        WHERE i.project_id = %s AND f.is_target = 1 AND f.superseded_by IS NULL
          AND f.embedding_model = %s
        """,
        (project["id"], face_client.EXPECTED_MODEL_ID),
        fetch=True,
    )

    if not confirmed_rows:
        return 0

    t_query = time.monotonic()

    # Load all confirmed embeddings to compute the centroid. Each is 512 float32
    # values (2048 bytes). Even 10K confirmed faces = ~20 MB — well within Toolforge
    # memory limits. Batching would add complexity with no practical benefit.
    confirmed_embeddings = []
    for row in confirmed_rows:
        validated = _load_embedding(row["encoding"], face_client.EMBEDDING_DIM, row.get("id", "confirmed"))
        if validated is None:
            continue
        confirmed_embeddings.append(validated)

    if not confirmed_embeddings:
        logger.warning(f"Project {project['id']}: all confirmed embeddings invalid, skipping inference")
        return 0

    try:
        # Mean of unit vectors, renormalised: the centroid must itself be a unit
        # vector for the cosine distance in _face_distance to be a dot product.
        centroid = np.mean(confirmed_embeddings, axis=0)
        centroid_norm = float(np.linalg.norm(centroid))
        if centroid_norm < 1e-6:
            logger.warning(f"Project {project['id']}: confirmed embeddings cancel out (norm {centroid_norm}), skipping")
            return 0
        centroid = centroid / centroid_norm
    except Exception:
        logger.exception(f"Project {project['id']}: failed to compute centroid")
        return 0
    t_centroid = time.monotonic()

    # Select candidate faces for model classification. A face is eligible only if:
    #   (a) unclassified (is_target IS NULL) and not superseded
    #   (b) no human has interacted with it (classified_by_user_id IS NULL)
    #   (c) no P180 removal is queued for it (sdc_removal_pending = 0). A model
    #       guess must not re-target a face whose claim a human asked to remove:
    #       the removal phase would then strip the claim of a fresh model match.
    #       Excluded faces become eligible again once the worker resolves the
    #       queue (or the user classifies them directly).
    # Bootstrapped images are included: bootstrap auto-classifies the single best face,
    # but multi-face images have additional unclassified faces that need inference.
    # Sibling state is deliberately not snapshotted here: it can change between this
    # query and the locked write (a human approval or undo), so _persist_inference_batch
    # re-reads it and vetoes a match only when a live target still exists.
    unclassified_rows = execute_query(
        """
        SELECT f.id, f.image_id, f.encoding
        FROM faces f
        JOIN images i ON f.image_id = i.id
        WHERE i.project_id = %s
          AND f.is_target IS NULL
          AND f.superseded_by IS NULL
          AND f.classified_by_user_id IS NULL
          AND f.sdc_removal_pending = 0
          AND f.embedding_model = %s
        """,
        (project["id"], face_client.EXPECTED_MODEL_ID),
        fetch=True,
    )

    candidate_count = len(unclassified_rows) if unclassified_rows else 0
    logger.info(
        f"Project {project['id']}: {candidate_count} candidate faces for inference (unclassified, no human edits)"
    )

    classified_count = 0
    threshold = project.get("distance_threshold", DEFAULT_DISTANCE_THRESHOLD)

    # Compute all classifications in memory first (pure numpy, very fast),
    # then batch-UPDATE in chunks to minimize DB round-trips.
    UPDATE_BATCH_SIZE = 500
    pending_updates: list[tuple[int, int, float]] = []  # (face_id, is_target, distance)

    t_distance_start = time.monotonic()

    # Phase 1: compute distances for all candidates
    face_distances: list[tuple[int, int, float]] = []  # (face_id, image_id, distance)
    for row in unclassified_rows:
        if shutdown_requested:
            break

        embedding = _load_embedding(row["encoding"], face_client.EMBEDDING_DIM, row["id"])
        if embedding is None:
            continue
        distance = _face_distance(centroid, embedding)
        face_distances.append((row["id"], row["image_id"], float(distance)))

    # Phase 2: per-image dedup — only the closest face below threshold is a match;
    # all other faces on the same image are rejected (a person appears once per image).
    # Compute this from distance alone, even if the image had a sibling target when
    # the candidates were selected. That target may have been undone since, and the
    # locked write re-checks live sibling state before vetoing a match.
    best_per_image: dict[int, tuple[int, float]] = {}  # image_id -> (face_id, distance)
    for face_id, image_id, distance in face_distances:
        if distance < threshold:
            if image_id not in best_per_image or distance < best_per_image[image_id][1]:
                best_per_image[image_id] = (face_id, distance)

    best_face_ids = {face_id for face_id, _ in best_per_image.values()}

    for face_id, _image_id, distance in face_distances:
        if face_id in best_face_ids:
            pending_updates.append((face_id, 1, distance))
        else:
            pending_updates.append((face_id, 0, distance))

    t_distance = time.monotonic()

    # Revalidate each batch under the same project lock as human edits.
    t_db_start = time.monotonic()
    matches = 0
    for i in range(0, len(pending_updates), UPDATE_BATCH_SIZE):
        if shutdown_requested:
            break
        batch = pending_updates[i : i + UPDATE_BATCH_SIZE]
        if not batch:
            break

        applied, batch_matches = _persist_inference_batch(project["id"], batch)
        classified_count += applied
        matches += batch_matches
    t_db = time.monotonic()

    logger.info(
        f"Inference complete for project {project['id']}: "
        f"{classified_count} classified ({matches} matches, {classified_count - matches} non-matches) | "
        f"query={t_query - t_start:.2f}s, centroid={t_centroid - t_query:.2f}s, "
        f"distances={t_distance - t_distance_start:.2f}s, "
        f"db_update={t_db - t_db_start:.2f}s, total={t_db - t_start:.2f}s"
    )

    # Record which settings were used for this inference run so the web UI
    # can detect whether a re-run is needed after settings change.
    try:
        execute_query(
            "UPDATE projects SET last_inference_threshold = %s, last_inference_min_confirmed = %s WHERE id = %s",
            (threshold, min_confirmed, project["id"]),
            fetch=False,
        )
    except Exception:
        logger.warning(f"Project {project['id']}: failed to record inference settings", exc_info=True)

    return classified_count


# Map Wikimedia API error codes to user-friendly messages
_SDC_ERROR_MESSAGES: dict[str, str] = {
    "permissiondenied": "Your account lacks edit permissions. Please re-register the OAuth consumer with the 'Edit existing pages' grant.",
    "notloggedin": "Your session has expired. Please log in again.",
    "badtoken": "Authentication token expired. Please try again or log in again.",
    "protectedpage": "This page is protected and cannot be edited.",
    "ratelimited": "Too many edits in a short period. Please wait and try again.",
    "readonly": "The Wikimedia database is currently in read-only mode. Please try again later.",
    "maxlag": "The Wikimedia servers are busy. Please try again later.",
    "no-such-entity": "A Commons file was deleted or its structured data entity does not exist. The file was skipped.",
}

# Errors that affect a single entity (file/page) rather than the entire session.
# These are skipped per-face instead of aborting the whole write batch.
_SDC_PER_FACE_SKIP_ERRORS = frozenset({"no-such-entity"})


def _sdc_error_message(error_code: str, error_info: str) -> str:
    """Return a user-friendly error message for an SDC API error code."""
    return _SDC_ERROR_MESSAGES.get(error_code, f"{error_code}: {error_info}")


def _clear_sdc_dedup_row(commons_page_id: int, qid: str) -> None:
    """Delete the page/QID deduplication row once its P180 claim left Commons.

    A surviving row with ``written_at`` set would make the next write run trust
    it and mark the face written without restoring the claim, so every removal
    path clears the row after the claim is gone.
    """
    execute_query(
        "DELETE FROM sdc_claims WHERE commons_page_id = %s AND wikidata_qid = %s",
        (commons_page_id, qid),
        fetch=False,
    )


def _keepalive_sdc_write(project_id: int) -> None:
    """Refresh liveness signals during a long SDC write run.

    ``write_sdc_claims`` runs synchronously on the main thread, so the heartbeat
    blocks in the main loop cannot run while it works. Without this, a run
    longer than five minutes would let ``healthcheck.sh`` see a stale heartbeat
    file (Toolforge restarts the pod mid-write), and a run longer than
    ``CLAIM_EXPIRY_MINUTES`` would let the other worker claim the same project
    and write the same queue concurrently. All three refreshes are best-effort.
    """
    try:
        execute_query(
            "REPLACE INTO worker_heartbeat (id, last_seen) VALUES (1, NOW())",
            fetch=False,
        )
    except DatabaseError:
        pass
    _touch_heartbeat_file()
    try:
        execute_query(
            "UPDATE projects SET worker_claimed_at = NOW() WHERE id = %s AND worker_claimed_by = %s",
            (project_id, _worker_id),
            fetch=False,
        )
    except DatabaseError:
        pass


def _has_pending_sdc_work(project_id: int) -> bool:
    """Check the same actionable writes and removals as the two SDC phases."""
    rows = execute_query(
        "SELECT EXISTS ("
        "  SELECT 1 FROM faces f JOIN images i ON f.image_id = i.id "
        "  WHERE i.project_id = %s AND f.superseded_by IS NULL AND ("
        "    (f.is_target = 1 AND f.sdc_written = 0 "
        "      AND f.classified_by != 'bootstrap' AND i.bootstrapped = 0) "
        "    OR (f.sdc_removal_pending = 1 AND NOT EXISTS ("
        "      SELECT 1 FROM faces f2 WHERE f2.image_id = f.image_id "
        "      AND f2.is_target = 1 AND f2.superseded_by IS NULL AND f2.id != f.id"
        "    ))"
        "  )"
        ") AS pending",
        (project_id,),
        fetch=True,
    )
    if not rows:
        raise DatabaseError("Could not read outstanding SDC work")
    return bool(rows[0]["pending"])


# Faces fetched per write pass, and pages per removal pass.
_SDC_WRITE_BATCH = 50
_SDC_REMOVAL_BATCH = 50

# While writing, the stop button is re-checked (and heartbeat, health-check file
# and project claim refreshed) every this many faces. The check made once per
# batch would otherwise allow up to _SDC_WRITE_BATCH unwanted Commons edits
# between a click on "Stop" and the worker noticing.
_PER_FACE_CANCEL_CHECK_EVERY = 5

# A face whose page is claimed by another project that is still writing is not
# written now, but the batch query selects it again at once. Without a pause the
# loop would hammer the shared database until that claim is written or expires
# (CLAIM_EXPIRY_MINUTES). After a pass in which EVERY face was held, wait, and
# give up after this many such passes in a row, leaving those faces for the next
# send and telling the user.
_SDC_BLOCKED_BACKOFF_SECONDS = 20
_SDC_BLOCKED_MAX_PASSES = 6

# Commons answers with one of these when the token is unusable. Each is handled
# the same way: refresh the token (bounded by MAX_TOKEN_RETRIES) and try again.
_SDC_AUTH_ERRORS = frozenset({"badtoken", "permissiondenied", "notloggedin", "mwoauth-invalid-authorization"})

_SDC_FACES_TO_WRITE_SQL = """
    SELECT f.id as face_id, i.id as image_id, i.commons_page_id
    FROM faces f
    JOIN images i ON f.image_id = i.id
    WHERE i.project_id = %s AND f.is_target = 1 AND f.sdc_written = 0
    AND f.classified_by != 'bootstrap' AND i.bootstrapped = 0
    AND f.superseded_by IS NULL
    LIMIT %s
"""

_SDC_PAGES_TO_REMOVE_SQL = """
    SELECT DISTINCT i.commons_page_id
    FROM faces f
    JOIN images i ON f.image_id = i.id
    WHERE i.project_id = %s AND f.sdc_removal_pending = 1
    AND f.superseded_by IS NULL
    AND NOT EXISTS (
        SELECT 1 FROM faces f2
        WHERE f2.image_id = f.image_id AND f2.is_target = 1
        AND f2.superseded_by IS NULL AND f2.id != f.id
    )
    LIMIT %s
"""

_SDC_REMOVAL_STILL_PENDING_SQL = """
    SELECT 1 FROM faces f
    JOIN images i ON f.image_id = i.id
    WHERE i.commons_page_id = %s AND i.project_id = %s
    AND f.sdc_removal_pending = 1 AND f.superseded_by IS NULL
    AND NOT EXISTS (
        SELECT 1 FROM faces f2
        WHERE f2.image_id = f.image_id AND f2.is_target = 1
        AND f2.superseded_by IS NULL AND f2.id != f.id
    )
    LIMIT 1
"""


class _Step(enum.Enum):
    """What happened to one face (write phase) or one page (removal phase)."""

    DONE = enum.auto()  # handled and counted in the run's totals
    SKIPPED = enum.auto()  # nothing changed; the item stays pending for a later pass
    BLOCKED = enum.auto()  # another project holds a fresh claim on the page; retry after a pause
    UNVERIFIED = enum.auto()  # the face could not be re-read from the database
    STOP = enum.auto()  # the run is over (aborted or cancelled): do no more work


def _abort_sdc_run(project_id: int, message: str) -> None:
    """End the run: clear the request flag and record why it stopped."""
    execute_query(
        "UPDATE projects SET sdc_write_requested = 0, sdc_write_user_id = NULL, sdc_write_error = %s WHERE id = %s",
        (message, project_id),
        fetch=False,
    )


def _find_p180_claim(claim_data: dict[str, Any], qid: str) -> dict[str, Any] | None:
    """Return the first P180 claim in a ``wbgetclaims`` reply whose value is *qid*."""
    for claim in claim_data.get("claims", {}).get("P180", []):
        if claim.get("mainsnak", {}).get("datavalue", {}).get("value", {}).get("id") == qid:
            return claim
    return None


class _SdcRun:
    """One ``write_sdc_claims`` call: write approved faces, then remove rejected claims.

    The two phases share the credentials and the retry budget for refreshed
    tokens. Every step returns a :class:`_Step`; ``STOP`` means the run already
    recorded its outcome on the project (or the user cancelled) and the caller
    must return immediately.
    """

    def __init__(self, project: dict[str, Any], user_id: int, access_token: str, csrf_token: str, numeric_id: int):
        self.project_id: int = project["id"]
        self.qid: str = project["wikidata_qid"]
        self.user_id = user_id
        self.access_token = access_token
        self.csrf_token = csrf_token
        self.numeric_id = numeric_id
        self.total_written = 0
        self.total_removed = 0
        self.token_retries = 0
        self.notice: str | None = None  # shown to the user when the run ends without an abort

    @classmethod
    def begin(cls, project: dict[str, Any]) -> Self | None:
        """Obtain credentials and validate the project, or record the failure and return None."""
        project_id = project["id"]
        user_id = project.get("sdc_write_user_id") or project["user_id"]
        logger.info(f"Starting SDC writes for project {project_id}")

        access_token = _refresh_worker_token(user_id)
        if not access_token:
            logger.error(f"Cannot obtain valid token for user {user_id}, aborting SDC writes")
            _abort_sdc_run(project_id, "Token expired or user not found. Please log in again.")
            return None

        try:
            csrf_token = _get_csrf_token(access_token)
        except Exception as e:
            logger.error(f"Failed to get CSRF token for project {project_id}: {e}")
            _abort_sdc_run(project_id, "Failed to get CSRF token")
            return None

        qid = project["wikidata_qid"]
        try:
            numeric_id = int(qid[1:])
        except ValueError:
            logger.error(f"Invalid QID format: {qid}")
            _abort_sdc_run(project_id, "Invalid QID format")
            return None

        return cls(project, user_id, access_token, csrf_token, numeric_id)

    # -- shared helpers -----------------------------------------------------

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.access_token}"}

    def abort(self, message: str) -> None:
        _abort_sdc_run(self.project_id, message)

    def _user_cancelled(self) -> bool:
        """True when the user pressed Stop. A failed check never cancels: keep writing."""
        try:
            rows = execute_query(
                "SELECT sdc_write_requested FROM projects WHERE id = %s",
                (self.project_id,),
                fetch=True,
            )
        except DatabaseError:
            return False
        return not rows or rows[0]["sdc_write_requested"] == 0

    def _lookup_claims(self, mid: str) -> dict[str, Any]:
        response = _api_request(
            COMMONS_API_URL,
            params={"action": "wbgetclaims", "entity": mid, "property": "P180", "format": "json"},
            headers=self._headers,
        )
        return response.json()

    def _refresh_credentials(self, error_code: str, error_info: str, step: str, mid: str) -> bool:
        """Refresh the token after an auth error. False means the run was aborted."""
        self.token_retries += 1
        if self.token_retries > MAX_TOKEN_RETRIES:
            logger.error(f"SDC token retry limit reached during {step} for project {self.project_id}: {error_code}")
            self.abort(_sdc_error_message(error_code, error_info))
            return False
        logger.warning(f"Auth error during {step} for {mid}: {error_code}, refreshing token")
        refreshed = _refresh_worker_token(self.user_id)
        if refreshed:
            self.access_token = refreshed
        try:
            self.csrf_token = _get_csrf_token(self.access_token)
        except Exception:
            logger.error(f"Failed to refresh tokens during {step} for project {self.project_id}")
            self.abort(_sdc_error_message(error_code, error_info))
            return False
        return True

    def _api_failure(
        self,
        reply: dict[str, Any],
        *,
        mid: str,
        step: str,
        on_skippable: Callable[[str], _Step],
    ) -> _Step | None:
        """Classify an error reply from Commons; None when the reply holds no error."""
        error = reply.get("error")
        if not error:
            return None
        code = error.get("code", "unknown")
        info = error.get("info", "Unknown error")
        if code in _SDC_AUTH_ERRORS:
            return _Step.SKIPPED if self._refresh_credentials(code, info, step, mid) else _Step.STOP
        if code in _SDC_PER_FACE_SKIP_ERRORS:
            return on_skippable(code)
        logger.error(f"SDC {step} error for {mid}: {error}")
        self.abort(_sdc_error_message(code, info))
        logger.info(
            f"SDC run aborted for project {self.project_id}: {self.total_written} written, "
            f"{self.total_removed} removed before the error"
        )
        return _Step.STOP

    # -- write phase --------------------------------------------------------

    def write_phase(self) -> bool:
        """Write every approved face. False means the run ended (aborted or cancelled)."""
        self.token_retries = 0
        blocked_passes = 0
        while not shutdown_requested:
            if self._user_cancelled():
                logger.info(
                    f"SDC write cancelled by user for project {self.project_id} ({self.total_written} written so far)"
                )
                return False
            faces = execute_query(_SDC_FACES_TO_WRITE_SQL, (self.project_id, _SDC_WRITE_BATCH), fetch=True)
            if not faces:
                break
            step = self._write_batch(faces)
            if step is _Step.STOP:
                return False
            if step is not _Step.BLOCKED:
                blocked_passes = 0
                continue
            blocked_passes += 1
            if blocked_passes >= _SDC_BLOCKED_MAX_PASSES:
                logger.warning(
                    f"SDC writes for project {self.project_id}: pages still held by another project after "
                    f"{blocked_passes} passes, leaving them for the next send"
                )
                self.notice = (
                    "Some files are being edited by another project and were skipped. Send the edits again later."
                )
                break
            if not self._wait_while_blocked():
                break
        return True

    def _wait_while_blocked(self) -> bool:
        """Pause between fully blocked passes, staying alive. False: a shutdown was requested."""
        for second in range(1, _SDC_BLOCKED_BACKOFF_SECONDS + 1):
            if shutdown_requested:
                return False
            time.sleep(1)
            if second % _PER_FACE_CANCEL_CHECK_EVERY == 0:
                _keepalive_sdc_write(self.project_id)
        return True

    def _write_batch(self, faces: list[dict[str, Any]]) -> _Step:
        faces_since_check = 0
        unverified = 0
        blocked = 0
        for row in faces:
            if shutdown_requested:
                break

            faces_since_check += 1
            if faces_since_check >= _PER_FACE_CANCEL_CHECK_EVERY:
                faces_since_check = 0
                _keepalive_sdc_write(self.project_id)
                if self._user_cancelled():
                    logger.info(
                        f"SDC write cancelled mid-batch by user for project {self.project_id} "
                        f"({self.total_written} written so far)"
                    )
                    return _Step.STOP

            step = self._write_face(row)
            if step is _Step.STOP:
                return _Step.STOP
            if step is _Step.DONE:
                self.total_written += 1
            elif step is _Step.UNVERIFIED:
                unverified += 1
            elif step is _Step.BLOCKED:
                blocked += 1

        if unverified >= len(faces):
            # Every face failed its live-state re-check (e.g. the DB flapped on
            # those single-row SELECTs while the batch SELECT still worked).
            # Nothing in this batch can make progress. Ending through the abort
            # path records why the faces stayed unwritten; falling through to the
            # normal cleanup would clear sdc_write_requested with no explanation
            # and no retry for the user.
            logger.warning(
                f"SDC writes for project {self.project_id}: no face in the batch could be verified, "
                f"aborting this run ({self.total_written} written)"
            )
            self.abort("Could not verify the faces before writing. Please try again.")
            return _Step.STOP
        return _Step.BLOCKED if blocked >= len(faces) else _Step.DONE

    def _face_state(self, face_id: int) -> bool | None:
        """Re-read a face. True: still an approved target. False: changed. None: unreadable.

        The batch was selected before earlier faces were written; the user can
        reject the face, undo its classification or edit its box while we work.
        """
        try:
            rows = execute_query(
                "SELECT is_target, superseded_by FROM faces WHERE id = %s",
                (face_id,),
                fetch=True,
            )
        except DatabaseError:
            return None
        return bool(rows) and rows[0]["is_target"] == 1 and rows[0]["superseded_by"] is None

    def _write_face(self, row: dict[str, Any]) -> _Step:
        face_id = row["face_id"]
        state = self._face_state(face_id)
        if state is None:
            # Unverifiable: skip and retry on a later pass rather than risk a stale claim.
            logger.warning(f"Could not re-check face {face_id} before SDC write, skipping")
            return _Step.UNVERIFIED
        if not state:
            logger.info(f"SDC write skipped for face {face_id}: no longer an approved target")
            return _Step.SKIPPED

        mid = f"M{row['commons_page_id']}"
        try:
            return self._write_verified_face(row, mid)
        except Exception as e:
            logger.error(f"Error writing SDC for face {face_id} on {mid}: {e}")
            self.abort(str(e)[:1024])
            logger.info(f"SDC writes aborted for project {self.project_id}: {self.total_written} written before error")
            return _Step.STOP

    def _write_verified_face(self, row: dict[str, Any], mid: str) -> _Step:
        page_id = row["commons_page_id"]
        face_id = row["face_id"]

        reserved = self._reserve_claim(page_id, face_id)
        if reserved is not None:
            return reserved

        reply = self._lookup_claims(mid)
        failure = self._api_failure(
            reply,
            mid=mid,
            step="idempotency check",
            on_skippable=lambda code: self._finish_face_without_edit(face_id, mid, code),
        )
        if failure is not None:
            return failure

        if _find_p180_claim(reply, self.qid):
            return self._adopt_existing_claim(row, mid)

        # The lookup was a network call (with retries): the face can have been
        # rejected, undone or deleted meanwhile (an undone manual face even
        # cascades its sdc_claims row away). Re-read it right before the edit
        # and release the reservation we still hold, so a later write for a
        # re-confirmed face starts clean.
        state = self._face_state(face_id)
        if state is None:
            logger.warning(f"Could not re-check face {face_id} before SDC edit, skipping")
            return _Step.UNVERIFIED
        if not state:
            logger.info(f"SDC write skipped for face {face_id}: no longer an approved target")
            execute_query(
                "DELETE FROM sdc_claims WHERE face_id = %s AND written_at IS NULL",
                (face_id,),
                fetch=False,
            )
            return _Step.SKIPPED

        reply = self._send_edit(mid)
        failure = self._api_failure(
            reply,
            mid=mid,
            step="write",
            on_skippable=lambda code: self._finish_face_without_edit(face_id, mid, code, release_claim=False),
        )
        if failure is not None:
            return failure

        if not reply.get("success"):
            logger.warning(
                f"SDC write for face {face_id} on {mid} returned unexpected response "
                f"(no 'success' key): {json.dumps(reply)[:500]}"
            )
        logger.info(f"SDC P180 claim for {self.qid} written to {mid} (face {face_id})")
        self._record_written(face_id, page_id, mid)
        time.sleep(1)  # Rate limiting for the MediaWiki API
        return _Step.DONE

    def _reserve_claim(self, page_id: int, face_id: int) -> _Step | None:
        """Claim this page+QID pair across projects. None: go on and write; else the step's result."""
        inserted = execute_query(
            "INSERT IGNORE INTO sdc_claims (commons_page_id, wikidata_qid, project_id, face_id) VALUES (%s, %s, %s, %s)",
            (page_id, self.qid, self.project_id, face_id),
            fetch=False,
        )
        if inserted != 0:
            return None

        # A row already exists for this page+qid: decide whether to skip, retry or reclaim.
        existing_rows = execute_query(
            "SELECT project_id, written_at, claimed_at FROM sdc_claims WHERE commons_page_id = %s AND wikidata_qid = %s",
            (page_id, self.qid),
            fetch=True,
        )
        if not existing_rows:
            # The row vanished between INSERT and SELECT (e.g. a cascade delete); retry next pass.
            logger.warning(
                f"SDC claim row missing for {self.qid} on M{page_id} after INSERT IGNORE, skipping face {face_id}"
            )
            return _Step.SKIPPED

        existing = existing_rows[0]
        if existing["written_at"] is not None:
            logger.info(
                f"SDC claim for {self.qid} on M{page_id} already written "
                f"(written_at={existing['written_at']}), marking face {face_id} as written"
            )
            execute_query("UPDATE faces SET sdc_written = 1 WHERE id = %s", (face_id,), fetch=False)
            return _Step.DONE

        if int(existing["project_id"]) == int(self.project_id):
            # This project reserved it but never finished (e.g. an earlier attempt failed midway).
            logger.info(
                f"SDC claim for {self.qid} on M{page_id} previously claimed by this project, "
                f"retrying write for face {face_id}"
            )
            return None

        # Another project holds an unwritten claim: take it over only if it went stale.
        reclaimed = execute_query(
            "UPDATE sdc_claims SET project_id = %s, face_id = %s, claimed_at = NOW() "
            "WHERE commons_page_id = %s AND wikidata_qid = %s "
            "AND written_at IS NULL "
            "AND claimed_at < NOW() - INTERVAL %s MINUTE",
            (self.project_id, face_id, page_id, self.qid, CLAIM_EXPIRY_MINUTES),
            fetch=False,
        )
        if reclaimed == 0:
            logger.info(
                f"SDC claim for {self.qid} on M{page_id} held by project {existing['project_id']} "
                f"(claimed_at={existing['claimed_at']}), skipping face {face_id} this cycle"
            )
            return _Step.BLOCKED
        logger.info(
            f"Reclaimed stale SDC claim for {self.qid} on M{page_id} from project {existing['project_id']} "
            f"for face {face_id}"
        )
        return None

    def _finish_face_without_edit(self, face_id: int, mid: str, code: str, *, release_claim: bool = True) -> _Step:
        """Commons says the entity is gone: mark the face terminal so it is not retried forever."""
        logger.warning(f"Skipping face {face_id} on {mid}: {code}")
        execute_query("UPDATE faces SET sdc_written = 1 WHERE id = %s", (face_id,), fetch=False)
        if release_claim:
            execute_query("DELETE FROM sdc_claims WHERE face_id = %s", (face_id,), fetch=False)
        return _Step.SKIPPED

    def _adopt_existing_claim(self, row: dict[str, Any], mid: str) -> _Step:
        """The P180 claim is already on Commons: record it instead of writing it again."""
        face_id = row["face_id"]
        # Only mark a face that is still a target. The claim predates this run, so a
        # face reset while wbgetclaims was in flight must not carry a written
        # marker: a later rejection/undo would remove a claim the worker did not
        # add. No removal is queued here for the same reason.
        marked = execute_query(
            "UPDATE faces SET sdc_written = 1 WHERE id = %s AND is_target = 1",
            (face_id,),
            fetch=False,
        )
        if marked:
            logger.info(f"SDC P180 claim for {self.qid} already exists on {mid}, marked face {face_id} as written")
        else:
            logger.warning(
                f"Face {face_id} is no longer a target; not marking the pre-existing P180 claim on {mid} as written"
            )
        execute_query(
            "UPDATE sdc_claims SET written_at = NOW() "
            "WHERE commons_page_id = %s AND wikidata_qid = %s AND written_at IS NULL",
            (row["commons_page_id"], self.qid),
            fetch=False,
        )
        execute_query(
            "UPDATE images SET bootstrapped = 1 WHERE id = %s AND bootstrapped = 0",
            (row["image_id"],),
            fetch=False,
        )
        return _Step.DONE

    def _send_edit(self, mid: str) -> dict[str, Any]:
        claim = {
            "mainsnak": {
                "snaktype": "value",
                "property": "P180",
                "datavalue": {
                    "type": "wikibase-entityid",
                    "value": {"numeric-id": self.numeric_id, "id": self.qid},
                },
            },
            "type": "statement",
            "rank": "normal",
        }
        response = _api_request(
            COMMONS_API_URL,
            data={
                "action": "wbeditentity",
                "id": mid,
                "data": json.dumps({"claims": [claim]}),
                "token": self.csrf_token,
                "summary": f"WikiVisage: Adding depicts (P180) claim for {self.qid}",
                "format": "json",
                "bot": "1",
                "maxlag": "5",
            },
            headers=self._headers,
            method="post",
        )
        return response.json()

    def _record_written(self, face_id: int, page_id: int, mid: str) -> None:
        # The face may have been reset, rejected or undone between the live-state
        # re-check and the wbeditentity call. Only mark it written while it is
        # still a target; otherwise the claim we just added is orphaned, so queue
        # its removal in the same run. An undo can restore is_target=0 (review
        # undo) as well as NULL (classify undo), so both need the recovery.
        written_now = execute_query(
            "UPDATE faces SET sdc_written = 1 WHERE id = %s AND is_target = 1",
            (face_id,),
            fetch=False,
        )
        if not written_now:
            logger.warning(f"Face {face_id} changed during the SDC write; queuing P180 removal for {mid}")
            execute_query(
                "UPDATE faces SET sdc_written = 1, sdc_removal_pending = 1 "
                "WHERE id = %s AND (is_target IS NULL OR is_target = 0) "
                "AND superseded_by IS NULL",
                (face_id,),
                fetch=False,
            )
        execute_query(
            "UPDATE sdc_claims SET written_at = NOW() WHERE commons_page_id = %s AND wikidata_qid = %s",
            (page_id, self.qid),
            fetch=False,
        )

    # -- removal phase ------------------------------------------------------

    def removal_phase(self) -> bool:
        """Remove P180 claims for rejected faces. False means the run ended (aborted)."""
        self.token_retries = 0
        while not shutdown_requested:
            if self._user_cancelled():
                logger.info(f"SDC removal cancelled by user for project {self.project_id}")
                break

            pages = execute_query(_SDC_PAGES_TO_REMOVE_SQL, (self.project_id, _SDC_REMOVAL_BATCH), fetch=True)
            if not pages:
                break

            for row in pages:
                if shutdown_requested:
                    break
                _keepalive_sdc_write(self.project_id)
                step = self._remove_page(row["commons_page_id"])
                if step is _Step.STOP:
                    return False
                if step is _Step.DONE:
                    self.total_removed += 1
        return True

    def _remove_page(self, page_id: int) -> _Step:
        mid = f"M{page_id}"
        try:
            still_pending = execute_query(
                _SDC_REMOVAL_STILL_PENDING_SQL,
                (page_id, self.project_id),
                fetch=True,
            )
            if not still_pending:
                logger.info(f"Skipping removal for M{page_id}: sibling approved since selection")
                return _Step.SKIPPED

            reply = self._lookup_claims(mid)
            failure = self._api_failure(
                reply,
                mid=mid,
                step="removal idempotency check",
                on_skippable=lambda code: self._settle_removal(page_id, f"{code} during idempotency check"),
            )
            if failure is not None:
                return failure

            claim = _find_p180_claim(reply, self.qid)
            guid = claim.get("id") if claim else None
            if not guid:
                logger.info(f"P180 claim for {self.qid} not found on {mid}, clearing removal flag")
                return self._settle_removal(page_id)

            reply = self._send_removal(guid)
            failure = self._api_failure(
                reply,
                mid=mid,
                step="removal",
                on_skippable=lambda code: self._settle_removal(page_id, f"{code} during claim removal"),
            )
            if failure is not None:
                return failure

            logger.info(f"SDC removal succeeded for {mid} (project {self.project_id}): removed claim {guid}")
            self._settle_removal(page_id)
            time.sleep(1)
            return _Step.DONE
        except Exception as e:
            logger.error(f"Error removing SDC for {mid}: {e}")
            self.abort(str(e)[:1024])
            logger.info(
                f"SDC removals aborted for project {self.project_id}: {self.total_removed} removed before error"
            )
            return _Step.STOP

    def _send_removal(self, guid: str) -> dict[str, Any]:
        response = _api_request(
            COMMONS_API_URL,
            data={
                "action": "wbremoveclaims",
                "claim": guid,
                "token": self.csrf_token,
                "summary": f"WikiVisage: Removing depicts (P180) claim for {self.qid} (human review)",
                "format": "json",
                "bot": "1",
                "maxlag": "5",
            },
            headers=self._headers,
            method="post",
        )
        return response.json()

    def _settle_removal(self, page_id: int, skipped_because: str | None = None) -> _Step:
        """The claim is gone from Commons (removed, or never there): clear every flag that said otherwise."""
        if skipped_because:
            logger.warning(f"Skipping removal for M{page_id}: {skipped_because}")
        execute_query(
            "UPDATE faces f JOIN images i ON f.image_id = i.id "
            "SET f.sdc_removal_pending = 0, f.sdc_written = 0 "
            "WHERE i.commons_page_id = %s AND i.project_id = %s "
            "AND f.sdc_removal_pending = 1",
            (page_id, self.project_id),
            fetch=False,
        )
        _clear_sdc_dedup_row(page_id, self.qid)
        return _Step.DONE

    # -- end of run ---------------------------------------------------------

    def finish(self) -> None:
        """Clear the request flag, reporting an unfinished run if a shutdown cut it short."""
        # A shutdown can arrive after the last successful edit, including during
        # its rate-limit sleep. The signal alone does not mean any work was lost.
        error_msg = self.notice
        if shutdown_requested:
            try:
                if _has_pending_sdc_work(self.project_id):
                    error_msg = (
                        "Worker shutdown left Commons edits pending. "
                        "Edits already sent were kept. Send the remaining edits again."
                    )
                    logger.info("SDC for project %s stopped with edits still pending", self.project_id)
                else:
                    logger.info("SDC for project %s finished before worker shutdown; no edits remain", self.project_id)
            except DatabaseError:
                logger.warning(
                    "Could not verify remaining SDC work on shutdown for project %s", self.project_id, exc_info=True
                )
                error_msg = (
                    "Worker shut down before completion could be verified. "
                    "Edits may already be on Commons. Please check the results."
                )

        execute_query(
            "UPDATE projects SET sdc_write_requested = 0, sdc_write_user_id = NULL, sdc_write_error = %s WHERE id = %s",
            (error_msg, self.project_id),
            fetch=False,
        )
        logger.info(f"SDC for project {self.project_id}: {self.total_written} written, {self.total_removed} removed")


def write_sdc_claims(project: dict[str, Any]) -> int:
    """Write SDC P180 claims for all faces marked as target, then remove rejected ones.

    Called when ``sdc_write_requested=1``. Processes faces in batches and clears the
    flag when done (or sets ``sdc_write_error`` on failure). Returns the number of
    claims written (or recorded as already present).
    """
    run = _SdcRun.begin(project)
    if run is None:
        return 0
    if run.write_phase() and run.removal_phase():
        run.finish()
    return run.total_written


# ---------------------------------------------------------------------------
# Distributed locking: claim / release / refresh helpers
# ---------------------------------------------------------------------------


def _claim_active_projects(max_count: int) -> list[dict[str, Any]]:
    """Atomically claim up to *max_count* unclaimed (or stale-claimed) active projects.

    Uses SELECT … FOR UPDATE SKIP LOCKED inside a transaction so two workers
    racing on the same poll cycle will never both claim the same project, and
    neither will block waiting for the other's lock.

    Only claims projects that have remaining work: at least one pending image
    to process OR at least one unclassified face.  This avoids wasting worker
    slots on fully-processed projects that the user hasn't marked completed.
    The work predicate is mirrored (negated) by ``_claim_shadow_project``,
    which must never take a project this function can claim; keep them in sync.

    Returns:
        List of project dicts that were successfully claimed by this worker.
    """

    def _txn(conn: Any, cursor: Any) -> list[dict[str, Any]]:
        cursor.execute(
            "SELECT p.id FROM projects p "
            "WHERE p.status = 'active' "
            "AND (p.worker_claimed_by IS NULL "
            "     OR p.worker_claimed_at < NOW() - INTERVAL %s MINUTE) "
            "AND ("
            "  NOT EXISTS (SELECT 1 FROM images i WHERE i.project_id = p.id) "
            "  OR EXISTS (SELECT 1 FROM images i WHERE i.project_id = p.id AND i.status = 'pending') "
            "  OR EXISTS ("
            "    SELECT 1 FROM faces f JOIN images i ON f.image_id = i.id "
            "    WHERE i.project_id = p.id AND f.is_target IS NULL AND f.superseded_by IS NULL"
            "  )"
            ") "
            "ORDER BY COALESCE(p.worker_claimed_at, '1970-01-01') ASC, "
            "p.images_total ASC, p.id DESC "
            "LIMIT %s "
            "FOR UPDATE SKIP LOCKED",
            (CLAIM_EXPIRY_MINUTES, max_count),
        )
        rows = cursor.fetchall()
        if not rows:
            return []

        ids = [r["id"] for r in rows]
        placeholders = ", ".join(["%s"] * len(ids))
        cursor.execute(
            f"UPDATE projects SET worker_claimed_by = %s, worker_claimed_at = NOW() WHERE id IN ({placeholders})",
            [_worker_id, *ids],
        )

        # Re-fetch the full project rows (now claimed) so callers get all columns.
        cursor.execute(
            f"SELECT p.*, "
            f"  (SELECT COUNT(*) FROM images i WHERE i.project_id = p.id) AS image_count "
            f"FROM projects p WHERE p.id IN ({placeholders})",
            ids,
        )
        return list(cursor.fetchall())

    try:
        result = execute_transaction(_txn)
        if result:
            logger.info(f"Claimed {len(result)} active project(s): {[p['id'] for p in result]}")
        return result
    except DatabaseError:
        logger.exception("Failed to claim active projects")
        return []


def _claim_inference_projects(max_count: int) -> list[dict[str, Any]]:
    """Atomically claim inference-eligible projects not already claimed by any worker.

    An inference-eligible project is active or completed, has enough human-confirmed
    target faces (>= min_confirmed), and still has inferable unclassified faces
    remaining (``run_autonomous_inference`` skips faces with a queued removal).
    The work predicate is mirrored (negated) by ``_claim_shadow_project``; keep
    them in sync.
    """

    def _txn(conn: Any, cursor: Any) -> list[dict[str, Any]]:
        cursor.execute(
            "SELECT p.id FROM projects p "
            "WHERE p.status IN ('active', 'completed') "
            "AND (p.worker_claimed_by IS NULL "
            "     OR p.worker_claimed_at < NOW() - INTERVAL %s MINUTE) "
            "AND ("
            "  SELECT COUNT(*) FROM faces f "
            "  JOIN images i ON f.image_id = i.id "
            "  WHERE i.project_id = p.id AND f.is_target = 1 AND f.superseded_by IS NULL"
            "  AND f.classified_by_user_id IS NOT NULL"
            ") >= p.min_confirmed "
            "AND EXISTS ("
            "  SELECT 1 FROM faces f "
            "  JOIN images i ON f.image_id = i.id "
            "  WHERE i.project_id = p.id AND f.is_target IS NULL AND f.superseded_by IS NULL "
            "  AND f.sdc_removal_pending = 0"
            ") "
            "ORDER BY COALESCE(p.worker_claimed_at, '1970-01-01') ASC, p.id ASC "
            "LIMIT %s "
            "FOR UPDATE SKIP LOCKED",
            (CLAIM_EXPIRY_MINUTES, max_count),
        )
        rows = cursor.fetchall()
        if not rows:
            return []

        ids = [r["id"] for r in rows]
        placeholders = ", ".join(["%s"] * len(ids))
        cursor.execute(
            f"UPDATE projects SET worker_claimed_by = %s, worker_claimed_at = NOW() WHERE id IN ({placeholders})",
            [_worker_id, *ids],
        )

        cursor.execute(
            f"SELECT p.* FROM projects p WHERE p.id IN ({placeholders})",
            ids,
        )
        return list(cursor.fetchall())

    try:
        result = execute_transaction(_txn)
        if result:
            logger.info(f"Claimed {len(result)} inference project(s): {[p['id'] for p in result]}")
        return result
    except DatabaseError:
        logger.exception("Failed to claim inference projects")
        return []


def _claim_sdc_projects() -> tuple[list[dict[str, Any]], set[int]]:
    """Atomically claim projects that have pending SDC write requests.

    Returns:
        A tuple of ``(projects, pre_claimed_ids)`` where *pre_claimed_ids* is
        the set of project IDs that were **already** claimed by this worker
        before this call (i.e. for active/inference processing).  Callers must
        **not** release those claims when SDC writing finishes, as another
        thread may still be processing them.
    """

    def _txn(conn: Any, cursor: Any) -> tuple[list[dict[str, Any]], set[int]]:
        cursor.execute(
            "SELECT p.id, (p.worker_claimed_by = %s) AS already_claimed "
            "FROM projects p "
            "WHERE p.sdc_write_requested = 1 AND p.status != 'deleted' "
            "AND (p.worker_claimed_by IS NULL "
            "     OR p.worker_claimed_by = %s "
            "     OR p.worker_claimed_at < NOW() - INTERVAL %s MINUTE) "
            "ORDER BY COALESCE(p.worker_claimed_at, '1970-01-01') ASC "
            "LIMIT %s "
            "FOR UPDATE SKIP LOCKED",
            (_worker_id, _worker_id, CLAIM_EXPIRY_MINUTES, MAX_CONCURRENT_PROJECTS),
        )
        rows = cursor.fetchall()
        if not rows:
            return [], set()

        ids = [r["id"] for r in rows]
        pre_claimed_ids = {r["id"] for r in rows if r["already_claimed"]}
        placeholders = ", ".join(["%s"] * len(ids))
        cursor.execute(
            f"UPDATE projects SET worker_claimed_by = %s, worker_claimed_at = NOW() WHERE id IN ({placeholders})",
            [_worker_id, *ids],
        )

        cursor.execute(
            f"SELECT p.* FROM projects p WHERE p.id IN ({placeholders})",
            ids,
        )
        return list(cursor.fetchall()), pre_claimed_ids

    try:
        result, pre_claimed_ids = execute_transaction(_txn)
        if result:
            logger.info(f"Claimed {len(result)} SDC project(s): {[p['id'] for p in result]}")
        return result, pre_claimed_ids
    except DatabaseError:
        logger.exception("Failed to claim SDC projects")
        return [], set()


def _process_sdc_writes() -> int:
    """Claim and process all pending SDC write requests.

    Returns the total number of claims written across all projects.
    Safe to call from any point in the main loop (top-of-cycle,
    mid-cycle wake handler, inference loop).
    """
    sdc_projects, pre_claimed_ids = _claim_sdc_projects()
    if not sdc_projects:
        return 0
    logger.info(f"Claimed {len(sdc_projects)} SDC project(s) for writing")
    total_written = 0
    for sdc_project in sdc_projects:
        if shutdown_requested:
            break
        # Refresh claim before starting potentially long SDC write
        try:
            execute_query(
                "UPDATE projects SET worker_claimed_at = NOW() WHERE id = %s AND worker_claimed_by = %s",
                (sdc_project["id"], _worker_id),
                fetch=False,
            )
        except DatabaseError:
            pass
        logger.info(f"Processing SDC write request for project {sdc_project['id']}")
        try:
            written = write_sdc_claims(sdc_project)
            total_written += written
            logger.info(f"SDC write complete for project {sdc_project['id']}: {written} claims written")
        except Exception as e:
            logger.error(f"SDC write failed for project {sdc_project['id']}: {e}")
            # Clear the flag so it doesn't retry endlessly
            try:
                execute_query(
                    "UPDATE projects SET sdc_write_requested = 0, sdc_write_user_id = NULL, sdc_write_error = %s WHERE id = %s",
                    (str(e)[:1000], sdc_project["id"]),
                    fetch=False,
                )
            except DatabaseError:
                pass
        finally:
            # Only release the claim if it was newly acquired for SDC writing.
            # Pre-existing claims (from active/inference processing) must not be
            # cleared here, as another thread may still be using them.
            if sdc_project["id"] not in pre_claimed_ids:
                _release_project(sdc_project["id"])
    return total_written


def _release_project(project_id: int) -> None:
    """Release this worker's claim on a single project."""
    try:
        execute_query(
            "UPDATE projects SET worker_claimed_by = NULL WHERE id = %s AND worker_claimed_by = %s",
            (project_id, _worker_id),
            fetch=False,
        )
    except DatabaseError:
        logger.warning("Failed to release claim on project %s", project_id)


def _release_all_claims() -> None:
    """Release all claims held by this worker (shutdown cleanup)."""
    try:
        released = execute_query(
            "UPDATE projects SET worker_claimed_by = NULL WHERE worker_claimed_by = %s",
            (_worker_id,),
            fetch=False,
        )
        if released:
            logger.info(f"Released {released} project claim(s) on shutdown")
    except DatabaseError:
        logger.warning("Failed to release claims on shutdown")


def _refresh_claims() -> None:
    """Refresh worker_claimed_at for all projects claimed by this worker.

    This prevents our claims from expiring while we are still actively
    processing them (the expiry window is CLAIM_EXPIRY_MINUTES).
    """
    try:
        execute_query(
            "UPDATE projects SET worker_claimed_at = NOW() WHERE worker_claimed_by = %s",
            (_worker_id,),
            fetch=False,
        )
    except DatabaseError:
        logger.warning("Failed to refresh claim timestamps")


def _process_and_release(project: dict[str, Any], *, skip_discovery: bool = False) -> None:
    """Wrapper: run process_project then release the claim no matter what."""
    try:
        process_project(project, skip_discovery=skip_discovery)
    finally:
        _release_project(project["id"])


def _infer_and_release(project: dict[str, Any]) -> int:
    """Wrapper: run run_autonomous_inference then release the claim."""
    try:
        return run_autonomous_inference(project)
    finally:
        _release_project(project["id"])


def _run_new_inference_projects(max_count: int = 1) -> int:
    """Claim and run up to ``max_count`` inference-eligible projects inline.

    Called from the wake handler and the 60 s heartbeat checks of the running
    loops, so a project whose classifications just crossed ``min_confirmed``
    does not have to wait for the next cycle while active projects hold the
    executor slots.  Inference is seconds of numpy plus one batched UPDATE, so
    running it inline keeps the project page's "within the next minute" promise.
    Returns the number of projects classified.
    """
    classified_projects = 0
    for project in _claim_inference_projects(max_count):
        try:
            classified = _infer_and_release(project)
            classified_projects += 1
            logger.info(f"Inference classified {classified} faces for project {project['id']}")
        except Exception:
            logger.exception(f"Inline inference failed for project {project['id']}")
    return classified_projects


def _has_inference_candidates(project_id: int) -> bool:
    """Return True when the project has a face inference could classify.

    Cheap pre-check for the early inference pass in :func:`process_project`.
    Mirrors the candidate filter in :func:`run_autonomous_inference`.
    """
    try:
        rows = execute_query(
            "SELECT 1 AS found FROM faces f "
            "JOIN images i ON f.image_id = i.id "
            "WHERE i.project_id = %s "
            "  AND f.is_target IS NULL "
            "  AND f.superseded_by IS NULL "
            "  AND f.classified_by_user_id IS NULL "
            "  AND f.sdc_removal_pending = 0 "
            "  AND f.embedding_model = %s "
            "LIMIT 1",
            (project_id, face_client.EXPECTED_MODEL_ID),
            fetch=True,
        )
        return bool(rows)
    except DatabaseError:
        logger.warning("Could not check inference candidates for project %s", project_id, exc_info=True)
        return False


def _project_still_active(project_id: int) -> bool:
    """Re-check project status and claim ownership from the database.

    False if the project is no longer active, or another worker has reclaimed it (for example
    because our claim expired). Refreshes the claim timestamp on success so the claim does not
    expire while we work. A database error counts as still active: keep going rather than
    silently drop work.
    """
    try:
        row = execute_query(
            "SELECT status, worker_claimed_by FROM projects WHERE id = %s",
            (project_id,),
            fetch=True,
        )
        if not row:
            logger.info(f"Project {project_id} not found, stopping processing")
            return False
        status = row[0]["status"]
        claimed_by = row[0].get("worker_claimed_by")
        if status != "active":
            logger.info(f"Project {project_id} is no longer active (status={status}), stopping processing")
            return False
        if claimed_by != _worker_id:
            logger.info(f"Project {project_id} not claimed by us (claimed_by={claimed_by}), stopping processing")
            return False
        execute_query(
            "UPDATE projects SET worker_claimed_at = NOW() WHERE id = %s AND worker_claimed_by = %s",
            (project_id, _worker_id),
            fetch=False,
        )
        return True
    except DatabaseError as exc:
        logger.warning(
            "DatabaseError while checking if project %s is still active; "
            "assuming it is still active to avoid dropping work: %s",
            project_id,
            exc,
        )
        return True


def _run_early_inference(project: dict[str, Any]) -> None:
    """Classify faces that are already waiting, before the (slow) discovery stages.

    A re-run resets model faces on a project that is still active; without this the user waits
    through the category crawl and SPARQL bootstrap before the inference card resolves. The
    final pass in ``process_project`` still handles faces from newly processed images.
    """
    project_id = project["id"]
    if shutdown_requested or not _has_inference_candidates(project_id):
        return
    fresh_rows = execute_query(
        "SELECT * FROM projects WHERE id = %s AND status != 'deleted'",
        (project_id,),
        fetch=True,
    )
    early_classified = run_autonomous_inference(fresh_rows[0] if fresh_rows else project)
    logger.info(f"Project {project_id}: early inference classified {early_classified} face(s) before discovery")


def _process_pending_images(project: dict[str, Any]) -> tuple[float, int]:
    """Download and detect every pending image. Returns (seconds spent, images processed)."""
    t0 = time.monotonic()
    total_processed = 0
    while not shutdown_requested:
        batch_count = process_images(project)
        if batch_count == 0:
            break
        total_processed += batch_count
        logger.info(f"Processed batch of {batch_count} images ({total_processed} total so far)")

        # Check if project was paused/completed between batches
        if not _project_still_active(project["id"]):
            break
    return time.monotonic() - t0, total_processed


_AUTO_COMPLETE_SQL = {
    "no_faces": (
        "UPDATE projects SET status = 'completed', completion_reason = 'no_faces' WHERE id = %s AND status = 'active'"
    ),
    "insufficient_faces": (
        "UPDATE projects SET status = 'completed', completion_reason = 'insufficient_faces' "
        "WHERE id = %s AND status = 'active'"
    ),
}


def _auto_complete_unusable_project(project: dict[str, Any]) -> bool:
    """Mark a project completed when it has no faces, or too few to train on. True if it did.

    Never auto-completes while images are still unprocessed. The image loop exits as soon as a
    batch yields zero successes, which also happens when every download fails (for example when
    Commons moves its CDN to a host the allowlist rejects). Without this guard a transient outage
    is recorded as a permanent "no faces in this category" verdict, and ``_claim_active_projects``
    only picks up status='active', so the remaining images would never be retried. Mirrors the
    guard in app.py.
    """
    project_id = project["id"]
    pending_row = execute_query(
        "SELECT COUNT(*) AS cnt FROM images WHERE project_id = %s AND status = 'pending'",
        (project_id,),
        fetch=True,
    )
    pending_images = pending_row[0]["cnt"] if pending_row else 0

    face_count_row = execute_query(
        "SELECT COUNT(*) AS cnt FROM faces f "
        "JOIN images i ON f.image_id = i.id "
        "WHERE i.project_id = %s AND f.superseded_by IS NULL",
        (project_id,),
        fetch=True,
    )
    total_faces = face_count_row[0]["cnt"] if face_count_row else 0

    # Fetch the latest min_confirmed from the database to avoid using a stale in-memory value.
    min_conf_row = execute_query(
        "SELECT min_confirmed FROM projects WHERE id = %s",
        (project_id,),
        fetch=True,
    )
    if min_conf_row and min_conf_row[0].get("min_confirmed") is not None:
        min_confirmed = min_conf_row[0]["min_confirmed"]
    else:
        # Fall back to the original project dict or default to preserve existing behavior.
        min_confirmed = project.get("min_confirmed", 5)

    if pending_images > 0:
        logger.warning(
            f"Project {project_id}: image processing stopped early with {pending_images} image(s) "
            f"still pending ({total_faces} faces so far). NOT auto-completing — the project stays "
            f"active and will retry next cycle. Check images.error_message for the cause."
        )
        return False

    if total_faces == 0:
        reason = "no_faces"
        outcome = f"0 faces detected, auto-completed ({reason})"
    elif total_faces <= 5 and total_faces < min_confirmed:
        reason = "insufficient_faces"
        outcome = f"only {total_faces} faces detected (need {min_confirmed}), auto-completed ({reason})"
    else:
        return False

    execute_query(_AUTO_COMPLETE_SQL[reason], (project_id,), fetch=False)
    logger.info(f"Project {project_id}: {outcome}")
    return True


def _run_final_inference(project: dict[str, Any]) -> float:
    """Classify whatever is still unclassified. Returns the seconds spent."""
    t0 = time.monotonic()
    if not shutdown_requested and _project_still_active(project["id"]):
        fresh = execute_query(
            "SELECT * FROM projects WHERE id = %s AND status != 'deleted'", (project["id"],), fetch=True
        )
        run_autonomous_inference(fresh[0] if fresh else project)
    return time.monotonic() - t0


def _log_project_summary(
    project_id: int,
    t_start: float,
    *,
    traversal: float,
    bootstrap: float,
    images: float,
    processed: int,
    inference: str,
) -> None:
    logger.info(
        f"--- Project {project_id} complete: "
        f"traversal={traversal:.1f}s, bootstrap={bootstrap:.1f}s, "
        f"images={images:.1f}s ({processed} processed), "
        f"inference={inference}, total={time.monotonic() - t_start:.1f}s ---"
    )


def process_project(project: dict[str, Any], *, skip_discovery: bool = False) -> None:
    """Orchestrate all phases for a single project.

    Args:
        project: Project row dict from DB (must include 'id', 'wikidata_qid', etc.).
        skip_discovery: If True, skip traverse_category and bootstrap_from_sparql.
    """
    t_project_start = time.monotonic()
    project_id = project["id"]
    logger.info(f"--- Processing project {project_id} ({project['wikidata_qid']}) ---")

    try:
        # Early bail-out: the project may have been deleted / paused between the moment it was
        # queued in the thread pool and the moment this thread starts running. Checking here
        # avoids wasted traversal and bootstrap work for projects that are no longer active.
        if not _project_still_active(project_id):
            return

        _run_early_inference(project)

        t_traversal = 0.0
        t_bootstrap = 0.0

        if skip_discovery:
            logger.info(f"Project {project_id}: skipping discovery (already fast-tracked)")
        else:
            # 1. Traversal
            t0 = time.monotonic()
            traverse_category(project)
            t_traversal = time.monotonic() - t0

            if shutdown_requested or not _project_still_active(project_id):
                return

            # 2. Bootstrap (if needed)
            t0 = time.monotonic()
            bootstrap_from_sparql(project)
            t_bootstrap = time.monotonic() - t0

            if shutdown_requested or not _project_still_active(project_id):
                return

        # 3. Image download and face detection: process all pending images
        t_images, total_processed = _process_pending_images(project)

        # 3a. Experimental LiftWing metadata pre-filter (shadow mode: verdicts are recorded for an
        #     offline readout and change no behaviour). Runs after detection on purpose: verdicts
        #     are metadata-only, and selecting already-processed images too means a backlog larger
        #     than one cycle's cap is fully covered over time. (An eventual `enforce` mode must
        #     instead select pending images and run *before* the image loop, so drops happen
        #     before any download.)
        if not shutdown_requested and _project_still_active(project_id):
            prefilter_images(project)

        # 3b. Auto-complete if no faces or insufficient faces were detected
        if not shutdown_requested and _project_still_active(project_id) and _auto_complete_unusable_project(project):
            _log_project_summary(
                project_id,
                t_project_start,
                traversal=t_traversal,
                bootstrap=t_bootstrap,
                images=t_images,
                processed=total_processed,
                inference="skipped",
            )
            return

        # 4. Final autonomous inference (catch any remaining unclassified faces)
        t_inference = _run_final_inference(project)

        # SDC writes are handled separately: triggered via the web UI, processed in the main
        # loop when sdc_write_requested=1.
        _log_project_summary(
            project_id,
            t_project_start,
            traversal=t_traversal,
            bootstrap=t_bootstrap,
            images=t_images,
            processed=total_processed,
            inference=f"{t_inference:.1f}s",
        )

    except Exception as e:
        logger.error(f"Failed to process project {project['id']}: {e}")


def signal_handler(signum, frame):
    """Graceful shutdown on SIGINT/SIGTERM."""
    global shutdown_requested
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    shutdown_requested = True


def _check_face_service_model() -> None:
    """Refuse to run against a face service that produces a different embedding.

    Readiness says nothing about *which* embedding the service returns. Storing
    vectors from the wrong model would corrupt every project, so a definite
    mismatch stops the worker (exit 1); anything transient is only logged and
    retried, because the service may still be starting.
    """
    try:
        service_model, service_dim = face_client.check_service_model()
        logger.info("Face service embedding model: %s (dim=%d)", service_model, service_dim)
    except FaceServiceModelMismatch as e:
        logger.critical(f"Face service runs a different embedding model than this worker expects: {e}")
        sys.exit(1)
    except FaceServiceError as e:
        logger.warning(f"Could not verify the face service's embedding model at startup: {e}")


def _record_liftwing_failure(error: str | None) -> None:
    """Record one failed LiftWing call; flip to 'down' at ``LIFTWING_FAIL_THRESHOLD``.

    Used for both probe failures and pre-filter request failures, so the banner
    reflects what real completions experience, not just what the probe sees.
    Hysteresis: one slow reply must not flap the banner, so 'down' needs
    consecutive failures, and a single success flips it back (see
    ``_probe_liftwing_if_due`` and ``_record_liftwing_success``).
    """
    rows = execute_query("SELECT status, consecutive_failures FROM service_status WHERE name = 'liftwing'", fetch=True)
    if not rows:
        return
    failures = int(rows[0]["consecutive_failures"]) + 1
    status = "down" if failures >= LIFTWING_FAIL_THRESHOLD else rows[0]["status"]
    execute_query(
        "UPDATE service_status SET status = %s, consecutive_failures = %s, last_error = %s WHERE name = 'liftwing'",
        (status, failures, (error or "")[:255] or None),
        fetch=False,
    )
    if status == "down":
        logger.warning(f"LiftWing marked down after {failures} consecutive failures: {error}")
    else:
        logger.info(f"LiftWing failure ({failures}/{LIFTWING_FAIL_THRESHOLD}): {error}")


def _record_liftwing_success() -> None:
    """Record one successful real LiftWing call: clear failures, mark it up.

    Called by the pre-filter after a stored batch, so a working service flips
    the banner back without waiting for the next probe.
    """
    try:
        execute_query(
            "UPDATE service_status SET status = 'up', consecutive_failures = 0, "
            "last_ok = NOW(), last_error = NULL WHERE name = 'liftwing'",
            fetch=False,
        )
    except DatabaseError:
        logger.debug("LiftWing success update skipped: database error")


def _probe_liftwing_if_due() -> None:
    """Probe LiftWing at most once per interval, across all worker instances.

    Only runs while at least one non-deleted project has the pre-filter on, so
    nothing calls LiftWing while the experiment is unused. The claim is a
    single atomic UPDATE on ``service_status``: whichever worker gets rowcount 1
    probes, so two workers cannot probe at once. Errors are swallowed — a probe
    must never take the main loop down.
    """
    if LIFTWING_PROBE_INTERVAL <= 0:
        return
    try:
        in_use = execute_query(
            "SELECT COUNT(*) AS cnt FROM projects WHERE status != 'deleted' AND prefilter_mode != 'off'",
            fetch=True,
        )
        if not in_use or not in_use[0]["cnt"]:
            return

        claimed = execute_query(
            "UPDATE service_status SET last_checked = NOW() "
            "WHERE name = 'liftwing' "
            "AND (last_checked IS NULL OR last_checked < NOW() - INTERVAL %s SECOND)",
            (LIFTWING_PROBE_INTERVAL,),
            fetch=False,
        )
        if not claimed:
            return

        up, error = liftwing_client.get_client().probe()
        if up:
            execute_query(
                "UPDATE service_status SET status = 'up', consecutive_failures = 0, "
                "last_ok = NOW(), last_error = NULL WHERE name = 'liftwing'",
                fetch=False,
            )
            logger.info("LiftWing probe: up")
        else:
            _record_liftwing_failure(error)
    except DatabaseError:
        logger.debug("LiftWing probe skipped: database error")
    except Exception:
        logger.exception("LiftWing probe failed unexpectedly")


def _parse_worker_id(argv: list[str]) -> str:
    """Worker id from ``--worker-id`` (used by deploy.yml), else ``<hostname>-<pid>``."""
    wid = None
    if "--worker-id" in argv:
        idx = argv.index("--worker-id")
        if idx + 1 < len(argv):
            wid = argv[idx + 1]
    worker_id = wid or f"{socket.gethostname()}-{os.getpid()}"
    # The id becomes part of the heartbeat file path: refuse anything that could traverse.
    if not re.fullmatch(r"[a-zA-Z0-9._-]+", worker_id):
        raise SystemExit(f"Invalid worker ID (must be alphanumeric/dot/hyphen/underscore): {worker_id!r}")
    return worker_id


def _db_pool_size() -> int:
    """Pool size for this worker: ``WIKIVISAGE_DB_POOL_SIZE``, else derived from the concurrency settings.

    Toolforge ToolsDB enforces max_user_connections=20 across ALL processes sharing the DB user
    (gunicorn + every worker instance). The web app uses 2 workers x 3 connections, so each of
    the two background workers is capped at 6 to stay at or below 20.
    """
    db_pool_env = os.environ.get("WIKIVISAGE_DB_POOL_SIZE")
    if db_pool_env:
        pool_size = int(db_pool_env)
    else:
        pool_size = min(MAX_CONCURRENT_PROJECTS * IMAGE_THREADS + 3, 6)
    if pool_size > 6:
        logger.warning(
            "DB pool size %d may exceed Toolforge max_user_connections (20 shared across all processes). "
            "With 2 workers + gunicorn, keep each worker's pool <= 6.",
            pool_size,
        )
    return pool_size


def _check_face_service_at_startup() -> None:
    """Probe the face service once so a misconfigured endpoint shows up in the logs at once.

    Non-fatal: the service may still be starting, and detection retries on every poll cycle.
    """
    if face_client.is_healthy():
        logger.info("Face service ready at %s (model=%s)", face_client.FACE_SERVICE_URL, face_client.FACE_SERVICE_MODEL)
        _check_face_service_model()
    else:
        logger.warning(
            "Face service not ready at %s (model=%s) — image processing will retry each poll cycle. "
            "Set WIKIVISAGE_FACE_SERVICE_URL if this is wrong.",
            face_client.FACE_SERVICE_URL,
            face_client.FACE_SERVICE_MODEL,
        )


def _log_project_diagnostics() -> None:
    """Log one line per project so a support session starts with the real counts."""
    try:
        diag = execute_query(
            "SELECT p.id, p.wikidata_qid, p.status, p.min_confirmed, p.faces_confirmed, "
            "  (SELECT COUNT(*) FROM faces f JOIN images i ON f.image_id = i.id "
            "   WHERE i.project_id = p.id AND f.is_target = 1 AND f.superseded_by IS NULL) AS actual_target_faces, "
            "  (SELECT COUNT(*) FROM faces f JOIN images i ON f.image_id = i.id "
            "   WHERE i.project_id = p.id AND f.is_target = 0 AND f.superseded_by IS NULL) AS actual_non_target, "
            "  (SELECT COUNT(*) FROM faces f JOIN images i ON f.image_id = i.id "
            "   WHERE i.project_id = p.id AND f.is_target IS NULL AND f.superseded_by IS NULL) AS unclassified_faces "
            "FROM projects p",
            fetch=True,
        )
        for row in diag or []:
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


def _archive_and_purge(conn: Any, cursor: Any) -> int:
    """Move deleted projects' classification counts into user_stats, then delete them."""
    cursor.execute(
        "INSERT INTO user_stats (user_id, classifications, sdc_tags) "
        "SELECT f.classified_by_user_id, "
        "  COUNT(*), "
        "  COUNT(CASE WHEN f.sdc_written = 1 THEN 1 END) "
        "FROM faces f "
        "INNER JOIN images i ON i.id = f.image_id "
        "INNER JOIN projects p ON p.id = i.project_id "
        "WHERE p.status = 'deleted' "
        "  AND f.classified_by_user_id IS NOT NULL "
        "  AND f.superseded_by IS NULL "
        "GROUP BY f.classified_by_user_id "
        "ON DUPLICATE KEY UPDATE "
        "  classifications = classifications + VALUES(classifications), "
        "  sdc_tags = sdc_tags + VALUES(sdc_tags)",
    )
    cursor.execute("DELETE FROM projects WHERE status = 'deleted'")
    return cursor.rowcount


def _purge_deleted_projects() -> None:
    """Purge soft-deleted projects first: it is quick and unblocks re-creating the same project."""
    try:
        deleted = execute_transaction(_archive_and_purge)
        if deleted:
            logger.info(f"Purged {deleted} soft-deleted project(s)")
    except DatabaseError as exc:
        logger.warning("Failed to purge soft-deleted projects: %s", exc)


def _consume_wake_signal() -> bool:
    """True once per wake-up request. The web app creates the file to ask for an immediate cycle."""
    if not os.path.exists(WAKE_FILE_PATH):
        return False
    try:
        os.remove(WAKE_FILE_PATH)
    except OSError:
        pass
    return True


def _keep_alive_during_cycle() -> None:
    """Once a minute while projects run: heartbeat, claims, LiftWing probe, newly inference-ready projects."""
    try:
        execute_query("REPLACE INTO worker_heartbeat (id, last_seen) VALUES (1, NOW())", fetch=False)
        _touch_heartbeat_file()
    except Exception:
        pass
    _refresh_claims()
    _probe_liftwing_if_due()
    _run_new_inference_projects(1)


def _supervise(
    futures: dict[Future[Any], int],
    *,
    on_wake: Callable[[dict[Future[Any], int]], None],
    on_done: Callable[[Future[Any], int], None],
) -> None:
    """Wait for *futures* while keeping this worker alive and responsive.

    The project threads can run for a long time. Meanwhile this loop refreshes the
    heartbeat and claims every minute, services SDC write requests (on a wake-up
    signal, and every 10 s because the wake file does not cross pods), and hands each
    finished future to *on_done*. *on_wake* runs when the web app asked for an
    immediate check; it may add futures to the dict.
    """
    last_heartbeat = last_sdc_check = time.time()
    while futures:
        if shutdown_requested:
            break

        if time.time() - last_heartbeat >= 60:
            _keep_alive_during_cycle()
            last_heartbeat = time.time()

        if _consume_wake_signal():
            on_wake(futures)
            last_sdc_check = time.time()

        if time.time() - last_sdc_check >= 10:
            _process_sdc_writes()
            last_sdc_check = time.time()

        done = {f for f in futures if f.done()}
        if not done:
            time.sleep(1)
            continue
        for future in done:
            on_done(future, futures.pop(future))


def _drain_inference_queue(active_ids: set[int], seen_ids: set[int]) -> None:
    """Run inference for every claimable project before the (slow) active projects start.

    Inference is a few seconds of numpy plus one batched UPDATE per project, whereas an
    active project can hold all slots for a long time looping through its image backlog.
    With the old order a project whose classifications had just crossed min_confirmed sat
    behind that backlog while its page said "Model inference in progress". Keep claiming
    until no unseen eligible project remains; ``seen_ids`` guarantees termination when a
    project stays eligible because some of its faces have unusable encodings.
    """

    def on_wake(_futures: dict[Future[Any], int]) -> None:
        _process_sdc_writes()
        # Pick up any project that just became inference-ready.
        _run_new_inference_projects(1)

    def on_done(future: Future[Any], project_id: int) -> None:
        try:
            logger.info(f"Inference classified {future.result()} faces for project {project_id}")
        except Exception as e:
            logger.error(f"Inference failed for project {project_id}: {e}")

    while not shutdown_requested:
        claimed = _claim_inference_projects(MAX_CONCURRENT_PROJECTS)
        inference_only = [p for p in claimed if p["id"] not in active_ids and p["id"] not in seen_ids]
        run_ids = {p["id"] for p in inference_only}
        # Release claims we will not run (already processed this cycle, or covered by the
        # active pipeline) instead of holding them until the claim expiry.
        for project in claimed:
            if project["id"] not in run_ids and project["id"] not in active_ids:
                _release_project(project["id"])
        if not inference_only:
            break

        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_PROJECTS) as executor:
            futures: dict[Future[Any], int] = {}
            for project in inference_only:
                if shutdown_requested:
                    break
                seen_ids.add(project["id"])
                logger.info(f"Running inference-only for project {project['id']} (status={project.get('status')})")
                futures[executor.submit(_infer_and_release, project)] = project["id"]
            _supervise(futures, on_wake=on_wake, on_done=on_done)


def _process_active_projects(active_projects: list[dict[str, Any]], seen_ids: set[int]) -> None:
    """Process the claimed projects concurrently (claims are released in the wrapper).

    Each thread handles its own discovery (traverse_category + bootstrap), so all claimed
    projects start working at once with no serial bottleneck.
    """
    if not active_projects:
        return

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_PROJECTS) as executor:
        futures: dict[Future[Any], int] = {}

        def submit(project: dict[str, Any]) -> None:
            seen_ids.add(project["id"])
            futures[executor.submit(_process_and_release, project)] = project["id"]

        def on_wake(current: dict[Future[Any], int]) -> None:
            logger.info("Wake-up signal received mid-cycle, checking for new work")
            _process_sdc_writes()

            free_slots = MAX_CONCURRENT_PROJECTS - len([f for f in current if not f.done()])
            if free_slots <= 0:
                logger.debug("All %d worker slots occupied, skipping mid-cycle claim", MAX_CONCURRENT_PROJECTS)
            else:
                for project in _claim_active_projects(free_slots):
                    if project["id"] not in seen_ids:
                        logger.info(f"Adding new project {project['id']} to current cycle")
                        submit(project)

            # A classification can make a project inference-ready even when all slots are
            # busy; inference is cheap, so run it inline instead of waiting for the next cycle.
            _run_new_inference_projects(1)

        def on_done(future: Future[Any], project_id: int) -> None:
            try:
                future.result()
            except Exception as e:
                logger.error(f"Project {project_id} processing failed: {e}")

        for project in active_projects:
            if shutdown_requested:
                break
            submit(project)
        _supervise(futures, on_wake=on_wake, on_done=on_done)


def _run_cycle() -> None:
    """One poll cycle: heartbeat, purge, SDC writes, inference, active projects."""
    execute_query("REPLACE INTO worker_heartbeat (id, last_seen) VALUES (1, NOW())", fetch=False)
    _touch_heartbeat_file()
    _probe_liftwing_if_due()

    _purge_deleted_projects()

    # User-triggered SDC writes come before the active pipeline, so users do not wait
    # minutes through crawl/detect/infer before "Send Edits to Wikimedia Commons" is serviced.
    _process_sdc_writes()

    active_projects = _claim_active_projects(MAX_CONCURRENT_PROJECTS)
    logger.info(f"Poll: claimed {len(active_projects)} active project(s)")

    seen_ids: set[int] = set()
    _drain_inference_queue({p["id"] for p in active_projects}, seen_ids)
    _process_active_projects(active_projects, seen_ids)


def _sleep_until_next_cycle() -> None:
    """Sleep one poll interval, waking early for a wake-up signal or finished SDC work.

    Random jitter (0-15 s) desynchronizes multiple workers so they do not all race to
    claim the same projects every cycle.
    """
    last_sdc_check = time.time()
    for _ in range(POLL_INTERVAL + random.randint(0, 15)):
        if shutdown_requested:
            break
        if _consume_wake_signal():
            logger.info("Wake-up signal received, starting next cycle")
            break
        if time.time() - last_sdc_check >= 10:
            written = _process_sdc_writes()
            last_sdc_check = time.time()
            if written:
                break
        time.sleep(1)


def _release_resources(sweep_thread: threading.Thread) -> None:
    """Shutdown: stop the sweep, give back claims, close pools. Order matters."""
    global shutdown_requested
    logger.info("Worker %s shutting down, releasing claims and closing resources.", _worker_id)
    # Signal the sweep and let it leave its batch before the claims and the DB/HTTP resources
    # go away: it checks shutdown_requested between batches, but an in-flight LiftWing request
    # can still outlive the bounded join. After that the daemon thread dies with the process.
    shutdown_requested = True
    sweep_thread.join(timeout=SWEEP_JOIN_TIMEOUT)
    if sweep_thread.is_alive():
        logger.warning(f"Shadow pre-filter sweep still running after {SWEEP_JOIN_TIMEOUT}s; closing resources anyway")
    _release_all_claims()
    face_client.reset_session()
    liftwing_client.reset_client()
    close_pool()


def main():
    """Main worker loop with distributed locking for concurrent multi-instance processing."""
    global _worker_id
    _worker_id = _parse_worker_id(sys.argv)
    logger.info(
        f"Starting WikiVisage Worker {_worker_id} (max_projects={MAX_CONCURRENT_PROJECTS}, image_threads={IMAGE_THREADS})"
    )

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        init_db(pool_size=_db_pool_size())
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        sys.exit(1)

    _check_face_service_at_startup()
    _log_project_diagnostics()

    # Background shadow sweep: records LiftWing verdicts for projects the processing pipelines
    # never claim (completed, paused or fully processed). Own daemon thread: one pass can take
    # minutes, and an inline call would stall the heartbeat refreshes the Toolforge health check
    # watches. The handle is kept so shutdown can join it before its resources go away.
    sweep_thread = threading.Thread(target=_shadow_prefilter_loop, name="shadow-prefilter-sweep", daemon=True)
    sweep_thread.start()

    try:
        while not shutdown_requested:
            try:
                _run_cycle()
                _sleep_until_next_cycle()
            except DatabaseError as e:
                logger.error(f"Database error in main loop: {e}")
                time.sleep(10)
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}")
                time.sleep(10)
    finally:
        _release_resources(sweep_thread)


if __name__ == "__main__":
    main()
