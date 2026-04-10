import io
import ipaddress
import json
import logging
import multiprocessing
import os
import queue
import random
import re
import signal
import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()

# Optional throttle (in seconds) between image downloads and processing to
# avoid overloading Commons or downstream services. Defaults to 0 (no delay).
COMMONS_DOWNLOAD_THROTTLE_SECONDS = float(os.getenv("COMMONS_DOWNLOAD_THROTTLE_SECONDS", "0"))

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
warnings.filterwarnings("ignore", message=".*pkg_resources.*", category=UserWarning)

import face_recognition
import numpy as np
import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from config import HEARTBEAT_FILE_DIR, WAKE_FILE_PATH
from database import DatabaseError, close_pool, execute_query, execute_transaction, init_db
from token_crypto import TokenDecryptionError, decrypt_token, encrypt_token

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("worker")

# Constants
COMMONS_API_URL = "https://commons.wikimedia.org/w/api.php"
SPARQL_URL = "https://commons-query.wikimedia.org/sparql"  # Requires OAuth; unused, kept for reference
FILE_PATH_URL = "https://commons.wikimedia.org/wiki/Special:FilePath/{file_title}?width=500"

USER_AGENT = "WikiVisage/1.0 (Wikimedia Toolforge; https://toolsadmin.wikimedia.org)"

POLL_INTERVAL = int(os.environ.get("WIKIVISAGE_WORKER_POLL_INTERVAL", 60))
BATCH_SIZE = int(os.environ.get("WIKIVISAGE_WORKER_BATCH_SIZE", 50))
MAX_CONCURRENT_PROJECTS = int(os.environ.get("WIKIVISAGE_WORKER_MAX_PROJECTS", 4))
IMAGE_THREADS = int(os.environ.get("WIKIVISAGE_WORKER_IMAGE_THREADS", 4))

# Distributed locking: stale claims from crashed workers expire after this many minutes
CLAIM_EXPIRY_MINUTES = 15

# Maximum image download size (50 MB) — prevents OOM on abnormally large files
MAX_IMAGE_DOWNLOAD_BYTES = 50 * 1024 * 1024

# Maximum image pixel area (100 megapixels) — prevents OOM in face detection subprocess
MAX_IMAGE_PIXELS = 100_000_000

# Maximum images per project — stops category traversal once this limit is reached
MAX_IMAGES_PER_PROJECT = 9000

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
TOKEN_REFRESH_BUFFER = 300

# Maximum number of token refresh retries per SDC write session
MAX_TOKEN_RETRIES = 2


# Non-image file extensions to skip during category traversal (video, audio)
_SKIP_EXTENSIONS = {".webm", ".ogv", ".ogg", ".mp3", ".wav", ".flac", ".opus", ".mid", ".oga"}


def _build_skip_extensions_regex(extensions: set[str]) -> str:
    """
    Build a regex that matches file titles ending with any of the non-image extensions.

    The generated pattern has the form: r"\\.(ext1|ext2|ext3)$"
    where the extensions come from the `_SKIP_EXTENSIONS` set.
    """
    # Normalize by removing leading dots and escaping for regex safety
    escaped_exts = [re.escape(ext.lstrip(".")) for ext in sorted(extensions)]
    if not escaped_exts:
        # Fallback that matches nothing if the set is ever empty
        return r"(?!)"
    return r"\.(" + "|".join(escaped_exts) + r")$"


# Regex derived from `_SKIP_EXTENSIONS` for consistent non-image filtering (e.g. in SQL REGEXP)
SKIP_EXTENSIONS_REGEX = _build_skip_extensions_regex(_SKIP_EXTENSIONS)

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


_ALLOWED_DOWNLOAD_HOSTS = frozenset({"commons.wikimedia.org", "upload.wikimedia.org"})


def _reject_private_ip(hostname: str) -> None:
    """Raise ValueError if *hostname* resolves to a private/reserved IP (DNS rebinding defense)."""
    for info in socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM):
        addr = ipaddress.ip_address(info[4][0])
        if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved:
            raise ValueError(f"Download host {hostname} resolved to private/reserved IP: {addr}")


def _download_image(url: str, max_bytes: int = MAX_IMAGE_DOWNLOAD_BYTES) -> bytes:
    """Download an image with streaming size cap to prevent OOM.

    Uses _get_session() for connection pooling / retry. Raises ValueError
    if the URL points to an untrusted host or if the response exceeds max_bytes.
    """
    parsed_url = urlparse(url)
    if parsed_url.scheme not in ("http", "https") or parsed_url.hostname not in _ALLOWED_DOWNLOAD_HOSTS:
        raise ValueError(f"Blocked download from untrusted host: {parsed_url.hostname}")
    _reject_private_ip(parsed_url.hostname)
    session = _get_session()
    resp = session.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=30,
        stream=True,
    )
    resp.raise_for_status()

    # Reject redirects to untrusted hosts (SSRF defense)
    final_host = urlparse(resp.url).hostname
    if final_host not in _ALLOWED_DOWNLOAD_HOSTS:
        resp.close()
        raise ValueError(f"Redirect to untrusted host: {final_host}")

    # Check Content-Length header first (fast reject)
    content_length = resp.headers.get("Content-Length")
    if content_length and int(content_length) > max_bytes:
        resp.close()
        raise ValueError(f"Image too large: {int(content_length)} bytes (limit {max_bytes})")

    # Stream with enforced cap
    chunks: list[bytes] = []
    downloaded = 0
    for chunk in resp.iter_content(chunk_size=65536):
        downloaded += len(chunk)
        if downloaded > max_bytes:
            resp.close()
            raise ValueError(f"Image download exceeded {max_bytes} bytes limit")
        chunks.append(chunk)
    resp.close()

    return b"".join(chunks)


def _validate_image_dimensions(image_bytes: bytes) -> None:
    """Check image dimensions via PIL header-only read. Raises ValueError if too large."""
    img = Image.open(io.BytesIO(image_bytes))
    w, h = img.size
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


def _refresh_worker_token(user_id: int) -> str | None:
    """Refresh an expired OAuth access token for SDC writes.

    Reads refresh_token from DB, calls the OAuth token endpoint,
    and updates the DB with the new credentials. Returns the new
    access_token or None if refresh failed.
    """
    user_row = execute_query(
        "SELECT access_token, refresh_token, token_expires_at FROM users WHERE id = %s",
        (user_id,),
        fetch=True,
    )
    if not user_row:
        return None

    user = user_row[0]
    access_token = user["access_token"]
    if isinstance(access_token, bytes):
        access_token = access_token.decode("utf-8")
    try:
        access_token = decrypt_token(access_token)
    except TokenDecryptionError:
        logger.error(
            f"Cannot decrypt access token for user {user_id} — possible key rotation; skipping SDC writes for this user"
        )
        return None
    refresh_token = user.get("refresh_token", "")
    if isinstance(refresh_token, bytes):
        refresh_token = refresh_token.decode("utf-8")
    try:
        refresh_token = decrypt_token(refresh_token)
    except TokenDecryptionError:
        logger.error(
            f"Cannot decrypt refresh token for user {user_id} — possible key rotation; skipping SDC writes for this user"
        )
        return None

    expires_at = user.get("token_expires_at")
    if expires_at:
        if isinstance(expires_at, str):
            expires_at = datetime.fromisoformat(expires_at)
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=UTC)
        now = datetime.now(UTC)
        if (expires_at - now).total_seconds() > TOKEN_REFRESH_BUFFER:
            return access_token

    if not refresh_token:
        logger.error(f"No refresh token for user {user_id}, cannot refresh")
        return None

    logger.info(f"Refreshing access token for user {user_id} (worker)")
    try:
        resp = _get_session().post(
            OAUTH_TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": OAUTH_CLIENT_ID,
                "client_secret": OAUTH_CLIENT_SECRET,
            },
            headers={"User-Agent": USER_AGENT},
            timeout=30,
        )
        try:
            resp.raise_for_status()
            new_token = resp.json()
        finally:
            resp.close()
    except Exception:
        logger.exception(f"Failed to refresh access token for user {user_id}")
        return None

    new_access = new_token.get("access_token")
    if not new_access:
        logger.error(f"Token refresh response missing access_token for user {user_id}")
        return None

    new_refresh = new_token.get("refresh_token", refresh_token)
    new_expires_at = datetime.now(UTC) + timedelta(seconds=new_token.get("expires_in", 14400))

    # Optimistic concurrency: only update if no other process refreshed first.
    # The old expires_at acts as a compare-and-swap guard.
    old_expires_str = expires_at.strftime("%Y-%m-%d %H:%M:%S") if expires_at else None

    try:
        if old_expires_str:
            rowcount = execute_query(
                "UPDATE users SET access_token = %s, refresh_token = %s, token_expires_at = %s "
                "WHERE id = %s AND token_expires_at = %s",
                (
                    encrypt_token(new_access),
                    encrypt_token(new_refresh),
                    new_expires_at.strftime("%Y-%m-%d %H:%M:%S"),
                    user_id,
                    old_expires_str,
                ),
                fetch=False,
            )
        else:
            rowcount = execute_query(
                "UPDATE users SET access_token = %s, refresh_token = %s, token_expires_at = %s WHERE id = %s",
                (
                    encrypt_token(new_access),
                    encrypt_token(new_refresh),
                    new_expires_at.strftime("%Y-%m-%d %H:%M:%S"),
                    user_id,
                ),
                fetch=False,
            )

        if rowcount == 0 and old_expires_str:
            # Another process already refreshed — re-read from DB
            logger.info(f"Token already refreshed by another process for user {user_id}")
            fresh = execute_query(
                "SELECT access_token FROM users WHERE id = %s",
                (user_id,),
                fetch=True,
            )
            if fresh:
                fresh_token = fresh[0]["access_token"]
                if isinstance(fresh_token, bytes):
                    fresh_token = fresh_token.decode("utf-8")
                try:
                    return decrypt_token(fresh_token)
                except TokenDecryptionError:
                    logger.error(f"Cannot decrypt fresh token for user {user_id}")
                    return None
            return None
    except DatabaseError:
        logger.exception(f"Failed to persist refreshed token for user {user_id}")

    return new_access


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


FACE_DETECT_TIMEOUT = 120  # seconds — kill subprocess if face detection hangs


# ---------------------------------------------------------------------------
# Persistent Subprocess Pool for Face Detection
# ---------------------------------------------------------------------------
# Instead of spawning a new multiprocessing.Process per image (which re-imports
# dlib/numpy/face_recognition each time — 2-5s overhead on Toolforge), we
# maintain a pool of long-lived subprocesses. Each subprocess imports the heavy
# libraries once at startup and processes images received via a Queue.
# If a subprocess crashes (e.g. dlib segfault), it is automatically respawned.


def _face_detect_worker_loop(
    task_queue: multiprocessing.Queue,
    result_queue: multiprocessing.Queue,
    worker_id: int,
) -> None:
    """Long-lived subprocess loop: receive image bytes, detect faces, send results.

    Runs until it receives a None sentinel on task_queue. Each task is
    (request_id, image_bytes). Results are (request_id, "ok", locations,
    encodings_bytes, width, height) or (request_id, "error", error_string).
    """
    # Heavy imports happen once per subprocess lifetime — this is the whole point
    import face_recognition as fr
    import numpy  # noqa: F401 — imported to ensure numpy is initialized

    while True:
        try:
            task = task_queue.get()
            if task is None:
                break  # Sentinel: clean shutdown

            request_id, image_bytes = task

            try:
                image_data = fr.load_image_file(io.BytesIO(image_bytes))
                img_height, img_width = image_data.shape[:2]
                face_locations = fr.face_locations(image_data, model="hog")
                face_encodings = fr.face_encodings(image_data, face_locations)

                encodings_as_bytes = [enc.tobytes() for enc in face_encodings]
                result_queue.put(
                    (
                        request_id,
                        "ok",
                        face_locations,
                        encodings_as_bytes,
                        img_width,
                        img_height,
                    )
                )
            except Exception as e:
                result_queue.put((request_id, "error", str(e)))

        except Exception:
            # Queue error or other fatal issue — subprocess exits, will be respawned
            logging.getLogger(__name__).exception("Fatal error in face detection subprocess, exiting")
            break


class PoolUnavailableError(RuntimeError):
    """Raised when the persistent face detection pool is not functional."""

    pass


class FaceDetectPool:
    """Pool of persistent subprocesses for face detection.

    Eliminates the 2-5 second per-image overhead of spawning a new Process
    (which re-imports dlib/face_recognition) by keeping N subprocesses alive.
    Each subprocess imports the heavy libraries once at startup.

    Thread-safe: multiple IMAGE_THREADS can call detect_faces() concurrently.
    A background dispatch thread routes results from the shared subprocess
    result queue to per-request queues, avoiding stash/requeue races.

    Provides crash isolation: if a subprocess segfaults on a bad image,
    it is automatically respawned for the next request.
    """

    def __init__(self, pool_size: int = IMAGE_THREADS):
        self._pool_size = pool_size
        self._task_queue: multiprocessing.Queue = multiprocessing.Queue()
        self._result_queue: multiprocessing.Queue = multiprocessing.Queue()
        self._workers: list[multiprocessing.Process] = []
        self._workers_lock = threading.Lock()
        self._request_counter = 0
        self._counter_lock = threading.Lock()
        # Per-request result routing: request_id -> queue.Queue holding the result
        self._pending: dict[int, queue.Queue] = {}
        self._pending_lock = threading.Lock()
        self._dispatcher_thread: threading.Thread | None = None
        self._shutdown_event = threading.Event()
        self._started = False

    def start(self) -> None:
        """Start the subprocess pool and result dispatcher. Idempotent."""
        if self._started:
            return
        logger.info(f"Starting face detection subprocess pool (size={self._pool_size})")
        self._shutdown_event.clear()
        with self._workers_lock:
            for i in range(self._pool_size):
                self._spawn_worker(i)
        # Start background thread that routes results to per-request queues
        self._dispatcher_thread = threading.Thread(
            target=self._dispatch_results, daemon=True, name="face-pool-dispatch"
        )
        self._dispatcher_thread.start()
        self._started = True

    def _spawn_worker(self, worker_id: int) -> None:
        """Spawn a single worker subprocess. Caller must hold self._workers_lock."""
        proc = multiprocessing.Process(
            target=_face_detect_worker_loop,
            args=(self._task_queue, self._result_queue, worker_id),
        )
        proc.start()
        if worker_id < len(self._workers):
            self._workers[worker_id] = proc
        else:
            self._workers.append(proc)
        logger.debug(f"Spawned face detection subprocess {worker_id} (pid={proc.pid})")

    def _ensure_workers_alive(self) -> None:
        """Check all workers and respawn any that have died."""
        with self._workers_lock:
            for i, proc in enumerate(self._workers):
                if not proc.is_alive():
                    exitcode = proc.exitcode
                    logger.warning(f"Face detection subprocess {i} died (exitcode={exitcode}), respawning")
                    self._spawn_worker(i)

    def _dispatch_results(self) -> None:
        """Background thread: drain _result_queue and route to per-request queues.

        Each result tuple starts with request_id. We look up the corresponding
        per-request queue in self._pending and deliver the result. If no pending
        request matches (e.g. caller timed out and unregistered), the result is
        discarded with a warning.
        """
        while not self._shutdown_event.is_set():
            try:
                # Short timeout so we can check shutdown_event periodically
                result = self._result_queue.get(timeout=1.0)
            except Exception:
                # queue.Empty on timeout — loop back and check shutdown
                continue

            request_id = result[0]
            with self._pending_lock:
                result_q = self._pending.get(request_id)

            if result_q is not None:
                result_q.put(result)
            else:
                logger.warning(f"Face detection result for unknown request_id={request_id} (caller may have timed out)")

    def is_healthy(self) -> bool:
        """Check if the pool is started and the dispatcher thread is alive."""
        return self._started and self._dispatcher_thread is not None and self._dispatcher_thread.is_alive()

    def detect_faces(self, image_bytes: bytes) -> tuple[list, list[bytes], int, int]:
        """Submit image for face detection and wait for result.

        Thread-safe: each caller gets a unique request_id and a private result
        queue. The dispatch thread routes the subprocess result to the correct
        caller without stashing or requeuing.

        Returns (locations, encoding_bytes_list, width, height).
        Raises PoolUnavailableError if the pool is not started or dispatcher is dead.
        Raises RuntimeError on timeout, subprocess crash, or detection error.
        """
        if not self.is_healthy():
            raise PoolUnavailableError(
                "Face detection pool is not running"
                + (" (dispatcher thread died)" if self._started else " (not started)")
            )
        self._ensure_workers_alive()

        # Allocate unique request_id under lock
        with self._counter_lock:
            self._request_counter += 1
            request_id = self._request_counter

        # Register a private result queue for this request
        result_q: queue.Queue = queue.Queue()
        with self._pending_lock:
            self._pending[request_id] = result_q

        try:
            # Submit task to subprocess pool
            self._task_queue.put((request_id, image_bytes))

            # Wait for our result — the dispatch thread will deliver it
            try:
                result = result_q.get(timeout=FACE_DETECT_TIMEOUT)
            except queue.Empty:
                logger.warning(
                    f"Face detection timed out after {FACE_DETECT_TIMEOUT}s (request_id={request_id}). "
                    "Orphaned task will be discarded when a subprocess picks it up."
                )
                raise RuntimeError(f"Face detection timed out after {FACE_DETECT_TIMEOUT}s")

            if result[1] == "error":
                raise RuntimeError(f"Face detection failed: {result[2]}")
            if result[1] != "ok":
                raise RuntimeError(f"Unexpected subprocess result: {result[1]}")
            return result[2], result[3], result[4], result[5]
        finally:
            # Unregister so dispatch thread doesn't hold a stale reference
            with self._pending_lock:
                self._pending.pop(request_id, None)

    def shutdown(self) -> None:
        """Gracefully shut down all worker subprocesses and the dispatch thread.

        Ordering: mark not-started (reject new work) → send sentinels to
        workers → join workers → stop dispatcher → cleanup.  The dispatcher
        must stay alive while workers drain so in-flight results are still
        routed to their callers.
        """
        if not self._started:
            return
        logger.info("Shutting down face detection subprocess pool")

        # 1. Reject new work immediately
        self._started = False

        # 2. Send sentinel to each worker subprocess so they exit cleanly
        with self._workers_lock:
            for _ in self._workers:
                try:
                    self._task_queue.put(None)
                except Exception:
                    pass

            # 3. Wait for workers to exit (dispatcher still routing results)
            for i, proc in enumerate(self._workers):
                proc.join(timeout=5)
                if proc.is_alive():
                    logger.warning(f"Force-killing face detection subprocess {i}")
                    proc.kill()
                    proc.join(timeout=2)

        # 4. Now stop the dispatcher — all workers are gone, no more results
        self._shutdown_event.set()
        if self._dispatcher_thread is not None:
            self._dispatcher_thread.join(timeout=5)

        # 5. Close and join multiprocessing queues to release OS resources
        for q in (self._task_queue, self._result_queue):
            try:
                q.close()
                q.join_thread()
            except Exception:
                pass

        with self._workers_lock:
            self._workers.clear()
        logger.info("Face detection subprocess pool shut down")


# Module-level pool instance — initialized in main() before processing starts
_face_pool: FaceDetectPool | None = None


def _detect_faces_in_subprocess(
    image_bytes: bytes,
    conn: Any,
) -> None:
    """Run face detection in an isolated subprocess. Sends results back via pipe.

    This isolates dlib's C++ code so a segfault kills only this subprocess,
    not the parent worker. Results are sent as (locations, encodings_bytes) or
    an error string.

    NOTE: This is the FALLBACK path used only when the persistent pool is not
    available (e.g. during single-image retries after pool crash).
    """
    try:
        image_data = face_recognition.load_image_file(io.BytesIO(image_bytes))
        img_height, img_width = image_data.shape[:2]
        face_locations = face_recognition.face_locations(image_data, model="hog")
        face_encodings = face_recognition.face_encodings(image_data, face_locations)

        encodings_as_bytes = [enc.tobytes() for enc in face_encodings]
        conn.send(("ok", face_locations, encodings_as_bytes, img_width, img_height))
    except Exception as e:
        conn.send(("error", str(e)))
    finally:
        conn.close()


def _run_face_detection_fallback(
    image_bytes: bytes,
) -> tuple[list, list[bytes], int, int]:
    """Fallback: run face detection in a one-shot subprocess (old behavior).

    Used only when the persistent pool is unavailable.
    """
    parent_conn, child_conn = multiprocessing.Pipe(duplex=False)
    proc = multiprocessing.Process(
        target=_detect_faces_in_subprocess,
        args=(image_bytes, child_conn),
    )
    proc.start()
    child_conn.close()

    elapsed = 0.0
    poll_slice = 2.0
    got_result = False
    while elapsed < FACE_DETECT_TIMEOUT:
        if shutdown_requested:
            parent_conn.close()
            proc.kill()
            proc.join(timeout=5)
            raise InterruptedError("Worker shutting down during face detection")
        wait = min(poll_slice, FACE_DETECT_TIMEOUT - elapsed)
        if parent_conn.poll(wait):
            got_result = True
            break
        elapsed += wait

    if got_result:
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

    return result[1], result[2], result[3], result[4]


def _run_face_detection(image_bytes: bytes) -> tuple[list, list[bytes], int, int]:
    """Run face detection — uses persistent pool if available, falls back to one-shot subprocess.

    Returns (locations, encoding_bytes_list, width, height).
    Raises RuntimeError if subprocess crashes (segfault) or times out.
    """
    if _face_pool is not None:
        try:
            return _face_pool.detect_faces(image_bytes)
        except PoolUnavailableError as e:
            logger.warning(f"Pool unavailable, falling back to one-shot subprocess: {e}")
    return _run_face_detection_fallback(image_bytes)


def _process_single_image(
    img_id: int, title: str, *, bootstrapped: bool = False, project_id: int | None = None
) -> bool:
    """Download one image, detect faces in subprocess, store encodings.

    If the image is bootstrapped (already has a P180 depicts claim on Commons)
    and exactly one face is detected, auto-classify it as a target match.
    Multi-face bootstrapped images are left for manual classification.

    Returns True on success.
    """
    clean_title = title[5:] if title.startswith("File:") else title
    url = FILE_PATH_URL.format(file_title=clean_title)

    try:
        t_start = time.monotonic()

        logger.debug(f"Downloading image {title}")
        image_bytes = _download_image(url)
        _validate_image_dimensions(image_bytes)
        t_download = time.monotonic()

        if COMMONS_DOWNLOAD_THROTTLE_SECONDS > 0:
            time.sleep(COMMONS_DOWNLOAD_THROTTLE_SECONDS)

        face_locations, encodings_bytes, det_width, det_height = _run_face_detection(image_bytes)
        del image_bytes
        t_detect = time.monotonic()

        face_count = len(face_locations)

        # Batch-insert detected faces in chunks to stay within max_allowed_packet.
        # All faces start unclassified (is_target=NULL). Classification
        # happens via human confirmation or autonomous inference.
        if face_locations:
            FACE_INSERT_CHUNK = 100  # ~100KB per chunk (1024-byte encoding + ints)
            for chunk_start in range(0, face_count, FACE_INSERT_CHUNK):
                chunk_locs = face_locations[chunk_start : chunk_start + FACE_INSERT_CHUNK]
                chunk_encs = encodings_bytes[chunk_start : chunk_start + FACE_INSERT_CHUNK]
                chunk_size = len(chunk_locs)
                placeholders = ", ".join(["(%s, %s, %s, %s, %s, %s)"] * chunk_size)
                flat_params: list = []
                for location, enc_bytes in zip(chunk_locs, chunk_encs):
                    top, right, bottom, left = location
                    flat_params.extend([img_id, enc_bytes, top, right, bottom, left])

                execute_query(
                    f"INSERT INTO faces (image_id, encoding, bbox_top, bbox_right, bbox_bottom, bbox_left) "
                    f"VALUES {placeholders}",
                    tuple(flat_params),
                    fetch=False,
                )

        execute_query(
            "UPDATE images SET status = 'processed', face_count = %s, "
            "detection_width = %s, detection_height = %s WHERE id = %s",
            (face_count, det_width, det_height, img_id),
            fetch=False,
        )

        # Auto-classify single-face bootstrapped images as target matches.
        # These images already have a P180 depicts claim on Commons, so the
        # single detected face is overwhelmingly likely to be the target person.
        # Multi-face bootstrapped images are left for manual classification.
        if bootstrapped and face_count == 1 and project_id is not None:
            auto_classified = execute_query(
                "UPDATE faces SET is_target = 1, classified_by = 'bootstrap', classified_at = NOW() "
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

        t_db = time.monotonic()

        logger.debug(
            f"Image {img_id} ({face_count} faces): "
            f"download={t_download - t_start:.2f}s, "
            f"detect={t_detect - t_download:.2f}s, "
            f"db={t_db - t_detect:.2f}s, "
            f"total={t_db - t_start:.2f}s"
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

    with ThreadPoolExecutor(max_workers=IMAGE_THREADS) as executor:
        futures = {}
        for img_row in pending_images:
            if shutdown_requested:
                break
            future = executor.submit(
                _process_single_image,
                img_row["id"],
                img_row["file_title"],
                bootstrapped=bool(img_row.get("bootstrapped")),
                project_id=project_id,
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

    batch_elapsed = time.monotonic() - batch_start
    batch_size = len(pending_images) if pending_images else 0
    per_image = batch_elapsed / batch_size if batch_size > 0 else 0
    logger.info(
        f"Batch complete: {processed_count}/{batch_size} images in {batch_elapsed:.1f}s "
        f"({per_image:.2f}s/image avg, {IMAGE_THREADS} threads)"
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
                        # matches (same logic as _process_single_image).
                        # Perform this in a single conditional UPDATE to avoid a per-image COUNT query.
                        auto = execute_query(
                            "UPDATE faces "
                            "SET is_target = 1, classified_by = 'bootstrap', classified_at = NOW() "
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
    ...     if "SELECT f.encoding FROM faces f" in sql:
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
    confirmed_rows = execute_query(
        """
        SELECT f.encoding FROM faces f
        JOIN images i ON f.image_id = i.id
        WHERE i.project_id = %s AND f.is_target = 1 AND f.superseded_by IS NULL
        """,
        (project["id"],),
        fetch=True,
    )

    if not confirmed_rows:
        return 0

    t_query = time.monotonic()

    # Load all confirmed encodings to compute centroid. Each encoding is 128 float64
    # values (1024 bytes). Even 10K confirmed faces = ~10 MB — well within Toolforge
    # memory limits. Batching would add complexity with no practical benefit.
    confirmed_encodings = []
    for row in confirmed_rows:
        enc = row["encoding"]
        if enc is None or len(enc) != 1024:
            logger.warning(
                f"Skipping confirmed face with invalid encoding length ({len(enc) if enc else 'None'} bytes)"
            )
            continue
        confirmed_encodings.append(np.frombuffer(enc, dtype=np.float64).copy())

    if not confirmed_encodings:
        logger.warning(f"Project {project['id']}: all confirmed encodings invalid, skipping inference")
        return 0

    try:
        centroid = np.mean(confirmed_encodings, axis=0)
    except Exception:
        logger.exception(f"Project {project['id']}: failed to compute centroid")
        return 0
    t_centroid = time.monotonic()

    # Select candidate faces for model classification. A face is eligible only if:
    #   (a) unclassified (is_target IS NULL) and not superseded
    #   (b) no human has interacted with it (classified_by_user_id IS NULL)
    # Bootstrapped images are included: bootstrap auto-classifies the single best face,
    # but multi-face images have additional unclassified faces that need inference.
    # The per-image dedup below ensures only the closest face per image is matched.
    unclassified_rows = execute_query(
        """
        SELECT f.id, f.image_id, f.encoding FROM faces f
        JOIN images i ON f.image_id = i.id
        WHERE i.project_id = %s
          AND f.is_target IS NULL
          AND f.superseded_by IS NULL
          AND f.classified_by_user_id IS NULL
        """,
        (project["id"],),
        fetch=True,
    )

    candidate_count = len(unclassified_rows) if unclassified_rows else 0
    logger.info(
        f"Project {project['id']}: {candidate_count} candidate faces for inference (non-bootstrap, no human edits)"
    )

    classified_count = 0
    threshold = project.get("distance_threshold", 0.6)

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

        enc = row["encoding"]
        if enc is None or len(enc) != 1024:
            logger.warning(
                f"Skipping candidate face {row['id']} with invalid encoding ({len(enc) if enc else 'None'} bytes)"
            )
            continue
        encoding = np.frombuffer(enc, dtype=np.float64).copy()
        distance = face_recognition.face_distance([centroid], encoding)[0]
        face_distances.append((row["id"], row["image_id"], float(distance)))

    # Phase 2: per-image dedup — only the closest face below threshold is a match;
    # all other faces on the same image are rejected (a person appears once per image)
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

    # Flush batch UPDATEs using CASE/WHEN for single round-trip per batch
    t_db_start = time.monotonic()
    for i in range(0, len(pending_updates), UPDATE_BATCH_SIZE):
        if shutdown_requested:
            break
        batch = pending_updates[i : i + UPDATE_BATCH_SIZE]
        if not batch:
            break

        face_ids = [item[0] for item in batch]
        id_placeholders = ",".join(["%s"] * len(face_ids))

        # Build CASE expressions for is_target and confidence
        target_cases = " ".join(["WHEN %s THEN %s" for _ in batch])
        conf_cases = " ".join(["WHEN %s THEN %s" for _ in batch])

        # Flatten params: target CASE pairs, confidence CASE pairs, then IN list
        params: list = []
        for face_id, is_target_val, _ in batch:
            params.extend([face_id, is_target_val])
        for face_id, _, dist in batch:
            params.extend([face_id, dist])
        params.extend(face_ids)

        execute_query(
            f"UPDATE faces SET "
            f"is_target = CASE id {target_cases} END, "
            f"confidence = CASE id {conf_cases} END, "
            f"classified_by = 'model', "
            f"classified_at = NOW() "
            f"WHERE id IN ({id_placeholders}) "
            f"AND classified_by_user_id IS NULL "
            f"AND superseded_by IS NULL",
            tuple(params),
            fetch=False,
        )
        classified_count += len(batch)
    t_db = time.monotonic()

    matches = sum(1 for _, t, _ in pending_updates if t == 1)
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


def write_sdc_claims(project: dict[str, Any]) -> int:
    """Write SDC P180 claims for all faces marked as target.

    Called when sdc_write_requested=1. Processes all pending faces in batches,
    clears the flag when done (or sets sdc_write_error on failure).
    """
    project_id = project["id"]
    user_id = project.get("sdc_write_user_id") or project["user_id"]
    logger.info(f"Starting SDC writes for project {project_id}")

    access_token = _refresh_worker_token(user_id)
    if not access_token:
        logger.error(f"Cannot obtain valid token for user {user_id}, aborting SDC writes")
        execute_query(
            "UPDATE projects SET sdc_write_requested = 0, sdc_write_user_id = NULL, "
            "sdc_write_error = 'Token expired or user not found. Please log in again.' WHERE id = %s",
            (project_id,),
            fetch=False,
        )
        return 0

    try:
        csrf_token = _get_csrf_token(access_token)
    except Exception as e:
        logger.error(f"Failed to get CSRF token for project {project_id}: {e}")
        execute_query(
            "UPDATE projects SET sdc_write_requested = 0, sdc_write_user_id = NULL, sdc_write_error = 'Failed to get CSRF token' WHERE id = %s",
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
            "UPDATE projects SET sdc_write_requested = 0, sdc_write_user_id = NULL, sdc_write_error = 'Invalid QID format' WHERE id = %s",
            (project_id,),
            fetch=False,
        )
        return 0

    total_written = 0
    token_retries = 0
    SDC_BATCH = 50  # Faces per DB fetch batch

    while not shutdown_requested:
        # Check if user cancelled the write via the stop button
        try:
            cancel_check = execute_query(
                "SELECT sdc_write_requested FROM projects WHERE id = %s",
                (project_id,),
                fetch=True,
            )
            if not cancel_check or cancel_check[0]["sdc_write_requested"] == 0:
                logger.info(f"SDC write cancelled by user for project {project_id} ({total_written} written so far)")
                return total_written
        except DatabaseError:
            pass  # Non-fatal: continue writing if we can't check

        # Fetch next batch of unwritten faces
        faces_to_write = execute_query(
            """
            SELECT f.id as face_id, i.id as image_id, i.commons_page_id
            FROM faces f
            JOIN images i ON f.image_id = i.id
            WHERE i.project_id = %s AND f.is_target = 1 AND f.sdc_written = 0
            AND f.classified_by != 'bootstrap' AND i.bootstrapped = 0
            AND f.superseded_by IS NULL
            LIMIT %s
            """,
            (project_id, SDC_BATCH),
            fetch=True,
        )

        if not faces_to_write:
            break

        for row in faces_to_write:
            if shutdown_requested:
                break

            page_id = row["commons_page_id"]
            face_id = row["face_id"]
            mid = f"M{page_id}"

            try:
                # Cross-project dedup: claim this page+qid pair before making the API call.
                # INSERT IGNORE returns rowcount=0 when a row already exists.
                claim_rows = execute_query(
                    "INSERT IGNORE INTO sdc_claims (commons_page_id, wikidata_qid, project_id, face_id) "
                    "VALUES (%s, %s, %s, %s)",
                    (page_id, qid, project_id, face_id),
                    fetch=False,
                )
                if claim_rows == 0:
                    # INSERT was a no-op: a row already exists for this page+qid.
                    # Inspect it before deciding whether to skip, retry, or reclaim.
                    existing_claim = execute_query(
                        "SELECT project_id, written_at, claimed_at FROM sdc_claims "
                        "WHERE commons_page_id = %s AND wikidata_qid = %s",
                        (page_id, qid),
                        fetch=True,
                    )
                    if not existing_claim:
                        # Row vanished between INSERT and SELECT (e.g. cascade delete).
                        # Skip this face; it will be retried next write cycle.
                        logger.warning(
                            f"SDC claim row missing for {qid} on M{page_id} after INSERT IGNORE, "
                            f"skipping face {face_id}"
                        )
                        continue

                    existing = existing_claim[0]

                    if existing["written_at"] is not None:
                        # The claim was already successfully written to Commons.
                        # Safe to mark this face as done.
                        logger.info(
                            f"SDC claim for {qid} on M{page_id} already written "
                            f"(written_at={existing['written_at']}), marking face {face_id} as written"
                        )
                        execute_query(
                            "UPDATE faces SET sdc_written = 1 WHERE id = %s",
                            (face_id,),
                            fetch=False,
                        )
                        total_written += 1
                        continue

                    if int(existing["project_id"]) == int(project_id):
                        # Same project owns this claim but hasn't written it yet
                        # (e.g. a previous write attempt failed mid-way).
                        # Fall through to the write logic below.
                        logger.info(
                            f"SDC claim for {qid} on M{page_id} previously claimed by this project, "
                            f"retrying write for face {face_id}"
                        )
                    else:
                        # A different project holds the claim with written_at IS NULL.
                        # Try to reclaim if the claim is stale (crashed/abandoned worker).
                        reclaimed = execute_query(
                            "UPDATE sdc_claims SET project_id = %s, face_id = %s, claimed_at = NOW() "
                            "WHERE commons_page_id = %s AND wikidata_qid = %s "
                            "AND written_at IS NULL "
                            "AND claimed_at < NOW() - INTERVAL %s MINUTE",
                            (project_id, face_id, page_id, qid, CLAIM_EXPIRY_MINUTES),
                            fetch=False,
                        )
                        if reclaimed == 0:
                            # Another project holds a fresh, non-stale claim — skip for now.
                            logger.info(
                                f"SDC claim for {qid} on M{page_id} held by project "
                                f"{existing['project_id']} (claimed_at={existing['claimed_at']}), "
                                f"skipping face {face_id} this cycle"
                            )
                            continue
                        logger.info(
                            f"Reclaimed stale SDC claim for {qid} on M{page_id} "
                            f"from project {existing['project_id']} for face {face_id}"
                        )

                # Idempotency check
                claim_params = {
                    "action": "wbgetclaims",
                    "entity": mid,
                    "property": "P180",
                    "format": "json",
                }
                claim_headers = {"Authorization": f"Bearer {access_token}"}

                claim_resp = _api_request(COMMONS_API_URL, params=claim_params, headers=claim_headers)
                claim_data = claim_resp.json()

                if "error" in claim_data:
                    error_code = claim_data["error"].get("code", "unknown")
                    error_info = claim_data["error"].get("info", "Unknown error")
                    if error_code in ("badtoken", "permissiondenied", "notloggedin", "mwoauth-invalid-authorization"):
                        token_retries += 1
                        if token_retries > MAX_TOKEN_RETRIES:
                            msg = _sdc_error_message(error_code, error_info)
                            logger.error(
                                f"SDC token retry limit reached during idempotency check for project {project_id}: "
                                f"{error_code}"
                            )
                            execute_query(
                                "UPDATE projects SET sdc_write_requested = 0, sdc_write_user_id = NULL, sdc_write_error = %s WHERE id = %s",
                                (msg, project_id),
                                fetch=False,
                            )
                            return total_written
                        logger.warning(f"Auth error on wbgetclaims for {mid}: {error_code}, refreshing token")
                        refreshed = _refresh_worker_token(user_id)
                        if refreshed:
                            access_token = refreshed
                            claim_headers = {"Authorization": f"Bearer {access_token}"}
                        try:
                            csrf_token = _get_csrf_token(access_token)
                        except Exception:
                            msg = _sdc_error_message(error_code, error_info)
                            logger.error(f"Failed to refresh tokens for project {project_id}")
                            execute_query(
                                "UPDATE projects SET sdc_write_requested = 0, sdc_write_user_id = NULL, sdc_write_error = %s WHERE id = %s",
                                (msg, project_id),
                                fetch=False,
                            )
                            return total_written
                        continue
                    elif error_code in _SDC_PER_FACE_SKIP_ERRORS:
                        logger.warning(f"Skipping face {face_id} on {mid}: {error_code} during idempotency check")
                        # Mark this face as completed for SDC writes and release its claim so it is not retried indefinitely.
                        execute_query(
                            "UPDATE faces SET sdc_written = 1 WHERE id = %s",
                            (face_id,),
                            fetch=False,
                        )
                        execute_query(
                            "DELETE FROM sdc_claims WHERE face_id = %s",
                            (face_id,),
                            fetch=False,
                        )
                        continue
                    else:
                        msg = _sdc_error_message(error_code, error_info)
                        logger.error(f"SDC idempotency check error for {mid}: {claim_data['error']}")
                        execute_query(
                            "UPDATE projects SET sdc_write_requested = 0, sdc_write_user_id = NULL, sdc_write_error = %s WHERE id = %s",
                            (msg, project_id),
                            fetch=False,
                        )
                        return total_written

                already_exists = False
                claims = claim_data.get("claims", {}).get("P180", [])
                for claim in claims:
                    snak = claim.get("mainsnak", {})
                    if snak.get("datavalue", {}).get("value", {}).get("id") == qid:
                        already_exists = True
                        break

                if already_exists:
                    logger.info(f"SDC P180 claim for {qid} already exists on {mid}, marking face {face_id} as written")
                    execute_query(
                        "UPDATE faces SET sdc_written = 1 WHERE id = %s",
                        (face_id,),
                        fetch=False,
                    )
                    execute_query(
                        "UPDATE sdc_claims SET written_at = NOW() "
                        "WHERE commons_page_id = %s AND wikidata_qid = %s AND written_at IS NULL",
                        (page_id, qid),
                        fetch=False,
                    )
                    image_id = row["image_id"]
                    execute_query(
                        "UPDATE images SET bootstrapped = 1 WHERE id = %s AND bootstrapped = 0",
                        (image_id,),
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
                    if error_code in ("badtoken", "permissiondenied", "notloggedin"):
                        token_retries += 1
                        if token_retries > MAX_TOKEN_RETRIES:
                            msg = _sdc_error_message(error_code, error_info)
                            logger.error(f"SDC token retry limit reached for project {project_id}: {error_code}")
                            execute_query(
                                "UPDATE projects SET sdc_write_requested = 0, sdc_write_user_id = NULL, sdc_write_error = %s WHERE id = %s",
                                (msg, project_id),
                                fetch=False,
                            )
                            return total_written
                        # Try refreshing the access token first
                        refreshed = _refresh_worker_token(user_id)
                        if refreshed:
                            access_token = refreshed
                            claim_headers = {"Authorization": f"Bearer {access_token}"}
                        try:
                            csrf_token = _get_csrf_token(access_token)
                        except Exception:
                            msg = _sdc_error_message(error_code, error_info)
                            logger.error(f"Failed to refresh tokens for project {project_id}")
                            execute_query(
                                "UPDATE projects SET sdc_write_requested = 0, sdc_write_user_id = NULL, sdc_write_error = %s WHERE id = %s",
                                (msg, project_id),
                                fetch=False,
                            )
                            return total_written
                        continue
                    elif error_code in _SDC_PER_FACE_SKIP_ERRORS:
                        logger.warning(f"Skipping face {face_id} on {mid}: {error_code} during write")
                        # Mark this face as terminal so it is not retried indefinitely
                        execute_query(
                            "UPDATE faces SET sdc_written = 1 WHERE id = %s",
                            (face_id,),
                            fetch=False,
                        )
                        continue
                    else:
                        # Any other API error — abort entire write
                        msg = _sdc_error_message(error_code, error_info)
                        logger.error(f"SDC Write error for {mid}: {edit_json['error']}")
                        execute_query(
                            "UPDATE projects SET sdc_write_requested = 0, sdc_write_user_id = NULL, sdc_write_error = %s WHERE id = %s",
                            (msg, project_id),
                            fetch=False,
                        )
                        logger.info(
                            f"SDC writes aborted for project {project_id}: {total_written} written before error"
                        )
                        return total_written

                if not edit_json.get("success"):
                    logger.warning(
                        f"SDC write for face {face_id} on {mid} returned unexpected response "
                        f"(no 'success' key): {json.dumps(edit_json)[:500]}"
                    )

                logger.info(f"SDC P180 claim for {qid} written to {mid} (face {face_id})")
                execute_query(
                    "UPDATE faces SET sdc_written = 1 WHERE id = %s",
                    (face_id,),
                    fetch=False,
                )
                execute_query(
                    "UPDATE sdc_claims SET written_at = NOW() WHERE commons_page_id = %s AND wikidata_qid = %s",
                    (page_id, qid),
                    fetch=False,
                )
                total_written += 1

                # Rate limiting for MediaWiki API
                time.sleep(1)

            except Exception as e:
                logger.error(f"Error writing SDC for face {face_id} on {mid}: {e}")
                execute_query(
                    "UPDATE projects SET sdc_write_requested = 0, sdc_write_user_id = NULL, sdc_write_error = %s WHERE id = %s",
                    (str(e)[:1024], project_id),
                    fetch=False,
                )
                logger.info(f"SDC writes aborted for project {project_id}: {total_written} written before error")
                return total_written

    # --- P180 Removal Phase ---
    # Process faces with sdc_removal_pending=1 (rejected bootstrap faces)
    total_removed = 0
    token_retries = 0
    REMOVAL_BATCH = 50

    while not shutdown_requested:
        # Check if user cancelled the write via the stop button
        try:
            cancel_check = execute_query(
                "SELECT sdc_write_requested FROM projects WHERE id = %s",
                (project_id,),
                fetch=True,
            )
            if not cancel_check or cancel_check[0]["sdc_write_requested"] == 0:
                logger.info(f"SDC removal cancelled by user for project {project_id}")
                break
        except DatabaseError:
            pass

        faces_to_remove = execute_query(
            """
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
            """,
            (project_id, REMOVAL_BATCH),
            fetch=True,
        )

        if not faces_to_remove:
            break

        for row in faces_to_remove:
            if shutdown_requested:
                break

            page_id = row["commons_page_id"]
            mid = f"M{page_id}"

            try:
                still_pending = execute_query(
                    """
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
                    """,
                    (page_id, project_id),
                    fetch=True,
                )
                if not still_pending:
                    logger.info(f"Skipping removal for M{page_id}: sibling approved since selection")
                    continue

                claim_params = {
                    "action": "wbgetclaims",
                    "entity": mid,
                    "property": "P180",
                    "format": "json",
                }
                claim_headers = {"Authorization": f"Bearer {access_token}"}

                claim_resp = _api_request(COMMONS_API_URL, params=claim_params, headers=claim_headers)
                claim_data = claim_resp.json()

                if "error" in claim_data:
                    error_code = claim_data["error"].get("code", "unknown")
                    error_info = claim_data["error"].get("info", "Unknown error")
                    if error_code in ("badtoken", "permissiondenied", "notloggedin", "mwoauth-invalid-authorization"):
                        token_retries += 1
                        if token_retries > MAX_TOKEN_RETRIES:
                            msg = _sdc_error_message(error_code, error_info)
                            logger.error(
                                f"SDC token retry limit reached during removal idempotency check "
                                f"for project {project_id}: {error_code}"
                            )
                            execute_query(
                                "UPDATE projects SET sdc_write_requested = 0, sdc_write_user_id = NULL, sdc_write_error = %s WHERE id = %s",
                                (msg, project_id),
                                fetch=False,
                            )
                            return total_written
                        logger.warning(f"Auth error on wbgetclaims (removal) for {mid}: {error_code}, refreshing token")
                        refreshed = _refresh_worker_token(user_id)
                        if refreshed:
                            access_token = refreshed
                            claim_headers = {"Authorization": f"Bearer {access_token}"}
                        try:
                            csrf_token = _get_csrf_token(access_token)
                        except Exception:
                            msg = _sdc_error_message(error_code, error_info)
                            logger.error(f"Failed to refresh tokens during removal for project {project_id}")
                            execute_query(
                                "UPDATE projects SET sdc_write_requested = 0, sdc_write_user_id = NULL, sdc_write_error = %s WHERE id = %s",
                                (msg, project_id),
                                fetch=False,
                            )
                            return total_written
                        continue
                    elif error_code in _SDC_PER_FACE_SKIP_ERRORS:
                        logger.warning(f"Skipping removal for {mid}: {error_code} during idempotency check")
                        execute_query(
                            "UPDATE faces f JOIN images i ON f.image_id = i.id "
                            "SET f.sdc_removal_pending = 0 "
                            "WHERE i.commons_page_id = %s AND i.project_id = %s "
                            "AND f.sdc_removal_pending = 1",
                            (page_id, project_id),
                            fetch=False,
                        )
                        total_removed += 1
                        continue
                    else:
                        msg = _sdc_error_message(error_code, error_info)
                        logger.error(f"SDC removal idempotency check error for {mid}: {claim_data['error']}")
                        execute_query(
                            "UPDATE projects SET sdc_write_requested = 0, sdc_write_user_id = NULL, sdc_write_error = %s WHERE id = %s",
                            (msg, project_id),
                            fetch=False,
                        )
                        logger.info(
                            f"SDC removals aborted for project {project_id}: {total_removed} removed before error"
                        )
                        return total_written

                claims = claim_data.get("claims", {}).get("P180", [])
                target_guid = None
                for claim in claims:
                    snak = claim.get("mainsnak", {})
                    if snak.get("datavalue", {}).get("value", {}).get("id") == qid:
                        target_guid = claim.get("id")
                        break

                if not target_guid:
                    logger.info(f"P180 claim for {qid} not found on {mid}, clearing removal flag")
                    execute_query(
                        "UPDATE faces f JOIN images i ON f.image_id = i.id "
                        "SET f.sdc_removal_pending = 0 "
                        "WHERE i.commons_page_id = %s AND i.project_id = %s "
                        "AND f.sdc_removal_pending = 1",
                        (page_id, project_id),
                        fetch=False,
                    )
                    total_removed += 1
                    continue

                remove_data = {
                    "action": "wbremoveclaims",
                    "claim": target_guid,
                    "token": csrf_token,
                    "summary": f"WikiVisage: Removing depicts (P180) claim for {qid} (human review)",
                    "format": "json",
                    "bot": "1",
                    "maxlag": "5",
                }

                remove_resp = _api_request(
                    COMMONS_API_URL,
                    data=remove_data,
                    headers=claim_headers,
                    method="post",
                )
                remove_json = remove_resp.json()

                if "error" in remove_json:
                    error_code = remove_json["error"].get("code")
                    error_info = remove_json["error"].get("info", "Unknown error")
                    if error_code in ("badtoken", "permissiondenied", "notloggedin"):
                        token_retries += 1
                        if token_retries > MAX_TOKEN_RETRIES:
                            msg = _sdc_error_message(error_code, error_info)
                            logger.error(f"SDC token retry limit reached during removal for project {project_id}")
                            execute_query(
                                "UPDATE projects SET sdc_write_requested = 0, sdc_write_user_id = NULL, sdc_write_error = %s WHERE id = %s",
                                (msg, project_id),
                                fetch=False,
                            )
                            return total_written
                        refreshed = _refresh_worker_token(user_id)
                        if refreshed:
                            access_token = refreshed
                            claim_headers = {"Authorization": f"Bearer {access_token}"}
                        try:
                            csrf_token = _get_csrf_token(access_token)
                        except Exception:
                            msg = _sdc_error_message(error_code, error_info)
                            logger.error(f"Failed to refresh tokens during removal for project {project_id}")
                            execute_query(
                                "UPDATE projects SET sdc_write_requested = 0, sdc_write_user_id = NULL, sdc_write_error = %s WHERE id = %s",
                                (msg, project_id),
                                fetch=False,
                            )
                            return total_written
                        continue
                    elif error_code in _SDC_PER_FACE_SKIP_ERRORS:
                        logger.warning(f"Skipping removal for {mid}: {error_code} during claim removal")
                        execute_query(
                            "UPDATE faces f JOIN images i ON f.image_id = i.id "
                            "SET f.sdc_removal_pending = 0 "
                            "WHERE i.commons_page_id = %s AND i.project_id = %s "
                            "AND f.sdc_removal_pending = 1",
                            (page_id, project_id),
                            fetch=False,
                        )
                        total_removed += 1
                        continue
                    else:
                        msg = _sdc_error_message(error_code, error_info)
                        logger.error(f"SDC removal error for {mid}: {remove_json['error']}")
                        execute_query(
                            "UPDATE projects SET sdc_write_requested = 0, sdc_write_user_id = NULL, sdc_write_error = %s WHERE id = %s",
                            (msg, project_id),
                            fetch=False,
                        )
                        logger.info(
                            f"SDC removals aborted for project {project_id}: {total_removed} removed before error"
                        )
                        return total_written

                logger.info(f"SDC removal succeeded for {mid} (project {project_id}): removed claim {target_guid}")

                execute_query(
                    "UPDATE faces f JOIN images i ON f.image_id = i.id "
                    "SET f.sdc_removal_pending = 0 "
                    "WHERE i.commons_page_id = %s AND i.project_id = %s "
                    "AND f.sdc_removal_pending = 1",
                    (page_id, project_id),
                    fetch=False,
                )
                total_removed += 1

                time.sleep(1)

            except Exception as e:
                logger.error(f"Error removing SDC for {mid}: {e}")
                execute_query(
                    "UPDATE projects SET sdc_write_requested = 0, sdc_write_user_id = NULL, sdc_write_error = %s WHERE id = %s",
                    (str(e)[:1024], project_id),
                    fetch=False,
                )
                logger.info(f"SDC removals aborted for project {project_id}: {total_removed} removed before error")
                return total_written

    # Clear the flag — all done successfully (or shutdown requested)
    error_msg = None
    if shutdown_requested:
        error_msg = "Worker shutdown interrupted SDC writes"

    execute_query(
        "UPDATE projects SET sdc_write_requested = 0, sdc_write_user_id = NULL, sdc_write_error = %s WHERE id = %s",
        (error_msg, project_id),
        fetch=False,
    )

    logger.info(f"SDC for project {project_id}: {total_written} written, {total_removed} removed")
    return total_written


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
    target faces (>= min_confirmed), and still has unclassified faces remaining.
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
            "  WHERE i.project_id = p.id AND f.is_target IS NULL AND f.superseded_by IS NULL"
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


def process_project(project: dict[str, Any], *, skip_discovery: bool = False) -> None:
    """Orchestrate all phases for a single project.

    Args:
        project: Project row dict from DB (must include 'id', 'wikidata_qid', etc.).
        skip_discovery: If True, skip traverse_category and bootstrap_from_sparql.
    """
    t_project_start = time.monotonic()
    project_id = project["id"]
    logger.info(f"--- Processing project {project_id} ({project['wikidata_qid']}) ---")

    def _is_still_active() -> bool:
        """Re-check project status and claim ownership from DB.

        Returns False if the project is no longer active or if another worker
        has reclaimed it (e.g. because our claim expired).  Also refreshes the
        claim timestamp on success so the claim does not expire while we work.
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
            # Refresh claim timestamp so it doesn't expire while we work
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
            return True  # On DB error, keep going rather than silently dropping work

    try:
        # Early bail-out: the project may have been deleted / paused between the
        # moment it was queued in the thread-pool and the moment this thread
        # actually starts running.  Checking here avoids wasted traversal and
        # bootstrap work for projects that are no longer active.
        if not _is_still_active():
            return

        t_traversal = 0.0
        t_bootstrap = 0.0

        if skip_discovery:
            logger.info(f"Project {project_id}: skipping discovery (already fast-tracked)")
        else:
            # 1. Traversal
            t0 = time.monotonic()
            traverse_category(project)
            t_traversal = time.monotonic() - t0

            if shutdown_requested or not _is_still_active():
                return

            # 2. Bootstrap (if needed)
            t0 = time.monotonic()
            bootstrap_from_sparql(project)
            t_bootstrap = time.monotonic() - t0

            if shutdown_requested or not _is_still_active():
                return

        # 3. Image Download & Face Detection — process all pending images
        #    Interleave inference every 5 batches so results appear progressively
        t0 = time.monotonic()
        total_processed = 0
        while not shutdown_requested:
            batch_count = process_images(project)
            if batch_count == 0:
                break
            total_processed += batch_count
            logger.info(f"Processed batch of {batch_count} images ({total_processed} total so far)")

            # Check if project was paused/completed between batches
            if not _is_still_active():
                break
        t_images = time.monotonic() - t0

        # 3b. Auto-complete if no faces or insufficient faces detected
        if not shutdown_requested and _is_still_active():
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
            if total_faces == 0:
                execute_query(
                    "UPDATE projects SET status = 'completed', completion_reason = 'no_faces' "
                    "WHERE id = %s AND status = 'active'",
                    (project_id,),
                    fetch=False,
                )
                logger.info(f"Project {project_id}: 0 faces detected, auto-completed (no_faces)")
                t_total = time.monotonic() - t_project_start
                logger.info(
                    f"--- Project {project['id']} complete: "
                    f"traversal={t_traversal:.1f}s, bootstrap={t_bootstrap:.1f}s, "
                    f"images={t_images:.1f}s ({total_processed} processed), "
                    f"inference=skipped, total={t_total:.1f}s ---"
                )
                return
            elif total_faces <= 5 and total_faces < min_confirmed:
                execute_query(
                    "UPDATE projects SET status = 'completed', completion_reason = 'insufficient_faces' "
                    "WHERE id = %s AND status = 'active'",
                    (project_id,),
                    fetch=False,
                )
                logger.info(
                    f"Project {project_id}: only {total_faces} faces detected "
                    f"(need {min_confirmed}), auto-completed (insufficient_faces)"
                )
                t_total = time.monotonic() - t_project_start
                logger.info(
                    f"--- Project {project['id']} complete: "
                    f"traversal={t_traversal:.1f}s, bootstrap={t_bootstrap:.1f}s, "
                    f"images={t_images:.1f}s ({total_processed} processed), "
                    f"inference=skipped, total={t_total:.1f}s ---"
                )
                return

        # 4. Final Autonomous Inference (catch any remaining unclassified faces)
        t0 = time.monotonic()
        if not shutdown_requested and _is_still_active():
            fresh = execute_query(
                "SELECT * FROM projects WHERE id = %s AND status != 'deleted'", (project_id,), fetch=True
            )
            if fresh:
                run_autonomous_inference(fresh[0])
            else:
                run_autonomous_inference(project)
        t_inference = time.monotonic() - t0

        # 6. SDC writes are handled separately — triggered via web UI,
        #    processed in the main loop when sdc_write_requested=1.

        t_total = time.monotonic() - t_project_start
        logger.info(
            f"--- Project {project['id']} complete: "
            f"traversal={t_traversal:.1f}s, bootstrap={t_bootstrap:.1f}s, "
            f"images={t_images:.1f}s ({total_processed} processed), "
            f"inference={t_inference:.1f}s, total={t_total:.1f}s ---"
        )

    except Exception as e:
        logger.error(f"Failed to process project {project['id']}: {e}")


def signal_handler(signum, frame):
    """Graceful shutdown on SIGINT/SIGTERM."""
    global shutdown_requested
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    shutdown_requested = True


def main():
    """Main worker loop with distributed locking for concurrent multi-instance processing."""
    global _worker_id
    # Accept --worker-id from command line (used in jobs.yaml), fall back to hostname-pid.
    wid = None
    if "--worker-id" in sys.argv:
        idx = sys.argv.index("--worker-id")
        if idx + 1 < len(sys.argv):
            wid = sys.argv[idx + 1]
    _worker_id = wid or f"{socket.gethostname()}-{os.getpid()}"
    # Sanitize worker ID to prevent path traversal in heartbeat file path
    if not re.fullmatch(r"[a-zA-Z0-9._-]+", _worker_id):
        raise SystemExit(f"Invalid worker ID (must be alphanumeric/dot/hyphen/underscore): {_worker_id!r}")
    logger.info(
        f"Starting WikiVisage Worker {_worker_id} "
        f"(max_projects={MAX_CONCURRENT_PROJECTS}, image_threads={IMAGE_THREADS})"
    )

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Respect WIKIVISAGE_DB_POOL_SIZE env var if set; otherwise compute from concurrency settings.
        # Toolforge ToolsDB enforces max_user_connections=20 across ALL processes sharing the same
        # DB user account (gunicorn + all worker instances). Default gunicorn uses pool_size=2 per
        # worker (2 workers = 4 conns). Cap each worker at 6 so 2 workers + web stays <= 20.
        db_pool_env = os.environ.get("WIKIVISAGE_DB_POOL_SIZE")
        if db_pool_env:
            worker_pool_size = int(db_pool_env)
        else:
            worker_pool_size = min(MAX_CONCURRENT_PROJECTS * IMAGE_THREADS + 3, 6)
        if worker_pool_size > 6:
            logger.warning(
                "DB pool size %d may exceed Toolforge max_user_connections (20 shared across all processes). "
                "With 2 workers + gunicorn, keep each worker's pool <= 6.",
                worker_pool_size,
            )
        init_db(pool_size=worker_pool_size)
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        sys.exit(1)

    # Start persistent face detection subprocess pool — each subprocess imports
    # dlib/face_recognition once, eliminating 2-5s per-image spawn overhead.
    global _face_pool
    _face_pool = FaceDetectPool(pool_size=IMAGE_THREADS)
    _face_pool.start()

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
        if diag:
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
                _touch_heartbeat_file()

                # ---- Purge soft-deleted projects FIRST (quick, unblocks re-creation) ----
                try:

                    def _archive_and_purge(conn: Any, cursor: Any) -> int:
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

                    deleted = execute_transaction(_archive_and_purge)
                    if deleted:
                        logger.info(f"Purged {deleted} soft-deleted project(s)")
                except DatabaseError as exc:
                    logger.warning("Failed to purge soft-deleted projects: %s", exc)

                # ---- SDC writes (user-triggered, latency-sensitive) ----
                # Process SDC write requests before the active pipeline so users
                # don't wait minutes through crawl/detect/infer before their
                # "Send Edits to Wikimedia Commons" request is serviced.
                _process_sdc_writes()

                # ---- Claim active projects (atomic, distributed-safe) ----
                active_projects = _claim_active_projects(MAX_CONCURRENT_PROJECTS)

                # ---- Claim inference-eligible projects (skip those we already claimed above) ----
                active_ids = {p["id"] for p in active_projects}
                inference_projects = _claim_inference_projects(MAX_CONCURRENT_PROJECTS)
                inference_only = [p for p in inference_projects if p["id"] not in active_ids]

                logger.info(f"Poll: claimed {len(active_projects)} active, {len(inference_only)} inference-only")

                # Process active projects concurrently (claims released in wrapper).
                # Each thread handles its own discovery (traverse_category + bootstrap)
                # so all claimed projects start working immediately — no serial bottleneck.
                #
                # A dedicated single-thread executor (sdc_executor) handles SDC writes so
                # they never block the main thread's heartbeat refresh or result processing.
                seen_ids = set()
                if active_projects:
                    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_PROJECTS) as executor:
                        futures = {}
                        for project in active_projects:
                            if shutdown_requested:
                                break
                            seen_ids.add(project["id"])
                            future = executor.submit(_process_and_release, project)
                            futures[future] = project["id"]

                        last_heartbeat = time.time()
                        last_sdc_check = time.time()
                        while futures:
                            if shutdown_requested:
                                break

                            # Refresh heartbeat + claim timestamps during long processing cycles
                            if time.time() - last_heartbeat >= 60:
                                try:
                                    execute_query(
                                        "REPLACE INTO worker_heartbeat (id, last_seen) VALUES (1, NOW())",
                                        fetch=False,
                                    )
                                    _touch_heartbeat_file()
                                except Exception:
                                    pass
                                _refresh_claims()
                                last_heartbeat = time.time()

                            # Check for wake-up signal every iteration (even if no futures done)
                            if os.path.exists(WAKE_FILE_PATH):
                                try:
                                    os.remove(WAKE_FILE_PATH)
                                except OSError:
                                    pass
                                logger.info("Wake-up signal received mid-cycle, checking for new work")

                                _process_sdc_writes()
                                last_sdc_check = time.time()

                                unfinished_futures = [f for f in futures if not f.done()]
                                free_slots = MAX_CONCURRENT_PROJECTS - len(unfinished_futures)
                                if free_slots <= 0:
                                    logger.debug(
                                        "All %d worker slots occupied, skipping mid-cycle claim",
                                        MAX_CONCURRENT_PROJECTS,
                                    )
                                else:
                                    new_projects = _claim_active_projects(free_slots)
                                    for project in new_projects:
                                        if project["id"] not in seen_ids:
                                            seen_ids.add(project["id"])
                                            logger.info(f"Adding new project {project['id']} to current cycle")
                                            future = executor.submit(_process_and_release, project)
                                            futures[future] = project["id"]

                            # Periodic SDC check (works across pods where wake file doesn't)
                            if time.time() - last_sdc_check >= 10:
                                _process_sdc_writes()
                                last_sdc_check = time.time()

                            done = {f for f in futures if f.done()}
                            if not done:
                                time.sleep(1)
                                continue

                            for future in done:
                                project_id = futures.pop(future)
                                try:
                                    future.result()
                                except Exception as e:
                                    logger.error(f"Project {project_id} processing failed: {e}")

                # Run inference for eligible projects not already processed (claims released in wrapper)
                if inference_only:
                    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_PROJECTS) as executor:
                        futures = {}
                        for project in inference_only:
                            if shutdown_requested:
                                break
                            logger.info(
                                f"Running inference-only for project {project['id']} (status={project.get('status')})"
                            )
                            future = executor.submit(_infer_and_release, project)
                            futures[future] = project["id"]

                        last_heartbeat_inf = time.time()
                        last_sdc_check_inf = time.time()
                        while futures:
                            if shutdown_requested:
                                break

                            # Refresh heartbeat + claims during inference
                            if time.time() - last_heartbeat_inf >= 60:
                                try:
                                    execute_query(
                                        "REPLACE INTO worker_heartbeat (id, last_seen) VALUES (1, NOW())",
                                        fetch=False,
                                    )
                                    _touch_heartbeat_file()
                                except Exception:
                                    pass
                                _refresh_claims()
                                last_heartbeat_inf = time.time()

                            # Check for wake-up signal during inference
                            if os.path.exists(WAKE_FILE_PATH):
                                try:
                                    os.remove(WAKE_FILE_PATH)
                                except OSError:
                                    pass
                                _process_sdc_writes()
                                last_sdc_check_inf = time.time()

                            # Periodic SDC check (works across pods where wake file doesn't)
                            if time.time() - last_sdc_check_inf >= 10:
                                _process_sdc_writes()
                                last_sdc_check_inf = time.time()

                            done = {f for f in futures if f.done()}
                            if not done:
                                time.sleep(1)
                                continue

                            for future in done:
                                project_id = futures.pop(future)
                                try:
                                    classified = future.result()
                                    logger.info(f"Inference classified {classified} faces for project {project_id}")
                                except Exception as e:
                                    logger.error(f"Inference failed for project {project_id}: {e}")

                # Add random jitter (0–15s) so multiple workers desynchronize
                # and don't all race to claim the same projects every cycle.
                jitter = random.randint(0, 15)
                last_sdc_check_sleep = time.time()
                for _ in range(POLL_INTERVAL + jitter):
                    if shutdown_requested:
                        break
                    if os.path.exists(WAKE_FILE_PATH):
                        try:
                            os.remove(WAKE_FILE_PATH)
                        except OSError:
                            pass
                        logger.info("Wake-up signal received, starting next cycle")
                        break
                    if time.time() - last_sdc_check_sleep >= 10:
                        written = _process_sdc_writes()
                        last_sdc_check_sleep = time.time()
                        if written:
                            break
                    time.sleep(1)

            except DatabaseError as e:
                logger.error(f"Database error in main loop: {e}")
                time.sleep(10)
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}")
                time.sleep(10)

    finally:
        logger.info("Worker %s shutting down, releasing claims and closing resources.", _worker_id)
        _release_all_claims()
        if _face_pool is not None:
            _face_pool.shutdown()
        close_pool()


if __name__ == "__main__":
    # spawn avoids fork-safety issues with threads + dlib's C++ code
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
    main()
