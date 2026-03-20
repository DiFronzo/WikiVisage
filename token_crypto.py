"""
Token encryption helpers for OAuth tokens stored in the database.

Uses Fernet symmetric encryption (AES-128-CBC + HMAC-SHA256) from the
``cryptography`` library.  Encryption is **opt-in**: when the environment
variable ``WIKIVISAGE_TOKEN_KEY`` is set to a valid Fernet key, tokens are
encrypted before DB writes and decrypted on reads.  When the variable is
absent, tokens pass through as plaintext — this keeps local development
and existing deployments working without changes.

Generate a key once::

    python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

Then set it::

    toolforge envvars create WIKIVISAGE_TOKEN_KEY "<key>"
"""

import logging
import os

logger = logging.getLogger(__name__)

_fernet = None
_TOKEN_KEY = os.environ.get("WIKIVISAGE_TOKEN_KEY", "")

if _TOKEN_KEY:
    try:
        from cryptography.fernet import Fernet

        _fernet = Fernet(_TOKEN_KEY.encode() if isinstance(_TOKEN_KEY, str) else _TOKEN_KEY)
        logger.info("Token encryption enabled (WIKIVISAGE_TOKEN_KEY is set)")
    except Exception:
        logger.exception(
            "WIKIVISAGE_TOKEN_KEY is set but invalid — tokens will NOT be encrypted. "
            'Generate a valid key with: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"'
        )
        _fernet = None
else:
    logger.info("Token encryption disabled (WIKIVISAGE_TOKEN_KEY not set)")


def encrypt_token(plaintext: str) -> str:
    """Encrypt a token string for database storage.

    Returns the Fernet ciphertext as a UTF-8 string when encryption is
    enabled, or the original plaintext when it is not.
    """
    if not _fernet or not plaintext:
        return plaintext
    return _fernet.encrypt(plaintext.encode("utf-8")).decode("utf-8")


def decrypt_token(stored: str | bytes) -> str:
    """Decrypt a token read from the database.

    Handles three cases gracefully:
    1. Encryption enabled + valid ciphertext → decrypted plaintext.
    2. Encryption enabled + legacy plaintext → returns as-is (migration-safe).
    3. Encryption disabled → returns as-is.

    The ``bytes`` input type is accepted because PyMySQL may return
    VARBINARY columns as ``bytes``.
    """
    if isinstance(stored, bytes):
        stored = stored.decode("utf-8")
    if not stored:
        return stored
    if not _fernet:
        return stored
    try:
        return _fernet.decrypt(stored.encode("utf-8")).decode("utf-8")
    except Exception:
        # Likely a legacy plaintext token stored before encryption was enabled.
        # Return as-is so the app keeps working during migration.
        return stored


def is_encryption_enabled() -> bool:
    """Return True when token encryption is active."""
    return _fernet is not None
