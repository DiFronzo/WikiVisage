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


class TokenDecryptionError(Exception):
    """Raised when a Fernet-encrypted token cannot be decrypted.

    This indicates either a wrong/rotated key or corrupted ciphertext.
    Callers should treat this as a fatal auth failure and trigger re-authentication.
    """


_fernet = None
_TOKEN_KEY = os.environ.get("WIKIVISAGE_TOKEN_KEY", "")

if _TOKEN_KEY:
    try:
        from cryptography.fernet import Fernet

        _fernet = Fernet(_TOKEN_KEY.encode() if isinstance(_TOKEN_KEY, str) else _TOKEN_KEY)
        logger.info("Token encryption enabled (WIKIVISAGE_TOKEN_KEY is set)")
    except Exception:
        raise RuntimeError(
            "WIKIVISAGE_TOKEN_KEY is set but invalid — refusing to start with broken encryption. "
            'Generate a valid key with: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"'
        ) from None
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


def _looks_like_fernet(value: str) -> bool:
    """Return True if *value* appears to be a Fernet ciphertext.

    Fernet tokens are URL-safe base64 strings whose first byte is always
    ``0x80`` (the version marker), which produces the fixed prefix ``gAAA``
    in base64url encoding.  Checking for this prefix reliably distinguishes
    encrypted tokens from legacy plaintext OAuth tokens.
    """
    return value.startswith("gAAA")


def decrypt_token(stored: str | bytes) -> str:
    """Decrypt a token read from the database.

    Handles three cases:
    1. Encryption enabled + valid ciphertext → decrypted plaintext.
    2. Encryption enabled + Fernet-looking ciphertext that cannot be decrypted
       (wrong key / corrupted) → logs a warning and raises :exc:`TokenDecryptionError`.
       Callers must handle this and trigger re-authentication.
    3. Encryption enabled + legacy plaintext (pre-encryption) → returned as-is.
    4. Encryption disabled → returned as-is.

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
        if _looks_like_fernet(stored):
            logger.warning(
                "Failed to decrypt a Fernet-looking token — possible key rotation or "
                "data corruption.  Triggering re-authentication."
            )
            raise TokenDecryptionError(
                "Token appears to be Fernet-encrypted but could not be decrypted. "
                "The encryption key may have been rotated or the token is corrupted."
            ) from None
        # Value does not look like a Fernet token: treat as legacy plaintext stored
        # before encryption was enabled and return it unchanged.
        return stored
