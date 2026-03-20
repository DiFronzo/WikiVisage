"""Tests for token_crypto module — Fernet encrypt/decrypt of OAuth tokens."""

import importlib
import os
from unittest.mock import patch

import pytest


@pytest.fixture()
def _no_token_key():
    """Ensure WIKIVISAGE_TOKEN_KEY is unset, then reload the module."""
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("WIKIVISAGE_TOKEN_KEY", None)
        import token_crypto

        importlib.reload(token_crypto)
        yield token_crypto


@pytest.fixture()
def _with_token_key():
    """Set a valid Fernet key, then reload the module."""
    from cryptography.fernet import Fernet

    key = Fernet.generate_key().decode()
    with patch.dict(os.environ, {"WIKIVISAGE_TOKEN_KEY": key}):
        import token_crypto

        importlib.reload(token_crypto)
        yield token_crypto


@pytest.fixture()
def _with_bad_key():
    """Set an invalid key, then reload the module."""
    with patch.dict(os.environ, {"WIKIVISAGE_TOKEN_KEY": "not-a-valid-fernet-key"}):
        import token_crypto

        importlib.reload(token_crypto)
        yield token_crypto


# --- Encryption disabled (no key) ---


class TestEncryptionDisabled:
    def test_is_encryption_enabled_false(self, _no_token_key):
        assert _no_token_key.is_encryption_enabled() is False

    def test_encrypt_passthrough(self, _no_token_key):
        token = "my-oauth-token-abc123"
        assert _no_token_key.encrypt_token(token) == token

    def test_decrypt_passthrough(self, _no_token_key):
        token = "my-oauth-token-abc123"
        assert _no_token_key.decrypt_token(token) == token

    def test_decrypt_bytes_passthrough(self, _no_token_key):
        token = b"my-oauth-token-abc123"
        assert _no_token_key.decrypt_token(token) == "my-oauth-token-abc123"

    def test_encrypt_empty_string(self, _no_token_key):
        assert _no_token_key.encrypt_token("") == ""

    def test_decrypt_empty_string(self, _no_token_key):
        assert _no_token_key.decrypt_token("") == ""


# --- Encryption enabled (valid key) ---


class TestEncryptionEnabled:
    def test_is_encryption_enabled_true(self, _with_token_key):
        assert _with_token_key.is_encryption_enabled() is True

    def test_encrypt_returns_different_value(self, _with_token_key):
        token = "my-oauth-token-abc123"
        encrypted = _with_token_key.encrypt_token(token)
        assert encrypted != token
        assert isinstance(encrypted, str)

    def test_roundtrip(self, _with_token_key):
        token = "ya29.a0AfH6SMBx_example_token"
        encrypted = _with_token_key.encrypt_token(token)
        decrypted = _with_token_key.decrypt_token(encrypted)
        assert decrypted == token

    def test_roundtrip_unicode(self, _with_token_key):
        token = "token-with-ünïcödé-chars"
        encrypted = _with_token_key.encrypt_token(token)
        assert _with_token_key.decrypt_token(encrypted) == token

    def test_decrypt_bytes_ciphertext(self, _with_token_key):
        """PyMySQL may return VARBINARY columns as bytes."""
        token = "my-oauth-token-abc123"
        encrypted = _with_token_key.encrypt_token(token)
        encrypted_bytes = encrypted.encode("utf-8")
        assert _with_token_key.decrypt_token(encrypted_bytes) == token

    def test_encrypt_empty_returns_empty(self, _with_token_key):
        assert _with_token_key.encrypt_token("") == ""

    def test_decrypt_empty_returns_empty(self, _with_token_key):
        assert _with_token_key.decrypt_token("") == ""

    def test_legacy_plaintext_fallback(self, _with_token_key):
        """When encryption is enabled but the stored value is plaintext
        (from before encryption was enabled), decrypt should return it as-is."""
        legacy_token = "plaintext-legacy-token-from-before"
        assert _with_token_key.decrypt_token(legacy_token) == legacy_token

    def test_legacy_plaintext_bytes_fallback(self, _with_token_key):
        legacy_token = b"plaintext-legacy-token-bytes"
        assert _with_token_key.decrypt_token(legacy_token) == "plaintext-legacy-token-bytes"


# --- Invalid key ---


class TestInvalidKey:
    def test_is_encryption_enabled_false(self, _with_bad_key):
        assert _with_bad_key.is_encryption_enabled() is False

    def test_encrypt_passthrough_on_bad_key(self, _with_bad_key):
        token = "my-oauth-token"
        assert _with_bad_key.encrypt_token(token) == token

    def test_decrypt_passthrough_on_bad_key(self, _with_bad_key):
        token = "my-oauth-token"
        assert _with_bad_key.decrypt_token(token) == token
