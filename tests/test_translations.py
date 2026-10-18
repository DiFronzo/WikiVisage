"""Freshness checks for the committed translation catalogs.

``translations/**/messages.mo`` are compiled artifacts committed to the
repository: Flask-Babel loads the ``.mo`` files at runtime, and the Toolforge
buildpacks only copy the repo, so nothing recompiles them during a build or
deploy. A ``.po`` change that was never recompiled would therefore ship stale
translations silently.

Re-sync the catalogs after every ``.po`` change with::

    pybabel compile -d translations
"""

import gettext
from pathlib import Path

import pytest
from babel.messages.catalog import Message
from babel.messages.pofile import read_po

REPO_ROOT = Path(__file__).resolve().parent.parent
TRANSLATIONS_DIR = REPO_ROOT / "translations"


def _compiled_forms(compiled: gettext.GNUTranslations, message: Message) -> list[str | None]:
    """The compiled translation forms of a message, in plural-index order."""
    if not message.pluralizable:
        if message.context:
            return [compiled.pgettext(message.context, message.id)]
        return [compiled.gettext(message.id)]

    singular, plural = message.id
    forms: list[str | None] = []
    for index in range(len(message.string)):
        # Select the form through the catalog's own plural rule — for an
        # "(n != 1)" locale form 0 is reached with n=1, so form index and n
        # are not interchangeable.
        n = next((candidate for candidate in range(50) if compiled.plural(candidate) == index), None)
        if n is None:
            forms.append(None)
        elif message.context:
            forms.append(compiled.npgettext(message.context, singular, plural, n))
        else:
            forms.append(compiled.ngettext(singular, plural, n))
    return forms


def _stale_entries(po_path: Path) -> list[str]:
    """Return the entries whose compiled form does not match the .po source."""
    mo_path = po_path.with_suffix(".mo")
    if not mo_path.is_file():
        return [f"{mo_path.name} is missing"]

    with po_path.open("rb") as fh:
        catalog = read_po(fh)
    with mo_path.open("rb") as fh:
        compiled = gettext.GNUTranslations(fh)

    stale: list[str] = []
    # Obsolete entries live in catalog.obsolete, not in the iteration itself.
    # Fuzzy entries are skipped exactly like ``pybabel compile`` skips them.
    for message in catalog:
        if not message.id or message.fuzzy:
            continue
        source_ids = (message.id,) if isinstance(message.id, str) else message.id
        expected_forms = message.string if message.pluralizable else (message.string,)
        actual_forms = _compiled_forms(compiled, message)
        for n, expected in enumerate(expected_forms):
            actual = actual_forms[n] if n < len(actual_forms) else None
            label = f"{message.id!r} form {n}" if message.pluralizable else repr(message.id)
            if expected:
                if actual != expected:
                    shown = actual if actual is not None else "(missing)"
                    stale.append(f"{label}: .po has {expected!r}, .mo has {shown}")
            elif actual is not None and actual not in ("", *source_ids):
                stale.append(f"{label}: untranslated in the .po, but the .mo still has {actual!r}")
    return stale


@pytest.mark.parametrize(
    "po_path",
    sorted(TRANSLATIONS_DIR.glob("*/LC_MESSAGES/messages.po")),
    ids=lambda path: path.parent.parent.name,
)
def test_compiled_catalog_matches_its_po_source(po_path: Path) -> None:
    """The committed .mo must serve exactly what its .po declares."""
    stale = _stale_entries(po_path)
    assert not stale, (
        f"{po_path.relative_to(REPO_ROOT)} is out of date: {len(stale)} stale entries. "
        f"Run `pybabel compile -d translations`. First mismatch: {stale[0]}"
    )
