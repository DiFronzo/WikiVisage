"""The gallery excludes audio/video files without running a regex per row."""

import importlib
from pathlib import Path

import pytest

TITLES = {
    "File:Portrait.jpg": True,
    "File:Portrait.JPEG": True,
    "File:Interview.webm": False,
    "File:Interview.WEBM": False,
    "File:Song.ogg": False,
    "File:Song.oga": False,
    "File:Take.opus": False,
    "File:Theme.mid": False,
    "File:Voice.mp3": False,
    "File:Tone.wav": False,
    "File:Lossless.flac": False,
    "File:Clip.ogv": False,
    # The extension must be the final segment, not merely present.
    "File:Talk.ogg.jpg": True,
    "File:No extension": True,
    "File:webm": True,
}


def test_both_gallery_queries_share_the_filter_without_a_regex():
    app_module = importlib.import_module("app")
    source = Path(app_module.__file__).read_text(encoding="utf-8")

    assert "NOT REGEXP" not in source
    assert source.count("_EXCLUDE_NON_VISUAL_SQL}") == 2
    assert "'.', -1" in app_module._EXCLUDE_NON_VISUAL_SQL


@pytest.mark.integration
def test_filter_matches_the_former_regex_on_real_rows(db_conn, seed_project):
    app_module = importlib.import_module("app")
    regex = r"\.(webm|ogv|ogg|mp3|wav|flac|opus|mid|oga)$"

    with db_conn.cursor() as cur:
        for n, title in enumerate(TITLES, start=1):
            cur.execute(
                "INSERT INTO images (project_id, commons_page_id, file_title, status) VALUES (%s, %s, %s, 'processed')",
                (seed_project["id"], 700000 + n, title),
            )

        cur.execute(
            f"SELECT i.file_title FROM images i WHERE i.project_id = %s AND {app_module._EXCLUDE_NON_VISUAL_SQL}",
            (seed_project["id"],),
        )
        kept = {row["file_title"] for row in cur.fetchall()}

        cur.execute(
            "SELECT i.file_title FROM images i WHERE i.project_id = %s AND LOWER(i.file_title) NOT REGEXP %s",
            (seed_project["id"], regex),
        )
        by_regex = {row["file_title"] for row in cur.fetchall()}

    assert kept == by_regex
    assert kept == {title for title, visible in TITLES.items() if visible}
