"""Unit tests for scripts/eval_prefilter.py's readout math.

The script is loaded by path (scripts/ is not a package). Importing it only
pulls in ``database`` and ``dotenv`` — no database connection is made until
``init_db`` runs inside ``main`` — so these tests need nothing else.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "eval_prefilter.py"
_spec = importlib.util.spec_from_file_location("eval_prefilter", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
eval_prefilter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval_prefilter)


def _row(**overrides):
    row = {
        "image_id": 1,
        "file_title": "File:Example.jpg",
        "status": "processed",
        "prefilter_keep": 1,
        "prefilter_reason": None,
        "human_targets": 0,
        "model_targets": 0,
        "faces": 0,
    }
    row.update(overrides)
    return row


def test_share_formats_and_handles_empty():
    assert eval_prefilter._share(0, 0) == "n/a"
    assert eval_prefilter._share(1, 4) == "1/4 (25.0%)"
    assert eval_prefilter._share(3, 3) == "3/3 (100.0%)"


def test_summarise_counts_verdicts_and_targets():
    rows = [
        _row(image_id=1, prefilter_keep=1, human_targets=1, faces=1),
        _row(image_id=2, prefilter_keep=0, human_targets=1, faces=2),  # human-confirmed, dropped
        _row(image_id=3, prefilter_keep=0, model_targets=1, faces=1),  # model target, dropped
        _row(image_id=4, prefilter_keep=None, status="error"),  # untagged, never processed
    ]

    stats = eval_prefilter._summarise(rows)

    assert stats["checked"] == 4
    assert stats["kept"] == 1
    assert stats["dropped"] == 2
    assert stats["untagged"] == 1
    assert len(stats["human"]) == 2
    assert len(stats["model"]) == 1
    # any_target is human OR model; a checked image can never be a bootstrap target.
    assert len(stats["any_target"]) == 3


def test_summarise_detector_signal_uses_processed_images_only():
    rows = [
        _row(image_id=1, prefilter_keep=0, status="processed", faces=1),  # drop, but a face was found
        _row(image_id=2, prefilter_keep=0, status="processed", faces=0),  # drop, nothing found
        _row(image_id=3, prefilter_keep=1, status="processed", faces=0),  # keep, nothing found
        _row(image_id=4, prefilter_keep=1, status="processed", faces=3),  # keep with faces
        _row(image_id=5, prefilter_keep=0, status="error", faces=0),  # unprocessed: no face data
    ]

    stats = eval_prefilter._summarise(rows)

    assert stats["processed"] == 4
    assert [row["image_id"] for row in stats["dropped_with_faces"]] == [1]
    assert [row["image_id"] for row in stats["kept_without_faces"]] == [3]
    assert stats["dropped_processed"] == 2
    assert stats["kept_processed"] == 2
