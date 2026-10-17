#!/usr/bin/env python3
"""Offline readout for the experimental LiftWing metadata pre-filter (Phase 2).

Reads the shadow verdicts the worker stored on ``images`` (``prefilter_keep``,
``prefilter_reason``, ``prefilter_checked_at``) and compares them with what the
face pipeline actually found, so the experiment can be evaluated before an
``enforce`` mode is ever built:

* skip rate — share of files the model wanted to drop;
* recall of keep — among checked images that really contain a target face
  (human classifications first, then model verdicts for context), the share the
  model would have kept. Anything below 100% is an image an ``enforce`` mode
  would have thrown away despite it containing the target;
* detector signal — among processed images, drops that nonetheless had a
  detected face (should be rare; a smell for over-filtering) and keeps that had
  no detected face at all (expected to dominate: most category files show
  nobody);
* reason samples for the drops, so wrong verdicts are visible.

Bootstrapped images never appear here: they always bypass the pre-filter
(they already carry a P180 claim), so their verdict is never recorded.

Latency is not stored per file; the worker logs it per batch instead (search
the worker log for "Pre-filter").

Usage::

    python scripts/eval_prefilter.py [--project-id N] [--reasons]

Uses the same database environment variables as the worker
(``TOOL_TOOLSDB_USER``, ``TOOL_TOOLSDB_PASSWORD``, ``WIKIVISAGE_DB_NAME``, ...),
loaded from ``.env`` in the repo root when present, exactly like the worker.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402  (path set above)

load_dotenv()  # same .env loading as worker.py/app.py

from database import execute_query, init_db  # noqa: E402  (path set above)

BASE_SQL = """
    SELECT
        i.id AS image_id,
        i.file_title,
        i.status,
        i.prefilter_keep,
        i.prefilter_reason,
        COALESCE(SUM(CASE WHEN f.is_target = 1 AND f.classified_by_user_id IS NOT NULL THEN 1 ELSE 0 END), 0)
            AS human_targets,
        COALESCE(SUM(CASE WHEN f.is_target = 1 AND f.classified_by = 'model' THEN 1 ELSE 0 END), 0)
            AS model_targets,
        COUNT(f.id) AS faces
    FROM images i
    LEFT JOIN faces f ON f.image_id = i.id AND f.superseded_by IS NULL
    WHERE i.prefilter_checked_at IS NOT NULL
"""


def _fetch_rows(project_id: int | None) -> list[dict[str, Any]]:
    if project_id is None:
        return list(execute_query(BASE_SQL + " GROUP BY i.id ORDER BY i.id"))
    return list(
        execute_query(
            BASE_SQL + " AND i.project_id = %s GROUP BY i.id ORDER BY i.id",
            (project_id,),
        )
    )


def _share(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{numerator}/{denominator} ({100.0 * numerator / denominator:.1f}%)"


def _summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Count the readout metrics from raw rows (pure function, unit-testable)."""
    kept = [row for row in rows if row["prefilter_keep"] == 1]
    dropped = [row for row in rows if row["prefilter_keep"] == 0]
    untagged = [row for row in rows if row["prefilter_keep"] is None]
    processed = [row for row in rows if row["status"] == "processed"]

    def with_target(column: str) -> list[dict[str, Any]]:
        return [row for row in rows if row[column] > 0]

    human = with_target("human_targets")
    model = with_target("model_targets")
    # Bootstrap targets cannot occur: bootstrapped images always bypass the
    # pre-filter, so no checked row can carry one.
    any_target = [row for row in rows if row["human_targets"] or row["model_targets"]]

    # Face counts only exist for processed images; error/pending rows have none.
    dropped_processed = [row for row in processed if row["prefilter_keep"] == 0]
    kept_processed = [row for row in processed if row["prefilter_keep"] == 1]

    return {
        "checked": len(rows),
        "kept": len(kept),
        "dropped": len(dropped),
        "untagged": len(untagged),
        "human": human,
        "model": model,
        "any_target": any_target,
        "processed": len(processed),
        "dropped_with_faces": [row for row in dropped_processed if row["faces"] > 0],
        "kept_without_faces": [row for row in kept_processed if row["faces"] == 0],
        "dropped_processed": len(dropped_processed),
        "kept_processed": len(kept_processed),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project-id", type=int, default=None, help="restrict the readout to one project")
    parser.add_argument("--reasons", action="store_true", help="print sample reasons for the drop verdicts")
    args = parser.parse_args()

    init_db(pool_size=1)
    rows = _fetch_rows(args.project_id)

    if not rows:
        print("No images have a recorded LiftWing verdict yet.")
        print("Set a project's Metadata Pre-Filter to 'shadow' and let the worker run a cycle.")
        return 0

    stats = _summarise(rows)
    human, model, any_target = stats["human"], stats["model"], stats["any_target"]

    print(f"Checked images: {stats['checked']}")
    print(f"  keep=1:  {stats['kept']}")
    print(f"  keep=0:  {stats['dropped']}  (the model's skip rate: {_share(stats['dropped'], stats['checked'])})")
    if stats["untagged"]:
        print(f"  keep=NULL: {stats['untagged']} (unexpected: a checked image should carry a verdict)")
    print()
    print("Recall of keep, over checked images that really contain a target face")
    print(f"  human-confirmed target:   {_share(sum(1 for r in human if r['prefilter_keep'] == 1), len(human))}")
    print(f"  model-classified target:  {_share(sum(1 for r in model if r['prefilter_keep'] == 1), len(model))}")
    print(
        f"  any target:               {_share(sum(1 for r in any_target if r['prefilter_keep'] == 1), len(any_target))}"
    )
    print()
    print("Detector signal (processed images only)")
    print(
        "  dropped images with at least one detected face: "
        f"{_share(len(stats['dropped_with_faces']), stats['dropped_processed'])}"
    )
    print(
        "  kept images with no detected face:              "
        f"{_share(len(stats['kept_without_faces']), stats['kept_processed'])}"
    )

    missed = [row for row in human if row["prefilter_keep"] == 0]
    if missed:
        print()
        print(f"Human-confirmed target images the filter wanted to drop ({len(missed)}):")
        for row in missed[:20]:
            reason = row["prefilter_reason"] or "no reason recorded"
            print(f"  - {row['file_title']}: {reason}")
        if len(missed) > 20:
            print(f"  ... and {len(missed) - 20} more")

    dropped = [row for row in rows if row["prefilter_keep"] == 0]
    if args.reasons and dropped:
        print()
        print(f"Sample drop reasons (of {len(dropped)}):")
        for row in dropped[:20]:
            reason = row["prefilter_reason"] or "no reason recorded"
            print(f"  - {row['file_title']}: {reason}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
