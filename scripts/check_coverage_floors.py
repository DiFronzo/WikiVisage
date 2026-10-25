#!/usr/bin/env python3
"""Fail when a module's line coverage drops below its floor.

Usage: coverage json -o coverage.json && python scripts/check_coverage_floors.py coverage.json

The floors sit about two points under the last measured value, so ordinary churn
does not trip them but a real loss of tests does. They are a ratchet: raise a floor
when a refactor or new tests lift a file, never lower one to make CI pass.
worker.py is the weakest (its main loop and SDC writer are the least covered code
in the repository), so it is the file that should move most. It went from 70% to
91% when write_sdc_claims and main were split into testable pieces.
"""

from __future__ import annotations

import json
import sys

FLOORS = {
    "app.py": 94,
    "worker.py": 89,
    "database.py": 85,
    "face_client.py": 93,
    "liftwing_client.py": 94,
    "migrate.py": 74,
    "config.py": 100,
    "redis_lock.py": 100,
    "token_crypto.py": 100,
    "oauth_refresh.py": 98,
    "api_errors.py": 100,
}


def main(path: str) -> int:
    with open(path, encoding="utf-8") as handle:
        files = json.load(handle)["files"]

    failures = []
    for name, floor in sorted(FLOORS.items()):
        entry = files.get(name)
        if entry is None:
            failures.append(f"{name}: not in the coverage report (was it renamed?)")
            continue
        percent = entry["summary"]["percent_covered"]
        status = "ok  " if percent >= floor else "FAIL"
        print(f"{status} {name:<20} {percent:5.1f}%  (floor {floor}%)")
        if percent < floor:
            failures.append(f"{name}: {percent:.1f}% < {floor}%")

    if failures:
        print("\nCoverage fell below a floor:\n  " + "\n  ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1]))
