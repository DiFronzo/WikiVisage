#!/usr/bin/env python3
"""Build a KServe V1 request body from local image files."""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="+", type=Path, help="Image files to encode into the request.")
    parser.add_argument(
        "--task",
        choices=("detect", "encode"),
        default="detect",
        help="'detect' finds faces; 'encode' embeds a known box (default: detect).",
    )
    parser.add_argument(
        "--box",
        action="append",
        default=[],
        metavar="TOP,RIGHT,BOTTOM,LEFT",
        help="Bounding box for --task encode. Repeat for multiple boxes.",
    )
    args = parser.parse_args()

    if args.task == "encode" and not args.box:
        parser.error("--task encode requires at least one --box TOP,RIGHT,BOTTOM,LEFT")

    boxes = []
    for raw in args.box:
        try:
            top, right, bottom, left = (int(part) for part in raw.split(","))
        except ValueError:
            parser.error(f"invalid --box {raw!r}; expected four integers TOP,RIGHT,BOTTOM,LEFT")
        boxes.append({"top": top, "right": right, "bottom": bottom, "left": left})

    instances = []
    for index, path in enumerate(args.images):
        if not path.is_file():
            print(f"error: no such file: {path}", file=sys.stderr)
            return 1
        instance = {
            "id": str(index),
            "task": args.task,
            "image": base64.b64encode(path.read_bytes()).decode("ascii"),
        }
        if args.task == "encode":
            instance["boxes"] = boxes
        instances.append(instance)

    json.dump({"instances": instances}, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
