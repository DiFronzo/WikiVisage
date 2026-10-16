#!/usr/bin/env bash
# Fail if the face service's model weights could be delivered by anything other
# than plain git blobs. Run in CI (lint job); see docs/plans/face-model-upgrade.md,
# section 1c.
#
#   scripts/check_no_lfs.sh [repo-root]
#
# The Toolforge build service does a plain clone. A Git LFS pointer file or an
# empty submodule directory would therefore build "successfully" and then fail
# (or worse, misbehave) at runtime. Three checks:
#   1. no .gitmodules and no submodule entries (mode 160000) in the index,
#   2. no .gitattributes rule routes any weight file through the lfs filter,
#   3. no weight file is an LFS pointer (starts with "version https://git-lfs").

set -euo pipefail

ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
status=0

fail() {
    echo "error: $*" >&2
    status=1
}

if [[ -e .gitmodules ]]; then
    fail ".gitmodules exists; the face service must not depend on submodules"
fi
if git rev-parse --git-dir >/dev/null 2>&1 && git ls-files --stage | grep -q '^160000 '; then
    fail "the index contains a submodule (mode 160000)"
fi

shopt -s nullglob
weights=(model-server/weights/*.onnx)
if [[ ${#weights[@]} -eq 0 ]]; then
    fail "no model-server/weights/*.onnx files found"
    exit 1
fi

for file in "${weights[@]}"; do
    if git rev-parse --git-dir >/dev/null 2>&1; then
        filter="$(git check-attr filter -- "$file" | awk '{print $NF}')"
        if [[ "$filter" == "lfs" ]]; then
            fail "$file is routed through Git LFS (.gitattributes filter=lfs)"
        fi
    fi
    if head -c 64 "$file" | grep -q 'version https://git-lfs'; then
        fail "$file is a Git LFS pointer file, not the model"
    fi
done

if [[ $status -eq 0 ]]; then
    echo "ok: ${#weights[@]} weight files are plain blobs; no LFS, no submodules"
fi
exit $status
