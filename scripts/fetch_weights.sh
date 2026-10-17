#!/usr/bin/env bash
# Re-create or verify model-server/weights/ from the official InsightFace release.
#
#   scripts/fetch_weights.sh           # download buffalo_sc.zip, verify, install
#   scripts/fetch_weights.sh --check   # verify the files already in place; no network
#
# Verification order (see docs/plans/face-model-upgrade.md, section 1c):
#   1. SHA-256 of the downloaded zip,
#   2. SHA-256 of both extracted files,
# and nothing is written to model-server/weights/ unless every check passes.
#
# Licence: these weights are for non-commercial research use only. Read
# model-server/weights/LICENSE-NOTICE before distributing them.

set -euo pipefail

ZIP_URL="https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip"
ZIP_SHA256="57d31b56b6ffa911c8a73cfc1707c73cab76efe7f13b675a05223bf42de47c72"
DET_SHA256="5e4447f50245bbd7966bd6c0fa52938c61474a04ec7def48753668a9d8b4ea3a"
REC_SHA256="9cc6e4a75f0e2bf0b1aed94578f144d15175f357bdc05e815e5c4a02b319eb4f"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/model-server/weights"

sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d' ' -f1
    else
        shasum -a 256 "$1" | cut -d' ' -f1
    fi
}

verify() {
    local file="$1" expected="$2" actual
    if [[ ! -f "$file" ]]; then
        echo "error: missing file: $file" >&2
        return 1
    fi
    actual="$(sha256_of "$file")"
    if [[ "$actual" != "$expected" ]]; then
        echo "error: SHA-256 mismatch for $file" >&2
        echo "  expected $expected" >&2
        echo "  actual   $actual" >&2
        return 1
    fi
    echo "ok  $expected  $(basename "$file")"
}

check_installed() {
    verify "$DEST/det_500m.onnx" "$DET_SHA256"
    verify "$DEST/w600k_mbf.onnx" "$REC_SHA256"
    # The committed manifest must agree with the pinned values above.
    local manifest
    manifest="$(tr -s ' *' ' ' <"$DEST/SHA256SUMS")"
    grep -qx "$DET_SHA256 det_500m.onnx" <<<"$manifest" || { echo "error: SHA256SUMS disagrees on det_500m.onnx" >&2; return 1; }
    grep -qx "$REC_SHA256 w600k_mbf.onnx" <<<"$manifest" || { echo "error: SHA256SUMS disagrees on w600k_mbf.onnx" >&2; return 1; }
    echo "ok  SHA256SUMS agrees with the pinned hashes"
}

if [[ "${1:-}" == "--check" ]]; then
    check_installed
    exit 0
fi

if [[ $# -gt 0 ]]; then
    echo "usage: $0 [--check]" >&2
    exit 2
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "downloading $ZIP_URL"
curl -fsSL --retry 3 -o "$WORK/buffalo_sc.zip" "$ZIP_URL"
verify "$WORK/buffalo_sc.zip" "$ZIP_SHA256"

mkdir "$WORK/extract"
unzip -q "$WORK/buffalo_sc.zip" det_500m.onnx w600k_mbf.onnx -d "$WORK/extract"
verify "$WORK/extract/det_500m.onnx" "$DET_SHA256"
verify "$WORK/extract/w600k_mbf.onnx" "$REC_SHA256"

mkdir -p "$DEST"
cp "$WORK/extract/det_500m.onnx" "$WORK/extract/w600k_mbf.onnx" "$DEST/"
printf '%s  det_500m.onnx\n%s  w600k_mbf.onnx\n' "$DET_SHA256" "$REC_SHA256" >"$DEST/SHA256SUMS"
echo "installed into $DEST"
