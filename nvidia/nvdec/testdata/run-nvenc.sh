#!/usr/bin/env bash
# Run the H.264 NVENC -> CPU reference -> NVDEC physical cell.
set -euo pipefail
export GOWORK=off
module=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$HOME/tmp"
artifacts=${1:-$(mktemp -d "$HOME/tmp/nvenc-nvdec.XXXXXX")}
mkdir -p "$artifacts"
artifacts=$(cd "$artifacts" && pwd)
export TMPDIR="$artifacts"
cd "$module"

NVENC_ARTIFACT_DIR="$artifacts" go test -v -count=1 -tags nvenc_physical \
    ./nvenc -run '^TestPhysicalEncode$' 2>&1 | tee "$artifacts/nvenc.log"
cp "$artifacts/output.h264" "$artifacts/three.h264"
ffmpeg -version > "$artifacts/ffmpeg-version.txt"
ffmpeg -v error -y -hwaccel none -threads 1 -i "$artifacts/three.h264" \
    -pix_fmt nv12 -f rawvideo "$artifacts/three.h264.nv12"
# NVENC is lossless; require the independent CPU reference to equal its input.
cmp "$artifacts/input.nv12" "$artifacts/three.h264.nv12"
NVDEC_FIXTURE_DIR="$artifacts" go test -v -count=1 -tags nvdec_physical \
    ./nvdec -run '^TestPhysicalDecode$/^h264$' 2>&1 | tee "$artifacts/nvdec.log"
printf 'Artifacts: %s\n' "$artifacts"
