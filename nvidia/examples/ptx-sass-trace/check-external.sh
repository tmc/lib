#!/bin/sh
set -eu

nvidia_dir=$(CDPATH='' cd -- "$(dirname -- "$0")/../.." && pwd)
tmp_root=$(mktemp -d "${HOME}/tmp/check-external-XXXXXX")
trap 'rm -rf "$tmp_root"' EXIT

client_dir="$tmp_root/client"
mkdir -p "$client_dir"

# Copy example files
cp "$nvidia_dir/examples/ptx-sass-trace/"*.go "$client_dir/"
cp "$nvidia_dir/examples/ptx-sass-trace/"*.cu "$client_dir/" 2>/dev/null || true
cp "$nvidia_dir/examples/ptx-sass-trace/"bundle.README.md "$client_dir/" 2>/dev/null || true

cd "$client_dir"
export GOWORK=off

cat > go.mod <<GOMOD
module example.com/ptx-sass-client

go 1.26.2

require github.com/tmc/lib/nvidia v0.0.0

replace github.com/tmc/lib/nvidia => $nvidia_dir
GOMOD

go mod tidy

# Negative control: verify that importing internal/profilebundle fails
cat > bad_internal.go <<'BAD'
package main

import _ "github.com/tmc/lib/nvidia/internal/profilebundle"
BAD

echo "Running negative control (should fail to import internal package)..."
if go build -o "$tmp_root/bad" . 2>"$tmp_root/bad.err"; then
    echo "ERROR: build succeeded when importing internal package!" >&2
    exit 1
fi

if grep -q "use of internal package .* not allowed" "$tmp_root/bad.err"; then
    echo "Expected failure observed: $(cat "$tmp_root/bad.err")"
else
    echo "ERROR: unexpected failure message: $(cat "$tmp_root/bad.err")" >&2
    exit 1
fi

# Remove bad internal import and build cleanly
rm bad_internal.go
echo "Building external client..."
go build -o "$tmp_root/ptx-sass-client" .

echo "Running help check on built client..."
"$tmp_root/ptx-sass-client" -h 2>&1 | grep -q "Usage: ptx-sass-trace" || "$tmp_root/ptx-sass-client" -h 2>&1 | grep -q "output directory"

echo "SUCCESS: check-external passed."
