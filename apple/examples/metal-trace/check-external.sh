#!/bin/sh
set -eu

lib_apple_dir=$(CDPATH='' cd -- "$(dirname -- "$0")/../.." && pwd)

# Allow explicit override via argument or env var, defaulting to dynamically resolved module
if [ "${1:-}" != "" ]; then
    tmc_apple_dir="$1"
elif [ "${TMC_APPLE_DIR:-}" != "" ]; then
    tmc_apple_dir="$TMC_APPLE_DIR"
else
    # Dynamically resolve from the current workspace/go environment
    tmc_apple_dir=$(cd "$lib_apple_dir" && go list -m -f '{{.Dir}}' github.com/tmc/apple 2>/dev/null || true)
    if [ -z "$tmc_apple_dir" ] || [ ! -d "$tmc_apple_dir" ]; then
        # Check standard relative path
        if [ -d "$lib_apple_dir/../../../tmc/apple" ]; then
            tmc_apple_dir=$(cd "$lib_apple_dir/../../../tmc/apple" && pwd)
        fi
    fi
fi

if [ -z "$tmc_apple_dir" ] || [ ! -d "$tmc_apple_dir" ]; then
    echo "ERROR: could not resolve github.com/tmc/apple directory. Set TMC_APPLE_DIR." >&2
    exit 1
fi

tmp_root=$(mktemp -d "${HOME}/tmp/check-external-metal-XXXXXX")
trap 'rm -rf "$tmp_root"' EXIT

# Create a disposable isolated copy of lib/apple in tmp_root with an internal package for negative control
dep_copy="$tmp_root/lib-apple-dep"
mkdir -p "$dep_copy"
cp -R "$lib_apple_dir/" "$dep_copy/"

# Add internal package fixture to disposable copy only
mkdir -p "$dep_copy/internal/fixture"
cat > "$dep_copy/internal/fixture/fixture.go" <<'FIXTURE'
package fixture

const InternalToken = "disposable-internal-token"
FIXTURE

client_dir="$tmp_root/client"
mkdir -p "$client_dir"

# Copy example files and assets
cp "$lib_apple_dir/examples/metal-trace/"*.go "$client_dir/"
cp "$lib_apple_dir/examples/metal-trace/"*.metal "$client_dir/" 2>/dev/null || true

cd "$client_dir"
export GOWORK=off

cat > go.mod <<GOMOD
module example.com/metal-trace-client

go 1.27.0

require (
	github.com/google/pprof v0.0.0-20260906184651-6331bc6350fe
	github.com/tmc/lib/apple v0.0.0
)

replace github.com/tmc/lib/apple => $dep_copy
replace github.com/tmc/apple => $tmc_apple_dir
GOMOD

go mod tidy

# Negative control: verify that importing an internal package in disposable github.com/tmc/lib/apple fails
cat > bad_internal.go <<'BAD'
package main

import (
	"fmt"
	"github.com/tmc/lib/apple/internal/fixture"
)

func init() {
	fmt.Println(fixture.InternalToken)
}
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
go build -o "$tmp_root/metal-trace-client" .

echo "Running help check on built client..."
"$tmp_root/metal-trace-client" -h 2>&1 | grep -q "timing mode"

echo "Exercising metalcapture public API from external module..."
cat > reader_writer_test.go <<'TEST'
package main

import (
	"bytes"
	"testing"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/apple/x/metal/metalcapture"
)

func TestExternalCaptureRoundTrip(t *testing.T) {
	p := &profile.Profile{
		SampleType: []*profile.ValueType{{Type: "gpu_time", Unit: "nanoseconds"}},
	}
	cap := &metalcapture.Capture{
		Profile: p,
		Files: map[string][]byte{
			"kernel.metal": []byte("kernel void add() {}\n"),
		},
		Code: metalcapture.CodeFiles{
			Source: "kernel.metal",
		},
	}
	var buf bytes.Buffer
	if err := metalcapture.Write(&buf, cap); err != nil {
		t.Fatalf("Write: %v", err)
	}
	readBack, err := metalcapture.Read(&buf)
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	if readBack.Code.Source != "kernel.metal" {
		t.Fatalf("unexpected source: %q", readBack.Code.Source)
	}
}
TEST

go test -v .

echo "SUCCESS: check-external passed."

