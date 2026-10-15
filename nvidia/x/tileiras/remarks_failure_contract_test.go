package tileiras_test

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/tileiras"
)

// The script is a deterministic synthetic process, not an assembler. It writes
// arbitrary output bytes so the test reaches Assemble's production remark parser.
const remarkFixture = `#!/bin/sh
set -eu
output=
remarks=
previous=
for arg do
  if [ "$previous" = -o ]; then output=$arg; fi
  case "$arg" in
    --remarks-output-file=*) remarks=${arg#*=} ;;
  esac
  previous=$arg
done
test -n "$output"
test -n "$remarks"
printf '%s' 'synthetic output bytes; no executable code' > "$output"
printf '%s' "$TILEIRAS_TEST_REMARKS" > "$remarks"
printf '%s' "$TILEIRAS_TEST_STDOUT"
printf '%s' "$TILEIRAS_TEST_STDERR" >&2
`

func TestAssembleRemarkFailureContract(t *testing.T) {
	path := filepath.Join(t.TempDir(), "synthetic-tileiras")
	if err := os.WriteFile(path, []byte(remarkFixture), 0700); err != nil {
		t.Fatal(err)
	}
	const stdout = "synthetic stdout\n"
	const stderr = "synthetic stderr\n"
	t.Setenv("TILEIRAS_TEST_STDOUT", stdout)
	t.Setenv("TILEIRAS_TEST_STDERR", stderr)
	for _, tt := range []struct {
		name string
		raw  string
	}{
		{"missing kind", "---\nPass: Memory\noriginal malformed bytes\n"},
		{"invalid location", "--- !Passed\nDebugLoc: { File: 'original.cu'\n"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("TILEIRAS_TEST_REMARKS", tt.raw)
			result, err := tileiras.Assemble(context.Background(), []byte("synthetic input"), tileiras.WithPath(path), tileiras.WithRemarks("all"))
			if result != nil || !errors.Is(err, tileiras.ErrMalformedRemark) {
				t.Fatalf("malformed assembler remarks: result=%+v err=%v", result, err)
			}
			for _, raw := range []string{tt.raw, stdout, stderr} {
				if !strings.Contains(err.Error(), raw) {
					t.Errorf("error lost original diagnostic bytes %q: %v", raw, err)
				}
			}
		})
	}
}

func TestAssembleUnknownRemarkContract(t *testing.T) {
	path := filepath.Join(t.TempDir(), "synthetic-tileiras")
	if err := os.WriteFile(path, []byte(remarkFixture), 0700); err != nil {
		t.Fatal(err)
	}
	const raw = "--- !FutureKind\nPass: FuturePass\nName: FutureName\nFunction: fixture\nArgs:\n  - FutureKey: original value\n...\n"
	t.Setenv("TILEIRAS_TEST_REMARKS", raw)
	t.Setenv("TILEIRAS_TEST_STDOUT", "synthetic stdout\n")
	t.Setenv("TILEIRAS_TEST_STDERR", "synthetic stderr\n")
	result, err := tileiras.Assemble(context.Background(), []byte("synthetic input"), tileiras.WithPath(path), tileiras.WithRemarks("all"))
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Remarks) != 1 {
		t.Fatalf("remarks = %+v, want one unknown remark", result.Remarks)
	}
	r := result.Remarks[0]
	if r.Kind != "FutureKind" || r.Pass != "FuturePass" || r.Name != "FutureName" || r.Function != "fixture" || r.Args["FutureKey"] != "original value" {
		t.Fatalf("unknown remark lost: %+v", r)
	}
	if string(result.CUBIN) != "synthetic output bytes; no executable code" || result.Log != "synthetic stdout\nsynthetic stderr\n" {
		t.Fatalf("synthetic output or log changed: %+v", result)
	}
}
