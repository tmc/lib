package cudasass

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"reflect"
	"strings"
	"testing"
	"time"
)

func checkKnownJSON(got, want *JSONDisassembly) error {
	// Compare interpreted metadata so insignificant JSON whitespace is ignored.
	for _, key := range []string{"ELF", "SM", "SchemaVersion", "Producer"} {
		var g, w any
		if err := json.Unmarshal(got.Metadata[key], &g); err != nil {
			return fmt.Errorf("metadata %s: %w", key, err)
		}
		if err := json.Unmarshal(want.Metadata[key], &w); err != nil {
			return fmt.Errorf("reference metadata %s: %w", key, err)
		}
		if !reflect.DeepEqual(g, w) {
			return fmt.Errorf("metadata %s = %v, want %v", key, g, w)
		}
	}
	if len(got.Functions) != len(want.Functions) {
		return fmt.Errorf("function count = %d, want %d", len(got.Functions), len(want.Functions))
	}
	for i := range want.Functions {
		if !reflect.DeepEqual(got.Functions[i], want.Functions[i]) {
			return fmt.Errorf("function %d differs: got %+v, want %+v", i, got.Functions[i], want.Functions[i])
		}
	}
	return nil
}

func knownJSONReference(t *testing.T) ([]byte, *JSONDisassembly) {
	t.Helper()
	b, err := os.ReadFile("testdata/nvdisasm-13.0-sm121.json")
	if err != nil {
		t.Fatal(err)
	}
	d, err := ParseJSON(bytes.NewReader(b))
	if err != nil {
		t.Fatal(err)
	}
	return b, d
}

// The negative controls remain valid JSON with a valid nvdisasm schema. They
// must fail the artifact comparison even though ParseJSON accepts them.
func TestKnownJSONSemanticChanges(t *testing.T) {
	b, want := knownJSONReference(t)
	for _, tt := range []struct{ name, old, replacement string }{
		{"unchanged", "", ""},
		{"architecture", `"major":12,"minor":1`, `"major":12,"minor":0`},
		{"function", `"function-name":"addone"`, `"function-name":"other"`},
		{"length", `"length":384`, `"length":368`},
		{"opcode", `"opcode":"FADD"`, `"opcode":"FMUL"`},
		{"operand", `"operands":"R5,R0,1"`, `"operands":"R5,R0,2"`},
		{"control flow", `"control-flow":"True"`, `"control-flow":"False"`},
		{"instruction count", `,{"opcode":"NOP"}`, ""},
	} {
		t.Run(tt.name, func(t *testing.T) {
			s := string(b)
			if tt.old != "" {
				if !strings.Contains(s, tt.old) {
					t.Fatal("mutation did not match the reference")
				}
				s = strings.Replace(s, tt.old, tt.replacement, 1)
			}
			got, err := ParseJSON(strings.NewReader(s))
			if err != nil {
				t.Fatalf("mutation should remain parseable: %v", err)
			}
			err = checkKnownJSON(got, want)
			if (err == nil) != (tt.name == "unchanged") {
				t.Fatalf("semantic comparison = %v", err)
			}
		})
	}
}

// TestInstalledKnownJSON compares the pinned artifact's complete function
// records, including operands, control-flow attributes and padding slots.
// This is binary inspection, not evidence that the kernel executes correctly.
func TestInstalledKnownJSON(t *testing.T) {
	p := os.Getenv("CUDASASS_TEST_KNOWN_CUBIN")
	if p == "" {
		t.Skip("set CUDASASS_TEST_KNOWN_CUBIN to the retained sm121 sass.cubin")
	}
	b, err := os.ReadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	const wantHash = "cc5746145dc7f00f91c458925f3513a4e4c91ff5a252be744272501055d41027"
	if got := fmt.Sprintf("%x", sha256.Sum256(b)); got != wantHash {
		t.Fatalf("CUBIN sha256 = %s, want pinned artifact %s", got, wantHash)
	}
	_, want := knownJSONReference(t)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	scratch := t.TempDir()
	d := Disassembler{Path: os.Getenv("NVDISASM"), TempDir: scratch}
	version, err := d.Version(ctx)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("input=%s sha256=%s tool=%s version=%s%s", p, wantHash, version.Path, version.Stdout, version.Stderr)
	r, err := d.Disassemble(ctx, b, Options{Mode: JSON})
	t.Logf("args=%q stderr=%s stdout_sha256=%x", r.Args, r.Stderr, sha256.Sum256(r.Stdout))
	if err != nil {
		t.Fatal(err)
	}
	got, err := ParseJSON(bytes.NewReader(r.Stdout))
	if err != nil {
		t.Fatal(err)
	}
	if err := checkKnownJSON(got, want); err != nil {
		t.Fatal(err)
	}
	remaining, err := os.ReadDir(scratch)
	if err != nil || len(remaining) != 0 {
		t.Fatalf("scratch after disassembly: %v, %v", remaining, err)
	}
	after, err := os.ReadFile(p)
	if err != nil || !bytes.Equal(b, after) {
		t.Fatalf("input changed: %v", err)
	}
	t.Log("verified pinned sm121 metadata and addone function: 24 instruction slots, operands and control-flow attributes")
}
