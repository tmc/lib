package cudakernel_test

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudakernel"
)

func TestArtifactExactCUBINEntrySymbol(t *testing.T) {
	// This existing fixture is NVCC 13.4.59 output for sm_90. Its README
	// records compiler/disassembler-only provenance, not GPU execution.
	encoded, err := os.ReadFile("../cudacapture/testdata/kernels.cubin.hex")
	if err != nil {
		t.Fatal(err)
	}
	if got := fmt.Sprintf("%x", sha256.Sum256(encoded)); got != "825fc635f5ed57052e8f76d563862114735a0534bfc19c3698069391bae36f12" {
		t.Fatalf("public CUBIN fixture changed: %s", got)
	}
	raw, err := hex.DecodeString(strings.TrimSpace(string(encoded)))
	if err != nil {
		t.Fatal(err)
	}
	if got := fmt.Sprintf("%x", sha256.Sum256(raw)); got != "0d0f044503d4b56333c0263409098603a84a58ac36ed82af2cbd5354111b6a90" {
		t.Fatalf("decoded public CUBIN fixture changed: %s", got)
	}
	info, err := cudakernel.ParseCUBIN(raw)
	if err != nil {
		t.Fatalf("public fixture setup: %v", err)
	}
	for _, name := range []string{"_Z5scalePfPKff", "fill"} {
		if !slices.Contains(info.Symbols, name) {
			t.Fatalf("public fixture setup: missing exact symbol %q", name)
		}
	}
	for _, name := range []string{"scale", "fil"} {
		if slices.Contains(info.Symbols, name) {
			t.Fatalf("public fixture setup: fragment %q is an exact symbol", name)
		}
	}
	for _, tt := range []struct {
		name     string
		exact    string
		fragment string
		params   []cudakernel.Param
	}{
		{
			name: "C++ linkage", exact: "_Z5scalePfPKff", fragment: "scale",
			params: []cudakernel.Param{
				{Name: "out", Type: "ptr", IsPointer: true},
				{Name: "in", Type: "ptr", IsPointer: true},
				{Name: "k", Type: "f32"},
			},
		},
		{
			name: "C linkage", exact: "fill", fragment: "fil",
			params: []cudakernel.Param{
				{Name: "out", Type: "ptr", IsPointer: true},
				{Name: "v", Type: "s32"},
			},
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			schema, err := cudakernel.NewSchema(tt.params...)
			if err != nil {
				t.Fatal(err)
			}
			a, err := cudakernel.NewArtifact(cudakernel.KindCUBIN, raw, tt.exact, schema, "sm_90")
			if err != nil {
				t.Fatalf("exact entry constructor: %v", err)
			}
			if a.EntryPoint() != tt.exact {
				t.Fatalf("stored entry = %q, want %q", a.EntryPoint(), tt.exact)
			}
			wire, err := json.Marshal(a)
			if err != nil {
				t.Fatal(err)
			}
			parsed, err := cudakernel.ParseArtifact(wire)
			if err != nil {
				t.Fatalf("exact entry roundtrip: %v", err)
			}
			if parsed.EntryPoint() != tt.exact {
				t.Errorf("roundtrip entry = %q, want %q", parsed.EntryPoint(), tt.exact)
			}
			if bad, err := cudakernel.NewArtifact(cudakernel.KindCUBIN, raw, tt.fragment, schema, "sm_90"); err == nil || bad != nil {
				t.Errorf("absent fragment constructor = %v, %v, want nil artifact and error", bad, err)
			}
			spec := a.Spec()
			spec.EntryPoint = tt.fragment
			if bad, err := cudakernel.NewArtifactFromSpec(spec); err == nil || bad != nil {
				t.Errorf("absent fragment spec constructor = %v, %v, want nil artifact and error", bad, err)
			}
			var fields map[string]json.RawMessage
			if err := json.Unmarshal(wire, &fields); err != nil {
				t.Fatal(err)
			}
			fields["entry_point"], err = json.Marshal(tt.fragment)
			if err != nil {
				t.Fatal(err)
			}
			wire, err = json.Marshal(fields)
			if err != nil {
				t.Fatal(err)
			}
			if bad, err := cudakernel.ParseArtifact(wire); err == nil || bad != nil {
				t.Errorf("absent fragment wire = %v, %v, want nil artifact and error", bad, err)
			}
		})
	}
}
