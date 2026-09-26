package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"os"
	"path/filepath"
	"testing"
)

var update = flag.Bool("update", false, "rewrite testdata/compare golden files")

// fusedPair is a synthetic base and target in which the target fuses two
// backward kernels into one, renames a forward kernel that runs twice as
// often, and leaves a GEMM alone.
func fusedPair() (base, target []rec) {
	for i := range uint64(4) {
		at := 1000 + 100*i
		base = append(base,
			rec{subtype: "kernel", name: "LayerNorm", start: at, end: at + 20},
			rec{subtype: "kernel", name: "gemm", start: at + 20, end: at + 50},
			rec{subtype: "kernel", name: "LayerNormBackward", start: at + 50, end: at + 70},
			rec{subtype: "kernel", name: "LayerNormParamPartial", start: at + 70, end: at + 82},
			rec{subtype: "kernel", name: "LayerNorm", start: at + 82, end: at + 102},
		)
		target = append(target,
			rec{subtype: "kernel", name: "LayerNormF32", start: at, end: at + 8},
			rec{subtype: "kernel", name: "gemm", start: at + 20, end: at + 50},
			rec{subtype: "kernel", name: "LayerNormBackwardParam", start: at + 50, end: at + 76},
			rec{subtype: "kernel", name: "LayerNormF32", start: at + 82, end: at + 92},
		)
	}
	return base, target
}

// TestCompareGolden checks the text and JSON output of compare on
// fusedPair, with and without an ops file. Run with -update to rewrite
// the golden files.
func TestCompareGolden(t *testing.T) {
	tests := []struct {
		name string
		ops  string
	}{
		{"fused-kernels", ""},
		{"fused-ops", "gemm gemm\n[Ll]ayer_?[Nn]orm layernorm\n"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var rules opRules
			if tt.ops != "" {
				rules = mustRules(t, tt.ops)
			}
			base, target := fusedPair()
			c, err := compareCaptures(compareCapture(base, phaseStatus(1000, 2000)), compareCapture(target, phaseStatus(1000, 2000)), rules, nil, compareOptions{phase: -1})
			if err != nil {
				t.Fatal(err)
			}
			c.Cudaprof = buildStamp{}
			var text bytes.Buffer
			if err := c.write(&text); err != nil {
				t.Fatal(err)
			}
			js, err := json.MarshalIndent(c, "", "\t")
			if err != nil {
				t.Fatal(err)
			}
			checkGolden(t, filepath.Join("testdata", "compare", tt.name+".txt"), text.Bytes())
			checkGolden(t, filepath.Join("testdata", "compare", tt.name+".json"), append(js, '\n'))
		})
	}
}

func checkGolden(t *testing.T, path string, got []byte) {
	t.Helper()
	if *update {
		if err := os.MkdirAll(filepath.Dir(path), 0o777); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, got, 0o666); err != nil {
			t.Fatal(err)
		}
		return
	}
	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, want) {
		t.Errorf("%s differs; rerun with -update and inspect the diff\ngot:\n%s", path, got)
	}
}
