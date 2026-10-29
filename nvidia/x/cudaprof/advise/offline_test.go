package advise

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudacapture"
)

// codeCapture assembles a compile-only capture from the testdata files of
// one kernel: its SASS, its ptxas -v report and, where present, its PTX.
func codeCapture(t *testing.T, stem string) *cudacapture.Capture {
	t.Helper()
	cap := &cudacapture.Capture{Files: map[string][]byte{}}
	for _, ext := range []string{".sass", ".ptxas.txt", ".ptx"} {
		data, err := os.ReadFile(filepath.Join("testdata", "tensor", stem+ext))
		if os.IsNotExist(err) {
			continue
		}
		if err != nil {
			t.Fatal(err)
		}
		name := stem + ext
		cap.Files[name] = data
		switch ext {
		case ".sass":
			cap.Code.SASS = name
		case ".ptx":
			cap.Code.PTX = name
		}
	}
	return cap
}

// TestCompileOnly runs advise on SASS and ptxas reports compiled for each
// target with no device, as a generator planning for GB300 must.
func TestCompileOnly(t *testing.T) {
	tests := []struct {
		stem      string
		arch      string
		rule      string // the tensor-path rule that must fire, or ""
		occupancy string // the limit the occupancy finding names, or ""
	}{
		{"gemm.sm_80", "sm_80", "", "registers"},
		{"gemm.sm_90a", "sm_90a", "warp-mma-wgmma", "registers"},
		{"gemm.sm_100a", "sm_100a", "warp-mma-tcgen05", "registers"},
		{"gemm.sm_103a", "sm_103a", "warp-mma-tcgen05", "registers"},
		{"gemm.sm_110a", "sm_110a", "warp-mma-tcgen05", "registers"},
		{"gemm.sm_120a", "sm_120a", "", "registers"},
		{"gemm.sm_121a", "sm_121a", "", "registers"},
		{"nvfp4.sm_103a", "sm_103a", "fp4-dequant", "registers"},
		{"nvfp4.sm_120a", "sm_120a", "fp4-dequant", "registers"},
		{"umma_bf16.sm_100a", "sm_100a", "", ""},
		{"umma_nvfp4.sm_103a", "sm_103a", "", "tensor_memory"},
	}
	all := map[string]bool{"warp-mma-wgmma": true, "warp-mma-tcgen05": true, "fp4-dequant": true}
	for _, tt := range tests {
		t.Run(tt.stem, func(t *testing.T) {
			cap := codeCapture(t, tt.stem)
			if got := parseArchFromCapture(cap); got != tt.arch {
				t.Fatalf("arch = %q, want %q", got, tt.arch)
			}
			report, err := Analyze(cap)
			if err != nil {
				t.Fatal(err)
			}
			if !report.CompileOnly || report.SampleOnly {
				t.Errorf("CompileOnly %v SampleOnly %v, want true, false", report.CompileOnly, report.SampleOnly)
			}
			if !strings.Contains(report.Markdown(), "compiled code only") {
				t.Error("markdown does not say the capture is compiled code only")
			}
			got := map[string]bool{}
			occupancy := ""
			var paths int
			walk(report.Items, func(it Item) {
				got[it.rule] = true
				if it.rule == "tensor-path" {
					paths++
				}
				if it.rule == "occupancy" {
					occupancy = it.Finding
					if !strings.Contains(it.Finding, "on "+strings.TrimSuffix(tt.arch, "a")) || cap.Code.PTX == "" && !strings.Contains(it.Finding, "128 threads assumed") {
						t.Errorf("occupancy finding names another architecture: %s", it.Finding)
					}
				}
			})
			for rule := range all {
				if got[rule] != (rule == tt.rule) {
					t.Errorf("rule %s fired %v, want %v", rule, got[rule], rule == tt.rule)
				}
			}
			if paths != 1 {
				t.Errorf("%d tensor-path findings, want 1", paths)
			}
			if tt.occupancy == "" && occupancy != "" || tt.occupancy != "" && !strings.Contains(occupancy, "constrained by "+tt.occupancy) {
				t.Errorf("occupancy finding = %q, want one constrained by %q", occupancy, tt.occupancy)
			}
		})
	}
}

// walk calls f on every item and every pattern nested in one.
func walk(items []Item, f func(Item)) {
	for _, it := range items {
		f(it)
		walk(it.Patterns, f)
	}
}
