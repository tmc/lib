package cublasltx_test

import (
	"slices"
	"strings"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/cublaslt"
	"github.com/tmc/lib/nvidia/x/cublasltx"
	"github.com/tmc/lib/nvidia/x/cudatest"
)

func TestAlgoValidation(t *testing.T) {
	if _, err := cublasltx.Heuristic(0, 0, 0, 0, 0, 0, 0, 1); err == nil || !strings.Contains(err.Error(), "nil handle") {
		t.Errorf("Heuristic with nil handle: error = %v, want nil handle", err)
	}
	if _, err := cublasltx.Heuristic(1, 0, 0, 0, 0, 0, 0, 0); err == nil || !strings.Contains(err.Error(), "out of range") {
		t.Errorf("Heuristic with no candidates: error = %v, want out of range", err)
	}
	if _, err := cublasltx.AlgoConfigOf(nil); err == nil || !strings.Contains(err.Error(), "nil algorithm") {
		t.Errorf("AlgoConfigOf(nil): error = %v, want nil algorithm", err)
	}
	if _, err := cublasltx.Pick(nil, nil); err == nil || !strings.Contains(err.Error(), "no candidate") {
		t.Errorf("Pick(nil): error = %v, want no candidate", err)
	}
}

// TestHeuristicPick asks for the candidates of a BF16 product and checks
// that they, and so a Pick over them, do not change between calls.
func TestHeuristicPick(t *testing.T) {
	dev := cudatest.Require(t)
	major, minor, patch, err := cublasltx.Version()
	if err != nil {
		t.Fatalf("Version: %v", err)
	}
	if major <= 0 {
		t.Errorf("Version = %d.%d.%d, want major > 0", major, minor, patch)
	}

	const m, n, k = 1024, 512, 768
	var configs [2][]cublasltx.AlgoConfig
	var cands []cublaslt.MatmulHeuristicResult
	err = dev.WithContext(func() error {
		h, err := cublaslt.Create()
		if err != nil {
			t.Fatalf("Create: %v", err)
		}
		defer h.Close()
		desc, err := cublaslt.NewMatmulDesc(cublaslt.Compute32F, cublaslt.R32F)
		if err != nil {
			t.Fatalf("NewMatmulDesc: %v", err)
		}
		defer desc.Close()
		layout := func(typ cublaslt.DataType, rows, cols int) cublaslt.MatrixLayout {
			l, err := cublaslt.NewMatrixLayout(typ, uint64(rows), uint64(cols), int64(rows))
			if err != nil {
				t.Fatalf("NewMatrixLayout: %v", err)
			}
			t.Cleanup(func() {
				if err := l.Close(); err != nil {
					t.Errorf("close layout: %v", err)
				}
			})
			return l
		}
		a := layout(cublaslt.R16BF, m, k)
		b := layout(cublaslt.R16BF, k, n)
		d := layout(cublaslt.R32F, m, n)
		pref, err := cublaslt.NewMatmulPreference()
		if err != nil {
			t.Fatalf("NewMatmulPreference: %v", err)
		}
		defer pref.Close()
		ws := uint64(32 << 20)
		if err := pref.SetAttribute(cublaslt.MatmulPrefMaxWorkspaceBytes, unsafe.Pointer(&ws), 8); err != nil {
			t.Fatalf("set workspace: %v", err)
		}
		for i := range configs {
			cands, err = cublasltx.Heuristic(h, desc, a, b, d, d, pref, 8)
			if err != nil {
				t.Fatalf("Heuristic: %v", err)
			}
			if len(cands) == 0 || len(cands) > 8 {
				t.Fatalf("Heuristic returned %d candidates, want 1 to 8", len(cands))
			}
			for _, r := range cands {
				c, err := cublasltx.AlgoConfigOf(&r.Algo)
				if err != nil {
					t.Fatalf("AlgoConfigOf: %v", err)
				}
				configs[i] = append(configs[i], c)
			}
		}
		return nil
	})
	if err != nil {
		t.Fatalf("WithContext: %v", err)
	}
	if !slices.Equal(configs[0], configs[1]) {
		t.Fatalf("candidates changed between calls:\n%v\n%v", configs[0], configs[1])
	}
	t.Logf("cuBLASLt %d.%d.%d candidates: %+v", major, minor, patch, configs[0])

	if i, err := cublasltx.Pick(cands, nil); err != nil || i != 0 {
		t.Errorf("Pick(nil match) = %d, %v, want 0", i, err)
	}
	if i, err := cublasltx.Pick(cands, func(cublasltx.AlgoConfig) bool { return false }); err != nil || i != 0 {
		t.Errorf("Pick(no match) = %d, %v, want 0", i, err)
	}
	last := configs[0][len(configs[0])-1]
	want := slices.Index(configs[0], last)
	if i, err := cublasltx.Pick(cands, func(c cublasltx.AlgoConfig) bool { return c == last }); err != nil || i != want {
		t.Errorf("Pick(last config) = %d, %v, want %d", i, err, want)
	}
}
