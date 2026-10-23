package hopper

import (
	"flag"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

var update = flag.Bool("update", false, "rewrite testdata/golden files")

// checkGolden compares got with testdata/golden/name, or rewrites it with -update.
// The goldens are the CUDA that nvcc compile checks (TestGoldensCompile) read.
func checkGolden(t *testing.T, name, got string) {
	t.Helper()
	path := filepath.Join("testdata", "golden", name)
	if *update {
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(got), 0o644); err != nil {
			t.Fatal(err)
		}
		return
	}
	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("%v (run go test -update)", err)
	}
	if string(want) != got {
		t.Errorf("%s differs from emitted source (run go test -update to accept)", path)
	}
}

func relL2(got, want []float32) float64 {
	var num, den float64
	for i := range want {
		d := float64(got[i]) - float64(want[i])
		num += d * d
		den += float64(want[i]) * float64(want[i])
	}
	if den == 0 {
		return math.Sqrt(num)
	}
	return math.Sqrt(num / den)
}

// closeTo reports whether every got[i] is within rel·|want[i]| + abs of want[i].
func closeTo(got, want []float32, rel, abs float64) (int, bool) {
	for i := range want {
		if d := math.Abs(float64(got[i]) - float64(want[i])); d > rel*math.Abs(float64(want[i]))+abs || math.IsNaN(float64(got[i])) {
			return i, false
		}
	}
	return 0, true
}

// TestGoldensCompile compiles every golden translation unit for sm_90a. It
// needs nvcc on PATH (not available on the macOS development host), so it is
// the check to run on a CUDA machine: go test -run GoldensCompile.
func TestGoldensCompile(t *testing.T) {
	nvcc, err := exec.LookPath("nvcc")
	if err != nil {
		t.Skip("nvcc not on PATH; goldens not compiled")
	}
	files, err := filepath.Glob(filepath.Join("testdata", "golden", "*.cu"))
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range files {
		t.Run(filepath.Base(f), func(t *testing.T) {
			obj := filepath.Join(t.TempDir(), "k.o")
			out, err := exec.Command(nvcc, "-std=c++20", "-arch=sm_90a", "-O3", "-c", "-o", obj, f).CombinedOutput()
			if err != nil {
				t.Fatalf("nvcc %s: %v\n%s", f, err, out)
			}
		})
	}
}
