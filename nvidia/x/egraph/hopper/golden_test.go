package hopper

import (
	"encoding/json"
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

// TestGoldensCompile compiles every golden translation unit for sm_90a.
//
// With CUDACOMPILECHECK set to a cudacompilecheck binary
// (github.com/tmc/lib/nvidia/x/cmd/cudacompilecheck) it compiles through
// that, and fails unless ptxas reports at least one entry function. To
// compile on another host, such as a Lima VM with the toolkit, also set
// CUDACOMPILECHECK_EXEC to its -exec prefix and CUDACOMPILECHECK_NVCC to
// nvcc's path there:
//
//	CUDACOMPILECHECK=$(go env GOPATH)/bin/cudacompilecheck \
//	CUDACOMPILECHECK_EXEC='limactl shell --workdir / okcnvcc --' \
//	CUDACOMPILECHECK_NVCC=/home/me.linux/venv/lib/python3.12/site-packages/nvidia/cu13/bin/nvcc \
//	go test -run GoldensCompile
//
// Otherwise it runs nvcc from PATH, and skips if there is none.
func TestGoldensCompile(t *testing.T) {
	files, err := filepath.Glob(filepath.Join("testdata", "golden", "*.cu"))
	if err != nil {
		t.Fatal(err)
	}
	if check := os.Getenv("CUDACOMPILECHECK"); check != "" {
		args := []string{"-arch", "sm_90a", "-std", "c++20", "-flags", "-O3", "-json"}
		if x := os.Getenv("CUDACOMPILECHECK_EXEC"); x != "" {
			args = append(args, "-exec", x)
		}
		if nvcc := os.Getenv("CUDACOMPILECHECK_NVCC"); nvcc != "" {
			args = append(args, "-nvcc", nvcc)
		}
		for _, f := range files {
			t.Run(filepath.Base(f), func(t *testing.T) {
				out, err := exec.Command(check, append(args, f)...).Output()
				if err != nil {
					var stderr []byte
					if ee, ok := err.(*exec.ExitError); ok {
						stderr = ee.Stderr
					}
					t.Fatalf("cudacompilecheck %s: %v\n%s%s", f, err, out, stderr)
				}
				var reports []struct{ Functions []struct{ Entry bool } }
				if err := json.Unmarshal(out, &reports); err != nil {
					t.Fatalf("cudacompilecheck %s: %v\n%s", f, err, out)
				}
				entries := 0
				for _, r := range reports {
					for _, fn := range r.Functions {
						if fn.Entry {
							entries++
						}
					}
				}
				if entries == 0 {
					t.Errorf("%s: ptxas reported no entry functions", f)
				}
			})
		}
		return
	}
	nvcc, err := exec.LookPath("nvcc")
	if err != nil {
		t.Skip("nvcc not on PATH and CUDACOMPILECHECK not set; goldens not compiled")
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
