package cudagroup

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/internal/cudatest"
)

func nativeFixture(t *testing.T) Fixture {
	t.Helper()
	warp, err := New(Warp, Dim3{X: 32, Y: 1, Z: 1})
	if err != nil {
		t.Fatal(err)
	}
	target := os.Getenv("CUDA_GROUP_NATIVE_ARCH")
	if target == "" {
		target = "sm_90"
	}
	f, err := GenerateReductionFixtureForTarget(warp, target)
	if err != nil {
		t.Fatal(err)
	}
	if err := f.ValidateLaunch(); err != nil {
		t.Fatal(err)
	}
	return f
}

func nativeDir(t *testing.T) string {
	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(home, "tmp"), 0700); err != nil {
		t.Fatal(err)
	}
	dir, err := os.MkdirTemp(filepath.Join(home, "tmp"), "cudagroup-native-")
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("retained native fixture: %s", dir)
	return dir
}

func requireNative(t *testing.T) {
	t.Helper()
	if os.Getenv("CUDA_GROUP_NATIVE") != "1" {
		t.Skip("native qualification disabled: set CUDA_GROUP_NATIVE=1")
	}
	// See cudashared/native_test.go: the lock exports the holder's pid, not "1".
	if os.Getenv("GPU_LOCK_HELD") == "" {
		t.Fatal("CUDA_GROUP_NATIVE=1 requires running under the GPU lock")
	}
}

func TestCompileReductionFixture(t *testing.T) {
	if os.Getenv("CUDA_GROUP_COMPILE") != "1" {
		t.Skip("compile qualification disabled: set CUDA_GROUP_COMPILE=1")
	}
	nvcc, _ := cudatest.NVCC(t)
	f := nativeFixture(t)
	dir := nativeDir(t)
	src := filepath.Join(dir, "fixture.cu")
	if err := os.WriteFile(src, []byte(f.Source), 0600); err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(nvcc, "-c", "-O0", "-arch="+f.Target, src, "-o", filepath.Join(dir, "fixture.o"))
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("nvcc compile: %v\n%s", err, out)
	}
}

func TestNativeGroupReference(t *testing.T) {
	requireNative(t)
	nvcc, _ := cudatest.NVCC(t)
	f := nativeFixture(t)
	dir := nativeDir(t)
	values := make([]float32, 32)
	for i := range values {
		values[i] = float32(i) - 7.5
	}
	want, err := ReduceSumReference(values)
	if err != nil {
		t.Fatal(err)
	}
	input := make([]string, len(values))
	for i, v := range values {
		input[i] = strconv.FormatFloat(float64(v), 'g', -1, 32)
	}
	host := fmt.Sprintf(`
int main() {
  cudaDeviceProp prop; int driver = 0, runtime = 0;
  if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess || cudaDriverGetVersion(&driver) != cudaSuccess || cudaRuntimeGetVersion(&runtime) != cudaSuccess) return 1;
  fprintf(stderr, "device=%%s sm_%%d%%d driver=%%d runtime=%%d\n", prop.name, prop.major, prop.minor, driver, runtime);
  float h[32] = {%s}; float *di = 0, *do_ = 0, out = 0;
  if (cudaMalloc(&di, sizeof(h)) != cudaSuccess || cudaMalloc(&do_, sizeof(out)) != cudaSuccess) return 2;
  if (cudaMemcpy(di, h, sizeof(h), cudaMemcpyHostToDevice) != cudaSuccess) return 3;
  cudagroup_reduce_sum<<<1, 32>>>(di, do_);
  if (cudaDeviceSynchronize() != cudaSuccess) return 4;
  if (cudaMemcpy(&out, do_, sizeof(out), cudaMemcpyDeviceToHost) != cudaSuccess) return 5;
  cudaFree(di); cudaFree(do_); printf("%%.9g\n", out); return 0;
}
`, strings.Join(input, ","))
	src := filepath.Join(dir, "native.cu")
	if err := os.WriteFile(src, []byte(f.Source+"\n"+host), 0600); err != nil {
		t.Fatal(err)
	}
	exe := filepath.Join(dir, "native")
	build := exec.Command(nvcc, "-O0", "-arch="+f.Target, src, "-o", exe)
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("nvcc build: %v\n%s", err, out)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	run := exec.CommandContext(ctx, exe)
	var stdout, stderr bytes.Buffer
	run.Stdout, run.Stderr = &stdout, &stderr
	if err := run.Run(); err != nil {
		t.Fatalf("native run: %v\nstderr: %s", err, stderr.String())
	}
	t.Logf("target=%s %s", f.Target, stderr.String())
	got, err := strconv.ParseFloat(strings.TrimSpace(stdout.String()), 32)
	if err != nil {
		t.Fatalf("parse native output %q: %v", stdout.String(), err)
	}
	if float32(got) != want {
		t.Fatalf("native reduction: got %v want %v", got, want)
	}
}
