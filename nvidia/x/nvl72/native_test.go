package nvl72

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaerrors"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/internal/cudatest"
)

// The native gate compiles emitted kernels with nvcc and runs those that a
// single GPU can execute. Multi-rank fabric and tcgen05 paths stay on
// cudahostemu.
const nvl72NativeEnv = "CUDA_NVL72_NATIVE"

type nvl72NativeGateClosedError struct{}

func (nvl72NativeGateClosedError) Error() string {
	return "nvl72: native gate is closed: CUDA_NVL72_NATIVE=1 unset, no device examined"
}

func (nvl72NativeGateClosedError) Unwrap() error { return cudaerrors.ErrHardwarePending }

func checkNVL72NativeGate(getenv func(string) string) error {
	if getenv(nvl72NativeEnv) != "1" {
		return nvl72NativeGateClosedError{}
	}
	if _, err := cudalock.Inherited(getenv); err != nil {
		return fmt.Errorf("nvl72: native gate open without inherited gpu lock: %w", err)
	}
	return nil
}

// nativeTarget returns the architecture native tests build for;
// CUDA_NVL72_NATIVE_ARCH names another.
func nativeTarget() string {
	if a := os.Getenv("CUDA_NVL72_NATIVE_ARCH"); a != "" {
		return a
	}
	return "sm_121a"
}

// requireNative skips t when the gate is closed and fails it when the gate
// is open without an inherited GPU lock. It returns nvcc.
func requireNative(t *testing.T) string {
	t.Helper()
	err := checkNVL72NativeGate(os.Getenv)
	if errors.Is(err, cudaerrors.ErrHardwarePending) {
		t.Skip(err)
	}
	if err != nil {
		t.Fatal(err)
	}
	nvcc, _ := cudatest.NVCC(t)
	cudatest.SkipUnlessTarget(t, nvcc, nativeTarget())
	return nvcc
}

// nativeCompile compiles src for target with nvcc using the extra flags
// (such as -cubin or -ptx) into out.
func nativeCompile(t *testing.T, nvcc, target, src, out string, flags ...string) {
	t.Helper()
	cu := filepath.Join(t.TempDir(), "k.cu")
	if err := os.WriteFile(cu, []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	args := append([]string{"-std=c++17", "-arch=" + target, "-o", out}, flags...)
	args = append(args, cu)
	if b, err := exec.CommandContext(t.Context(), nvcc, args...).CombinedOutput(); err != nil {
		t.Fatalf("nvcc %s: %v\n%s", strings.Join(args, " "), err, b)
	}
}

// nativeRun builds src plus a main for the native target and returns what
// the program prints.
func nativeRun(t *testing.T, nvcc, src string) string {
	t.Helper()
	bin := filepath.Join(t.TempDir(), "prog")
	nativeCompile(t, nvcc, nativeTarget(), src, bin)
	ctx := t.Context()
	cmd := exec.CommandContext(ctx, bin)
	cmd.WaitDelay = 5 * time.Second
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("%s: %v\n%s", bin, err, out)
	}
	return string(out)
}

// parseFields returns the numbers on the line of out that starts with key.
func parseFields(t *testing.T, out, key string) []float64 {
	t.Helper()
	for line := range strings.Lines(out) {
		f := strings.Fields(line)
		if len(f) == 0 || f[0] != key {
			continue
		}
		v := make([]float64, len(f)-1)
		for i, s := range f[1:] {
			x, err := strconv.ParseFloat(s, 64)
			if err != nil {
				t.Fatalf("%s field %d: %v", key, i, err)
			}
			v[i] = x
		}
		return v
	}
	t.Fatalf("no %q line in output:\n%s", key, out)
	return nil
}

func cArray(ctype, name string, n int, elem func(i int) string) string {
	var b strings.Builder
	fmt.Fprintf(&b, "static const %s %s[%d] = {", ctype, name, n)
	for i := range n {
		if i > 0 {
			b.WriteString(",")
		}
		b.WriteString(elem(i))
	}
	b.WriteString("};\n")
	return b.String()
}

func TestNVL72NativeGateClosed(t *testing.T) {
	if os.Getenv(nvl72NativeEnv) == "1" {
		t.Skip("closed-gate control applies when CUDA_NVL72_NATIVE != 1")
	}
	if err := checkNVL72NativeGate(os.Getenv); !errors.Is(err, cudaerrors.ErrHardwarePending) {
		t.Fatalf("closed gate = %v, want ErrHardwarePending", err)
	}
	presenceOnly := func(k string) string {
		switch k {
		case nvl72NativeEnv, "GPU_LOCK_HELD":
			return "1"
		}
		return ""
	}
	if err := checkNVL72NativeGate(presenceOnly); err == nil || errors.Is(err, cudaerrors.ErrHardwarePending) {
		t.Fatalf("gate with GPU_LOCK_HELD=1 = %v, want a lock error", err)
	}
}
