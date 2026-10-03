//go:build native

package cudamem_test

import (
	"crypto/sha256"
	_ "embed"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"math"
	"os"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudalayout"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudatest"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

//go:embed copy.go
var numericalCopyImplementation string

//go:embed mem.go
var numericalMemoryImplementation string

func TestNumericalOracle_CopyHealthyAndCorrupted(t *testing.T) {
	if os.Getenv("GPU_LOCK_HELD") == "" {
		t.Skip("native numerical copy requires gpu-lock")
	}
	dev := cudatest.RequireDevice(t)
	defer func() {
		if err := dev.Close(); err != nil {
			t.Error(err)
		}
	}()
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		t.Fatal(err)
	}
	const source = `extern "C" __global__ void corrupt_copy(float* out) {
  if (blockIdx.x == 0 && threadIdx.x == 0) out[1002] = 123456.0f;
}`
	compiled, err := nvrtcx.Compile(source, nvrtcx.WithArch(major, minor))
	if err != nil {
		t.Fatal(err)
	}
	module, _, err := compiled.LoadOwnedOutput(dev, nvrtcx.OutputPTX)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := module.Close(); err != nil {
			t.Error(err)
		}
	}()
	store, err := module.Function("corrupt_copy")
	if err != nil {
		t.Fatal(err)
	}
	const n = 1003
	want := make([]float32, n)
	inputBytes := make([]byte, 4*n)
	for i := range want {
		want[i] = float32(i-501) / 8
		binary.LittleEndian.PutUint32(inputBytes[4*i:], math.Float32bits(want[i]))
	}
	src, err := cudamem.Alloc[float32](dev, n)
	defer func() {
		if err := src.Free(); err != nil {
			t.Error(err)
		}
	}()
	if err != nil {
		t.Fatal(err)
	}
	base, err := cudamem.Alloc[float32](dev, n+2)
	defer func() {
		if err := base.Free(); err != nil {
			t.Error(err)
		}
	}()
	if err != nil {
		t.Fatal(err)
	}
	dst := base.Slice(1, n+1)
	if err := src.CopyFromHost(want); err != nil {
		t.Fatal(err)
	}
	layout, err := cudalayout.ParseLayout("1003:1")
	if err != nil {
		t.Fatal(err)
	}
	sourceHash := sha256.Sum256([]byte(numericalCopyImplementation + numericalMemoryImplementation + source))
	inputHash := sha256.Sum256(inputBytes)
	p := cudatest.CUDAOutputPolicy{Target: fmt.Sprintf("sm_%d%d", major, minor), DType: "fp32", MathMode: "exact FP32 D2D copy; NVRTC default corruption store", SourceDigest: hex.EncodeToString(sourceHash[:]), InputDigest: hex.EncodeToString(inputHash[:]), Elements: n, NaN: "reject", Infinity: "reject", Zero: "bits"}
	for _, corrupt := range []bool{false, true} {
		poison := make([]float32, n+2)
		for i := range poison {
			poison[i] = -99999
		}
		poison[0] = 123
		poison[n+1] = -123
		if err := base.CopyFromHost(poison); err != nil {
			t.Fatal(err)
		}
		plan, err := cudamem.CopyWithLayouts(dst, src, layout, layout)
		if err != nil {
			t.Fatal(err)
		}
		if corrupt {
			ptr := dst.Ptr()
			if err := dev.WithContext(func() error {
				if err := store.Launch([3]uint32{1, 1, 1}, [3]uint32{1, 1, 1}, 0, 0, []unsafe.Pointer{unsafe.Pointer(&ptr)}); err != nil {
					return err
				}
				return driver.CtxSynchronize()
			}); err != nil {
				t.Fatal(err)
			}
		}
		got := make([]float32, n+2)
		if err := base.CopyToHost(got); err != nil {
			t.Fatal(err)
		}
		if got[0] != 123 || got[n+1] != -123 {
			t.Fatal("copy output guard overwritten")
		}
		comparison := cudatest.CompareCUDAOutputFP32(got[1:n+1], want, p)
		if corrupt {
			var mismatch *cudatest.CUDAOutputMismatch
			if !errors.As(comparison, &mismatch) || mismatch.Index != 1002 || mismatch.ActualBits != math.Float32bits(123456) {
				t.Fatalf("device corrupt store escaped: %v", comparison)
			}
			t.Logf("device corruption rejected: %v", comparison)
		} else if comparison != nil {
			t.Fatal(comparison)
		}
		t.Logf("copy corrupt=%t elements=%d vector=%d source=%s input=%s target=%s", corrupt, n, plan.VectorWidth, p.SourceDigest, p.InputDigest, p.Target)
	}
}
