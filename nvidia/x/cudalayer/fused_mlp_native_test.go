//go:build native

package cudalayer_test

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"math"
	"os"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudalayer"
	"github.com/tmc/lib/nvidia/x/cudatest"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

type fusedMLPKernelSetter struct{ kernel *cuda.Kernel }

func (s fusedMLPKernelSetter) SetAttribute(attr driver.FunctionAttribute, value int32) error {
	return s.kernel.SetAttribute(attr, int(value))
}

func TestDeviceFunc_HardwarePendingOnNoGPU(t *testing.T) {
	runNumericalMLP(t, false)
}

func TestNumericalOracle_MLPHealthyAndNaNStore(t *testing.T) {
	t.Run("healthy", func(t *testing.T) { runNumericalMLP(t, false) })
	t.Run("device NaN store", func(t *testing.T) { runNumericalMLP(t, true) })
}

func runNumericalMLP(t *testing.T, corrupt bool) {
	t.Helper()
	if os.Getenv("GPU_LOCK_HELD") == "" {
		t.Skip("SKIPPED, NOT VERIFIED: native launch requires GPU_LOCK_HELD or gpu-lock")
	}
	dev := cudatest.RequireDevice(t)
	if dev == nil {
		t.Fatalf("RequireDevice returned nil without skipping")
	}
	defer func() {
		if err := dev.Close(); err != nil {
			t.Errorf("close device: %v", err)
		}
	}()

	major, minor, err := dev.ComputeCapability()
	if err != nil {
		t.Fatal(err)
	}
	mlp, err := cudalayer.NewFusedMLP(64, 64, 64, 3)
	if err != nil {
		t.Fatal(err)
	}
	unit, err := cudair.EmitTree(mlp, "fused_mlp64_eval")
	if err != nil {
		t.Fatal(err)
	}
	subs := map[string]string{"FORWARD_DEVICE_FUNCTION": unit.Source, "LOSS_DEVICE_FUNCTION": "", "KERNEL_NAME": "mlp64_kernel", "FUNC_NAME": "fused_mlp64_eval", "IN_WIDTH": "64", "OUT_WIDTH": "64"}
	const nanStore = `
extern "C" __global__ void numerical_nan_store(float* out) {
  if (blockIdx.x == 0 && threadIdx.x == 0) out[1003] = __int_as_float(0x7fc00123);
}
`
	template := cudalayer.OuterKernelTemplate
	if corrupt {
		template += nanStore
	}
	source, err := nvrtcx.SpliceTemplate(template, subs)
	if err != nil {
		t.Fatal(err)
	}
	compiled, err := nvrtcx.SpliceAndCompile(dev, template, subs, nvrtcx.WithArch(major, minor))
	if err != nil {
		t.Fatalf("compile spliced MLP: %v", err)
	}
	module, _, err := compiled.LoadOwnedOutput(dev, nvrtcx.OutputPTX)
	if err != nil {
		t.Fatalf("load spliced MLP: %v", err)
	}
	defer func() {
		if err := module.Close(); err != nil {
			t.Errorf("close module: %v", err)
		}
	}()
	kernel, err := module.Function("mlp64_kernel")
	if err != nil {
		t.Fatal(err)
	}
	if err := nvrtcx.PrepareDeviceFuncLaunch(fusedMLPKernelSetter{kernel}, unit); err != nil {
		t.Fatal(err)
	}

	const rows = 65
	input := make([]float32, rows*64)
	for i := range input {
		input[i] = float32((i*11)%37-18) / 9
	}
	params := make([]float32, 3*64*64)
	for layer := 0; layer < 3; layer++ {
		for in := 0; in < 64; in++ {
			for out := 0; out < 64; out++ {
				params[layer*4096+in*64+out] = float32((in*7+out*3+layer*5)%13-6) / 128
			}
		}
	}
	got := make([]float32, rows*64)
	var count int32 = rows
	var inPtr, paramPtr, outPtr driver.DevicePtr
	const block = 32
	grid := uint32((rows + block - 1) / block)
	allocBytes := func(n int) (driver.DevicePtr, error) { return driver.MemAllocBytes(uint(n * 4)) }
	if err := dev.WithContext(func() error {
		var err error
		inPtr, err = allocBytes(len(input))
		if err != nil {
			return err
		}
		defer func() {
			if inPtr != 0 {
				_ = inPtr.Free()
			}
		}()
		paramPtr, err = allocBytes(len(params))
		if err != nil {
			return err
		}
		defer func() {
			if paramPtr != 0 {
				_ = paramPtr.Free()
			}
		}()
		outPtr, err = allocBytes(len(got))
		if err != nil {
			return err
		}
		defer func() {
			if outPtr != 0 {
				_ = outPtr.Free()
			}
		}()
		if err = inPtr.CopyFromHost(input); err != nil {
			return err
		}
		if err = paramPtr.CopyFromHost(params); err != nil {
			return err
		}
		args := []unsafe.Pointer{unsafe.Pointer(&count), unsafe.Pointer(&inPtr), unsafe.Pointer(&paramPtr), unsafe.Pointer(&outPtr)}
		if err = kernel.Launch([3]uint32{grid, 1, 1}, [3]uint32{block, 1, 1}, 0, 0, args); err != nil {
			return err
		}
		if err = driver.CtxSynchronize(); err != nil {
			return err
		}
		if corrupt {
			store, e := module.Function("numerical_nan_store")
			if e != nil {
				return e
			}
			if e = store.Launch([3]uint32{1, 1, 1}, [3]uint32{1, 1, 1}, 0, 0, []unsafe.Pointer{unsafe.Pointer(&outPtr)}); e != nil {
				return e
			}
			if e = driver.CtxSynchronize(); e != nil {
				return e
			}
		}
		return outPtr.CopyToHost(got)
	}); err != nil {
		t.Fatal(err)
	}
	want := append([]float32(nil), input...)
	for layer := 0; layer < 3; layer++ {
		next := make([]float32, rows*64)
		for row := 0; row < rows; row++ {
			for out := 0; out < 64; out++ {
				var sum float32
				for in := 0; in < 64; in++ {
					sum += want[row*64+in] * params[layer*4096+in*64+out]
				}
				if layer < 2 && sum < 0 {
					sum = 0
				}
				next[row*64+out] = sum
			}
		}
		want = next
	}
	inputBytes := make([]byte, 4*(len(input)+len(params)))
	for i, v := range input {
		binary.LittleEndian.PutUint32(inputBytes[4*i:], math.Float32bits(v))
	}
	for i, v := range params {
		binary.LittleEndian.PutUint32(inputBytes[4*(len(input)+i):], math.Float32bits(v))
	}
	sourceHash, inputHash := sha256.Sum256([]byte(source)), sha256.Sum256(inputBytes)
	p := cudatest.CUDAOutputPolicy{Target: fmt.Sprintf("sm_%d%d", major, minor), DType: "fp32", MathMode: "NVRTC default FP32", SourceDigest: hex.EncodeToString(sourceHash[:]), InputDigest: hex.EncodeToString(inputHash[:]), Elements: len(want), Absolute: 1e-4, NaN: "reject", Infinity: "reject", Zero: "equal"}
	comparison := cudatest.CompareCUDAOutputFP32(got, want, p)
	if corrupt {
		var mismatch *cudatest.CUDAOutputMismatch
		if !errors.As(comparison, &mismatch) || mismatch.Index != 1003 || mismatch.ActualBits != 0x7fc00123 {
			t.Fatalf("device NaN control escaped: %v", comparison)
		}
		t.Logf("device NaN rejected through MLP comparison: %v", comparison)
	} else if comparison != nil {
		t.Fatal(comparison)
	}
	imageHash := sha256.Sum256([]byte(compiled.PTX()))
	t.Logf("compiled, loaded and launched fused MLP on sm_%d%d; corrupt=%t, %d values across %d blocks checked; PTX=%x source=%s input=%s math=%s", major, minor, corrupt, len(got), grid, imageHash, p.SourceDigest, p.InputDigest, p.MathMode)
}
