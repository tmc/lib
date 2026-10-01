//go:build native

package cudalayer_test

import (
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
	compiled, err := nvrtcx.SpliceAndCompile(dev, cudalayer.OuterKernelTemplate, subs, nvrtcx.WithArch(major, minor))
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
	for i, wantValue := range want {
		if math.Abs(float64(got[i]-wantValue)) > 1e-4 {
			t.Fatalf("output[%d]=%g, want %g", i, got[i], wantValue)
		}
	}
	t.Logf("compiled, loaded and launched fused MLP on sm_%d%d; %d values across %d blocks matched reference", major, minor, len(got), grid)
}
