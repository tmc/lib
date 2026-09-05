package nvrtcx

import (
	"errors"
	"os"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
)

func TestDriverJITDiagnostics(t *testing.T) {
	if os.Getenv("CUDA_JIT_TEST") != "1" {
		t.Skip("CUDA_JIT_TEST=1 requires a CUDA device and driver JIT")
	}
	dev, err := cuda.NewDevice(0)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := dev.Close(); err != nil {
			t.Error(err)
		}
	}()
	const header = ".version 8.0\n.target sm_90\n.address_size 64\n.visible .entry probe() {\n"
	for _, tt := range []struct {
		name, body string
		invalid    bool
	}{
		{"invalid", "not_a_ptx_instruction;\nret;\n}", true},
		{"valid_after_failure", "ret;\n}", false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			cm := &CompiledModule{output: OutputPTX, ptx: header + tt.body}
			mod, log, err := cm.LoadOutputWithLog(dev, OutputPTX)
			if mod != 0 {
				defer func() {
					if err := dev.WithContext(mod.Unload); err != nil {
						t.Error(err)
					}
				}()
			}
			t.Logf("output=%d info=%q error=%q load=%v", log.Output, log.Info, log.Error, err)
			if tt.invalid {
				if !errors.Is(err, driver.ErrInvalidPtx) || log.Error == "" || mod != 0 {
					t.Fatalf("invalid PTX result: module=%v log=%+v err=%v", mod, log, err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if err := dev.WithContext(func() error {
				fn, err := mod.GetFunction("probe")
				if err != nil {
					return err
				}
				if err := fn.LaunchKernel(1, 1, 1, 1, 1, 1, 0, 0, nil); err != nil {
					return err
				}
				return driver.CtxSynchronize()
			}); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestOwnedCompilerModule(t *testing.T) {
	if os.Getenv("CUDA_JIT_TEST") != "1" {
		t.Skip("CUDA_JIT_TEST=1 requires CUDA and NVRTC")
	}
	check := func(err error) {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
	}
	dev, err := cuda.NewDevice(0)
	check(err)
	defer func() {
		if err := dev.Close(); err != nil {
			t.Error(err)
		}
	}()
	major, minor, err := dev.ComputeCapability()
	check(err)
	cm, err := Compile(`extern "C" __global__ void fill(int *out) { out[threadIdx.x] = threadIdx.x + 9; }`, WithArch(major, minor), WithOutput(OutputPTX), WithNoCache())
	check(err)
	owner, log, err := cm.LoadOwnedOutput(dev, OutputPTX)
	check(err)
	defer func() {
		if err := owner.Close(); err != nil {
			t.Error(err)
		}
	}()
	if log.Output != OutputPTX {
		t.Fatal("lost artifact identity")
	}
	kernel, err := owner.Function("fill")
	check(err)
	var p driver.DevicePtr
	check(dev.WithContext(func() error { var err error; p, err = driver.MemAlloc(64 * 4); return err }))
	defer func() {
		if err := dev.WithContext(p.MemFree); err != nil {
			t.Error(err)
		}
	}()
	check(kernel.Launch([3]uint32{1, 1, 1}, [3]uint32{64, 1, 1}, 0, 0, []unsafe.Pointer{unsafe.Pointer(&p)}))
	// Close must finish the queued kernel before unloading it.
	check(owner.Close())
	var got [64]int32
	check(dev.WithContext(func() error { return p.MemcpyDtoH(unsafe.Pointer(&got[0]), 64*4) }))
	for i, v := range got {
		if v != int32(i+9) {
			t.Fatalf("output[%d]=%d", i, v)
		}
	}
	if err := kernel.Launch([3]uint32{1, 1, 1}, [3]uint32{1, 1, 1}, 0, 0, nil); err == nil {
		t.Fatal("closed kernel launched")
	}
	t.Log("NVRTC -> owned PTX module -> 64 matching outputs; Close synchronized launch and invalidated kernel")
}
