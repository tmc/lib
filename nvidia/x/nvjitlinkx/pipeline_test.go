package nvjitlinkx_test

import (
	"fmt"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"os"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/nvjitlink"
	"github.com/tmc/lib/nvidia/x/nvjitlinkx"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

func ExampleLink() {
	_, err := nvjitlinkx.Link(nil, "-arch=sm_80")
	fmt.Println(err)
	// Output: nvjitlinkx: no inputs
}

// TestLTOPipeline requires a compatible NVRTC/nvJitLink installation, but does
// not normally load a CUDA module or launch GPU work. CUDA_LINK_EXEC_TEST=1
// additionally targets the present device and verifies the linked kernel result.
func TestLTOPipeline(t *testing.T) {
	if os.Getenv("CUDA_COMPILER_TEST") != "1" && os.Getenv("CUDA_LINK_EXEC_TEST") != "1" {
		t.Skip("CUDA_COMPILER_TEST=1 required")
	}
	major, minor := 8, 0
	var dev *cuda.Device
	if os.Getenv("CUDA_LINK_EXEC_TEST") == "1" {
		var err error
		dev, err = cuda.NewDevice(0)
		if err != nil {
			t.Fatal(err)
		}
		defer func() {
			if err := dev.Close(); err != nil {
				t.Error(err)
			}
		}()
		major, minor, err = dev.ComputeCapability()
		if err != nil {
			t.Fatal(err)
		}
	}
	arch := fmt.Sprintf("-arch=sm_%d%d", major, minor)
	sources := []string{
		`extern "C" __device__ float twice(float x) { return 2*x; }`,
		`extern "C" __device__ float twice(float);
extern "C" __global__ void apply(float* out) { out[0] = twice(3); }`,
	}
	var inputs []nvjitlinkx.Input
	for i, source := range sources {
		cm, err := nvrtcx.Compile(source, nvrtcx.WithName(fmt.Sprintf("unit%d.cu", i)), nvrtcx.WithArch(major, minor), nvrtcx.WithOutput(nvrtcx.OutputLTOIR), nvrtcx.WithNoCache())
		if err != nil {
			t.Fatal(err)
		}
		if len(cm.LTOIR()) == 0 || cm.PTX() != "" || len(cm.CUBIN()) != 0 {
			t.Fatal("wrong compiler artifact")
		}
		inputs = append(inputs, nvjitlinkx.Input{Name: fmt.Sprintf("unit%d", i), Kind: nvjitlink.InputLtoir, Data: cm.LTOIR()})
	}
	result, err := nvjitlinkx.Link(inputs, arch, "-lto")
	if result != nil {
		defer func() {
			if err := result.Close(); err != nil {
				t.Error(err)
			}
		}()
	}
	if err != nil {
		t.Fatal(err)
	}
	if len(result.CUBIN) < 4 || string(result.CUBIN[:4]) != "\x7fELF" {
		t.Fatal("no ELF CUBIN")
	}
	if dev != nil {
		check := func(err error) {
			t.Helper()
			if err != nil {
				t.Fatal(err)
			}
		}
		var raw driver.Module
		check(dev.WithContext(func() error {
			var err error
			raw, err = driver.NewModuleLoadData(unsafe.Pointer(&result.CUBIN[0]))
			return err
		}))
		owner, err := cuda.OwnModule(dev, raw)
		if err != nil {
			_ = dev.WithContext(raw.Unload)
			t.Fatal(err)
		}
		defer func() {
			if err := owner.Close(); err != nil {
				t.Error(err)
			}
		}()
		kernel, err := owner.Function("apply")
		check(err)
		var p driver.DevicePtr
		check(dev.WithContext(func() error { var err error; p, err = driver.MemAlloc(4); return err }))
		defer func() {
			if err := dev.WithContext(p.MemFree); err != nil {
				t.Error(err)
			}
		}()
		check(kernel.Launch([3]uint32{1, 1, 1}, [3]uint32{1, 1, 1}, 0, 0, []unsafe.Pointer{unsafe.Pointer(&p)}))
		check(owner.Close())
		var got float32
		check(dev.WithContext(func() error { return p.MemcpyDtoH(unsafe.Pointer(&got), 4) }))
		if got != 6 {
			t.Fatalf("linked kernel output=%g want6", got)
		}
		t.Log("two NVRTC LTOIR units linked, loaded, and executed: twice(3)=6")
	}
	bad, err := nvjitlinkx.Link(inputs[1:], arch, "-lto")
	if bad != nil {
		defer func() {
			if err := bad.Close(); err != nil {
				t.Error(err)
			}
		}()
	}
	if err == nil {
		t.Fatal("unresolved device function linked successfully")
	}
	if bad == nil || bad.ErrorLog == "" {
		t.Fatal("missing unresolved-symbol diagnostic")
	}
}
