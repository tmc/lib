package nvjitlinkx_test

import (
	"crypto/sha256"
	"fmt"
	"os"
	"runtime"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvjitlink"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/nvjitlinkx"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// TestCUBINPipeline requires a GPU and compatible NVRTC/nvJitLink libraries.
// It compiles two relocatable CUBINs, resolves a device function between them,
// and compares every output with a CPU calculation. Run with a process timeout.
func TestCUBINPipeline(t *testing.T) {
	if os.Getenv("CUDA_LINK_EXEC_TEST") != "1" {
		t.Skip("CUDA_LINK_EXEC_TEST=1 requires a GPU and matching compiler/linker")
	}
	check := func(err error) {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
	}
	cleanup := func(err error) {
		t.Helper()
		if err != nil {
			t.Error(err)
		}
	}
	dev, err := cuda.NewDevice(0)
	check(err)
	defer func() { cleanup(dev.Close()) }()
	major, minor, err := dev.ComputeCapability()
	check(err)
	arch := fmt.Sprintf("sm_%d%d", major, minor)
	sources := []string{
		`extern "C" __device__ int affine(int x) { return 3*x-7; }`,
		`extern "C" __device__ int affine(int);
extern "C" __global__ void apply(int* out) {
 int i = threadIdx.x;
 out[i] = affine(i*i-17);
}`,
	}
	inputs := make([]nvjitlinkx.Input, len(sources))
	for i, source := range sources {
		cm, err := nvrtcx.Compile(source, nvrtcx.WithName(fmt.Sprintf("cubin%d.cu", i)),
			nvrtcx.WithFlag("--gpu-architecture="+arch), nvrtcx.WithFlag("--device-c"),
			nvrtcx.WithOutput(nvrtcx.OutputCUBIN), nvrtcx.WithNoCache())
		check(err)
		data := cm.CUBIN()
		if len(data) < 4 || string(data[:4]) != "\x7fELF" || cm.PTX() != "" || len(cm.LTOIR()) != 0 {
			t.Fatalf("unit %d: expected only ELF CUBIN", i)
		}
		t.Logf("unit=%d source_sha256=%x cubin_bytes=%d cubin_sha256=%x log=%q", i, sha256.Sum256([]byte(source)), len(data), sha256.Sum256(data), cm.Log())
		inputs[i] = nvjitlinkx.Input{Name: fmt.Sprintf("unit%d.cubin", i), Kind: nvjitlink.InputCubin, Data: data}
	}
	// Omitting the provider must fail; otherwise the fixture does not prove linking.
	bad, err := nvjitlinkx.Link(inputs[1:], "-arch="+arch)
	if bad != nil {
		defer func() { cleanup(bad.Close()) }()
	}
	if err == nil || bad == nil || bad.ErrorLog == "" {
		t.Fatalf("missing provider: result=%v err=%v", bad, err)
	}
	t.Logf("missing-provider diagnostic=%q", bad.ErrorLog)
	result, err := nvjitlinkx.Link(inputs, "-arch="+arch)
	if result != nil {
		defer func() { cleanup(result.Close()) }()
	}
	check(err)
	if len(result.CUBIN) < 4 || string(result.CUBIN[:4]) != "\x7fELF" || len(result.PTX) != 0 {
		t.Fatal("linked output is not CUBIN")
	}
	t.Logf("linked_bytes=%d sha256=%x info=%q error=%q", len(result.CUBIN), sha256.Sum256(result.CUBIN), result.InfoLog, result.ErrorLog)
	var raw driver.Module
	err = dev.WithContext(func() error {
		var err error
		raw, err = driver.NewModuleLoadData(unsafe.Pointer(&result.CUBIN[0]))
		return err
	})
	runtime.KeepAlive(result)
	if err != nil {
		if raw != 0 {
			cleanup(dev.WithContext(raw.Unload))
		}
		t.Fatal(err)
	}
	owner, err := cuda.OwnModule(dev, raw)
	if err != nil {
		cleanup(dev.WithContext(raw.Unload))
		t.Fatal(err)
	}
	defer func() { cleanup(owner.Close()) }()
	kernel, err := owner.Function("apply")
	check(err)
	var p driver.DevicePtr
	check(dev.WithContext(func() error { var err error; p, err = driver.MemAlloc(64 * 4); return err }))
	defer func() { cleanup(dev.WithContext(p.MemFree)) }()
	check(kernel.Launch([3]uint32{1, 1, 1}, [3]uint32{64, 1, 1}, 0, 0, []unsafe.Pointer{unsafe.Pointer(&p)}))
	check(owner.Close()) // Completes submitted work before unloading.
	var got [64]int32
	check(dev.WithContext(func() error { return p.MemcpyDtoH(unsafe.Pointer(&got[0]), 64*4) }))
	for i, value := range got {
		want := int32(3*(i*i-17) - 7)
		if value != want {
			t.Fatalf("output[%d]=%d want %d", i, value, want)
		}
	}
	t.Log("verified 64 CPU-referenced outputs from two linked relocatable CUBINs")
}
