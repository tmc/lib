//go:build nvjitlink_physical

// Package nvjitlink_test links real PTX into a real cubin and then runs it.
// Checking that the link succeeds proves very little on its own, so the linked
// cubin is loaded through the CUDA driver and launched; the kernel writes a
// value the test predicts in advance for every element, which is the
// independent oracle. A mislinked or truncated cubin fails to load, and a
// binding defect in the buffer-size or output-pointer handling shows up as a
// short or corrupt image rather than as a bad status code.
//
// Set NVJITLINK_PHYSICAL_PERTURB=expected to run the negative control.
package nvjitlink_test

import (
	"bytes"
	"os"
	"runtime"
	"strconv"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvjitlink"
	"github.com/tmc/lib/nvidia/nvrtc"
)

// kernelSource writes a value derived from the thread index so that the host
// can predict every element of the result independently of the GPU.
const kernelSource = `
extern "C" __global__ void fill(int *out, int n, int bias) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) out[i] = i * i + bias;
}
`

const (
	elems = 256
	bias  = 7
)

// want returns the expected contents of the output buffer.
func want() []int32 {
	xs := make([]int32, elems)
	for i := range xs {
		xs[i] = int32(i*i + bias)
	}
	return xs
}

// gpuArch returns the compute capability of device 0 as a "89" style string.
func gpuArch(t *testing.T) string {
	t.Helper()
	dev, err := driver.NewDeviceGet(0)
	if err != nil {
		t.Skipf("no CUDA device: %v", err)
	}
	var major, minor int32
	if err := dev.ComputeCapability(&major, &minor); err != nil {
		t.Fatalf("compute capability: %v", err)
	}
	return strconv.Itoa(int(major)) + strconv.Itoa(int(minor))
}

// compilePTX builds kernelSource to PTX with NVRTC.
func compilePTX(t *testing.T, arch string) []byte {
	t.Helper()
	prog, err := nvrtc.CreateProgram(kernelSource, "fill.cu", nil, nil)
	if err != nil {
		t.Skipf("NVRTC unavailable: %v", err)
	}
	defer func() {
		if err := nvrtc.DestroyProgram(&prog); err != nil {
			t.Errorf("destroy program: %v", err)
		}
	}()
	if err := prog.Compile([]string{"--gpu-architecture=compute_" + arch}); err != nil {
		log, _ := prog.GetProgramLog()
		t.Fatalf("compile: %v\n%s", err, log)
	}
	ptx, err := prog.GetPTX()
	if err != nil {
		t.Fatalf("GetPTX: %v", err)
	}
	if !bytes.Contains(ptx, []byte(".entry fill")) {
		t.Fatalf("PTX does not define the expected entry point:\n%s", ptx)
	}
	return ptx
}

// loadOverride points the loader at NVJITLINK_LIB when it is set. Other tests
// in this package load the system library first, and the loader refuses to
// switch libraries while one is open, so an already-open library is closed.
func loadOverride(t *testing.T) {
	t.Helper()
	lib := os.Getenv("NVJITLINK_LIB")
	if lib == "" {
		return
	}
	if nvjitlink.Loaded() && nvjitlink.Path() != lib {
		if err := nvjitlink.Close(); err != nil {
			t.Fatalf("close %s: %v", nvjitlink.Path(), err)
		}
	}
	if err := nvjitlink.Load(lib); err != nil {
		t.Fatalf("load %s: %v", lib, err)
	}
	t.Logf("loaded %s", nvjitlink.Path())
}

// TestLinkAndRun links NVRTC-produced PTX with nvJitLink and executes the
// result. The launch and the readback are the assertion; the link status is
// not.
func TestLinkAndRun(t *testing.T) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if err := driver.Init(0); err != nil {
		t.Skipf("CUDA driver unavailable: %v", err)
	}
	arch := gpuArch(t)
	t.Logf("device compute capability sm_%s", arch)
	dev, err := driver.NewDeviceGet(0)
	if err != nil {
		t.Skipf("no CUDA device: %v", err)
	}
	ctx, err := dev.PrimaryCtxRetain()
	if err != nil {
		t.Fatalf("retain primary context: %v", err)
	}
	defer func() {
		if err := dev.PrimaryCtxRelease(); err != nil {
			t.Errorf("release context: %v", err)
		}
	}()
	if err := ctx.SetCurrent(); err != nil {
		t.Fatalf("set current context: %v", err)
	}

	ptx := compilePTX(t, arch)

	// The generated bindings resolve the CUDA 13 versioned symbol names
	// (__nvJitLinkCreate_13_0 and friends), which a CUDA 12 libnvJitLink does
	// not export. NVJITLINK_LIB points the loader at a specific library so the
	// wrapper shapes can be exercised independently of which toolkit is
	// installed system-wide.
	loadOverride(t)
	if major, minor, err := nvjitlink.Version(); err == nil {
		t.Logf("nvJitLink version %d.%d", major, minor)
	}

	handle, err := nvjitlink.Create([]string{"-arch=sm_" + arch})
	if err != nil {
		t.Skipf("nvJitLink unavailable: %v", err)
	}
	defer func() {
		if err := nvjitlink.Destroy(&handle); err != nil {
			t.Errorf("destroy linker: %v", err)
		}
	}()
	if err := handle.AddData(nvjitlink.InputPtx, unsafe.Pointer(&ptx[0]), uint(len(ptx)), "fill.ptx"); err != nil {
		log, _ := handle.GetErrorLog()
		t.Fatalf("AddData: %v\n%s", err, log)
	}
	if err := handle.Complete(); err != nil {
		log, _ := handle.GetErrorLog()
		t.Fatalf("Complete: %v\n%s", err, log)
	}
	size, err := handle.GetLinkedCubinSize()
	if err != nil {
		t.Fatalf("GetLinkedCubinSize: %v", err)
	}
	cubin, err := handle.GetLinkedCubin()
	if err != nil {
		t.Fatalf("GetLinkedCubin: %v", err)
	}
	// The two-call size/get pair is the classic place for an out-parameter
	// defect: a wrong size returns a short buffer that still loads.
	if uint(len(cubin)) != size {
		t.Fatalf("GetLinkedCubin returned %d bytes, GetLinkedCubinSize said %d", len(cubin), size)
	}
	if !bytes.HasPrefix(cubin, []byte("\x7fELF")) {
		t.Fatalf("linked cubin is not an ELF image: % x", cubin[:min(16, len(cubin))])
	}
	t.Logf("linked %d bytes of PTX into a %d byte ELF cubin", len(ptx), len(cubin))

	// Info log is optional but should at least be readable.
	if info, err := handle.GetInfoLog(); err == nil && info != "" {
		t.Logf("linker info: %s", info)
	}

	got := runCubin(t, cubin)
	expected := want()
	if os.Getenv("NVJITLINK_PHYSICAL_PERTURB") == "expected" {
		expected[elems/2]++
	}
	for i := range expected {
		if got[i] != expected[i] {
			t.Fatalf("out[%d] = %d, want %d", i, got[i], expected[i])
		}
	}
	t.Logf("kernel output matches the predicted i*i+%d for all %d elements", bias, elems)
}

// runCubin loads a cubin, launches the fill kernel and returns the output.
func runCubin(t *testing.T, cubin []byte) []int32 {
	t.Helper()
	mod, err := driver.NewModuleLoadData(unsafe.Pointer(&cubin[0]))
	if err != nil {
		t.Fatalf("ModuleLoadData: %v", err)
	}
	defer func() {
		if err := mod.Unload(); err != nil {
			t.Errorf("unload module: %v", err)
		}
	}()
	fn, err := mod.GetFunction("fill")
	if err != nil {
		t.Fatalf("GetFunction: %v", err)
	}
	const size = elems * 4
	out, err := driver.MemAlloc(size)
	if err != nil {
		t.Fatalf("MemAlloc: %v", err)
	}
	defer func() {
		if err := out.MemFree(); err != nil {
			t.Errorf("free device buffer: %v", err)
		}
	}()
	// Fill with a sentinel so that a kernel that never runs cannot pass.
	sentinel := make([]int32, elems)
	for i := range sentinel {
		sentinel[i] = -1
	}
	if err := out.MemcpyHtoD(unsafe.Pointer(&sentinel[0]), size); err != nil {
		t.Fatalf("MemcpyHtoD: %v", err)
	}
	n, b := int32(elems), int32(bias)
	params := []unsafe.Pointer{
		unsafe.Pointer(&out),
		unsafe.Pointer(&n),
		unsafe.Pointer(&b),
	}
	if err := fn.LaunchKernel(elems/64, 1, 1, 64, 1, 1, 0, 0, params); err != nil {
		t.Fatalf("LaunchKernel: %v", err)
	}
	if err := driver.CtxSynchronize(); err != nil {
		t.Fatalf("synchronize: %v", err)
	}
	got := make([]int32, elems)
	if err := out.MemcpyDtoH(unsafe.Pointer(&got[0]), size); err != nil {
		t.Fatalf("MemcpyDtoH: %v", err)
	}
	return got
}

// TestVersionedSymbolPinning documents a defect in the generated bindings.
//
// Every nvJitLink entry point is a macro in nvJitLink.h that expands to a
// symbol carrying the toolkit version, so the header this package was
// generated from fixed the suffix at _13_0. libnvJitLink.so.12 exports
// __nvJitLinkCreate_12_2 through __nvJitLinkCreate_12_9 and an unsuffixed
// nvJitLinkCreate, but nothing named _13_0, so against any CUDA 12 toolkit
// every call in this package fails at symbol resolution and the package is
// unusable. Only Version, whose symbol is not versioned, resolves.
//
// The wrapper shapes themselves are correct: TestLinkAndRun passes end to end
// when NVJITLINK_LIB points at a CUDA 13 library. The fix belongs in the
// generator, which should fall back to the unsuffixed alias or probe the
// suffixes the loaded library actually exports.
func TestVersionedSymbolPinning(t *testing.T) {
	loadOverride(t)
	major, minor, err := nvjitlink.Version()
	if err != nil {
		t.Skipf("nvJitLink unavailable: %v", err)
	}
	t.Logf("loaded %s, reporting version %d.%d", nvjitlink.Path(), major, minor)
	if nvjitlink.HasCreate() {
		return
	}
	// nm on this library shows __nvJitLinkCreate_12_2 through _12_9 plus an
	// unsuffixed nvJitLinkCreate, so an alias that would have worked is there.
	// Has only reports symbols this package registered, so it cannot see it.
	t.Fatalf("bindings require __nvJitLinkCreate_13_0, which libnvJitLink %d.%d does not export; "+
		"every entry point except Version fails to resolve and the package is unusable on this toolkit",
		major, minor)
}
