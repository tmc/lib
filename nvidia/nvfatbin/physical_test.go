//go:build nvfatbin_physical

// Package nvfatbin_test builds a real fatbin from NVRTC output and then runs a
// kernel out of it through the CUDA driver. The status codes are not the
// assertion: the assertion is that the driver accepts the container this
// package produced and that the kernel inside it writes the values the host
// predicted.
//
// Set NVFATBIN_PHYSICAL_PERTURB=expected to run the negative control.
package nvfatbin_test

import (
	"encoding/binary"
	"os"
	"runtime"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvfatbin"
	"github.com/tmc/lib/nvidia/nvrtc"
)

const kernelSource = `
extern "C" __global__ void fill(int *out, int n, int bias) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) out[i] = i * 3 + bias;
}
`

const (
	elems = 128
	bias  = 11
)

// fatbinMagic is the value at the head of every CUDA fatbin container
// (FATBIN_MAGIC in nvcc's fatbinary format).
const fatbinMagic = 0xBA55ED50

func TestBuildFatbinAndRun(t *testing.T) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if err := driver.Init(0); err != nil {
		t.Skipf("CUDA driver unavailable: %v", err)
	}
	dev, err := driver.NewDeviceGet(0)
	if err != nil {
		t.Skipf("no CUDA device: %v", err)
	}
	var major, minor int32
	if err := dev.ComputeCapability(&major, &minor); err != nil {
		t.Fatalf("compute capability: %v", err)
	}
	arch := string(rune('0'+major)) + string(rune('0'+minor))
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

	// NVRTC produces a cubin for this exact GPU, which is what goes into the
	// fatbin. nvFatbin names architectures by their bare number, not sm_NN.
	prog, err := nvrtc.CreateProgram(kernelSource, "fill.cu", nil, nil)
	if err != nil {
		t.Skipf("NVRTC unavailable: %v", err)
	}
	defer func() {
		if err := nvrtc.DestroyProgram(&prog); err != nil {
			t.Errorf("destroy program: %v", err)
		}
	}()
	if err := prog.Compile([]string{"--gpu-architecture=sm_" + arch}); err != nil {
		log, _ := prog.GetProgramLog()
		t.Fatalf("compile: %v\n%s", err, log)
	}
	cubin, err := prog.GetCUBIN()
	if err != nil {
		t.Fatalf("GetCUBIN: %v", err)
	}

	majorV, minorV, err := nvfatbin.Version()
	if err != nil {
		t.Skipf("nvFatbin unavailable: %v", err)
	}
	t.Logf("nvFatbin version %d.%d, target sm_%s", majorV, minorV, arch)

	h, err := nvfatbin.Create([]string{"-64"})
	if err != nil {
		t.Skipf("nvFatbin unavailable: %v", err)
	}
	defer func() {
		if err := nvfatbin.Destroy(&h); err != nil {
			t.Errorf("destroy fatbin: %v", err)
		}
	}()
	if err := h.AddCubin(unsafe.Pointer(&cubin[0]), uint(len(cubin)), arch, "fill"); err != nil {
		t.Fatalf("AddCubin: %v", err)
	}

	size, err := h.Size()
	if err != nil {
		t.Fatalf("Size: %v", err)
	}
	fatbin, err := h.Get()
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	// Size and Get are separate calls into the C API; a wrong out-parameter in
	// either would produce a container the driver still might accept.
	if uint(len(fatbin)) != size {
		t.Fatalf("Get returned %d bytes, Size said %d", len(fatbin), size)
	}
	if len(fatbin) < 16 {
		t.Fatalf("fatbin is only %d bytes", len(fatbin))
	}
	if got := binary.LittleEndian.Uint32(fatbin[:4]); got != fatbinMagic {
		t.Fatalf("fatbin magic = %#08x, want %#08x", got, fatbinMagic)
	}
	// The fatbin header records its own payload size at offset 8; it must
	// account for everything after the 16 byte header.
	if got, want := binary.LittleEndian.Uint64(fatbin[8:16]), uint64(len(fatbin)-16); got != want {
		t.Fatalf("fatbin header payload size = %d, but %d bytes follow the header", got, want)
	}
	t.Logf("built a %d byte fatbin containing a %d byte cubin", len(fatbin), len(cubin))

	mod, err := driver.NewModuleLoadFatBinary(unsafe.Pointer(&fatbin[0]))
	if err != nil {
		t.Fatalf("ModuleLoadFatBinary: %v", err)
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

	const nbytes = elems * 4
	out, err := driver.MemAlloc(nbytes)
	if err != nil {
		t.Fatalf("MemAlloc: %v", err)
	}
	defer func() {
		if err := out.MemFree(); err != nil {
			t.Errorf("free device buffer: %v", err)
		}
	}()
	sentinel := make([]int32, elems)
	for i := range sentinel {
		sentinel[i] = -1
	}
	if err := out.MemcpyHtoD(unsafe.Pointer(&sentinel[0]), nbytes); err != nil {
		t.Fatalf("MemcpyHtoD: %v", err)
	}
	n, b := int32(elems), int32(bias)
	params := []unsafe.Pointer{unsafe.Pointer(&out), unsafe.Pointer(&n), unsafe.Pointer(&b)}
	if err := fn.LaunchKernel(elems/32, 1, 1, 32, 1, 1, 0, 0, params); err != nil {
		t.Fatalf("LaunchKernel: %v", err)
	}
	if err := driver.CtxSynchronize(); err != nil {
		t.Fatalf("synchronize: %v", err)
	}
	got := make([]int32, elems)
	if err := out.MemcpyDtoH(unsafe.Pointer(&got[0]), nbytes); err != nil {
		t.Fatalf("MemcpyDtoH: %v", err)
	}
	for i := range got {
		want := int32(i*3 + bias)
		if os.Getenv("NVFATBIN_PHYSICAL_PERTURB") == "expected" && i == elems/2 {
			want++
		}
		if got[i] != want {
			t.Fatalf("out[%d] = %d, want %d", i, got[i], want)
		}
	}
	t.Logf("kernel loaded from the fatbin wrote the predicted i*3+%d for all %d elements", bias, elems)
}
