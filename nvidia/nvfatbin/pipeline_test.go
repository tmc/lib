package nvfatbin_test

import (
	"crypto/sha256"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvfatbin"
)

// TestFatbinPipeline packages PTX into a fatbinary, loads it with the CUDA
// driver and checks the kernel's output.
//
// It is opt-in so that absent hardware cannot masquerade as success. Once
// NVFATBIN_PHYSICAL_TEST=1 is set, every missing prerequisite is a failure and
// nothing below skips.
//
// Prerequisites, all of them:
//   - libnvfatbin.so.13 on the loader path, or NVFATBIN_LIB_PATH pointing at
//     it. This one is host-side: it needs no GPU.
//   - libcuda.so.1 from an installed NVIDIA driver, and a visible GPU.
//   - a GPU whose compute capability the installed ptxas accepts. The driver
//     JIT-compiles the packaged PTX at module load, so the driver's PTX
//     version support, not just the device, has to cover .version 8.8.
func TestFatbinPipeline(t *testing.T) {
	if os.Getenv("NVFATBIN_PHYSICAL_TEST") != "1" {
		t.Skip("set NVFATBIN_PHYSICAL_TEST=1 to run against a real libnvfatbin and NVIDIA GPU")
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	check := func(err error) {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
	}

	major, minor, err := nvfatbin.Version()
	check(err)
	t.Logf("nvFatbin version %d.%d from %s", major, minor, nvfatbin.Path())

	check(driver.Init(0))
	dev, err := driver.NewDeviceGet(0)
	check(err)
	ccMajor, err := dev.GetAttribute(driver.DevAttrComputeCapabilityMajor)
	check(err)
	ccMinor, err := dev.GetAttribute(driver.DevAttrComputeCapabilityMinor)
	check(err)
	arch := fmt.Sprintf("%d%d", ccMajor, ccMinor)
	t.Logf("target sm_%s", arch)
	ctx, err := driver.NewContext(nil, 0, dev)
	check(err)
	defer func() { check(ctx.Destroy()) }()

	// The pinned PTX names sm_90 so that it assembles on its own; retarget it
	// at the device actually present rather than requiring one.
	source, err := os.ReadFile("testdata/addone.ptx")
	check(err)
	ptx := strings.Replace(string(source), ".target sm_90", ".target sm_"+arch, 1)
	if !strings.Contains(ptx, ".target sm_"+arch) {
		t.Fatalf("could not retarget testdata/addone.ptx at sm_%s", arch)
	}

	handle, err := nvfatbin.Create([]string{"-64", "-cuda"})
	check(err)
	defer func() { check(nvfatbin.Destroy(&handle)) }()
	check(handle.AddPTXString(ptx, arch, "addone.ptx", ""))

	size, err := handle.Size()
	check(err)
	if size == 0 {
		t.Fatal("nvFatbinSize reported 0 bytes after adding PTX")
	}
	fatbin, err := handle.Get()
	check(err)
	if uint(len(fatbin)) != size {
		t.Fatalf("Get returned %d bytes, Size reported %d", len(fatbin), size)
	}
	t.Logf("fatbinary bytes=%d sha256=%x", len(fatbin), sha256.Sum256(fatbin))
	if dir := os.Getenv("NVFATBIN_ARTIFACT_DIR"); dir != "" {
		check(os.MkdirAll(dir, 0o755))
		check(os.WriteFile(filepath.Join(dir, "addone.fatbin"), fatbin, 0o644))
	}

	// Loading the fatbinary with the driver is what distinguishes a correct
	// package from a nonempty one. A length assertion alone would pass on
	// bytes the driver cannot read.
	mod, err := driver.NewModuleLoadData(unsafe.Pointer(unsafe.SliceData(fatbin)))
	check(err)
	defer func() { check(mod.Unload()) }()
	runtime.KeepAlive(fatbin)
	kernel, err := mod.GetFunction("addone")
	check(err)

	data := make([]float32, 128)
	for i := range data {
		data[i] = float32(i)
	}
	ptr, err := driver.MemAlloc(uint(len(data) * 4))
	check(err)
	defer func() { check(ptr.MemFree()) }()
	check(ptr.MemcpyHtoD(unsafe.Pointer(unsafe.SliceData(data)), uint(len(data)*4)))
	check(kernel.LaunchKernel(1, 1, 1, uint32(len(data)), 1, 1, 0, 0, []unsafe.Pointer{unsafe.Pointer(&ptr)}))
	check(driver.CtxSynchronize())
	check(ptr.MemcpyDtoH(unsafe.Pointer(unsafe.SliceData(data)), uint(len(data)*4)))
	for i, got := range data {
		if want := float32(i) + 1; got != want {
			t.Fatalf("result[%d]=%g, want %g", i, got, want)
		}
	}
	t.Logf("verified %d results from the packaged kernel", len(data))

	// Packaging PTX does not validate its instructions. Confirm that the
	// driver rejects a packaged kernel with an invalid instruction.
	bad, err := nvfatbin.Create([]string{"-64", "-cuda"})
	check(err)
	defer func() { check(nvfatbin.Destroy(&bad)) }()
	invalid := strings.Replace(ptx, "add.f32", "invalid_instruction.f32", 1)
	check(bad.AddPTXString(invalid, arch, "invalid.ptx", ""))
	badFatbin, err := bad.Get()
	check(err)
	badModule, err := driver.NewModuleLoadData(unsafe.Pointer(unsafe.SliceData(badFatbin)))
	runtime.KeepAlive(badFatbin)
	if err == nil {
		check(badModule.Unload())
		t.Fatal("driver accepted PTX with an invalid instruction")
	}
	t.Logf("driver rejected invalid PTX: %v", err)
}

// TestDestroyZeroesHandle checks the C contract that nvFatbinDestroy sets the
// caller's handle to nullptr, so a use-after-destroy is a zero Handle rather
// than a dangling one.
//
// Same gate, and for the same reason: it calls into the real library.
func TestDestroyZeroesHandle(t *testing.T) {
	if os.Getenv("NVFATBIN_PHYSICAL_TEST") != "1" {
		t.Skip("set NVFATBIN_PHYSICAL_TEST=1 to run against a real libnvfatbin")
	}
	handle, err := nvfatbin.Create(nil)
	if err != nil {
		t.Fatal(err)
	}
	if handle == 0 {
		t.Fatal("Create returned a zero handle with no error")
	}
	if err := nvfatbin.Destroy(&handle); err != nil {
		t.Fatal(err)
	}
	if handle != 0 {
		t.Errorf("handle = %#x after Destroy, want 0", uintptr(handle))
	}
}

// TestUnrecognizedOption checks that a bad option reaches the caller as a
// typed result rather than as a panic or a silent success. It needs
// libnvfatbin but no GPU, so it is gated separately: a host with the CUDA
// libraries and no device can still run it.
func TestUnrecognizedOption(t *testing.T) {
	if os.Getenv("NVFATBIN_LIBRARY_TEST") != "1" {
		t.Skip("set NVFATBIN_LIBRARY_TEST=1 to run against a real libnvfatbin (no GPU needed)")
	}
	handle, err := nvfatbin.Create([]string{"-not-an-option"})
	if err == nil {
		nvfatbin.Destroy(&handle)
		t.Fatal("Create accepted an unrecognized option")
	}
	var code nvfatbin.Result
	if !errors.As(err, &code) {
		t.Fatalf("Create error = %v (%T), want nvfatbin.Result", err, err)
	}
	if code != nvfatbin.ErrUnrecognizedOption {
		t.Errorf("Create error = %v, want ErrUnrecognizedOption", code)
	}
}
