package nvvm_test

import (
	"crypto/sha256"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvjitlink"
	"github.com/tmc/lib/nvidia/nvvm"
)

// TestCompilerPipeline is opt-in so missing hardware cannot masquerade as a
// successful GPU test. Once requested, unavailable libraries are failures.
func TestCompilerPipeline(t *testing.T) {
	if os.Getenv("CUDA_COMPILER_TEST") != "1" {
		t.Skip("set CUDA_COMPILER_TEST=1 and CUDA_LIBDEVICE to run on NVIDIA hardware")
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	check := func(err error) {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
	}
	check(driver.Init(0))
	dev, err := driver.DeviceGet(0)
	check(err)
	major, err := dev.Attribute(driver.DevAttrComputeCapabilityMajor)
	check(err)
	minor, err := dev.Attribute(driver.DevAttrComputeCapabilityMinor)
	check(err)
	arch := fmt.Sprintf("%d%d", major, minor)
	t.Logf("target sm_%s", arch)
	irFile := "testdata/sin-typed.ll"
	if nvvm.Has("nvvmLLVMVersion") {
		llvm, err := nvvm.LLVMVersion("compute_" + arch)
		check(err)
		if llvm < 1 {
			t.Fatalf("invalid LLVM version %d", llvm)
		}
		t.Logf("NVVM LLVM version %d", llvm)
		irFile = testIR(int(llvm))
	}
	ctx, err := driver.NewContext(dev, 0)
	check(err)
	defer func() { check(ctx.Close()) }()

	ir, err := os.ReadFile(irFile)
	check(err)
	libdevice, err := os.ReadFile(os.Getenv("CUDA_LIBDEVICE"))
	check(err)
	t.Logf("ir sha256=%x libdevice sha256=%x", sha256.Sum256(ir), sha256.Sum256(libdevice))
	program, err := nvvm.CreateProgram()
	check(err)
	defer func() { check(program.Destroy()) }()
	check(program.AddModuleTo(unsafe.Pointer(unsafe.SliceData(ir)), uint(len(ir)), "sin.ll"))
	check(program.LazyAddModuleTo(unsafe.Pointer(unsafe.SliceData(libdevice)), uint(len(libdevice)), "libdevice.10.bc"))
	if err := program.Compile("-arch=compute_"+arch, "-gen-lto"); err != nil {
		log, _ := program.Log()
		t.Fatalf("compile: %v\n%s", err, log)
	}
	runtime.KeepAlive(libdevice)
	lto, err := program.CompiledResult()
	check(err)
	if len(lto) == 0 {
		t.Fatal("empty LTOIR")
	}
	linker, err := nvjitlink.Create("-arch=sm_"+arch, "-lto")
	check(err)
	defer func() { check(linker.Destroy()) }()
	check(linker.AddData(nvjitlink.InputLtoir, unsafe.Pointer(unsafe.SliceData(lto)), uint(len(lto)), "sin.ltoir"))
	if err := linker.Complete(); err != nil {
		log, _ := linker.ErrorLog()
		t.Fatalf("link: %v\n%s", err, log)
	}
	cubin, err := linker.LinkedCubin()
	check(err)
	if len(cubin) < 4 || string(cubin[:4]) != "\x7fELF" {
		t.Fatal("output is not a cubin ELF")
	}
	if dir := os.Getenv("CUDA_ARTIFACT_DIR"); dir != "" {
		check(os.MkdirAll(dir, 0755))
		check(os.WriteFile(filepath.Join(dir, "sin.ltoir"), lto, 0644))
		check(os.WriteFile(filepath.Join(dir, "sin.cubin"), cubin, 0644))
	}
	t.Logf("ltoir bytes=%d sha256=%x cubin bytes=%d sha256=%x", len(lto), sha256.Sum256(lto), len(cubin), sha256.Sum256(cubin))
	mod, err := driver.ModuleLoadData(cubin)
	check(err)
	defer func() { check(mod.Close()) }()
	kernel, err := mod.Function("sine")
	check(err)
	data := make([]float32, 128)
	for i := range data {
		data[i] = float32(i)/32 - 2
	}
	ptr, err := driver.MemAllocBytes(uint(len(data) * 4))
	check(err)
	defer func() { check(ptr.Free()) }()
	check(ptr.CopyFromHostPtr(unsafe.Pointer(unsafe.SliceData(data)), uint(len(data)*4)))
	check(kernel.LaunchKernel(driver.Dim3{X: 1, Y: 1, Z: 1}, driver.Dim3{X: uint32(len(data)), Y: 1, Z: 1}, 0, 0, []unsafe.Pointer{unsafe.Pointer(&ptr)}))
	check(driver.CtxSynchronize())
	check(ptr.CopyToHostPtr(unsafe.Pointer(unsafe.SliceData(data)), uint(len(data)*4)))
	for i, got := range data {
		want := math.Sin(float64(float32(i)/32 - 2))
		if math.IsNaN(float64(got)) || math.Abs(float64(got)-want) > 1e-6 {
			t.Fatalf("result[%d]=%g, want %g", i, got, want)
		}
	}
	t.Logf("verified %d sine results", len(data))

	// Tensor maps need compute capability 9.0; an older device reports
	// CUDA_ERROR_NOT_SUPPORTED. The step stays on this locked thread, where the
	// context is current, so it is logged as skipped rather than a subtest.
	if major < 9 {
		t.Logf("tensor-map step skipped: requires compute capability 9.0; device is %d.%d", major, minor)
	} else {
		// The descriptor API writes the full 128-byte object into aligned storage.
		descriptor := driver.NewTensorMap()
		dim := [2]uint64{32, 4}
		globalStride := uint64(128)
		box, stride := [2]uint32{32, 1}, [2]uint32{1, 1}
		check(driver.TensorMapEncodeTiled(descriptor, driver.TensorMapDataTypeFloat32, 2, ptr, &dim[0], &globalStride, &box[0], &stride[0], driver.TensorMapInterleaveNone, driver.TensorMapSwizzleNone, 0, 0))
		if *descriptor == (driver.TensorMap{}) {
			t.Fatal("tensor-map encode left a zero descriptor")
		}
		check(driver.TensorMapReplaceAddress(descriptor, ptr))
		t.Log("verified tensor-map encoding and address replacement")
	}
	invalid, err := nvvm.CreateProgram()
	check(err)
	defer func() { check(invalid.Destroy()) }()
	badIR := []byte("invalid NVVM IR")
	check(invalid.AddModuleTo(unsafe.Pointer(unsafe.SliceData(badIR)), uint(len(badIR)), "invalid.ll"))
	if err := invalid.Compile("-arch=compute_" + arch); err == nil {
		t.Fatal("compiler accepted invalid IR")
	}
	log, err := invalid.Log()
	check(err)
	if log == "" {
		t.Fatal("compiler omitted diagnostics for invalid IR")
	}
	t.Log("verified invalid-IR rejection and compiler diagnostics")
}

// testIR returns the test kernel in the IR dialect of an NVVM whose target
// reports llvm as its LLVM version. Before LLVM 17 NVVM parses typed pointers,
// which is every target of CUDA 12 and pre-Blackwell targets of CUDA 13; later
// targets parse opaque pointers. An NVVM too old to report a version takes the
// typed form.
func testIR(llvm int) string {
	if llvm >= 17 {
		return "testdata/sin.ll"
	}
	return "testdata/sin-typed.ll"
}
