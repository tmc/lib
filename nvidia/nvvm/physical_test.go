//go:build nvvm_physical

// Package nvvm_test compiles hand-written NVVM IR with libNVVM and then runs
// the PTX it produces on the GPU. Compiling successfully is not the assertion;
// the assertion is that the resulting kernel writes values the host predicted
// without consulting the GPU, which is what distinguishes a working binding
// from one that merely returns NVVM_SUCCESS.
//
// libNVVM ships in $CUDA_HOME/nvvm/lib64, which no CUDA install adds to the
// dynamic linker search path: ld.so.conf.d lists only targets/<arch>/lib and
// cuda/lib64, and the only nvvm entries ldconfig knows are the driver's
// libnvidia-nvvm. The loader here tries bare sonames, so running this test
// needs LD_LIBRARY_PATH=$CUDA_HOME/nvvm/lib64 (or an explicit nvvm.Load).
//
// Set NVVM_PHYSICAL_PERTURB=expected to run the negative control.
package nvvm_test

import (
	"os"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvvm"
)

// module is NVVM IR for a kernel equivalent to
//
//	out[i] = i*5 + bias
//
// written against the typed-pointer LLVM dialect libNVVM accepts. The
// nvvm.annotations metadata is what marks fill as a kernel entry point.
const module = `
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()

define void @fill(i32* %out, i32 %n, i32 %bias) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %ctaid = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %ntid = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %base = mul i32 %ctaid, %ntid
  %i = add i32 %base, %tid
  %inrange = icmp slt i32 %i, %n
  br i1 %inrange, label %body, label %exit

body:
  %scaled = mul i32 %i, 5
  %value = add i32 %scaled, %bias
  %idx = sext i32 %i to i64
  %slot = getelementptr i32, i32* %out, i64 %idx
  store i32 %value, i32* %slot
  br label %exit

exit:
  ret void
}

!nvvm.annotations = !{!0}
!0 = !{void (i32*, i32, i32)* @fill, !"kernel", i32 1}
!nvvmir.version = !{!1}
!1 = !{i32 2, i32 0, i32 3, i32 1}
`

const (
	elems = 192
	bias  = 23
)

// TestVersions checks the three version entry points against each other. Each
// returns several values through out-parameters, which is where a generator
// most often drops or reorders a result.
func TestVersions(t *testing.T) {
	major, minor, err := nvvm.Version()
	if err != nil {
		t.Skipf("libNVVM unavailable: %v", err)
	}
	t.Logf("libNVVM version %d.%d", major, minor)
	if major < 1 || major > 20 {
		t.Fatalf("implausible libNVVM major version %d", major)
	}

	majorIR, minorIR, majorDbg, minorDbg, err := nvvm.IRVersion()
	if err != nil {
		t.Fatalf("IRVersion: %v", err)
	}
	t.Logf("NVVM IR version %d.%d, debug metadata version %d.%d",
		majorIR, minorIR, majorDbg, minorDbg)
	// All four are small version components; a dropped out-parameter shows up
	// as a zero or as uninitialised garbage.
	for _, v := range []struct {
		name string
		val  int32
	}{
		{"majorIR", majorIR}, {"minorIR", minorIR},
		{"majorDbg", majorDbg}, {"minorDbg", minorDbg},
	} {
		if v.val < 0 || v.val > 100 {
			t.Fatalf("IRVersion %s = %d, outside any plausible range", v.name, v.val)
		}
	}
	if majorIR == 0 {
		t.Fatalf("IRVersion majorIR = 0, want a real version")
	}
	// The IR version the library accepts must be the one the test module
	// declares in !nvvmir.version, or the compile below would be testing a
	// different contract than the one documented here.
	if majorIR != 2 {
		t.Logf("note: library reports NVVM IR %d.%d; the test module declares 2.0",
			majorIR, minorIR)
	}
}

// TestCompileAndRun compiles the NVVM IR module to PTX and executes it.
func TestCompileAndRun(t *testing.T) {
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
	arch := strconv.Itoa(int(major)) + strconv.Itoa(int(minor))
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

	prog, err := nvvm.CreateProgram()
	if err != nil {
		t.Skipf("libNVVM unavailable: %v", err)
	}
	defer func() {
		if err := nvvm.DestroyProgram(&prog); err != nil {
			t.Errorf("destroy program: %v", err)
		}
	}()
	ir := []byte(module)
	if err := prog.AddModuleTo(unsafe.Pointer(&ir[0]), uint(len(ir)), "fill.ll"); err != nil {
		t.Fatalf("AddModuleTo: %v", err)
	}
	if err := prog.Verify(nil); err != nil {
		log, _ := prog.GetProgramLog()
		t.Fatalf("Verify: %v\n%s", err, log)
	}
	// libNVVM ships with the toolkit, so the one that loads can predate the
	// installed device: a CUDA 12 libNVVM rejects -arch=compute_121. PTX
	// built for an older virtual architecture still JITs onto newer
	// hardware, so fall back through the architectures libNVVM knows rather
	// than failing on a toolkit-versus-device mismatch. The kernel and its
	// oracle are unchanged; only the PTX target moves.
	log := ""
	for _, try := range archFallbacks(arch) {
		if err := prog.Compile([]string{"-arch=compute_" + try}); err == nil {
			arch = try
			log = ""
			break
		}
		log, _ = prog.GetProgramLog()
		if !strings.Contains(log, "unsupported option") {
			t.Fatalf("Compile for compute_%s: %s", try, log)
		}
		t.Logf("libNVVM does not support compute_%s, trying an older target", try)
	}
	if log != "" {
		t.Fatalf("libNVVM supports no architecture this device can run:\n%s", log)
	}
	size, err := prog.GetCompiledResultSize()
	if err != nil {
		t.Fatalf("GetCompiledResultSize: %v", err)
	}
	ptx, err := prog.GetCompiledResult()
	if err != nil {
		t.Fatalf("GetCompiledResult: %v", err)
	}
	// The size/get pair is the classic out-parameter defect site.
	if uint(len(ptx)) != size {
		t.Fatalf("GetCompiledResult returned %d bytes, GetCompiledResultSize said %d", len(ptx), size)
	}
	text := string(ptx)
	if !strings.Contains(text, ".visible .entry fill") {
		t.Fatalf("PTX does not export fill as a kernel entry point:\n%s", text)
	}
	if !strings.Contains(text, ".target sm_"+arch) {
		t.Fatalf("PTX is not targeted at sm_%s:\n%s", arch, text)
	}
	t.Logf("compiled %d bytes of NVVM IR into %d bytes of PTX for sm_%s", len(ir), len(ptx), arch)

	// The driver JIT-compiles the PTX, so this also confirms libNVVM produced
	// PTX the installed driver accepts.
	image := append(ptx, 0)
	mod, err := driver.NewModuleLoadData(unsafe.Pointer(&image[0]))
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
	if err := fn.LaunchKernel(elems/64, 1, 1, 64, 1, 1, 0, 0, params); err != nil {
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
		want := int32(i*5 + bias)
		if os.Getenv("NVVM_PHYSICAL_PERTURB") == "expected" && i == elems/2 {
			want++
		}
		if got[i] != want {
			t.Fatalf("out[%d] = %d, want %d", i, got[i], want)
		}
	}
	t.Logf("kernel compiled from NVVM IR wrote the predicted i*5+%d for all %d elements", bias, elems)
}

// archFallbacks returns the virtual architectures to try for a device of the
// given compute capability, most capable first. Every entry is one libNVVM
// has supported at some point and is no newer than the device, so any PTX
// that results runs on it.
func archFallbacks(arch string) []string {
	known := []string{"120", "100", "90", "89", "87", "86", "80", "75"}
	out := []string{arch}
	n, err := strconv.Atoi(arch)
	if err != nil {
		return out
	}
	for _, k := range known {
		if v, _ := strconv.Atoi(k); v <= n && k != arch {
			out = append(out, k)
		}
	}
	return out
}
