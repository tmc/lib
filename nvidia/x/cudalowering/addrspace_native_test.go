//go:build native

package cudalowering

import (
	"bytes"
	"crypto/sha256"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudaptx"
	"github.com/tmc/lib/nvidia/x/cudasass"
	"github.com/tmc/lib/nvidia/x/nvvmx"
)

// The production optimization gate must retain a nonvacuous native control.
func TestLowering_AddressSpaceInferenceAB(t *testing.T) {
	testAddressSpaceInferenceAB(t, 3)
}

// Opt0 exposes an observed conversion mechanism; it does not qualify the
// production optimized A/B gate or a performance benefit.
func TestLowering_AddressSpaceInferenceMechanismOpt0(t *testing.T) {
	testAddressSpaceInferenceAB(t, 0)
}

func testAddressSpaceInferenceAB(t *testing.T, opt int) {
	t.Helper()
	if os.Getenv("GPU_LOCK_HELD") == "" {
		t.Skip("requires GPU_LOCK_HELD; run under gpu-lock")
	}
	ptxas, nvdisasm := cudaTool(t, "ptxas"), cudaTool(t, "nvdisasm")
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if err := driver.Init(0); err != nil {
		t.Skipf("CUDA driver unavailable: %v", err)
	}
	dev, err := driver.DeviceGet(0)
	if err != nil {
		t.Skipf("CUDA device unavailable: %v", err)
	}
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		t.Fatal(err)
	}
	ctx, err := driver.NewContext(dev, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := ctx.Close(); err != nil {
			t.Error(err)
		}
	}()

	// The fixture uses opaque pointers, requiring NVVM IR 2.0. Declare the
	// same metadata contract as the native nvvm compilation fixture.
	input := append(addressSpaceABInput(), []byte("!nvvmir.version = !{!1}\n!1 = !{i32 2, i32 0, i32 3, i32 2}\n")...)
	input = bytes.Replace(input, []byte("e-p:64:64:64-p3:32:32:32"), []byte("e-p:64:64:64-p3:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64-a:8:8"), 1)
	inferred, _, err := nvvmx.InferAddressSpaces(input)
	if err != nil {
		t.Fatal(err)
	}
	arch := fmt.Sprintf("%d%d", major, minor)
	compilePTX := func(name string, source []byte) []byte {
		t.Logf("%s exact NVVM input sha256=%x opt=%d", name, sha256.Sum256(source), opt)
		r, err := nvvmx.CompileWithOptions([]nvvmx.Input{{Name: name, Data: source}}, nvvmx.WithArch("compute_"+arch), nvvmx.WithOpt(opt))
		if err != nil {
			t.Fatalf("compile %s: %v", name, err)
		}
		defer r.Close()
		return append([]byte(nil), r.PTX...)
	}
	before := compilePTX("before", input)
	after := compilePTX("after", inferred)

	work := t.TempDir()
	disassemble := func(name string, ptx []byte) (*cudaptx.Module, *cudasass.Disassembly, []byte) {
		ptxPath := filepath.Join(work, name+".ptx")
		cubinPath := filepath.Join(work, name+".cubin")
		if err := os.WriteFile(ptxPath, ptx, 0o600); err != nil {
			t.Fatal(err)
		}
		runTool(t, ptxas, "-arch=sm_"+arch, ptxPath, "-o", cubinPath)
		out := runTool(t, nvdisasm, "-c", cubinPath)
		cubin, err := os.ReadFile(cubinPath)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("%s actual CUBIN sha256=%x", name, sha256.Sum256(cubin))
		mod, err := cudaptx.Parse(bytes.NewReader(ptx))
		if err != nil {
			t.Fatalf("parse %s PTX: %v", name, err)
		}
		dis, err := cudasass.Parse(bytes.NewReader(out))
		if err != nil {
			t.Fatalf("parse %s SASS: %v", name, err)
		}
		t.Logf("%s NVVM opt=%d PTX sha256=%x SASS sha256=%x", name, opt, sha256.Sum256(ptx), sha256.Sum256(out))
		t.Logf("%s PTX:\n%s\n%s SASS:\n%s", name, ptx, name, out)
		return mod, dis, cubin
	}
	beforePTX, beforeSASS, beforeCUBIN := disassemble("before", before)
	afterPTX, afterSASS, afterCUBIN := disassemble("after", after)
	without, err := VerifyNoSharedConversions(beforePTX, beforeSASS, "@kernel/%slot")
	if err != nil {
		t.Fatal(err)
	}
	if !hasSharedConversion(without) {
		t.Fatalf("uninferred control contains no detected shared conversion:\n%s", without)
	}
	if !hasSASSSharedConversion(without) {
		t.Fatalf("uninferred control has no recognized SASS conversion for @kernel/%%slot:\n%s", without)
	}
	with, err := VerifyNoSharedConversions(afterPTX, afterSASS, "@kernel/%slot")
	if err != nil {
		t.Fatal(err)
	}
	if hasSharedConversion(with) {
		t.Fatalf("inferred output still contains a shared conversion:\n%s", with)
	}
	beforeValue, afterValue := launchAddressSpaceKernel(t, beforeCUBIN), launchAddressSpaceKernel(t, afterCUBIN)
	if beforeValue != 42 || afterValue != 42 {
		t.Fatalf("A/B output: before=%d after=%d want both42", beforeValue, afterValue)
	}
	// Unique slot attribution is confined to this exact single-slot fixture;
	// caller labels supplied to Compare are not a general mapping proof.
	t.Logf("NVVM opt=%d A/B passed on sm_%s; exact single-slot input sha256=%x; output before=%d after=%d", opt, arch, sha256.Sum256(input), beforeValue, afterValue)
}

func cudaTool(t *testing.T, name string) string {
	t.Helper()
	if root := os.Getenv("CUDA"); root != "" {
		path := filepath.Join(root, "bin", name)
		if _, err := os.Stat(path); err == nil {
			return path
		}
	}
	path, err := exec.LookPath(name)
	if err != nil {
		t.Skipf("CUDA %s unavailable", name)
	}
	return path
}

func runTool(t *testing.T, name string, args ...string) []byte {
	t.Helper()
	cmd := exec.Command(name, args...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("%s %s: %v\n%s", name, strings.Join(args, " "), err, out)
	}
	return out
}

func hasSharedConversion(report *Report) bool {
	for _, div := range report.Divergences {
		if div.Kind == SharedConversion && (div.PTXCount > 0 || div.SASSCount > 0) {
			return true
		}
	}
	return false
}

func hasSASSSharedConversion(report *Report) bool {
	for _, div := range report.Divergences {
		if div.Kind == SharedConversion && div.SASSCount > 0 {
			return true
		}
	}
	return false
}

func launchAddressSpaceKernel(t *testing.T, cubin []byte) uint32 {
	t.Helper()
	t.Logf("loaded numerical CUBIN sha256=%x", sha256.Sum256(cubin))
	module, err := driver.ModuleLoadData(cubin)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := module.Close(); err != nil {
			t.Error(err)
		}
	}()
	fn, err := module.Function("kernel")
	if err != nil {
		t.Fatal(err)
	}
	p, err := driver.MemAllocBytes(4)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := p.Free(); err != nil {
			t.Error(err)
		}
	}()
	input := uint32(35)
	if err := fn.LaunchKernel(driver.Dim3{X: 1, Y: 1, Z: 1}, driver.Dim3{X: 1, Y: 1, Z: 1}, 0, 0,
		[]unsafe.Pointer{unsafe.Pointer(&p), unsafe.Pointer(&input)}); err != nil {
		t.Fatal(err)
	}
	if err := driver.CtxSynchronize(); err != nil {
		t.Fatal(err)
	}
	var output uint32
	if err := p.CopyToHostPtr(unsafe.Pointer(&output), 4); err != nil {
		t.Fatal(err)
	}
	return output
}
