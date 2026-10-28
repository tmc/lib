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
		t.Skip("requires an inherited GPU lease; run under gpu-lock")
	}
	if err := withAddressSpaceNativeLease(os.Getenv, func() {
		testAddressSpaceInferenceABAdmitted(t, opt)
	}); err != nil {
		t.Fatalf("native address-space test requires an inherited GPU lease: %v", err)
	}
}

func testAddressSpaceInferenceABAdmitted(t *testing.T, opt int) {
	t.Helper()
	rowKind := "production-opt3"
	if opt == 0 {
		rowKind = "mechanism-opt0"
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

	input := twoSlotAddressSpaceABInput()
	rewritten, inference, err := nvvmx.InferAddressSpaces(input)
	if err != nil {
		t.Fatal(err)
	}
	provenance, err := verifyTwoSlotInferenceOrigins(input, rewritten, inference)
	if err != nil {
		t.Fatal(err)
	}

	arch := fmt.Sprintf("%d%d", major, minor)
	beforeOpts := []nvvmx.Option{nvvmx.WithArch("compute_" + arch), nvvmx.WithOpt(opt)}
	t.Logf("row=%s arm=before input=before.ll orig_ir_sha256=%s arch=compute_%s opt=%d infer=false",
		rowKind, provenance.OriginalSHA256, arch, opt)
	beforeResult, err := nvvmx.CompileWithOptions([]nvvmx.Input{{Name: "before.ll", Data: input}}, beforeOpts...)
	if err != nil {
		t.Fatalf("compile before: %v", err)
	}
	defer func() {
		if err := beforeResult.Close(); err != nil {
			t.Errorf("close before result: %v", err)
		}
	}()
	beforePTX := append([]byte(nil), beforeResult.PTX...)
	if err := beforeResult.Close(); err != nil {
		t.Fatalf("close before result: %v", err)
	}

	afterOpts := []nvvmx.Option{nvvmx.WithArch("compute_" + arch), nvvmx.WithOpt(opt), nvvmx.WithAddressSpaceInference()}
	t.Logf("row=%s arm=after input=after.ll orig_ir_sha256=%s prep_ir_sha256=%s arch=compute_%s opt=%d infer=true",
		rowKind, provenance.OriginalSHA256, provenance.PreparedSHA256, arch, opt)
	afterResult, err := nvvmx.CompileWithOptions([]nvvmx.Input{{Name: "after.ll", Data: input}}, afterOpts...)
	if err != nil {
		t.Fatalf("compile after: %v", err)
	}
	defer func() {
		if err := afterResult.Close(); err != nil {
			t.Errorf("close after result: %v", err)
		}
	}()
	afterPTX := append([]byte(nil), afterResult.PTX...)

	work := t.TempDir()
	disassemble := func(name string, ptx []byte) ([]byte, []byte) {
		ptxPath := filepath.Join(work, name+".ptx")
		cubinPath := filepath.Join(work, name+".cubin")
		if err := os.WriteFile(ptxPath, ptx, 0o600); err != nil {
			t.Fatal(err)
		}
		runTool(t, ptxas, "-arch=sm_"+arch, ptxPath, "-o", cubinPath)
		sass := runTool(t, nvdisasm, "-c", cubinPath)
		cubin, err := os.ReadFile(cubinPath)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := cudaptx.Parse(bytes.NewReader(ptx)); err != nil {
			t.Fatalf("parse %s PTX: %v", name, err)
		}
		if _, err := cudasass.Parse(bytes.NewReader(sass)); err != nil {
			t.Fatalf("parse %s SASS: %v", name, err)
		}
		t.Logf("row=%s arm=%s opt=%d ptxas=%s nvdisasm=%s ptx_sha256=%x cubin_sha256=%x sass_sha256=%x",
			rowKind, name, opt, ptxas, nvdisasm, sha256.Sum256(ptx), sha256.Sum256(cubin), sha256.Sum256(sass))
		t.Logf("%s PTX:\n%s\n%s SASS:\n%s", name, ptx, name, sass)
		return sass, cubin
	}
	beforeSASS, beforeCUBIN := disassemble("before", beforePTX)
	afterSASS, afterCUBIN := disassemble("after", afterPTX)
	if bytes.Equal(beforePTX, afterPTX) || bytes.Equal(beforeCUBIN, afterCUBIN) {
		t.Fatalf("A/B compilation produced identical PTX or CUBIN artifacts (row=%s opt=%d)", rowKind, opt)
	}

	// Before arm compiled without inference; evaluate per-slot provenance from
	// the original two-slot IR origins and actual before-arm PTX/SASS without
	// claiming compile custody on the uninferred result.
	without, err := VerifyInferredSharedConversions(input, inference, beforePTX, beforeSASS)
	if err != nil {
		t.Fatalf("before-arm per-slot provenance failed (row=%s opt=%d): %v\n%s", rowKind, opt, err, without)
	}
	if err := verifyTwoSlotBeforeReport(without, provenance.OriginalSHA256, provenance.PreparedSHA256, sha256Hex(beforePTX), sha256Hex(beforeSASS)); err != nil {
		t.Fatalf("before-arm per-slot report invalid (row=%s opt=%d): %v\n%s", rowKind, opt, err, without)
	}

	// Retain afterResult until VerifyCompiledSharedConversions has verified its
	// copied PTX and the actual nvdisasm listing for afterCUBIN.
	compiledProv, err := afterResult.AddressSpaceProvenance()
	if err != nil {
		t.Fatalf("after-arm compile provenance unavailable: %v", err)
	}
	if len(compiledProv.Inputs) != 1 || compiledProv.Inputs[0].Name != "after.ll" ||
		compiledProv.PTXSHA256 != sha256Hex(afterPTX) ||
		compiledProv.Inputs[0].Provenance.OriginalSHA256 != provenance.OriginalSHA256 ||
		compiledProv.Inputs[0].Provenance.PreparedSHA256 != provenance.PreparedSHA256 {
		t.Fatalf("after-arm compile provenance binding mismatch: %+v", compiledProv)
	}
	with, err := VerifyCompiledSharedConversions(afterResult, afterSASS)
	if err != nil {
		t.Fatalf("after-arm compiled per-slot provenance failed (row=%s opt=%d): %v\n%s", rowKind, opt, err, with)
	}
	if err := verifyTwoSlotAfterReport(with, provenance.OriginalSHA256, provenance.PreparedSHA256, sha256Hex(afterPTX), sha256Hex(afterSASS)); err != nil {
		t.Fatalf("after-arm per-slot report invalid (row=%s opt=%d): %v\n%s", rowKind, opt, err, with)
	}
	if err := afterResult.Close(); err != nil {
		t.Fatalf("close after result: %v", err)
	}

	beforeValue, afterValue := launchAddressSpaceKernel(t, beforeCUBIN), launchAddressSpaceKernel(t, afterCUBIN)
	if beforeValue != afterValue {
		t.Fatalf("A/B numerical output mismatch: before=%d after=%d", beforeValue, afterValue)
	}
	if err := verifyTwoSlotAddressSpaceOutput(twoSlotAddressSpaceSeed, beforeValue); err != nil {
		t.Fatalf("before A/B output failed two-slot oracle: %v", err)
	}
	if err := verifyTwoSlotAddressSpaceOutput(twoSlotAddressSpaceSeed, afterValue); err != nil {
		t.Fatalf("after A/B output failed two-slot oracle: %v", err)
	}
	t.Logf("row=%s NVVM opt=%d A/B passed on sm_%s; slots=[%s,%s]; orig_ir_sha256=%s prep_ir_sha256=%s; output before=%d after=%d want=%d",
		rowKind, opt, arch, twoSlotSlotA, twoSlotSlotB, provenance.OriginalSHA256, provenance.PreparedSHA256, beforeValue, afterValue, twoSlotExpectedOutput(twoSlotAddressSpaceSeed))
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
	input := twoSlotAddressSpaceSeed
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
