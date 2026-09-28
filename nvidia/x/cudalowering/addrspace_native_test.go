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

func TestLowering_AddressSpaceInferenceAB(t *testing.T) {
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

	input := addressSpaceABInput()
	inferred, _, err := nvvmx.InferAddressSpaces(input)
	if err != nil {
		t.Fatal(err)
	}
	arch := fmt.Sprintf("%d%d", major, minor)
	compilePTX := func(name string, source []byte) []byte {
		r, err := nvvmx.CompileWithOptions([]nvvmx.Input{{Name: name, Data: source}}, nvvmx.WithArch("compute_"+arch))
		if err != nil {
			t.Fatalf("compile %s: %v", name, err)
		}
		defer r.Close()
		return append([]byte(nil), r.PTX...)
	}
	before := compilePTX("before", input)
	after := compilePTX("after", inferred)

	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(home, "tmp"), 0o700); err != nil {
		t.Fatal(err)
	}
	work, err := os.MkdirTemp(filepath.Join(home, "tmp"), "nvvmx-addrspace-")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(work)
	disassemble := func(name string, ptx []byte) (*cudaptx.Module, *cudasass.Disassembly) {
		ptxPath := filepath.Join(work, name+".ptx")
		cubinPath := filepath.Join(work, name+".cubin")
		if err := os.WriteFile(ptxPath, ptx, 0o600); err != nil {
			t.Fatal(err)
		}
		runTool(t, ptxas, "-arch=sm_"+arch, ptxPath, "-o", cubinPath)
		out := runTool(t, nvdisasm, "-c", cubinPath)
		mod, err := cudaptx.Parse(bytes.NewReader(ptx))
		if err != nil {
			t.Fatalf("parse %s PTX: %v", name, err)
		}
		dis, err := cudasass.Parse(bytes.NewReader(out))
		if err != nil {
			t.Fatalf("parse %s SASS: %v", name, err)
		}
		t.Logf("%s PTX sha256=%x SASS sha256=%x", name, sha256.Sum256(ptx), sha256.Sum256(out))
		return mod, dis
	}
	beforePTX, beforeSASS := disassemble("before", before)
	afterPTX, afterSASS := disassemble("after", after)
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
	if got, want := launchAddressSpaceKernel(t, before), launchAddressSpaceKernel(t, after); got != want {
		t.Fatalf("A/B output differs: before=%d after=%d", got, want)
	}
	t.Logf("NVVM A/B passed on sm_%s; input sha256=%x", arch, sha256.Sum256(input))
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

func launchAddressSpaceKernel(t *testing.T, ptx []byte) uint32 {
	t.Helper()
	image := append(append([]byte(nil), ptx...), 0)
	module, err := driver.ModuleLoadData(image)
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
