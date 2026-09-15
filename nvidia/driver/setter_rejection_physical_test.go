//go:build linux

package driver_test

import (
	"os"
	"runtime"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
)

// The runtime API was measured to reject out-of-range setter arguments with
// cudaErrorInvalidValue while leaving the prior value installed, so that a
// readback truthfully reports a configuration the caller does not believe is
// in effect. These tests ask the same question of the driver API, which is
// where these bindings live and which that measurement did not cover.
//
// Every case runs a positive control before the rejection: a valid set whose
// readback must change. Without it a readback that is constant for any reason
// would satisfy the "prior value survives" assertion for free.

// emptyKernelPTX is JIT-compiled by the driver, so these tests need no cubin
// and no toolkit. sm_52 is forward-compatible with every supported device.
const emptyKernelPTX = `
.version 6.0
.target sm_52
.address_size 64

.visible .entry empty()
{
	ret;
}
` + "\x00"

func physicalContext(t *testing.T) driver.Device {
	t.Helper()
	if os.Getenv("CUDA_DRIVER_HARDWARE_TEST") != "1" {
		t.Skip("set CUDA_DRIVER_HARDWARE_TEST=1 for native driver validation")
	}
	runtime.LockOSThread()
	t.Cleanup(runtime.UnlockOSThread)
	if err := driver.Init(0); err != nil {
		t.Fatal(err)
	}
	dev, err := driver.NewDeviceGet(0)
	if err != nil {
		t.Fatal(err)
	}
	ctx, err := dev.PrimaryCtxRetain()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { dev.PrimaryCtxRelease() })
	if err := ctx.SetCurrent(); err != nil {
		t.Fatal(err)
	}
	return dev
}

// TestNativeCtxSetLimitRejection records whether cuCtxSetLimit rejects an
// unknown limit and what the prior value does when it does.
func TestNativeCtxSetLimitRejection(t *testing.T) {
	physicalContext(t)

	const invalidLimit = driver.Limit(0x7f)

	base, err := driver.CtxGetLimit(driver.LimitPrintfFifoSize)
	if err != nil {
		t.Fatal(err)
	}

	// Positive control: a valid set must move the readback, or the assertion
	// below proves nothing.
	control := base * 2
	if err := driver.CtxSetLimit(driver.LimitPrintfFifoSize, control); err != nil {
		t.Fatalf("positive control rejected: %v", err)
	}
	got, err := driver.CtxGetLimit(driver.LimitPrintfFifoSize)
	if err != nil {
		t.Fatal(err)
	}
	if got == base {
		t.Fatalf("positive control did not move the readback: still %d; this instrument cannot detect a stale value", base)
	}
	t.Logf("positive control: PrintfFifoSize %d -> %d (requested %d)", base, got, control)
	installed := got

	if err := driver.CtxSetLimit(invalidLimit, 4096); err == nil {
		t.Errorf("CtxSetLimit(%v) accepted an unknown limit", invalidLimit)
	} else {
		t.Logf("CtxSetLimit(%v) rejected: %v", invalidLimit, err)
	}

	after, err := driver.CtxGetLimit(driver.LimitPrintfFifoSize)
	if err != nil {
		t.Fatal(err)
	}
	if after != installed {
		t.Errorf("after rejection PrintfFifoSize = %d, want the prior %d", after, installed)
	}
	t.Logf("after rejection: PrintfFifoSize reads back %d (prior value survived: %t)", after, after == installed)
}

// TestNativeCtxSetCacheConfigRejection does the same for cuCtxSetCacheConfig,
// whose argument is a small enum rather than a size.
func TestNativeCtxSetCacheConfigRejection(t *testing.T) {
	physicalContext(t)

	const invalidConfig = driver.FuncCacheConfig(99)

	base, err := driver.CtxGetCacheConfig()
	if err != nil {
		t.Fatal(err)
	}

	// Positive control. On devices with a fixed L1/shared split the driver may
	// legally ignore the preference, in which case this instrument cannot
	// answer the question and says so rather than asserting.
	control := driver.FuncCachePreferShared
	if base == control {
		control = driver.FuncCachePreferL1
	}
	if err := driver.CtxSetCacheConfig(control); err != nil {
		t.Fatalf("positive control rejected: %v", err)
	}
	installed, err := driver.CtxGetCacheConfig()
	if err != nil {
		t.Fatal(err)
	}
	if installed == base {
		t.Skipf("positive control did not move the readback (%v); cache config is advisory on this device and the rejection assertion would be vacuous", base)
	}
	t.Logf("positive control: cache config %v -> %v", base, installed)

	if err := driver.CtxSetCacheConfig(invalidConfig); err == nil {
		t.Errorf("CtxSetCacheConfig(%d) accepted an out-of-range config", invalidConfig)
	} else {
		t.Logf("CtxSetCacheConfig(%d) rejected: %v", invalidConfig, err)
	}

	after, err := driver.CtxGetCacheConfig()
	if err != nil {
		t.Fatal(err)
	}
	if after != installed {
		t.Errorf("after rejection cache config = %v, want the prior %v", after, installed)
	}
	t.Logf("after rejection: cache config reads back %v (prior value survived: %t)", after, after == installed)
}

// TestNativeFunctionSetAttributeRejection covers cuFuncSetAttribute, the
// driver-API counterpart of cudaFuncSetAttribute, which the runtime-API
// measurement found rejects while leaving the prior value installed.
func TestNativeFunctionSetAttributeRejection(t *testing.T) {
	dev := physicalContext(t)

	fn := loadEmptyKernel(t)

	optin, err := dev.GetAttribute(driver.DevAttrMaxSharedMemoryPerBlockOptin)
	if err != nil {
		t.Fatal(err)
	}

	base, err := fn.GetAttribute(driver.FuncAttributeMaxDynamicSharedSizeBytes)
	if err != nil {
		t.Fatal(err)
	}

	// Positive control: half the opt-in maximum is always a legal request.
	control := optin / 2
	if control == base {
		control = optin / 4
	}
	if err := fn.SetAttribute(driver.FuncAttributeMaxDynamicSharedSizeBytes, control); err != nil {
		t.Fatalf("positive control rejected: %v", err)
	}
	installed, err := fn.GetAttribute(driver.FuncAttributeMaxDynamicSharedSizeBytes)
	if err != nil {
		t.Fatal(err)
	}
	if installed == base {
		t.Fatalf("positive control did not move the readback: still %d; this instrument cannot detect a stale value", base)
	}
	t.Logf("positive control: MaxDynamicSharedSizeBytes %d -> %d (opt-in max %d)", base, installed, optin)

	// Beyond the opt-in maximum is the documented rejection.
	if err := fn.SetAttribute(driver.FuncAttributeMaxDynamicSharedSizeBytes, optin*4); err == nil {
		t.Errorf("Function.SetAttribute accepted %d bytes against an opt-in maximum of %d", optin*4, optin)
	} else {
		t.Logf("Function.SetAttribute(%d) rejected: %v", optin*4, err)
	}

	after, err := fn.GetAttribute(driver.FuncAttributeMaxDynamicSharedSizeBytes)
	if err != nil {
		t.Fatal(err)
	}
	if after != installed {
		t.Errorf("after rejection MaxDynamicSharedSizeBytes = %d, want the prior %d", after, installed)
	}
	t.Logf("after rejection: MaxDynamicSharedSizeBytes reads back %d (prior value survived: %t)", after, after == installed)

	// FuncAttributeNumRegs is read-only; setting it must fail.
	if err := fn.SetAttribute(driver.FuncAttributeNumRegs, 8); err == nil {
		t.Error("Function.SetAttribute accepted a write to the read-only NumRegs attribute")
	} else {
		t.Logf("Function.SetAttribute(NumRegs) rejected: %v", err)
	}
}

// TestNativeFunctionSetCacheConfigRejection covers cuFuncSetCacheConfig. It
// has no readback of its own, so the only observable is the return value.
func TestNativeFunctionSetCacheConfigRejection(t *testing.T) {
	physicalContext(t)

	fn := loadEmptyKernel(t)

	if err := fn.SetCacheConfig(driver.FuncCachePreferShared); err != nil {
		t.Fatalf("positive control rejected: %v", err)
	}
	t.Log("positive control: SetCacheConfig(PreferShared) accepted")

	const invalidConfig = driver.FuncCacheConfig(99)
	if err := fn.SetCacheConfig(invalidConfig); err == nil {
		t.Errorf("Function.SetCacheConfig(%d) accepted an out-of-range config", invalidConfig)
	} else {
		t.Logf("Function.SetCacheConfig(%d) rejected: %v", invalidConfig, err)
	}
	// There is no cuFuncGetCacheConfig, so a caller cannot check what is
	// installed. Record that gap; it is the reason this setter is riskier
	// than the ones above, not a defect in the test.
	t.Log("no readback exists for per-function cache config: the installed value is unobservable from the driver API")
}

func loadEmptyKernel(t *testing.T) driver.Function {
	t.Helper()
	src := []byte(emptyKernelPTX)
	mod, err := driver.NewModuleLoadData(unsafe.Pointer(&src[0]))
	if err != nil {
		t.Fatalf("loading PTX: %v", err)
	}
	t.Cleanup(func() { mod.Unload() })
	fn, err := mod.GetFunction("empty")
	if err != nil {
		t.Fatalf("GetFunction: %v", err)
	}
	return fn
}
