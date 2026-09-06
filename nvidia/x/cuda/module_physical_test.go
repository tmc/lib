package cuda

import (
	"errors"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
)

// TestModuleOwnership needs a real device. It used to be gated on
// CUDA_MODULE_TEST=1, whose skip message read "requires a CUDA device" -- which
// is a probe an env var cannot perform, and the reason it was never run by
// default even on a machine that has one. It probes instead.
//
// This file is in package cuda, and cudatest imports cuda, so the shared
// helper is unreachable from here; the guard is inlined the same way the
// lifetime tests in this package inline theirs. Loading the driver is what
// separates absent from present, and a failure past that point is real.
func TestModuleOwnership(t *testing.T) {
	if err := driver.Load(""); err != nil {
		t.Skipf("skipping test: CUDA driver not available: %v", err)
	}
	if err := Init(); err != nil {
		t.Fatalf("cuInit failed with libcuda loaded from %q: %v", driver.Path(), err)
	}
	count, err := DeviceCount()
	if err != nil {
		t.Fatalf("cuDeviceGetCount failed after cuInit succeeded: %v", err)
	}
	if count == 0 {
		t.Skip("skipping test: no CUDA devices available")
	}
	check := func(err error) {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
	}
	dev, err := NewDevice(0)
	check(err)
	// The kernel below is PTX with an explicit .target, so it will not load on
	// a device older than that. This is a property of the fixture rather than
	// of the code under test, and skipping names it.
	if major, minor, err := dev.ComputeCapability(); err != nil {
		t.Fatal(err)
	} else if major < 9 {
		_ = dev.Close()
		t.Skipf("skipping test: embedded PTX targets sm_90, device is sm_%d%d", major, minor)
	}
	defer func() {
		if err := dev.Close(); err != nil {
			t.Error(err)
		}
	}()
	image := []byte(`.version 8.0
.target sm_90
.address_size 64
.visible .entry set_value(.param .u64 output) {
.reg .u64 %p;
ld.param.u64 %p, [output];
st.global.u32 [%p], 7;
ret;
}` + "\x00")
	var raw driver.Module
	check(dev.WithContext(func() error {
		var err error
		raw, err = driver.NewModuleLoadData(unsafe.Pointer(&image[0]))
		return err
	}))
	m, err := OwnModule(dev, raw)
	if err != nil {
		_ = dev.WithContext(raw.Unload)
		t.Fatal(err)
	}
	defer func() {
		if err := m.Close(); err != nil {
			t.Error(err)
		}
	}()
	var p driver.DevicePtr
	check(m.ctx.Run(func() error { var err error; p, err = driver.MemAlloc(4); return err }))
	defer func() {
		if p != 0 {
			if err := m.ctx.Run(p.MemFree); err != nil {
				t.Error(err)
			}
		}
	}()
	maxThreads, err := dev.Attribute(driver.DevAttrMaxThreadsPerBlock)
	check(err)
	check(dev.Close()) // The module's retained context must keep working.
	if _, err := m.Function("missing_symbol"); err == nil {
		t.Fatal("missing kernel accepted")
	}
	k, err := m.Function("set_value")
	check(err)
	badLaunch := k.Launch([3]uint32{1, 1, 1}, [3]uint32{uint32(maxThreads + 1), 1, 1}, 0, 0, []unsafe.Pointer{unsafe.Pointer(&p)})
	if !errors.Is(badLaunch, driver.ErrInvalidValue) {
		t.Fatalf("oversized block: %v", badLaunch)
	}
	t.Logf("driver rejected block exceeding %d threads: %v", maxThreads, badLaunch)
	check(k.Launch([3]uint32{1, 1, 1}, [3]uint32{1, 1, 1}, 0, 0, []unsafe.Pointer{unsafe.Pointer(&p)}))
	var got uint32
	check(m.ctx.Run(func() error {
		if err := driver.CtxSynchronize(); err != nil {
			return err
		}
		return p.MemcpyDtoH(unsafe.Pointer(&got), 4)
	}))
	if got != 7 {
		t.Fatalf("got %d want 7", got)
	}
	check(m.ctx.Run(p.MemFree))
	p = 0
	check(m.Close())
	check(m.Close())
	if err := k.Launch([3]uint32{1, 1, 1}, [3]uint32{1, 1, 1}, 0, 0, nil); err == nil {
		t.Fatal("kernel launched after close")
	}
	if _, err := m.Function("set_value"); err == nil {
		t.Fatal("lookup after close")
	}
	t.Log("module outlived Device; output7 matched; lookup failure recovered; post-close operations rejected")
}
