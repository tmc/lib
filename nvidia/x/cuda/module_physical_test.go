package cuda

import (
	"errors"
	"os"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
)

func TestModuleOwnership(t *testing.T) {
	if os.Getenv("CUDA_MODULE_TEST") != "1" {
		t.Skip("CUDA_MODULE_TEST=1 requires a CUDA device")
	}
	check := func(err error) {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
	}
	dev, err := NewDevice(0)
	check(err)
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
