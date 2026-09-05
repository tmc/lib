package sanitizer_test

import (
	"errors"
	"os"
	"runtime"
	"sync/atomic"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/sanitizer"
)

func TestNativeAllocationCallback(t *testing.T) {
	path := os.Getenv("SANITIZER_PHYSICAL_LIBRARY")
	if path == "" {
		t.Skip("set SANITIZER_PHYSICAL_LIBRARY to run native callback validation")
	}
	check := func(err error) {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
	}
	check(sanitizer.Load(path))
	var events, bytes, badToken atomic.Uint64
	callback := func(token uintptr, domain sanitizer.CallbackDomain, id uint32, data unsafe.Pointer) {
		if token != 73 {
			badToken.Add(1)
		}
		if domain == sanitizer.CbDomainResource && sanitizer.CallbackIdResource(id) == sanitizer.CbidResourceDeviceMemoryAlloc && data != nil {
			event := *(*sanitizer.ResourceMemoryData)(data)
			events.Add(1)
			bytes.Add(uint64(event.Size))
		}
	}
	sub, err := sanitizer.SubscribeFunc(callback, 73)
	check(err)
	defer func() { check(sub.Unsubscribe()) }()
	second, err := sanitizer.SubscribeFunc(callback, 73)
	if second != nil {
		check(second.Unsubscribe())
	}
	if !errors.Is(err, sanitizer.ErrMaxLimitReached) {
		t.Fatalf("second subscribe: %v, want ErrMaxLimitReached", err)
	}
	check(sub.EnableDomain(true, sanitizer.CbDomainResource))
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	check(driver.Init(0))
	dev, err := driver.NewDeviceGet(0)
	check(err)
	ctx, err := driver.NewContext(nil, 0, dev)
	check(err)
	defer func() { check(ctx.Destroy()) }()
	before := bytes.Load()
	p, err := driver.MemAlloc(4096)
	check(err)
	check(p.MemFree())
	if events.Load() == 0 || bytes.Load()-before < 4096 || badToken.Load() != 0 {
		t.Fatalf("events=%d bytes=%d bad tokens=%d", events.Load(), bytes.Load()-before, badToken.Load())
	}
	t.Logf("allocation callbacks=%d bytes since allocation=%d", events.Load(), bytes.Load()-before)
}
