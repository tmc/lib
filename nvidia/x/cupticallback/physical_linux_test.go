//go:build linux && cupti_physical

package cupticallback_test

import (
	"runtime"
	"sync"
	"testing"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cupticallback"
)

func withContext(t *testing.T) uintptr {
	t.Helper()
	runtime.LockOSThread()
	t.Cleanup(runtime.UnlockOSThread)
	if err := driver.Init(0); err != nil {
		t.Skipf("CUDA driver unavailable: %v", err)
	}
	dev, err := driver.NewDeviceGet(0)
	if err != nil {
		t.Skipf("no CUDA device: %v", err)
	}
	ctx, err := dev.PrimaryCtxRetain()
	if err != nil {
		t.Fatalf("retain primary context: %v", err)
	}
	t.Cleanup(func() {
		if err := dev.PrimaryCtxRelease(); err != nil {
			t.Errorf("release context: %v", err)
		}
	})
	if err := ctx.SetCurrent(); err != nil {
		t.Fatalf("set current context: %v", err)
	}
	return uintptr(ctx)
}

func TestPhysicalCallbackLifecycle(t *testing.T) {
	_ = withContext(t)

	if err := cupticallback.CheckVersion(); err != nil {
		t.Skipf("CUPTI unavailable or unqualified version: %v", err)
	}

	var (
		mu         sync.Mutex
		seenEnter  bool
		seenExit   bool
		funcsSeen  = make(map[string]int)
		totalCalls int
	)

	sub, err := cupticallback.SubscribeFunc(func(evt *cupticallback.Event) {
		if evt.Domain == cupti.CbDomainDriverApi && evt.Api != nil {
			mu.Lock()
			totalCalls++
			funcsSeen[evt.Api.FunctionName]++
			if evt.Api.CallbackSite == cupti.ApiEnter {
				seenEnter = true
			}
			if evt.Api.CallbackSite == cupti.ApiExit {
				seenExit = true
			}
			mu.Unlock()
		}
	})
	if err != nil {
		t.Skipf("Subscribe: %v (CUPTI may be restricted)", err)
	}
	defer func() {
		if err := sub.Unsubscribe(); err != nil {
			t.Errorf("sub.Unsubscribe defer: %v", err)
		}
	}()

	if err := sub.EnableDomain(cupti.CbDomainDriverApi, true); err != nil {
		t.Fatalf("EnableDomain: %v", err)
	}

	// Trigger driver API calls under observation
	ptr, err := driver.MemAlloc(4096)
	if err != nil {
		t.Fatalf("MemAlloc: %v", err)
	}
	if err := ptr.MemFree(); err != nil {
		t.Fatalf("MemFree: %v", err)
	}

	if err := sub.EnableDomain(cupti.CbDomainDriverApi, false); err != nil {
		t.Fatalf("disable domain: %v", err)
	}

	mu.Lock()
	calls := totalCalls
	enter := seenEnter
	exit := seenExit
	allocCount := funcsSeen["cuMemAlloc_v2"]
	freeCount := funcsSeen["cuMemFree_v2"]
	mu.Unlock()

	if calls == 0 {
		t.Fatal("no callbacks recorded for driver API calls")
	}
	if !enter || !exit {
		t.Errorf("expected both enter and exit sites: enter=%v exit=%v", enter, exit)
	}
	if allocCount == 0 {
		t.Errorf("cuMemAlloc_v2 not observed in callbacks: %v", funcsSeen)
	}
	if freeCount == 0 {
		t.Errorf("cuMemFree_v2 not observed in callbacks: %v", funcsSeen)
	}

	// Verify unsubscription is idempotent
	if err := sub.Unsubscribe(); err != nil {
		t.Fatalf("sub.Unsubscribe: %v", err)
	}
	if err := sub.Unsubscribe(); err != nil {
		t.Errorf("idempotent Unsubscribe returned error: %v", err)
	}
}
