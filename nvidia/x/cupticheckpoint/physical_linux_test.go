//go:build linux && cupti_physical

package cupticheckpoint_test

import (
	"runtime"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cupticheckpoint"
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

func TestPhysicalCheckpointLifecycle(t *testing.T) {
	ctx := withContext(t)

	if err := cupticheckpoint.CheckVersion(); err != nil {
		t.Skipf("CUPTI unavailable or unqualified version: %v", err)
	}

	opts := cupticheckpoint.Options{
		AllowOverwrite: true,
		Optimizations:  cupticheckpoint.OptNone,
	}

	cp, err := cupticheckpoint.Save(ctx, opts)
	if err != nil {
		t.Fatalf("Save: %v", err)
	}
	defer func() {
		if err := cp.Free(); err != nil {
			t.Errorf("cp.Free defer: %v", err)
		}
	}()

	if err := cp.Restore(); err != nil {
		t.Fatalf("cp.Restore: %v", err)
	}

	if err := cp.Free(); err != nil {
		t.Fatalf("cp.Free: %v", err)
	}
}
