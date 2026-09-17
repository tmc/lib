package cudalaunch_test

import (
	"strings"
	"sync"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
)

func TestCheckSharedMemLimitValidation(t *testing.T) {
	if _, err := cudalaunch.CheckSharedMemLimit(nil, 0, 1024); err == nil {
		t.Error("expected error for nil device, got nil")
	}
	if _, err := cudalaunch.CheckSharedMemLimit(nil, -1, 1024); err == nil {
		t.Error("expected error for negative static memory, got nil")
	}
	if _, err := cudalaunch.CheckSharedMemLimit(nil, 0, -1); err == nil {
		t.Error("expected error for negative dynamic memory, got nil")
	}
}

func TestPrepareSharedMem_NilLauncher(t *testing.T) {
	var l *cudalaunch.Launcher
	if err := l.PrepareSharedMem(); err == nil {
		t.Error("expected error for nil launcher, got nil")
	}
}

func TestPrepareSharedMem_ZeroSharedMemNoop(t *testing.T) {
	l := cudalaunch.Kernel(driver.Function(1)).SharedMem(0)
	if err := l.PrepareSharedMem(); err != nil {
		t.Errorf("zero shared memory should be no-op, got: %v", err)
	}
}

func TestPrepareSharedMem_ClosedKernel(t *testing.T) {
	l := cudalaunch.OwnedKernel(new(cuda.Kernel)).SharedMem(1024)
	if err := l.PrepareSharedMem(); err == nil {
		t.Error("expected error preparing closed kernel, got nil")
	}
}

func TestPrepareSharedMem_RawFunctionRequiresDevice(t *testing.T) {
	l := cudalaunch.Kernel(driver.Function(1)).SharedMem(1024)
	if err := l.PrepareSharedMem(); err == nil || !strings.Contains(err.Error(), "requires a device or stream") {
		t.Errorf("expected error requiring device or stream, got: %v", err)
	}
}

func TestDescription_PrepareSharedMem(t *testing.T) {
	desc := cudalaunch.Description{
		Function:  driver.Function(1),
		SharedMem: 0,
	}
	if err := desc.PrepareSharedMem(); err != nil {
		t.Errorf("zero shared memory description should succeed, got: %v", err)
	}

	descWithShmem := cudalaunch.Description{
		Function:  driver.Function(1),
		SharedMem: 1024,
	}
	if err := descWithShmem.PrepareSharedMem(); err == nil {
		t.Error("expected error without device for bare function description, got nil")
	}
}

func TestConcurrentPrepareSharedMemSimulation(t *testing.T) {
	// Verify that multiple goroutines can safely invoke PrepareSharedMem logic concurrently
	var wg sync.WaitGroup
	for i := 0; i < 16; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			l := cudalaunch.Kernel(driver.Function(1)).SharedMem(0)
			if err := l.PrepareSharedMem(); err != nil {
				t.Errorf("concurrent prepare: %v", err)
			}
		}()
	}
	wg.Wait()
}
