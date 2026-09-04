package metal_test

import (
	"errors"
	"fmt"
	"sync"
	"testing"

	"github.com/tmc/apple/objectivec"
	libmetal "github.com/tmc/lib/apple/x/metal"
)

const ownershipKernel = `
#include <metal_stdlib>
using namespace metal;
kernel void ownership_noop(device uint *out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
	out[i] = i;
}
`

func TestDeviceCloseInvalidatesOwnedHandles(t *testing.T) {
	dev, err := libmetal.NewDevice()
	if err != nil {
		t.Skipf("Metal device unavailable: %v", err)
	}
	lib, err := dev.CompileLibrary(ownershipKernel)
	if err != nil {
		t.Fatal(err)
	}
	pso, err := dev.NewComputePipeline(lib, "ownership_noop")
	if err != nil {
		t.Fatal(err)
	}
	buf, err := dev.AllocBuffer(4)
	if err != nil {
		t.Fatal(err)
	}
	objectivec.ObjectFromID(buf.GetID()).Release()
	objectivec.ObjectFromID(pso.GetID()).Release()
	objectivec.ObjectFromID(lib.GetID()).Release()
	if err := dev.Close(); err != nil {
		t.Fatal(err)
	}
	if err := dev.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
	if dev.Raw() != nil || dev.Queue() != nil || dev.Name() != "" {
		t.Fatal("closed device retained a borrowed handle")
	}
	if _, err := dev.CompileLibrary(ownershipKernel); !errors.Is(err, libmetal.ErrClosed) {
		t.Fatalf("CompileLibrary after Close = %v, want ErrClosed", err)
	}
	if _, err := dev.AllocBuffer(4); !errors.Is(err, libmetal.ErrClosed) {
		t.Fatalf("AllocBuffer after Close = %v, want ErrClosed", err)
	}
}

func TestDeviceConcurrentCompileClose(t *testing.T) {
	dev, err := libmetal.NewDevice()
	if err != nil {
		t.Skipf("Metal device unavailable: %v", err)
	}
	const workers = 16
	start := make(chan struct{})
	errs := make(chan error, workers)
	var wg sync.WaitGroup
	for range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			lib, err := dev.CompileLibrary(ownershipKernel)
			if lib != nil {
				objectivec.ObjectFromID(lib.GetID()).Release()
			}
			errs <- err
		}()
	}
	close(start)
	if err := dev.Close(); err != nil {
		t.Fatal(err)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil && !errors.Is(err, libmetal.ErrClosed) {
			t.Errorf("CompileLibrary concurrent with Close: %v", err)
		}
	}
}

func TestDeviceConcurrentDerivedObjectLifetime(t *testing.T) {
	dev, err := libmetal.NewDevice()
	if err != nil {
		t.Skipf("Metal device unavailable: %v", err)
	}
	lib, err := dev.CompileLibrary(ownershipKernel)
	if err != nil {
		dev.Close()
		t.Fatal(err)
	}
	before := dev.Raw().CurrentAllocatedSize()

	const workers = 16
	start := make(chan struct{})
	errs := make(chan error, workers)
	var wg sync.WaitGroup
	for range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			pso, err := dev.NewComputePipeline(lib, "ownership_noop")
			if err != nil {
				errs <- err
				return
			}
			buf, err := dev.AllocBuffer(4096)
			if err != nil {
				objectivec.ObjectFromID(pso.GetID()).Release()
				errs <- err
				return
			}
			if buf.GetID() == 0 || pso.GetID() == 0 {
				errs <- fmt.Errorf("constructed a nil derived object")
			}
			objectivec.ObjectFromID(buf.GetID()).Release()
			objectivec.ObjectFromID(pso.GetID()).Release()
		}()
	}
	close(start)
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
	if _, err := dev.NewComputePipeline(lib, "missing_function"); err == nil {
		t.Error("missing function did not fail")
	}
	after := dev.Raw().CurrentAllocatedSize()
	t.Logf("Metal currentAllocatedSize before=%d after=%d", before, after)
	objectivec.ObjectFromID(lib.GetID()).Release()
	if err := dev.Close(); err != nil {
		t.Fatal(err)
	}
	if err := dev.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
}
