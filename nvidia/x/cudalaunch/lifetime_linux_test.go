//go:build linux

package cudalaunch_test

import (
	"context"
	"sync"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudatest"
)

const lifetimePTX = `.version 8.0
.target sm_80
.address_size 64

.visible .entry spin(.param .u64 cycles) {
	.reg .pred %p;
	.reg .u64 %cycles, %start, %now;
	ld.param.u64 %cycles, [cycles];
	mov.u64 %start, %clock64;
again:
	mov.u64 %now, %clock64;
	sub.u64 %now, %now, %start;
	setp.lt.u64 %p, %now, %cycles;
	@%p bra again;
	ret;
}
`

func loadLifetimeKernel(t *testing.T, dev *cuda.Device) (driver.Module, driver.Function) {
	t.Helper()
	image := append([]byte(lifetimePTX), 0)
	var mod driver.Module
	var fn driver.Function
	if err := dev.WithContext(func() error {
		var err error
		mod, err = driver.NewModuleLoadData(unsafe.Pointer(&image[0]))
		if err != nil {
			return err
		}
		fn, err = mod.GetFunction("spin")
		return err
	}); err != nil {
		t.Fatalf("load PTX: %v", err)
	}
	t.Cleanup(func() {
		if err := dev.WithContext(mod.Unload); err != nil {
			t.Errorf("Unload: %v", err)
		}
	})
	return mod, fn
}

func TestCounterCountsSubmissionNotCompletion(t *testing.T) {
	dev := cudatest.RequireDevice(t)
	t.Cleanup(func() { _ = dev.Close() })
	_, fn := loadLifetimeKernel(t, dev)
	stream, err := dev.NewStreamWithFlags(uint32(driver.StreamNonBlocking))
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Destroy()

	var counter cudalaunch.Counter
	launch := cudalaunch.Kernel(fn).On(stream).Count(&counter)
	if err := launch.Launch(uint64(1_000_000_000)); err != nil {
		t.Fatalf("Launch: %v", err)
	}
	if got := counter.Value(); got != 1 {
		t.Fatalf("counter after submission = %d, want 1", got)
	}
	idle, err := stream.Query()
	if err != nil {
		t.Fatal(err)
	}
	if idle {
		t.Fatal("stream completed before submission/completion distinction was observed")
	}

	launch.Grid(0)
	if err := launch.Launch(uint64(1)); err == nil {
		t.Fatal("zero-grid launch succeeded")
	}
	if got := counter.Value(); got != 1 {
		t.Fatalf("counter after rejected launch = %d, want 1", got)
	}
	if err := stream.Synchronize(context.Background()); err != nil {
		t.Fatalf("Synchronize: %v", err)
	}
}

func TestConcurrentStreamLaunchesOwnContext(t *testing.T) {
	dev := cudatest.RequireDevice(t)
	t.Cleanup(func() { _ = dev.Close() })
	_, fn := loadLifetimeKernel(t, dev)

	const streams = 4
	var counter cudalaunch.Counter
	var wg sync.WaitGroup
	errs := make(chan error, streams)
	for range streams {
		stream, err := dev.NewStreamWithFlags(uint32(driver.StreamNonBlocking))
		if err != nil {
			t.Fatal(err)
		}
		wg.Add(1)
		go func() {
			defer wg.Done()
			defer stream.Destroy()
			if err := cudalaunch.Kernel(fn).On(stream).Count(&counter).Launch(uint64(1_000_000)); err != nil {
				errs <- err
				return
			}
			errs <- stream.Synchronize(context.Background())
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Errorf("concurrent launch: %v", err)
		}
	}
	if got := counter.Value(); got != streams {
		t.Fatalf("counter = %d, want %d successful submissions", got, streams)
	}
}

// TestDescribeDoesNotDetectAClosedModule locks in the limit the Description
// lifetime note states. Closing a module unloads it whatever still refers to
// it: cuda.Module.close zeroes the module handle and leaves every Kernel's
// cached function alone, and Kernel.Raw returns that cached value without
// consulting the module. Describe therefore still succeeds and still reports a
// non-zero Function.
//
// This is a wart, not a guarantee. It is asserted so that adding a real lease
// to cuda, or a liveness check to Kernel.Raw, has to come here and say so.
func TestDescribeDoesNotDetectAClosedModule(t *testing.T) {
	dev := cudatest.Require(t)

	image := append([]byte(lifetimePTX), 0)
	var raw driver.Module
	if err := dev.WithContext(func() error {
		var err error
		raw, err = driver.NewModuleLoadData(unsafe.Pointer(&image[0]))
		return err
	}); err != nil {
		t.Fatalf("load PTX: %v", err)
	}
	mod, err := cuda.OwnModule(dev, raw)
	if err != nil {
		t.Fatal(err)
	}
	kern, err := mod.Function("spin")
	if err != nil {
		t.Fatal(err)
	}

	before, err := cudalaunch.OwnedKernel(kern).Describe()
	if err != nil {
		t.Fatalf("Describe before Close: %v", err)
	}
	if before.Function == 0 {
		t.Fatal("Describe returned a zero Function for a loaded module")
	}

	if err := mod.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	after, err := cudalaunch.OwnedKernel(kern).Describe()
	if err != nil {
		t.Fatalf("Describe after Close returned an error: %v\n"+
			"cuda gained a liveness check; update the Description lifetime note, which says it has none", err)
	}
	if after.Function != before.Function {
		t.Fatalf("Describe after Close returned Function %#x, want the stale %#x it caches",
			uintptr(after.Function), uintptr(before.Function))
	}
	// The kernel the Description retains is the closed one; nothing about
	// holding it kept the module loaded.
	if after.Kernel() != kern {
		t.Error("Description does not retain the kernel it was built from")
	}
}
