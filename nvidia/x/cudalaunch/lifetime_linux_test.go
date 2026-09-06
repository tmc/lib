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
