package metalmem

import (
	"fmt"
	"sync"
	"testing"

	"github.com/tmc/apple/objectivec"
	"github.com/tmc/lib/apple/x/metal"
)

func TestPoolSizeClassLargestIntTerminates(t *testing.T) {
	if _, err := sizeClass(int(^uint(0) >> 1)); err == nil {
		t.Fatal("sizeClass accepted the largest int")
	}
}

func TestPoolDoesNotReturnNilBuffer(t *testing.T) {
	p := NewPool(nil)
	if err := p.Put(nil, 1024); err == nil {
		t.Fatal("Put accepted a nil buffer")
	}
	buf, err := p.Get(1024)
	if err == nil || buf != nil {
		t.Fatalf("Get = %v, %v; want nil buffer and error", buf, err)
	}
}

func TestPoolPhysicalReuseAndOwnership(t *testing.T) {
	dev, err := metal.NewDevice()
	if err != nil {
		t.Skipf("Metal device unavailable: %v", err)
	}
	defer dev.Close()
	p := NewPool(dev)
	buf, err := p.Get(1500)
	if err != nil {
		t.Fatal(err)
	}
	id := buf.GetID()
	if err := p.Put(buf, 1500); err != nil {
		t.Fatal(err)
	}
	if err := p.Put(buf, 1500); err == nil {
		t.Fatal("second Put of the same buffer succeeded")
	}
	reused, err := p.Get(1500)
	if err != nil {
		t.Fatal(err)
	}
	if reused.GetID() != id {
		t.Fatalf("Get returned buffer %v, want recycled %v", reused.GetID(), id)
	}
	if err := p.Put(reused, int(reused.Length())); err != nil {
		t.Fatal(err)
	}
	p.Free()
	p.Free()

	// A checked-out buffer remains caller-owned when the pool is freed.
	checkedOut, err := p.Get(1024)
	if err != nil {
		t.Fatal(err)
	}
	p.Free()
	objectivec.ObjectFromID(checkedOut.GetID()).Release()
}

func TestPoolConcurrentPhysicalReuse(t *testing.T) {
	dev, err := metal.NewDevice()
	if err != nil {
		t.Skipf("Metal device unavailable: %v", err)
	}
	defer dev.Close()
	p := NewPool(dev)
	before := dev.Raw().CurrentAllocatedSize()

	const workers = 16
	start := make(chan struct{})
	errs := make(chan error, workers)
	active := make(map[uintptr]bool)
	var activeMu sync.Mutex
	var wg sync.WaitGroup
	for worker := range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			minBytes := 1024 << (worker % 4)
			for range 50 {
				buf, err := p.Get(minBytes)
				if err != nil {
					errs <- err
					return
				}
				id := uintptr(buf.GetID())
				activeMu.Lock()
				if active[id] {
					activeMu.Unlock()
					errs <- fmt.Errorf("buffer %x checked out twice", id)
					return
				}
				active[id] = true
				activeMu.Unlock()
				buf.Contents()
				activeMu.Lock()
				delete(active, id)
				activeMu.Unlock()
				if err := p.Put(buf, minBytes); err != nil {
					errs <- err
					return
				}
			}
		}()
	}
	close(start)
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
	peak := dev.Raw().CurrentAllocatedSize()
	p.Free()
	p.Free()
	after := dev.Raw().CurrentAllocatedSize()
	t.Logf("Metal currentAllocatedSize before=%d pooled=%d after=%d", before, peak, after)
	if after > peak {
		t.Fatalf("currentAllocatedSize grew across pool release: pooled=%d after=%d", peak, after)
	}
}
