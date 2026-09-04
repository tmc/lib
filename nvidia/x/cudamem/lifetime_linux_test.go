//go:build linux

package cudamem

import (
	"context"
	"runtime"
	"runtime/debug"
	"sync"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
)

func testDevice(t *testing.T) *cuda.Device {
	t.Helper()
	dev, err := cuda.NewDevice(0)
	if err != nil {
		t.Fatalf("NewDevice: %v", err)
	}
	t.Cleanup(func() {
		if err := dev.Close(); err != nil {
			t.Errorf("Close: %v", err)
		}
	})
	return dev
}

func TestAllocationAliasesReleaseOnce(t *testing.T) {
	dev := testDevice(t)
	tests := []struct {
		name  string
		alloc func() (Slice[byte], error)
	}{
		{"device", func() (Slice[byte], error) { return Alloc[byte](dev, 4096) }},
		{"managed", func() (Slice[byte], error) { return AllocManaged[byte](dev, 4096) }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s, err := tt.alloc()
			if err != nil {
				t.Fatalf("allocate: %v", err)
			}
			alias := s.Slice(0, s.Len())
			if err := alias.Free(); err != nil {
				t.Fatalf("alias Free: %v", err)
			}
			if err := s.Free(); err != nil {
				t.Fatalf("owner Free after alias: %v", err)
			}
		})
	}
}

func TestTrackedAliasReleaseAccountsOnce(t *testing.T) {
	dev := testDevice(t)
	var tracker Tracker
	s, err := AllocTracked[byte](dev, &tracker, 4096)
	if err != nil {
		t.Fatal(err)
	}
	alias := s.Slice(0, s.Len())
	if err := alias.Free(); err != nil {
		t.Fatal(err)
	}
	if err := s.Free(); err != nil {
		t.Fatal(err)
	}
	current, _ := tracker.Snapshot()
	if current != 0 {
		t.Fatalf("tracked bytes after release = %d, want 0", current)
	}
}

func TestHostAllocationAliasesReleaseOnce(t *testing.T) {
	dev := testDevice(t)
	var h HostAllocation[byte]
	err := dev.WithContext(func() error {
		var err error
		h, err = AllocHost[byte](4096)
		return err
	})
	if err != nil {
		t.Fatal(err)
	}
	alias := h
	if err := dev.WithContext(alias.Free); err != nil {
		t.Fatalf("alias Free: %v", err)
	}
	if err := dev.WithContext(h.Free); err != nil {
		t.Fatalf("owner Free after alias: %v", err)
	}
	if h.Ptr() != 0 || h.Len() != 0 || h.Slice() != nil {
		t.Fatal("released host alias still exposes memory")
	}
}

func TestReleasedSliceMethodsFail(t *testing.T) {
	dev := testDevice(t)
	s, err := Alloc[byte](dev, 16)
	if err != nil {
		t.Fatal(err)
	}
	alias := s.Slice(0, s.Len())
	if err := s.Free(); err != nil {
		t.Fatal(err)
	}
	if alias.Ptr() != 0 {
		t.Fatal("released alias exposes a device pointer")
	}
	tests := []struct {
		name string
		do   func() error
	}{
		{"copy from host", func() error { return alias.CopyFromHost(make([]byte, alias.Len())) }},
		{"copy to host", func() error { return alias.CopyToHost(make([]byte, alias.Len())) }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.do(); err == nil {
				t.Fatal("operation on released alias succeeded")
			}
		})
	}
}

func TestDeviceCopyHoldsBothAllocations(t *testing.T) {
	dev := testDevice(t)
	src, err := Alloc[byte](dev, 4096)
	if err != nil {
		t.Fatal(err)
	}
	dst, err := Alloc[byte](dev, 4096)
	if err != nil {
		t.Fatal(err)
	}
	defer dst.Free()

	entered := make(chan struct{})
	release := make(chan struct{})
	copyDone := make(chan error, 1)
	go func() {
		copyDone <- dst.withSource(src, func(_, _ driver.DevicePtr) error {
			close(entered)
			<-release
			return nil
		})
	}()
	<-entered
	freeDone := make(chan error, 1)
	go func() { freeDone <- src.Free() }()
	select {
	case err := <-freeDone:
		t.Fatalf("source Free completed during copy critical section: %v", err)
	default:
	}
	close(release)
	if err := <-copyDone; err != nil {
		t.Fatal(err)
	}
	if err := <-freeDone; err != nil {
		t.Fatal(err)
	}
}

func TestDeviceSelfCopyDoesNotDeadlock(t *testing.T) {
	dev := testDevice(t)
	s, err := Alloc[byte](dev, 4096)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Free()
	done := make(chan error, 1)
	go func() { done <- s.CopyDeviceToDevice(s) }()
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("self-copy deadlocked")
	}
}

func TestAsyncAllocationAliasesReleaseOnce(t *testing.T) {
	dev := testDevice(t)
	stream, err := dev.NewStreamWithFlags(uint32(driver.StreamNonBlocking))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = stream.Destroy() })
	s, err := AllocAsync[byte](stream, 4096)
	if err != nil {
		t.Fatal(err)
	}
	alias := s
	if err := alias.FreeAsync(stream); err != nil {
		t.Fatalf("alias FreeAsync: %v", err)
	}
	if err := s.FreeAsync(stream); err != nil {
		t.Fatalf("owner FreeAsync after alias: %v", err)
	}
	if err := stream.Synchronize(context.Background()); err != nil {
		t.Fatalf("Synchronize: %v", err)
	}
}

func TestAsyncFreeAccountsAtCompletion(t *testing.T) {
	dev := testDevice(t)
	stream, err := dev.NewStreamWithFlags(uint32(driver.StreamNonBlocking))
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Destroy()
	var tracker Tracker
	s, err := AllocTracked[byte](dev, &tracker, 4096)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.FreeAsync(stream); err != nil {
		t.Fatal(err)
	}
	current, _ := tracker.Snapshot()
	if current != 4096 {
		t.Fatalf("tracked bytes before completion = %d, want 4096", current)
	}
	if err := stream.Synchronize(context.Background()); err != nil {
		t.Fatal(err)
	}
	current, _ = tracker.Snapshot()
	if current != 0 {
		t.Fatalf("tracked bytes after completion = %d, want 0", current)
	}
}

func TestConcurrentStreamsAsyncCopiesAndCancellation(t *testing.T) {
	dev := testDevice(t)
	const n = 4 << 20
	type transfer struct {
		stream *cuda.Stream
		device Slice[byte]
		src    []byte
		dst    []byte
	}
	transfers := make([]transfer, 2)
	for i := range transfers {
		s, err := dev.NewStreamWithFlags(uint32(driver.StreamNonBlocking))
		if err != nil {
			t.Fatal(err)
		}
		d, err := Alloc[byte](dev, n)
		if err != nil {
			_ = s.Destroy()
			t.Fatal(err)
		}
		transfers[i] = transfer{s, d, make([]byte, n), make([]byte, n)}
		for j := range transfers[i].src {
			transfers[i].src[j] = byte(i + j)
		}
		t.Cleanup(func() {
			_ = d.Free()
			_ = s.Destroy()
		})
	}

	for i := range transfers {
		tr := &transfers[i]
		if err := tr.device.CopyFromHostAsync(tr.src, tr.stream); err != nil {
			t.Fatal(err)
		}
		if err := tr.device.CopyToHostAsync(tr.dst, tr.stream); err != nil {
			t.Fatal(err)
		}
	}

	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	cleaned := false
	transfers[0].stream.AddCleanup(func() { cleaned = true })
	if err := transfers[0].stream.Synchronize(cancelled); err != context.Canceled {
		t.Fatalf("cancelled Synchronize = %v, want context.Canceled", err)
	}
	if cleaned {
		t.Fatal("cancelled synchronization ran completion cleanup")
	}

	var wg sync.WaitGroup
	errs := make(chan error, len(transfers))
	for i := range transfers {
		wg.Add(1)
		go func(tr *transfer) {
			defer wg.Done()
			errs <- tr.stream.Synchronize(context.Background())
		}(&transfers[i])
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatal(err)
		}
	}
	if !cleaned {
		t.Fatal("successful synchronization did not run completion cleanup")
	}
	for i := range transfers {
		if string(transfers[i].dst) != string(transfers[i].src) {
			t.Fatalf("transfer %d data mismatch", i)
		}
	}
}

func TestArenaConstructorRollback(t *testing.T) {
	dev := testDevice(t)
	var tracker Tracker
	a := NewTrackedArena(dev, &tracker)
	_ = AllocIn[byte](a, "first", 4096)
	_ = AllocIn[byte](a, "fail", -1)
	_ = AllocIn[byte](a, "not-attempted", 4096)
	if a.Err() == nil {
		t.Fatal("arena did not retain allocation failure")
	}
	if err := a.Free(); err != nil {
		t.Fatal(err)
	}
	current, _ := tracker.Snapshot()
	if current != 0 {
		t.Fatalf("tracked bytes after rollback = %d, want 0", current)
	}
}

func TestPostSynchronizeAllocationBaseline(t *testing.T) {
	dev := testDevice(t)
	memInfo := func() uint {
		t.Helper()
		var free uint
		if err := dev.WithContext(func() error {
			var err error
			free, _, err = driver.MemGetInfo()
			return err
		}); err != nil {
			t.Fatalf("MemGetInfo: %v", err)
		}
		return free
	}
	before := memInfo()
	var tracker Tracker
	s, err := AllocTracked[byte](dev, &tracker, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	stream, err := dev.NewStreamWithFlags(uint32(driver.StreamNonBlocking))
	if err != nil {
		t.Fatal(err)
	}
	src := make([]byte, 64<<20)
	if err := s.CopyFromHostAsync(src, stream); err != nil {
		t.Fatal(err)
	}
	if err := stream.Synchronize(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := s.Free(); err != nil {
		t.Fatal(err)
	}
	if err := stream.Destroy(); err != nil {
		t.Fatal(err)
	}
	src = nil
	runtime.GC()
	debug.FreeOSMemory()
	current, peak := tracker.Snapshot()
	if current != 0 || peak != 64<<20 {
		t.Fatalf("tracker after synchronized release = current %d peak %d", current, peak)
	}
	after := memInfo()
	t.Logf("device free bytes before=%d after=%d delta=%d tracker_current=%d tracker_peak=%d", before, after, int64(after)-int64(before), current, peak)
}
