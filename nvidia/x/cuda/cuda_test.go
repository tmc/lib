package cuda_test

import (
	"context"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
)

func TestInitAndDevices(t *testing.T) {
	if err := driver.Load(""); err != nil {
		t.Skipf("skipping test: CUDA driver not available: %v", err)
	}
	// Past this point libcuda is present, so every remaining failure is real.
	if err := cuda.Init(); err != nil {
		t.Fatalf("cuInit failed with libcuda loaded from %q: %v", driver.Path(), err)
	}

	count, err := cuda.DeviceCount()
	if err != nil {
		t.Fatalf("DeviceCount failed: %v", err)
	}
	t.Logf("Found %d CUDA devices", count)

	if count == 0 {
		t.Skip("no CUDA devices available")
	}

	dev, err := cuda.NewDevice(0)
	if err != nil {
		t.Fatalf("NewDevice(0) failed: %v", err)
	}

	name, err := dev.Name()
	if err != nil {
		t.Fatalf("dev.Name() failed: %v", err)
	}
	t.Logf("Device 0 Name: %s", name)

	maj, min, err := dev.ComputeCapability()
	if err != nil {
		t.Fatalf("ComputeCapability failed: %v", err)
	}
	t.Logf("Compute Capability: sm_%d%d", maj, min)

	mem, err := dev.TotalMem()
	if err != nil {
		t.Fatalf("TotalMem failed: %v", err)
	}
	t.Logf("Total Memory: %d bytes (%.2f GiB)", mem, float64(mem)/(1024*1024*1024))
}

func TestStreamAndEventLifecycle(t *testing.T) {
	if err := driver.Load(""); err != nil {
		t.Skipf("CUDA driver not available: %v", err)
	}
	// Past this point libcuda is present, so every remaining failure is real.
	if err := cuda.Init(); err != nil {
		t.Fatalf("cuInit failed with libcuda loaded from %q: %v", driver.Path(), err)
	}

	count, err := cuda.DeviceCount()
	if err != nil || count == 0 {
		t.Skip("no CUDA devices available")
	}

	dev, err := cuda.NewDevice(0)
	if err != nil {
		t.Fatalf("NewDevice(0) failed: %v", err)
	}

	stream, err := dev.NewStream()
	if err != nil {
		t.Fatalf("NewStream failed: %v", err)
	}
	defer stream.Destroy()

	eventStart, err := dev.NewEvent()
	if err != nil {
		t.Fatalf("NewEvent (start) failed: %v", err)
	}
	defer eventStart.Destroy()

	eventEnd, err := dev.NewEvent()
	if err != nil {
		t.Fatalf("NewEvent (end) failed: %v", err)
	}
	defer eventEnd.Destroy()

	if err := eventStart.Record(stream); err != nil {
		t.Fatalf("eventStart.Record failed: %v", err)
	}

	if err := eventEnd.Record(stream); err != nil {
		t.Fatalf("eventEnd.Record failed: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	if err := stream.Synchronize(ctx); err != nil {
		t.Fatalf("stream.Synchronize failed: %v", err)
	}

	dur, err := eventEnd.ElapsedSince(eventStart)
	if err != nil {
		t.Fatalf("eventEnd.ElapsedSince failed: %v", err)
	}
	t.Logf("Event elapsed time: %v", dur)
}

func TestStreamCleanup(t *testing.T) {
	if err := driver.Load(""); err != nil {
		t.Skipf("CUDA driver not available: %v", err)
	}
	// Past this point libcuda is present, so every remaining failure is real.
	if err := cuda.Init(); err != nil {
		t.Fatalf("cuInit failed with libcuda loaded from %q: %v", driver.Path(), err)
	}
	count, err := cuda.DeviceCount()
	if err != nil || count == 0 {
		t.Skip("no CUDA devices available")
	}

	dev, err := cuda.NewDevice(0)
	if err != nil {
		t.Fatalf("NewDevice(0) failed: %v", err)
	}

	stream, err := dev.NewStream()
	if err != nil {
		t.Fatalf("NewStream failed: %v", err)
	}
	defer stream.Destroy()

	cleaned := false
	stream.AddCleanup(func() {
		cleaned = true
	})

	if cleaned {
		t.Fatal("cleanup should not run before stream synchronization")
	}

	if err := stream.Synchronize(context.Background()); err != nil {
		t.Fatalf("Synchronize failed: %v", err)
	}

	if !cleaned {
		t.Fatal("cleanup must be executed upon stream synchronization")
	}
}
