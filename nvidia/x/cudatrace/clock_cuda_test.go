package cudatrace_test

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudatest"
	"github.com/tmc/lib/nvidia/x/cudatrace"
)

func TestCalibrateNilArgs(t *testing.T) {
	if _, err := cudatrace.Calibrate(context.Background(), nil, nil); err == nil {
		t.Fatal("expected error for nil stream")
	}
}

// TestClockPlacesDeviceWorkOnHostTimeline checks that a span timed by CUDA
// events and placed with Clock lands inside the host-clock window that brackets
// the same device work, and that its width matches the event interval.
func TestClockPlacesDeviceWorkOnHostTimeline(t *testing.T) {
	dev := cudatest.Require(t)
	defer dev.Close()

	stream, err := dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Destroy()

	anchor, err := dev.NewEvent()
	if err != nil {
		t.Fatal(err)
	}
	defer anchor.Destroy()
	start, err := dev.NewEvent()
	if err != nil {
		t.Fatal(err)
	}
	defer start.Destroy()
	end, err := dev.NewEvent()
	if err != nil {
		t.Fatal(err)
	}
	defer end.Destroy()

	clock, err := cudatrace.Calibrate(context.Background(), stream, anchor)
	if err != nil {
		t.Fatalf("Calibrate: %v", err)
	}

	other, err := cuda.NewDevice(dev.Ordinal())
	if err != nil {
		t.Fatal(err)
	}
	defer other.Close()
	otherAnchor, err := other.NewEvent()
	if err != nil {
		t.Fatal(err)
	}
	defer otherAnchor.Destroy()
	if _, err := cudatrace.Calibrate(context.Background(), stream, otherAnchor); err != nil {
		t.Fatalf("Calibrate with another handle for the same primary context: %v", err)
	}
	if _, err := clock.HostNs(otherAnchor); err != nil {
		t.Fatalf("HostNs with another handle for the same primary context: %v", err)
	}

	const n = 16 << 20
	src, err := cudamem.Alloc[byte](dev, n)
	if err != nil {
		t.Fatal(err)
	}
	defer src.Free()
	dst, err := cudamem.Alloc[byte](dev, n)
	if err != nil {
		t.Fatal(err)
	}
	defer dst.Free()

	hostBefore := time.Now().UnixNano()
	if err := start.Record(stream); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 8; i++ {
		if err := dst.CopyDeviceToDeviceAsync(src, stream); err != nil {
			t.Fatal(err)
		}
	}
	if err := end.Record(stream); err != nil {
		t.Fatal(err)
	}
	if err := stream.Synchronize(context.Background()); err != nil {
		t.Fatal(err)
	}
	hostAfter := time.Now().UnixNano()

	startNs, err := clock.HostNs(start)
	if err != nil {
		t.Fatalf("HostNs(start): %v", err)
	}
	endNs, err := clock.HostNs(end)
	if err != nil {
		t.Fatalf("HostNs(end): %v", err)
	}
	elapsed, err := end.ElapsedSince(start)
	if err != nil {
		t.Fatal(err)
	}

	if endNs < startNs {
		t.Fatalf("span inverted: start %d end %d", startNs, endNs)
	}
	// The span width must reproduce the event interval: both come from the same
	// device clock. They can differ by a nanosecond because each endpoint is
	// converted from its own float32 millisecond offset to the anchor.
	if diff := int64(endNs-startNs) - int64(elapsed); diff < -4 || diff > 4 {
		t.Errorf("span width %d ns, event interval %d ns", int64(endNs-startNs), int64(elapsed))
	}
	// The span must sit inside the host window that brackets the same work,
	// allowing for the calibration uncertainty at each edge.
	slack := clock.UncertaintyNs() + int64(time.Millisecond)
	if int64(startNs) < hostBefore-slack || int64(endNs) > hostAfter+slack {
		t.Errorf("span [%d,%d] outside host window [%d,%d] with slack %d",
			startNs, endNs, hostBefore, hostAfter, slack)
	}
	if elapsed <= 0 {
		t.Errorf("device work measured %v, want a positive interval", elapsed)
	}
	t.Logf("uncertainty=%dns device=%v span=[%d,%d] host=[%d,%d]",
		clock.UncertaintyNs(), elapsed, startNs, endNs, hostBefore, hostAfter)

	// Null control: an empty interval must not acquire a duration from the
	// calibration itself.
	if err := start.Record(stream); err != nil {
		t.Fatal(err)
	}
	if err := end.Record(stream); err != nil {
		t.Fatal(err)
	}
	if err := stream.Synchronize(context.Background()); err != nil {
		t.Fatal(err)
	}
	idle, err := end.ElapsedSince(start)
	if err != nil {
		t.Fatal(err)
	}
	nullStart, err := clock.HostNs(start)
	if err != nil {
		t.Fatal(err)
	}
	nullEnd, err := clock.HostNs(end)
	if err != nil {
		t.Fatal(err)
	}
	if nullEnd-nullStart > uint64(elapsed)/10 {
		t.Errorf("empty span %d ns, want well under the %v of real work", nullEnd-nullStart, elapsed)
	}
	t.Logf("null control: empty span %d ns (device %v)", nullEnd-nullStart, idle)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := cudatrace.Calibrate(ctx, stream, anchor); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled Calibrate = %v, want context.Canceled", err)
	}
	if err := anchor.Destroy(); err != nil {
		t.Fatal(err)
	}
	if _, err := clock.HostNs(end); err == nil {
		t.Fatal("HostNs succeeded after anchor destruction")
	}
}
