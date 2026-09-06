//go:build linux

package cudamem

import (
	"context"
	"errors"
	"slices"
	"testing"
)

func TestPinnedTransferCUDA(t *testing.T) {
	dev := testDevice(t)
	stream, err := dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := stream.Destroy(); err != nil {
			t.Error(err)
		}
	}()
	device, err := Alloc[uint32](dev, 257)
	if err != nil {
		t.Fatal(err)
	}
	var host HostAllocation[uint32]
	if err := dev.WithContext(func() error { var err error; host, err = AllocHost[uint32](device.Len()); return err }); err != nil {
		_ = device.Free()
		t.Fatal(err)
	}
	defer func() {
		if err := stream.Synchronize(nil); err != nil {
			t.Error(err)
			return
		}
		if err := device.Free(); err != nil {
			t.Error(err)
		}
		if err := dev.WithContext(host.Free); err != nil {
			t.Error(err)
		}
	}()
	want := make([]uint32, device.Len())
	for i := range want {
		want[i] = uint32(i*5 + 11)
	}
	copy(host.Slice(), want)
	toDevice, err := device.CopyFromPinnedAsync(host, stream)
	if toDevice != nil {
		defer func() {
			if err := toDevice.Close(); err != nil {
				t.Error(err)
			}
		}()
	}
	if err != nil {
		t.Fatal(err)
	}
	if err := host.Free(); !errors.Is(err, ErrInUse) {
		t.Fatalf("host Free = %v", err)
	}
	if err := device.Free(); !errors.Is(err, ErrInUse) {
		t.Fatalf("device Free = %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := toDevice.Wait(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled Wait = %v", err)
	}
	if err := host.Free(); !errors.Is(err, ErrInUse) {
		t.Fatalf("host Free after cancellation = %v", err)
	}
	if err := toDevice.Close(); err != nil {
		t.Fatal(err)
	}
	// No stream-wide synchronization: the transfer event alone permits access.
	clear(host.Slice())
	toHost, err := device.CopyToPinnedAsync(host, stream)
	if toHost != nil {
		defer func() {
			if err := toHost.Close(); err != nil {
				t.Error(err)
			}
		}()
	}
	if err != nil {
		t.Fatal(err)
	}
	if err := toHost.Close(); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(host.Slice(), want) {
		t.Fatal("pinned round-trip mismatch")
	}
	if err := device.Free(); err != nil {
		t.Fatal(err)
	}
	if err := dev.WithContext(host.Free); err != nil {
		t.Fatal(err)
	}
}

func TestTransferLeaseCUDA(t *testing.T) {
	dev := testDevice(t)
	stream, err := dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := stream.Destroy(); err != nil {
			t.Error(err)
		}
	}()
	a, err := Alloc[uint32](dev, 257)
	if err != nil {
		t.Fatal(err)
	}
	b, err := Alloc[uint32](dev, a.Len())
	if err != nil {
		_ = a.Free()
		t.Fatal(err)
	}
	defer func() {
		if err := stream.Synchronize(nil); err != nil {
			t.Error(err)
			return
		}
		if err := a.Free(); err != nil {
			t.Error(err)
		}
		if err := b.Free(); err != nil {
			t.Error(err)
		}
	}()
	want := make([]uint32, a.Len())
	for i := range want {
		want[i] = uint32(i*3 + 7)
	}
	if err := a.CopyFromHostAsync(want, stream); err != nil {
		t.Fatal(err)
	}
	if err := a.Slice(1, 2).Free(); !errors.Is(err, ErrInUse) {
		t.Fatalf("in-flight alias Free = %v", err)
	}
	if err := b.CopyDeviceToDeviceAsync(a, stream); err != nil {
		t.Fatal(err)
	}
	got := make([]uint32, b.Len())
	if err := b.CopyToHostAsync(got, stream); err != nil {
		t.Fatal(err)
	}
	if err := b.Free(); !errors.Is(err, ErrInUse) {
		t.Fatalf("in-flight destination Free = %v", err)
	}
	if err := stream.Synchronize(nil); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, want) {
		t.Fatal("round-trip mismatch")
	}
	if err := a.Free(); err != nil {
		t.Fatal(err)
	}
	if err := b.Free(); err != nil {
		t.Fatal(err)
	}
}
