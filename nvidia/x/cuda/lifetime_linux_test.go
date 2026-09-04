//go:build linux

package cuda_test

import (
	"context"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
)

func TestContextRunRestoresCurrent(t *testing.T) {
	dev, err := cuda.NewDevice(0)
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	ctx, err := dev.PrimaryContext()
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.Release()

	defer driver.Context(0).SetCurrent()
	if err := driver.Context(0).SetCurrent(); err != nil {
		t.Fatal(err)
	}
	if err := ctx.Run(func() error {
		current, err := driver.NewContextGetCurrent()
		if err != nil {
			return err
		}
		if current != ctx.Raw() {
			t.Errorf("current context = %v, want %v", current, ctx.Raw())
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	current, err := driver.NewContextGetCurrent()
	if err != nil {
		t.Fatal(err)
	}
	if current != 0 {
		t.Fatalf("current context after Run = %v, want none", current)
	}
}

func TestRepeatedRelease(t *testing.T) {
	dev, err := cuda.NewDevice(0)
	if err != nil {
		t.Fatal(err)
	}
	ctx, err := dev.PrimaryContext()
	if err != nil {
		t.Fatal(err)
	}
	stream, err := dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	event, err := dev.NewEvent()
	if err != nil {
		t.Fatal(err)
	}
	if err := event.Record(stream); err != nil {
		t.Fatal(err)
	}
	if err := stream.Synchronize(context.Background()); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name    string
		release func() error
	}{
		{"event", event.Destroy},
		{"stream", stream.Destroy},
		{"context", ctx.Release},
		{"device", dev.Close},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.release(); err != nil {
				t.Fatalf("first release: %v", err)
			}
			if err := tt.release(); err != nil {
				t.Fatalf("second release: %v", err)
			}
		})
	}
}
