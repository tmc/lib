package cudaplan_test

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudaplan"
)

func TestTakeoverFailedDrainRetainsAndReleasesOnce(t *testing.T) {
	var attempts atomic.Int32
	var releases atomic.Int32
	comp := cudaplan.ExportNewHostCompletion(func(context.Context) error {
		if attempts.Add(1) == 1 {
			return errors.New("drain unavailable")
		}
		return nil
	}, nil, func() { releases.Add(1) })
	if err := comp.Close(); err == nil {
		t.Fatal("first failed drain returned success")
	}
	if releases.Load() != 0 {
		t.Fatalf("released after failed drain: %d", releases.Load())
	}
	if err := comp.Close(); err != nil {
		t.Fatalf("retry drain: %v", err)
	}
	if releases.Load() != 1 {
		t.Fatalf("release count = %d, want 1", releases.Load())
	}
	if err := comp.Close(); err != nil {
		t.Fatalf("idempotent close: %v", err)
	}
	if releases.Load() != 1 {
		t.Fatalf("release count after repeated close = %d", releases.Load())
	}
}

func TestTakeoverRejectInputOutputOverlap(t *testing.T) {
	b := cudaplan.NewBindings()
	if err := b.SetInput("x", 0x1000, 32, nil); err != nil {
		t.Fatal(err)
	}
	if err := b.SetOutput("y", 0x1010, 32, nil); err != nil {
		t.Fatal(err)
	}
	if err := b.Validate(nil, map[string]int{"x": 32}, map[string]int{"y": 32}); !errors.Is(err, cudaplan.ErrOverlappingOutputs) {
		t.Fatalf("overlap validation: %v", err)
	}
}

func TestTakeoverRejectWrappingBinding(t *testing.T) {
	b := cudaplan.NewBindings()
	err := b.SetOutput("y", driver.DevicePtr(^uint64(0)-7), 16, nil)
	if err == nil {
		err = b.Validate(nil, nil, map[string]int{"y": 16})
	}
	if err == nil {
		t.Fatal("wrapping device address accepted")
	}
}

func TestTakeoverRejectUnknownDevice(t *testing.T) {
	b := cudaplan.NewBindings()
	if err := b.SetInput("x", 0x1000, 32, nil); err != nil {
		t.Fatal(err)
	}
	if err := b.Validate(&cuda.Device{}, map[string]int{"x": 32}, nil); err == nil {
		t.Fatal("unknown device affinity accepted")
	}
}

func TestTakeoverRejectImplicitHostExecution(t *testing.T) {
	p, err := cudaplan.Compile(buildAffineGraph(t))
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	b := cudaplan.NewBindings()
	for name, size := range p.RequiredInputs() {
		if err := b.SetInput(name, driver.DevicePtr(0x1000+len(name)*0x1000), size, nil); err != nil {
			t.Fatal(err)
		}
	}
	for name, size := range p.RequiredOutputs() {
		if err := b.SetOutput(name, 0x10000, size, nil); err != nil {
			t.Fatal(err)
		}
	}
	c, err := p.Execute(nil, b)
	if err == nil {
		if c != nil {
			_ = c.Close()
		}
		t.Fatal("missing device/stream returned successful no-op execution")
	}
}

func TestTakeoverHostBudget(t *testing.T) {
	p, err := cudaplan.Compile(buildAffineGraph(t), cudaplan.WithPeakWorkspaceBudget(1))
	if p != nil {
		_ = p.Close()
	}
	if !errors.Is(err, cudaplan.ErrWorkspaceBudgetExceeded) {
		t.Fatalf("host plan budget: %v", err)
	}
}
