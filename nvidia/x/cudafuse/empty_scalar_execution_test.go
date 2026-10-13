package cudafuse

import (
	"context"
	"errors"
	"fmt"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudair"
	"slices"
	"testing"
)

func TestEmptyScalarExecutionBeforeBorrow(t *testing.T) {
	for _, left := range []bool{false, true} {
		for _, eager := range []bool{false, true} {
			t.Run(fmt.Sprintf("%t/%t", left, eager), func(t *testing.T) {
				ctx := NewLazyContext()
				zero := NewLazyArray(ctx, []int{2, 0}, cudair.Float32)
				scalar := NewLazyArray(ctx, nil, cudair.Float32)
				a, b := zero, scalar
				if left {
					a, b = scalar, zero
				}
				root := a.Add(b)
				p, err := root.Materialize()
				if err != nil {
					t.Fatal(err)
				}
				borrows := 0
				bindings := map[string]executionInput{
					zero.Node.Name(): {description: executionBindingDescription{shape: []int{2, 0}, dtype: cudair.Float32}, borrow: func(*cuda.Device) (driver.DevicePtr, func(), error) {
						borrows++
						return 0, nil, errors.New("unexpected empty borrow")
					}},
					scalar.Node.Name(): {description: executionBindingDescription{shape: nil, dtype: cudair.Float32, length: 1, ptr: 4096, bytes: 4}, borrow: func(*cuda.Device) (driver.DevicePtr, func(), error) { borrows++; return 4096, func() {}, nil }},
				}
				// executeInputs is the same checked route called by public Execute/Eager;
				// descriptions/borrows here are host fixtures, not fabricated device storage.
				e, err := p.executeInputs(context.Background(), nil, bindings, eager)
				if err != nil || e == nil || borrows != 0 || !slices.Equal(e.Shape(), []int{2, 0}) {
					t.Fatal(e, err, borrows)
				}
				values, err := e.Readback(context.Background())
				if err != nil || len(values) != 0 || e.Stats() != (ExecutionStats{}) {
					t.Fatal(values, err)
				}
				if err := e.Close(); err != nil {
					t.Fatal(err)
				}
				bad := bindings[scalar.Node.Name()]
				bad.description.length = 0
				bindings[scalar.Node.Name()] = bad
				if _, err := p.executeInputs(context.Background(), nil, bindings, eager); err == nil || borrows != 0 {
					t.Fatal("zero bypassed binding validation")
				}
			})
		}
	}
}
func TestEmptyPublicExecutionAndCancellation(t *testing.T) {
	ctx := NewLazyContext()
	a := NewLazyArray(ctx, []int{0}, cudair.Float32)
	b := NewLazyArray(ctx, []int{0}, cudair.Float32)
	p, err := a.Add(b).Materialize()
	if err != nil {
		t.Fatal(err)
	}
	bindings := map[string]ExecutionBinding{a.Node.Name(): {Shape: []int{0}, DType: cudair.Float32}, b.Node.Name(): {Shape: []int{0}, DType: cudair.Float32}}
	for _, execute := range []func(context.Context, *cuda.Stream, map[string]ExecutionBinding) (*Execution, error){p.Execute, p.ExecuteEager} {
		e, err := execute(context.Background(), nil, bindings)
		if err != nil || e == nil {
			t.Fatal(e, err)
		}
		if err := e.Close(); err != nil {
			t.Fatal(err)
		}
		cancelled, cancel := context.WithCancel(context.Background())
		cancel()
		if _, err := execute(cancelled, nil, bindings); !errors.Is(err, context.Canceled) {
			t.Fatal(err)
		}
	}
}
