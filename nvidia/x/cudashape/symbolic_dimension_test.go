package cudashape_test

import (
	"errors"
	"fmt"
	"slices"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudashape"
)

func TestNamedDim(t *testing.T) {
	for _, name := range []string{"", "batch", "seq_len", " batch ", "π"} {
		t.Run(name, func(t *testing.T) {
			dim, err := cudashape.NamedDim(name)
			if name == "" {
				if err == nil || err.Error() != "cudashape: empty symbol name" || dim != (cudashape.Dim{}) {
					t.Fatalf("NamedDim(empty) = (%v, %v), want zero dimension and contextual error", dim, err)
				}
				return
			}
			if err != nil || !dim.IsSymbol() || dim.Name() != name || dim.String() != name {
				t.Fatalf("NamedDim(%q) = (%v, %v), symbolic identity changed", name, dim, err)
			}
		})
	}
}

func TestNamedDimShapeBinding(t *testing.T) {
	for _, batch := range []int{0, 1, 16} {
		t.Run(fmt.Sprint(batch), func(t *testing.T) {
			dim, err := cudashape.NamedDim("batch")
			if err != nil {
				t.Fatal(err)
			}
			shape := cudashape.NewSymbolicShape(dim, cudashape.ConstDim(8))
			env := cudashape.NewEnvironment()
			if _, err := shape.Bind(env); !errors.Is(err, cudashape.ErrUnboundSymbol) {
				t.Fatalf("unbound dimension error = %v", err)
			}
			if err := env.Bind(dim.Name(), batch); err != nil {
				t.Fatal(err)
			}
			bound, err := shape.Bind(env)
			if err != nil || !slices.Equal(bound.Dims(), []int{batch, 8}) {
				t.Fatalf("bound shape = (%v, %v), want [%d, 8]", bound, err, batch)
			}
			if err := env.Bind(dim.Name(), batch+1); !errors.Is(err, cudashape.ErrContradictoryBinding) {
				t.Fatalf("contradiction error = %v", err)
			}
			bound, err = shape.Bind(env)
			if err != nil || !slices.Equal(bound.Dims(), []int{batch, 8}) {
				t.Fatalf("failed rebinding changed original shape: (%v, %v)", bound, err)
			}
		})
	}
}
