package cudafuse_test

import (
	"errors"
	"github.com/tmc/lib/nvidia/x/cudafuse"
	"github.com/tmc/lib/nvidia/x/cudair"
	"slices"
	"testing"
)

func TestBinaryShapeConstruction(t *testing.T) {
	for _, op := range []string{"add", "sub", "mul", "div"} {
		for _, tt := range []struct {
			name       string
			a, b, want []int
			invalid    bool
		}{
			{"equal", []int{2, 3}, []int{2, 3}, []int{2, 3}, false},
			{"scalar left", nil, []int{2, 3}, []int{2, 3}, false},
			{"scalar right", []int{2, 3}, []int{1, 1}, []int{2, 3}, false},
			{"both scalar", []int{1}, nil, []int{1}, false},
			{"empty", []int{2, 0}, []int{1}, []int{2, 0}, false},
			{"same count", []int{2, 3}, []int{6}, nil, true},
			{"axes", []int{2, 3}, []int{3, 2}, nil, true},
			{"unequal empty", []int{0}, []int{2, 0}, nil, true},
		} {
			t.Run(op+"/"+tt.name, func(t *testing.T) {
				ctx := cudafuse.NewLazyContext()
				a := cudafuse.NewLazyArray(ctx, tt.a, cudair.Float32)
				b := cudafuse.NewLazyArray(ctx, tt.b, cudair.Float32)
				r := a.Map2(op, b)
				if tt.invalid {
					if !errors.Is(r.Node.Err, cudair.ErrShapeMismatch) {
						t.Fatal(r.Node.Err)
					}
					return
				}
				if r.Node.Err != nil || !slices.Equal(r.Shape, tt.want) || !slices.Equal(r.Node.Shape, tt.want) {
					t.Fatal(r.Shape, r.Node.Err)
				}
				if ctx.Tracker.Snapshot().Allocations != 0 {
					t.Fatal("eager bookkeeping")
				}
			})
		}
	}
}
