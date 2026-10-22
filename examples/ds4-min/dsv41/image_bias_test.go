package dsv41

import (
	"math"
	"testing"

	"github.com/tmc/lib/gguf"
)

func TestModelOptionalImageBias(t *testing.T) {
	const name = "blk.7.exp_probs_b_vl.bias"
	for _, tc := range []struct {
		name    string
		present bool
		mutate  func(*gguf.Tensor)
		ok      bool
	}{
		{"absent", false, nil, true},
		{"valid", true, nil, true},
		{"shape", true, func(d *gguf.Tensor) { d.Dimensions[0] = 383 }, false},
		{"rank", true, func(d *gguf.Tensor) { d.Dimensions = []uint64{384, 1} }, false},
		{"encoding", true, func(d *gguf.Tensor) { d.TypeID = 1 }, false},
		{"overflow", true, func(d *gguf.Tensor) { d.Offset = math.MaxUint64 }, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			g := modelFixture(t)
			if tc.present {
				d := gguf.Tensor{Name: name, TypeID: 0, Dimensions: []uint64{384}, Offset: g.Size - g.DataOffset}
				g.Size += 384 * 4
				if tc.mutate != nil {
					tc.mutate(&d)
				}
				g.Tensors = append(g.Tensors, d)
			}
			m, err := Bind(new(countReader), int64(g.Size), g)
			if (err == nil) != tc.ok {
				t.Fatalf("optional image bind ok=%v: %v", tc.ok, err)
			}
			if !tc.ok {
				if m != nil {
					t.Fatal("invalid optional image tensor published model")
				}
				return
			}
			d, ok := m.Tensor(name)
			if ok != tc.present {
				t.Fatalf("optional descriptor present=%v want=%v", ok, tc.present)
			}
			if ok {
				d.Dimensions[0] = 1
				again, _ := m.Tensor(name)
				if again.Dimensions[0] != 384 {
					t.Fatal("optional image descriptor dimensions aliased")
				}
			}
		})
	}
}
