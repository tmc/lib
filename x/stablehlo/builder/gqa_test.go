package builder

import "testing"

func TestGroupedAttentionValidation(t *testing.T) {
	for _, tt := range []struct {
		name    string
		q, k, v []int64
	}{
		{"ratio", []int64{3, 2, 4}, []int64{2, 3, 4}, []int64{2, 3, 5}},
		{"zero heads", []int64{0, 2, 4}, []int64{2, 3, 4}, []int64{2, 3, 5}},
		{"value heads", []int64{4, 2, 4}, []int64{2, 3, 4}, []int64{1, 3, 5}},
		{"batch", []int64{2, 4, 2, 4}, []int64{3, 2, 3, 4}, []int64{2, 2, 3, 5}},
		{"feature", []int64{4, 2, 4}, []int64{2, 3, 6}, []int64{2, 3, 5}},
		{"sequence", []int64{4, 2, 4}, []int64{2, 3, 4}, []int64{2, 2, 5}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			typ := func(shape []int64) Type { return Type{Shape: shape, ElementType: "f32"} }
			b := NewFunc("main", typ(tt.q), typ(tt.k), typ(tt.v))
			if _, err := b.GroupedQueryAttention(b.Arg(0), b.Arg(1), b.Arg(2)); err == nil {
				t.Fatal("accepted invalid GQA")
			}
			if len(b.Ops) != 0 || b.nextID != 0 {
				t.Fatal("invalid GQA changed builder")
			}
		})
	}
}

func TestGroupedAttentionMaskRollback(t *testing.T) {
	typ := func(heads, sequence, features int64) Type {
		return Type{Shape: []int64{heads, sequence, features}, ElementType: "bf16"}
	}
	b := NewFunc("main", typ(4, 2, 4), typ(2, 3, 4), typ(2, 3, 5), Type{Shape: []int64{2, 4}, ElementType: "i1"})
	if _, err := b.MaskedGroupedQueryAttention(b.Arg(0), b.Arg(1), b.Arg(2), b.Arg(3)); err == nil {
		t.Fatal("accepted invalid grouped mask")
	}
	if len(b.Ops) != 0 || b.nextID != 0 {
		t.Fatal("failed grouped mask left emitted operations")
	}
}
