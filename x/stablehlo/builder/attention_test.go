package builder

import (
	"strings"
	"testing"
)

func TestAttention(t *testing.T) {
	q := Type{Shape: []int64{2, 3, 4}, ElementType: "f32"}
	k := Type{Shape: []int64{2, 5, 4}, ElementType: "f32"}
	v := Type{Shape: []int64{2, 5, 6}, ElementType: "f32"}
	b := NewFunc("main", q, k, v)
	out, err := b.ScaledDotProductAttention(b.Arg(0), b.Arg(1), b.Arg(2))
	if err != nil {
		t.Fatal(err)
	}
	if out.Type.String() != "tensor<2x3x6xf32>" {
		t.Fatalf("result type: %s", out.Type)
	}
	text := strings.Join(b.Ops, "\n")
	for _, part := range []string{"batching_dims = [0] x [0]", "stablehlo.maximum", "stablehlo.exponential", "0xFF800000"} {
		if !strings.Contains(text, part) {
			t.Errorf("missing %s", part)
		}
	}
}

func TestAttentionRejectsBadShape(t *testing.T) {
	for _, shape := range [][]int64{{2, 5, 7}, {3, 5, 4}, {2, 0, 4}, {2, -1, 4}, {4}} {
		b := NewFunc("main", Type{Shape: []int64{2, 3, 4}, ElementType: "f32"}, Type{Shape: shape, ElementType: "f32"}, Type{Shape: []int64{2, 5, 6}, ElementType: "f32"})
		if _, err := b.ScaledDotProductAttention(b.Arg(0), b.Arg(1), b.Arg(2)); err == nil {
			t.Fatalf("accepted key shape %v", shape)
		}
		if len(b.Ops) != 0 || b.nextID != 0 {
			t.Fatal("failed attention changed builder")
		}
	}
}

func TestSoftmaxRejectsEmptyOrDynamicAxis(t *testing.T) {
	for _, shape := range [][]int64{{2, 0}, {2, -1}} {
		b := NewFunc("main", Type{Shape: shape, ElementType: "f32"})
		if _, err := b.Softmax(b.Arg(0), 1); err == nil {
			t.Fatalf("accepted shape %v", shape)
		}
		if len(b.Ops) != 0 || b.nextID != 0 {
			t.Fatal("failed softmax changed builder")
		}
	}
}
