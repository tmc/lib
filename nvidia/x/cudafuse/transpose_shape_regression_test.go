package cudafuse_test

import (
	"slices"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudafuse"
	"github.com/tmc/lib/nvidia/x/cudair"
)

func TestTransposeArrayShape(t *testing.T) {
	tests := []struct {
		name              string
		shape, perm, want []int
	}{
		{"matrix", []int{2, 3}, []int{1, 0}, []int{3, 2}},
		{"rank3", []int{2, 3, 4}, []int{2, 0, 1}, []int{4, 2, 3}},
		{"identity", []int{2, 3}, []int{0, 1}, []int{2, 3}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			input := cudafuse.NewLazyArray(cudafuse.NewLazyContext(), tt.shape, cudair.Float32)
			output := input.Transpose(tt.perm)
			if output.Node.Err != nil {
				t.Fatal(output.Node.Err)
			}
			if !slices.Equal(output.Shape, tt.want) || !slices.Equal(output.Node.Shape, tt.want) {
				t.Fatalf("array shape %v, node shape %v; want %v", output.Shape, output.Node.Shape, tt.want)
			}
			if output.DType != input.DType || output.Ctx != input.Ctx || output.Node.Inputs[0] != input.Node {
				t.Fatal("transpose changed input, context, or dtype")
			}
			output.Shape[0] = 99
			if !slices.Equal(output.Node.Shape, tt.want) || !slices.Equal(input.Shape, tt.shape) {
				t.Fatal("array shape aliases node or input shape")
			}
		})
	}
}
