package builder

import (
	"bytes"
	"strings"
	"testing"
)

func TestBuilderOperations(t *testing.T) {
	tests := []struct {
		name     string
		build    func() *FuncBuilder
		wantOps  []string
	}{
		{
			name: "add",
			build: func() *FuncBuilder {
				b := NewFunc("test_add", Type{Shape: []int64{2, 2}, ElementType: "f32"})
				arg := b.Arg(0)
				res, _ := b.Add(arg, arg)
				b.Return(res)
				return b
			},
			wantOps: []string{
				"%0 = stablehlo.add %arg0, %arg0 : tensor<2x2xf32>",
			},
		},
		{
			name: "dot_general",
			build: func() *FuncBuilder {
				b := NewFunc("test_dot", Type{Shape: []int64{2, 3}, ElementType: "f32"}, Type{Shape: []int64{3, 4}, ElementType: "f32"})
				res, _ := b.DotGeneral(b.Arg(0), b.Arg(1), nil, nil, []int64{1}, []int64{0}, Type{Shape: []int64{2, 4}, ElementType: "f32"})
				b.Return(res)
				return b
			},
			wantOps: []string{
				"%0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>",
			},
		},
		{
			name: "rmsnorm",
			build: func() *FuncBuilder {
				b := NewFunc("test_rmsnorm", Type{Shape: []int64{2, 4}, ElementType: "f32"}, Type{Shape: []int64{4}, ElementType: "f32"})
				res, _ := b.RMSNorm(b.Arg(0), b.Arg(1), 1, 1e-5)
				b.Return(res)
				return b
			},
			wantOps: []string{
				"%0 = stablehlo.multiply %arg0, %arg0 : tensor<2x4xf32>",
				"%1 = stablehlo.constant dense<0.000000> : tensor<f32>",
				"%2 = stablehlo.reduce(%0 init: %1) applies stablehlo.add across dimensions = [1] : (tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>",
				"%3 = stablehlo.constant dense<4.000000> : tensor<2xf32>",
				"%4 = stablehlo.divide %2, %3 : tensor<2xf32>",
				"%5 = stablehlo.constant dense<0.000010> : tensor<2xf32>",
				"%6 = stablehlo.add %4, %5 : tensor<2xf32>",
				"%7 = stablehlo.rsqrt %6 : tensor<2xf32>",
				"%8 = stablehlo.broadcast_in_dim %7, dims = [0] : (tensor<2xf32>) -> tensor<2x4xf32>",
				"%9 = stablehlo.multiply %arg0, %8 : tensor<2x4xf32>",
				"%10 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<4xf32>) -> tensor<2x4xf32>",
				"%11 = stablehlo.multiply %9, %10 : tensor<2x4xf32>",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			b := tt.build()
			var buf bytes.Buffer
			if err := b.Emit(&buf); err != nil {
				t.Fatalf("Emit failed: %v", err)
			}
			out := buf.String()
			for _, want := range tt.wantOps {
				if !strings.Contains(out, want) {
					t.Errorf("Emit output missing expected op %q\nGot:\n%s", want, out)
				}
			}
		})
	}
}
