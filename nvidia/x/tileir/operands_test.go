package tileir_test

import (
	"reflect"
	"testing"

	"github.com/tmc/lib/nvidia/x/tileir"
)

func TestSplitOperands(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  []string
	}{
		{
			name:  "empty",
			input: "",
			want:  nil,
		},
		{
			name:  "whitespace only",
			input: "   ",
			want:  nil,
		},
		{
			name:  "simple ssa values",
			input: "%0, %tok1, %w_f32",
			want:  []string{"%0", "%tok1", "%w_f32"},
		},
		{
			name:  "types and shapes in angle brackets",
			input: "!cuda_tile.tile<64x64xf16>, tensor_view<1024x1024xf32>",
			want:  []string{"!cuda_tile.tile<64x64xf16>", "tensor_view<1024x1024xf32>"},
		},
		{
			name:  "nested pointer in tile",
			input: "!cuda_tile.tile<ptr<f16, none>>, !cuda_tile.token",
			want:  []string{"!cuda_tile.tile<ptr<f16, none>>", "!cuda_tile.token"},
		},
		{
			name:  "attribute blocks and loc refs",
			input: "%acc, {inbounds = true}, loc(#loc1)",
			want:  []string{"%acc", "{inbounds = true}", "loc(#loc1)"},
		},
		{
			name:  "nested braces and brackets",
			input: "[%r1, %r2], {dim = [16, 16]}, (%tok0, %tok1)",
			want:  []string{"[%r1, %r2]", "{dim = [16, 16]}", "(%tok0, %tok1)"},
		},
		{
			name:  "string literal with commas",
			input: `loc("kernel,v2.cu":14:10), %tok0`,
			want:  []string{`loc("kernel,v2.cu":14:10)`, "%tok0"},
		},
		{
			name:  "mma operands with loc",
			input: "%vA, %vB, %cst, loc(#loc6)",
			want:  []string{"%vA", "%vB", "%cst", "loc(#loc6)"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tileir.SplitOperands(tt.input)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("SplitOperands(%q) = %#v, want %#v", tt.input, got, tt.want)
			}
		})
	}
}
