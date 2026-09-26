package main

import (
	"reflect"
	"testing"
)

func TestPairReplacements(t *testing.T) {
	side := func(kernels ...any) map[string]map[string]*opTotals {
		m := make(map[string]map[string]*opTotals)
		for i := 0; i < len(kernels); i += 3 {
			op, key, n := kernels[i].(string), kernels[i+1].(string), kernels[i+2].(int)
			if m[op] == nil {
				m[op] = make(map[string]*opTotals)
			}
			m[op][key] = &opTotals{launches: n, exclusive: int64(10 * n), summed: int64(10 * n)}
		}
		return m
	}
	type row struct {
		op           string
		base, target []string
		launches     int
	}
	tests := []struct {
		name         string
		base, target map[string]map[string]*opTotals
		grouped      bool
		want         []row
	}{
		{
			name:   "prefix and count",
			base:   side("LayerNorm", "LayerNorm", 8, "LayerNormBackward", "LayerNormBackward", 4, "LayerNormParamPartial", "LayerNormParamPartial", 4, "gemm", "gemm", 4),
			target: side("LayerNormF32", "LayerNormF32", 8, "LayerNormBackwardParam", "LayerNormBackwardParam", 4, "gemm", "gemm", 4),
			want: []row{
				{"", []string{"LayerNormBackward", "LayerNormParamPartial"}, []string{"LayerNormBackwardParam"}, 4},
				{"", []string{"LayerNorm"}, []string{"LayerNormF32"}, 8},
			},
		},
		{
			name:   "no shared prefix",
			base:   side("vectorized_layer_norm", "vectorized_layer_norm", 8),
			target: side("LayerNormF32", "LayerNormF32", 8),
		},
		{
			name:   "unequal counts",
			base:   side("LayerNorm", "LayerNorm", 8),
			target: side("LayerNormF32", "LayerNormF32", 6),
		},
		{
			name:    "grouped with leftovers",
			base:    side("ln", "vectorized_layer_norm", 8, "ln", "grad_input", 4, "ln", "GammaBeta", 4, "ew", "add", 3),
			target:  side("ln", "LayerNormF32", 8, "ln", "LayerNormBackwardParam", 4, "ln", "ColumnSum", 2, "ew", "mul", 5),
			grouped: true,
			want: []row{
				{"ln", []string{"GammaBeta", "grad_input"}, []string{"LayerNormBackwardParam"}, 4},
				{"ew", []string{"add"}, []string{"mul"}, 0},
				{"ln", []string{"vectorized_layer_norm"}, []string{"LayerNormF32"}, 8},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var got []row
			for _, r := range pairReplacements(tt.base, tt.target, tt.grouped) {
				got = append(got, row{r.Op, r.Base, r.Target, r.Launches})
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("rows =\n%v\nwant\n%v", got, tt.want)
			}
		})
	}
}
