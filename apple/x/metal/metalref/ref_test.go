package metalref

import (
	"math"
	"testing"
)

func TestBlockReduceSum(t *testing.T) {
	tests := []struct {
		name  string
		input []float32
		want  float32
	}{
		{"nil", nil, 0},
		{"values", []float32{1.5, -2, 4.25}, 3.75},
		{"sequential rounding", []float32{1e20, 1, -1e20}, 0},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := BlockReduceSum(test.input); got != test.want {
				t.Fatalf("BlockReduceSum() = %v, want %v", got, test.want)
			}
		})
	}
	if got := BlockReduceSum([]float32{1, float32(math.NaN())}); !math.IsNaN(float64(got)) {
		t.Fatalf("NaN input produced %v", got)
	}
}
