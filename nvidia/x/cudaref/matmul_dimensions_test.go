package cudaref_test

import (
	"reflect"
	"strconv"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaref"
)

func TestMatMulRejectsNegativeDimensions(t *testing.T) {
	got, err := cudaref.MatMul([]float32{7}, []float32{9}, -1, -1, -1)
	if err == nil {
		t.Fatalf("MatMul with negative dimensions returned %v, want error", got)
	}
}

func TestMatMulRejectsDimensionProductOverflow(t *testing.T) {
	large := int(1) << (strconv.IntSize - 2)
	maxInt := int(^uint(0) >> 1)
	largeBuffer := maxInt/4 + 1
	tests := []struct {
		name    string
		a, b    []float32
		m, n, k int
	}{
		{"left input", []float32{1}, nil, large, 0, 4},
		{"right input", nil, nil, 0, large, 4},
		{"output", nil, nil, large, 4, 0},
		{"left input bytes", nil, nil, largeBuffer, 0, 1},
		{"right input bytes", nil, nil, 0, largeBuffer, 1},
		{"output bytes", nil, nil, largeBuffer, 1, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err, panicValue := matMulCapturingPanic(tt.a, tt.b, tt.m, tt.n, tt.k)
			if panicValue != nil {
				t.Fatalf("MatMul panicked: %v", panicValue)
			}
			if err == nil {
				t.Fatalf("MatMul returned %v, want dimension-overflow error", got)
			}
		})
	}
}

func matMulCapturingPanic(a, b []float32, m, n, k int) (out []float32, err error, panicValue any) {
	defer func() {
		panicValue = recover()
	}()
	out, err = cudaref.MatMul(a, b, m, n, k)
	return out, err, nil
}

func TestMatMulPreservesValidZeroExtents(t *testing.T) {
	tests := []struct {
		name    string
		a, b    []float32
		m, n, k int
		want    []float32
	}{
		{"zero reduction", nil, nil, 2, 3, 0, []float32{0, 0, 0, 0, 0, 0}},
		{"zero rows", nil, []float32{1, 2, 3, 4, 5, 6}, 0, 3, 2, []float32{}},
		{"zero columns", []float32{1, 2, 3, 4, 5, 6}, nil, 2, 0, 3, []float32{}},
		{"large empty output", nil, nil, int(1) << (strconv.IntSize - 2), 0, 0, []float32{}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := cudaref.MatMul(tt.a, tt.b, tt.m, tt.n, tt.k)
			if err != nil {
				t.Fatalf("MatMul() error = %v", err)
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("MatMul() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestMatMulChecksInputLengthsWithZeroOutput(t *testing.T) {
	tests := []struct {
		name    string
		a, b    []float32
		m, n, k int
	}{
		{"missing right input", nil, nil, 0, 3, 2},
		{"missing left input", nil, nil, 2, 0, 3},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := cudaref.MatMul(tt.a, tt.b, tt.m, tt.n, tt.k); err == nil {
				t.Fatal("MatMul() accepted an input length mismatch")
			}
		})
	}
}
