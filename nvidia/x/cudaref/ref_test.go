package cudaref_test

import (
	"math"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaref"
)

func TestRefMatMul(t *testing.T) {
	// A: 2x2 [[1, 2], [3, 4]]
	a := []float32{1, 2, 3, 4}
	// B: 2x2 [[5, 6], [7, 8]]
	b := []float32{5, 6, 7, 8}

	// C = A x B = [[19, 22], [43, 50]]
	c, err := cudaref.MatMul(a, b, 2, 2, 2)
	if err != nil {
		t.Fatalf("MatMul failed: %v", err)
	}

	expected := []float32{19, 22, 43, 50}
	for i := range expected {
		if c[i] != expected[i] {
			t.Errorf("at %d: want %v, got %v", i, expected[i], c[i])
		}
	}
}

func TestRefSoftmax(t *testing.T) {
	src := []float32{1.0, 2.0, 3.0}
	res := cudaref.Softmax(src)

	var sum float32
	for _, v := range res {
		sum += v
	}

	if math.Abs(float64(sum-1.0)) > 1e-5 {
		t.Errorf("softmax sum != 1.0: %v", sum)
	}
}

func TestRefReductions(t *testing.T) {
	var warpVals [32]float32
	for i := range warpVals {
		warpVals[i] = float32(i + 1)
	}

	// 1 + 2 + ... + 32 = 32*33/2 = 528
	wSum := cudaref.WarpReduceSum(warpVals)
	if wSum != 528 {
		t.Errorf("expected 528, got %v", wSum)
	}
}
