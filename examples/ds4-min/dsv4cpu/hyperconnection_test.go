package dsv4cpu

import (
	"fmt"
	"testing"
)

func ExampleSplit() {
	out := make([]float32, 3)
	_ = Split(out, []float32{0, 0, 0}, []float32{1, 1, 1}, []float32{0, 0, 0}, 1, 20, 1e-6)
	fmt.Printf("%.3f %.3f %.3f\n", out[0], out[1], out[2])
	// Output: 0.500 1.000 1.000
}

func ExampleCollapse() {
	out := make([]float32, 2)
	_ = Collapse(out, []float32{1, 2, 3, 4}, []float32{.25, .75})
	fmt.Println(out)
	// Output: [2.5 3.5]
}

func ExampleCombine() {
	out := make([]float32, 4)
	_ = Combine(out, []float32{1, 2}, []float32{3, 4, 5, 6}, []float32{1, 2}, []float32{1, 0, 0, 1})
	fmt.Println(out)
	// Output: [4 6 7 10]
}

func TestHyperconnectionFullWidth(t *testing.T) {
	const dim = 4096
	state := make([]float32, 4*dim)
	for h := 0; h < 4; h++ {
		for d := 0; d < dim; d++ {
			state[h*dim+d] = float32(h + 1)
		}
	}
	x := make([]float32, dim)
	if err := Collapse(x, state, []float32{1, 2, 3, 4}); err != nil {
		t.Fatal(err)
	}
	for i, v := range x {
		if v != 30 {
			t.Fatalf("collapse %d=%g", i, v)
		}
	}
	comb := make([]float32, 16)
	for i := 0; i < 4; i++ {
		comb[i+i*4] = 1
	}
	out := make([]float32, len(state))
	if err := Combine(out, x, state, []float32{1, 2, 3, 4}, comb); err != nil {
		t.Fatal(err)
	}
	for h := 0; h < 4; h++ {
		for d := 0; d < dim; d++ {
			if out[h*dim+d] != float32(31*(h+1)) {
				t.Fatalf("combine h%d d%d", h, d)
			}
		}
	}
}
