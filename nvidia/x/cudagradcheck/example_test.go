package cudagradcheck_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudagradcheck"
)

func Example() {
	x := []float64{-1, 0.5, 2}
	dy := []float64{1, -2, 0.5}
	// In a kernel test, grad would be copied back from the device.
	grad := cudagradcheck.GELUBackward(x, dy)

	f := func(x []float64) float64 { return cudagradcheck.Dot(cudagradcheck.GELU(x), dy) }
	_, err := cudagradcheck.CheckGrad(f, x, grad, cudagradcheck.GradOptions{})
	fmt.Println("gradient ok:", err == nil)

	grad[1] += 0.01
	r, err := cudagradcheck.CheckGrad(f, x, grad, cudagradcheck.GradOptions{})
	fmt.Println("perturbed:", err != nil, "worst element", r.Index)
	// Output:
	// gradient ok: true
	// perturbed: true worst element 1
}

func ExampleCompare() {
	want := []float64{1.0, 2.0, 3.0}
	got := []float32{1.0001, 1.9999, 3.0002} // TF32-sized errors
	tol := cudagradcheck.Accumulated(cudagradcheck.TF32Roundoff, 64, 1)
	_, err := cudagradcheck.Compare(got, want, tol)
	fmt.Println("tf32 ok:", err == nil)
	_, err = cudagradcheck.Compare(got, want, cudagradcheck.Accumulated(cudagradcheck.Float32Roundoff, 64, 1))
	fmt.Println("float32 ok:", err == nil)
	// Output:
	// tf32 ok: true
	// float32 ok: false
}
