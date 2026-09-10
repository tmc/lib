package builder

import "fmt"

func ExampleFuncBuilder_FFT() {
	b := NewFunc("main", Type{Shape: []int64{4}, ElementType: "complex<f32>"})
	z, _ := b.FFT(b.Arg(0), FFTForward, []int64{4})
	fmt.Println(z.Type)
	// Output:
	// tensor<4xcomplex<f32>>
}
