package builder

import "fmt"

func ExampleFuncBuilder_Complex() {
	b := NewFunc("main", Type{Shape: []int64{2}, ElementType: "f32"}, Type{Shape: []int64{2}, ElementType: "f32"})
	z, _ := b.Complex(b.Arg(0), b.Arg(1))
	fmt.Println(z.Type)
	// Output:
	// tensor<2xcomplex<f32>>
}

func ExampleFuncBuilder_Conj() {
	b := NewFunc("main", Type{ElementType: "complex<f64>"})
	z, _ := b.Conj(b.Arg(0))
	fmt.Println(z.Type)
	// Output:
	// tensor<complex<f64>>
}

func ExampleFuncBuilder_Real() {
	b := NewFunc("main", Type{Shape: []int64{2}, ElementType: "complex<f32>"})
	z, _ := b.Real(b.Arg(0))
	fmt.Println(z.Type)
	// Output:
	// tensor<2xf32>
}

func ExampleFuncBuilder_Imag() {
	b := NewFunc("main", Type{Shape: []int64{2}, ElementType: "complex<f32>"})
	z, _ := b.Imag(b.Arg(0))
	fmt.Println(z.Type)
	// Output:
	// tensor<2xf32>
}

func ExampleFuncBuilder_ComplexConstant() {
	b := NewFunc("main")
	z, _ := b.ComplexConstant(1, -2, Type{Shape: []int64{2}, ElementType: "complex<f32>"})
	fmt.Println(z.Type)
	// Output:
	// tensor<2xcomplex<f32>>
}
