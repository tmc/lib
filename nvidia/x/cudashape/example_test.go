package cudashape_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudashape"
)

func ExampleNew() {
	s, err := cudashape.New(2, 3, 4)
	if err != nil {
		panic(err)
	}
	n, _ := s.NumElements()
	bytes, _ := s.SizeBytes(4) // float32 elements
	fmt.Printf("shape: %s, elements: %d, bytes: %d\n", s, n, bytes)
	// Output:
	// shape: [2, 3, 4], elements: 24, bytes: 96
}

func ExampleParseDimTokens() {
	tokens, err := cudashape.ParseDimTokens("a12BC", true)
	if err != nil {
		panic(err)
	}
	fmt.Println(tokens)
	// Output:
	// [A12 B C]
}

func ExampleRowMajorStrides() {
	strides, err := cudashape.RowMajorStrides([]int{2, 3, 4})
	if err != nil {
		panic(err)
	}
	fmt.Printf("strides: %v\n", strides)
	// Output:
	// strides: [12 4 1]
}

func ExampleSymbolicShape_Bind() {
	env := cudashape.NewEnvironment()
	_ = env.AddConstraint(cudashape.DivisibleBy("hidden", 64))
	_ = env.Bind("batch", 4)
	_ = env.Bind("seq", 128)
	_ = env.Bind("hidden", 256)

	symShape := cudashape.NewSymbolicShape(
		cudashape.NamedDim("batch"),
		cudashape.NamedDim("seq"),
		cudashape.NamedDim("hidden"),
	)

	shape, err := symShape.Bind(env)
	if err != nil {
		panic(err)
	}
	fmt.Println(shape)
	// Output:
	// [4, 128, 256]
}
