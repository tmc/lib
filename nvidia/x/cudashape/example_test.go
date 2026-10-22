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

	batch, err := cudashape.NamedDim("batch")
	if err != nil {
		fmt.Println(err)
		return
	}
	seq, err := cudashape.NamedDim("seq")
	if err != nil {
		fmt.Println(err)
		return
	}
	hidden, err := cudashape.NamedDim("hidden")
	if err != nil {
		fmt.Println(err)
		return
	}
	symShape := cudashape.NewSymbolicShape(batch, seq, hidden)

	shape, err := symShape.Bind(env)
	if err != nil {
		panic(err)
	}
	fmt.Println(shape)
	// Output:
	// [4, 128, 256]
}

func ExampleNamedDim() {
	dim, err := cudashape.NamedDim("batch")
	fmt.Println(dim.Name(), dim.IsSymbol(), err)
	_, err = cudashape.NamedDim("")
	fmt.Println(err)
	// Output:
	// batch true <nil>
	// cudashape: empty symbol name
}

func ExampleInferGrid() {
	tensor, err := cudashape.New(1024, 2048)
	if err != nil {
		panic(err)
	}
	tile, err := cudashape.New(64, 128)
	if err != nil {
		panic(err)
	}
	grid, err := cudashape.InferGrid(tensor, tile)
	if err != nil {
		panic(err)
	}
	fmt.Printf("grid: %s\n", grid)
	// Output:
	// grid: (16, 16, 1)
}
