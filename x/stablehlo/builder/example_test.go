package builder_test

import (
	"bytes"
	"fmt"
	"os"

	"github.com/tmc/lib/x/stablehlo"
	"github.com/tmc/lib/x/stablehlo/builder"
)

func ExampleFuncBuilder_Add() {
	// Create a new function builder with a 2x2 f32 tensor argument
	t := builder.Type{Shape: []int64{2, 2}, ElementType: "f32"}
	b := builder.NewFunc("main", t)
	
	// Add the argument to itself
	arg := b.Arg(0)
	res, _ := b.Add(arg, arg)
	
	// Return the result
	b.Return(res)
	
	// Emit the MLIR function
	b.Emit(os.Stdout)
	
	// Output:
	//   func.func @main(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
	//     %0 = stablehlo.add %arg0, %arg0 : tensor<2x2xf32>
	//     return %0 : tensor<2x2xf32>
	//   }
}

func ExampleFuncBuilder_DotGeneral() {
	// Create a builder for matrix multiplication
	lhsType := builder.Type{Shape: []int64{2, 3}, ElementType: "f32"}
	rhsType := builder.Type{Shape: []int64{3, 4}, ElementType: "f32"}
	retType := builder.Type{Shape: []int64{2, 4}, ElementType: "f32"}
	
	b := builder.NewFunc("matmul", lhsType, rhsType)
	
	// Perform dot_general with contracting dimensions
	res, _ := b.DotGeneral(b.Arg(0), b.Arg(1), nil, nil, []int64{1}, []int64{0}, retType)
	b.Return(res)
	
	b.Emit(os.Stdout)
	
	// Output:
	//   func.func @matmul(%arg0: tensor<2x3xf32>, %arg1: tensor<3x4xf32>) -> tensor<2x4xf32> {
	//     %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
	//     return %0 : tensor<2x4xf32>
	//   }
}

func Example() {
	// Build a module containing a main function that does an addition
	m := &stablehlo.Module{}
	
	t := builder.Type{Shape: []int64{4}, ElementType: "f32"}
	b := builder.NewFunc("main", t, t)
	
	res, _ := b.Add(b.Arg(0), b.Arg(1))
	b.Return(res)
	
	m.Funcs = append(m.Funcs, b)
	
	// Emit the full module
	var buf bytes.Buffer
	m.Emit(&buf)
	fmt.Print(buf.String())
	
	// Output:
	// module {
	//   func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
	//     %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32>
	//     return %0 : tensor<4xf32>
	//   }
	// }
}
