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

func ExampleFuncBuilder_Softmax() {
	b := builder.NewFunc("main", builder.Type{Shape: []int64{2, 3}, ElementType: "f32"})
	out, err := b.Softmax(b.Arg(0), 1)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(out.Type)
	// Output: tensor<2x3xf32>
}

func ExampleFuncBuilder_ScaledDotProductAttention() {
	q := builder.Type{Shape: []int64{2, 3, 4}, ElementType: "f32"}
	k := builder.Type{Shape: []int64{2, 5, 4}, ElementType: "f32"}
	v := builder.Type{Shape: []int64{2, 5, 6}, ElementType: "f32"}
	b := builder.NewFunc("main", q, k, v)
	out, err := b.ScaledDotProductAttention(b.Arg(0), b.Arg(1), b.Arg(2))
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(out.Type)
	// Output: tensor<2x3x6xf32>
}

func ExampleFuncBuilder_Convert() {
	b := builder.NewFunc("main", builder.Type{Shape: []int64{4}, ElementType: "bf16"})
	out, err := b.Convert(b.Arg(0), "f32")
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(out.Type)
	// Output: tensor<4xf32>
}

func ExampleFuncBuilder_CausalMask() {
	b := builder.NewFunc("main")
	mask, err := b.CausalMask(1, 8, 7)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(mask.Type)
	// Output: tensor<1x8xi1>
}

func ExampleFuncBuilder_MaskedScaledDotProductAttention() {
	typ := builder.Type{Shape: []int64{3, 4}, ElementType: "f32"}
	b := builder.NewFunc("main", typ, typ, typ)
	mask, err := b.CausalMask(3, 3, 0)
	if err != nil {
		fmt.Println(err)
		return
	}
	out, err := b.MaskedScaledDotProductAttention(b.Arg(0), b.Arg(1), b.Arg(2), mask)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(out.Type)
	// Output: tensor<3x4xf32>
}

func ExampleFuncBuilder_And() {
	typ := builder.Type{Shape: []int64{2, 3}, ElementType: "i1"}
	b := builder.NewFunc("main", typ, typ)
	out, err := b.And(b.Arg(0), b.Arg(1))
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(out.Type)
	// Output: tensor<2x3xi1>
}

func ExampleFuncBuilder_IntegerConstant() {
	b := builder.NewFunc("main")
	out, err := b.IntegerConstant(0, builder.Type{ElementType: "i64"})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(out.Type)
	// Output: tensor<i64>
}

func ExampleFuncBuilder_DynamicUpdateSlice() {
	typ := func(n int64) builder.Type { return builder.Type{Shape: []int64{n}, ElementType: "f32"} }
	b := builder.NewFunc("main", typ(8), typ(1), builder.Type{ElementType: "i64"})
	out, err := b.DynamicUpdateSlice(b.Arg(0), b.Arg(1), []builder.Value{b.Arg(2)})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(out.Type)
	// Output: tensor<8xf32>
}

func ExampleFuncBuilder_CausalMaskAt() {
	b := builder.NewFunc("main", builder.Type{ElementType: "i64"})
	out, err := b.CausalMaskAt(1, 8, b.Arg(0))
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(out.Type)
	// Output: tensor<1x8xi1>
}

func ExampleFuncBuilder_GroupedQueryAttention() {
	typ := func(heads, sequence, features int64) builder.Type {
		return builder.Type{Shape: []int64{heads, sequence, features}, ElementType: "f32"}
	}
	b := builder.NewFunc("main", typ(4, 2, 8), typ(2, 3, 8), typ(2, 3, 6))
	out, err := b.GroupedQueryAttention(b.Arg(0), b.Arg(1), b.Arg(2))
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(out.Type)
	// Output: tensor<4x2x6xf32>
}

func ExampleFuncBuilder_MaskedGroupedQueryAttention() {
	typ := func(heads, sequence, features int64) builder.Type {
		return builder.Type{Shape: []int64{heads, sequence, features}, ElementType: "f32"}
	}
	b := builder.NewFunc("main", typ(4, 2, 8), typ(2, 3, 8), typ(2, 3, 6))
	mask, err := b.CausalMask(2, 3, 0)
	if err != nil {
		fmt.Println(err)
		return
	}
	out, err := b.MaskedGroupedQueryAttention(b.Arg(0), b.Arg(1), b.Arg(2), mask)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(out.Type)
	// Output: tensor<4x2x6xf32>
}

func ExampleFuncBuilder_ScatterRows() {
	operand := builder.Type{Shape: []int64{8, 4}, ElementType: "f32"}
	indices := builder.Type{Shape: []int64{2}, ElementType: "i64"}
	updates := builder.Type{Shape: []int64{2, 4}, ElementType: "f32"}
	b := builder.NewFunc("main", operand, indices, updates)
	out, err := b.ScatterRows(b.Arg(0), b.Arg(1), b.Arg(2))
	fmt.Println(out.Type, err)
	// Output: tensor<8x4xf32> <nil>
}

func ExampleFuncBuilder_BatchedCausalMask() {
	meta := builder.Type{Shape: []int64{2}, ElementType: "i64"}
	b := builder.NewFunc("main", meta, meta)
	mask, err := b.BatchedCausalMask(3, 8, b.Arg(0), b.Arg(1))
	fmt.Println(mask.Type, err)
	// Output: tensor<2x3x8xi1> <nil>
}

func ExampleFuncBuilder_GatherRows() {
	b := builder.NewFunc("main", builder.Type{Shape: []int64{32, 8}, ElementType: "f32"}, builder.Type{Shape: []int64{2, 3}, ElementType: "i64"})
	out, err := b.GatherRows(b.Arg(0), b.Arg(1))
	fmt.Println(out.Type, err)
	// Output: tensor<2x3x8xf32> <nil>
}

func ExampleFuncBuilder_SiLU() {
	b := builder.NewFunc("main", builder.Type{Shape: []int64{2, 8}, ElementType: "bf16"})
	out, err := b.SiLU(b.Arg(0))
	fmt.Println(out.Type, err)
	// Output: tensor<2x8xbf16> <nil>
}

func ExampleFuncBuilder_SwiGLU() {
	typ := builder.Type{Shape: []int64{2, 8}, ElementType: "f16"}
	b := builder.NewFunc("main", typ, typ)
	out, err := b.SwiGLU(b.Arg(0), b.Arg(1))
	fmt.Println(out.Type, err)
	// Output: tensor<2x8xf16> <nil>
}

func ExampleFuncBuilder_RotaryEmbedding() {
	x := builder.Type{Shape: []int64{2, 4, 3, 8}, ElementType: "bf16"}
	table := builder.Type{Shape: []int64{3, 4}, ElementType: "f32"}
	b := builder.NewFunc("main", x, table, table)
	out, err := b.RotaryEmbedding(b.Arg(0), b.Arg(1), b.Arg(2))
	fmt.Println(out.Type, err)
	// Output: tensor<2x4x3x8xbf16> <nil>
}
