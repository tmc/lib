package gomlx

import (
	"math"
	"os"
	"reflect"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/gomlx/core/graph"
)

func TestNativeBitcast(t *testing.T) {
	path := os.Getenv("TMC_COMPUTE_PLUGIN")
	if path == "" {
		t.Skip("set TMC_COMPUTE_PLUGIN")
	}
	b, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := b.Close(); err != nil {
			t.Error(err)
		}
	}()

	t.Run("SameWidthFloat32Uint32", func(t *testing.T) {
		// Verify raw bits, NaN payloads, negative zero, positive zero
		nanBits := uint32(0x7fc01234)
		negZeroBits := math.Float32bits(float32(math.Copysign(0, -1)))
		posZeroBits := math.Float32bits(0.0)
		normBits := math.Float32bits(1.2345)

		inputFloats := []float32{
			math.Float32frombits(nanBits),
			math.Float32frombits(negZeroBits),
			math.Float32frombits(posZeroBits),
			math.Float32frombits(normBits),
		}
		wantUints := []uint32{nanBits, negZeroBits, posZeroBits, normBits}

		e, err := graph.NewExec(b, func(x *graph.Node) *graph.Node {
			return graph.Bitcast(x, dtypes.Uint32)
		})
		if err != nil {
			t.Fatal(err)
		}
		defer e.Finalize()

		out, err := e.Call(inputFloats)
		if err != nil {
			t.Fatal(err)
		}
		defer out[0].FinalizeAll()

		if got := out[0].Value(); !reflect.DeepEqual(got, wantUints) {
			t.Fatalf("got %v, want %v", got, wantUints)
		}
	})

	t.Run("NarrowingInt32ToInt8", func(t *testing.T) {
		// Smaller target: appends trailing ratio dimension (32/8 = 4)
		// Shape [2] -> [2, 4]
		// Little-endian byte layout:
		// 0x01020304 -> [4, 3, 2, 1]
		// 0x05060708 -> [8, 7, 6, 5]
		input := []int32{0x01020304, 0x05060708}
		want := [][]int8{
			{4, 3, 2, 1},
			{8, 7, 6, 5},
		}

		e, err := graph.NewExec(b, func(x *graph.Node) *graph.Node {
			return graph.Bitcast(x, dtypes.Int8)
		})
		if err != nil {
			t.Fatal(err)
		}
		defer e.Finalize()

		out, err := e.Call(input)
		if err != nil {
			t.Fatal(err)
		}
		defer out[0].FinalizeAll()

		if got := out[0].Value(); !reflect.DeepEqual(got, want) {
			t.Fatalf("got %v, want %v", got, want)
		}
	})

	t.Run("WideningInt8ToInt32", func(t *testing.T) {
		// Larger target: collapses trailing ratio dimension (32/8 = 4)
		// Shape [2, 4] -> [2]
		input := [][]int8{
			{4, 3, 2, 1},
			{8, 7, 6, 5},
		}
		want := []int32{0x01020304, 0x05060708}

		e, err := graph.NewExec(b, func(x *graph.Node) *graph.Node {
			return graph.Bitcast(x, dtypes.Int32)
		})
		if err != nil {
			t.Fatal(err)
		}
		defer e.Finalize()

		out, err := e.Call(input)
		if err != nil {
			t.Fatal(err)
		}
		defer out[0].FinalizeAll()

		if got := out[0].Value(); !reflect.DeepEqual(got, want) {
			t.Fatalf("got %v, want %v", got, want)
		}
	})

	t.Run("Rank1ToScalarWidening", func(t *testing.T) {
		// Rank 1 [4] int8 -> scalar () int32
		input := []int8{4, 3, 2, 1}
		want := int32(0x01020304)

		e, err := graph.NewExec(b, func(x *graph.Node) *graph.Node {
			return graph.Bitcast(x, dtypes.Int32)
		})
		if err != nil {
			t.Fatal(err)
		}
		defer e.Finalize()

		out, err := e.Call(input)
		if err != nil {
			t.Fatal(err)
		}
		defer out[0].FinalizeAll()

		if got := out[0].Value(); !reflect.DeepEqual(got, want) {
			t.Fatalf("got %v, want %v", got, want)
		}
	})

	t.Run("ScalarNarrowing", func(t *testing.T) {
		// Scalar int32 -> [4] int8
		input := int32(0x01020304)
		want := []int8{4, 3, 2, 1}

		e, err := graph.NewExec(b, func(x *graph.Node) *graph.Node {
			return graph.Bitcast(x, dtypes.Int8)
		})
		if err != nil {
			t.Fatal(err)
		}
		defer e.Finalize()

		out, err := e.Call(input)
		if err != nil {
			t.Fatal(err)
		}
		defer out[0].FinalizeAll()

		if got := out[0].Value(); !reflect.DeepEqual(got, want) {
			t.Fatalf("got %v, want %v", got, want)
		}
	})

	t.Run("F16AndBF16Patterns", func(t *testing.T) {
		// Float16 <-> Uint16-like reinterpretation via Int16/Uint32
		// Or Float16 <-> Int8 narrowing: [2] Float16 -> [2, 2] Int8
		f16Val := float16.FromFloat32(3.14)
		bits := f16Val.Bits()
		b0 := int8(byte(bits))
		b1 := int8(byte(bits >> 8))

		bf16Val := bfloat16.FromFloat32(3.14)
		bfBits := bf16Val.Bits()
		bf0 := int8(byte(bfBits))
		bf1 := int8(byte(bfBits >> 8))

		eF16, err := graph.NewExec(b, func(x *graph.Node) *graph.Node {
			return graph.Bitcast(x, dtypes.Int8)
		})
		if err != nil {
			t.Fatal(err)
		}
		defer eF16.Finalize()

		outF16, err := eF16.Call([]float16.Float16{f16Val})
		if err != nil {
			t.Fatal(err)
		}
		defer outF16[0].FinalizeAll()
		wantF16 := [][]int8{{b0, b1}}
		if got := outF16[0].Value(); !reflect.DeepEqual(got, wantF16) {
			t.Fatalf("F16 bitcast: got %v, want %v", got, wantF16)
		}

		eBF16, err := graph.NewExec(b, func(x *graph.Node) *graph.Node {
			return graph.Bitcast(x, dtypes.Int8)
		})
		if err != nil {
			t.Fatal(err)
		}
		defer eBF16.Finalize()

		outBF16, err := eBF16.Call([]bfloat16.BFloat16{bf16Val})
		if err != nil {
			t.Fatal(err)
		}
		defer outBF16[0].FinalizeAll()
		wantBF16 := [][]int8{{bf0, bf1}}
		if got := outBF16[0].Value(); !reflect.DeepEqual(got, wantBF16) {
			t.Fatalf("BF16 bitcast: got %v, want %v", got, wantBF16)
		}
	})

	t.Run("EmptyShapes", func(t *testing.T) {
		// Empty same-width [0] Float32 -> [0] Uint32
		shapeIn := shapes.Make(dtypes.Float32, 0)
		inBuf, err := b.BufferFromFlatData(0, []float32{}, shapeIn)
		if err != nil {
			t.Fatal(err)
		}
		defer inBuf.Finalize()

		builder := b.Builder("empty_bitcast_test")
		f := builder.Main()
		p, err := f.Parameter("x", shapeIn, nil)
		if err != nil {
			t.Fatal(err)
		}
		node, err := f.Bitcast(p, dtypes.Uint32)
		if err != nil {
			t.Fatal(err)
		}
		if err := f.Return([]compute.Value{node}, nil); err != nil {
			t.Fatal(err)
		}
		exec, err := builder.Compile()
		if err != nil {
			t.Fatal(err)
		}
		defer exec.Finalize()

		outBufs, err := exec.Execute([]compute.Buffer{inBuf}, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		defer outBufs[0].Finalize()

		outFlat := make([]uint32, 0)
		if err := outBufs[0].ToFlatData(outFlat); err != nil {
			t.Fatal(err)
		}
		if len(outFlat) != 0 {
			t.Fatalf("expected empty outFlat, got len %d", len(outFlat))
		}

		// Empty widening [0, 4] Int8 -> [0] Int32
		shapeIn2 := shapes.Make(dtypes.Int8, 0, 4)
		inBuf2, err := b.BufferFromFlatData(0, []int8{}, shapeIn2)
		if err != nil {
			t.Fatal(err)
		}
		defer inBuf2.Finalize()

		builder2 := b.Builder("empty_bitcast_widen")
		f2 := builder2.Main()
		p2, err := f2.Parameter("x", shapeIn2, nil)
		if err != nil {
			t.Fatal(err)
		}
		node2, err := f2.Bitcast(p2, dtypes.Int32)
		if err != nil {
			t.Fatal(err)
		}
		if err := f2.Return([]compute.Value{node2}, nil); err != nil {
			t.Fatal(err)
		}
		exec2, err := builder2.Compile()
		if err != nil {
			t.Fatal(err)
		}
		defer exec2.Finalize()

		outBufs2, err := exec2.Execute([]compute.Buffer{inBuf2}, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		defer outBufs2[0].Finalize()

		outFlat2 := make([]int32, 0)
		if err := outBufs2[0].ToFlatData(outFlat2); err != nil {
			t.Fatal(err)
		}
		if len(outFlat2) != 0 {
			t.Fatalf("expected empty outFlat2, got len %d", len(outFlat2))
		}
	})

	t.Run("GraphConsumer", func(t *testing.T) {
		// Feed bitcast into arithmetic graph downstream
		e, err := graph.NewExec(b, func(x *graph.Node) *graph.Node {
			u := graph.Bitcast(x, dtypes.Uint32)
			mask := graph.Const(x.Graph(), uint32(0x7fffffff))
			masked := graph.BitwiseAnd(u, mask)
			return graph.Bitcast(masked, dtypes.Float32)
		})
		if err != nil {
			t.Fatal(err)
		}
		defer e.Finalize()

		out, err := e.Call([]float32{-1.0, -2.5, 3.5})
		if err != nil {
			t.Fatal(err)
		}
		defer out[0].FinalizeAll()

		want := []float32{1.0, 2.5, 3.5}
		if got := out[0].Value(); !reflect.DeepEqual(got, want) {
			t.Fatalf("graph consumer bitcast abs: got %v, want %v", got, want)
		}
	})

	t.Run("BooleanPackingNativeLimitation", func(t *testing.T) {
		// Bool-to-Bool identity bitcast must remain valid and return identical node.
		eBool, err := graph.NewExec(b, func(x *graph.Node) *graph.Node {
			return graph.Bitcast(x, dtypes.Bool)
		})
		if err != nil {
			t.Fatal(err)
		}
		defer eBool.Finalize()
		outBool, err := eBool.Call([]bool{true, false, true})
		if err != nil {
			t.Fatalf("bool identity bitcast failed: %v", err)
		}
		defer outBool[0].FinalizeAll()
		wantBool := []bool{true, false, true}
		if got := outBool[0].Value(); !reflect.DeepEqual(got, wantBool) {
			t.Fatalf("bool identity bitcast: got %v, want %v", got, wantBool)
		}

		// GoMLX shapes.DType.Bits() reports 8 for Bool, but StableHLO i1 is a 1-bit boolean.
		// When bitcasting between byte-sized types and Bool (i1), StableHLO requires a shape rank/dimension change,
		// and the PJRT CPU plugin rejects direct bitcast_convert between i1 and ui8/i8.
		// The adapter explicitly rejects cross-type Bool conversions with ErrNotImplemented before emitting IR.
		builder := b.Builder("bool_bitcast_test")
		f := builder.Main()
		p, err := f.Parameter("x", shapes.Make(dtypes.Uint8, 1), nil)
		if err != nil {
			t.Fatal(err)
		}
		_, err = f.Bitcast(p, dtypes.Bool)
		if err == nil {
			t.Fatalf("expected error for bitcast between ui8 and Bool, got nil")
		}
		t.Logf("correctly rejected cross-type boolean bitcasting before IR: %v", err)
	})
}

func TestBitcastValidation(t *testing.T) {
	path := os.Getenv("TMC_COMPUTE_PLUGIN")
	if path == "" {
		t.Skip("set TMC_COMPUTE_PLUGIN")
	}
	b, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()

	builder := b.Builder("validation")
	f := builder.Main()

	// 1. Scalar widening must be rejected before shapeinference panics
	scalarShape := shapes.Make(dtypes.Int8)
	paramScalar, err := f.Parameter("scalar", scalarShape, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.Bitcast(paramScalar, dtypes.Int32); err == nil {
		t.Fatal("expected error on scalar widening, got nil")
	} else {
		t.Logf("correctly rejected scalar widening: %v", err)
	}

	// 2. Invalid trailing size on widening: [3] Int8 -> Int32 requires trailing dimension 4
	shape3 := shapes.Make(dtypes.Int8, 3)
	param3, err := f.Parameter("p3", shape3, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.Bitcast(param3, dtypes.Int32); err == nil {
		t.Fatal("expected error on invalid trailing dimension ratio, got nil")
	} else {
		t.Logf("correctly rejected invalid trailing ratio: %v", err)
	}

	// 3. Unsupported target dtype before IR
	shape4 := shapes.Make(dtypes.Int8, 4)
	param4, err := f.Parameter("p4", shape4, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.Bitcast(param4, dtypes.DType(9999)); err == nil {
		t.Fatal("expected unsupported target dtype to fail, got nil")
	} else {
		t.Logf("correctly rejected unsupported target dtype: %v", err)
	}

	// 4. Closed function rejection
	if err := f.Return([]compute.Value{param4}, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := f.Bitcast(param4, dtypes.Int32); err == nil {
		t.Fatal("expected bitcast on closed function to fail, got nil")
	} else {
		t.Logf("correctly rejected closed function: %v", err)
	}
}
