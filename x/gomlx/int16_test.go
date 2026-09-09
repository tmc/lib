package gomlx

import (
	"fmt"
	"math"
	"os"
	"reflect"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/testutil"
	"github.com/gomlx/gomlx/core/graph"
)

func TestNativeInt16Uint16(t *testing.T) {
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

	// 1. Host transfer roundtrips (scalar, vector, empty) and buffer byte sizing
	t.Run("HostTransferRoundtrips", func(t *testing.T) {
		// Signed Int16
		i16Vals := []int16{0, 1, -1, math.MaxInt16, math.MinInt16, -1234, 5678}
		bI16, err := b.BufferFromFlatData(0, i16Vals, shapes.Make(dtypes.Int16, len(i16Vals)))
		if err != nil {
			t.Fatalf("BufferFromFlatData Int16: %v", err)
		}
		defer bI16.Finalize()

		sI16, err := bI16.Shape()
		if err != nil {
			t.Fatal(err)
		}
		if sI16.DType != dtypes.Int16 || sI16.Size() != len(i16Vals) {
			t.Fatalf("Int16 shape: got %v", sI16)
		}
		gotI16 := make([]int16, len(i16Vals))
		if err := bI16.ToFlatData(gotI16); err != nil {
			t.Fatalf("ToFlatData Int16: %v", err)
		}
		if !reflect.DeepEqual(gotI16, i16Vals) {
			t.Fatalf("Int16 roundtrip: got %v, want %v", gotI16, i16Vals)
		}

		// Unsigned Uint16
		ui16Vals := []uint16{0, 1, math.MaxUint16, 0x8000, 0x7fff, 12345}
		bUI16, err := b.BufferFromFlatData(0, ui16Vals, shapes.Make(dtypes.Uint16, len(ui16Vals)))
		if err != nil {
			t.Fatalf("BufferFromFlatData Uint16: %v", err)
		}
		defer bUI16.Finalize()

		sUI16, err := bUI16.Shape()
		if err != nil {
			t.Fatal(err)
		}
		if sUI16.DType != dtypes.Uint16 || sUI16.Size() != len(ui16Vals) {
			t.Fatalf("Uint16 shape: got %v", sUI16)
		}
		gotUI16 := make([]uint16, len(ui16Vals))
		if err := bUI16.ToFlatData(gotUI16); err != nil {
			t.Fatalf("ToFlatData Uint16: %v", err)
		}
		if !reflect.DeepEqual(gotUI16, ui16Vals) {
			t.Fatalf("Uint16 roundtrip: got %v, want %v", gotUI16, ui16Vals)
		}

		// Scalar roundtrips
		bScalar, err := b.BufferFromFlatData(0, []int16{-42}, shapes.Make(dtypes.Int16))
		if err != nil {
			t.Fatalf("Scalar buffer: %v", err)
		}
		defer bScalar.Finalize()
		gotScalar := make([]int16, 1)
		if err := bScalar.ToFlatData(gotScalar); err != nil {
			t.Fatal(err)
		}
		if gotScalar[0] != -42 {
			t.Fatalf("scalar got %d, want -42", gotScalar[0])
		}
		bUnsignedScalar, err := b.BufferFromFlatData(0, []uint16{0x8000}, shapes.Make(dtypes.Uint16))
		if err != nil {
			t.Fatalf("unsigned scalar buffer: %v", err)
		}
		defer bUnsignedScalar.Finalize()
		gotUnsignedScalar := make([]uint16, 1)
		if err := bUnsignedScalar.ToFlatData(gotUnsignedScalar); err != nil {
			t.Fatal(err)
		}
		if gotUnsignedScalar[0] != 0x8000 {
			t.Fatalf("unsigned scalar got %x, want 8000", gotUnsignedScalar[0])
		}

		// Empty shape roundtrips
		bEmpty, err := b.BufferFromFlatData(0, []uint16{}, shapes.Make(dtypes.Uint16, 0))
		if err != nil {
			t.Fatalf("Empty buffer: %v", err)
		}
		defer bEmpty.Finalize()
		gotEmpty := make([]uint16, 0)
		if err := bEmpty.ToFlatData(gotEmpty); err != nil {
			t.Fatal(err)
		}
		if len(gotEmpty) != 0 {
			t.Fatalf("expected empty data, got len %d", len(gotEmpty))
		}
		bSignedEmpty, err := b.BufferFromFlatData(0, []int16{}, shapes.Make(dtypes.Int16, 0))
		if err != nil {
			t.Fatalf("signed empty buffer: %v", err)
		}
		defer bSignedEmpty.Finalize()
		if err := bSignedEmpty.ToFlatData(make([]int16, 0)); err != nil {
			t.Fatal(err)
		}

		if _, err := b.BufferFromFlatData(0, []int16{1}, shapes.Make(dtypes.Uint16, 1)); err == nil {
			t.Fatal("accepted mismatched transfer dtype")
		}
		if _, err := b.BufferFromFlatData(0, []uint16{1}, shapes.Make(dtypes.Uint16, 2)); err == nil {
			t.Fatal("accepted mismatched transfer length")
		}
		if err := bI16.ToFlatData(make([]int16, len(i16Vals)-1)); err == nil {
			t.Fatal("accepted mismatched readback length")
		}
		if err := bUI16.ToFlatData(make([]int16, len(ui16Vals))); err == nil {
			t.Fatal("accepted mismatched readback dtype")
		}
	})

	// 2. Constants
	t.Run("Constants", func(t *testing.T) {
		builder := b.Builder("const_test")
		f := builder.Main()
		cI16, err := f.Constant([]int16{math.MinInt16, -1, 0, 1, math.MaxInt16}, 5)
		if err != nil {
			t.Fatalf("Constant Int16: %v", err)
		}
		cUI16, err := f.Constant([]uint16{0, 1, 0x8000, math.MaxUint16}, 4)
		if err != nil {
			t.Fatalf("Constant Uint16: %v", err)
		}
		if err := f.Return([]compute.Value{cI16, cUI16}, nil); err != nil {
			t.Fatal(err)
		}
		exe, err := builder.Compile()
		if err != nil {
			t.Fatalf("Compile constants: %v", err)
		}
		defer exe.Finalize()

		out, err := exe.Execute(nil, nil, 0)
		if err != nil {
			t.Fatalf("Execute constants: %v", err)
		}
		defer out[0].Finalize()
		defer out[1].Finalize()

		gotI16 := make([]int16, 5)
		if err := out[0].ToFlatData(gotI16); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(gotI16, []int16{math.MinInt16, -1, 0, 1, math.MaxInt16}) {
			t.Fatalf("got %v", gotI16)
		}

		gotUI16 := make([]uint16, 4)
		if err := out[1].ToFlatData(gotUI16); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(gotUI16, []uint16{0, 1, 0x8000, math.MaxUint16}) {
			t.Fatalf("got %v", gotUI16)
		}
	})

	// 3. Conversions & Bitcasts
	t.Run("ConversionsAndBitcasts", func(t *testing.T) {
		// Convert Int16 -> Int32, Float32, Uint16
		i16Data := []int16{-1, 0, 1, 1000, -1000}
		eConv, err := graph.NewExec(b, func(x *graph.Node) (*graph.Node, *graph.Node, *graph.Node) {
			return graph.ConvertDType(x, dtypes.Int32),
				graph.ConvertDType(x, dtypes.Float32),
				graph.ConvertDType(x, dtypes.Uint16)
		})
		if err != nil {
			t.Fatalf("NewExec conv: %v", err)
		}
		defer eConv.Finalize()

		outConv, err := eConv.Call(i16Data)
		if err != nil {
			t.Fatalf("Call conv: %v", err)
		}
		defer outConv[0].FinalizeAll()
		defer outConv[1].FinalizeAll()
		defer outConv[2].FinalizeAll()

		gotI32 := outConv[0].Value().([]int32)
		gotF32 := outConv[1].Value().([]float32)
		gotUI16 := outConv[2].Value().([]uint16)

		wantI32 := []int32{-1, 0, 1, 1000, -1000}
		wantF32 := []float32{-1, 0, 1, 1000, -1000}
		wantUI16 := []uint16{65535, 0, 1, 1000, 64536}

		if !reflect.DeepEqual(gotI32, wantI32) {
			t.Fatalf("Convert to Int32: got %v, want %v", gotI32, wantI32)
		}
		if !reflect.DeepEqual(gotF32, wantF32) {
			t.Fatalf("Convert to Float32: got %v, want %v", gotF32, wantF32)
		}
		if !reflect.DeepEqual(gotUI16, wantUI16) {
			t.Fatalf("Convert to Uint16: got %v, want %v", gotUI16, wantUI16)
		}

		// Bitcast: Int16 <-> Uint16 (same bitwidth, bitcast_convert)
		eBitcast, err := graph.NewExec(b, func(x *graph.Node) *graph.Node {
			return graph.Bitcast(x, dtypes.Uint16)
		})
		if err != nil {
			t.Fatalf("NewExec bitcast: %v", err)
		}
		defer eBitcast.Finalize()

		outBitcast, err := eBitcast.Call([]int16{-1, 0x1234})
		if err != nil {
			t.Fatalf("Call bitcast: %v", err)
		}
		defer outBitcast[0].FinalizeAll()

		gotBitcastUI16 := outBitcast[0].Value().([]uint16)
		wantBitcastUI16 := []uint16{0xffff, 0x1234}
		if !reflect.DeepEqual(gotBitcastUI16, wantBitcastUI16) {
			t.Fatalf("Bitcast Int16->Uint16: got %x, want %x", gotBitcastUI16, wantBitcastUI16)
		}
	})

	// 4. Comparisons and Total Order
	t.Run("ComparisonsAndTotalOrder", func(t *testing.T) {
		// Signed Int16 comparisons
		xsI16 := []int16{-10, 0, 5, 20}
		ysI16 := []int16{-5, 0, 10, 15}

		eCmpI16, err := graph.NewExec(b, func(x, y *graph.Node) []*graph.Node {
			return []*graph.Node{
				graph.Equal(x, y),
				graph.NotEqual(x, y),
				graph.LessThan(x, y),
				graph.LessOrEqual(x, y),
				graph.GreaterThan(x, y),
				graph.GreaterOrEqual(x, y),
			}
		})
		if err != nil {
			t.Fatal(err)
		}
		defer eCmpI16.Finalize()

		outI16, err := eCmpI16.Call(xsI16, ysI16)
		if err != nil {
			t.Fatal(err)
		}
		for _, v := range outI16 {
			defer v.FinalizeAll()
		}

		gotEQ := outI16[0].Value().([]bool)
		gotNE := outI16[1].Value().([]bool)
		gotLT := outI16[2].Value().([]bool)
		gotLE := outI16[3].Value().([]bool)
		gotGT := outI16[4].Value().([]bool)
		gotGE := outI16[5].Value().([]bool)

		wantEQ := []bool{false, true, false, false}
		wantNE := []bool{true, false, true, true}
		wantLT := []bool{true, false, true, false}
		wantLE := []bool{true, true, true, false}
		wantGT := []bool{false, false, false, true}
		wantGE := []bool{false, true, false, true}

		if !reflect.DeepEqual(gotEQ, wantEQ) || !reflect.DeepEqual(gotNE, wantNE) ||
			!reflect.DeepEqual(gotLT, wantLT) || !reflect.DeepEqual(gotLE, wantLE) ||
			!reflect.DeepEqual(gotGT, wantGT) || !reflect.DeepEqual(gotGE, wantGE) {
			t.Fatalf("Int16 comparison mismatch")
		}

		// Unsigned Uint16 comparisons (verifying 0x8000 > 0x7fff)
		xsUI16 := []uint16{0x8000, 100}
		ysUI16 := []uint16{0x7fff, 100}

		eCmpUI16, err := graph.NewExec(b, func(x, y *graph.Node) (*graph.Node, *graph.Node) {
			return graph.GreaterThan(x, y), graph.EqualTotalOrder(x, y)
		})
		if err != nil {
			t.Fatal(err)
		}
		defer eCmpUI16.Finalize()

		outUI16, err := eCmpUI16.Call(xsUI16, ysUI16)
		if err != nil {
			t.Fatal(err)
		}
		for _, v := range outUI16 {
			defer v.FinalizeAll()
		}

		gotGT_UI := outUI16[0].Value().([]bool)
		gotTotalEQ_UI := outUI16[1].Value().([]bool)
		if !reflect.DeepEqual(gotGT_UI, []bool{true, false}) {
			t.Fatalf("Uint16 GT: got %v, want [true, false]", gotGT_UI)
		}
		if !reflect.DeepEqual(gotTotalEQ_UI, []bool{false, true}) {
			t.Fatalf("Uint16 TotalEQ: got %v, want [false, true]", gotTotalEQ_UI)
		}
	})

	// 5. Basic Arithmetic (Add, Sub, Mul, Div, Rem, Neg)
	t.Run("Arithmetic", func(t *testing.T) {
		a := []int16{10, -20, 100}
		c := []int16{3, 4, 7}

		eArith, err := graph.NewExec(b, func(x, y *graph.Node) []*graph.Node {
			add := graph.Add(x, y)
			sub := graph.Sub(x, y)
			mul := graph.Mul(x, y)
			div := graph.Div(x, y)
			rem := graph.Rem(x, y)
			return []*graph.Node{add, sub, mul, div, rem}
		})
		if err != nil {
			t.Fatal(err)
		}
		defer eArith.Finalize()

		out, err := eArith.Call(a, c)
		if err != nil {
			t.Fatal(err)
		}
		for _, v := range out {
			defer v.FinalizeAll()
		}

		gotAdd := out[0].Value().([]int16)
		gotSub := out[1].Value().([]int16)
		gotMul := out[2].Value().([]int16)
		gotDiv := out[3].Value().([]int16)
		gotRem := out[4].Value().([]int16)

		wantAdd := []int16{13, -16, 107}
		wantSub := []int16{7, -24, 93}
		wantMul := []int16{30, -80, 700}
		wantDiv := []int16{3, -5, 14}
		wantRem := []int16{1, 0, 2}

		if !reflect.DeepEqual(gotAdd, wantAdd) || !reflect.DeepEqual(gotSub, wantSub) ||
			!reflect.DeepEqual(gotMul, wantMul) || !reflect.DeepEqual(gotDiv, wantDiv) ||
			!reflect.DeepEqual(gotRem, wantRem) {
			t.Fatalf("Int16 arithmetic results mismatch")
		}
	})

	// 6. Reductions (ReduceSum, ReduceProduct, ReduceMax, ReduceMin, ReduceBitwiseAnd/Or/Xor)
	t.Run("Reductions", func(t *testing.T) {
		// Signed Int16 matrix [2, 2]:
		// [[10, -5],
		//  [20, 3]]
		vals := []int16{10, -5, 20, 3}
		out, err := testutil.Exec(b, []any{vals}, func(f compute.Function, v []compute.Value) ([]compute.Value, error) {
			sum, err := f.ReduceSum(v[0], 0)
			if err != nil {
				return nil, err
			}
			max, err := f.ReduceMax(v[0], 0)
			if err != nil {
				return nil, err
			}
			min, err := f.ReduceMin(v[0], 0)
			if err != nil {
				return nil, err
			}
			prod, err := f.ReduceProduct(v[0], 0)
			if err != nil {
				return nil, err
			}
			return []compute.Value{sum, max, min, prod}, nil
		})
		if err != nil {
			t.Fatal(err)
		}
		gotSum := out[0].(int16)
		gotMax := out[1].(int16)
		gotMin := out[2].(int16)
		gotProd := out[3].(int16)

		wantSum := int16(28)
		wantMax := int16(20)
		wantMin := int16(-5)
		wantProd := int16(-3000)

		if gotSum != wantSum || gotMax != wantMax || gotMin != wantMin || gotProd != wantProd {
			t.Fatalf("Int16 reduction got sum=%d, max=%d, min=%d, prod=%d", gotSum, gotMax, gotMin, gotProd)
		}

		// Unsigned Uint16 Reductions: Min with initial MaxUint16
		uiVals := [][]uint16{{100, 20}, {5, 300}}
		eUIRed, err := graph.NewExec(b, func(x *graph.Node) (*graph.Node, *graph.Node) {
			return graph.ReduceMin(x), graph.ReduceMax(x)
		})
		if err != nil {
			t.Fatal(err)
		}
		defer eUIRed.Finalize()

		outUI, err := eUIRed.Call(uiVals)
		if err != nil {
			t.Fatal(err)
		}
		defer outUI[0].FinalizeAll()
		defer outUI[1].FinalizeAll()

		gotUIMin := outUI[0].Value().(uint16)
		gotUIMax := outUI[1].Value().(uint16)
		if gotUIMin != 5 || gotUIMax != 300 {
			t.Fatalf("Uint16 reduction got min=%d, max=%d", gotUIMin, gotUIMax)
		}
	})

	// 7. Shifts (ShiftLeft, ShiftRightArithmetic, ShiftRightLogical) on Int16 and Uint16
	t.Run("ShiftsAndBoundaries", func(t *testing.T) {
		// Table of shifts on Int16 and Uint16
		tests := []struct {
			name string
			x, c any
			op   func(f compute.Function, l, r compute.Value) (compute.Value, error)
			want any
		}{
			// Int16
			{"Int16_SHL", []int16{16, -16}, []int16{2, 2}, func(f compute.Function, l, r compute.Value) (compute.Value, error) { return f.ShiftLeft(l, r) }, []int16{64, -64}},
			{"Int16_SRA", []int16{16, -16}, []int16{2, 2}, func(f compute.Function, l, r compute.Value) (compute.Value, error) {
				return f.ShiftRightArithmetic(l, r)
			}, []int16{4, -4}},
			{"Int16_SRL", []int16{16, -16}, []int16{2, 2}, func(f compute.Function, l, r compute.Value) (compute.Value, error) { return f.ShiftRightLogical(l, r) }, []int16{4, 16380}}, // 0xfff0 >>> 2 = 0x3ffc = 16380

			// Uint16
			{"Uint16_SHL", []uint16{0x8000, 16}, []uint16{1, 1}, func(f compute.Function, l, r compute.Value) (compute.Value, error) { return f.ShiftLeft(l, r) }, []uint16{0, 32}},
			{"Uint16_SRA", []uint16{0x8000, 16}, []uint16{1, 1}, func(f compute.Function, l, r compute.Value) (compute.Value, error) {
				return f.ShiftRightArithmetic(l, r)
			}, []uint16{0xc000, 8}}, // MSB sign-extension
			{"Uint16_SRL", []uint16{0x8000, 16}, []uint16{1, 1}, func(f compute.Function, l, r compute.Value) (compute.Value, error) { return f.ShiftRightLogical(l, r) }, []uint16{0x4000, 8}},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				got, err := testutil.Exec1(b, []any{tc.x, tc.c}, func(f compute.Function, v []compute.Value) (compute.Value, error) {
					return tc.op(f, v[0], v[1])
				})
				if err != nil {
					t.Fatalf("%s failed: %v", tc.name, err)
				}
				if !reflect.DeepEqual(got, tc.want) {
					t.Fatalf("%s: got %v, want %v", tc.name, got, tc.want)
				}
			})
		}

		// Width boundaries on Int16: count == 0, width - 1, width, and larger counts.
		boundaryTests := []struct {
			count   int16
			wantSHL []int16
			wantSRA []int16
			wantSRL []int16
		}{
			{0, []int16{16, -16}, []int16{16, -16}, []int16{16, -16}},
			{15, []int16{0, 0}, []int16{0, -1}, []int16{0, 1}}, // -16 = 0xfff0; >>> 15 = 1
			{16, []int16{0, 0}, []int16{0, -1}, []int16{0, 0}},
			{17, []int16{0, 0}, []int16{0, -1}, []int16{0, 0}},
			{32, []int16{0, 0}, []int16{0, -1}, []int16{0, 0}},
			{-1, []int16{0, 0}, []int16{0, -1}, []int16{0, 0}},
		}
		vals := []int16{16, -16}
		for _, tc := range boundaryTests {
			t.Run(fmt.Sprintf("count-%d", tc.count), func(t *testing.T) {
				out, err := testutil.Exec(b, []any{vals, []int16{tc.count, tc.count}}, func(f compute.Function, v []compute.Value) ([]compute.Value, error) {
					shl, err := f.ShiftLeft(v[0], v[1])
					if err != nil {
						return nil, err
					}
					sra, err := f.ShiftRightArithmetic(v[0], v[1])
					if err != nil {
						return nil, err
					}
					srl, err := f.ShiftRightLogical(v[0], v[1])
					if err != nil {
						return nil, err
					}
					return []compute.Value{shl, sra, srl}, nil
				})
				if err != nil {
					t.Fatalf("count %d error: %v", tc.count, err)
				}
				shl := out[0].([]int16)
				sra := out[1].([]int16)
				srl := out[2].([]int16)

				if !reflect.DeepEqual(shl, tc.wantSHL) {
					t.Fatalf("count %d SHL: got %v, want %v", tc.count, shl, tc.wantSHL)
				}
				if !reflect.DeepEqual(sra, tc.wantSRA) {
					t.Fatalf("count %d SRA: got %v, want %v", tc.count, sra, tc.wantSRA)
				}
				if !reflect.DeepEqual(srl, tc.wantSRL) {
					t.Fatalf("count %d SRL: got %v, want %v", tc.count, srl, tc.wantSRL)
				}
			})
		}

		unsignedBoundaryTests := []struct {
			count   uint16
			wantSHL []uint16
			wantSRA []uint16
			wantSRL []uint16
		}{
			{0, []uint16{16, 0x8000}, []uint16{16, 0x8000}, []uint16{16, 0x8000}},
			{15, []uint16{0, 0}, []uint16{0, 0xffff}, []uint16{0, 1}},
			{16, []uint16{0, 0}, []uint16{0, 0xffff}, []uint16{0, 0}},
			{17, []uint16{0, 0}, []uint16{0, 0xffff}, []uint16{0, 0}},
		}
		unsignedVals := []uint16{16, 0x8000}
		for _, tc := range unsignedBoundaryTests {
			t.Run(fmt.Sprintf("uint16-count-%d", tc.count), func(t *testing.T) {
				out, err := testutil.Exec(b, []any{unsignedVals, []uint16{tc.count, tc.count}}, func(f compute.Function, v []compute.Value) ([]compute.Value, error) {
					shl, err := f.ShiftLeft(v[0], v[1])
					if err != nil {
						return nil, err
					}
					sra, err := f.ShiftRightArithmetic(v[0], v[1])
					if err != nil {
						return nil, err
					}
					srl, err := f.ShiftRightLogical(v[0], v[1])
					if err != nil {
						return nil, err
					}
					return []compute.Value{shl, sra, srl}, nil
				})
				if err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(out[0], tc.wantSHL) || !reflect.DeepEqual(out[1], tc.wantSRA) || !reflect.DeepEqual(out[2], tc.wantSRL) {
					t.Fatalf("count %d: got %v, want %v/%v/%v", tc.count, out, tc.wantSHL, tc.wantSRA, tc.wantSRL)
				}
			})
		}
	})

	// 8. Invalid Input Validation
	t.Run("InvalidInputs", func(t *testing.T) {
		f := b.Builder("val_i16").Main()
		pI16, err := f.Parameter("p16", shapes.Make(dtypes.Int16, 2), nil)
		if err != nil {
			t.Fatal(err)
		}
		pUI16, err := f.Parameter("pU16", shapes.Make(dtypes.Uint16, 2), nil)
		if err != nil {
			t.Fatal(err)
		}

		// Mismatched signed/unsigned integer types in shift
		if _, err := f.ShiftLeft(pI16, pUI16); err == nil {
			t.Fatal("expected error on mismatched Int16 and Uint16 shift, got nil")
		}

		// Non-float in Atan2
		if _, err := f.Atan2(pI16, pI16); err == nil {
			t.Fatal("expected error on integer Atan2, got nil")
		}
	})
}

func TestInt16Encoding(t *testing.T) {
	for _, tc := range []struct {
		name string
		flat any
		want []byte
	}{
		{"signed", []int16{0x0102, -1, math.MinInt16}, []byte{2, 1, 0xff, 0xff, 0, 0x80}},
		{"unsigned", []uint16{0x0102, 0xffff, 0x8000}, []byte{2, 1, 0xff, 0xff, 0, 0x80}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dtype, n := flatShape(tc.flat)
			if n*2 != len(tc.want) {
				t.Fatal("test fixture has invalid byte length")
			}
			raw := make([]byte, len(tc.want))
			encodeFlat(tc.flat, raw)
			if !reflect.DeepEqual(raw, tc.want) {
				t.Fatalf("encoded %x, want %x", raw, tc.want)
			}
			decoded := newFlat(dtype, n)
			decodeFlat(decoded, raw)
			if !reflect.DeepEqual(decoded, tc.flat) {
				t.Fatalf("decoded %v, want %v", decoded, tc.flat)
			}
		})
	}
}

func TestNativeInt16Uint16Additional(t *testing.T) {
	path := os.Getenv("TMC_COMPUTE_PLUGIN")
	if path == "" {
		t.Skip("set TMC_COMPUTE_PLUGIN")
	}
	b, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()

	t.Run("Conversions", func(t *testing.T) {
		for _, tc := range []struct {
			name  string
			input any
			dtype dtypes.DType
			want  any
		}{
			{"i32 to i16 narrows", []int32{-32768, -1, 0, 32767, 65536}, dtypes.Int16, []int16{-32768, -1, 0, 32767, 0}},
			{"i16 to ui16", []int16{-1, 0, 32767, -32768}, dtypes.Uint16, []uint16{0xffff, 0, 0x7fff, 0x8000}},
			{"ui16 high bit widens", []uint16{0, 0x7fff, 0x8000, 0xffff}, dtypes.Int32, []int32{0, 0x7fff, 0x8000, 0xffff}},
			{"f32 to i16", []float32{-32768, -1, 0, 32767}, dtypes.Int16, []int16{-32768, -1, 0, 32767}},
			{"f32 to ui16", []float32{0, 1, 32768, 65535}, dtypes.Uint16, []uint16{0, 1, 32768, 65535}},
		} {
			t.Run(tc.name, func(t *testing.T) {
				got, err := testutil.Exec1(b, []any{tc.input}, func(f compute.Function, v []compute.Value) (compute.Value, error) {
					return f.ConvertDType(v[0], tc.dtype)
				})
				if err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(got, tc.want) {
					t.Fatalf("got %v, want %v", got, tc.want)
				}
			})
		}
	})

	t.Run("Bitcasts", func(t *testing.T) {
		for _, tc := range []struct {
			name  string
			input any
			dtype dtypes.DType
			want  any
		}{
			{"i16 to ui16", []int16{math.MinInt16, -1, 0, math.MaxInt16}, dtypes.Uint16, []uint16{0x8000, 0xffff, 0, 0x7fff}},
			{"ui16 to i16", []uint16{0, 0x7fff, 0x8000, 0xffff}, dtypes.Int16, []int16{0, math.MaxInt16, math.MinInt16, -1}},
			{"i16 to i8", []int16{0x0102, -1}, dtypes.Int8, [][]int8{{2, 1}, {-1, -1}}},
			{"i8 to i16", [][]int8{{2, 1}, {-1, -1}}, dtypes.Int16, []int16{0x0102, -1}},
		} {
			t.Run(tc.name, func(t *testing.T) {
				got, err := testutil.Exec1(b, []any{tc.input}, func(f compute.Function, v []compute.Value) (compute.Value, error) {
					return f.Bitcast(v[0], tc.dtype)
				})
				if err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(got, tc.want) {
					t.Fatalf("got %v, want %v", got, tc.want)
				}
			})
		}
	})

	t.Run("Comparisons", func(t *testing.T) {
		for _, tc := range []struct {
			name string
			x, y any
		}{
			{"i16", []int16{math.MinInt16, -1, 0, math.MaxInt16}, []int16{math.MinInt16, 0, -1, math.MaxInt16}},
			{"ui16", []uint16{0, 0x7fff, 0x8000, 0xffff}, []uint16{0, 0x8000, 0x7fff, 0xffff}},
		} {
			t.Run(tc.name, func(t *testing.T) {
				got, err := testutil.Exec(b, []any{tc.x, tc.y}, func(f compute.Function, v []compute.Value) ([]compute.Value, error) {
					return []compute.Value{
						mustComparison(f.Equal(v[0], v[1])), mustComparison(f.NotEqual(v[0], v[1])),
						mustComparison(f.LessThan(v[0], v[1])), mustComparison(f.LessOrEqual(v[0], v[1])),
						mustComparison(f.GreaterThan(v[0], v[1])), mustComparison(f.GreaterOrEqual(v[0], v[1])),
						mustComparison(f.EqualTotalOrder(v[0], v[1])), mustComparison(f.NotEqualTotalOrder(v[0], v[1])),
						mustComparison(f.LessThanTotalOrder(v[0], v[1])), mustComparison(f.LessOrEqualTotalOrder(v[0], v[1])),
						mustComparison(f.GreaterThanTotalOrder(v[0], v[1])), mustComparison(f.GreaterOrEqualTotalOrder(v[0], v[1])),
					}, nil
				})
				if err != nil {
					t.Fatal(err)
				}
				want := [][]bool{
					{true, false, false, true}, {false, true, true, false}, {false, true, false, false}, {true, true, false, true}, {false, false, true, false}, {true, false, true, true},
				}
				want = append(want, want...)
				for i := range want {
					if !reflect.DeepEqual(got[i], want[i]) {
						t.Fatalf("comparison %d: got %v, want %v", i, got[i], want[i])
					}
				}
			})
		}
	})

	t.Run("Reductions", func(t *testing.T) {
		for _, tc := range []struct {
			name  string
			input any
			want  []any
		}{
			{"i16", []int16{6, 3, 5}, []any{int16(14), int16(90), int16(0), int16(7), int16(0), int16(3), int16(6)}},
			{"ui16", []uint16{6, 3, 5}, []any{uint16(14), uint16(90), uint16(0), uint16(7), uint16(0), uint16(3), uint16(6)}},
		} {
			t.Run(tc.name, func(t *testing.T) {
				got, err := testutil.Exec(b, []any{tc.input}, func(f compute.Function, v []compute.Value) ([]compute.Value, error) {
					sum, err := f.ReduceSum(v[0])
					if err != nil {
						return nil, err
					}
					prod, err := f.ReduceProduct(v[0])
					if err != nil {
						return nil, err
					}
					and, err := f.ReduceBitwiseAnd(v[0])
					if err != nil {
						return nil, err
					}
					or, err := f.ReduceBitwiseOr(v[0])
					if err != nil {
						return nil, err
					}
					xor, err := f.ReduceBitwiseXor(v[0])
					if err != nil {
						return nil, err
					}
					min, err := f.ReduceMin(v[0])
					if err != nil {
						return nil, err
					}
					max, err := f.ReduceMax(v[0])
					if err != nil {
						return nil, err
					}
					return []compute.Value{sum, prod, and, or, xor, min, max}, nil
				})
				if err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(got, tc.want) {
					t.Fatalf("got %v, want %v", got, tc.want)
				}
			})
		}
	})

	t.Run("Uint16Arithmetic", func(t *testing.T) {
		got, err := testutil.Exec(b, []any{[]uint16{10, 20, 100}, []uint16{3, 4, 7}}, func(f compute.Function, v []compute.Value) ([]compute.Value, error) {
			add, err := f.Add(v[0], v[1])
			if err != nil {
				return nil, err
			}
			sub, err := f.Sub(v[0], v[1])
			if err != nil {
				return nil, err
			}
			mul, err := f.Mul(v[0], v[1])
			if err != nil {
				return nil, err
			}
			div, err := f.Div(v[0], v[1])
			if err != nil {
				return nil, err
			}
			rem, err := f.Rem(v[0], v[1])
			if err != nil {
				return nil, err
			}
			return []compute.Value{add, sub, mul, div, rem}, nil
		})
		if err != nil {
			t.Fatal(err)
		}
		want := []any{[]uint16{13, 24, 107}, []uint16{7, 16, 93}, []uint16{30, 80, 700}, []uint16{3, 5, 14}, []uint16{1, 0, 2}}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("got %v, want %v", got, want)
		}
	})

	t.Run("ArgMinMaxOutput", func(t *testing.T) {
		for _, dtype := range []dtypes.DType{dtypes.Int16, dtypes.Uint16} {
			t.Run(dtype.String(), func(t *testing.T) {
				input := make([]int16, 300)
				input[299] = math.MaxInt16
				got, err := testutil.Exec1(b, []any{input}, func(f compute.Function, v []compute.Value) (compute.Value, error) {
					return f.ArgMinMax(v[0], 0, dtype, false)
				})
				if err != nil {
					t.Fatal(err)
				}
				if want := map[dtypes.DType]any{dtypes.Int16: int16(299), dtypes.Uint16: uint16(299)}[dtype]; !reflect.DeepEqual(got, want) {
					t.Fatalf("got %v, want %v", got, want)
				}
			})
		}
	})

	t.Run("DynamicUpdateSliceIndices", func(t *testing.T) {
		for _, tc := range []struct {
			name  string
			start any
		}{
			{"i16", int16(99)},
			{"ui16", uint16(math.MaxUint16)},
		} {
			t.Run(tc.name, func(t *testing.T) {
				got, err := testutil.Exec1(b, []any{[]int16{1, 2, 3}, []int16{9}, tc.start}, func(f compute.Function, v []compute.Value) (compute.Value, error) {
					return f.DynamicUpdateSlice(v[0], v[1], []compute.Value{v[2]})
				})
				if err != nil {
					t.Fatal(err)
				}
				if want := []int16{1, 2, 9}; !reflect.DeepEqual(got, want) {
					t.Fatalf("got %v, want %v", got, want)
				}
			})
		}
	})

	t.Run("ReduceWindow", func(t *testing.T) {
		for _, tc := range []struct {
			name        string
			input, want any
			op          compute.ReduceOpType
		}{
			{"i16 max padding", []int16{math.MinInt16, -1}, []int16{math.MinInt16, -1, -1}, compute.ReduceOpMax},
			{"ui16 min padding", []uint16{0xffff, 3}, []uint16{0xffff, 3, 3}, compute.ReduceOpMin},
		} {
			t.Run(tc.name, func(t *testing.T) {
				got, err := testutil.Exec1(b, []any{tc.input}, func(f compute.Function, v []compute.Value) (compute.Value, error) {
					return f.ReduceWindow(v[0], tc.op, []int{2}, []int{1}, nil, nil, [][2]int{{1, 1}})
				})
				if err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(got, tc.want) {
					t.Fatalf("got %v, want %v", got, tc.want)
				}
			})
		}
	})
}

func mustComparison(v compute.Value, err error) compute.Value {
	if err != nil {
		panic(err)
	}
	return v
}
