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

// float64TotalOrderKey maps an IEEE 754 float64 to an unsigned integer whose natural
// order mirrors the IEEE 754-2008 totalOrder binary relation:
// -NaN (highest payload first) < -Inf < -finite < -0.0 < +0.0 < +finite < +Inf < +NaN (lowest payload first)
func float64TotalOrderKey(f float64) uint64 {
	u := math.Float64bits(f)
	if u&(1<<63) != 0 {
		return ^u
	}
	return u ^ (1 << 63)
}

func float32TotalOrderKey(f float32) uint32 {
	u := math.Float32bits(f)
	if u&(1<<31) != 0 {
		return ^u
	}
	return u ^ (1 << 31)
}

func TestNativeComparison(t *testing.T) {
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

	t.Run("DefaultFloatNotEqual", func(t *testing.T) {
		// IEEE standard comparison:
		// NaN != NaN is true
		// -0.0 != +0.0 is false
		// 1.0 != 1.0 is false
		// 1.0 != 2.0 is true
		nan := float32(math.NaN())
		negZero := float32(math.Copysign(0, -1))
		posZero := float32(0.0)

		e, err := graph.NewExec(b, func(x, y *graph.Node) []*graph.Node {
			return []*graph.Node{
				graph.Equal(x, y),
				graph.NotEqual(x, y),
			}
		})
		if err != nil {
			t.Fatal(err)
		}
		defer e.Finalize()

		xs := []float32{nan, negZero, 1.0, 1.0}
		ys := []float32{nan, posZero, 1.0, 2.0}
		out, err := e.Call(xs, ys)
		if err != nil {
			t.Fatal(err)
		}
		defer out[0].FinalizeAll()
		defer out[1].FinalizeAll()

		wantEQ := []bool{false, true, true, false}
		wantNE := []bool{true, false, false, true}

		if got := out[0].Value(); !reflect.DeepEqual(got, wantEQ) {
			t.Fatalf("Equal: got %v, want %v", got, wantEQ)
		}
		if got := out[1].Value(); !reflect.DeepEqual(got, wantNE) {
			t.Fatalf("NotEqual: got %v, want %v", got, wantNE)
		}
	})

	t.Run("Float64TotalOrderOracle", func(t *testing.T) {
		// Exercise all 6 total-order relations against the independent bitwise key oracle.
		// Includes NaN payloads, negative zero, positive zero, true subnormals, infinities, finites.
		vals := []float64{
			math.Float64frombits(0xfff8000000000002), // -NaN payload 2
			math.Float64frombits(0xfff8000000000001), // -NaN payload 1
			math.Inf(-1),                             // -Inf
			-1000.0,
			-1.0,
			math.Float64frombits(0x8000000000000002), // -subnormal 2
			math.Float64frombits(0x8000000000000001), // -subnormal 1 (smallest negative subnormal)
			math.Copysign(0, -1),                     // -0.0
			0.0,                                      // +0.0
			math.Float64frombits(0x0000000000000001), // +subnormal 1 (smallest positive subnormal)
			math.Float64frombits(0x0000000000000002), // +subnormal 2
			1.0,
			1000.0,
			math.Inf(1),                              // +Inf
			math.Float64frombits(0x7ff8000000000001), // +NaN payload 1
			math.Float64frombits(0x7ff8000000000002), // +NaN payload 2
		}

		e, err := graph.NewExec(b, func(x, y *graph.Node) []*graph.Node {
			return []*graph.Node{
				graph.EqualTotalOrder(x, y),
				graph.NotEqualTotalOrder(x, y),
				graph.LessThanTotalOrder(x, y),
				graph.LessOrEqualTotalOrder(x, y),
				graph.GreaterThanTotalOrder(x, y),
				graph.GreaterOrEqualTotalOrder(x, y),
			}
		})
		if err != nil {
			t.Fatal(err)
		}
		defer e.Finalize()

		var xs, ys []float64
		var wantEQ, wantNE, wantLT, wantLE, wantGT, wantGE []bool
		for _, x := range vals {
			kx := float64TotalOrderKey(x)
			for _, y := range vals {
				ky := float64TotalOrderKey(y)
				xs = append(xs, x)
				ys = append(ys, y)
				wantEQ = append(wantEQ, kx == ky)
				wantNE = append(wantNE, kx != ky)
				wantLT = append(wantLT, kx < ky)
				wantLE = append(wantLE, kx <= ky)
				wantGT = append(wantGT, kx > ky)
				wantGE = append(wantGE, kx >= ky)
			}
		}

		out, err := e.Call(xs, ys)
		if err != nil {
			t.Fatalf("Call: %v", err)
		}
		for _, v := range out {
			defer v.FinalizeAll()
		}

		wants := [][]bool{wantEQ, wantNE, wantLT, wantLE, wantGT, wantGE}
		names := []string{"EqualTotalOrder", "NotEqualTotalOrder", "LessThanTotalOrder", "LessOrEqualTotalOrder", "GreaterThanTotalOrder", "GreaterOrEqualTotalOrder"}
		for i, name := range names {
			got := out[i].Value().([]bool)
			if !reflect.DeepEqual(got, wants[i]) {
				for j := range got {
					if got[j] != wants[i][j] {
						t.Fatalf("%s mismatch at index %d: x=%x, y=%x: got %v, want %v",
							name, j, math.Float64bits(xs[j]), math.Float64bits(ys[j]), got[j], wants[i][j])
					}
				}
			}
		}
	})

	t.Run("Float32TotalOrderOracle", func(t *testing.T) {
		vals := []float32{
			math.Float32frombits(0xffc00002), // -NaN payload 2
			math.Float32frombits(0xffc00001), // -NaN payload 1
			float32(math.Inf(-1)),
			-100.0,
			-1.0,
			math.Float32frombits(0x80000002), // -subnormal 2
			math.Float32frombits(0x80000001), // -subnormal 1
			float32(math.Copysign(0, -1)),    // -0.0
			0.0,                              // +0.0
			math.Float32frombits(0x00000001), // +subnormal 1
			math.Float32frombits(0x00000002), // +subnormal 2
			1.0,
			100.0,
			float32(math.Inf(1)),
			math.Float32frombits(0x7fc00001), // +NaN payload 1
			math.Float32frombits(0x7fc00002), // +NaN payload 2
		}

		e, err := graph.NewExec(b, func(x, y *graph.Node) []*graph.Node {
			return []*graph.Node{
				graph.EqualTotalOrder(x, y),
				graph.NotEqualTotalOrder(x, y),
				graph.LessThanTotalOrder(x, y),
				graph.LessOrEqualTotalOrder(x, y),
				graph.GreaterThanTotalOrder(x, y),
				graph.GreaterOrEqualTotalOrder(x, y),
			}
		})
		if err != nil {
			t.Fatal(err)
		}
		defer e.Finalize()

		var xs, ys []float32
		var wantEQ, wantNE, wantLT, wantLE, wantGT, wantGE []bool
		for _, x := range vals {
			kx := float32TotalOrderKey(x)
			for _, y := range vals {
				ky := float32TotalOrderKey(y)
				xs = append(xs, x)
				ys = append(ys, y)
				wantEQ = append(wantEQ, kx == ky)
				wantNE = append(wantNE, kx != ky)
				wantLT = append(wantLT, kx < ky)
				wantLE = append(wantLE, kx <= ky)
				wantGT = append(wantGT, kx > ky)
				wantGE = append(wantGE, kx >= ky)
			}
		}

		out, err := e.Call(xs, ys)
		if err != nil {
			t.Fatalf("Call: %v", err)
		}
		for _, v := range out {
			defer v.FinalizeAll()
		}

		wants := [][]bool{wantEQ, wantNE, wantLT, wantLE, wantGT, wantGE}
		names := []string{"EqualTotalOrder", "NotEqualTotalOrder", "LessThanTotalOrder", "LessOrEqualTotalOrder", "GreaterThanTotalOrder", "GreaterOrEqualTotalOrder"}
		for i, name := range names {
			got := out[i].Value().([]bool)
			if !reflect.DeepEqual(got, wants[i]) {
				for j := range got {
					if got[j] != wants[i][j] {
						t.Fatalf("%s mismatch at index %d: x=%x, y=%x: got %v, want %v",
							name, j, math.Float32bits(xs[j]), math.Float32bits(ys[j]), got[j], wants[i][j])
					}
				}
			}
		}
	})

	t.Run("F16AndBF16TotalOrderOracle", func(t *testing.T) {
		// Float16 total order key
		f16TotalOrderKey := func(u uint16) uint16 {
			if u&(1<<15) != 0 {
				return ^u
			}
			return u ^ (1 << 15)
		}

		f16Raw := []uint16{
			0xfe02, // -NaN payload 2
			0xfe01, // -NaN payload 1
			0xfc00, // -Inf
			0xc000, // -2.0
			0x8002, // -subnormal 2
			0x8001, // -subnormal 1
			0x8000, // -0.0
			0x0000, // +0.0
			0x0001, // +subnormal 1
			0x0002, // +subnormal 2
			0x4000, // +2.0
			0x7c00, // +Inf
			0x7e01, // +NaN payload 1
			0x7e02, // +NaN payload 2
		}

		var f16Vals []float16.Float16
		for _, u := range f16Raw {
			f16Vals = append(f16Vals, float16.FromBits(u))
		}

		eF16, err := graph.NewExec(b, func(x, y *graph.Node) []*graph.Node {
			return []*graph.Node{
				graph.EqualTotalOrder(x, y),
				graph.NotEqualTotalOrder(x, y),
				graph.LessThanTotalOrder(x, y),
				graph.LessOrEqualTotalOrder(x, y),
				graph.GreaterThanTotalOrder(x, y),
				graph.GreaterOrEqualTotalOrder(x, y),
			}
		})
		if err != nil {
			t.Fatal(err)
		}
		defer eF16.Finalize()

		var xsF16, ysF16 []float16.Float16
		var wantEQ16, wantNE16, wantLT16, wantLE16, wantGT16, wantGE16 []bool
		for _, x := range f16Vals {
			kx := f16TotalOrderKey(x.Bits())
			for _, y := range f16Vals {
				ky := f16TotalOrderKey(y.Bits())
				xsF16 = append(xsF16, x)
				ysF16 = append(ysF16, y)
				wantEQ16 = append(wantEQ16, kx == ky)
				wantNE16 = append(wantNE16, kx != ky)
				wantLT16 = append(wantLT16, kx < ky)
				wantLE16 = append(wantLE16, kx <= ky)
				wantGT16 = append(wantGT16, kx > ky)
				wantGE16 = append(wantGE16, kx >= ky)
			}
		}

		outF16, err := eF16.Call(xsF16, ysF16)
		if err != nil {
			t.Fatalf("Call F16: %v", err)
		}
		for _, v := range outF16 {
			defer v.FinalizeAll()
		}

		wants16 := [][]bool{wantEQ16, wantNE16, wantLT16, wantLE16, wantGT16, wantGE16}
		names := []string{"EqualTotalOrder", "NotEqualTotalOrder", "LessThanTotalOrder", "LessOrEqualTotalOrder", "GreaterThanTotalOrder", "GreaterOrEqualTotalOrder"}
		for i, name := range names {
			got := outF16[i].Value().([]bool)
			if !reflect.DeepEqual(got, wants16[i]) {
				for j := range got {
					if got[j] != wants16[i][j] {
						t.Fatalf("F16 %s mismatch at index %d: x=%x, y=%x: got %v, want %v",
							name, j, xsF16[j].Bits(), ysF16[j].Bits(), got[j], wants16[i][j])
					}
				}
			}
		}

		// BFloat16 has 8-bit exponent and 7-bit mantissa (same as upper 16 bits of Float32).
		// Values corresponding to IEEE total order:
		// -NaN (0xffc2, 0xffc1), -Inf (0xff80), -finite (-2.0 -> 0xc000), -subnormal (0x8002, 0x8001),
		// -0.0 (0x8000), +0.0 (0x0000), +subnormal (0x0001, 0x0002), +finite (2.0 -> 0x4000),
		// +Inf (0x7f80), +NaN (0x7fc1, 0x7fc2)
		bf16Raw := []uint16{
			0xffc2, // -NaN payload 2
			0xffc1, // -NaN payload 1
			0xff80, // -Inf
			0xc000, // -2.0
			0x8002, // -subnormal 2
			0x8001, // -subnormal 1
			0x8000, // -0.0
			0x0000, // +0.0
			0x0001, // +subnormal 1
			0x0002, // +subnormal 2
			0x4000, // +2.0
			0x7f80, // +Inf
			0x7fc1, // +NaN payload 1
			0x7fc2, // +NaN payload 2
		}

		var bf16Vals []bfloat16.BFloat16
		for _, u := range bf16Raw {
			bf16Vals = append(bf16Vals, bfloat16.FromBits(u))
		}
		eBF16, err := graph.NewExec(b, func(x, y *graph.Node) []*graph.Node {
			return []*graph.Node{
				graph.EqualTotalOrder(x, y),
				graph.NotEqualTotalOrder(x, y),
				graph.LessThanTotalOrder(x, y),
				graph.LessOrEqualTotalOrder(x, y),
				graph.GreaterThanTotalOrder(x, y),
				graph.GreaterOrEqualTotalOrder(x, y),
			}
		})
		if err != nil {
			t.Fatal(err)
		}
		defer eBF16.Finalize()

		var xsBF16, ysBF16 []bfloat16.BFloat16
		var wantEQ_BF, wantNE_BF, wantLT_BF, wantLE_BF, wantGT_BF, wantGE_BF []bool
		for _, x := range bf16Vals {
			kx := f16TotalOrderKey(x.Bits())
			for _, y := range bf16Vals {
				ky := f16TotalOrderKey(y.Bits())
				xsBF16 = append(xsBF16, x)
				ysBF16 = append(ysBF16, y)
				wantEQ_BF = append(wantEQ_BF, kx == ky)
				wantNE_BF = append(wantNE_BF, kx != ky)
				wantLT_BF = append(wantLT_BF, kx < ky)
				wantLE_BF = append(wantLE_BF, kx <= ky)
				wantGT_BF = append(wantGT_BF, kx > ky)
				wantGE_BF = append(wantGE_BF, kx >= ky)
			}
		}

		outBF16, err := eBF16.Call(xsBF16, ysBF16)
		if err != nil {
			t.Fatalf("Call BF16: %v", err)
		}
		for _, v := range outBF16 {
			defer v.FinalizeAll()
		}

		wantsBF := [][]bool{wantEQ_BF, wantNE_BF, wantLT_BF, wantLE_BF, wantGT_BF, wantGE_BF}
		for i, name := range names {
			got := outBF16[i].Value().([]bool)
			if !reflect.DeepEqual(got, wantsBF[i]) {
				for j := range got {
					if got[j] != wantsBF[i][j] {
						t.Fatalf("BF16 %s mismatch at index %d: x=%x, y=%x: got %v, want %v",
							name, j, xsBF16[j].Bits(), ysBF16[j].Bits(), got[j], wantsBF[i][j])
					}
				}
			}
		}
	})

	t.Run("SemanticMutantFailsOracle", func(t *testing.T) {
		// Control test: if EqualTotalOrder were implemented using standard FLOAT compare (e.g. graph.Equal),
		// it would incorrectly return true for -0.0 == +0.0 and false for NaN == NaN.
		// Verify that our oracle asserts this failure deterministically.
		negZero := float32(math.Copysign(0, -1))
		posZero := float32(0.0)

		eMutant, err := graph.NewExec(b, func(x, y *graph.Node) *graph.Node {
			return graph.Equal(x, y) // FLOAT compare mutant
		})
		if err != nil {
			t.Fatal(err)
		}
		defer eMutant.Finalize()

		outMutant, err := eMutant.Call([]float32{negZero}, []float32{posZero})
		if err != nil {
			t.Fatal(err)
		}
		defer outMutant[0].FinalizeAll()

		mutantResult := outMutant[0].Value().([]bool)[0]
		oracleExpectation := float32TotalOrderKey(negZero) == float32TotalOrderKey(posZero) // false

		if mutantResult == oracleExpectation {
			t.Fatalf("semantic mutant (FLOAT instead of TOTALORDER) was not caught by oracle!")
		}
	})

	t.Run("ExactLargeIntegersAndBoolTotalOrder", func(t *testing.T) {
		// Exercise all six total-order relations for signed int64, unsigned uint64, and bool.
		// Standard integers and bools have total ordering equivalent to normal ordering.
		eTotalOrder, err := graph.NewExec(b, func(x, y *graph.Node) []*graph.Node {
			return []*graph.Node{
				graph.EqualTotalOrder(x, y),
				graph.NotEqualTotalOrder(x, y),
				graph.LessThanTotalOrder(x, y),
				graph.LessOrEqualTotalOrder(x, y),
				graph.GreaterThanTotalOrder(x, y),
				graph.GreaterOrEqualTotalOrder(x, y),
			}
		})
		if err != nil {
			t.Fatal(err)
		}
		defer eTotalOrder.Finalize()

		names := []string{"EqualTotalOrder", "NotEqualTotalOrder", "LessThanTotalOrder", "LessOrEqualTotalOrder", "GreaterThanTotalOrder", "GreaterOrEqualTotalOrder"}

		// 1. Signed int64 extreme values
		xsI64 := []int64{math.MinInt64, -9007199254740993, 0, 9007199254740992, math.MaxInt64}
		ysI64 := []int64{math.MinInt64 + 1, -9007199254740992, 0, 9007199254740993, math.MaxInt64}
		outI64, err := eTotalOrder.Call(xsI64, ysI64)
		if err != nil {
			t.Fatalf("Call int64: %v", err)
		}
		for _, v := range outI64 {
			defer v.FinalizeAll()
		}
		wantEQ_I64 := []bool{false, false, true, false, true}
		wantNE_I64 := []bool{true, true, false, true, false}
		wantLT_I64 := []bool{true, true, false, true, false}
		wantLE_I64 := []bool{true, true, true, true, true}
		wantGT_I64 := []bool{false, false, false, false, false}
		wantGE_I64 := []bool{false, false, true, false, true}
		wantsI64 := [][]bool{wantEQ_I64, wantNE_I64, wantLT_I64, wantLE_I64, wantGT_I64, wantGE_I64}
		for i, name := range names {
			got := outI64[i].Value().([]bool)
			if !reflect.DeepEqual(got, wantsI64[i]) {
				t.Fatalf("int64 %s: got %v, want %v", name, got, wantsI64[i])
			}
		}

		// 2. Unsigned uint64 extreme values
		xsU64 := []uint64{0, 1, 9007199254740992, math.MaxUint64 - 1, math.MaxUint64}
		ysU64 := []uint64{1, 0, 9007199254740993, math.MaxUint64, math.MaxUint64}
		outU64, err := eTotalOrder.Call(xsU64, ysU64)
		if err != nil {
			t.Fatalf("Call uint64: %v", err)
		}
		for _, v := range outU64 {
			defer v.FinalizeAll()
		}
		wantEQ_U64 := []bool{false, false, false, false, true}
		wantNE_U64 := []bool{true, true, true, true, false}
		wantLT_U64 := []bool{true, false, true, true, false}
		wantLE_U64 := []bool{true, false, true, true, true}
		wantGT_U64 := []bool{false, true, false, false, false}
		wantGE_U64 := []bool{false, true, false, false, true}
		wantsU64 := [][]bool{wantEQ_U64, wantNE_U64, wantLT_U64, wantLE_U64, wantGT_U64, wantGE_U64}
		for i, name := range names {
			got := outU64[i].Value().([]bool)
			if !reflect.DeepEqual(got, wantsU64[i]) {
				t.Fatalf("uint64 %s: got %v, want %v", name, got, wantsU64[i])
			}
		}

		// 3. Boolean total-order relations (false < true)
		xsBool := []bool{false, false, true, true}
		ysBool := []bool{false, true, false, true}
		outBool, err := eTotalOrder.Call(xsBool, ysBool)
		if err != nil {
			t.Fatalf("Call bool: %v", err)
		}
		for _, v := range outBool {
			defer v.FinalizeAll()
		}
		wantEQ_Bool := []bool{true, false, false, true}
		wantNE_Bool := []bool{false, true, true, false}
		wantLT_Bool := []bool{false, true, false, false}
		wantLE_Bool := []bool{true, true, false, true}
		wantGT_Bool := []bool{false, false, true, false}
		wantGE_Bool := []bool{true, false, true, true}
		wantsBool := [][]bool{wantEQ_Bool, wantNE_Bool, wantLT_Bool, wantLE_Bool, wantGT_Bool, wantGE_Bool}
		for i, name := range names {
			got := outBool[i].Value().([]bool)
			if !reflect.DeepEqual(got, wantsBool[i]) {
				t.Fatalf("bool %s: got %v, want %v", name, got, wantsBool[i])
			}
		}
	})

	t.Run("Broadcasting", func(t *testing.T) {
		// Scalar or broadcasted tensor comparison: [3, 1] vs [1, 2] -> [3, 2]
		e, err := graph.NewExec(b, func(x, y *graph.Node) *graph.Node {
			return graph.LessThan(x, y)
		})
		if err != nil {
			t.Fatal(err)
		}
		defer e.Finalize()

		x := [][]int32{{1}, {3}, {5}}
		y := [][]int32{{2, 4}}
		out, err := e.Call(x, y)
		if err != nil {
			t.Fatal(err)
		}
		defer out[0].FinalizeAll()

		want := [][]bool{
			{true, true},   // 1 < 2, 1 < 4
			{false, true},  // 3 < 2 (F), 3 < 4 (T)
			{false, false}, // 5 < 2 (F), 5 < 4 (F)
		}
		if got := out[0].Value(); !reflect.DeepEqual(got, want) {
			t.Fatalf("broadcast compare: got %v, want %v", got, want)
		}
	})

	t.Run("GraphConsumer", func(t *testing.T) {
		// Feed comparison results into Where and LogicalAnd downstream
		e, err := graph.NewExec(b, func(x, y, threshold *graph.Node) *graph.Node {
			diff := graph.NotEqual(x, y)
			small := graph.LessThanTotalOrder(x, threshold)
			active := graph.LogicalAnd(diff, small)
			return graph.Where(active, x, y)
		})
		if err != nil {
			t.Fatal(err)
		}
		defer e.Finalize()

		out, err := e.Call([]float32{1.0, 5.0, 2.0}, []float32{1.0, 3.0, 4.0}, float32(3.0))
		if err != nil {
			t.Fatal(err)
		}
		defer out[0].FinalizeAll()

		// x=1, y=1: diff=F -> y=1
		// x=5, y=3: diff=T, small=(5 < 3)=F -> active=F -> y=3
		// x=2, y=4: diff=T, small=(2 < 3)=T -> active=T -> x=2
		want := []float32{1.0, 3.0, 2.0}
		if got := out[0].Value(); !reflect.DeepEqual(got, want) {
			t.Fatalf("graph consumer: got %v, want %v", got, want)
		}
	})
}

func TestComparisonValidation(t *testing.T) {
	path := os.Getenv("TMC_COMPUTE_PLUGIN")
	if path == "" {
		t.Skip("set TMC_COMPUTE_PLUGIN")
	}
	b, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()

	builder := b.Builder("cmp_validation")
	f := builder.Main()

	p1, err := f.Parameter("p1", shapes.Make(dtypes.Float32, 2), nil)
	if err != nil {
		t.Fatal(err)
	}
	p2, err := f.Parameter("p2", shapes.Make(dtypes.Int32, 2), nil)
	if err != nil {
		t.Fatal(err)
	}
	pBadShape, err := f.Parameter("p3", shapes.Make(dtypes.Float32, 3), nil)
	if err != nil {
		t.Fatal(err)
	}

	// 1. Mismatched dtypes must fail
	if _, err := f.NotEqual(p1, p2); err == nil {
		t.Fatal("expected error comparing mismatched dtypes, got nil")
	} else {
		t.Logf("correctly rejected mismatched dtypes: %v", err)
	}

	// 2. Incompatible shapes must fail
	if _, err := f.EqualTotalOrder(p1, pBadShape); err == nil {
		t.Fatal("expected error comparing incompatible shapes, got nil")
	} else {
		t.Logf("correctly rejected incompatible shapes: %v", err)
	}

	// 3. Foreign/invalid value
	foreignVal := compute.Value(nil)
	if _, err := f.LessThanTotalOrder(p1, foreignVal); err == nil {
		t.Fatal("expected error on foreign nil value, got nil")
	} else {
		t.Logf("correctly rejected nil value: %v", err)
	}

	// 3b. Actual parameter from another function / builder must fail with lexical scope error
	foreignBuilder := b.Builder("foreign_builder")
	foreignFunc := foreignBuilder.Main()
	foreignParam, err := foreignFunc.Parameter("foreign_p", shapes.Make(dtypes.Float32, 2), nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.LessThanTotalOrder(p1, foreignParam); err == nil {
		t.Fatal("expected error on foreign function parameter, got nil")
	} else {
		t.Logf("correctly rejected foreign parameter from another function: %v", err)
	}

	// 4. Closed function rejection
	if err := f.Return([]compute.Value{p1}, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := f.GreaterThanTotalOrder(p1, p1); err == nil {
		t.Fatal("expected error on closed function, got nil")
	} else {
		t.Logf("correctly rejected closed function: %v", err)
	}
}
