package gomlx

import (
	"math"
	"os"
	"reflect"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
)

func TestNativeReductionIdentities(t *testing.T) {
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
	for _, tc := range []struct {
		name  string
		dtype dtypes.DType
		op    string
		want  any
	}{
		{"float min", dtypes.Float32, "min", []float32{float32(math.Inf(1))}},
		{"signed8 min", dtypes.Int8, "min", []int8{math.MaxInt8}},
		{"signed32 min", dtypes.Int32, "min", []int32{math.MaxInt32}},
		{"signed64 min", dtypes.Int64, "min", []int64{math.MaxInt64}},
		{"unsigned8 min", dtypes.Uint8, "min", []uint8{math.MaxUint8}},
		{"unsigned32 min", dtypes.Uint32, "min", []uint32{math.MaxUint32}},
		{"unsigned64 min", dtypes.Uint64, "min", []uint64{math.MaxUint64}},
		{"signed and", dtypes.Int64, "and", []int64{-1}},
		{"unsigned and", dtypes.Uint64, "and", []uint64{math.MaxUint64}},
		{"unsigned or", dtypes.Uint64, "or", []uint64{0}},
		{"unsigned xor", dtypes.Uint64, "xor", []uint64{0}},
		{"bool and", dtypes.Bool, "and", []bool{true}},
		{"bool or", dtypes.Bool, "or", []bool{false}},
		{"bool xor", dtypes.Bool, "xor", []bool{false}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			builder := b.Builder("identity")
			f := builder.Main()
			shape := shapes.Make(tc.dtype, 0)
			x, err := f.Parameter("x", shape, nil)
			if err != nil {
				t.Fatal(err)
			}
			var y compute.Value
			switch tc.op {
			case "min":
				y, err = f.ReduceMin(x)
			case "and":
				if tc.dtype == dtypes.Bool {
					y, err = f.ReduceLogicalAnd(x)
				} else {
					y, err = f.ReduceBitwiseAnd(x)
				}
			case "or":
				if tc.dtype == dtypes.Bool {
					y, err = f.ReduceLogicalOr(x)
				} else {
					y, err = f.ReduceBitwiseOr(x)
				}
			case "xor":
				if tc.dtype == dtypes.Bool {
					y, err = f.ReduceLogicalXor(x)
				} else {
					y, err = f.ReduceBitwiseXor(x)
				}
			}
			if err != nil {
				t.Fatal(err)
			}
			if err := f.Return([]compute.Value{y}, nil); err != nil {
				t.Fatal(err)
			}
			e, err := builder.Compile()
			if err != nil {
				t.Fatal(err)
			}
			defer e.Finalize()
			in, err := b.BufferFromFlatData(0, newFlat(tc.dtype, 0), shape)
			if err != nil {
				t.Fatal(err)
			}
			defer in.Finalize()
			out, err := e.Execute([]compute.Buffer{in}, nil, 0)
			if err != nil {
				t.Fatal(err)
			}
			for _, v := range out {
				defer v.Finalize()
				got := newFlat(tc.dtype, 1)
				if err := v.ToFlatData(got); err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(got, tc.want) {
					t.Fatalf("got%v want%v", got, tc.want)
				}
			}
		})
	}
}
