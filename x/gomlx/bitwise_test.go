package gomlx

import (
	"math/bits"
	"os"
	"reflect"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/testutil"
)

func TestNativeBitwise(t *testing.T) {
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
	for _, dtype := range []dtypes.DType{dtypes.Int8, dtypes.Int32, dtypes.Int64, dtypes.Uint8, dtypes.Uint32, dtypes.Uint64} {
		t.Run(dtype.String(), func(t *testing.T) {
			input := newFlat(dtype, 4)
			rv := reflect.ValueOf(input)
			width := rv.Type().Elem().Bits()
			mask := ^uint64(0) >> (64 - width)
			values := []uint64{0, 1, 1 << uint(width-1), mask}
			set := func(v reflect.Value, n uint64) {
				if dtype.IsUnsigned() {
					v.SetUint(n)
				} else {
					v.SetInt(int64(n))
				}
			}
			for i, n := range values {
				set(rv.Index(i), n)
			}
			other := reflect.New(rv.Type().Elem()).Elem()
			set(other, 0x55)
			want := make([]any, 5)
			for op := range want {
				flat := newFlat(dtype, 4)
				v := reflect.ValueOf(flat)
				for i, n := range values {
					var r uint64
					switch op {
					case 0:
						r = n & 0x55
					case 1:
						r = n | 0x55
					case 2:
						r = n ^ 0x55
					case 3:
						r = uint64(bits.OnesCount64(n))
					case 4:
						r = uint64(bits.LeadingZeros64(n) - (64 - width))
					}
					set(v.Index(i), r)
				}
				want[op] = flat
			}
			got, err := testutil.Exec(b, []any{input, other.Interface()}, func(f compute.Function, v []compute.Value) ([]compute.Value, error) {
				var out []compute.Value
				for _, op := range []func(compute.Value, compute.Value) (compute.Value, error){f.BitwiseAnd, f.BitwiseOr, f.BitwiseXor} {
					x, err := op(v[0], v[1])
					if err != nil {
						return nil, err
					}
					out = append(out, x)
				}
				for _, op := range []func(compute.Value) (compute.Value, error){f.BitCount, f.Clz} {
					x, err := op(v[0])
					if err != nil {
						return nil, err
					}
					out = append(out, x)
				}
				return out, nil
			})
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("got %v want %v", got, want)
			}
		})
	}
	got, err := testutil.Exec1(b, []any{[]bool{false, false, true, true}, []bool{false, true, false, true}}, func(f compute.Function, v []compute.Value) (compute.Value, error) { return f.LogicalXor(v[0], v[1]) })
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, []bool{false, true, true, false}) {
		t.Fatalf("logical xor %v", got)
	}
}

func TestBitwiseValidation(t *testing.T) {
	for _, logical := range []bool{false, true} {
		f := (&Backend{}).Builder("invalid").Main()
		dtype := dtypes.Float32
		if logical {
			dtype = dtypes.Int32
		}
		x, err := f.Parameter("x", shapes.Make(dtype, 3), nil)
		if err != nil {
			t.Fatal(err)
		}
		y, err := f.Parameter("y", shapes.Make(dtype), nil)
		if err != nil {
			t.Fatal(err)
		}
		raw := f.(*function).raw
		n := len(raw.Ops)
		ops := []func(compute.Value, compute.Value) (compute.Value, error){f.BitwiseAnd, f.BitwiseOr, f.BitwiseXor}
		if logical {
			ops = []func(compute.Value, compute.Value) (compute.Value, error){f.LogicalXor}
		}
		for _, op := range ops {
			if _, err := op(x, y); err == nil {
				t.Fatal("accepted invalid dtype")
			}
		}
		if !logical {
			for _, op := range []func(compute.Value) (compute.Value, error){f.BitCount, f.Clz} {
				if _, err := op(x); err == nil {
					t.Fatal("accepted float count")
				}
			}
		}
		if len(raw.Ops) != n {
			t.Fatal("invalid bitwise operation emitted IR")
		}
	}
}
