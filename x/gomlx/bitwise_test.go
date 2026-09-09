package gomlx

import (
	"fmt"
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

func TestNativeShifts(t *testing.T) {
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

	// Signed Int32: ShiftLeft, ShiftRightArithmetic, ShiftRightLogical
	t.Run("Int32ShiftsAndNegativeValues", func(t *testing.T) {
		vals := []int32{1, -2, -16, 1024, -1}
		counts := []int32{0, 1, 2, 4, 31}

		out, err := testutil.Exec(b, []any{vals, counts}, func(f compute.Function, v []compute.Value) ([]compute.Value, error) {
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

		gotSHL := out[0].([]int32)
		gotSRA := out[1].([]int32)
		gotSRL := out[2].([]int32)

		for i := range vals {
			x := vals[i]
			c := uint(counts[i])

			wantSHL := int32(uint32(x) << c)
			wantSRA := x >> c
			wantSRL := int32(uint32(x) >> c)

			if gotSHL[i] != wantSHL {
				t.Fatalf("[%d] SHL %d << %d: got %d, want %d", i, x, c, gotSHL[i], wantSHL)
			}
			if gotSRA[i] != wantSRA {
				t.Fatalf("[%d] SRA %d >> %d: got %d, want %d", i, x, c, gotSRA[i], wantSRA)
			}
			if gotSRL[i] != wantSRL {
				t.Fatalf("[%d] SRL %d >>> %d: got %d, want %d", i, x, c, gotSRL[i], wantSRL)
			}
		}
	})

	// Unsigned Uint32
	t.Run("Uint32Shifts", func(t *testing.T) {
		vals := []uint32{0x80000000, 0xffffffff, 1, 0x12345678}
		counts := []uint32{1, 4, 31, 8}

		out, err := testutil.Exec(b, []any{vals, counts}, func(f compute.Function, v []compute.Value) ([]compute.Value, error) {
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

		gotSHL := out[0].([]uint32)
		gotSRA := out[1].([]uint32)
		gotSRL := out[2].([]uint32)

		for i := range vals {
			x := vals[i]
			c := counts[i]
			wantSHL := x << c
			wantSRA := uint32(int32(x) >> c) // shift_right_arithmetic treats MSB as sign bit
			wantSRL := x >> c

			if gotSHL[i] != wantSHL {
				t.Fatalf("uint32 SHL: got %x, want %x", gotSHL[i], wantSHL)
			}
			if gotSRA[i] != wantSRA {
				t.Fatalf("uint32 SRA: got %x, want %x", gotSRA[i], wantSRA)
			}
			if gotSRL[i] != wantSRL {
				t.Fatalf("uint32 SRL: got %x, want %x", gotSRL[i], wantSRL)
			}
		}
	})

	// Full 18-combination matrix table-driven test: 6 integer dtypes x 3 shift operations
	t.Run("All18CombinationsMatrix", func(t *testing.T) {
		table := []struct {
			name   string
			x, c   any
			op     func(f compute.Function, lhs, rhs compute.Value) (compute.Value, error)
			want   any
		}{
			// Int8
			{"Int8_SHL", []int8{16, -16}, []int8{2, 2}, func(f compute.Function, l, r compute.Value) (compute.Value, error) { return f.ShiftLeft(l, r) }, []int8{64, -64}},
			{"Int8_SRA", []int8{16, -16}, []int8{2, 2}, func(f compute.Function, l, r compute.Value) (compute.Value, error) { return f.ShiftRightArithmetic(l, r) }, []int8{4, -4}},
			{"Int8_SRL", []int8{16, -16}, []int8{2, 2}, func(f compute.Function, l, r compute.Value) (compute.Value, error) { return f.ShiftRightLogical(l, r) }, []int8{4, 60}},

			// Uint8 (including high-bit arithmetic right shift sign-extension)
			{"Uint8_SHL", []uint8{0x80, 16}, []uint8{1, 1}, func(f compute.Function, l, r compute.Value) (compute.Value, error) { return f.ShiftLeft(l, r) }, []uint8{0, 32}},
			{"Uint8_SRA", []uint8{0x80, 16}, []uint8{1, 1}, func(f compute.Function, l, r compute.Value) (compute.Value, error) { return f.ShiftRightArithmetic(l, r) }, []uint8{192, 8}},
			{"Uint8_SRL", []uint8{0x80, 16}, []uint8{1, 1}, func(f compute.Function, l, r compute.Value) (compute.Value, error) { return f.ShiftRightLogical(l, r) }, []uint8{64, 8}},

			// Int32
			{"Int32_SHL", []int32{16, -16}, []int32{2, 2}, func(f compute.Function, l, r compute.Value) (compute.Value, error) { return f.ShiftLeft(l, r) }, []int32{64, -64}},
			{"Int32_SRA", []int32{16, -16}, []int32{2, 2}, func(f compute.Function, l, r compute.Value) (compute.Value, error) { return f.ShiftRightArithmetic(l, r) }, []int32{4, -4}},
			{"Int32_SRL", []int32{16, -16}, []int32{2, 2}, func(f compute.Function, l, r compute.Value) (compute.Value, error) { return f.ShiftRightLogical(l, r) }, []int32{4, 1073741820}},

			// Uint32 (including high-bit arithmetic right shift sign-extension)
			{"Uint32_SHL", []uint32{0x80000000, 16}, []uint32{1, 1}, func(f compute.Function, l, r compute.Value) (compute.Value, error) { return f.ShiftLeft(l, r) }, []uint32{0, 32}},
			{"Uint32_SRA", []uint32{0x80000000, 16}, []uint32{1, 1}, func(f compute.Function, l, r compute.Value) (compute.Value, error) { return f.ShiftRightArithmetic(l, r) }, []uint32{0xc0000000, 8}},
			{"Uint32_SRL", []uint32{0x80000000, 16}, []uint32{1, 1}, func(f compute.Function, l, r compute.Value) (compute.Value, error) { return f.ShiftRightLogical(l, r) }, []uint32{0x40000000, 8}},

			// Int64
			{"Int64_SHL", []int64{16, -16}, []int64{2, 2}, func(f compute.Function, l, r compute.Value) (compute.Value, error) { return f.ShiftLeft(l, r) }, []int64{64, -64}},
			{"Int64_SRA", []int64{16, -16}, []int64{2, 2}, func(f compute.Function, l, r compute.Value) (compute.Value, error) { return f.ShiftRightArithmetic(l, r) }, []int64{4, -4}},
			{"Int64_SRL", []int64{16, -16}, []int64{2, 2}, func(f compute.Function, l, r compute.Value) (compute.Value, error) { return f.ShiftRightLogical(l, r) }, []int64{4, 4611686018427387900}},

			// Uint64 (including high-bit arithmetic right shift sign-extension)
			{"Uint64_SHL", []uint64{0x8000000000000000, 16}, []uint64{1, 1}, func(f compute.Function, l, r compute.Value) (compute.Value, error) { return f.ShiftLeft(l, r) }, []uint64{0, 32}},
			{"Uint64_SRA", []uint64{0x8000000000000000, 16}, []uint64{1, 1}, func(f compute.Function, l, r compute.Value) (compute.Value, error) { return f.ShiftRightArithmetic(l, r) }, []uint64{0xc000000000000000, 8}},
			{"Uint64_SRL", []uint64{0x8000000000000000, 16}, []uint64{1, 1}, func(f compute.Function, l, r compute.Value) (compute.Value, error) { return f.ShiftRightLogical(l, r) }, []uint64{0x4000000000000000, 8}},
		}

		for _, tc := range table {
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
	})

	// Boundary and overflow counts on Int32: count == width-1, count == width, count > width, negative counts
	t.Run("CountBoundariesAndOversizedCounts", func(t *testing.T) {
		// Observations scoped to PJRT CPU plugin for Int32 fixtures (values [16, -16]):
		// - At count 31 (width - 1):
		//   SHL produces [0 0] (since 16<<31 overflows 32 bits, -16<<31 overflows)
		//   SRA produces [0 -1] (sign extension)
		//   SRL produces [0 1] (-16 = 0xfffffff0; >>> 31 gives 1)
		// - At count 32 (count == width), 33, 64 (count > width), and -1 (count < 0):
		//   SHL produces [0 0]
		//   SRL produces [0 0]
		//   SRA produces [0 -1]
		tests := []struct {
			count   int32
			wantSHL []int32
			wantSRA []int32
			wantSRL []int32
		}{
			{31, []int32{0, 0}, []int32{0, -1}, []int32{0, 1}},
			{32, []int32{0, 0}, []int32{0, -1}, []int32{0, 0}},
			{33, []int32{0, 0}, []int32{0, -1}, []int32{0, 0}},
			{64, []int32{0, 0}, []int32{0, -1}, []int32{0, 0}},
			{-1, []int32{0, 0}, []int32{0, -1}, []int32{0, 0}},
		}
		vals := []int32{16, -16}

		for _, tc := range tests {
			t.Run(fmt.Sprintf("count_%d", tc.count), func(t *testing.T) {
				out, err := testutil.Exec(b, []any{vals, []int32{tc.count, tc.count}}, func(f compute.Function, v []compute.Value) ([]compute.Value, error) {
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
					t.Fatalf("count %d returned error: %v", tc.count, err)
				}
				shl := out[0].([]int32)
				sra := out[1].([]int32)
				srl := out[2].([]int32)

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
	})

	// Broadcasting: [3, 1] vs [1, 2] -> [3, 2]
	t.Run("Broadcasting", func(t *testing.T) {
		xs := [][]int32{{1}, {2}, {4}}
		counts := [][]int32{{1, 2}}

		out, err := testutil.Exec1(b, []any{xs, counts}, func(f compute.Function, v []compute.Value) (compute.Value, error) {
			return f.ShiftLeft(v[0], v[1])
		})
		if err != nil {
			t.Fatal(err)
		}
		got := out.([][]int32)
		want := [][]int32{
			{2, 4},
			{4, 8},
			{8, 16},
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("broadcast shift: got %v, want %v", got, want)
		}
	})
}

func TestShiftValidation(t *testing.T) {
	b := &Backend{}
	f := b.Builder("val").Main()

	pFloat, err := f.Parameter("pF", shapes.Make(dtypes.Float32, 2), nil)
	if err != nil {
		t.Fatal(err)
	}
	pInt, err := f.Parameter("pI", shapes.Make(dtypes.Int32, 2), nil)
	if err != nil {
		t.Fatal(err)
	}
	pInt64, err := f.Parameter("pI64", shapes.Make(dtypes.Int64, 2), nil)
	if err != nil {
		t.Fatal(err)
	}

	// Float operands rejected
	if _, err := f.ShiftLeft(pFloat, pFloat); err == nil {
		t.Fatal("expected error on float ShiftLeft, got nil")
	}
	if _, err := f.ShiftRightArithmetic(pFloat, pFloat); err == nil {
		t.Fatal("expected error on float ShiftRightArithmetic, got nil")
	}
	if _, err := f.ShiftRightLogical(pFloat, pFloat); err == nil {
		t.Fatal("expected error on float ShiftRightLogical, got nil")
	}

	// Mismatched integer types rejected
	if _, err := f.ShiftLeft(pInt, pInt64); err == nil {
		t.Fatal("expected error on mismatched integer types, got nil")
	}
}
