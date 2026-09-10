package gomlx

import (
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

func TestComplexEncoding(t *testing.T) {
	for _, tc := range []struct {
		name string
		flat any
		want []byte
	}{
		{"c64", []complex64{complex(math.Float32frombits(0x80000000), math.Float32frombits(0x7fc01234))}, []byte{0, 0, 0, 0x80, 0x34, 0x12, 0xc0, 0x7f}},
		{"c128", []complex128{complex(math.Float64frombits(0x8000000000000000), math.Float64frombits(0x7ff8000000001234))}, []byte{0, 0, 0, 0, 0, 0, 0, 0x80, 0x34, 0x12, 0, 0, 0, 0, 0xf8, 0x7f}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dtype, n := flatShape(tc.flat)
			raw := make([]byte, len(tc.want))
			encodeFlat(tc.flat, raw)
			if !reflect.DeepEqual(raw, tc.want) {
				t.Fatalf("encoded %x, want %x", raw, tc.want)
			}
			decoded := newFlat(dtype, n)
			decodeFlat(decoded, raw)
			if !reflect.DeepEqual(complexBits(decoded), complexBits(tc.flat)) {
				t.Fatalf("decoded %#v, want %#v", decoded, tc.flat)
			}
		})
	}
}

func complexBits(v any) []uint64 {
	switch x := v.(type) {
	case []complex64:
		out := make([]uint64, len(x))
		for i, z := range x {
			out[i] = uint64(math.Float32bits(real(z)))<<32 | uint64(math.Float32bits(imag(z)))
		}
		return out
	case []complex128:
		out := make([]uint64, 2*len(x))
		for i, z := range x {
			out[2*i] = math.Float64bits(real(z))
			out[2*i+1] = math.Float64bits(imag(z))
		}
		return out
	}
	return nil
}

func TestComplexValidation(t *testing.T) {
	f := (&Backend{}).Builder("complex").Main()
	f32, err := f.Parameter("f32", shapes.Make(dtypes.Float32, 2), nil)
	if err != nil {
		t.Fatal(err)
	}
	c64, err := f.Parameter("c64", shapes.Make(dtypes.Complex64, 2), nil)
	if err != nil {
		t.Fatal(err)
	}
	i32, err := f.Parameter("i32", shapes.Make(dtypes.Int32, 2), nil)
	if err != nil {
		t.Fatal(err)
	}
	i32Scalar, err := f.Parameter("i32scalar", shapes.Make(dtypes.Int32), nil)
	if err != nil {
		t.Fatal(err)
	}
	before := len(f.(*function).raw.Ops)
	if _, err := f.Complex(f32, c64); err == nil {
		t.Fatal("accepted complex component dtype mismatch")
	}
	for _, op := range []func(any, any) error{
		func(a, b any) error { _, err := f.LessThan(a.(compute.Value), b.(compute.Value)); return err },
		func(a, b any) error { _, err := f.Rem(a.(compute.Value), b.(compute.Value)); return err },
		func(a, b any) error { _, err := f.Max(a.(compute.Value), b.(compute.Value)); return err },
		func(a, b any) error { _, err := f.ReduceMax(a.(compute.Value)); return err },
		func(a, b any) error { _, err := f.ReduceMin(a.(compute.Value)); return err },
		func(a, b any) error { _, err := f.Pow(a.(compute.Value), b.(compute.Value)); return err },
	} {
		if err := op(c64, c64); err == nil {
			t.Fatal("accepted complex operation with no StableHLO lowering")
		}
	}
	if _, err := f.Real(f32); err == nil {
		t.Fatal("accepted real float input")
	}
	if _, err := f.Bitcast(c64, dtypes.Uint64); err == nil {
		t.Fatal("accepted cross-type complex bitcast")
	}
	if _, err := f.ConvertDType(c64, dtypes.Float32); err == nil {
		t.Fatal("accepted complex-to-float conversion")
	}
	if _, err := f.Complex(i32, i32Scalar); err == nil {
		t.Fatal("accepted integer complex components")
	}
	if len(f.(*function).raw.Ops) != before {
		t.Fatal("invalid complex operation emitted IR")
	}
}

func TestComplexConstantLiteral(t *testing.T) {
	for _, tc := range []struct {
		name string
		flat any
		want string
	}{
		{"c64", []complex64{complex(float32(math.Copysign(0, -1)), math.Float32frombits(0x7fa00001))}, "%constant0 = stablehlo.constant dense<[(0x80000000, 0x7FA00001)]> : tensor<1xcomplex<f32>>"},
		{"c128", []complex128{complex(math.Copysign(0, -1), math.Float64frombits(0x7ff4000000000001))}, "%constant0 = stablehlo.constant dense<[(0x8000000000000000, 0x7FF4000000000001)]> : tensor<1xcomplex<f64>>"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := (&Backend{}).Builder("complex_constant_literal").Main()
			if _, err := f.Constant(tc.flat, 1); err != nil {
				t.Fatal(err)
			}
			got := f.(*function).raw.Ops
			if len(got) != 1 || got[0] != tc.want {
				t.Fatalf("constant literal %q, want %q", got, tc.want)
			}
		})
	}
}

func TestNativeComplex(t *testing.T) {
	path := os.Getenv("TMC_COMPUTE_PLUGIN")
	if path == "" {
		t.Skip("set TMC_COMPUTE_PLUGIN")
	}
	b, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()

	t.Run("Transport", func(t *testing.T) {
		for _, tc := range []struct {
			name  string
			flat  any
			shape shapes.Shape
		}{
			{"c64 scalar", []complex64{complex(float32(math.Copysign(0, -1)), 1)}, shapes.Make(dtypes.Complex64)},
			{"c64 vector", []complex64{1 + 2i, -3 - 4i}, shapes.Make(dtypes.Complex64, 2)},
			{"c64 empty", []complex64{}, shapes.Make(dtypes.Complex64, 0)},
			{"c128 scalar", []complex128{complex(math.Copysign(0, -1), 1)}, shapes.Make(dtypes.Complex128)},
			{"c128 vector", []complex128{1 + 2i, -3 - 4i}, shapes.Make(dtypes.Complex128, 2)},
			{"c128 empty", []complex128{}, shapes.Make(dtypes.Complex128, 0)},
		} {
			t.Run(tc.name, func(t *testing.T) {
				in, err := b.BufferFromFlatData(0, tc.flat, tc.shape)
				if err != nil {
					t.Fatal(err)
				}
				defer in.Finalize()
				got := newFlat(tc.shape.DType, tc.shape.Size())
				if err := in.ToFlatData(got); err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(complexBits(got), complexBits(tc.flat)) {
					t.Fatalf("got %#v, want %#v", got, tc.flat)
				}
			})
		}
	})

	t.Run("Constants", func(t *testing.T) {
		builder := b.Builder("complex_constants")
		f := builder.Main()
		c64Input := []complex64{1 + 2i, complex(float32(math.Copysign(0, -1)), math.Float32frombits(0x7fa00001))}
		c64, err := f.Constant(c64Input, 2)
		if err != nil {
			t.Fatal(err)
		}
		c128Input := []complex128{5 - 6i, complex(math.Copysign(0, -1), math.Float64frombits(0x7ff4000000000001))}
		c128, err := f.Constant(c128Input, 2)
		if err != nil {
			t.Fatal(err)
		}
		if err := f.Return([]compute.Value{c64, c128}, nil); err != nil {
			t.Fatal(err)
		}
		exe, err := builder.Compile()
		if err != nil {
			t.Fatal(err)
		}
		defer exe.Finalize()
		out, err := exe.Execute(nil, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		defer out[0].Finalize()
		defer out[1].Finalize()
		got64 := make([]complex64, 2)
		got128 := make([]complex128, 2)
		if err := out[0].ToFlatData(got64); err != nil {
			t.Fatal(err)
		}
		if err := out[1].ToFlatData(got128); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(complexBits(got64), complexBits(c64Input)) || !reflect.DeepEqual(complexBits(got128), complexBits(c128Input)) {
			t.Fatalf("got %v %v", got64, got128)
		}
	})

	t.Run("Complex64Primitives", func(t *testing.T) {
		real := []float32{1, -2, 3}
		imag := []float32{2, 4, -1}
		got, err := testutil.Exec(b, []any{real, imag}, func(f compute.Function, v []compute.Value) ([]compute.Value, error) {
			z, err := f.Complex(v[0], v[1])
			if err != nil {
				return nil, err
			}
			conj, err := f.Conj(z)
			if err != nil {
				return nil, err
			}
			r, err := f.Real(z)
			if err != nil {
				return nil, err
			}
			i, err := f.Imag(z)
			if err != nil {
				return nil, err
			}
			add, err := f.Add(z, z)
			if err != nil {
				return nil, err
			}
			mul, err := f.Mul(z, conj)
			if err != nil {
				return nil, err
			}
			neg, err := f.Neg(z)
			if err != nil {
				return nil, err
			}
			sum, err := f.ReduceSum(z)
			if err != nil {
				return nil, err
			}
			prod, err := f.ReduceProduct(z)
			if err != nil {
				return nil, err
			}
			dot, err := f.DotGeneral(z, []int{0}, nil, z, []int{0}, nil, compute.DotGeneralConfig{})
			if err != nil {
				return nil, err
			}
			return []compute.Value{z, conj, r, i, add, mul, neg, sum, prod, dot}, nil
		})
		if err != nil {
			t.Fatal(err)
		}
		z := []complex64{1 + 2i, -2 + 4i, 3 - 1i}
		conj := []complex64{1 - 2i, -2 - 4i, 3 + 1i}
		want := []any{z, conj, real, imag, []complex64{2 + 4i, -4 + 8i, 6 - 2i}, []complex64{5, 20, 10}, []complex64{-1 - 2i, 2 - 4i, -3 + 1i}, complex64(2 + 5i), complex64(-30 + 10i), complex64(-7 - 18i)}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("got %#v, want %#v", got, want)
		}
	})

	t.Run("Complex128Primitives", func(t *testing.T) {
		real := []float64{1, -2}
		imag := []float64{2, 4}
		got, err := testutil.Exec(b, []any{real, imag}, func(f compute.Function, v []compute.Value) ([]compute.Value, error) {
			z, err := f.Complex(v[0], v[1])
			if err != nil {
				return nil, err
			}
			conj, err := f.Conj(z)
			if err != nil {
				return nil, err
			}
			r, err := f.Real(z)
			if err != nil {
				return nil, err
			}
			i, err := f.Imag(z)
			if err != nil {
				return nil, err
			}
			add, err := f.Add(z, z)
			if err != nil {
				return nil, err
			}
			sum, err := f.ReduceSum(z)
			if err != nil {
				return nil, err
			}
			prod, err := f.ReduceProduct(z)
			if err != nil {
				return nil, err
			}
			dot, err := f.DotGeneral(z, []int{0}, nil, z, []int{0}, nil, compute.DotGeneralConfig{})
			if err != nil {
				return nil, err
			}
			return []compute.Value{z, conj, r, i, add, sum, prod, dot}, nil
		})
		if err != nil {
			t.Fatal(err)
		}
		want := []any{[]complex128{1 + 2i, -2 + 4i}, []complex128{1 - 2i, -2 - 4i}, real, imag, []complex128{2 + 4i, -4 + 8i}, complex128(-1 + 6i), complex128(-10), complex128(-15 - 12i)}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("got %#v, want %#v", got, want)
		}
	})

	t.Run("Convert", func(t *testing.T) {
		input := []complex64{1 + 2i, -3 - 4i}
		got, err := testutil.Exec(b, []any{input}, func(f compute.Function, v []compute.Value) ([]compute.Value, error) {
			wide, err := f.ConvertDType(v[0], dtypes.Complex128)
			if err != nil {
				return nil, err
			}
			narrow, err := f.ConvertDType(wide, dtypes.Complex64)
			if err != nil {
				return nil, err
			}
			return []compute.Value{wide, narrow}, nil
		})
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got[0], []complex128{1 + 2i, -3 - 4i}) || !reflect.DeepEqual(got[1], input) {
			t.Fatalf("got %v", got)
		}
	})

	t.Run("SignedZeroAndNaN", func(t *testing.T) {
		real := []float32{math.Float32frombits(0x80000000), math.Float32frombits(0x7fc01234), float32(math.Inf(1))}
		imag := []float32{0, math.Float32frombits(0x80000000), float32(math.Inf(-1))}
		got, err := testutil.Exec(b, []any{real, imag}, func(f compute.Function, v []compute.Value) ([]compute.Value, error) {
			z, err := f.Complex(v[0], v[1])
			if err != nil {
				return nil, err
			}
			c, err := f.Conj(z)
			if err != nil {
				return nil, err
			}
			return []compute.Value{z, c}, nil
		})
		if err != nil {
			t.Fatal(err)
		}
		want := []complex64{complex(real[0], imag[0]), complex(real[1], imag[1]), complex(real[2], imag[2])}
		wantConj := []complex64{complex(real[0], -imag[0]), complex(real[1], -imag[1]), complex(real[2], -imag[2])}
		if !reflect.DeepEqual(complexBits(got[0]), complexBits(want)) || !reflect.DeepEqual(complexBits(got[1]), complexBits(wantConj)) {
			t.Fatalf("got bits %x %x, want %x %x", complexBits(got[0]), complexBits(got[1]), complexBits(want), complexBits(wantConj))
		}
	})

	t.Run("GraphConsumer", func(t *testing.T) {
		exec, err := graph.NewExec(b, func(real, imag *graph.Node) (*graph.Node, *graph.Node) {
			z := graph.Complex(real, imag)
			return graph.Real(graph.Conj(z)), graph.Imag(graph.Conj(z))
		})
		if err != nil {
			t.Fatal(err)
		}
		defer exec.Finalize()
		out, err := exec.Call([]float32{1, -2}, []float32{3, 4})
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(out[0].Value(), []float32{1, -2}) || !reflect.DeepEqual(out[1].Value(), []float32{-3, -4}) {
			t.Fatalf("got %v %v", out[0].Value(), out[1].Value())
		}
	})
}
