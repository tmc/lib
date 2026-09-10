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

func TestFFTValidation(t *testing.T) {
	f := (&Backend{}).Builder("fft_validation").Main()
	c64, err := f.Parameter("c64", shapes.Make(dtypes.Complex64, 4), nil)
	if err != nil {
		t.Fatal(err)
	}
	c64Three, err := f.Parameter("c64three", shapes.Make(dtypes.Complex64, 3), nil)
	if err != nil {
		t.Fatal(err)
	}
	c64Multi, err := f.Parameter("c64multi", shapes.Make(dtypes.Complex64, 2, 3), nil)
	if err != nil {
		t.Fatal(err)
	}
	f32, err := f.Parameter("f32", shapes.Make(dtypes.Float32, 4), nil)
	if err != nil {
		t.Fatal(err)
	}
	i32, err := f.Parameter("i32", shapes.Make(dtypes.Int32, 4), nil)
	if err != nil {
		t.Fatal(err)
	}
	scalar, err := f.Parameter("scalar", shapes.Make(dtypes.Complex64), nil)
	if err != nil {
		t.Fatal(err)
	}
	before := len(f.(*function).raw.Ops)
	for _, tc := range []struct {
		value  compute.Value
		typ    compute.FFTType
		length []int
	}{
		{i32, compute.FFTForward, []int{4}},
		{scalar, compute.FFTForward, nil},
		{f32, compute.FFTForwardReal, []int{3}},
		{c64Three, compute.FFTInverseReal, []int{6}},
		{c64Multi, compute.FFTInverseReal, []int{4, 4}},
		{c64, compute.FFTForward, []int{-1}},
		{c64, compute.FFTType(99), []int{4}},
	} {
		if _, err := f.FFT(tc.value, tc.typ, tc.length); err == nil {
			t.Fatal("accepted invalid FFT")
		}
	}
	if len(f.(*function).raw.Ops) != before {
		t.Fatal("invalid FFT emitted IR")
	}
}

func TestNativeFFT(t *testing.T) {
	path := os.Getenv("TMC_COMPUTE_PLUGIN")
	if path == "" {
		t.Skip("set TMC_COMPUTE_PLUGIN")
	}
	b, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()

	t.Run("Complex64ImpulseAndRoundTrip", func(t *testing.T) {
		input := []complex64{1, 0, 0, 0}
		got, err := testutil.Exec(b, []any{input}, func(f compute.Function, v []compute.Value) ([]compute.Value, error) {
			fft, err := f.FFT(v[0], compute.FFTForward, nil)
			if err != nil {
				return nil, err
			}
			inv, err := f.FFT(fft, compute.FFTInverse, nil)
			if err != nil {
				return nil, err
			}
			return []compute.Value{fft, inv}, nil
		})
		if err != nil {
			t.Fatal(err)
		}
		assertComplex64Close(t, got[0].([]complex64), []complex64{1, 1, 1, 1})
		assertComplex64Close(t, got[1].([]complex64), input)
	})

	t.Run("Complex128DFT", func(t *testing.T) {
		input := []complex128{1 + 2i, -2 + 1i, 3 - 1i, 2 + 0i}
		got, err := testutil.Exec1(b, []any{input}, func(f compute.Function, v []compute.Value) (compute.Value, error) {
			return f.FFT(v[0], compute.FFTForward, []int{4})
		})
		if err != nil {
			t.Fatal(err)
		}
		assertComplex128Close(t, got.([]complex128), dft(input, false))
	})

	t.Run("RealOddEvenRoundTrip", func(t *testing.T) {
		for _, input := range [][]float64{{1, 2, 3, 4}, {1, -2, 3, -4, 5}} {
			got, err := testutil.Exec(b, []any{input}, func(f compute.Function, v []compute.Value) ([]compute.Value, error) {
				fft, err := f.FFT(v[0], compute.FFTForwardReal, nil)
				if err != nil {
					return nil, err
				}
				inv, err := f.FFT(fft, compute.FFTInverseReal, []int{len(input)})
				if err != nil {
					return nil, err
				}
				return []compute.Value{fft, inv}, nil
			})
			if err != nil {
				t.Fatal(err)
			}
			want := dft(realComplex(input), false)[:len(input)/2+1]
			assertComplex128Close(t, got[0].([]complex128), want)
			if !reflect.DeepEqual(got[1], input) {
				t.Fatalf("inverse got %v, want %v", got[1], input)
			}
		}
	})

	t.Run("BatchedTwoDimensionalReal", func(t *testing.T) {
		input := make([]float64, 2*3*4)
		input[0] = 1
		input[3*4] = 2
		got := executeFFT(t, b, input, shapes.Make(dtypes.Float64, 2, 3, 4), compute.FFTForwardReal, []int{3, 4})
		want := make([]complex128, 2*3*3)
		for i := 0; i < 3*3; i++ {
			want[i] = 1
			want[3*3+i] = 2
		}
		assertComplex128Close(t, got.([]complex128), want)
	})

	t.Run("RemainingModes", func(t *testing.T) {
		rfft := executeFFT(t, b, []float32{1, 0, 0, 0}, shapes.Make(dtypes.Float32, 4), compute.FFTForwardReal, []int{4})
		assertComplex64Close(t, rfft.([]complex64), []complex64{1, 1, 1})
		irfft := executeFFT(t, b, []complex64{1, 1, 1}, shapes.Make(dtypes.Complex64, 3), compute.FFTInverseReal, []int{4})
		if !reflect.DeepEqual(irfft, []float32{1, 0, 0, 0}) {
			t.Fatalf("irfft got %v", irfft)
		}
		input := []complex128{1 + 2i, -2 + 1i, 3 - 1i, 2}
		freq := dft(input, false)
		got, err := testutil.Exec1(b, []any{freq}, func(f compute.Function, v []compute.Value) (compute.Value, error) {
			return f.FFT(v[0], compute.FFTInverse, []int{4})
		})
		if err != nil {
			t.Fatal(err)
		}
		assertComplex128Close(t, got.([]complex128), input)
	})

	t.Run("MultiDimensionalInverseReal", func(t *testing.T) {
		input := []complex128{1, 1, 1, 1, 1, 1}
		got := executeFFT(t, b, input, shapes.Make(dtypes.Complex128, 2, 3), compute.FFTInverseReal, []int{2, 4})
		want := []float64{1, 0, 0, 0, 0, 0, 0, 0}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("got %v, want %v", got, want)
		}
	})

	t.Run("ThreeDimensionalComplex", func(t *testing.T) {
		input := []complex64{1, 0, 0, 0, 0, 0, 0, 0}
		fft := executeFFT(t, b, input, shapes.Make(dtypes.Complex64, 2, 2, 2), compute.FFTForward, []int{2, 2, 2})
		assertComplex64Close(t, fft.([]complex64), []complex64{1, 1, 1, 1, 1, 1, 1, 1})
		inv := executeFFT(t, b, fft, shapes.Make(dtypes.Complex64, 2, 2, 2), compute.FFTInverse, []int{2, 2, 2})
		assertComplex64Close(t, inv.([]complex64), input)
	})

	t.Run("ZeroExtent", func(t *testing.T) {
		got := executeFFT(t, b, []float32{}, shapes.Make(dtypes.Float32, 0), compute.FFTForwardReal, nil)
		if len(got.([]complex64)) != 0 {
			t.Fatalf("got %v", got)
		}
	})

	t.Run("GraphWrapper", func(t *testing.T) {
		exec, err := graph.NewExec(b, func(x *graph.Node) *graph.Node { return graph.InverseFFT(graph.FFT(x)) })
		if err != nil {
			t.Fatal(err)
		}
		defer exec.Finalize()
		out, err := exec.Call([]complex64{1 + 2i, -2 + 1i, 3 - 1i, 2})
		if err != nil {
			t.Fatal(err)
		}
		assertComplex64Close(t, out[0].Value().([]complex64), []complex64{1 + 2i, -2 + 1i, 3 - 1i, 2})
	})

	t.Run("GraphGradientConsumer", func(t *testing.T) {
		exec, err := graph.NewExec(b, func(x *graph.Node) *graph.Node {
			y := graph.FFT(x)
			return graph.Gradient(graph.ReduceAllSum(graph.Real(y)), x)[0]
		})
		if err != nil {
			t.Fatal(err)
		}
		defer exec.Finalize()
		out, err := exec.Call([]complex64{1, 0, 0, 0})
		if err != nil {
			t.Fatal(err)
		}
		assertComplex64Close(t, out[0].Value().([]complex64), []complex64{4, 0, 0, 0})
	})
}

func executeFFT(t *testing.T, b *Backend, flat any, shape shapes.Shape, fftType compute.FFTType, fftLength []int) any {
	t.Helper()
	in, err := b.BufferFromFlatData(0, flat, shape)
	if err != nil {
		t.Fatal(err)
	}
	defer in.Finalize()
	builder := b.Builder("fft")
	f := builder.Main()
	param, err := f.Parameter("input", shape, nil)
	if err != nil {
		t.Fatal(err)
	}
	out, err := f.FFT(param, fftType, fftLength)
	if err != nil {
		t.Fatal(err)
	}
	outShape, err := f.Shape(out)
	if err != nil {
		t.Fatal(err)
	}
	if err := f.Return([]compute.Value{out}, nil); err != nil {
		t.Fatal(err)
	}
	exe, err := builder.Compile()
	if err != nil {
		t.Fatal(err)
	}
	defer exe.Finalize()
	outputs, err := exe.Execute([]compute.Buffer{in}, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer outputs[0].Finalize()
	got := newFlat(outShape.DType, outShape.Size())
	if err := outputs[0].ToFlatData(got); err != nil {
		t.Fatal(err)
	}
	return got
}

func dft(input []complex128, inverse bool) []complex128 {
	out := make([]complex128, len(input))
	sign := -1.0
	if inverse {
		sign = 1
	}
	for k := range out {
		for n, x := range input {
			angle := sign * 2 * math.Pi * float64(k*n) / float64(len(input))
			out[k] += x * complex(math.Cos(angle), math.Sin(angle))
		}
		if inverse {
			out[k] /= complex(float64(len(input)), 0)
		}
	}
	return out
}

func realComplex(input []float64) []complex128 {
	out := make([]complex128, len(input))
	for i, x := range input {
		out[i] = complex(x, 0)
	}
	return out
}

func assertComplex64Close(t *testing.T, got, want []complex64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("got length %d, want %d", len(got), len(want))
	}
	for i := range got {
		if math.Abs(float64(real(got[i]-want[i]))) > 1e-5 || math.Abs(float64(imag(got[i]-want[i]))) > 1e-5 {
			t.Fatalf("got %v, want %v", got, want)
		}
	}
}

func assertComplex128Close(t *testing.T, got, want []complex128) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("got length %d, want %d", len(got), len(want))
	}
	for i := range got {
		if cmplxAbs(got[i]-want[i]) > 1e-10 {
			t.Fatalf("got %v, want %v", got, want)
		}
	}
}

func cmplxAbs(x complex128) float64 { return math.Hypot(real(x), imag(x)) }
