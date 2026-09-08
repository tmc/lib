package gomlx

import (
	"os"
	"reflect"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/testutil"
)

func TestNativeHalfBits(t *testing.T) {
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
	f16 := make([]float16.Float16, 1<<16)
	bf16 := make([]bfloat16.BFloat16, 1<<16)
	for i := range f16 {
		f16[i] = float16.FromBits(uint16(i))
		bf16[i] = bfloat16.FromBits(uint16(i))
	}
	for _, tc := range []struct {
		name  string
		dtype dtypes.DType
		data  any
	}{
		{"float16", dtypes.Float16, f16}, {"bfloat16", dtypes.BFloat16, bf16},
	} {
		t.Run(tc.name, func(t *testing.T) {
			input, err := b.BufferFromFlatData(0, tc.data, shapes.Make(tc.dtype, 1<<16))
			if err != nil {
				t.Fatal(err)
			}
			defer input.Finalize()
			got := newFlat(tc.dtype, 1<<16)
			if err := input.ToFlatData(got); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, tc.data) {
				t.Fatal("transport changed raw bits")
			}
			out, err := testutil.Exec1(b, nil, func(f compute.Function, _ []compute.Value) (compute.Value, error) { return f.Constant(tc.data, 1<<16) })
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(out, tc.data) {
				a, z := reflect.ValueOf(out), reflect.ValueOf(tc.data)
				for i := 0; i < a.Len(); i++ {
					if !reflect.DeepEqual(a.Index(i).Interface(), z.Index(i).Interface()) {
						t.Fatalf("constant bit pattern %04x got%v want%v", i, a.Index(i).Interface(), z.Index(i).Interface())
					}
				}
			}
		})
	}
}

func TestNativeHalfDot(t *testing.T) {
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
		data  any
		x     float32
	}{
		{"float16", dtypes.Float16, []float16.Float16{float16.FromBits(0x3c01)}, float16.FromBits(0x3c01).Float32()},
		{"bfloat16", dtypes.BFloat16, []bfloat16.BFloat16{bfloat16.FromBits(0x3f81)}, bfloat16.FromBits(0x3f81).Float32()},
	} {
		t.Run(tc.name, func(t *testing.T) {
			run := func(output dtypes.DType) any {
				got, err := testutil.Exec1(b, []any{tc.data, tc.data}, func(f compute.Function, v []compute.Value) (compute.Value, error) {
					return f.DotGeneral(v[0], []int{0}, nil, v[1], []int{0}, nil, compute.DotGeneralConfig{AccumulatorDType: dtypes.Float32, OutputDType: output})
				})
				if err != nil {
					t.Fatal(err)
				}
				return got
			}
			want := tc.x * tc.x
			got := run(dtypes.Float32)
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("f32 result %v (%T), want %v", got, got, want)
			}
			rounded := run(dtypes.InvalidDType)
			var value float32
			switch x := rounded.(type) {
			case float16.Float16:
				value = x.Float32()
			case bfloat16.BFloat16:
				value = x.Float32()
			default:
				t.Fatalf("default output %T", rounded)
			}
			if value == want {
				t.Fatal("rounding discriminator did not distinguish half output")
			}
			if dtypes.FromAny(rounded) != tc.dtype {
				t.Fatal("default output changed input dtype")
			}
			t.Logf("f32 result %.10g differs from half-rounded %.10g", want, value)
		})
	}
}
