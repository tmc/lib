//go:build darwin || linux

package curand

import (
	"errors"
	"math"
	"os"
	"slices"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/internal/abistub"
)

// abiStub stands in for libcurand. Its functions take float and double
// arguments after integer and pointer ones, and one takes a double first,
// so the floating-point words travel in their own registers.
const abiStub = abistub.Prelude + `
#ifdef STUB_REAL_HEADER
#include <curand.h>
#else
typedef int curandStatus_t;
typedef struct curandGenerator_st *curandGenerator_t;
typedef struct curandDiscreteDistribution_st *curandDiscreteDistribution_t;
#endif

curandStatus_t curandGenerateNormal(curandGenerator_t g, float *out, size_t n, float mean, float stddev) {
	BEGIN; R(g); R(out); R(n); F32(mean); F32(stddev); return RET;
}
curandStatus_t curandGenerateLogNormal(curandGenerator_t g, float *out, size_t n, float mean, float stddev) {
	BEGIN; R(g); R(out); R(n); F32(mean); F32(stddev); return RET;
}
curandStatus_t curandGenerateNormalDouble(curandGenerator_t g, double *out, size_t n, double mean, double stddev) {
	BEGIN; R(g); R(out); R(n); F64(mean); F64(stddev); return RET;
}
curandStatus_t curandGenerateBinomial(curandGenerator_t g, unsigned int *out, size_t num, unsigned int n, double p) {
	BEGIN; R(g); R(out); R(num); R(n); F64(p); return RET;
}
curandStatus_t curandGeneratePoisson(curandGenerator_t g, unsigned int *out, size_t n, double lambda) {
	BEGIN; R(g); R(out); R(n); F64(lambda); return RET;
}
curandStatus_t curandCreatePoissonDistribution(double lambda, curandDiscreteDistribution_t *d) {
	BEGIN; F64(lambda);
	*d = (curandDiscreteDistribution_t)0xffff00000000d15cull;
	return RET;
}
`

const (
	abiGen = Generator(0xfedcba9876543210)
	abiPtr = driver.DevicePtr(0xffffffffffff0000)
	abiN   = uint(0x8000000000000040)
)

// TestABI checks, without a GPU, that the generated curand wrappers deliver
// mixed integer and floating-point arguments to C bit for bit and do not
// allocate. It runs in a child process, because Load binds the stub for the
// life of the process.
func TestABI(t *testing.T) {
	if path := os.Getenv("CURAND_ABI_STUB"); path != "" {
		testABI(t, path)
		return
	}
	variants := map[string][]string{"mirror": nil}
	if inc := abistub.CUDAInclude("curand.h"); inc != "" {
		variants["curand.h"] = []string{"-DSTUB_REAL_HEADER", "-I", inc}
	}
	for name, flags := range variants {
		t.Run(name, func(t *testing.T) {
			abistub.Child(t, "TestABI", "CURAND_ABI_STUB", abistub.Build(t, abiStub, flags...))
		})
	}
}

func testABI(t *testing.T, path string) {
	if err := Load(path); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { Close() })
	stub, err := abistub.NewRecorder(func(name string) (uintptr, error) { return lookupLibrarySymbol(lib, name) })
	if err != nil {
		t.Fatal(err)
	}
	f32 := func(f float32) uint64 { return uint64(math.Float32bits(f)) }
	f64 := math.Float64bits
	tests := []struct {
		name string
		call func() error
		want []uint64
	}{
		{
			name: "GenerateNormal",
			call: func() error { return abiGen.GenerateNormal(abiPtr, abiN, -1.5, 3e38) },
			want: []uint64{uint64(abiGen), uint64(abiPtr), uint64(abiN), f32(-1.5), f32(3e38)},
		},
		{
			name: "GenerateLogNormal",
			call: func() error { return abiGen.GenerateLogNormal(abiPtr, abiN, float32(math.Inf(-1)), 1e-40) },
			want: []uint64{uint64(abiGen), uint64(abiPtr), uint64(abiN), f32(float32(math.Inf(-1))), f32(1e-40)},
		},
		{
			name: "GenerateNormalDouble",
			call: func() error { return abiGen.GenerateNormalDouble(abiPtr, abiN, -2.5e300, math.SmallestNonzeroFloat64) },
			want: []uint64{uint64(abiGen), uint64(abiPtr), uint64(abiN), f64(-2.5e300), f64(math.SmallestNonzeroFloat64)},
		},
		{
			name: "GenerateBinomial",
			call: func() error { return abiGen.GenerateBinomial(abiPtr, abiN, 0x80000001, 0.125) },
			want: []uint64{uint64(abiGen), uint64(abiPtr), uint64(abiN), 0x80000001, f64(0.125)},
		},
		{
			name: "GeneratePoisson",
			call: func() error { return abiGen.GeneratePoisson(abiPtr, abiN, 1e-3) },
			want: []uint64{uint64(abiGen), uint64(abiPtr), uint64(abiN), f64(1e-3)},
		},
		{
			name: "CreatePoissonDistribution",
			call: func() error {
				d, err := CreatePoissonDistribution(math.Copysign(0, -1))
				if err == nil && d != DiscreteDistribution(0xffff00000000d15c) {
					t.Errorf("CreatePoissonDistribution = %#x, want 0xffff00000000d15c", d)
				}
				return err
			},
			want: []uint64{f64(math.Copysign(0, -1))},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			stub.Reset(0)
			if err := tt.call(); err != nil {
				t.Fatal(err)
			}
			if got := stub.Recorded(); !slices.Equal(got, tt.want) {
				t.Errorf("C received\n\t%#x\nwant\n\t%#x", got, tt.want)
			}
			if n := testing.AllocsPerRun(100, func() { tt.call() }); n != 0 {
				t.Errorf("%v allocs/op, want 0", n)
			}

			stub.Reset(int32(ErrLaunchFailure))
			defer stub.Reset(0)
			var ce *CallError
			if err := tt.call(); !errors.As(err, &ce) || ce.Result != ErrLaunchFailure {
				t.Errorf("with the stub failing: %v, want a *CallError with ErrLaunchFailure", err)
			}
		})
	}
}
