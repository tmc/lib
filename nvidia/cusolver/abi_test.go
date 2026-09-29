//go:build darwin || linux

package cusolver

import (
	"errors"
	"math"
	"os"
	"slices"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/internal/abistub"
)

// abiStub stands in for libcusolver. The gesvd functions take more
// arguments than there are integer argument registers, with signed chars,
// ints and 64-bit values among them, so narrow words land on the stack,
// where Apple's arm64 ABI packs them at their natural size.
const abiStub = abistub.Prelude + `
#ifdef STUB_REAL_HEADER
#include <cusolverDn.h>
#else
typedef int cusolverStatus_t;
typedef int cudaDataType;
typedef struct cusolverDnContext *cusolverDnHandle_t;
typedef struct cusolverDnParams *cusolverDnParams_t;
typedef struct gesvdjInfo *gesvdjInfo_t;
typedef int cusolverEigType_t;
typedef int cusolverEigMode_t;
typedef int cusolverEigRange_t;
typedef int cublasFillMode_t;
#endif

cusolverStatus_t cusolverDnSgesvd(cusolverDnHandle_t h, signed char jobu, signed char jobvt,
	int m, int n, float *A, int lda, float *S, float *U, int ldu, float *VT, int ldvt,
	float *work, int lwork, float *rwork, int *info) {
	BEGIN; R(h); S(jobu); S(jobvt); S(m); S(n); R(A); S(lda); R(S); R(U); S(ldu); R(VT); S(ldvt);
	R(work); S(lwork); R(rwork); R(info);
	return RET;
}

cusolverStatus_t cusolverDnXgesvd(cusolverDnHandle_t h, cusolverDnParams_t p, signed char jobu, signed char jobvt,
	int64_t m, int64_t n, cudaDataType tA, void *A, int64_t lda, cudaDataType tS, void *S,
	cudaDataType tU, void *U, int64_t ldu, cudaDataType tVT, void *VT, int64_t ldvt,
	cudaDataType compute, void *bufDev, size_t nDev, void *bufHost, size_t nHost, int *info) {
	BEGIN; R(h); R(p); S(jobu); S(jobvt); S(m); S(n); R((uint32_t)tA); R(A); S(lda); R((uint32_t)tS); R(S);
	R((uint32_t)tU); R(U); S(ldu); R((uint32_t)tVT); R(VT); S(ldvt); R((uint32_t)compute);
	R(bufDev); R(nDev); R(bufHost); R(nHost); R(info);
	return RET;
}

// cusolverDnSsygvdx passes ldb, il and iu as adjacent ints on the stack, and
// vl and vu in floating-point registers.
cusolverStatus_t cusolverDnSsygvdx(cusolverDnHandle_t h, cusolverEigType_t itype, cusolverEigMode_t jobz,
	cusolverEigRange_t range, cublasFillMode_t uplo, int n, float *A, int lda, float *B, int ldb,
	float vl, float vu, int il, int iu, int *meig, float *W, float *work, int lwork, int *info) {
	BEGIN; R(h); R((uint32_t)itype); R((uint32_t)jobz); R((uint32_t)range); R((uint32_t)uplo); S(n); R(A); S(lda);
	R(B); S(ldb); F32(vl); F32(vu); S(il); S(iu); R(meig); R(W); R(work); S(lwork); R(info);
	*meig = -0x7ffffff0;
	return RET;
}

cusolverStatus_t cusolverDnXgesvdjSetTolerance(gesvdjInfo_t info, double tol) {
	BEGIN; R(info); F64(tol); return RET;
}
`

// TestABI checks, without a GPU, that the generated cusolver wrappers
// deliver arguments past the register file to C bit for bit and do not
// allocate. It runs in a child process, because Load binds the stub for the
// life of the process.
func TestABI(t *testing.T) {
	if path := os.Getenv("CUSOLVER_ABI_STUB"); path != "" {
		testABI(t, path)
		return
	}
	variants := map[string][]string{"mirror": nil}
	if inc := abistub.CUDAInclude("cusolverDn.h"); inc != "" {
		variants["cusolverDn.h"] = []string{"-DSTUB_REAL_HEADER", "-I", inc}
	}
	for name, flags := range variants {
		t.Run(name, func(t *testing.T) {
			abistub.Child(t, "TestABI", "CUSOLVER_ABI_STUB", abistub.Build(t, abiStub, flags...))
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
	const h = Handle(0xfedcba9876543210)
	p := func(i uint64) driver.DevicePtr { return driver.DevicePtr(0xffff000000000000 | i<<8) }
	u := func(v int64) uint64 { return uint64(v) }
	tests := []struct {
		name string
		call func() error
		want []uint64
	}{
		{
			name: "Sgesvd",
			call: func() error {
				return h.Sgesvd('A', -2, -3, 0x7ffffff4, p(1), -5, p(2), p(3), -6, p(4), 0x7ffffff7, p(5), -8, p(6), p(7))
			},
			want: []uint64{uint64(h), 'A', u(-2), u(-3), 0x7ffffff4, uint64(p(1)), u(-5), uint64(p(2)), uint64(p(3)), u(-6),
				uint64(p(4)), 0x7ffffff7, uint64(p(5)), u(-8), uint64(p(6)), uint64(p(7))},
		},
		{
			name: "Xgesvd",
			call: func() error {
				return h.Xgesvd(Params(0x8000000000000001), 'S', -1, -0x100000000, 0x7fffffffffffffff,
					DataType(-0x7ffffffe), p(1), -3, DataType(4), p(2), DataType(0x7fffffff), p(3), -5,
					DataType(-2), p(4), -6, DataType(7), p(5), 0x8000000000000009, 0xc000000000000000, 0xffffffffffffffff, p(6))
			},
			want: []uint64{uint64(h), 0x8000000000000001, 'S', u(-1), u(-0x100000000), 0x7fffffffffffffff,
				0x80000002, uint64(p(1)), u(-3), 4, uint64(p(2)), 0x7fffffff, uint64(p(3)), u(-5),
				0xfffffffe, uint64(p(4)), u(-6), 7, uint64(p(5)), 0x8000000000000009, 0xc000000000000000, 0xffffffffffffffff, uint64(p(6))},
		},
		{
			name: "Ssygvdx",
			call: func() error {
				abiMeig = 0
				err := h.Ssygvdx(EigType(-1), EigMode(0x7ffffff2), EigRange(-3), FillMode(0x40000004), -5, p(1), 0x7ffffff6,
					p(2), -7, -0.5, float32(math.Inf(1)), 0x7ffffff8, -9, &abiMeig, p(3), p(4), -10, p(5))
				if err == nil && abiMeig != -0x7ffffff0 {
					t.Errorf("Ssygvdx wrote meig %#x, want -0x7ffffff0", abiMeig)
				}
				return err
			},
			want: []uint64{uint64(h), 0xffffffff, 0x7ffffff2, 0xfffffffd, 0x40000004, u(-5), uint64(p(1)), 0x7ffffff6,
				uint64(p(2)), u(-7), uint64(math.Float32bits(-0.5)), uint64(math.Float32bits(float32(math.Inf(1)))),
				0x7ffffff8, u(-9), uint64(uintptr(unsafe.Pointer(&abiMeig))), uint64(p(3)), uint64(p(4)), u(-10), uint64(p(5))},
		},
		{
			name: "XgesvdjSetTolerance",
			call: func() error { return GesvdjInfo(0x8000000000000abc).XgesvdjSetTolerance(-math.MaxFloat64) },
			want: []uint64{0x8000000000000abc, math.Float64bits(-math.MaxFloat64)},
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

			stub.Reset(int32(ErrExecutionFailed))
			defer stub.Reset(0)
			var ce *CallError
			if err := tt.call(); !errors.As(err, &ce) || ce.Result != ErrExecutionFailed {
				t.Errorf("with the stub failing: %v, want a *CallError with ErrExecutionFailed", err)
			}
		})
	}
}

// abiMeig receives Ssygvdx's eigenvalue count; it is package-level so the
// call itself does not allocate.
var abiMeig int32
