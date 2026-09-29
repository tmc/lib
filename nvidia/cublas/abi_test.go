//go:build darwin || linux

package cublas

import (
	"errors"
	"os"
	"slices"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/internal/abistub"
)

// abiStub stands in for libcublas. Its functions take a driver.Stream, a
// type from another package, which the binding passes as a machine word.
const abiStub = abistub.Prelude + `
#ifdef STUB_REAL_HEADER
#include <cublas_api.h>
#else
typedef int cublasStatus_t;
typedef struct cublasContext *cublasHandle_t;
typedef struct CUstream_st *cudaStream_t;
#endif

cublasStatus_t cublasSetStream_v2(cublasHandle_t h, cudaStream_t s) {
	BEGIN; R(h); R(s); return RET;
}
cublasStatus_t cublasSetVectorAsync(int n, int elemSize, const void *host, int incx, void *dev, int incy, cudaStream_t s) {
	BEGIN; S(n); S(elemSize); R(host); S(incx); R(dev); S(incy); R(s); return RET;
}
cublasStatus_t cublasGetMatrixAsync_64(int64_t rows, int64_t cols, int64_t elemSize, const void *A, int64_t lda,
	void *B, int64_t ldb, cudaStream_t s) {
	BEGIN; S(rows); S(cols); S(elemSize); R(A); S(lda); R(B); S(ldb); R(s); return RET;
}
`

// TestABI checks, without a GPU, that the generated cublas stream functions
// deliver their arguments to C bit for bit and do not allocate. It runs in
// a child process, because Load binds the stub for the life of the process.
func TestABI(t *testing.T) {
	if path := os.Getenv("CUBLAS_ABI_STUB"); path != "" {
		testABI(t, path)
		return
	}
	variants := map[string][]string{"mirror": nil}
	if inc := abistub.CUDAInclude("cublas_api.h"); inc != "" {
		variants["cublas_api.h"] = []string{"-DSTUB_REAL_HEADER", "-I", inc}
	}
	for name, flags := range variants {
		t.Run(name, func(t *testing.T) {
			abistub.Child(t, "TestABI", "CUBLAS_ABI_STUB", abistub.Build(t, abiStub, flags...))
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
	const (
		h = Handle(0xfedcba9876543210)
		s = driver.Stream(0x8000000000000008)
		d = driver.DevicePtr(0xffff000000000100)
	)
	host := unsafe.Pointer(&abiHost)
	u := func(v int64) uint64 { return uint64(v) }
	tests := []struct {
		name string
		call func() error
		want []uint64
	}{
		{
			name: "SetStream",
			call: func() error { return h.SetStream(s) },
			want: []uint64{uint64(h), uint64(s)},
		},
		{
			name: "SetVectorAsync",
			call: func() error { return SetVectorAsync(-1, 0x7ffffff2, host, -3, d, 0x7ffffff4, s) },
			want: []uint64{u(-1), 0x7ffffff2, uint64(uintptr(host)), u(-3), uint64(d), 0x7ffffff4, uint64(s)},
		},
		{
			name: "GetMatrixAsync64",
			call: func() error {
				return GetMatrixAsync64(-0x100000000, 0x7fffffffffffffff, -2, d, -4, uintptr(host), 0x4000000000000005, s)
			},
			want: []uint64{u(-0x100000000), 0x7fffffffffffffff, u(-2), uint64(d), u(-4), uint64(uintptr(host)), 0x4000000000000005, uint64(s)},
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

			stub.Reset(int32(ErrInvalidValue))
			defer stub.Reset(0)
			var ce *CallError
			if err := tt.call(); !errors.As(err, &ce) || ce.Result != ErrInvalidValue {
				t.Errorf("with the stub failing: %v, want a *CallError with ErrInvalidValue", err)
			}
		})
	}
}

// abiHost is the host buffer the copies name; it is package-level so the
// calls themselves do not allocate.
var abiHost [16]byte
