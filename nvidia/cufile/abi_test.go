//go:build linux

package cufile

import (
	"errors"
	"os"
	"slices"
	"testing"

	"github.com/ebitengine/purego"
	"github.com/tmc/lib/nvidia/internal/abistub"
)

// abiStub stands in for libcufile. Every cuFile status is CUfileError_t, an
// 8-byte struct of two enums that the binding reads back from the integer
// result register. The stub returns stubRet as err and, when it is nonzero,
// a cu_err with its high bit set, so a lost or swapped half shows.
const abiStub = abistub.Prelude + `
#include <stdbool.h>
#ifdef STUB_REAL_HEADER
#include <cufile.h>
#else
typedef int CUfileOpError;
typedef int CUresult;
typedef struct CUfileError { CUfileOpError err; CUresult cu_err; } CUfileError_t;
#endif

#define ERR ((CUfileError_t){(CUfileOpError)RET, RET ? (CUresult)0x80000abc : (CUresult)0})

CUfileError_t cuFileDriverOpen(void) {
	BEGIN; return ERR;
}
CUfileError_t cuFileBufRegister(const void *base, size_t length, int flags) {
	BEGIN; R(base); R(length); S(flags); return ERR;
}
CUfileError_t cuFileDriverSetPollMode(bool poll, size_t threshold) {
	BEGIN; R(poll); R(threshold); return ERR;
}
`

// TestABI checks, without a GPU, that the generated cufile wrappers deliver
// their arguments to C bit for bit, read both halves of the CUfileError_t
// result, and do not allocate on success. It runs in a child process,
// because Load binds the stub for the life of the process.
func TestABI(t *testing.T) {
	if path := os.Getenv("CUFILE_ABI_STUB"); path != "" {
		testABI(t, path)
		return
	}
	variants := map[string][]string{"mirror": nil}
	if inc := abistub.CUDAInclude("cufile.h"); inc != "" {
		variants["cufile.h"] = []string{"-DSTUB_REAL_HEADER", "-I", inc}
	}
	for name, flags := range variants {
		t.Run(name, func(t *testing.T) {
			abistub.Child(t, "TestABI", "CUFILE_ABI_STUB", abistub.Build(t, abiStub, flags...))
		})
	}
}

func testABI(t *testing.T, path string) {
	if err := Load(path); err != nil {
		t.Fatal(err)
	}
	stub, err := abistub.NewRecorder(func(name string) (uintptr, error) { return purego.Dlsym(lib, name) })
	if err != nil {
		t.Fatal(err)
	}
	const base = uintptr(0xfedcba9876543210)
	tests := []struct {
		name string
		call func() error
		want []uint64
	}{
		{name: "DriverOpen", call: DriverOpen},
		{
			name: "BufRegister",
			call: func() error { return BufRegister(base, 1<<40+3, -7) },
			want: []uint64{uint64(base), 1<<40 + 3, 0xfffffffffffffff9},
		},
		{
			name: "DriverSetPollMode",
			call: func() error { return DriverSetPollMode(true, 0x8000000000000001) },
			want: []uint64{1, 0x8000000000000001},
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

			stub.Reset(int32(InvalidValue))
			defer stub.Reset(0)
			var ce *CallError
			want := Error{Err: InvalidValue, CuErr: CudaResult(-0x7ffff544)}
			if err := tt.call(); !errors.As(err, &ce) || ce.Result != want {
				t.Errorf("with the stub failing: %v, want a *CallError with %+v", err, want)
			}
		})
	}
}

// BenchmarkABI measures a call with a CUfileError_t result against the
// stub, loaded in this process.
func BenchmarkABI(b *testing.B) {
	if err := Load(abistub.Build(b, abiStub)); err != nil {
		b.Fatal(err)
	}
	for _, bb := range []struct {
		name string
		call func() error
	}{
		{"DriverOpen", DriverOpen},
		{"BufRegister", func() error { return BufRegister(0x1000, 4096, 0) }},
	} {
		b.Run(bb.name, func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				if err := bb.call(); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
