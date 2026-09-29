//go:build darwin || linux

package cupti

import (
	"errors"
	"os"
	"slices"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/internal/abistub"
)

// abiStub stands in for libcupti. Its functions take a driver.Context and a
// driver.Stream, types from another package, which the binding passes as
// machine words. The stubs with an output write a high-bit value through it
// rather than recording its address, which is a pooled cell.
const abiStub = abistub.Prelude + `
#ifdef STUB_REAL_HEADER
#include <cupti.h>
#else
typedef int CUptiResult;
typedef int CUpti_ActivityKind;
typedef struct CUctx_st *CUcontext;
typedef struct CUstream_st *CUstream;
#endif

CUptiResult cuptiActivityEnableContext(CUcontext c, CUpti_ActivityKind kind) {
	BEGIN; R(c); R((uint32_t)kind); return RET;
}
CUptiResult cuptiGetContextId(CUcontext c, uint32_t *id) {
	BEGIN; R(c); *id = 0x80000001; return RET;
}
CUptiResult cuptiGetStreamIdEx(CUcontext c, CUstream s, uint8_t perThread, uint32_t *id) {
	BEGIN; R(c); R(s); R(perThread); *id = 0xfffffffe; return RET;
}
`

// TestABI checks, without a GPU, that the generated cupti wrappers deliver
// context and stream handles to C bit for bit and do not allocate. It runs
// in a child process, because Load binds the stub for the life of the
// process.
func TestABI(t *testing.T) {
	if path := os.Getenv("CUPTI_ABI_STUB"); path != "" {
		testABI(t, path)
		return
	}
	variants := map[string][]string{"mirror": nil}
	if inc := abistub.CUDAInclude("cupti.h"); inc != "" {
		variants["cupti.h"] = []string{"-DSTUB_REAL_HEADER", "-I", inc}
	}
	for name, flags := range variants {
		t.Run(name, func(t *testing.T) {
			abistub.Child(t, "TestABI", "CUPTI_ABI_STUB", abistub.Build(t, abiStub, flags...))
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
		c = driver.Context(0xfedcba9876543210)
		s = driver.Stream(0x8000000000000008)
	)
	tests := []struct {
		name string
		call func() error
		want []uint64
	}{
		{
			name: "ActivityEnableContext",
			call: func() error { return ActivityEnableContext(c, ActivityKind(-3)) },
			want: []uint64{uint64(c), 0xfffffffd},
		},
		{
			name: "GetContextId",
			call: func() error {
				id, err := GetContextId(c)
				if err == nil && id != 0x80000001 {
					t.Errorf("GetContextId = %#x, want 0x80000001", id)
				}
				return err
			},
			want: []uint64{uint64(c)},
		},
		{
			name: "GetStreamIdEx",
			call: func() error {
				id, err := GetStreamIdEx(c, s, 0xff)
				if err == nil && id != 0xfffffffe {
					t.Errorf("GetStreamIdEx = %#x, want 0xfffffffe", id)
				}
				return err
			},
			want: []uint64{uint64(c), uint64(s), 0xff},
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

			stub.Reset(int32(ErrInvalidParameter))
			defer stub.Reset(0)
			var ce *CallError
			if err := tt.call(); !errors.As(err, &ce) || ce.Result != ErrInvalidParameter {
				t.Errorf("with the stub failing: %v, want a *CallError with ErrInvalidParameter", err)
			}
		})
	}
}
