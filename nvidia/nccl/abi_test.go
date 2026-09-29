//go:build darwin || linux

package nccl

import (
	"errors"
	"os"
	"slices"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/internal/abistub"
)

// abiStub stands in for libnccl. Every collective takes a driver.Stream,
// a type from another package, which the binding passes as a machine word.
const abiStub = abistub.Prelude + `
#ifdef STUB_REAL_HEADER
#include <nccl.h>
#else
typedef int ncclResult_t;
typedef int ncclDataType_t;
typedef int ncclRedOp_t;
typedef struct ncclComm *ncclComm_t;
typedef struct CUstream_st *cudaStream_t;
#endif

ncclResult_t ncclAllReduce(const void *send, void *recv, size_t n, ncclDataType_t dt, ncclRedOp_t op, ncclComm_t c, cudaStream_t s) {
	BEGIN; R(send); R(recv); R(n); R((uint32_t)dt); R((uint32_t)op); R(c); R(s); return RET;
}
ncclResult_t ncclSend(const void *send, size_t n, ncclDataType_t dt, int peer, ncclComm_t c, cudaStream_t s) {
	BEGIN; R(send); R(n); R((uint32_t)dt); S(peer); R(c); R(s); return RET;
}
ncclResult_t ncclRecv(void *recv, size_t n, ncclDataType_t dt, int peer, ncclComm_t c, cudaStream_t s) {
	BEGIN; R(recv); R(n); R((uint32_t)dt); S(peer); R(c); R(s); return RET;
}
`

// TestABI checks, without a GPU, that the generated nccl collectives
// deliver their arguments to C bit for bit, and reports what each call
// allocates. It runs in a child process, because Load binds the stub for
// the life of the process.
func TestABI(t *testing.T) {
	if path := os.Getenv("NCCL_ABI_STUB"); path != "" {
		testABI(t, path)
		return
	}
	variants := map[string][]string{"mirror": nil}
	if inc := abistub.CUDAInclude("nccl.h"); inc != "" {
		variants["nccl.h"] = []string{"-DSTUB_REAL_HEADER", "-I", inc}
	}
	for name, flags := range variants {
		t.Run(name, func(t *testing.T) {
			abistub.Child(t, "TestABI", "NCCL_ABI_STUB", abistub.Build(t, abiStub, flags...))
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
		c = Comm(0xfedcba9876543210)
		s = driver.Stream(0x8000000000000008)
		p = driver.DevicePtr(0xffffffffffff0000)
		q = driver.DevicePtr(0xffff000000000100)
		n = uint(0x8000000000000040)
	)
	tests := []struct {
		name string
		call func() error
		want []uint64
	}{
		{
			name: "AllReduce",
			call: func() error { return c.AllReduce(p, q, n, DataType(-2), RedOp(0x7ffffff3), s) },
			want: []uint64{uint64(p), uint64(q), uint64(n), 0xfffffffe, 0x7ffffff3, uint64(c), uint64(s)},
		},
		{
			name: "Send",
			call: func() error { return c.Send(p, n, DataType(7), -5, s) },
			want: []uint64{uint64(p), uint64(n), 7, 0xfffffffffffffffb, uint64(c), uint64(s)},
		},
		{
			name: "Recv",
			call: func() error { return c.Recv(q, n, DataType(7), 0x7ffffff6, s) },
			want: []uint64{uint64(q), uint64(n), 7, 0x7ffffff6, uint64(c), uint64(s)},
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
			// driver.Stream is a word, so the collectives are positional
			// calls and allocate nothing.
			if n := testing.AllocsPerRun(100, func() { tt.call() }); n != 0 {
				t.Errorf("%v allocs/op, want 0", n)
			}

			stub.Reset(int32(ErrUnhandledCudaError))
			defer stub.Reset(0)
			var ce *CallError
			if err := tt.call(); !errors.As(err, &ce) || ce.Result != ErrUnhandledCudaError {
				t.Errorf("with the stub failing: %v, want a *CallError with ErrUnhandledCudaError", err)
			}
		})
	}
}
