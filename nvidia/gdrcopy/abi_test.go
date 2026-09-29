//go:build linux

package gdrcopy

import (
	"errors"
	"os"
	"slices"
	"syscall"
	"testing"
	"unsafe"

	"github.com/ebitengine/purego"
	"github.com/tmc/lib/nvidia/internal/abistub"
)

// abiStub stands in for libgdrapi. gdr_mh_t is a struct of one unsigned
// long, which the binding passes as the MemHandle word; the stub records
// its member. The copies record the addresses they were given rather than
// dereferencing them.
const abiStub = abistub.Prelude + `
struct gdr;
typedef struct gdr *gdr_t;
typedef struct gdr_mh_s { unsigned long h; } gdr_mh_t;

int gdr_get_callback_flag(gdr_t g, gdr_mh_t handle, int *flag) {
	BEGIN; R(g); R(handle.h); *flag = 1; return RET;
}
int gdr_copy_to_mapping(gdr_mh_t handle, void *map_d_ptr, const void *h_ptr, size_t size) {
	BEGIN; R(handle.h); R(map_d_ptr); R(h_ptr); R(size); return RET;
}
int gdr_copy_from_mapping(gdr_mh_t handle, void *h_ptr, const void *map_d_ptr, size_t size) {
	BEGIN; R(handle.h); R(h_ptr); R(map_d_ptr); R(size); return RET;
}
`

// abiBuf is package data, so passing it allocates nothing.
var abiBuf = make([]byte, 64)

// TestABI checks, without a GPU, that the generated wrappers deliver a
// MemHandle to C bit for bit, next to the other arguments, and that the
// copies do not allocate. It runs in a child process, because Load binds the
// stub for the life of the process.
func TestABI(t *testing.T) {
	if path := os.Getenv("GDRCOPY_ABI_STUB"); path != "" {
		testABI(t, path)
		return
	}
	abistub.Child(t, "TestABI", "GDRCOPY_ABI_STUB", abistub.Build(t, abiStub))
}

func testABI(t *testing.T, path string) {
	if err := Load(path); err != nil {
		t.Fatal(err)
	}
	stub, err := abistub.NewRecorder(func(name string) (uintptr, error) { return purego.Dlsym(lib, name) })
	if err != nil {
		t.Fatal(err)
	}
	const (
		g  = Handle(0x8000000000000010)
		mh = MemHandle(0xfedcba9876543210)
		va = uintptr(0x7f0000001000)
	)
	seedPin(t, g, mh, va, 0x1000, true)
	buf := uint64(uintptr(unsafe.Pointer(&abiBuf[0])))
	tests := []struct {
		name   string
		call   func() error
		want   []uint64
		allocs bool // the wrapper passes the address of a local
	}{
		{
			name: "GetCallbackFlag",
			call: func() error {
				f, err := g.GetCallbackFlag(mh)
				if err == nil && !f {
					t.Error("GetCallbackFlag = false, want true")
				}
				return err
			},
			want:   []uint64{uint64(g), uint64(mh)},
			allocs: true,
		},
		{
			name: "CopyToMapping",
			call: func() error { return mh.CopyToMapping(MappedPtr(va+8), abiBuf) },
			want: []uint64{uint64(mh), uint64(va + 8), buf, 64},
		},
		{
			name: "CopyFromMapping",
			call: func() error { return mh.CopyFromMapping(abiBuf[:32], MappedPtr(va+16)) },
			want: []uint64{uint64(mh), buf, uint64(va + 16), 32},
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
			if n := testing.AllocsPerRun(100, func() { tt.call() }); n != 0 && !tt.allocs {
				t.Errorf("%v allocs/op, want 0", n)
			}

			stub.Reset(int32(syscall.EINVAL))
			defer stub.Reset(0)
			if err := tt.call(); !errors.Is(err, syscall.EINVAL) {
				t.Errorf("with the stub failing: %v, want EINVAL", err)
			}
		})
	}
}
