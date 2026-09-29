//go:build darwin || linux

package driver

import (
	"errors"
	"math"
	"os"
	"slices"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/internal/abistub"
)

// abiStub stands in for libcuda. Each function records the arguments it
// received in stubRec, widened to 64 bits (signed integers sign-extended,
// enums as their 32 bits, since cuda.h's enums without negative values are
// unsigned, and floating-point values as their bit patterns), writes a recognizable value
// to each output, and returns stubRet. The test compares the record with
// what it passed, bit for bit.
//
// Built with -DSTUB_REAL_HEADER, the stub takes its types from the real
// cuda.h, so the compiler rejects a stub prototype that differs from the
// library's, and the layouts the stub reads are the library's own.
const abiStub = abistub.Prelude + `
#ifdef STUB_REAL_HEADER
#include <cuda.h>
#else
typedef int CUresult;
typedef int CUdevice;
typedef unsigned long long CUdeviceptr;
typedef struct CUctx_st *CUcontext;
typedef struct CUfunc_st *CUfunction;
typedef struct CUstream_st *CUstream;
typedef struct CUevent_st *CUevent;
typedef struct CUgraphExec_st *CUgraphExec;
typedef struct CUtexref_st *CUtexref;
typedef int CUmem_advise;
typedef int CUmemLocationType;
typedef struct { CUmemLocationType type; union { int id; }; } CUmemLocation;
typedef struct { char reserved[64]; } CUipcMemHandle;
typedef struct { char reserved[64]; } CUipcEventHandle;
typedef int CUlaunchAttributeID;
typedef union {
	char pad[64];
	int cooperative;
	struct { unsigned int x, y, z; } clusterDim;
	int priority;
} CUlaunchAttributeValue;
typedef struct {
	CUlaunchAttributeID id;
	char pad[8 - sizeof(CUlaunchAttributeID)];
	CUlaunchAttributeValue value;
} CUlaunchAttribute;
typedef struct {
	unsigned int gridDimX, gridDimY, gridDimZ;
	unsigned int blockDimX, blockDimY, blockDimZ;
	unsigned int sharedMemBytes;
	CUstream hStream;
	CUlaunchAttribute *attrs;
	unsigned int numAttrs;
} CUlaunchConfig;
typedef uint32_t cuuint32_t;
typedef uint64_t cuuint64_t;
typedef struct { cuuint64_t opaque[16]; } CUtensorMap;
typedef int CUtensorMapDataType;
typedef int CUtensorMapInterleave;
typedef int CUtensorMapSwizzle;
typedef int CUtensorMapL2promotion;
typedef int CUtensorMapFloatOOBfill;
#endif

const uint64_t stubLayout[] = {
	sizeof(CUlaunchConfig),
	offsetof(CUlaunchConfig, sharedMemBytes),
	offsetof(CUlaunchConfig, hStream),
	offsetof(CUlaunchConfig, attrs),
	offsetof(CUlaunchConfig, numAttrs),
	sizeof(CUlaunchAttribute),
	offsetof(CUlaunchAttribute, value),
	sizeof(CUmemLocation),
	sizeof(CUipcMemHandle),
};

static void handle(const char *p) {
	uint64_t lo, hi;
	memcpy(&lo, p, 8);
	memcpy(&hi, p + 56, 8);
	R(lo);
	R(hi);
}

CUresult cuLaunchKernel(CUfunction f,
	unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
	unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
	unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra) {
	BEGIN;
	R(f); R(gridDimX); R(gridDimY); R(gridDimZ);
	R(blockDimX); R(blockDimY); R(blockDimZ);
	R(sharedMemBytes); R(hStream); R(kernelParams); R(extra);
	if (kernelParams) {
		R(*(uint64_t *)kernelParams[0]);
		S(*(int32_t *)kernelParams[1]);
		F32(*(float *)kernelParams[2]);
	}
	return RET;
}

CUresult cuLaunchKernelEx(const CUlaunchConfig *c, CUfunction f, void **kernelParams, void **extra) {
	BEGIN;
	R(c->gridDimX); R(c->gridDimY); R(c->gridDimZ);
	R(c->blockDimX); R(c->blockDimY); R(c->blockDimZ);
	R(c->sharedMemBytes); R(c->hStream); R(c->numAttrs);
	for (unsigned int i = 0; i < c->numAttrs; i++) {
		R(c->attrs[i].id);
		R(((uint32_t *)&c->attrs[i].value)[0]);
		R(((uint32_t *)&c->attrs[i].value)[1]);
		R(((uint32_t *)&c->attrs[i].value)[2]);
	}
	R(f); R(kernelParams); R(extra);
	if (kernelParams) {
		R(*(uint64_t *)kernelParams[0]);
	}
	return RET;
}

CUresult cuCtxGetCurrent(CUcontext *pctx) {
	BEGIN;
	*pctx = (CUcontext)0xffff00000000c7c0ull;
	return RET;
}

CUresult cuCtxSetCurrent(CUcontext ctx) { BEGIN; R(ctx); return RET; }
CUresult cuEventRecord(CUevent e, CUstream s) { BEGIN; R(e); R(s); return RET; }
CUresult cuGraphLaunch(CUgraphExec g, CUstream s) { BEGIN; R(g); R(s); return RET; }

CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr dst, const void *src, size_t n, CUstream s) {
	BEGIN; R(dst); R(src); R(n); R(s); return RET;
}

CUresult cuMemcpyDtoHAsync_v2(void *dst, CUdeviceptr src, size_t n, CUstream s) {
	BEGIN; R(dst); R(src); R(n); R(s); return RET;
}

CUresult cuMemsetD8_v2(CUdeviceptr d, unsigned char v, size_t n) { BEGIN; R(d); R(v); R(n); return RET; }
CUresult cuMemsetD16_v2(CUdeviceptr d, unsigned short v, size_t n) { BEGIN; R(d); R(v); R(n); return RET; }
CUresult cuMemsetD32_v2(CUdeviceptr d, unsigned int v, size_t n) { BEGIN; R(d); R(v); R(n); return RET; }
CUresult cuMemsetD8Async(CUdeviceptr d, unsigned char v, size_t n, CUstream s) { BEGIN; R(d); R(v); R(n); R(s); return RET; }
CUresult cuMemsetD16Async(CUdeviceptr d, unsigned short v, size_t n, CUstream s) { BEGIN; R(d); R(v); R(n); R(s); return RET; }
CUresult cuMemsetD32Async(CUdeviceptr d, unsigned int v, size_t n, CUstream s) { BEGIN; R(d); R(v); R(n); R(s); return RET; }

CUresult cuParamSetf(CUfunction f, int offset, float v) { BEGIN; R(f); S(offset); F32(v); return RET; }
CUresult cuParamSeti(CUfunction f, int offset, unsigned int v) { BEGIN; R(f); S(offset); R(v); return RET; }

CUresult cuTexRefSetMipmapLevelClamp(CUtexref t, float lo, float hi) {
	BEGIN; R(t); F32(lo); F32(hi); return RET;
}

CUresult cuMemAdvise_v2(CUdeviceptr d, size_t n, CUmem_advise a, CUmemLocation loc) {
	BEGIN; R(d); R(n); R((uint32_t)a); R((uint32_t)loc.type); S(loc.id); return RET;
}

CUresult cuMemPrefetchAsync_v2(CUdeviceptr d, size_t n, CUmemLocation loc, unsigned int flags, CUstream s) {
	BEGIN; R(d); R(n); R((uint32_t)loc.type); S(loc.id); R(flags); R(s); return RET;
}

CUresult cuIpcOpenMemHandle_v2(CUdeviceptr *p, CUipcMemHandle h, unsigned int flags) {
	BEGIN; handle(h.reserved); R(flags);
	*p = 0xffffffff00001000ull;
	return RET;
}

CUresult cuIpcOpenEventHandle(CUevent *e, CUipcEventHandle h) {
	BEGIN; handle(h.reserved);
	*e = (CUevent)0x8000000000000e0eull;
	return RET;
}

// cuTensorMapEncodeTiled's last four arguments are ints on the stack, which
// Apple's arm64 ABI packs into consecutive 4-byte slots.
CUresult cuTensorMapEncodeTiled(CUtensorMap *m, CUtensorMapDataType dt, cuuint32_t rank, void *addr,
	const cuuint64_t *dim, const cuuint64_t *strides, const cuuint32_t *box, const cuuint32_t *estr,
	CUtensorMapInterleave il, CUtensorMapSwizzle sw, CUtensorMapL2promotion l2, CUtensorMapFloatOOBfill oob) {
	BEGIN; R(m); R((uint32_t)dt); R(rank); R(addr);
	R(dim[0]); R(dim[2]); R(strides[0]); R(strides[1]); R(box[0]); R(box[2]); R(estr[0]); R(estr[2]);
	R((uint32_t)il); R((uint32_t)sw); R((uint32_t)l2); R((uint32_t)oob);
	m->opaque[0] = 0xfeedfacecafebeefull;
	m->opaque[15] = 0x8000000000000001ull;
	return RET;
}

CUresult cuMemGetInfo_v2(size_t *free, size_t *total) {
	BEGIN;
	*free = 0x8000000000000123ull;
	*total = 0xfffffffffffff000ull;
	return RET;
}

CUresult cuDeviceTotalMem_v2(size_t *bytes, CUdevice dev) {
	BEGIN; S(dev);
	*bytes = 0x8000000000000456ull;
	return RET;
}
`

// Arguments with high bits set, so a narrowed, sign-extended or misplaced
// word does not survive the comparison.
const (
	abiFunc   = Function(0xfedcba9876543210)
	abiStream = Stream(0x8000000000000008)
	abiPtr    = DevicePtr(0xffffffffffff0000)
	abiSize   = uint(0x8000000000000040)
)

// TestABI checks, without a GPU, that the generated driver wrappers deliver
// every argument of the hot calls to C bit for bit, read their outputs and
// errors back, and do not allocate on success. It runs in a child process,
// because Load binds the stub for the life of the process.
func TestABI(t *testing.T) {
	if lib := os.Getenv("DRIVER_ABI_STUB"); lib != "" {
		testABI(t, lib)
		return
	}
	variants := map[string][]string{"mirror": nil}
	if inc := abistub.CUDAInclude("cuda.h"); inc != "" {
		variants["cuda.h"] = []string{"-DSTUB_REAL_HEADER", "-I", inc}
	}
	for name, flags := range variants {
		t.Run(name, func(t *testing.T) {
			abistub.Child(t, "TestABI", "DRIVER_ABI_STUB", abistub.Build(t, abiStub, flags...))
		})
	}
}

func testABI(t *testing.T, path string) {
	if err := Load(path); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { Close() })
	if f := BindFailures(); len(f) != 0 {
		t.Fatalf("BindFailures() = %v", f)
	}
	stub, err := abistub.NewRecorder(func(name string) (uintptr, error) { return lookupLibrarySymbol(lib, name) })
	if err != nil {
		t.Fatal(err)
	}

	t.Run("layout", func(t *testing.T) {
		var c cLaunchConfig
		var a LaunchAttribute
		want := []uint64{
			uint64(unsafe.Sizeof(c)),
			uint64(unsafe.Offsetof(c.SharedMemBytes)),
			uint64(unsafe.Offsetof(c.HStream)),
			uint64(unsafe.Offsetof(c.Attrs)),
			uint64(unsafe.Offsetof(c.NumAttrs)),
			uint64(unsafe.Sizeof(a)),
			uint64(unsafe.Offsetof(a.Value)),
			uint64(unsafe.Sizeof(MemLocation{})),
			uint64(unsafe.Sizeof(IpcMemHandle{})),
		}
		got, err := stub.Uint64s("stubLayout", len(want))
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(got, want) {
			t.Errorf("C layout %v, Go layout %v", got, want)
		}
	})

	// Kernel parameters live in package-level storage, as they would in a
	// caller that reuses them, so the test measures the call, not their
	// allocation.
	params := abiParams
	cfg := &LaunchConfig{
		GridDim:        Dim3{X: 0x80000001, Y: 0x7fff0002, Z: 3},
		BlockDim:       Dim3{X: 1024, Y: 0xfffe0005, Z: 6},
		SharedMemBytes: 0xc0010000,
		HStream:        abiStream,
		Attrs:          make([]LaunchAttribute, 3),
	}
	cfg.Attrs[0].Id = LaunchAttributeCooperative
	*cfg.Attrs[0].Value.Cooperative() = 1
	cfg.Attrs[1].Id = LaunchAttributeClusterDimension
	*cfg.Attrs[1].Value.ClusterDim() = Dim3{X: 2, Y: 0x80000001, Z: 4}
	cfg.Attrs[2].Id = LaunchAttributePriority
	*cfg.Attrs[2].Value.Priority() = -3
	host := make([]byte, 64)
	hostPtr := unsafe.Pointer(&host[0])
	var mh IpcMemHandle
	var eh IpcEventHandle
	for i := range mh.Reserved {
		mh.Reserved[i] = byte(i + 1)
		eh.Reserved[i] = byte(0xff - i)
	}
	loc := MemLocation{Type: MemLocationType(1), Id: -7}

	tests := []struct {
		name   string
		call   func() error
		want   []uint64 // what the stub recorded
		purego bool     // bound through purego.RegisterFunc, which allocates
	}{
		{
			name: "LaunchKernel/nil",
			call: func() error {
				return abiFunc.LaunchKernel(cfg.GridDim, cfg.BlockDim, cfg.SharedMemBytes, abiStream, nil)
			},
			want: []uint64{uint64(abiFunc), 0x80000001, 0x7fff0002, 3, 1024, 0xfffe0005, 6, 0xc0010000, uint64(abiStream), 0, 0},
		},
		{
			name: "LaunchKernel/params",
			call: func() error {
				return abiFunc.LaunchKernel(cfg.GridDim, cfg.BlockDim, cfg.SharedMemBytes, abiStream, params)
			},
			want: []uint64{uint64(abiFunc), 0x80000001, 0x7fff0002, 3, 1024, 0xfffe0005, 6, 0xc0010000, uint64(abiStream),
				uint64(uintptr(unsafe.Pointer(&params[0]))), 0,
				abiArg0, i64(int64(abiArg1)), uint64(math.Float32bits(abiArg2))},
		},
		{
			name: "LaunchKernelEx/nil",
			call: func() error { return abiFunc.LaunchKernelEx(cfg, nil) },
			want: []uint64{0x80000001, 0x7fff0002, 3, 1024, 0xfffe0005, 6, 0xc0010000, uint64(abiStream), 3,
				uint64(LaunchAttributeCooperative), 1, 0, 0,
				uint64(LaunchAttributeClusterDimension), 2, 0x80000001, 4,
				uint64(LaunchAttributePriority), uint64(uint32(0xfffffffd)), 0, 0,
				uint64(abiFunc), 0, 0},
		},
		{
			name: "LaunchKernelEx/params",
			call: func() error { return abiFunc.LaunchKernelEx(cfg, params) },
			want: []uint64{0x80000001, 0x7fff0002, 3, 1024, 0xfffe0005, 6, 0xc0010000, uint64(abiStream), 3,
				uint64(LaunchAttributeCooperative), 1, 0, 0,
				uint64(LaunchAttributeClusterDimension), 2, 0x80000001, 4,
				uint64(LaunchAttributePriority), uint64(uint32(0xfffffffd)), 0, 0,
				uint64(abiFunc), uint64(uintptr(unsafe.Pointer(&params[0]))), 0, abiArg0},
		},
		{
			name: "CtxGetCurrent",
			call: func() error {
				c, err := CtxGetCurrent()
				if err == nil && c != Context(0xffff00000000c7c0) {
					t.Errorf("CtxGetCurrent = %#x, want 0xffff00000000c7c0", c)
				}
				return err
			},
			want: []uint64{},
		},
		{
			name: "Context.SetCurrent",
			call: func() error { return Context(0xffff00000000c7c0).SetCurrent() },
			want: []uint64{0xffff00000000c7c0},
		},
		{
			name: "Event.Record",
			call: func() error { return Event(0x8000000000000e0e).Record(abiStream) },
			want: []uint64{0x8000000000000e0e, uint64(abiStream)},
		},
		{
			name: "GraphExec.Launch",
			call: func() error { return GraphExec(0xc000000000000abc).Launch(abiStream) },
			want: []uint64{0xc000000000000abc, uint64(abiStream)},
		},
		{
			name: "CopyFromHostAsyncPtr",
			call: func() error { return abiPtr.CopyFromHostAsyncPtr(hostPtr, abiSize, abiStream) },
			want: []uint64{uint64(abiPtr), uint64(uintptr(hostPtr)), uint64(abiSize), uint64(abiStream)},
		},
		{
			name: "CopyToHostAsyncPtr",
			call: func() error { return abiPtr.CopyToHostAsyncPtr(hostPtr, abiSize, abiStream) },
			want: []uint64{uint64(uintptr(hostPtr)), uint64(abiPtr), uint64(abiSize), uint64(abiStream)},
		},
		{
			name: "MemsetD8",
			call: func() error { return abiPtr.MemsetD8(0xa5, abiSize) },
			want: []uint64{uint64(abiPtr), 0xa5, uint64(abiSize)},
		},
		{
			name: "MemsetD16",
			call: func() error { return abiPtr.MemsetD16(0xbeef, abiSize) },
			want: []uint64{uint64(abiPtr), 0xbeef, uint64(abiSize)},
		},
		{
			name: "MemsetD32",
			call: func() error { return abiPtr.MemsetD32(0xdeadbeef, abiSize) },
			want: []uint64{uint64(abiPtr), 0xdeadbeef, uint64(abiSize)},
		},
		{
			name: "MemsetD8Async",
			call: func() error { return abiPtr.MemsetD8Async(0x81, abiSize, abiStream) },
			want: []uint64{uint64(abiPtr), 0x81, uint64(abiSize), uint64(abiStream)},
		},
		{
			name: "MemsetD16Async",
			call: func() error { return abiPtr.MemsetD16Async(0x8001, abiSize, abiStream) },
			want: []uint64{uint64(abiPtr), 0x8001, uint64(abiSize), uint64(abiStream)},
		},
		{
			name: "MemsetD32Async",
			call: func() error { return abiPtr.MemsetD32Async(0x80000001, abiSize, abiStream) },
			want: []uint64{uint64(abiPtr), 0x80000001, uint64(abiSize), uint64(abiStream)},
		},
		{
			name: "ParamSetf",
			call: func() error { return abiFunc.ParamSetf(-8, -1.25) },
			want: []uint64{uint64(abiFunc), i64(-8), uint64(math.Float32bits(-1.25))},
		},
		{
			name: "ParamSeti",
			call: func() error { return abiFunc.ParamSeti(-12, 0xfffffffe) },
			want: []uint64{uint64(abiFunc), i64(-12), 0xfffffffe},
		},
		{
			name: "TexRef.SetMipmapLevelClamp",
			call: func() error { return TexRef(0x8000000000000777).SetMipmapLevelClamp(0.5, -1e30) },
			want: []uint64{0x8000000000000777, uint64(math.Float32bits(0.5)), uint64(math.Float32bits(-1e30))},
		},
		{
			name:   "MemAdvise",
			call:   func() error { return abiPtr.MemAdvise(abiSize, MemAdvise(-2), loc) },
			want:   []uint64{uint64(abiPtr), uint64(abiSize), 0xfffffffe, 1, i64(-7)},
			purego: true,
		},
		{
			name:   "MemPrefetchAsync",
			call:   func() error { return abiPtr.MemPrefetchAsync(abiSize, loc, 0x80000001, abiStream) },
			want:   []uint64{uint64(abiPtr), uint64(abiSize), 1, i64(-7), 0x80000001, uint64(abiStream)},
			purego: true,
		},
		{
			name: "IpcOpenMemHandle",
			call: func() error {
				p, err := IpcOpenMemHandle(mh, IpcmemFlags(0x7fffffff))
				if err == nil && p != 0xffffffff00001000 {
					t.Errorf("IpcOpenMemHandle = %#x, want 0xffffffff00001000", p)
				}
				return err
			},
			want:   []uint64{0x0807060504030201, 0x403f3e3d3c3b3a39, 0x7fffffff},
			purego: true,
		},
		{
			name: "IpcOpenEventHandle",
			call: func() error {
				e, err := IpcOpenEventHandle(eh)
				if err == nil && e != 0x8000000000000e0e {
					t.Errorf("IpcOpenEventHandle = %#x, want 0x8000000000000e0e", e)
				}
				return err
			},
			want:   []uint64{0xf8f9fafbfcfdfeff, 0xc0c1c2c3c4c5c6c7},
			purego: true,
		},
		{
			name: "TensorMapEncodeTiled",
			call: func() error {
				abiTensorMap = TensorMap{}
				err := TensorMapEncodeTiled(&abiTensorMap, TensorMapDataType(-2), 3, abiPtr,
					&abiDims[0], &abiStrides[0], &abiBox[0], &abiElemStrides[0],
					TensorMapInterleave(0x7ffffff1), TensorMapSwizzle(-3), TensorMapL2promotion(0x40000004), TensorMapFloatOOBfill(-5))
				if err == nil && (abiTensorMap.Opaque[0] != 0xfeedfacecafebeef || abiTensorMap.Opaque[15] != 0x8000000000000001) {
					t.Errorf("TensorMapEncodeTiled wrote %#x ... %#x", abiTensorMap.Opaque[0], abiTensorMap.Opaque[15])
				}
				return err
			},
			want: []uint64{uint64(uintptr(unsafe.Pointer(&abiTensorMap))), 0xfffffffe, 3, uint64(abiPtr),
				abiDims[0], abiDims[2], abiStrides[0], abiStrides[1], uint64(abiBox[0]), uint64(abiBox[2]),
				uint64(abiElemStrides[0]), uint64(abiElemStrides[2]),
				0x7ffffff1, 0xfffffffd, 0x40000004, 0xfffffffb},
		},
		{
			name: "MemGetInfo",
			call: func() error {
				free, total, err := MemGetInfo()
				if err == nil && (free != 0x8000000000000123 || total != 0xfffffffffffff000) {
					t.Errorf("MemGetInfo = %#x, %#x, want 0x8000000000000123, 0xfffffffffffff000", free, total)
				}
				return err
			},
			want: []uint64{},
		},
		{
			name: "Device.TotalMem",
			call: func() error {
				n, err := Device(-5).TotalMem()
				if err == nil && n != 0x8000000000000456 {
					t.Errorf("TotalMem = %#x, want 0x8000000000000456", n)
				}
				return err
			},
			want: []uint64{i64(-5)},
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
			n := testing.AllocsPerRun(100, func() { tt.call() })
			switch {
			case tt.purego:
				t.Logf("%v allocs/op through purego.RegisterFunc", n)
			case n != 0:
				t.Errorf("%v allocs/op, want 0", n)
			}

			// The same call reports a failure the library returns.
			stub.Reset(int32(ErrLaunchFailed))
			defer stub.Reset(0)
			var ce *CallError
			if err := tt.call(); !errors.As(err, &ce) || ce.Result != ErrLaunchFailed {
				t.Errorf("with the stub failing: %v, want a *CallError with ErrLaunchFailed", err)
			}
		})
	}

	t.Run("absent", func(t *testing.T) {
		if HasStreamSynchronize() {
			t.Fatal("HasStreamSynchronize() = true for a library without cuStreamSynchronize")
		}
		var se *SymbolError
		if err := abiStream.Synchronize(); !errors.As(err, &se) || se.Symbol != "cuStreamSynchronize" {
			t.Errorf("Synchronize() = %v, want *SymbolError for cuStreamSynchronize", err)
		}
		if !HasLaunchKernel() {
			t.Error("HasLaunchKernel() = false for a library with cuLaunchKernel")
		}
	})
}

const (
	abiArg0 = uint64(0xdeadbeefcafef00d)
	abiArg1 = int32(-2)
	abiArg2 = float32(1.5)
)

var (
	abiArgs = struct {
		a uint64
		b int32
		c float32
	}{abiArg0, abiArg1, abiArg2}
	abiParams = []unsafe.Pointer{unsafe.Pointer(&abiArgs.a), unsafe.Pointer(&abiArgs.b), unsafe.Pointer(&abiArgs.c)}
)

// Tensor map arguments live in package-level storage, as a caller's
// descriptor would, so the test measures the call.
var (
	abiTensorMap   TensorMap
	abiDims        = [3]uint64{0x8000000000000100, 2, 0xffffffff00000003}
	abiStrides     = [2]uint64{0x8000000000000200, 0xc000000000000000}
	abiBox         = [3]uint32{0x80000001, 2, 0xfffffff3}
	abiElemStrides = [3]uint32{1, 2, 0x80000004}
)

func i64(v int64) uint64 { return uint64(v) }
