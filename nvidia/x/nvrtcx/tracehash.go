package nvrtcx

import (
	"fmt"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
)

// Constants matching FNV-1a 64-bit parameters.
const (
	// TraceHashOffsetBasis is the 64-bit FNV-1a offset basis (14695981039346656037).
	TraceHashOffsetBasis uint64 = 14695981039346656037

	// TraceHashPrime is the 64-bit FNV-1a prime multiplier (1099511628211).
	TraceHashPrime uint64 = 1099511628211

	// TraceSymbolName is the device global variable name holding the accumulator.
	TraceSymbolName = "rl_trace"

	// TraceResetKernelName is the entry point name of the reset kernel.
	TraceResetKernelName = "trace_reset_kernel"

	// TraceReadKernelName is the entry point name of the readback kernel.
	TraceReadKernelName = "trace_read_kernel"
)

// TraceHeader returns the CUDA C++ device declarations for the trace hash accumulator,
// fold functions (dump_var), reset kernel, and readback kernel.
func TraceHeader() string {
	return `// Generated trace hash device declarations
#ifndef TMC_TRACEHASH_H
#define TMC_TRACEHASH_H

extern "C" {
__device__ unsigned long long rl_trace = 14695981039346656037ULL;

__device__ inline void trace_reset(void) {
    rl_trace = 14695981039346656037ULL;
}

__device__ inline unsigned long long trace_finish(void) {
    return rl_trace;
}

__global__ void trace_reset_kernel(void) {
    trace_reset();
}

__global__ void trace_read_kernel(unsigned long long *out) {
    if (out != (unsigned long long*)0) {
        *out = trace_finish();
    }
}
}

__device__ inline void trace_write_byte(unsigned char byte) {
    unsigned long long old_val = rl_trace;
    while (true) {
        unsigned long long new_val = (old_val ^ (unsigned long long)byte) * 1099511628211ULL;
        unsigned long long assumed = atomicCAS(&rl_trace, old_val, new_val);
        if (assumed == old_val) break;
        old_val = assumed;
    }
}

__device__ inline void dump_var(unsigned char v) {
    trace_write_byte(v);
}

__device__ inline void dump_var(char v) {
    trace_write_byte((unsigned char)v);
}

__device__ inline void dump_var(signed char v) {
    trace_write_byte((unsigned char)v);
}

__device__ inline void dump_var(bool v) {
    trace_write_byte(v ? 1 : 0);
}

__device__ inline void dump_var(unsigned short v) {
    trace_write_byte((unsigned char)(v & 0xff));
    trace_write_byte((unsigned char)((v >> 8) & 0xff));
}

__device__ inline void dump_var(short v) {
    dump_var((unsigned short)v);
}

__device__ inline void dump_var(unsigned int v) {
    trace_write_byte((unsigned char)(v & 0xff));
    trace_write_byte((unsigned char)((v >> 8) & 0xff));
    trace_write_byte((unsigned char)((v >> 16) & 0xff));
    trace_write_byte((unsigned char)((v >> 24) & 0xff));
}

__device__ inline void dump_var(int v) {
    dump_var((unsigned int)v);
}

__device__ inline void dump_var(unsigned long long v) {
    dump_var((unsigned int)(v & 0xffffffffULL));
    dump_var((unsigned int)(v >> 32));
}

__device__ inline void dump_var(long long v) {
    dump_var((unsigned long long)v);
}

__device__ inline void dump_var(float v) {
    union {
        float f;
        unsigned int u;
    } cvt;
    cvt.f = v;
    dump_var(cvt.u);
}

__device__ inline void dump_var(double v) {
    union {
        double d;
        unsigned long long u;
    } cvt;
    cvt.d = v;
    dump_var(cvt.u);
}

template <typename T>
__device__ inline void dump_vars(const T *vals, int count) {
    for (int i = 0; i < count; ++i) {
        dump_var(vals[i]);
    }
}

#endif // TMC_TRACEHASH_H
`
}

// EmitDumpVar returns a CUDA C++ statement folding expr into the device accumulator.
func EmitDumpVar(expr string) string {
	return fmt.Sprintf("dump_var(%s);\n", expr)
}

// EmitDumpVars returns a CUDA C++ statement folding an array of count elements into the device accumulator.
func EmitDumpVars(ptrExpr string, countExpr string) string {
	return fmt.Sprintf("dump_vars(%s, %s);\n", ptrExpr, countExpr)
}

// ReadDeviceTrace reads the 64-bit trace accumulator from the loaded module's global symbol.
func ReadDeviceTrace(mod driver.Module) (uint64, error) {
	dptr, size, err := mod.Global(TraceSymbolName)
	if err != nil {
		return 0, fmt.Errorf("nvrtcx.ReadDeviceTrace: find symbol %q: %w", TraceSymbolName, err)
	}
	if size != 8 {
		return 0, fmt.Errorf("nvrtcx.ReadDeviceTrace: unexpected symbol %q size %d (want 8)", TraceSymbolName, size)
	}
	var val uint64
	if err := dptr.CopyToHostPtr(unsafe.Pointer(&val), 8); err != nil {
		return 0, fmt.Errorf("nvrtcx.ReadDeviceTrace: readback %q: %w", TraceSymbolName, err)
	}
	return val, nil
}

// ResetDeviceTrace resets the 64-bit trace accumulator in the loaded module to TraceHashOffsetBasis.
func ResetDeviceTrace(mod driver.Module) error {
	dptr, size, err := mod.Global(TraceSymbolName)
	if err != nil {
		return fmt.Errorf("nvrtcx.ResetDeviceTrace: find symbol %q: %w", TraceSymbolName, err)
	}
	if size != 8 {
		return fmt.Errorf("nvrtcx.ResetDeviceTrace: unexpected symbol %q size %d (want 8)", TraceSymbolName, size)
	}
	basis := TraceHashOffsetBasis
	if err := dptr.CopyFromHostPtr(unsafe.Pointer(&basis), 8); err != nil {
		return fmt.Errorf("nvrtcx.ResetDeviceTrace: write %q: %w", TraceSymbolName, err)
	}
	return nil
}
