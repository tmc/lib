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
	TraceSymbolName         = "rl_trace"
	TraceNextSymbolName     = "rl_trace_next"
	TraceLastBitsSymbolName = "rl_trace_last_bits"
)

// TraceHeader returns the CUDA C++ device declarations for the trace hash accumulator,
// fold functions (dump_var), reset kernel, and readback kernel.
func TraceHeader() string {
	return `// Generated trace hash device declarations
#ifndef TMC_TRACEHASH_H
#define TMC_TRACEHASH_H

extern "C" {
__device__ unsigned long long rl_trace = 14695981039346656037ULL;
__device__ unsigned int rl_trace_next = 0;
__device__ unsigned int rl_trace_last_bits = 0;

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

// Generated kernels pass node-major, row-major tickets. A single block owns
// the ordered sequence, so waiting cannot depend on another unscheduled block.
__device__ inline void trace_ordered_float(float value, unsigned int ticket, unsigned int fold_limit) {
    while (atomicAdd(&rl_trace_next, 0U) != ticket) { }
    union { float f; unsigned int u; } bits;
    bits.f = value;
    if (ticket < fold_limit) {
        rl_trace_last_bits = bits.u;
        trace_write_byte((unsigned char)(bits.u & 0xff));
        trace_write_byte((unsigned char)((bits.u >> 8) & 0xff));
        trace_write_byte((unsigned char)((bits.u >> 16) & 0xff));
        trace_write_byte((unsigned char)((bits.u >> 24) & 0xff));
    }
    __threadfence();
    atomicAdd(&rl_trace_next, 1U);
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

// EmitOrderedDumpVar folds expr at a canonical sequence ticket.
func EmitOrderedDumpVar(expr, ticket, limit string) string {
	return fmt.Sprintf("trace_ordered_float(%s, %s, %s);\n", expr, ticket, limit)
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

// ReadDeviceTraceNext reports the next ordered fold ticket for generated kernels.
func ReadDeviceTraceNext(mod driver.Module) (uint32, error) {
	dptr, size, err := mod.Global(TraceNextSymbolName)
	if err != nil {
		return 0, fmt.Errorf("nvrtcx.ReadDeviceTraceNext: find symbol %q: %w", TraceNextSymbolName, err)
	}
	if size != 4 {
		return 0, fmt.Errorf("nvrtcx.ReadDeviceTraceNext: unexpected symbol %q size %d (want 4)", TraceNextSymbolName, size)
	}
	var next uint32
	if err := dptr.CopyToHostPtr(unsafe.Pointer(&next), 4); err != nil {
		return 0, fmt.Errorf("nvrtcx.ReadDeviceTraceNext: read %q: %w", TraceNextSymbolName, err)
	}
	return next, nil
}

// ReadDeviceTraceLastBits reports the most recently folded float32 bit pattern.
func ReadDeviceTraceLastBits(mod driver.Module) (uint32, error) {
	dptr, size, err := mod.Global(TraceLastBitsSymbolName)
	if err != nil {
		return 0, fmt.Errorf("nvrtcx.ReadDeviceTraceLastBits: find symbol %q: %w", TraceLastBitsSymbolName, err)
	}
	if size != 4 {
		return 0, fmt.Errorf("nvrtcx.ReadDeviceTraceLastBits: unexpected symbol %q size %d (want 4)", TraceLastBitsSymbolName, size)
	}
	var bits uint32
	if err := dptr.CopyToHostPtr(unsafe.Pointer(&bits), 4); err != nil {
		return 0, fmt.Errorf("nvrtcx.ReadDeviceTraceLastBits: read %q: %w", TraceLastBitsSymbolName, err)
	}
	return bits, nil
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
	next, size, err := mod.Global(TraceNextSymbolName)
	if err != nil {
		return fmt.Errorf("nvrtcx.ResetDeviceTrace: find symbol %q: %w", TraceNextSymbolName, err)
	}
	if size != 4 {
		return fmt.Errorf("nvrtcx.ResetDeviceTrace: unexpected symbol %q size %d (want 4)", TraceNextSymbolName, size)
	}
	var zero uint32
	if err := next.CopyFromHostPtr(unsafe.Pointer(&zero), 4); err != nil {
		return fmt.Errorf("nvrtcx.ResetDeviceTrace: write %q: %w", TraceNextSymbolName, err)
	}
	last, size, err := mod.Global(TraceLastBitsSymbolName)
	if err != nil {
		return fmt.Errorf("nvrtcx.ResetDeviceTrace: find symbol %q: %w", TraceLastBitsSymbolName, err)
	}
	if size != 4 {
		return fmt.Errorf("nvrtcx.ResetDeviceTrace: unexpected symbol %q size %d (want 4)", TraceLastBitsSymbolName, size)
	}
	if err := last.CopyFromHostPtr(unsafe.Pointer(&zero), 4); err != nil {
		return fmt.Errorf("nvrtcx.ResetDeviceTrace: write %q: %w", TraceLastBitsSymbolName, err)
	}
	return nil
}
