package stage

import (
	"fmt"
	"reflect"
	"unsafe"
)

// Buffer identifies one retained device allocation. Name is a NUL-terminated
// string that remains live through the call. Pointer is a CUDA device address,
// not host storage. Bytes is its declared accessible extent.
type Buffer struct {
	Name    *byte
	Pointer uint64
	Bytes   uint64
}

// Function identifies a CUDA function retained by the caller's generation pin.
// A stage library does not acquire or release this pin.
type Function struct {
	Name   *byte
	Handle uintptr
}

// Context is the pointed-to stage_ctx ABI. The caller retains all named storage,
// stream and function pins until actual GPU completion, including on errors.
// Stage code may use this storage during Call but must not retain its address.
type Context struct {
	Version       uint32
	Size          uint32
	Buffers       *Buffer
	BufferCount   uint64
	Stream        uintptr
	Functions     *Function
	FunctionCount uint64
}

const abiVersion = 1

type abiValues struct {
	Count  uint64
	Values [24]uint64
}

// checkEntrySignature is used for every function registered with purego. In
// particular, passing an aggregate by value is never an admitted entry ABI.
func checkEntrySignature(t reflect.Type) error {
	if t == nil || t.Kind() != reflect.Func || t.IsVariadic() || t.NumIn() != 1 || t.In(0).Kind() != reflect.Pointer || t.NumOut() != 1 || t.Out(0).Kind() != reflect.Int32 {
		return fmt.Errorf("stage entry requires one pointer argument and an int32 result")
	}
	return nil
}

func expectedABI() []uint64 {
	return []uint64{
		abiVersion,
		uint64(unsafe.Sizeof(Context{})), uint64(unsafe.Alignof(Context{})),
		uint64(unsafe.Offsetof(Context{}.Version)), uint64(unsafe.Offsetof(Context{}.Size)),
		uint64(unsafe.Offsetof(Context{}.Buffers)), uint64(unsafe.Offsetof(Context{}.BufferCount)),
		uint64(unsafe.Offsetof(Context{}.Stream)), uint64(unsafe.Offsetof(Context{}.Functions)), uint64(unsafe.Offsetof(Context{}.FunctionCount)),
		uint64(unsafe.Sizeof(Buffer{})), uint64(unsafe.Alignof(Buffer{})),
		uint64(unsafe.Offsetof(Buffer{}.Name)), uint64(unsafe.Offsetof(Buffer{}.Pointer)), uint64(unsafe.Offsetof(Buffer{}.Bytes)),
		uint64(unsafe.Sizeof(Function{})), uint64(unsafe.Alignof(Function{})),
		uint64(unsafe.Offsetof(Function{}.Name)), uint64(unsafe.Offsetof(Function{}.Handle)),
	}
}

const header = `#ifndef CUDAHOTSWAP_STAGE_ABI_H
#define CUDAHOTSWAP_STAGE_ABI_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
struct stage_buffer { const char *name; uint64_t pointer; uint64_t bytes; };
struct stage_function { const char *name; void *handle; };
struct stage_ctx {
 uint32_t version; uint32_t size;
 struct stage_buffer *buffers; uint64_t buffer_count;
 void *stream;
 struct stage_function *functions; uint64_t function_count;
};
struct stage_abi_values { uint64_t count; uint64_t values[24]; };
int stage(struct stage_ctx *);
int stage_abi(struct stage_abi_values *);
#ifdef __cplusplus
}
#endif
#endif
`

const witness = `
#ifdef __cplusplus
#define STAGE_ALIGNOF(T) alignof(T)
extern "C"
#else
#define STAGE_ALIGNOF(T) _Alignof(T)
#endif
int stage_abi(struct stage_abi_values *out) {
 const uint64_t values[] = {
 1,
 sizeof(struct stage_ctx), STAGE_ALIGNOF(struct stage_ctx),
 offsetof(struct stage_ctx, version), offsetof(struct stage_ctx, size),
 offsetof(struct stage_ctx, buffers), offsetof(struct stage_ctx, buffer_count),
 offsetof(struct stage_ctx, stream), offsetof(struct stage_ctx, functions), offsetof(struct stage_ctx, function_count),
 sizeof(struct stage_buffer), STAGE_ALIGNOF(struct stage_buffer),
 offsetof(struct stage_buffer, name), offsetof(struct stage_buffer, pointer), offsetof(struct stage_buffer, bytes),
 sizeof(struct stage_function), STAGE_ALIGNOF(struct stage_function),
 offsetof(struct stage_function, name), offsetof(struct stage_function, handle)
 };
 if (!out || out->count != sizeof(values)/sizeof(values[0])) return -1;
 for (unsigned int i=0; i<sizeof(values)/sizeof(values[0]); ++i) out->values[i]=values[i];
 return 0;
}
`
