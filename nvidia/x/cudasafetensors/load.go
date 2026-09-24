package cudasafetensors

import (
	"errors"
	"fmt"
	"reflect"
	"unsafe"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/safetensors"
)

// A TensorReader reads a named tensor. [*safetensors.File] and
// [*safetensors.Sharded] implement it.
type TensorReader interface {
	Tensor(name string) (safetensors.Tensor, error)
}

// LoadSlice reads the named tensor from r, allocates device memory for it on
// dev, and copies the tensor's bytes there unchanged. T must have the size
// and kind of the tensor's dtype: float32 for F32, uint16 for F16 or BF16,
// uint8 for U8 or the F8 types, and so on.
//
// On error a nonzero slice must still be freed before dev is closed. It holds
// either a partial allocation or a buffer whose release after a failed copy
// also failed.
func LoadSlice[T any](r TensorReader, name string, dev *cuda.Device) (cudamem.Slice[T], error) {
	return loadSlice(r, name, func(length int) (cudamem.Slice[T], error) {
		return cudamem.Alloc[T](dev, length)
	})
}

type deviceAllocation[T any] interface {
	CopyFromHost([]T) error
	Free() error
}

func loadSlice[T any, A deviceAllocation[T]](r TensorReader, name string, alloc func(int) (A, error)) (A, error) {
	var zero A
	t, err := r.Tensor(name)
	if err != nil {
		return zero, fmt.Errorf("cudasafetensors: %w", err)
	}
	var elem T
	if !compatibleStorage(reflect.TypeOf(elem), t.Dtype) {
		return zero, fmt.Errorf("cudasafetensors: tensor %q dtype %s is incompatible with %T storage", name, t.Dtype, elem)
	}
	n := len(t.Data) / int(unsafe.Sizeof(elem))
	if n == 0 {
		return zero, fmt.Errorf("cudasafetensors: tensor %q has no elements", name)
	}

	dst, err := alloc(n)
	if err != nil {
		return dst, fmt.Errorf("cudasafetensors: allocate tensor %q: %w", name, err)
	}
	src := unsafe.Slice((*T)(unsafe.Pointer(&t.Data[0])), n)
	if err := dst.CopyFromHost(src); err != nil {
		if freeErr := dst.Free(); freeErr != nil {
			return dst, fmt.Errorf("cudasafetensors: copy tensor %q to device: %w", name, errors.Join(err, fmt.Errorf("free allocation: %w", freeErr)))
		}
		return zero, fmt.Errorf("cudasafetensors: copy tensor %q to device: %w", name, err)
	}
	return dst, nil
}

// compatibleStorage reports whether elements of type t hold dtype values
// bit for bit.
func compatibleStorage(t reflect.Type, dtype safetensors.Dtype) bool {
	if t == nil {
		return false
	}
	switch t.Kind() {
	case reflect.Bool:
		return dtype == safetensors.BOOL
	case reflect.Int8:
		return dtype == safetensors.I8
	case reflect.Int16:
		return dtype == safetensors.I16
	case reflect.Int32:
		return dtype == safetensors.I32
	case reflect.Int64:
		return dtype == safetensors.I64
	case reflect.Uint8:
		return dtype == safetensors.U8 || dtype == safetensors.F8E4M3 || dtype == safetensors.F8E5M2
	case reflect.Uint16:
		return dtype == safetensors.U16 || dtype == safetensors.F16 || dtype == safetensors.BF16
	case reflect.Uint32:
		return dtype == safetensors.U32
	case reflect.Uint64:
		return dtype == safetensors.U64
	case reflect.Float32:
		return dtype == safetensors.F32
	case reflect.Float64:
		return dtype == safetensors.F64
	}
	return false
}
