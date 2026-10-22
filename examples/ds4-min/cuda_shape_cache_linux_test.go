//go:build linux

package main

import (
	"errors"
	"io"
	"math"
	"reflect"
	"testing"
	"unsafe"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
	"github.com/tmc/lib/examples/ds4-min/dsv4shape"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

func cacheTestSlice(t *testing.T, n, identity int) cudamem.Slice[float32] {
	t.Helper()
	type header struct {
		ptr      driver.DevicePtr
		len, cap int
		elemSize uintptr
		alloc    unsafe.Pointer
	}
	var v cudamem.Slice[float32]
	typ := reflect.TypeOf(v)
	if typ.NumField() != 5 || unsafe.Sizeof(v) != unsafe.Sizeof(header{}) {
		t.Fatal("float slice test layout changed")
	}
	names := []string{"ptr", "len", "cap", "elemSize", "alloc"}
	types := []string{"driver.DevicePtr", "int", "int", "uintptr", "*cudamem.allocation"}
	for i, offset := range []uintptr{unsafe.Offsetof(header{}.ptr), unsafe.Offsetof(header{}.len), unsafe.Offsetof(header{}.cap), unsafe.Offsetof(header{}.elemSize), unsafe.Offsetof(header{}.alloc)} {
		f := typ.Field(i)
		if f.Offset != offset || f.Name != names[i] || f.Type.String() != types[i] {
			t.Fatal("float slice private fields changed")
		}
	}
	*(*header)(unsafe.Pointer(&v)) = header{ptr: driver.DevicePtr(4096 + identity*64), len: n, cap: n, elemSize: 4}
	return v
}

func TestShapeCacheReleasedClose(t *testing.T) {
	s := &cudaShapeCache{buffers: []cudamem.Slice[float32]{cacheTestSlice(t, 1, 1)}, ops: shapeCacheOperations{
		drain: func() error { return nil },
		free:  func(v *cudamem.Slice[float32]) error { *v = cudamem.Slice[float32]{}; return nil },
	}}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	s.ops.drain = func() error { t.Fatal("released cache touched foundation"); return nil }
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestShapeCacheOwnership(t *testing.T) {
	for _, fault := range []string{"allocation", "initialize", "drain", "free", "restore"} {
		t.Run(fault, func(t *testing.T) {
			c, _ := dsv4shape.For(dsv4shape.Pro)
			r := &cudaFoundation{device: new(cuda.Device)}
			plan, err := planShapeCache(c, 1, math.MaxUint64, 0)
			if err != nil {
				t.Fatal(err)
			}
			cause := errors.New("fault")
			failed := true
			allocs, inits, frees := 0, 0, 0
			ops := shapeCacheOperations{
				available: func() (uint64, error) { return math.MaxUint64, nil },
				alloc: func(n int) (cudamem.Slice[float32], error) {
					allocs++
					v := cacheTestSlice(t, n, allocs)
					if failed && fault == "allocation" {
						return v, cause
					}
					return v, nil
				},
				initialize: func(v cudamem.Slice[float32], bits uint32) error {
					if bits != plan.buffers[inits].initial || v.Len() != plan.buffers[inits].elements {
						t.Fatal("incorrect initialization")
					}
					inits++
					if failed && fault == "initialize" {
						return cause
					}
					return nil
				},
				drain: func() error {
					if failed && fault == "drain" {
						return cause
					}
					return nil
				},
				free: func(v *cudamem.Slice[float32]) error {
					frees++
					if failed {
						if fault == "restore" {
							*v = cudamem.Slice[float32]{}
						}
						return cause
					}
					return nil
				},
			}
			s, err := newCUDAShapeCacheWithOperations(r, c, 1, 0, ops)
			var owner io.Closer
			if fault == "allocation" || fault == "initialize" || fault == "drain" {
				if s != nil || !errors.Is(err, cause) || !errors.As(err, &owner) {
					t.Fatalf("missing retained owner: %v %v", s, err)
				}
				if fault == "drain" && frees != 0 {
					t.Fatal("freed without drain")
				}
			} else {
				if err != nil {
					t.Fatal(err)
				}
				owner = s
				if !errors.Is(s.Close(), cause) {
					t.Fatal("release fault did not fire")
				}
			}
			failed = false
			before := frees
			if err := owner.Close(); err != nil {
				t.Fatal(err)
			}
			if fault == "restore" && frees != before {
				t.Fatal("freed twice after restore failure")
			}
			before = frees
			if err := owner.Close(); err != nil {
				t.Fatal(err)
			}
			if frees != before {
				t.Fatal("cleanup not idempotent")
			}
		})
	}
}

func TestShapeCacheRejectsBeforeAllocation(t *testing.T) {
	c, _ := dsv4shape.For(dsv4shape.Pro)
	r := &cudaFoundation{device: new(cuda.Device)}
	ops := shapeCacheOperations{available: func() (uint64, error) { return 1, nil }, alloc: func(int) (cudamem.Slice[float32], error) {
		t.Fatal("allocation before capacity gate")
		return cudamem.Slice[float32]{}, nil
	}}
	if _, err := newCUDAShapeCacheWithOperations(r, c, 4096, 0, ops); err == nil {
		t.Fatal("insufficient capacity admitted")
	}
	r.closing = true
	if _, err := newCUDAShapeCacheWithOperations(r, c, 4096, 0, ops); !errors.Is(err, dsv4.ErrClosed) {
		t.Fatal("closed foundation admitted")
	}
}
