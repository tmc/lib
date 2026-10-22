//go:build linux

package main

import (
	"errors"
	"io"
	"reflect"
	"testing"
	"unsafe"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
	"github.com/tmc/lib/examples/ds4-min/dsv4shape"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// Only the test constructs opaque slice identities; no driver operation may
// receive them. Check the private layout before using the approved fault seam.
func shapeTestSlice(t *testing.T, n int) cudamem.Slice[int32] {
	t.Helper()
	type header struct {
		ptr      driver.DevicePtr
		len, cap int
		size     uintptr
		alloc    unsafe.Pointer
	}
	var v cudamem.Slice[int32]
	typ := reflect.TypeOf(v)
	if typ.NumField() != 5 {
		t.Fatal("slice field count changed")
	}
	names := []string{"ptr", "len", "cap", "elemSize", "alloc"}
	types := []string{"driver.DevicePtr", "int", "int", "uintptr", "*cudamem.allocation"}
	for i, want := range []uintptr{unsafe.Offsetof(header{}.ptr), unsafe.Offsetof(header{}.len), unsafe.Offsetof(header{}.cap), unsafe.Offsetof(header{}.size), unsafe.Offsetof(header{}.alloc)} {
		if typ.Field(i).Offset != want || typ.Field(i).Name != names[i] || typ.Field(i).Type.String() != types[i] {
			t.Fatal("slice test layout changed")
		}
	}
	if unsafe.Sizeof(v) != unsafe.Sizeof(header{}) {
		t.Fatal("slice test size changed")
	}
	*(*header)(unsafe.Pointer(&v)) = header{ptr: driver.DevicePtr(4096 + n*64), len: n, cap: n, size: 4}
	return v
}

func TestShapeGroupingOwnership(t *testing.T) {
	for _, fault := range []string{"allocation", "load", "drain", "free", "unload", "restore"} {
		t.Run(fault, func(t *testing.T) {
			cause := errors.New("fault")
			r := &cudaFoundation{device: new(cuda.Device)}
			failed := true
			frees, unloads := 0, 0
			run := func(fn func() error) error { return fn() }
			ops := shapeGroupingOperations{
				alloc: func(n int) (cudamem.Slice[int32], error) {
					v := shapeTestSlice(t, n)
					if failed && fault == "allocation" {
						return v, cause
					}
					return v, nil
				},
				load: func() (driver.Module, driver.Function, error) {
					if failed && fault == "load" {
						return 1, 0, cause
					}
					return 1, 2, nil
				},
				run: run, closeRun: func(fn func() error) error {
					if err := fn(); err != nil {
						return err
					}
					if failed && fault == "restore" && unloads > 0 {
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
				free: func(*cudamem.Slice[int32]) error {
					frees++
					if failed && (fault == "free" || fault == "allocation") {
						return cause
					}
					return nil
				},
				unload: func(driver.Module) error {
					unloads++
					if failed && (fault == "unload" || fault == "load") {
						return cause
					}
					return nil
				},
			}
			c, _ := dsv4shape.For(dsv4shape.Pro)
			s, err := newCUDAShapeGroupingWithOperations(r, c, 1, ops)
			var owner io.Closer
			if fault == "allocation" || fault == "load" {
				if s != nil || !errors.Is(err, cause) || !errors.As(err, &owner) {
					t.Fatalf("lost error owner: %v %v", s, err)
				}
			} else {
				if err != nil {
					t.Fatal(err)
				}
				owner = s
				if err = s.Close(); !errors.Is(err, cause) {
					t.Fatalf("fault did not fire: %v", err)
				}
				if fault == "drain" && (frees != 0 || unloads != 0) {
					t.Fatal("released before drain")
				}
				if fault == "restore" && (s.module != 0 || s.fn != 0) {
					t.Fatal("successful unload ownership retained")
				}
			}
			failed = false
			before := unloads
			if err := owner.Close(); err != nil {
				t.Fatal(err)
			}
			if fault == "restore" && unloads != before {
				t.Fatal("unloaded twice after restore error")
			}
			beforeF, beforeU := frees, unloads
			if err := owner.Close(); err != nil {
				t.Fatal(err)
			}
			if frees != beforeF || unloads != beforeU {
				t.Fatal("release not idempotent")
			}
		})
	}
}

func TestShapeGroupingClosedFoundation(t *testing.T) {
	r := &cudaFoundation{closing: true}
	c, _ := dsv4shape.For(dsv4shape.Pro)
	if _, err := newCUDAShapeGrouping(r, c, 1); !errors.Is(err, dsv4.ErrClosed) {
		t.Fatal("constructed on closed foundation")
	}
	s := &cudaShapeGrouping{r: r, plan: shapeGroupingPlan{pairs: 6}, module: 1, fn: 2}
	if err := s.EncodeAndValidate(shapeTestSlice(t, 6)); !errors.Is(err, dsv4.ErrClosed) {
		t.Fatal("encoded on closed foundation")
	}
}

func TestShapeGroupingSuccessfulOwnerAdmission(t *testing.T) {
	for _, fault := range []string{"null allocation", "null pointer", "short allocation", "long allocation", "null module", "null function", "valid"} {
		t.Run(fault, func(t *testing.T) {
			r := &cudaFoundation{device: new(cuda.Device)}
			allocations, frees, unloads := 0, 0, 0
			run := func(fn func() error) error { return fn() }
			ops := shapeGroupingOperations{
				alloc: func(n int) (cudamem.Slice[int32], error) {
					allocations++
					if fault == "null allocation" {
						return cudamem.Slice[int32]{}, nil
					}
					if fault == "short allocation" {
						n--
					}
					if fault == "long allocation" {
						n++
					}
					v := shapeTestSlice(t, n)
					if fault == "null pointer" {
						*(*driver.DevicePtr)(unsafe.Pointer(&v)) = 0
					}
					return v, nil
				},
				load: func() (driver.Module, driver.Function, error) {
					if fault == "null module" {
						return 0, 2, nil
					}
					if fault == "null function" {
						return 1, 0, nil
					}
					return 1, 2, nil
				},
				run: run, closeRun: run, drain: func() error { return nil },
				free:   func(*cudamem.Slice[int32]) error { frees++; return nil },
				unload: func(driver.Module) error { unloads++; return nil },
			}
			c, _ := dsv4shape.For(dsv4shape.Pro)
			s, err := newCUDAShapeGroupingWithOperations(r, c, 1, ops)
			if fault == "valid" {
				if err != nil {
					t.Fatal(err)
				}
				if err := s.Close(); err != nil {
					t.Fatal(err)
				}
			} else if s != nil || err == nil {
				t.Fatalf("invalid successful owner admitted: %v %v", s, err)
			}
			wantFrees := allocations
			if fault == "null allocation" || fault == "null pointer" {
				wantFrees = 0
			}
			wantUnloads := 0
			if fault == "null function" || fault == "valid" {
				wantUnloads = 1
			}
			if frees != wantFrees || unloads != wantUnloads {
				t.Fatalf("cleanup frees=%d/%d unloads=%d/%d", frees, wantFrees, unloads, wantUnloads)
			}
		})
	}
}

func TestShapeGroupingReleasedClose(t *testing.T) {
	s := &cudaShapeGrouping{r: &cudaFoundation{}}
	if err := s.Close(); err != nil {
		t.Fatalf("empty owner: %v", err)
	}
	if !s.closed {
		t.Fatal("empty owner remained open")
	}
}

func TestShapeGroupingConsumedFree(t *testing.T) {
	cause := errors.New("free after consume")
	r := &cudaFoundation{device: new(cuda.Device)}
	s := &cudaShapeGrouping{r: r, offsets: shapeTestSlice(t, 8)}
	run := func(fn func() error) error { return fn() }
	frees, drains := 0, 0
	s.ops = shapeGroupingOperations{closeRun: run, drain: func() error { drains++; return nil },
		free: func(p *cudamem.Slice[int32]) error { frees++; *p = cudamem.Slice[int32]{}; return cause }}
	if err := s.Close(); !errors.Is(err, cause) {
		t.Fatal(err)
	}
	if s.offsets.Ptr() != 0 {
		t.Fatal("consumed allocation retained")
	}
	r.device = nil
	if err := s.Close(); err != nil {
		t.Fatalf("released owner: %v", err)
	}
	if frees != 1 || drains != 1 {
		t.Fatalf("released twice: frees=%d drains=%d", frees, drains)
	}
}

func TestShapeGroupingMissingOwnerRetainsResources(t *testing.T) {
	v := shapeTestSlice(t, 8)
	s := &cudaShapeGrouping{r: &cudaFoundation{}, offsets: v, module: 1, fn: 2}
	if err := s.Close(); err == nil {
		t.Fatal("live resources lost missing-owner error")
	}
	if s.offsets.Ptr() != v.Ptr() || s.module != 1 || s.fn != 2 {
		t.Fatal("live resources lost without release")
	}
}
