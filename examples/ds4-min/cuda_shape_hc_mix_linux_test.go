//go:build linux

package main

import (
	"errors"
	"fmt"
	"io"
	"reflect"
	"strings"
	"testing"
	"unsafe"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
	"github.com/tmc/lib/examples/ds4-min/dsv4shape"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

func TestShapeOutputHCAdmission(t *testing.T) {
	c, _ := dsv4shape.For(dsv4shape.Pro)
	r := new(cudaFoundation)
	state := &cudaShapeForward{r: r, plan: shapeForwardPlan{config: c, context: 1}}
	collapse := &cudaShapeHC{r: r, config: c, module: 1, collapse: 1, expand: 2}
	for _, fault := range []string{"closed after drain", "closed after unload", "missing norm", "missing project", "valid"} {
		t.Run(fault, func(t *testing.T) {
			hc := &cudaShapeHCMix{r: r, config: c, module: 1, norm: 2, project: 3, sinkhorn: 4}
			cause := errors.New("cleanup fault")
			if strings.HasPrefix(fault, "closed") {
				hc.ops = shapeHCMixOperations{
					drain: func() error {
						if fault == "closed after drain" {
							return cause
						}
						return nil
					},
					closeRun: func(fn func() error) error { return fn() },
					unload:   func(driver.Module) error { return cause },
				}
				if err := hc.Close(); !errors.Is(err, cause) || !hc.closed || hc.module == 0 {
					t.Fatalf("retained closed owner not established: %v", err)
				}
			}
			if fault == "missing norm" {
				hc.norm = 0
			}
			if fault == "missing project" {
				hc.project = 0
			}
			// Nil binding is the next safe admission boundary. Invalid owners
			// must be rejected before even this boundary, never context/launch.
			err := encodeShapeOutput(state, nil, hc, collapse, 1, shapeOutputWeights{}, cudamem.Slice[byte]{}, cudamem.Slice[float32]{}, 1e-6, 1e-6)
			if err == nil {
				t.Fatal("invalid output admitted")
			}
			if fault == "valid" {
				if strings.Contains(err.Error(), "invalid state or module") {
					t.Fatal("valid owner refused")
				}
			} else if !strings.Contains(err.Error(), "invalid state or module") {
				t.Fatalf("terminal or missing-entry HC reached downstream admission: %v", err)
			}
		})
	}
}

func mixTestHalfSlice(t *testing.T, n int) cudamem.Slice[uint16] {
	t.Helper()
	type header struct {
		ptr      driver.DevicePtr
		len, cap int
		elemSize uintptr
		alloc    unsafe.Pointer
	}
	var v cudamem.Slice[uint16]
	typ := reflect.TypeOf(v)
	if typ.NumField() != 5 || unsafe.Sizeof(v) != unsafe.Sizeof(header{}) {
		t.Fatal("half slice layout changed")
	}
	names := []string{"ptr", "len", "cap", "elemSize", "alloc"}
	types := []string{"driver.DevicePtr", "int", "int", "uintptr", "*cudamem.allocation"}
	for i, off := range []uintptr{unsafe.Offsetof(header{}.ptr), unsafe.Offsetof(header{}.len), unsafe.Offsetof(header{}.cap), unsafe.Offsetof(header{}.elemSize), unsafe.Offsetof(header{}.alloc)} {
		f := typ.Field(i)
		if f.Offset != off || f.Name != names[i] || f.Type.String() != types[i] {
			t.Fatal("half slice fields changed")
		}
	}
	*(*header)(unsafe.Pointer(&v)) = header{ptr: 4096, len: n, cap: n, elemSize: 2}
	return v
}

func TestShapeHCMixOwnership(t *testing.T) {
	for _, fault := range []string{"allocation", "module", "function", "missing-entry", "drain", "free", "unload", "restore"} {
		t.Run(fault, func(t *testing.T) {
			c, _ := dsv4shape.For(dsv4shape.Pro)
			b := forwardTestBinding(t, c)
			r := new(cudaFoundation)
			weights := make(map[string]shapeHCWeights)
			for layer := 0; layer < c.Layers; layer++ {
				for _, phase := range []string{"attn", "ffn"} {
					weights[fmt.Sprintf("blk.%d.hc_%s", layer, phase)] = shapeHCWeights{fn: mixTestHalfSlice(t, c.HC*c.Embed*24), scale: cacheTestSlice(t, 3, 20), base: cacheTestSlice(t, 24, 21)}
				}
			}
			cause := errors.New("fault")
			failed := true
			allocs, frees, unloads := 0, 0, 0
			ops := shapeHCMixOperations{
				alloc: func(n int) (cudamem.Slice[float32], error) {
					allocs++
					v := cacheTestSlice(t, n, allocs)
					if failed && fault == "allocation" {
						return v, cause
					}
					return v, nil
				},
				load: func() (driver.Module, driver.Function, driver.Function, driver.Function, error) {
					if failed && fault == "module" {
						return 1, 0, 0, 0, cause
					}
					if failed && fault == "function" {
						return 1, 1, 2, 0, cause
					}
					if failed && fault == "missing-entry" {
						return 1, 1, 2, 0, nil
					}
					return 1, 1, 2, 3, nil
				},
				drain: func() error {
					if r.closing {
						t.Fatal("released Close touched borrowed foundation")
					}
					if failed && fault == "drain" {
						return cause
					}
					return nil
				},
				free: func(v *cudamem.Slice[float32]) error {
					frees++
					if failed && (fault == "free" || fault == "allocation") {
						return cause
					}
					*v = cudamem.Slice[float32]{}
					return nil
				},
				unload: func(driver.Module) error {
					unloads++
					if failed && (fault == "unload" || fault == "module" || fault == "function" || fault == "missing-entry") {
						return cause
					}
					return nil
				},
				closeRun: func(f func() error) error {
					if err := f(); err != nil {
						return err
					}
					if failed && fault == "restore" {
						return cause
					}
					return nil
				},
			}
			s, err := newCUDAShapeHCMixWithOperations(r, c, b, weights, ops)
			var owner io.Closer
			if fault == "allocation" || fault == "module" || fault == "function" || fault == "missing-entry" {
				if s != nil || !errors.Is(err, cause) || !errors.As(err, &owner) {
					t.Fatalf("lost owner: %v", err)
				}
			} else {
				if err != nil {
					t.Fatal(err)
				}
				owner = s
				if !errors.Is(s.Close(), cause) {
					t.Fatal("fault absent")
				}
			}
			if fault == "drain" && (frees != 0 || unloads != 0) {
				t.Fatal("released before drain")
			}
			beforeUnload := unloads
			failed = false
			if err := owner.Close(); err != nil {
				t.Fatal(err)
			}
			if fault == "restore" && unloads != beforeUnload {
				t.Fatal("unloaded twice")
			}
			beforeFree, beforeUnload := frees, unloads
			r.closing = true
			if err := owner.Close(); err != nil {
				t.Fatal(err)
			}
			if frees != beforeFree || unloads != beforeUnload {
				t.Fatal("released Close touched borrowed foundation")
			}
		})
	}
}

func TestShapeHCMixClosedAdmission(t *testing.T) {
	r := &cudaFoundation{closing: true}
	if _, err := newCUDAShapeHCMixWithOperations(r, dsv4shape.Config{}, nil, nil, shapeHCMixOperations{}); !errors.Is(err, dsv4.ErrClosed) {
		t.Fatalf("constructor error = %v", err)
	}
	s := &cudaShapeHCMix{r: r, module: 1}
	if err := s.Prepare(new(cudaShapeForward), 0, "attn", 1e-6, 1e-6); !errors.Is(err, dsv4.ErrClosed) {
		t.Fatalf("prepare error = %v", err)
	}
}
