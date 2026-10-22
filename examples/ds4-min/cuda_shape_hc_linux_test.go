//go:build linux

package main

import (
	"errors"
	"fmt"
	"io"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
	"github.com/tmc/lib/examples/ds4-min/dsv4shape"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

func TestShapeHCExtentAdmission(t *testing.T) {
	for _, family := range []dsv4shape.Variant{dsv4shape.Flash, dsv4shape.Pro} {
		c, err := dsv4shape.For(family)
		if err != nil {
			t.Fatal(err)
		}
		for _, phase := range []string{"attn", "ffn"} {
			for _, name := range []string{"cur", "next", phase + "_cur", phase + "_out"} {
				t.Run(fmt.Sprintf("%d/%s/%s", family, phase, name), func(t *testing.T) {
					r := new(cudaFoundation)
					hc := &cudaShapeHC{r: r, config: c, module: 1}
					state := &cudaShapeForward{r: r, plan: shapeForwardPlan{config: c}, scratch: map[string]cudamem.Slice[float32]{}}
					for i, key := range []string{"cur", "next", phase + "_cur", phase + "_out"} {
						n := c.Embed
						if key == "cur" || key == "next" {
							n *= c.HC
						}
						if key == name {
							n--
						}
						state.scratch[key] = cacheTestSlice(t, n, i+1)
					}
					var got error
					if name == "cur" || name == phase+"_cur" {
						got = hc.Collapse(state, phase, cacheTestSlice(t, c.HC, 20))
					} else {
						got = hc.Expand(state, phase, cacheTestSlice(t, c.HC, 20), cacheTestSlice(t, c.HC*c.HC, 21))
					}
					if got == nil {
						t.Fatal("accepted truncated HC operand")
					}
				})
			}
		}
	}
}

func TestShapeHCModuleOwnership(t *testing.T) {
	for _, fault := range []string{"load", "function", "drain", "unload", "restore"} {
		t.Run(fault, func(t *testing.T) {
			c, _ := dsv4shape.For(dsv4shape.Pro)
			r := new(cudaFoundation)
			failed := true
			cause := errors.New("fault")
			unloads := 0
			ops := shapeHCOperations{
				load: func() (driver.Module, driver.Function, driver.Function, error) {
					if failed && fault == "load" {
						return 1, 0, 0, cause
					}
					if failed && fault == "function" {
						return 1, 1, 0, cause
					}
					return 1, 1, 2, nil
				},
				drain: func() error {
					if failed && fault == "drain" {
						return cause
					}
					return nil
				},
				unload: func(driver.Module) error {
					unloads++
					if failed && (fault == "load" || fault == "function" || fault == "unload") {
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
			s, err := newCUDAShapeHCWithOperations(r, c, ops)
			var owner io.Closer
			if fault == "load" || fault == "function" {
				if s != nil || !errors.Is(err, cause) || !errors.As(err, &owner) {
					t.Fatalf("lost module: %v", err)
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
			if fault == "drain" && unloads != 0 {
				t.Fatal("unloaded before drain")
			}
			before := unloads
			failed = false
			if err := owner.Close(); err != nil {
				t.Fatal(err)
			}
			if fault == "restore" && unloads != before {
				t.Fatal("successful unload repeated")
			}
			before = unloads
			if err := owner.Close(); err != nil {
				t.Fatal(err)
			}
			if unloads != before || r.closing {
				t.Fatal("cleanup repeated or foundation closed")
			}
		})
	}
}

func TestShapeHCClosedAdmission(t *testing.T) {
	r := &cudaFoundation{closing: true}
	if _, err := newCUDAShapeHCWithOperations(r, dsv4shape.Config{}, shapeHCOperations{}); !errors.Is(err, dsv4.ErrClosed) {
		t.Fatal(err)
	}
}

func TestShapeHCReleasedClose(t *testing.T) {
	c, _ := dsv4shape.For(dsv4shape.Pro)
	r := new(cudaFoundation)
	calls := 0
	ops := shapeHCOperations{load: func() (driver.Module, driver.Function, driver.Function, error) { return 1, 1, 2, nil }, drain: func() error { calls++; return nil }, closeRun: func(f func() error) error { calls++; return f() }, unload: func(driver.Module) error { calls++; return nil }}
	s, err := newCUDAShapeHCWithOperations(r, c, ops)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	before := calls
	r.closing = true
	s.ops.drain = func() error { t.Fatal("released HC touched foundation"); return dsv4.ErrClosed }
	if err := s.Close(); err != nil || calls != before {
		t.Fatalf("released close: %v calls=%d", err, calls)
	}
}

func TestShapeHCMissingEntry(t *testing.T) {
	c, _ := dsv4shape.For(dsv4shape.Pro)
	r := new(cudaFoundation)
	for _, handles := range [][3]uint64{{0, 1, 2}, {1, 0, 2}, {1, 1, 0}} {
		unloads := 0
		ops := shapeHCOperations{load: func() (driver.Module, driver.Function, driver.Function, error) {
			return driver.Module(handles[0]), driver.Function(handles[1]), driver.Function(handles[2]), nil
		}, drain: func() error { return nil }, closeRun: func(f func() error) error { return f() }, unload: func(driver.Module) error { unloads++; return nil }}
		if s, err := newCUDAShapeHCWithOperations(r, c, ops); s != nil || err == nil {
			t.Fatal("missing entry admitted")
		}
		want := 0
		if handles[0] != 0 {
			want = 1
		}
		if unloads != want {
			t.Fatal("missing entry module leaked")
		}
	}
}
