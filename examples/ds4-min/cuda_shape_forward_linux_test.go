//go:build linux

package main

import (
	"errors"
	"io"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
	"github.com/tmc/lib/examples/ds4-min/dsv4shape"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

func TestShapeForwardOwnership(t *testing.T) {
	for _, fault := range []string{"allocation", "module", "function", "drain", "free", "unload"} {
		t.Run(fault, func(t *testing.T) {
			c, _ := dsv4shape.For(dsv4shape.Pro)
			b := forwardTestBinding(t, c)
			r := &cudaFoundation{device: new(cuda.Device)}
			weights := map[string]cudamem.Slice[float32]{}
			p, err := planShapeForward(c, b, 1)
			if err != nil {
				t.Fatal(err)
			}
			for name := range p.norms {
				weights[name] = cacheTestSlice(t, c.Embed, 2000)
			}
			cause := errors.New("fault")
			failed := true
			allocs, frees, unloads := 0, 0, 0
			ops := shapeForwardOperations{
				cache: func(uint64) (*cudaShapeCache, error) { return nil, nil },
				alloc: func(n int) (cudamem.Slice[float32], error) {
					allocs++
					v := cacheTestSlice(t, n, allocs)
					if failed && fault == "allocation" {
						return v, cause
					}
					return v, nil
				},
				load: func() (driver.Module, driver.Function, error) {
					if failed && (fault == "module" || fault == "function") {
						return 1, 0, cause
					}
					return 1, 1, nil
				},
				drain: func() error {
					if failed && fault == "drain" {
						return cause
					}
					return nil
				},
				free: func(v *cudamem.Slice[float32]) error {
					frees++
					if failed && (fault == "allocation" || fault == "module" || fault == "function" || fault == "free") {
						return cause
					}
					*v = cudamem.Slice[float32]{}
					return nil
				},
				closeRun: func(f func() error) error { return f() }, unload: func(driver.Module) error {
					unloads++
					if failed && fault == "unload" {
						return cause
					}
					return nil
				},
			}
			s, err := newCUDAShapeForwardWithOperations(r, c, b, weights, 1, 0, ops)
			var owner io.Closer
			if fault == "allocation" || fault == "module" || fault == "function" {
				if s != nil || !errors.Is(err, cause) || !errors.As(err, &owner) {
					t.Fatalf("lost owner: %v", err)
				}
			} else {
				if err != nil {
					t.Fatal(err)
				}
				owner = s
				if !errors.Is(s.Close(), cause) {
					t.Fatal("fault did not fire")
				}
				if fault == "drain" && (frees != 0 || unloads != 0) {
					t.Fatal("released before drain")
				}
			}
			failed = false
			if err := owner.Close(); err != nil {
				t.Fatal(err)
			}
			nf, nu := frees, unloads
			if err := owner.Close(); err != nil {
				t.Fatal(err)
			}
			if frees != nf || unloads != nu {
				t.Fatal("cleanup not idempotent")
			}
			if r.closing {
				t.Fatal("borrowed foundation closed")
			}
		})
	}
}

func TestShapeForwardClosedAdmission(t *testing.T) {
	r := &cudaFoundation{closing: true}
	if _, err := newCUDAShapeForwardWithOperations(r, dsv4shape.Config{}, nil, nil, 1, 0, shapeForwardOperations{}); !errors.Is(err, dsv4.ErrClosed) {
		t.Fatal(err)
	}
}

func TestShapeForwardCacheOwnership(t *testing.T) {
	for _, fault := range []string{"partial", "cache-close", "drain"} {
		t.Run(fault, func(t *testing.T) {
			c, _ := dsv4shape.For(dsv4shape.Pro)
			b := forwardTestBinding(t, c)
			r := &cudaFoundation{device: new(cuda.Device)}
			p, err := planShapeForward(c, b, 1)
			if err != nil {
				t.Fatal(err)
			}
			weights := make(map[string]cudamem.Slice[float32])
			for name := range p.norms {
				weights[name] = cacheTestSlice(t, c.Embed, 2000)
			}
			cause := errors.New("cache fault")
			failed := true
			drained, cacheDrained, cacheFrees, scratchFrees := false, false, 0, 0
			cache := &cudaShapeCache{r: r, buffers: []cudamem.Slice[float32]{cacheTestSlice(t, 512, 1000)}, ops: shapeCacheOperations{
				drain: func() error {
					if !drained {
						t.Fatal("cache drain before state drain")
					}
					cacheDrained = true
					return nil
				},
				free: func(v *cudamem.Slice[float32]) error {
					if !drained || !cacheDrained {
						t.Fatal("cache free before drain")
					}
					cacheFrees++
					if failed && (fault == "partial" || fault == "cache-close") {
						return cause
					}
					*v = cudamem.Slice[float32]{}
					return nil
				},
			}}
			ops := shapeForwardOperations{
				cache: func(uint64) (*cudaShapeCache, error) {
					if fault == "partial" {
						return cache, cause
					}
					return cache, nil
				},
				alloc: func(n int) (cudamem.Slice[float32], error) { return cacheTestSlice(t, n, 10), nil },
				load:  func() (driver.Module, driver.Function, error) { return 1, 1, nil },
				drain: func() error {
					if failed && fault == "drain" {
						return cause
					}
					drained = true
					return nil
				},
				free: func(v *cudamem.Slice[float32]) error {
					if !drained {
						t.Fatal("scratch free before drain")
					}
					if v.Ptr() == weights["blk.0.attn_norm.weight"].Ptr() {
						t.Fatal("borrowed norm freed")
					}
					scratchFrees++
					*v = cudamem.Slice[float32]{}
					return nil
				},
				closeRun: func(f func() error) error { return f() }, unload: func(driver.Module) error { return nil },
			}
			s, err := newCUDAShapeForwardWithOperations(r, c, b, weights, 1, 0, ops)
			var owner io.Closer
			if fault == "partial" {
				if s != nil || !errors.Is(err, cause) || !errors.As(err, &owner) {
					t.Fatalf("partial cache owner lost: %v", err)
				}
			} else {
				if err != nil {
					t.Fatal(err)
				}
				owner = s
				if !errors.Is(s.Close(), cause) {
					t.Fatal("cache close fault absent")
				}
			}
			if fault == "drain" && (cacheFrees != 0 || scratchFrees != 0) {
				t.Fatal("freed with failed state drain")
			}
			failed = false
			if err := owner.Close(); err != nil {
				t.Fatal(err)
			}
			want := 2
			if fault == "drain" {
				want = 1
			}
			if cacheFrees != want {
				t.Fatalf("cache free attempts=%d want=%d", cacheFrees, want)
			}
			n := cacheFrees
			if err := owner.Close(); err != nil {
				t.Fatal(err)
			}
			if cacheFrees != n {
				t.Fatal("cache freed twice")
			}
			if cache.buffers[0].Ptr() != 0 || r.closing {
				t.Fatal("cache retained or borrowed foundation closed")
			}
			for _, w := range weights {
				if w.Ptr() == 0 {
					t.Fatal("borrowed norm changed")
				}
			}
		})
	}
}

func TestShapeForwardReleasedClose(t *testing.T) {
	r := new(cudaFoundation)
	calls := 0
	s := &cudaShapeForward{r: r, scratch: map[string]cudamem.Slice[float32]{"one": cacheTestSlice(t, 1, 1)}, module: 1, ops: shapeForwardOperations{drain: func() error { calls++; return nil }, free: func(v *cudamem.Slice[float32]) error { calls++; *v = cudamem.Slice[float32]{}; return nil }, closeRun: func(f func() error) error { calls++; return f() }, unload: func(driver.Module) error { calls++; return nil }}}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	before := calls
	r.closing = true
	s.ops.drain = func() error { t.Fatal("released forward touched foundation"); return dsv4.ErrClosed }
	if err := s.Close(); err != nil || calls != before {
		t.Fatalf("released close: %v calls=%d", err, calls)
	}
}

func TestShapeForwardMissingEntry(t *testing.T) {
	c, _ := dsv4shape.For(dsv4shape.Pro)
	b := forwardTestBinding(t, c)
	p, err := planShapeForward(c, b, 1)
	if err != nil {
		t.Fatal(err)
	}
	r := new(cudaFoundation)
	weights := map[string]cudamem.Slice[float32]{}
	for name := range p.norms {
		weights[name] = cacheTestSlice(t, c.Embed, 2000)
	}
	for _, handles := range [][2]uint64{{0, 1}, {1, 0}} {
		unloads, frees := 0, 0
		ops := shapeForwardOperations{cache: func(uint64) (*cudaShapeCache, error) { return nil, nil }, alloc: func(n int) (cudamem.Slice[float32], error) { return cacheTestSlice(t, n, 1), nil }, load: func() (driver.Module, driver.Function, error) {
			return driver.Module(handles[0]), driver.Function(handles[1]), nil
		}, drain: func() error { return nil }, free: func(v *cudamem.Slice[float32]) error { frees++; *v = cudamem.Slice[float32]{}; return nil }, closeRun: func(f func() error) error { return f() }, unload: func(driver.Module) error { unloads++; return nil }}
		if s, err := newCUDAShapeForwardWithOperations(r, c, b, weights, 1, 0, ops); s != nil || err == nil {
			t.Fatal("missing forward entry admitted")
		}
		want := 0
		if handles[0] != 0 {
			want = 1
		}
		if unloads != want || frees != len(p.buffers) {
			t.Fatal("missing forward entry leaked resources")
		}
	}
}
