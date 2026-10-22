//go:build linux

package main

import (
	"errors"
	"io"
	"math"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
	"github.com/tmc/lib/examples/ds4-min/dsv4shape"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

type shapeConsumerPartialLoadOwner struct {
	blocked bool
	closes  int
}

func (e *shapeConsumerPartialLoadOwner) Error() string { return "partial module load" }
func (e *shapeConsumerPartialLoadOwner) Close() error {
	e.closes++
	if e.blocked {
		return errors.New("partial module cleanup blocked")
	}
	return nil
}

func TestShapeConsumerPartialErrorOwner(t *testing.T) {
	c, _ := dsv4shape.For(dsv4shape.Pro)
	partial := &shapeConsumerPartialLoadOwner{blocked: true}
	allocs, frees := 0, 0
	ops := shapeConsumerOperations{
		available: func() (uint64, error) { return math.MaxUint64, nil },
		alloc: func(n int) (cudamem.Slice[byte], error) {
			allocs++
			return cudamem.Reinterpret[byte](cacheTestSlice(t, n/4, allocs))
		},
		load:  func(string, string) (driver.Module, driver.Function, error) { return 0, 0, partial },
		drain: func() error { return nil },
		free:  func(v *cudamem.Slice[byte]) error { frees++; *v = cudamem.Slice[byte]{}; return nil },
	}
	s, err := newCUDAShapeConsumersWithOperations(new(cudaFoundation), c, 8, 0, ops)
	if s != nil || err == nil {
		t.Fatal("partial load admitted")
	}
	var owner io.Closer
	if !errors.As(err, &owner) {
		t.Fatal("partial error owner lost")
	}
	if frees != 0 || partial.closes != 0 {
		t.Fatal("constructor released dependent buffers")
	}
	if err := owner.Close(); err == nil || frees != 0 || partial.closes != 1 {
		t.Fatalf("blocked child released dependencies: %v", err)
	}
	partial.blocked = false
	if err := owner.Close(); err != nil {
		t.Fatal(err)
	}
	if frees != allocs || partial.closes != 2 {
		t.Fatalf("retry: frees %d/%d closes %d", frees, allocs, partial.closes)
	}
	if err := owner.Close(); err != nil || frees != allocs || partial.closes != 2 {
		t.Fatal("consumed owner reused")
	}
}

func TestShapeConsumerConsumedFreeError(t *testing.T) {
	v, err := cudamem.Reinterpret[byte](cacheTestSlice(t, 1, 1))
	if err != nil {
		t.Fatal(err)
	}
	cause := errors.New("restore fault")
	frees := 0
	s := &cudaShapeConsumers{buffers: map[string]cudamem.Slice[byte]{"one": v}, ops: shapeConsumerOperations{
		drain: func() error { return nil },
		free: func(v *cudamem.Slice[byte]) error {
			frees++
			*v = cudamem.Slice[byte]{}
			return cause
		},
	}}
	if err := s.Close(); !errors.Is(err, cause) {
		t.Fatal(err)
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	if frees != 1 {
		t.Fatalf("free count=%d want=1", frees)
	}
}

func TestShapeConsumerOwnership(t *testing.T) {
	for _, fault := range []string{"none", "allocation", "module", "drain", "free", "unload", "restore"} {
		t.Run(fault, func(t *testing.T) {
			c, _ := dsv4shape.For(dsv4shape.Pro)
			r := new(cudaFoundation)
			cause := errors.New("fault")
			failed, drained := true, false
			allocs, frees, loads, unloads, drains := 0, 0, 0, 0, 0
			ops := shapeConsumerOperations{
				available: func() (uint64, error) { return math.MaxUint64, nil },
				alloc: func(n int) (cudamem.Slice[byte], error) {
					allocs++
					v, err := cudamem.Reinterpret[byte](cacheTestSlice(t, n/4, allocs))
					if err != nil {
						t.Fatal(err)
					}
					if fault == "allocation" && failed {
						return v, cause
					}
					return v, nil
				},
				load: func(source, entry string) (driver.Module, driver.Function, error) {
					loads++
					if source == "" || entry == "" {
						t.Fatal("missing source or entry")
					}
					if fault == "module" && failed {
						return driver.Module(loads), 0, cause
					}
					return driver.Module(loads), driver.Function(loads), nil
				},
				drain: func() error {
					drains++
					if r.closing {
						t.Fatal("borrowed foundation used after release")
					}
					if fault == "drain" && failed {
						return cause
					}
					drained = true
					return nil
				},
				free: func(v *cudamem.Slice[byte]) error {
					if !drained {
						t.Fatal("free before drain")
					}
					if fault == "free" && failed {
						return cause
					}
					frees++
					return nil
				},
				closeRun: func(fn func() error) error {
					if err := fn(); err != nil {
						return err
					}
					if fault == "restore" && failed {
						return cause
					}
					return nil
				},
				unload: func(m driver.Module) error {
					if !drained || m == 0 {
						t.Fatal("unload before drain or zero handle")
					}
					if fault == "unload" && failed {
						return cause
					}
					unloads++
					return nil
				},
			}
			s, err := newCUDAShapeConsumersWithOperations(r, c, 32768, 0, ops)
			if fault == "allocation" || fault == "module" {
				if s != nil || !errors.Is(err, cause) {
					t.Fatalf("constructor=%v error=%v", s, err)
				}
				if frees != allocs || unloads != loads {
					t.Fatalf("CONSUMER_OWNER_LOST alloc=%d free=%d load=%d unload=%d", allocs, frees, loads, unloads)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			err = s.Close()
			if fault == "none" {
				if err != nil {
					t.Fatal(err)
				}
			} else if !errors.Is(err, cause) {
				t.Fatalf("close error=%v", err)
			}
			if fault == "drain" && (frees != 0 || unloads != 0) {
				t.Fatal("CONSUMER_RELEASE_BEFORE_DRAIN")
			}
			failed = false
			if err := s.Close(); err != nil {
				t.Fatal(err)
			}
			if frees != allocs || unloads != loads {
				t.Fatalf("CONSUMER_RETRY_OWNER alloc=%d free=%d load=%d unload=%d", allocs, frees, loads, unloads)
			}
			r.closing = true
			before := drains
			if err := s.Close(); err != nil || drains != before {
				t.Fatalf("CONSUMER_RELEASED_CLOSE error=%v drains=%d before=%d", err, drains, before)
			}
		})
	}
}

func TestShapeConsumerRetainedAllocation(t *testing.T) {
	c, _ := dsv4shape.For(dsv4shape.Flash)
	cause := errors.New("fault")
	failed := true
	ops := shapeConsumerOperations{
		available: func() (uint64, error) { return math.MaxUint64, nil },
		alloc: func(n int) (cudamem.Slice[byte], error) {
			v, err := cudamem.Reinterpret[byte](cacheTestSlice(t, n/4, 1))
			if err != nil {
				t.Fatal(err)
			}
			return v, cause
		},
		drain: func() error { return nil }, free: func(v *cudamem.Slice[byte]) error {
			if failed {
				return cause
			}
			return nil
		},
	}
	s, err := newCUDAShapeConsumersWithOperations(new(cudaFoundation), c, 32768, 0, ops)
	var owner io.Closer
	if s != nil || !errors.Is(err, cause) || !errors.As(err, &owner) {
		t.Fatalf("CONSUMER_RETAINED_OWNER state=%v error=%v", s, err)
	}
	failed = false
	if err := owner.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestShapeConsumerClosedFoundation(t *testing.T) {
	c, _ := dsv4shape.For(dsv4shape.Pro)
	_, err := newCUDAShapeConsumersWithOperations(&cudaFoundation{closing: true}, c, 32768, 0, shapeConsumerOperations{})
	if !errors.Is(err, dsv4.ErrClosed) {
		t.Fatalf("CONSUMER_CLOSED_ADMISSION error=%v", err)
	}
}
