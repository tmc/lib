//go:build linux

package main

import (
	"errors"
	"fmt"
	"io"
	"reflect"
	"testing"
	"unsafe"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
	"github.com/tmc/lib/nvidia/cublas"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudagraph"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// This host-only ownership fixture never enters CUDA. Assert the private
// representation before fabricating a slice; injected release callbacks must
// consume it, never Slice.Free. A layout change fails this test closed.
func cleanupTestSlice[T any](t *testing.T, ptr driver.DevicePtr) cudamem.Slice[T] {
	t.Helper()
	var s cudamem.Slice[T]
	typ := reflect.TypeOf(s)
	fields := []struct {
		name   string
		offset uintptr
		typ    reflect.Type
	}{
		{"ptr", 0, reflect.TypeOf(ptr)},
		{"len", 8, reflect.TypeOf(int(0))},
		{"cap", 16, reflect.TypeOf(int(0))},
		{"elemSize", 24, reflect.TypeOf(uintptr(0))},
	}
	if unsafe.Sizeof(uintptr(0)) != 8 || typ.NumField() != 5 || typ.Size() != 40 {
		t.Fatal("unsupported Slice fixture layout")
	}
	for _, want := range fields {
		got, ok := typ.FieldByName(want.name)
		if !ok || got.Offset != want.offset || got.Type != want.typ {
			t.Fatalf("Slice field %s changed: %v", want.name, got)
		}
	}
	alloc, ok := typ.FieldByName("alloc")
	if !ok || alloc.Offset != 32 || alloc.Type.Kind() != reflect.Pointer || alloc.Type.Elem().Name() != "allocation" {
		t.Fatal("Slice allocation field changed")
	}
	v := reflect.ValueOf(&s).Elem()
	set := func(name string, value reflect.Value) {
		f := v.FieldByName(name)
		reflect.NewAt(f.Type(), unsafe.Pointer(f.UnsafeAddr())).Elem().Set(value)
	}
	set("ptr", reflect.ValueOf(ptr))
	set("len", reflect.ValueOf(1))
	set("cap", reflect.ValueOf(1))
	var value T
	set("elemSize", reflect.ValueOf(unsafe.Sizeof(value)))
	return s
}

func TestCUDAArtifactCleanupRetry(t *testing.T) {
	for _, kind := range []string{"moe", "dense", "grouping"} {
		for _, restore := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/restore-%v", kind, restore), func(t *testing.T) {
				cause := errors.New("release fault")
				fail, raw := true, make(map[driver.DevicePtr]int)
				release := func(ptr driver.DevicePtr) error {
					if ptr == 2 && fail {
						if restore && raw[ptr] == 0 {
							raw[ptr]++
						}
						return cause
					}
					if raw[ptr] == 0 {
						raw[ptr]++
					}
					return nil
				}
				bytes := func(s *cudamem.Slice[byte]) error { return release(s.Ptr()) }
				ints := func(s *cudamem.Slice[int32]) error { return release(s.Ptr()) }
				var close func() error
				var retained func() bool
				switch kind {
				case "moe":
					a := &cudaMoEArtifact{gate: cleanupTestSlice[byte](t, 1), up: cleanupTestSlice[byte](t, 2), down: cleanupTestSlice[byte](t, 3)}
					close = func() error { return a.freeWith(bytes) }
					retained = func() bool { return a.up.Ptr() == 2 && a.down.Ptr() == 3 }
				case "dense":
					a := &cudaDenseArtifact{gate: cleanupTestSlice[byte](t, 1), up: cleanupTestSlice[byte](t, 2), down: cleanupTestSlice[byte](t, 3)}
					close = func() error { return a.freeWith(bytes) }
					retained = func() bool { return a.up.Ptr() == 2 && a.down.Ptr() == 3 }
				case "grouping":
					g := &cudaMoEGrouping{offsets: cleanupTestSlice[int32](t, 1), pairs: cleanupTestSlice[int32](t, 2), active: cleanupTestSlice[int32](t, 3)}
					close = func() error { return g.freeWith(ints) }
					retained = func() bool { return g.pairs.Ptr() == 2 && g.active.Ptr() == 3 }
				}
				if err := close(); !errors.Is(err, cause) {
					t.Fatalf("error = %v", err)
				}
				if !retained() || raw[1] != 1 || raw[3] != 0 {
					t.Fatalf("owner/order lost: %v", raw)
				}
				fail = false
				if err := close(); err != nil {
					t.Fatal(err)
				}
				if raw[1] != 1 || raw[2] != 1 || raw[3] != 1 {
					t.Fatalf("raw release counts: %v", raw)
				}
			})
		}
	}
}

func TestCUDAHCCleanupRetry(t *testing.T) {
	for _, stage := range []string{"sync", "destroy", "restore", "arena"} {
		t.Run(stage, func(t *testing.T) {
			cause := errors.New("HC close fault")
			fail, drains, destroys := true, 0, 0
			h := &cudaHC{r: &cudaFoundation{device: new(cuda.Device)}, arena: new(cudamem.Arena), blas: cublas.Handle(1)}
			h.cleanupOps = &cudaHCCleanupOperations{
				synchronize: func(*cudaHC) error {
					drains++
					if fail && stage == "sync" {
						return cause
					}
					return nil
				},
				destroy: func(_ *cudaHC, released func()) error {
					if fail && stage == "destroy" {
						return cause
					}
					destroys++
					released()
					if fail && stage == "restore" {
						return cause
					}
					return nil
				},
				arena: func(*cudamem.Arena) error {
					if fail && stage == "arena" {
						return cause
					}
					return nil
				},
			}
			if err := h.Close(); !errors.Is(err, cause) {
				t.Fatal(err)
			}
			if h.arena == nil || !h.closing {
				t.Fatal("cleanup owner lost")
			}
			if stage == "destroy" && h.blas == 0 {
				t.Fatal("live handle cleared")
			}
			if stage == "restore" && h.blas != 0 {
				t.Fatal("released handle retained")
			}
			fail = false
			if err := h.Close(); err != nil {
				t.Fatal(err)
			}
			if h.arena != nil || h.blas != 0 || destroys != 1 {
				t.Fatalf("retry: handle %v destroys %d", h.blas, destroys)
			}
			wantDrains := 1
			if stage == "sync" {
				wantDrains = 2
			}
			if drains != wantDrains {
				t.Fatalf("drains %d, want %d", drains, wantDrains)
			}
		})
	}
}

func TestCUDARuntimeCleanupRetry(t *testing.T) {
	for _, stage := range []string{"sync", "graph", "hc", "arena", "batch", "foundation"} {
		t.Run(stage, func(t *testing.T) {
			cause, fail := errors.New("runtime close fault"), true
			calls := make(map[string]int)
			call := func(name string) error {
				calls[name]++
				if fail && name == stage {
					return cause
				}
				return nil
			}
			r := &cudaRuntime{cudaFoundation: &cudaFoundation{device: new(cuda.Device), workStream: new(cuda.Stream)}, hcMath: new(cudaHC), arena: new(cudamem.Arena), batch: new(cudaBatchResources), model: new(cudaModel)}
			r.graphs[0].qkv = new(cudagraph.Exec)
			ops := defaultCUDAChildCleanupOperations()
			ops.synchronize = func(*cudaRuntime) error { return call("sync") }
			ops.graph = func(*cudagraph.Exec) error { return call("graph") }
			ops.hc = func(*cudaHC) error { return call("hc") }
			ops.arena = func(*cudamem.Arena) error { return call("arena") }
			ops.batch = func(*cudaBatchResources) error { return call("batch") }
			ops.foundation = func(*cudaFoundation) error { return call("foundation") }
			r.cleanupOps = &ops
			if err := r.Close(); !errors.Is(err, cause) {
				t.Fatal(err)
			}
			if r.cudaFoundation == nil || !errors.Is(r.failed, dsv4.ErrClosed) {
				t.Fatal("runtime ownership/admission lost")
			}
			if _, err := r.newIndependentState(4096); !errors.Is(err, dsv4.ErrClosed) {
				t.Fatalf("new state after Close: %v", err)
			}
			fail = false
			if err := r.Close(); err != nil {
				t.Fatal(err)
			}
			if r.cudaFoundation != nil || calls["foundation"] < 1 {
				t.Fatal("foundation not released")
			}
			if stage != "sync" && calls["sync"] != 1 {
				t.Fatalf("drained twice: %v", calls)
			}
		})
	}
}

func TestCUDABatchCleanupRetry(t *testing.T) {
	for _, stage := range []string{"graph", "grouping", "hc-batch", "hc", "arena"} {
		t.Run(stage, func(t *testing.T) {
			cause, fail := errors.New("batch close fault"), true
			call := func(name string) error {
				if fail && name == stage {
					return cause
				}
				return nil
			}
			b := &cudaBatchResources{grouping: new(cudaMoEGrouping), hcBatch: new(cudaHCBatch), hc: []*cudaHC{new(cudaHC)}, arena: new(cudamem.Arena)}
			b.graphs[0].qkv = new(cudagraph.Exec)
			ops := defaultCUDAChildCleanupOperations()
			ops.graph = func(*cudagraph.Exec) error { return call("graph") }
			ops.grouping = func(*cudaMoEGrouping) error { return call("grouping") }
			ops.hcBatch = func(*cudaHCBatch) error { return call("hc-batch") }
			ops.hc = func(*cudaHC) error { return call("hc") }
			ops.arena = func(*cudamem.Arena) error { return call("arena") }
			b.cleanupOps = &ops
			if err := b.Close(); !errors.Is(err, cause) {
				t.Fatal(err)
			}
			if b.arena == nil || !b.closing {
				t.Fatal("batch owner lost")
			}
			fail = false
			if err := b.Close(); err != nil {
				t.Fatal(err)
			}
			if b.arena != nil || b.grouping != nil || b.hcBatch != nil || b.hc[0] != nil {
				t.Fatal("remaining owner after retry")
			}
		})
	}
}

func TestCUDASnapshotCleanupRetry(t *testing.T) {
	cause, fail, raw := errors.New("snapshot close fault"), true, 0
	s := &cudaStateSnapshot{arena: new(cudamem.Arena)}
	s.closeArena = func(*cudamem.Arena) error {
		if raw == 0 {
			raw++
		}
		if fail {
			return cause
		}
		return nil
	}
	if err := s.Close(); !errors.Is(err, cause) {
		t.Fatal(err)
	}
	if s.closed || !s.closing || s.arena == nil {
		t.Fatal("snapshot owner lost")
	}
	fail = false
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	if !s.closed || s.arena != nil || raw != 1 {
		t.Fatal("snapshot retry failed")
	}
}

func TestCUDASnapshotCloneFailureOwner(t *testing.T) {
	t.Run("failed-admission", func(t *testing.T) {
		cause := errors.New("request fault")
		state := &cudaRequestState{runtime: &cudaRuntime{failed: errors.Join(dsv4.ErrDesynchronized, cause)}}
		ops := cudaSnapshotOperations{
			synchronize: func(*cudaRuntime) error { t.Fatal("synchronized failed request"); return nil },
			arena:       func(*cudaRuntime) (*cudamem.Arena, error) { t.Fatal("allocated for failed request"); return nil, nil },
		}
		got, err := state.snapshotWithOperations(ops)
		if got != nil || !errors.Is(err, dsv4.ErrDesynchronized) || !errors.Is(err, cause) {
			t.Fatalf("failed admission: snapshot=%v error=%v", got, err)
		}
	})
	for _, stage := range []string{"arena", "clone", "arena-error"} {
		t.Run(stage, func(t *testing.T) {
			cause, cleanupCause, fail := errors.New("clone fault"), errors.New("cleanup fault"), true
			a := new(cudamem.Arena)
			fault := func(name string) error {
				if stage == name {
					return cause
				}
				return nil
			}
			ops := cudaSnapshotOperations{
				synchronize: func(*cudaRuntime) error { return nil },
				arena:       func(*cudaRuntime) (*cudamem.Arena, error) { return a, fault("arena") },
				clone: func(*cudamem.Arena, cudaSessionState) (cudaSessionState, error) {
					return cudaSessionState{}, fault("clone")
				},
				arenaError: func(*cudamem.Arena) error { return fault("arena-error") },
				free: func(got *cudamem.Arena) error {
					if got != a {
						t.Fatal("wrong arena")
					}
					if fail {
						return cleanupCause
					}
					return nil
				},
			}
			state := &cudaRequestState{runtime: &cudaRuntime{cudaFoundation: new(cudaFoundation)}}
			_, err := state.snapshotWithOperations(ops)
			var owner io.Closer
			if !errors.Is(err, cause) || !errors.Is(err, cleanupCause) || !errors.As(err, &owner) {
				t.Fatalf("owner lost: %v", err)
			}
			fail = false
			if err := owner.Close(); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestCUDARuntimeConstructorFailureOwner(t *testing.T) {
	for _, stage := range []string{"foundation", "bind", "moe", "dense", "ballast", "allocate", "hc", "reset"} {
		t.Run(stage, func(t *testing.T) {
			cause, cleanupCause, fail := errors.New("construct fault"), errors.New("foundation close fault"), true
			foundation := &cudaFoundation{device: new(cuda.Device), workStream: new(cuda.Stream)}
			fault := func(name string) error {
				if name == stage {
					return cause
				}
				return nil
			}
			cleanup := defaultCUDAChildCleanupOperations()
			cleanup.synchronize = func(*cudaRuntime) error { return nil }
			cleanup.hc = func(*cudaHC) error { return nil }
			cleanup.arena = func(*cudamem.Arena) error { return nil }
			cleanup.foundation = func(got *cudaFoundation) error {
				if got != foundation {
					t.Fatal("wrong foundation")
				}
				if fail {
					return cleanupCause
				}
				return nil
			}
			ops := cudaRuntimeConstructorOperations{
				foundation: func(string, cudaRuntimeOptions) (*cudaFoundation, error) { return foundation, fault("foundation") },
				bind:       func(*cudaFoundation) (*cudaModel, error) { return new(cudaModel), fault("bind") },
				moe:        func(*cudaRuntime, int) error { return fault("moe") },
				dense:      func(*cudaRuntime, int) error { return fault("dense") },
				ballast:    func(*cudaRuntime) error { return fault("ballast") },
				allocate:   func(r *cudaRuntime) error { r.arena = new(cudamem.Arena); return fault("allocate") },
				hc:         func(*cudaFoundation) (*cudaHC, error) { return new(cudaHC), fault("hc") },
				reset:      func(*cudaRuntime) error { return fault("reset") },
				cleanup:    &cleanup,
			}
			_, err := newCUDARuntimeWithOperations("fixture", 4096, cudaRuntimeOptions{}, ops)
			var owner io.Closer
			if !errors.Is(err, cause) || !errors.Is(err, cleanupCause) || !errors.As(err, &owner) {
				t.Fatalf("constructor owner lost: %v", err)
			}
			fail = false
			if err := owner.Close(); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestCUDAMoEStageFailureOwner(t *testing.T) {
	cause, cleanupCause, fail := errors.New("allocation restore fault"), errors.New("stage close fault"), true
	buffer := cleanupTestSlice[byte](t, 42)
	_, err := newCUDAMoEStageWithOperations(new(cuda.Device), func(*cuda.Device, int) (cudamem.Slice[byte], error) { return buffer, cause }, func(s *cudamem.Slice[byte]) error {
		if s.Ptr() != 42 {
			t.Fatal("stage buffer owner lost")
		}
		if fail {
			return cleanupCause
		}
		return nil
	})
	var owner io.Closer
	if !errors.Is(err, cause) || !errors.Is(err, cleanupCause) || !errors.As(err, &owner) {
		t.Fatalf("stage owner missing: %v", err)
	}
	fail = false
	if err := owner.Close(); err != nil {
		t.Fatal(err)
	}
}
