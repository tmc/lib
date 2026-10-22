//go:build linux

package main

import (
	"errors"
	"io"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

type requestStreamFixture struct {
	parent                                                       *cudaRuntime
	created                                                      *cuda.Stream
	createErr, allocateErr, hcErr, resetErr, syncErr, destroyErr error
	creates, syncs, destroys, rawDestroys                        int
	destroyed                                                    bool
}

func newRequestStreamFixture() *requestStreamFixture {
	f := &requestStreamFixture{created: new(cuda.Stream)}
	f.parent = &cudaRuntime{
		cudaFoundation: &cudaFoundation{device: new(cuda.Device), workStream: new(cuda.Stream)},
		model:          new(cudaModel), arena: new(cudamem.Arena), ctx: 4096,
	}
	f.parent.requestOps = &cudaRequestOperations{
		create:          func(*cuda.Device) (*cuda.Stream, error) { f.creates++; return f.created, f.createErr },
		allocate:        func(r *cudaRuntime) error { r.arena = new(cudamem.Arena); return f.allocateErr },
		hyperconnection: func(*cudaRuntime) error { return f.hcErr },
		reset:           func(*cudaRuntime) error { return f.resetErr },
		synchronize: func(r *cudaRuntime) error {
			if f.destroyed {
				return errors.New("synchronize destroyed stream")
			}
			if r.workStream != nil {
				f.syncs++
			}
			return f.syncErr
		},
		destroy: func(r *cudaRuntime) error {
			f.destroys++
			if r.workStream != f.created {
				tpanic("destroyed backend stream")
			}
			if !f.destroyed && f.destroyErr == nil {
				f.rawDestroys++
				f.destroyed = true
			}
			return f.destroyErr
		},
	}
	return f
}

// This panic is a fixture invariant, never an API failure path.
func tpanic(s string) { panic(s) }

func TestCUDARequestStreamConstruction(t *testing.T) {
	cause := errors.New("constructor fault")
	release := errors.New("release fault")
	for _, stage := range []string{"nil-create", "created-restore", "arena", "hyperconnection", "reset"} {
		for _, retain := range []bool{false, true} {
			t.Run(stage+map[bool]string{false: "/released", true: "/retained"}[retain], func(t *testing.T) {
				f := newRequestStreamFixture()
				switch stage {
				case "nil-create":
					f.created, f.createErr = nil, cause
				case "created-restore":
					f.createErr = cause
				case "arena":
					f.allocateErr = cause
				case "hyperconnection":
					f.hcErr = cause
				case "reset":
					f.resetErr = cause
				}
				if retain {
					f.destroyErr = release
				}
				originalFoundation := f.parent.cudaFoundation
				originalStream := f.parent.workStream
				state, err := f.parent.newIndependentState(4096)
				if state != nil || !errors.Is(err, cause) {
					t.Fatalf("state=%v err=%v", state, err)
				}
				var owner io.Closer
				retained := retain && stage != "nil-create"
				if errors.As(err, &owner) != retained {
					t.Fatalf("cleanup owner=%v, want %v", owner, retained)
				}
				if retained {
					if !errors.Is(err, release) {
						t.Fatal("release cause lost")
					}
					f.destroyErr = nil
					if err := owner.Close(); err != nil {
						t.Fatal(err)
					}
					if err := owner.Close(); err != nil {
						t.Fatal(err)
					}
				}
				want := 1
				if stage == "nil-create" {
					want = 0
				}
				if f.rawDestroys != want {
					t.Fatalf("raw destroys=%d, want %d", f.rawDestroys, want)
				}
				if f.parent.cudaFoundation != originalFoundation || f.parent.workStream != originalStream || f.parent.device == nil {
					t.Fatal("immutable backend ownership changed")
				}
			})
		}
	}
}

func TestCUDARequestStreamCloseRetry(t *testing.T) {
	for _, stage := range []string{"sync", "destroy", "destroy-restore"} {
		t.Run(stage, func(t *testing.T) {
			f := newRequestStreamFixture()
			r, err := f.parent.newIndependentState(4096)
			if err != nil {
				t.Fatal(err)
			}
			cause := errors.New(stage)
			if stage == "sync" {
				f.syncErr = cause
			} else {
				f.destroyErr = cause
			}
			if stage == "destroy-restore" {
				original := f.parent.requestOps.destroy
				f.parent.requestOps.destroy = func(r *cudaRuntime) error {
					if !f.destroyed {
						f.destroyed, f.rawDestroys = true, 1
					}
					return original(r)
				}
			}
			state := &cudaRequestState{runtime: r}
			if err := state.Close(); !errors.Is(err, cause) {
				t.Fatalf("close=%v", err)
			}
			if state.closed || state.runtime != r || r.cudaFoundation == nil {
				t.Fatal("failed close lost ownership")
			}
			if !errors.Is(r.failed, dsv4.ErrClosed) || !errors.Is(r.failed, dsv4.ErrDesynchronized) {
				t.Fatal("failed close did not poison execution")
			}
			if state.logits() != nil {
				t.Fatal("closing state exposed logits")
			}
			if err := state.Reset(); !errors.Is(err, dsv4.ErrClosed) {
				t.Fatalf("closing reset=%v", err)
			}
			if err := state.eval(0, 0); !errors.Is(err, dsv4.ErrClosed) {
				t.Fatalf("closing eval=%v", err)
			}
			f.syncErr, f.destroyErr = nil, nil
			if err := state.Close(); err != nil {
				t.Fatal(err)
			}
			if err := state.Close(); err != nil {
				t.Fatal(err)
			}
			if !state.closed || state.runtime != nil || f.rawDestroys != 1 {
				t.Fatalf("closed=%v runtime=%v raw destroys=%d", state.closed, state.runtime, f.rawDestroys)
			}
			if f.parent.device == nil || f.parent.workStream == f.created {
				t.Fatal("backend invalidated")
			}
		})
	}
}

func TestCUDAStateAdapterCloseRetry(t *testing.T) {
	f := newRequestStreamFixture()
	r, err := f.parent.newIndependentState(4096)
	if err != nil {
		t.Fatal(err)
	}
	a := &cudaStateAdapter{runtime: r}
	f.syncErr = errors.New("sync")
	if err := a.Close(); err == nil || a.closed {
		t.Fatalf("err=%v closed=%v", err, a.closed)
	}
	f.syncErr = nil
	if err := a.Close(); err != nil {
		t.Fatal(err)
	}
	if err := a.Close(); err != nil {
		t.Fatal(err)
	}
	if !a.closed || f.rawDestroys != 1 {
		t.Fatalf("closed=%v raw destroys=%d", a.closed, f.rawDestroys)
	}
}
