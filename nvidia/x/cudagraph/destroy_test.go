package cudagraph

import (
	"errors"
	"testing"
)

func TestDestroyFailureState(t *testing.T) {
	failure := errors.New("injected failure")
	for _, kind := range []string{"graph", "exec"} {
		for _, stage := range []string{"entry", "destroy", "restore"} {
			t.Run(kind+"/"+stage, func(t *testing.T) {
				g := &Graph{raw: 1}
				e := &Exec{raw: 1, kernelOrder: []string{"kernel"}}
				close := g.destroy
				alive := func() bool { return g.raw != 0 }
				if kind == "exec" {
					close = e.destroy
					alive = func() bool { return e.raw != 0 }
				}
				calls := 0
				destroy := func() error {
					calls++
					if stage == "destroy" {
						return failure
					}
					return nil
				}
				run := func(fn func() error) error {
					if stage == "entry" {
						return failure
					}
					if err := fn(); err != nil {
						return err
					}
					return failure // Restoration fails after successful destruction.
				}
				if err := close(run, destroy); !errors.Is(err, failure) {
					t.Fatalf("destroy error = %v, want injected failure", err)
				}
				wantAlive := stage != "restore"
				if alive() != wantAlive {
					t.Fatalf("handle retained = %v, want %v", alive(), wantAlive)
				}
				if kind == "exec" && (len(e.kernelOrder) != 0) != wantAlive {
					t.Fatal("kernel order lifetime differs from handle lifetime")
				}
				wantCalls := 1
				if stage == "entry" {
					wantCalls = 0
				}
				if calls != wantCalls {
					t.Fatalf("destroy calls = %d, want %d", calls, wantCalls)
				}
				if wantAlive {
					wantCalls++
				}
				for range 2 {
					if err := close(func(fn func() error) error { return fn() }, func() error {
						calls++
						return nil
					}); err != nil {
						t.Fatal(err)
					}
				}
				if alive() || calls != wantCalls {
					t.Fatalf("after retries: retained=%v calls=%d, want false/%d", alive(), calls, wantCalls)
				}
			})
		}
	}
}

func TestDestroyEmpty(t *testing.T) {
	for _, g := range []*Graph{nil, {}} {
		if err := g.Destroy(); err != nil {
			t.Fatal(err)
		}
	}
	for _, e := range []*Exec{nil, {}} {
		if err := e.Destroy(); err != nil {
			t.Fatal(err)
		}
	}
}
