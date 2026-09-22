package cublasltx

import (
	"errors"
	"github.com/tmc/lib/nvidia/cublaslt"
	"testing"
)

func TestCloseEmpty(t *testing.T) {
	for _, e := range []*Engine{nil, {}} {
		if err := e.Close(); err != nil {
			t.Fatal(err)
		}
	}
}

func TestCleanupContextFailure(t *testing.T) {
	failure := errors.New("injected failure")
	for _, stage := range []string{"entry", "destroy", "restore"} {
		t.Run(stage, func(t *testing.T) {
			owner := &Engine{handle: 1}
			calls := 0
			run := func(fn func() error) error {
				if stage == "entry" {
					return failure
				}
				if err := fn(); err != nil {
					return err
				}
				return failure
			}
			destroy := func(handle cublaslt.Handle) error {
				if handle != 1 {
					t.Fatalf("handle = %v", handle)
				}
				calls++
				if stage == "destroy" {
					return failure
				}
				return nil
			}
			err := owner.closeWith(run, destroy, func() error { t.Fatal("unexpected workspace free"); return nil })
			if !errors.Is(err, failure) {
				t.Fatalf("error = %v", err)
			}
			wantLive := stage != "restore"
			if (owner.handle != 0) != wantLive {
				t.Fatalf("incorrect retained handle: %+v", owner)
			}

			wantCalls := 1
			if stage == "entry" {
				wantCalls = 0
			}
			if calls != wantCalls {
				t.Fatalf("calls = %d, want %d", calls, wantCalls)
			}
			if wantLive {
				wantCalls++
			}
			run = func(fn func() error) error { return fn() }
			destroy = func(cublaslt.Handle) error { calls++; return nil }
			for range 2 {
				err = owner.closeWith(run, destroy, func() error { t.Fatal("unexpected workspace free"); return nil })
				if err != nil {
					t.Fatal(err)
				}
			}
			if calls != wantCalls || owner.handle != 0 {
				t.Fatalf("after retries: calls=%d want=%d", calls, wantCalls)
			}

		})
	}
}
