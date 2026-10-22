//go:build linux

package main

import (
	"bytes"
	"errors"
	"reflect"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

func observerHistoryState(t *testing.T) *cudaRequestState {
	t.Helper()
	s := &cudaRequestState{runtime: &cudaRuntime{ctx: 4096, pos: 2}}
	done, err := s.runtime.headObserver.begin(0, []int{3, 4})
	if err != nil {
		t.Fatal(err)
	}
	done(true)
	return s
}

func requireObserverHistory(t *testing.T, s *cudaRequestState, want []int) {
	t.Helper()
	if !reflect.DeepEqual(s.runtime.headObserver.committed, want) || s.runtime.headObserver.pending != nil {
		t.Fatalf("history = %v pending=%v, want %v", s.runtime.headObserver.committed, s.runtime.headObserver.pending, want)
	}
}

func TestOutputHeadRejectedStateTransitionsPreserveHistory(t *testing.T) {
	for _, test := range []struct {
		name   string
		reject func(*cudaRequestState) error
	}{
		{"load-nil", func(s *cudaRequestState) error { return s.Load(nil) }},
		{"load-malformed", func(s *cudaRequestState) error { return s.Load(bytes.NewReader([]byte{1})) }},
		{"restore-nil", func(s *cudaRequestState) error { return s.Restore(nil) }},
	} {
		t.Run(test.name, func(t *testing.T) {
			s := observerHistoryState(t)
			if err := test.reject(s); err == nil {
				t.Fatal("invalid state transition admitted")
			}
			requireObserverHistory(t, s, []int{3, 4})
		})
	}
}

func TestOutputHeadEvalEpochPreservesHistory(t *testing.T) {
	s := observerHistoryState(t)
	if err := s.completeSpeculativeTransition(nil); err != nil {
		t.Fatal(err)
	}
	requireObserverHistory(t, s, []int{3, 4})
	done, err := s.runtime.headObserver.begin(2, []int{5})
	if err != nil {
		t.Fatal(err)
	}
	done(true)
	if err := s.completeSpeculativeTransition(nil); err != nil {
		t.Fatal(err)
	}
	requireObserverHistory(t, s, []int{3, 4, 5})
}

func TestOutputHeadStateRollbackPolicy(t *testing.T) {
	for _, rollbackFails := range []bool{false, true} {
		s := observerHistoryState(t)
		primary := errors.New("copy failed")
		err := cudaStateTransaction(func() error { return primary }, func() error {
			if rollbackFails {
				return errors.New("rollback failed")
			}
			return nil
		}, func(err error) { s.runtime.headObserver.invalidate(); s.runtime.failed = err })
		if !errors.Is(err, primary) {
			t.Fatal(err)
		}
		if rollbackFails {
			requireObserverHistory(t, s, nil)
			if !errors.Is(s.runtime.failed, dsv4.ErrDesynchronized) {
				t.Fatal("failed rollback did not poison state")
			}
		} else {
			requireObserverHistory(t, s, []int{3, 4})
			if s.runtime.failed != nil {
				t.Fatal("successful rollback poisoned state")
			}
		}
	}
}

func TestOutputHeadCacheReplacement(t *testing.T) {
	for _, err := range []error{nil, errors.New("rejected before replacement")} {
		s := observerHistoryState(t)
		got := s.completeObservedCacheReplacement(err)
		if !errors.Is(got, err) {
			t.Fatal(got)
		}
		if err == nil {
			requireObserverHistory(t, s, nil)
		} else {
			requireObserverHistory(t, s, []int{3, 4})
		}
	}
}

func TestOutputHeadCacheReplacementEpochExhaustion(t *testing.T) {
	s := observerHistoryState(t)
	s.lifecycleEpoch = ^uint64(0)
	if err := s.completeObservedCacheReplacement(nil); !errors.Is(err, dsv4.ErrDesynchronized) {
		t.Fatal(err)
	}
	requireObserverHistory(t, s, nil)
}

func TestOutputHeadReplacementInvalidationMutants(t *testing.T) {
	for _, test := range []struct {
		name     string
		complete func(*cudaRequestState) error
		want     bool
	}{
		{"production", func(s *cudaRequestState) error { return s.completeObservedCacheReplacement(nil) }, true},
		{"deleted-invalidation", func(s *cudaRequestState) error { return s.advanceSpeculativeEpoch() }, false},
		{"late-invalidation", func(s *cudaRequestState) error {
			if err := s.advanceSpeculativeEpoch(); err != nil {
				return err
			}
			s.runtime.headObserver.invalidate()
			return nil
		}, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			s := observerHistoryState(t)
			s.lifecycleEpoch = ^uint64(0)
			err := test.complete(s)
			passed := errors.Is(err, dsv4.ErrDesynchronized) && len(s.runtime.headObserver.committed) == 0 && s.runtime.headObserver.pending == nil
			if passed != test.want {
				t.Fatalf("epoch exhaustion invalidation gate=%v, want %v", passed, test.want)
			}
		})
	}
}
