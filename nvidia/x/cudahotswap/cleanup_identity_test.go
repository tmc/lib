package cudahotswap

import (
	"context"
	"errors"
	"io"
	"sync"
	"sync/atomic"
	"testing"
)

type identityCloser struct {
	token any
	calls *atomic.Int32
	fail  *atomic.Bool
}

func (c identityCloser) Close() error {
	c.calls.Add(1)
	if c.fail != nil && c.fail.Load() {
		return errors.New("close still pending")
	}
	return nil
}

func identityCleanup(t *testing.T, owners ...io.Closer) (*requestCleanup, *CleanupIdentityError) {
	t.Helper()
	s := &Service{}
	r := s.retainCleanup(nil)
	for _, owner := range owners {
		r.adopt(owner)
	}
	var identity *CleanupIdentityError
	if err := r.Close(); !errors.As(err, &identity) {
		t.Fatalf("close = %v, want unresolved identity", err)
	}
	return r, identity
}

func TestCleanupIdentityInvalidPartitions(t *testing.T) {
	for _, groups := range [][][]int{nil, {{}}, {{0}}, {{0}, {0, 1}}, {{-1}, {1}}, {{0}, {2}}} {
		t.Run("invalid", func(t *testing.T) {
			calls := new(atomic.Int32)
			owner := identityCloser{token: []byte{1}, calls: calls}
			r, identity := identityCleanup(t, owner, owner)
			if err := identity.ResolveAliases(groups...); err == nil {
				t.Fatal("invalid partition accepted")
			}
			if r.Close() != identity || calls.Load() != 0 {
				t.Fatal("invalid resolution changed retained cleanup")
			}
			if err := identity.ResolveAliases([]int{1, 0}); err != nil {
				t.Fatal(err)
			}
			if err := r.Close(); err != nil || calls.Load() != 1 {
				t.Fatalf("alias cleanup = %v, count %d", err, calls.Load())
			}
		})
	}
	unknown := identityCloser{token: []byte{1}, calls: new(atomic.Int32)}
	a := identityCloser{token: "a", calls: new(atomic.Int32)}
	b := identityCloser{token: "b", calls: new(atomic.Int32)}
	r, identity := identityCleanup(t, unknown, a, b)
	if err := identity.ResolveAliases([]int{0, 1, 2}); err == nil {
		t.Fatal("known distinct comparable owners merged through ambiguous member")
	}
	if err := identity.ResolveAliases([]int{0}, []int{1}, []int{2}); err != nil {
		t.Fatal(err)
	}
	if err := r.Close(); err != nil || unknown.calls.Load() != 1 || a.calls.Load() != 1 || b.calls.Load() != 1 {
		t.Fatal("distinct resolution discarded an owner", err)
	}
}

func TestCleanupIdentityStaleAdoption(t *testing.T) {
	calls := new(atomic.Int32)
	owner := identityCloser{token: []byte{1}, calls: calls}
	r, old := identityCleanup(t, owner, owner)
	r.adopt(owner)
	if err := old.ResolveAliases([]int{0, 1}); err == nil {
		t.Fatal("stale owner snapshot resolved newer obligations")
	}
	var current *CleanupIdentityError
	if err := r.Close(); !errors.As(err, &current) || len(current.Owners()) != 3 {
		t.Fatalf("new identity snapshot = %v", err)
	}
	if err := current.ResolveAliases([]int{0, 1, 2}); err != nil {
		t.Fatal(err)
	}
	if err := r.Close(); err != nil || calls.Load() != 1 {
		t.Fatal("stale adoption cleanup", err)
	}
}

func TestCleanupIdentityConcurrentResolution(t *testing.T) {
	for range 20 {
		calls := new(atomic.Int32)
		owner := identityCloser{token: []byte{1}, calls: calls}
		r, identity := identityCleanup(t, owner, owner)
		var wg sync.WaitGroup
		wg.Add(2)
		var resolveErr, closeErr error
		go func() { defer wg.Done(); resolveErr = identity.ResolveAliases([]int{0, 1}) }()
		go func() { defer wg.Done(); closeErr = r.Close() }()
		wg.Wait()
		if resolveErr != nil || closeErr != nil && closeErr != identity {
			t.Fatalf("concurrent resolution = %v, close = %v", resolveErr, closeErr)
		}
		if err := r.Close(); err != nil || calls.Load() != 1 {
			t.Fatalf("final close = %v, count %d", err, calls.Load())
		}
	}
}

func TestCleanupIdentityRetainsDrainAndRetry(t *testing.T) {
	ownerCalls, leaseCalls := new(atomic.Int32), new(atomic.Int32)
	fail := new(atomic.Bool)
	fail.Store(true)
	owner := identityCloser{token: []byte{1}, calls: ownerCalls, fail: fail}
	r, identity := identityCleanup(t, owner, owner)
	r.lease = identityCloser{calls: leaseCalls}
	drains, collects := 0, 0
	r.drain = func(context.Context) error { drains++; return nil }
	r.collect = func() error { collects++; return nil }
	if r.Close() != identity || drains != 0 || collects != 0 || leaseCalls.Load() != 0 {
		t.Fatal("unresolved identity allowed dependent cleanup")
	}
	if err := identity.ResolveAliases([]int{0, 1}); err != nil {
		t.Fatal(err)
	}
	if err := r.Close(); err == nil || drains != 1 || ownerCalls.Load() != 1 || leaseCalls.Load() != 0 {
		t.Fatal("failed close lost owner/lease", err)
	}
	fail.Store(false)
	if err := r.Close(); err != nil || drains != 1 || ownerCalls.Load() != 2 || collects != 1 || leaseCalls.Load() != 1 {
		t.Fatal("retry repeated drain or lost cleanup", err)
	}
}
