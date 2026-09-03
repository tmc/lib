// SPDX-License-Identifier: Apache-2.0

package kvcache

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
)

func testSpec(start int) Spec {
	return Spec{
		Owner:  "request-1",
		Tokens: Range{Start: start, End: start + 16},
		Format: Format{Layer: 3, DType: "float16", Layout: "bhsd", KVHeads: 8, HeadDim: 64},
	}
}

func testResidency(tier string) Residency {
	return Residency{Backend: "test", Tier: tier, Domain: "device-0"}
}

func TestStoreLeaseLifecycle(t *testing.T) {
	s, err := NewStore(1)
	if err != nil {
		t.Fatal(err)
	}
	owner, err := s.Allocate(testSpec(0), testResidency("device"))
	if err != nil {
		t.Fatal(err)
	}
	id, generation := owner.ID()
	if id == 0 || generation == 0 {
		t.Fatalf("identity = %d/%d, want nonzero", id, generation)
	}
	shared, err := s.Share(owner)
	if err != nil {
		t.Fatal(err)
	}
	desc, err := s.Describe(shared)
	if err != nil {
		t.Fatal(err)
	}
	if desc.References != 2 || desc.Spec != testSpec(0) {
		t.Fatalf("descriptor = %+v", desc)
	}
	if err := s.Evict(context.Background(), owner, func(context.Context, Descriptor) error { return nil }); !errors.Is(err, ErrBusy) {
		t.Fatalf("evict shared block: %v, want ErrBusy", err)
	}
	if err := shared.Release(); err != nil {
		t.Fatal(err)
	}
	if err := shared.Release(); !errors.Is(err, ErrStale) {
		t.Fatalf("release twice: %v, want ErrStale", err)
	}
	if err := s.Evict(context.Background(), owner, func(context.Context, Descriptor) error { return nil }); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Acquire(id, generation); !errors.Is(err, ErrStale) {
		t.Fatalf("acquire evicted generation: %v, want ErrStale", err)
	}
	if _, err := s.Describe(owner); !errors.Is(err, ErrStale) {
		t.Fatalf("describe evicted lease: %v, want ErrStale", err)
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Allocate(testSpec(16), testResidency("device")); !errors.Is(err, ErrClosed) {
		t.Fatalf("allocate after close: %v, want ErrClosed", err)
	}
}

func TestStoreRejectsZeroAndStaleLeases(t *testing.T) {
	s, _ := NewStore(0)
	if _, err := s.Share(Lease{}); !errors.Is(err, ErrStale) {
		t.Fatalf("share zero lease: %v, want ErrStale", err)
	}
	if _, err := s.Acquire(0, 0); !errors.Is(err, ErrInvalid) {
		t.Fatalf("acquire zero identity: %v, want ErrInvalid", err)
	}
	l, err := s.Allocate(testSpec(0), testResidency("device"))
	if err != nil {
		t.Fatal(err)
	}
	id, generation := l.ID()
	if _, err := s.Acquire(id, generation+1); !errors.Is(err, ErrStale) {
		t.Fatalf("acquire wrong generation: %v, want ErrStale", err)
	}
	if err := s.Close(); !errors.Is(err, ErrBusy) {
		t.Fatalf("close live store: %v, want ErrBusy", err)
	}
}

func TestStoreExhaustionIsTransactional(t *testing.T) {
	s, _ := NewStore(1)
	first, err := s.Allocate(testSpec(0), testResidency("device"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.Allocate(testSpec(16), testResidency("device")); !errors.Is(err, ErrExhausted) {
		t.Fatalf("second allocate: %v, want ErrExhausted", err)
	}
	if err := s.Evict(context.Background(), first, func(context.Context, Descriptor) error { return nil }); err != nil {
		t.Fatal(err)
	}
	second, err := s.Allocate(testSpec(16), testResidency("device"))
	if err != nil {
		t.Fatal(err)
	}
	firstID, firstGeneration := first.ID()
	secondID, secondGeneration := second.ID()
	if firstID != secondID || firstGeneration == secondGeneration {
		t.Fatalf("slot generation = first %d/%d, second %d/%d; want reused ID with new generation", firstID, firstGeneration, secondID, secondGeneration)
	}
	if _, err := s.Acquire(firstID, firstGeneration); !errors.Is(err, ErrStale) {
		t.Fatalf("acquire retired generation: %v, want ErrStale", err)
	}
}

func TestStoreMigrationCommitRollbackAndCancel(t *testing.T) {
	tests := []struct {
		name    string
		move    func(context.Context, Descriptor, Residency) error
		wantErr error
		want    string
	}{
		{name: "commit", move: func(context.Context, Descriptor, Residency) error { return nil }, want: "host"},
		{name: "rollback", move: func(context.Context, Descriptor, Residency) error { return errors.New("partial copy") }, wantErr: errors.New("partial copy"), want: "device"},
		{name: "cancel", move: func(ctx context.Context, _ Descriptor, _ Residency) error { return ctx.Err() }, wantErr: context.Canceled, want: "device"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s, _ := NewStore(0)
			l, _ := s.Allocate(testSpec(0), testResidency("device"))
			ctx := context.Background()
			if tt.name == "cancel" {
				var cancel context.CancelFunc
				ctx, cancel = context.WithCancel(ctx)
				cancel()
			}
			err := s.Migrate(ctx, l, testResidency("host"), tt.move)
			if tt.wantErr == nil && err != nil {
				t.Fatal(err)
			}
			if tt.wantErr != nil && (err == nil || err.Error() != tt.wantErr.Error()) {
				t.Fatalf("migrate error = %v, want %v", err, tt.wantErr)
			}
			desc, err := s.Describe(l)
			if err != nil {
				t.Fatal(err)
			}
			if desc.Residency.Tier != tt.want {
				t.Fatalf("tier = %q, want %q", desc.Residency.Tier, tt.want)
			}
		})
	}
}

func TestStoreMigrationCancellationAtCommitPoint(t *testing.T) {
	s, _ := NewStore(0)
	l, _ := s.Allocate(testSpec(0), testResidency("device"))
	ctx, cancel := context.WithCancel(context.Background())
	err := s.Migrate(ctx, l, testResidency("host"), func(context.Context, Descriptor, Residency) error {
		cancel()
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	desc, err := s.Describe(l)
	if err != nil {
		t.Fatal(err)
	}
	if desc.Residency.Tier != "host" {
		t.Fatalf("tier = %q, want committed host", desc.Residency.Tier)
	}
}

func TestStoreMigrationRequiresSoleLease(t *testing.T) {
	s, _ := NewStore(0)
	owner, _ := s.Allocate(testSpec(0), testResidency("device"))
	shared, err := s.Share(owner)
	if err != nil {
		t.Fatal(err)
	}
	called := false
	err = s.Migrate(context.Background(), owner, testResidency("host"), func(context.Context, Descriptor, Residency) error {
		called = true
		return nil
	})
	if !errors.Is(err, ErrBusy) {
		t.Fatalf("migrate shared block: %v, want ErrBusy", err)
	}
	if called {
		t.Fatal("migration callback called for shared block")
	}
	if err := shared.Release(); err != nil {
		t.Fatal(err)
	}
}

func TestStoreEvictionRollback(t *testing.T) {
	s, _ := NewStore(0)
	l, _ := s.Allocate(testSpec(0), testResidency("device"))
	want := errors.New("discard failed")
	if err := s.Evict(context.Background(), l, func(context.Context, Descriptor) error { return want }); !errors.Is(err, want) {
		t.Fatalf("evict error = %v, want %v", err, want)
	}
	if _, err := s.Describe(l); err != nil {
		t.Fatalf("describe after failed eviction: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	called := false
	if err := s.Evict(ctx, l, func(context.Context, Descriptor) error { called = true; return nil }); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled eviction = %v, want context.Canceled", err)
	}
	if called {
		t.Fatal("canceled eviction called discard")
	}
	if _, err := s.Describe(l); err != nil {
		t.Fatalf("describe after canceled eviction: %v", err)
	}
}

func TestStoreMigrationExcludesLifecycleChanges(t *testing.T) {
	s, _ := NewStore(0)
	l, _ := s.Allocate(testSpec(0), testResidency("device"))
	entered := make(chan struct{})
	proceed := make(chan struct{})
	done := make(chan error, 1)
	go func() {
		done <- s.Migrate(context.Background(), l, testResidency("host"), func(context.Context, Descriptor, Residency) error {
			close(entered)
			<-proceed
			return nil
		})
	}()
	<-entered
	id, generation := l.ID()
	checks := []struct {
		name string
		err  error
	}{
		{"acquire", func() error { _, err := s.Acquire(id, generation); return err }()},
		{"share", func() error { _, err := s.Share(l); return err }()},
		{"release", l.Release()},
		{"evict", s.Evict(context.Background(), l, func(context.Context, Descriptor) error { return nil })},
	}
	for _, check := range checks {
		if !errors.Is(check.err, ErrBusy) {
			t.Errorf("%s during migration: %v, want ErrBusy", check.name, check.err)
		}
	}
	close(proceed)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if err := l.Release(); !errors.Is(err, ErrBusy) {
		t.Fatalf("release last lease: %v, want ErrBusy", err)
	}
}

func TestStoreConcurrentLeaseAccounting(t *testing.T) {
	s, _ := NewStore(0)
	owner, _ := s.Allocate(testSpec(0), testResidency("device"))
	const workers = 32
	var wg sync.WaitGroup
	var failures atomic.Int32
	for range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 100 {
				l, err := s.Share(owner)
				if err != nil {
					failures.Add(1)
					return
				}
				if _, err := s.Describe(l); err != nil {
					failures.Add(1)
				}
				if err := l.Release(); err != nil {
					failures.Add(1)
				}
			}
		}()
	}
	wg.Wait()
	if got := failures.Load(); got != 0 {
		t.Fatalf("concurrent failures = %d", got)
	}
	desc, err := s.Describe(owner)
	if err != nil {
		t.Fatal(err)
	}
	if desc.References != 1 {
		t.Fatalf("references = %d, want 1", desc.References)
	}
}

func TestStoreValidation(t *testing.T) {
	tests := []struct {
		name string
		spec Spec
		res  Residency
	}{
		{"owner", func() Spec { s := testSpec(0); s.Owner = ""; return s }(), testResidency("device")},
		{"range", func() Spec { s := testSpec(0); s.Tokens.End = s.Tokens.Start; return s }(), testResidency("device")},
		{"format", func() Spec { s := testSpec(0); s.Format.DType = ""; return s }(), testResidency("device")},
		{"residency", testSpec(0), Residency{}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s, _ := NewStore(0)
			if _, err := s.Allocate(tt.spec, tt.res); !errors.Is(err, ErrInvalid) {
				t.Fatalf("Allocate(%+v, %+v) = %v, want ErrInvalid", tt.spec, tt.res, err)
			}
		})
	}
	if _, err := NewStore(-1); !errors.Is(err, ErrInvalid) {
		t.Fatalf("NewStore(-1) = %v, want ErrInvalid", err)
	}
}

func TestStoreAllowsAllLayerFormat(t *testing.T) {
	s, _ := NewStore(1)
	lease, err := s.Allocate(Spec{
		Owner: "request", Tokens: Range{Start: 0, End: 16},
		Format: Format{Layer: AllLayers, DType: "float32", Layout: "cuda-paged-all-layers", KVHeads: 8, HeadDim: 64},
	}, Residency{Backend: "cuda", Tier: "device", Domain: "gpu0"})
	if err != nil {
		t.Fatalf("Allocate: %v", err)
	}
	if err := s.Evict(context.Background(), lease, func(context.Context, Descriptor) error { return nil }); err != nil {
		t.Fatalf("Evict: %v", err)
	}
}

func ExampleStore() {
	store, _ := NewStore(8)
	lease, _ := store.Allocate(testSpec(0), Residency{Backend: "metal", Tier: "unified", Domain: "gpu-0"})
	desc, _ := store.Describe(lease)
	fmt.Println(desc.Spec.Tokens, desc.Residency.Tier)
	// Output: {0 16} unified
}
