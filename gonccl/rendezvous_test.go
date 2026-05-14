package gonccl

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"testing"
)

type fakeStore struct {
	values map[string][]byte
}

func newFakeStore() *fakeStore {
	return &fakeStore{values: map[string][]byte{}}
}

func (s *fakeStore) Set(ctx context.Context, key string, value []byte) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if s.values == nil {
		s.values = map[string][]byte{}
	}
	s.values[key] = append([]byte(nil), value...)
	return nil
}

func (s *fakeStore) Get(ctx context.Context, key string) ([]byte, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if s.values == nil {
		return nil, fmt.Errorf("%w: rendezvous key %q", ErrInvalidArgument, key)
	}
	value, ok := s.values[key]
	if !ok {
		return nil, fmt.Errorf("%w: rendezvous key %q", ErrInvalidArgument, key)
	}
	return append([]byte(nil), value...), nil
}

func (s *fakeStore) Delete(ctx context.Context, key string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	delete(s.values, key)
	return nil
}

func TestGeneratedFakeStoreRoundTrip(t *testing.T) {
	ctx := context.Background()
	store := newFakeStore()
	var _ Store = store
	if err := store.Set(ctx, "rank/0", []byte("ready")); err != nil {
		t.Fatal(err)
	}
	got, err := store.Get(ctx, "rank/0")
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, []byte("ready")) {
		t.Fatalf("Get = %q, want ready", got)
	}
	got[0] = 'x'
	again, err := store.Get(ctx, "rank/0")
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(again, []byte("ready")) {
		t.Fatalf("store returned aliased value %q", again)
	}
	if err := store.Delete(ctx, "rank/0"); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Get(ctx, "rank/0"); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("Get after Delete error = %v, want ErrInvalidArgument", err)
	}
}
