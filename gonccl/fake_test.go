package gonccl

import (
	"context"
	"errors"
	"testing"
)

type fakeBackend struct {
	closed bool
}

func (f *fakeBackend) Name() string                  { return "fake" }
func (f *fakeBackend) Close() error                  { f.closed = true; return nil }
func (f *fakeBackend) Capabilities() Capabilities    { return Capabilities{} }
func (f *fakeBackend) barrier(context.Context) error { return nil }
func (f *fakeBackend) allReduce(ctx context.Context, dst Buffer, src Buffer, op ReduceOp) error {
	return nil
}
func (f *fakeBackend) allGather(ctx context.Context, dst Buffer, src Buffer) error { return nil }
func (f *fakeBackend) broadcast(ctx context.Context, dst Buffer, src Buffer, root int) error {
	return nil
}
func (f *fakeBackend) allToAll(ctx context.Context, dst Buffer, src Buffer) error { return nil }
func (f *fakeBackend) reduceScatter(ctx context.Context, dst Buffer, src Buffer, op ReduceOp) error {
	return nil
}
func (f *fakeBackend) send(ctx context.Context, peer int, src Buffer) error { return nil }
func (f *fakeBackend) recv(ctx context.Context, peer int, dst Buffer) error { return nil }

func TestGeneratedGroupCloseUsesBackend(t *testing.T) {
	f := &fakeBackend{}
	x := &Group{b: f, rank: 0, size: 1}
	if err := x.Close(); err != nil {
		t.Fatal(err)
	}
	if !f.closed {
		t.Fatal("backend was not closed")
	}
}

func TestGeneratedBarrierRejectsNilResource(t *testing.T) {
	err := ((*Group)(nil)).Barrier(nil)
	if err == nil {
		t.Fatal("nil resource accepted")
	}
	var opErr *OperationError
	if !errors.As(err, &opErr) {
		t.Fatalf("err = %T, want *OperationError", err)
	}
	if opErr.Op != "Barrier" {
		t.Fatalf("op = %q, want Barrier", opErr.Op)
	}
}
