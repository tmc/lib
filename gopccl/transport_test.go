package gopccl

import (
	"context"
	"errors"
	"fmt"
	"testing"
)

var errFakeTransportDrop = fmt.Errorf("%w: fake transport drop", ErrInvalidArgument)

type fakeTransport struct {
	dropEvery int
	sends     int
}

func newFakeTransport() *fakeTransport {
	return &fakeTransport{dropEvery: 2}
}

func (t *fakeTransport) send(ctx context.Context, payload []byte) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if len(payload) == 0 {
		return fmt.Errorf("%w: empty fake transport payload", ErrInvalidArgument)
	}
	t.sends++
	if t.dropEvery > 0 && t.sends%t.dropEvery == 0 {
		return errFakeTransportDrop
	}
	return nil
}

func (t *fakeTransport) sendWithRetry(ctx context.Context, payload []byte, retries int) error {
	var err error
	for attempt := 0; attempt <= retries; attempt++ {
		err = t.send(ctx, payload)
		if err == nil {
			return nil
		}
		if !errors.Is(err, errFakeTransportDrop) {
			return err
		}
	}
	return err
}

func TestGeneratedFakeTransportDropRetry(t *testing.T) {
	ctx := context.Background()
	transport := newFakeTransport()
	if err := transport.send(ctx, []byte("warmup")); err != nil {
		t.Fatal(err)
	}
	if err := transport.sendWithRetry(ctx, []byte("payload"), 1); err != nil {
		t.Fatal(err)
	}
	if transport.sends != 3 {
		t.Fatalf("sends = %d, want 3", transport.sends)
	}

	alwaysDrop := &fakeTransport{dropEvery: 1}
	if err := alwaysDrop.sendWithRetry(ctx, []byte("payload"), 1); !errors.Is(err, errFakeTransportDrop) {
		t.Fatalf("sendWithRetry error = %v, want fake transport drop", err)
	}
}
