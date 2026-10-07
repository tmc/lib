//go:build native && linux

package cudaheadroom_test

import (
	"errors"
	"fmt"
	"io"
	"reflect"
	"testing"
)

type partialOwner struct{ owner io.Closer }

func (p *partialOwner) capture(err error) error {
	if errors.As(err, &p.owner) {
		return errors.Join(err, p.retry())
	}
	return err
}

func (p *partialOwner) retry() error {
	if p.owner == nil {
		return nil
	}
	if err := p.owner.Close(); err != nil {
		return fmt.Errorf("headroom fixture: retained partial owner: %w", err)
	}
	p.owner = nil
	return nil
}

func (p *partialOwner) closeDevice(close func() error) error {
	if p.owner != nil {
		return errors.New("headroom fixture: device retained because partial owner cleanup failed")
	}
	return close()
}

type fakePartialOwner struct {
	operation, cleanup error
	failures, calls    int
	order              *[]string
}

func (f *fakePartialOwner) Error() string { return f.operation.Error() }
func (f *fakePartialOwner) Unwrap() error { return f.operation }
func (f *fakePartialOwner) Close() error {
	f.calls++
	*f.order = append(*f.order, "owner")
	if f.calls <= f.failures {
		return f.cleanup
	}
	return nil
}

// This selector calls no CUDA, NVRTC, loader, lock or device function.
func TestNativePartialOwnerCleanupCPU(t *testing.T) {
	operation, cleanup := errors.New("operation failed"), errors.New("cleanup failed")
	for _, tt := range []struct {
		name     string
		failures int
	}{{"close succeeds", 0}, {"retry succeeds", 1}, {"retained failure", 2}} {
		t.Run(tt.name, func(t *testing.T) {
			var order []string
			f := &fakePartialOwner{operation: operation, cleanup: cleanup, failures: tt.failures, order: &order}
			var p partialOwner
			err := p.capture(fmt.Errorf("compile or load: %w", f))
			if !errors.Is(err, operation) || f.calls != 1 {
				t.Fatalf("initial cleanup = %v, calls %d", err, f.calls)
			}
			if errors.Is(err, cleanup) != (tt.failures > 0) {
				t.Fatalf("cleanup error lost: %v", err)
			}
			if tt.failures > 0 && p.owner != f {
				t.Fatal("failed owner dropped")
			}
			retryErr := p.retry()
			if (retryErr != nil) != (tt.failures == 2) || errors.Is(retryErr, cleanup) != (tt.failures == 2) {
				t.Fatalf("retry = %v", retryErr)
			}
			deviceErr := p.closeDevice(func() error { order = append(order, "device"); return nil })
			if tt.failures == 2 {
				if p.owner != f || deviceErr == nil || !reflect.DeepEqual(order, []string{"owner", "owner"}) {
					t.Fatalf("persistent ownership/order = %+v %v %v", p, order, deviceErr)
				}
			} else {
				want := []string{"owner", "device"}
				if tt.failures == 1 {
					want = []string{"owner", "owner", "device"}
				}
				if p.owner != nil || deviceErr != nil || !reflect.DeepEqual(order, want) {
					t.Fatalf("retirement order = %v, %v", order, deviceErr)
				}
			}
		})
	}
	var p partialOwner
	if got := p.capture(operation); got != operation || p.owner != nil {
		t.Fatalf("plain error = %v", got)
	}
	if err := p.capture(nil); err != nil {
		t.Fatal(err)
	}
}
