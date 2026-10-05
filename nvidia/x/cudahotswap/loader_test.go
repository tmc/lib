package cudahotswap

import (
	"context"
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/x/cuda"
)

func TestLoaderRefusals(t *testing.T) {
	if _, err := NewNVRTCLoader(nil); err == nil {
		t.Fatal("nil device accepted")
	}
	var absent *NVRTCLoader
	if _, err := absent.Compile(context.Background(), Source{}); err == nil {
		t.Fatal("nil compiler accepted")
	}
	if _, err := absent.Load(context.Background(), nil); err == nil {
		t.Fatal("nil loader accepted")
	}
	// These refusals must happen before touching an unopened CUDA device.
	l, err := NewNVRTCLoader(new(cuda.Device))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := l.Compile(nil, Source{}); err == nil {
		t.Fatal("nil compile context accepted")
	}
	if _, err := l.Load(nil, nil); err == nil {
		t.Fatal("nil load context accepted")
	}
	for _, c := range []*Compilation{nil, {}, {Entry: "entry", loader: l}} {
		if k, err := l.Load(context.Background(), c); k != nil || err == nil {
			t.Fatalf("invalid compilation: kernel=%v err=%v", k, err)
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := l.Compile(ctx, Source{}); !errors.Is(err, context.Canceled) {
		t.Fatalf("compile cancellation: %v", err)
	}
	if _, err := l.Load(ctx, nil); !errors.Is(err, context.Canceled) {
		t.Fatalf("load cancellation: %v", err)
	}
	if _, err := l.Compile(context.Background(), Source{}); err == nil {
		t.Fatal("empty source accepted")
	}
}

func TestNativeKernelRefusals(t *testing.T) {
	var k *nativeKernel
	if k.Raw() != 0 || k.Close() != nil {
		t.Fatal("nil kernel does not behave as an empty owner")
	}
	if err := k.Wait(context.Background(), nil); err == nil {
		t.Fatal("nil wait establishes completion")
	}
	owner := &nativeKernel{module: new(cuda.Module), dev: new(cuda.Device)}
	if err := owner.Wait(nil, nil); err == nil {
		t.Fatal("nil wait context accepted")
	}
}
