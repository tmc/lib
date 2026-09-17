package cudakv

import (
	"sync"
	"testing"
)

func TestDeviceCommandRingLifecycle(t *testing.T) {
	r := NewDeviceCommandRing(2)
	if err := r.Publish(Descriptor{SeqID: 1, Generation: 7}); err != nil {
		t.Fatal(err)
	}
	if err := r.Publish(Descriptor{SeqID: 2, Generation: 7}); err != nil {
		t.Fatal(err)
	}
	if err := r.Publish(Descriptor{SeqID: 3}); err == nil {
		t.Fatal("expected bounded overflow")
	}
	got, ok := r.Consume()
	if !ok || got.SeqID != 1 {
		t.Fatalf("Consume = %+v, %v", got, ok)
	}
	r.Close()
	if err := r.Publish(Descriptor{}); err == nil {
		t.Fatal("publish after close succeeded")
	}
	if _, ok := r.Consume(); !ok {
		t.Fatal("queued descriptor was lost on close")
	}
}

func TestDeviceCommandRingConcurrentPublish(t *testing.T) {
	r := NewDeviceCommandRing(64)
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func(i int) { defer wg.Done(); _ = r.Publish(Descriptor{SeqID: uint64(i)}) }(i)
	}
	wg.Wait()
	if n := r.Len(); n > 64 {
		t.Fatalf("Len = %d", n)
	}
}
