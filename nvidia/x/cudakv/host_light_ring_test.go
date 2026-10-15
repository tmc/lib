package cudakv

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"sync"
	"testing"
	"time"
)

func TestDeviceCommandRingLifecycle(t *testing.T) {
	r, err := NewDeviceCommandRing(2, 7)
	if err != nil {
		t.Fatal(err)
	}
	if err := r.Publish(Descriptor{SeqID: 1, Generation: 7}); err != nil {
		t.Fatal(err)
	}
	if err := r.Publish(Descriptor{SeqID: 2, Generation: 7}); err != nil {
		t.Fatal(err)
	}
	if err := r.Publish(Descriptor{SeqID: 3, Generation: 7}); !errors.Is(err, errRingFull) {
		t.Fatal("expected bounded overflow")
	}
	got, ok := r.Consume()
	if !ok || got.SeqID != 1 {
		t.Fatalf("Consume = %+v, %v", got, ok)
	}
	r.Close()
	if err := r.Publish(Descriptor{Generation: 7}); !errors.Is(err, errRingClosed) {
		t.Fatal("publish after close succeeded")
	}
	if _, ok := r.Consume(); !ok {
		t.Fatal("queued descriptor was lost on close")
	}
}

func TestDeviceCommandRingConcurrentPublish(t *testing.T) {
	const producers, perProducer = 4, 128
	r, err := NewDeviceCommandRing(3, 7)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	var wg sync.WaitGroup
	errorsFound := make(chan error, producers+1)
	for producer := 0; producer < producers; producer++ {
		wg.Add(1)
		go func(producer int) {
			defer wg.Done()
			for i := 0; i < perProducer; i++ {
				d := Descriptor{SeqID: uint64(producer*perProducer + i), Generation: 7, PromptToken: int32(i)}
				for {
					if err := ctx.Err(); err != nil {
						errorsFound <- err
						return
					}
					err := r.Publish(d)
					if err == nil {
						break
					}
					if !errors.Is(err, errRingFull) {
						errorsFound <- err
						return
					}
					runtime.Gosched()
				}
			}
		}(producer)
	}
	done := make(chan struct{})
	go func() { wg.Wait(); r.Close(); close(done) }()
	seen := make([]bool, producers*perProducer)
	last := [producers]int{-1, -1, -1, -1}
	consumed := 0
	for consumed < len(seen) && ctx.Err() == nil {
		d, ok := r.Consume()
		if !ok {
			runtime.Gosched()
			continue
		}
		if d.SeqID >= uint64(len(seen)) || seen[d.SeqID] || d.Generation != 7 {
			cancel()
			errorsFound <- fmt.Errorf("invalid or repeated descriptor %+v", d)
			break
		}
		producer, item := int(d.SeqID)/perProducer, int(d.SeqID)%perProducer
		if d.PromptToken != int32(item) || item != last[producer]+1 {
			cancel()
			errorsFound <- fmt.Errorf("producer %d reordered item %d after %d", producer, item, last[producer])
			break
		}
		last[producer] = item
		seen[d.SeqID] = true
		consumed++
	}
	cancel()
	wg.Wait()
	<-done
	close(errorsFound)
	for err := range errorsFound {
		t.Error(err)
	}
	if consumed != len(seen) || r.Len() != 0 {
		t.Fatalf("consumed %d of %d, queued %d", consumed, len(seen), r.Len())
	}
}
