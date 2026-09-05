package cuptiactivity

import (
	"errors"
	"runtime"
	"sync"
	"testing"
)

func TestBufferOwnership(t *testing.T) {
	p, err := newBufferPool(64, 1)
	if err != nil {
		t.Fatal(err)
	}
	ptr, size := p.request()
	if ptr == 0 || ptr%8 != 0 || size != 64 {
		t.Fatalf("pointer=%x size=%d", ptr, size)
	}
	if ptr, size := p.request(); ptr != 0 || size != 0 {
		t.Fatal("exceeded buffer budget")
	}
	runtime.GC()
	want := errors.New("consumer error")
	err = p.finish(ptr, size, 16, func(data []byte) error {
		if len(data) != 16 {
			t.Fatal("wrong valid region")
		}
		pending, _ := p.status()
		if pending != 1 {
			t.Fatal("released buffer before consumer returned")
		}
		data[15] = 42
		return want
	})
	if !errors.Is(err, want) {
		t.Fatalf("finish = %v", err)
	}
	if pending, refused := p.status(); pending != 0 || refused != 1 {
		t.Fatalf("pending=%d refused=%d", pending, refused)
	}
	if err := p.finish(ptr, size, 0, nil); err == nil {
		t.Fatal("accepted duplicate completion")
	}
}

func TestBufferValidation(t *testing.T) {
	for _, tt := range []struct{ bytes, limit int }{{0, 1}, {7, 1}, {9, 1}, {8, 0}, {8, -1}, {16, int(^uint(0) >> 1)}} {
		if _, err := newBufferPool(tt.bytes, tt.limit); err == nil {
			t.Fatalf("accepted %+v", tt)
		}
	}
	for _, tt := range []struct{ size, valid uint }{{63, 0}, {64, 65}, {128, 16}} {
		p, _ := newBufferPool(64, 1)
		ptr, _ := p.request()
		if err := p.finish(ptr, tt.size, tt.valid, func([]byte) error { t.Fatal("read invalid buffer"); return nil }); err == nil {
			t.Fatal("accepted invalid bounds")
		}
		if pending, _ := p.status(); pending != 0 {
			t.Fatal("leaked returned buffer after invalid metadata")
		}
	}
}

func TestBufferConcurrentCompletion(t *testing.T) {
	p, _ := newBufferPool(64, 1)
	ptr, size := p.request()
	entered, proceed := make(chan struct{}), make(chan struct{})
	var wg sync.WaitGroup
	wg.Go(func() {
		if err := p.finish(ptr, size, 1, func([]byte) error { close(entered); <-proceed; return nil }); err != nil {
			t.Error(err)
		}
	})
	<-entered
	if err := p.finish(ptr, size, 1, nil); err == nil {
		t.Error("accepted overlapping completion")
	}
	if ptr, _ := p.request(); ptr != 0 {
		t.Error("reused buffer during completion")
	}
	close(proceed)
	wg.Wait()
	ptr, size = p.request()
	if ptr == 0 {
		t.Fatal("did not recover budget")
	}
	if err := p.finish(ptr, size, 0, nil); err != nil {
		t.Fatal(err)
	}
}
