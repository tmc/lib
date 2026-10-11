//go:build linux

package bpfprobe

import (
	"context"
	"encoding/binary"
	"errors"
	"os"
	"syscall"
	"testing"
	"time"
)

func testRing() *Ring {
	page := os.Getpagesize()
	return &Ring{consumer: make([]byte, page), producer: make([]byte, page+2*page), capacity: uint64(page), done: make(chan struct{})}
}

func putRecord(r *Ring, pos uint64, header uint32, payload string) {
	offset := uint64(os.Getpagesize()) + (pos & (r.capacity - 1))
	binary.NativeEndian.PutUint32(r.producer[offset:], header)
	copy(r.producer[offset+8:], payload)
}

func TestRingRecords(t *testing.T) {
	for _, tt := range []struct {
		name     string
		position uint64
	}{
		{"ordinary", 0},
		{"wrap", uint64(os.Getpagesize()) - 8},
		{"position_overflow", ^uint64(0) - 7},
	} {
		t.Run(tt.name, func(t *testing.T) {
			r := testRing()
			binary.NativeEndian.PutUint64(r.consumer, tt.position)
			binary.NativeEndian.PutUint64(r.producer, tt.position+16)
			putRecord(r, tt.position, 3, "abc")
			data, ready, err := r.readRecord()
			if err != nil || !ready || string(data) != "abc" {
				t.Fatalf("read = %q, %v, %v", data, ready, err)
			}
			if got := binary.NativeEndian.Uint64(r.consumer); got != tt.position+16 {
				t.Fatalf("consumer = %d, want %d", got, tt.position+16)
			}
			putRecord(r, tt.position, 3, "xyz")
			if string(data) != "abc" {
				t.Fatal("returned record aliases mapped storage")
			}
		})
	}
}

func TestRingBusyDiscardAndMalformed(t *testing.T) {
	for _, tt := range []struct {
		name      string
		header    uint32
		available uint64
		wantErr   bool
	}{
		{"busy", 1<<31 | 3, 16, false},
		{"discard", 1<<30 | 3, 16, false},
		{"short_header", 0, 7, true},
		{"over_capacity", 0, uint64(os.Getpagesize()) + 8, true},
		{"short_payload", 16, 16, true},
		{"oversize_payload", uint32(os.Getpagesize()), uint64(os.Getpagesize()), true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			r := testRing()
			binary.NativeEndian.PutUint64(r.producer, tt.available)
			putRecord(r, 0, tt.header, "abc")
			_, ready, err := r.readRecord()
			if ready || (err != nil) != tt.wantErr {
				t.Fatalf("read = ready %v, err %v", ready, err)
			}
			wantPos := uint64(0)
			if tt.name == "discard" {
				wantPos = 16
			}
			if got := binary.NativeEndian.Uint64(r.consumer); got != wantPos {
				t.Fatalf("consumer = %d, want %d", got, wantPos)
			}
		})
	}
}

func TestRingDiscardThenRecord(t *testing.T) {
	r := testRing()
	binary.NativeEndian.PutUint64(r.producer, 32)
	putRecord(r, 0, 1<<30|3, "bad")
	putRecord(r, 16, 3, "yes")
	data, ready, err := r.readRecord()
	if err != nil || !ready || string(data) != "yes" {
		t.Fatalf("read = %q, %v, %v", data, ready, err)
	}
}

func TestRingCancellationAndZeroValue(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := testRing().Read(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled read: %v", err)
	}
	var zero Ring
	if _, err := zero.Read(context.Background()); !errors.Is(err, os.ErrClosed) {
		t.Fatalf("zero read: %v", err)
	}
	if err := zero.Close(); err != nil {
		t.Fatal(err)
	}
	if err := zero.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestOpenRingInvalid(t *testing.T) {
	for _, capacity := range []uint32{0, 1, 3, uint32(os.Getpagesize())} {
		if _, err := OpenRing(-1, capacity); err == nil {
			t.Fatalf("accepted fd -1 capacity %d", capacity)
		}
	}
}

func TestOpenRingRejectsNonMap(t *testing.T) {
	path := t.TempDir() + "/not-a-map"
	if err := os.WriteFile(path, []byte("not a map"), 0600); err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{"/dev/null", path} {
		t.Run(path, func(t *testing.T) {
			f, err := os.Open(path)
			if err != nil {
				t.Fatal(err)
			}
			defer f.Close()
			if r, err := OpenRing(int(f.Fd()), uint32(os.Getpagesize())); err == nil {
				r.Close()
				t.Fatal("accepted non-map descriptor")
			}
		})
	}
}

func TestRingCloseInterruptsRead(t *testing.T) {
	page := os.Getpagesize()
	consumer, err := syscall.Mmap(-1, 0, page, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_PRIVATE|syscall.MAP_ANON)
	if err != nil {
		t.Fatal(err)
	}
	producer, err := syscall.Mmap(-1, 0, 3*page, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_PRIVATE|syscall.MAP_ANON)
	if err != nil {
		_ = syscall.Munmap(consumer)
		t.Fatal(err)
	}
	r := &Ring{consumer: consumer, producer: producer, capacity: uint64(page), done: make(chan struct{})}
	defer r.Close()
	result := make(chan error, 1)
	go func() { _, err := r.Read(context.Background()); result <- err }()
	if err := r.Close(); err != nil {
		t.Fatal(err)
	}
	select {
	case err := <-result:
		if !errors.Is(err, os.ErrClosed) {
			t.Fatalf("read after close: %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("close did not interrupt read")
	}
}
