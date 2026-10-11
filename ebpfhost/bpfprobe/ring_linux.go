//go:build linux

package bpfprobe

import (
	"context"
	"errors"
	"fmt"
	"os"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
	"unsafe"

	"github.com/tmc/lib/ebpfhost/bpfload"
)

// Ring reads records from a BPF_MAP_TYPE_RINGBUF map. The zero value is not
// usable. Only one caller may call Read at a time. Close may run concurrently.
// The map descriptor remains owned by the caller.
type Ring struct {
	mu                 sync.Mutex
	consumer, producer []byte
	capacity           uint64
	closed             bool
	done               chan struct{}
}

// OpenRing maps a ring-buffer map descriptor. Capacity must match the map's
// max_entries and must be a power of two and a multiple of the system page size.
func OpenRing(fd int, capacity uint32) (*Ring, error) {
	page := os.Getpagesize()
	c := uint64(capacity)
	maxInt := uint64(^uint(0) >> 1)
	if fd < 0 || c == 0 || c&(c-1) != 0 || c%uint64(page) != 0 || c > (maxInt-uint64(page))/2 {
		return nil, errors.New("bpf ring: invalid descriptor or capacity")
	}
	info, err := bpfload.InspectMap(fd)
	if err != nil {
		return nil, fmt.Errorf("bpf ring: inspect map: %w", err)
	}
	if info.Type != 27 || info.MaxEntries != capacity || info.KeySize != 0 || info.ValueSize != 0 {
		return nil, errors.New("bpf ring: map type or capacity mismatch")
	}
	consumer, err := syscall.Mmap(fd, 0, page, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_SHARED)
	if err != nil {
		return nil, fmt.Errorf("bpf ring: map consumer: %w", err)
	}
	producer, err := syscall.Mmap(fd, int64(page), page+2*int(capacity), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		_ = syscall.Munmap(consumer)
		return nil, fmt.Errorf("bpf ring: map producer: %w", err)
	}
	return &Ring{consumer: consumer, producer: producer, capacity: c, done: make(chan struct{})}, nil
}

// Read waits for the next committed record and returns an owned copy. Discarded
// records are skipped. Cancellation leaves the next unconsumed record intact.
func (r *Ring) Read(ctx context.Context) ([]byte, error) {
	if ctx == nil {
		return nil, errors.New("bpf ring: nil context")
	}
	for {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		r.mu.Lock()
		if r.closed || r.done == nil {
			r.mu.Unlock()
			return nil, os.ErrClosed
		}
		data, ready, err := r.readRecord()
		done := r.done
		r.mu.Unlock()
		if err != nil || ready {
			return data, err
		}
		timer := time.NewTimer(time.Millisecond)
		select {
		case <-ctx.Done():
			timer.Stop()
			return nil, ctx.Err()
		case <-done:
			timer.Stop()
			return nil, os.ErrClosed
		case <-timer.C:
		}
	}
}

func (r *Ring) readRecord() ([]byte, bool, error) {
	consumer := (*uint64)(unsafe.Pointer(&r.consumer[0]))
	producer := (*uint64)(unsafe.Pointer(&r.producer[0]))
	for {
		pos := atomic.LoadUint64(consumer)
		available := atomic.LoadUint64(producer) - pos
		if available == 0 {
			return nil, false, nil
		}
		if available < 8 || available > r.capacity {
			return nil, false, errors.New("bpf ring: invalid producer position")
		}
		offset := uint64(os.Getpagesize()) + (pos & (r.capacity - 1))
		header := atomic.LoadUint32((*uint32)(unsafe.Pointer(&r.producer[offset])))
		if header&(1<<31) != 0 {
			return nil, false, nil
		}
		length := uint64(header & ((1 << 30) - 1))
		advance := (length + 8 + 7) &^ 7
		if advance > r.capacity || advance > available {
			return nil, false, errors.New("bpf ring: invalid record length")
		}
		if header&(1<<30) != 0 {
			atomic.StoreUint64(consumer, pos+advance)
			continue
		}
		data := append([]byte{}, r.producer[offset+8:offset+8+length]...)
		atomic.StoreUint64(consumer, pos+advance)
		return data, true, nil
	}
}

// Close releases both mappings and interrupts a pending Read. It is idempotent.
func (r *Ring) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return nil
	}
	r.closed = true
	if r.done != nil {
		close(r.done)
	}
	var errs []error
	if r.consumer != nil {
		errs = append(errs, syscall.Munmap(r.consumer))
	}
	if r.producer != nil {
		errs = append(errs, syscall.Munmap(r.producer))
	}
	r.consumer, r.producer = nil, nil
	return errors.Join(errs...)
}
