package cuptipcsampling

import (
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"
	"unsafe"

	"github.com/tmc/lib/nvidia/cupti"
)

func TestRingBufferValidation(t *testing.T) {
	tests := []struct {
		name string
		cfg  RingBufferConfig
	}{
		{
			name: "capacity too small",
			cfg:  RingBufferConfig{Capacity: 1, PCCount: 1, ReasonCount: 1, MaxBytes: 1024},
		},
		{
			name: "zero capacity",
			cfg:  RingBufferConfig{Capacity: 0, PCCount: 1, ReasonCount: 1, MaxBytes: 1024},
		},
		{
			name: "negative capacity",
			cfg:  RingBufferConfig{Capacity: -2, PCCount: 1, ReasonCount: 1, MaxBytes: 1024},
		},
		{
			name: "no pc count and no activity size",
			cfg:  RingBufferConfig{Capacity: 2, PCCount: 0, ActivitySize: 0},
		},
		{
			name: "activity size not multiple of 8",
			cfg:  RingBufferConfig{Capacity: 2, ActivitySize: 15},
		},
		{
			name: "activity size smaller than 8",
			cfg:  RingBufferConfig{Capacity: 2, ActivitySize: 4},
		},
		{
			name: "invalid pc count",
			cfg:  RingBufferConfig{Capacity: 2, PCCount: 0, ReasonCount: 1, MaxBytes: 1024},
		},
		{
			name: "invalid reason count",
			cfg:  RingBufferConfig{Capacity: 2, PCCount: 1, ReasonCount: 0, MaxBytes: 1024},
		},
		{
			name: "pc sampling exceeds max bytes",
			cfg:  RingBufferConfig{Capacity: 2, PCCount: 100, ReasonCount: 10, MaxBytes: 10},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rb, err := NewRingBufferWithConfig(tt.cfg)
			if err == nil {
				rb.Close()
				t.Fatalf("expected error for config %+v, got nil", tt.cfg)
			}
		})
	}
}

func TestRingBufferLifecycle(t *testing.T) {
	rb, err := NewRingBuffer(3, 2, 2, 1024)
	if err != nil {
		t.Fatalf("NewRingBuffer failed: %v", err)
	}

	stats := rb.Stats()
	if stats.Capacity != 3 || stats.Available != 3 || stats.InFlight != 0 || stats.Ready != 0 {
		t.Fatalf("unexpected initial stats: %+v", stats)
	}

	// Idempotent Close
	if err := rb.Close(); err != nil {
		t.Fatalf("first Close failed: %v", err)
	}
	if err := rb.Close(); err != nil {
		t.Fatalf("second Close failed: %v", err)
	}

	// Operations after Close return ErrRingBufferClosed
	if _, err := rb.Acquire(); !errors.Is(err, ErrRingBufferClosed) {
		t.Fatalf("expected ErrRingBufferClosed on Acquire, got: %v", err)
	}
	if err := rb.Produce(&RingSlot{}); !errors.Is(err, ErrRingBufferClosed) {
		t.Fatalf("expected ErrRingBufferClosed on Produce, got: %v", err)
	}
	if _, err := rb.Consume(); !errors.Is(err, ErrRingBufferClosed) {
		t.Fatalf("expected ErrRingBufferClosed on Consume, got: %v", err)
	}
	if _, err := rb.TryConsume(); !errors.Is(err, ErrRingBufferClosed) {
		t.Fatalf("expected ErrRingBufferClosed on TryConsume, got: %v", err)
	}
	if _, err := rb.Handoff(nil); !errors.Is(err, ErrRingBufferClosed) {
		t.Fatalf("expected ErrRingBufferClosed on Handoff, got: %v", err)
	}
}

func TestRingBufferProduceConsume(t *testing.T) {
	rb, err := NewRingBuffer(2, 2, 2, 1024)
	if err != nil {
		t.Fatalf("NewRingBuffer: %v", err)
	}
	defer rb.Close()

	slot0, err := rb.Acquire()
	if err != nil {
		t.Fatalf("Acquire slot0: %v", err)
	}
	if slot0.Index() != 0 {
		t.Fatalf("expected index 0, got %d", slot0.Index())
	}

	stats := rb.Stats()
	if stats.Available != 1 || stats.InFlight != 1 || stats.HighWaterMark != 1 {
		t.Fatalf("stats after 1 acquire: %+v", stats)
	}

	if err := rb.Produce(slot0); err != nil {
		t.Fatalf("Produce: %v", err)
	}

	stats = rb.Stats()
	if stats.Ready != 1 {
		t.Fatalf("expected Ready=1, got %d", stats.Ready)
	}

	cSlot, err := rb.Consume()
	if err != nil {
		t.Fatalf("Consume: %v", err)
	}
	if cSlot.Index() != 0 {
		t.Fatalf("expected consumed slot index 0, got %d", cSlot.Index())
	}

	rb.Release(cSlot)

	stats = rb.Stats()
	if stats.Available != 2 || stats.InFlight != 0 || stats.Ready != 0 {
		t.Fatalf("stats after release: %+v", stats)
	}
}

func TestRingBufferHandoff(t *testing.T) {
	rb, err := NewRingBuffer(2, 2, 2, 1024)
	if err != nil {
		t.Fatalf("NewRingBuffer: %v", err)
	}
	defer rb.Close()

	s0, err := rb.Acquire()
	if err != nil {
		t.Fatalf("Acquire s0: %v", err)
	}
	if s0.Index() != 0 {
		t.Fatalf("expected slot index 0, got %d", s0.Index())
	}

	// Handoff s0: enqueues s0 to ready queue, acquires next free slot s1
	s1, err := rb.Handoff(s0)
	if err != nil {
		t.Fatalf("Handoff s0: %v", err)
	}
	if s1.Index() != 1 {
		t.Fatalf("expected next slot index 1, got %d", s1.Index())
	}

	// Consumer consumes s0
	c0, err := rb.Consume()
	if err != nil {
		t.Fatalf("Consume c0: %v", err)
	}
	if c0.Index() != 0 {
		t.Fatalf("expected c0 index 0, got %d", c0.Index())
	}
	rb.Release(c0)

	// Handoff s1: enqueues s1, acquires reused slot s0
	s0Reused, err := rb.Handoff(s1)
	if err != nil {
		t.Fatalf("Handoff s1: %v", err)
	}
	if s0Reused.Index() != 0 {
		t.Fatalf("expected s0Reused index 0, got %d", s0Reused.Index())
	}

	// Consumer consumes s1
	c1, err := rb.Consume()
	if err != nil {
		t.Fatalf("Consume c1: %v", err)
	}
	if c1.Index() != 1 {
		t.Fatalf("expected c1 index 1, got %d", c1.Index())
	}
	rb.Release(c1)
	rb.Release(s0Reused)
}

func TestRingBufferOverflow(t *testing.T) {
	const cap = 3
	rb, err := NewRingBuffer(cap, 2, 2, 1024)
	if err != nil {
		t.Fatalf("NewRingBuffer: %v", err)
	}
	defer rb.Close()

	slots := make([]*RingSlot, cap)
	for i := 0; i < cap; i++ {
		s, err := rb.Acquire()
		if err != nil {
			t.Fatalf("Acquire %d failed: %v", i, err)
		}
		slots[i] = s
	}

	// Fourth acquire must fail with ErrRingBufferFull
	overflowSlot, err := rb.Acquire()
	if !errors.Is(err, ErrRingBufferFull) {
		t.Fatalf("expected ErrRingBufferFull, got %v (slot=%v)", err, overflowSlot)
	}

	stats := rb.Stats()
	if stats.Overflows != 1 {
		t.Fatalf("expected 1 overflow, got %d", stats.Overflows)
	}
	if stats.HighWaterMark != cap {
		t.Fatalf("expected high water mark %d, got %d", cap, stats.HighWaterMark)
	}

	// Produce slots so consumer can drain
	for _, s := range slots {
		if err := rb.Produce(s); err != nil {
			t.Fatalf("Produce failed: %v", err)
		}
	}

	// Handoff while full still queues the filled buffer and reports overflow
	sOverflow, err := rb.Handoff(nil)
	if !errors.Is(err, ErrRingBufferFull) || sOverflow != nil {
		t.Fatalf("expected ErrRingBufferFull and nil slot from full Handoff, got slot=%v err=%v", sOverflow, err)
	}
	if rb.Stats().Overflows != 2 {
		t.Fatalf("expected 2 overflows, got %d", rb.Stats().Overflows)
	}

	// Consumer drains one slot
	cSlot, err := rb.Consume()
	if err != nil {
		t.Fatalf("Consume failed: %v", err)
	}
	rb.Release(cSlot)

	// Now Acquire succeeds
	recovered, err := rb.Acquire()
	if err != nil {
		t.Fatalf("Acquire after drain failed: %v", err)
	}
	rb.Release(recovered)

	// Test RecordDrops
	rb.RecordDrops(42)
	if rb.Stats().Drops != 42 {
		t.Fatalf("expected 42 drops, got %d", rb.Stats().Drops)
	}
}

func TestRingBufferConcurrent(t *testing.T) {
	const (
		cap   = 4
		items = 10000
	)

	rb, err := NewActivityRingBuffer(cap, 64)
	if err != nil {
		t.Fatalf("NewActivityRingBuffer: %v", err)
	}
	defer rb.Close()

	var wg sync.WaitGroup
	wg.Add(2)

	// Producer goroutine
	go func() {
		defer wg.Done()
		slot, err := rb.Acquire()
		if err != nil {
			t.Errorf("initial Acquire: %v", err)
			return
		}

		for i := 0; i < items; i++ {
			// Write sequence index into buffer
			*(*uint64)(unsafe.Pointer(&slot.ActivityBuffer()[0])) = uint64(i)
			slot.SetValidBytes(8)

			if i == items-1 {
				if err := rb.Produce(slot); err != nil {
					t.Errorf("final Produce: %v", err)
				}
				break
			}

			next, err := rb.Handoff(slot)
			for errors.Is(err, ErrRingBufferFull) {
				// Retry if consumer is temporarily behind
				next, err = rb.Acquire()
			}
			if err != nil {
				t.Errorf("Handoff %d: %v", i, err)
				return
			}
			slot = next
		}
	}()

	// Consumer goroutine
	consumed := make([]uint64, 0, items)
	go func() {
		defer wg.Done()
		for len(consumed) < items {
			slot, err := rb.Consume()
			if err != nil {
				t.Errorf("Consume at %d: %v", len(consumed), err)
				return
			}
			val := *(*uint64)(unsafe.Pointer(&slot.ActivityBuffer()[0]))
			consumed = append(consumed, val)
			rb.Release(slot)
		}
	}()

	wg.Wait()

	if len(consumed) != items {
		t.Fatalf("expected %d items consumed, got %d", items, len(consumed))
	}
	for i, v := range consumed {
		if v != uint64(i) {
			t.Fatalf("out-of-order item at %d: got %d", i, v)
		}
	}

	stats := rb.Stats()
	if stats.InFlight != 0 {
		t.Fatalf("expected 0 in-flight after test completion, got %d", stats.InFlight)
	}
	if stats.HighWaterMark < 1 || stats.HighWaterMark > cap {
		t.Fatalf("invalid high water mark %d for capacity %d", stats.HighWaterMark, cap)
	}
}

func TestRingBufferZeroAllocations(t *testing.T) {
	rb, err := NewRingBuffer(4, 2, 2, 1024)
	if err != nil {
		t.Fatalf("NewRingBuffer: %v", err)
	}
	defer rb.Close()

	// Steady-state Acquire -> Produce -> Consume -> Release rotation
	allocs := testing.AllocsPerRun(1000, func() {
		slot, err := rb.Acquire()
		if err != nil {
			t.Fatal(err)
		}
		if err := rb.Produce(slot); err != nil {
			t.Fatal(err)
		}
		consumed, err := rb.Consume()
		if err != nil {
			t.Fatal(err)
		}
		rb.Release(consumed)
	})

	if allocs != 0.0 {
		t.Fatalf("expected 0 allocations in steady-state Acquire/Produce/Consume/Release, got %f", allocs)
	}

	// Steady-state Handoff rotation
	activeSlot, err := rb.Acquire()
	if err != nil {
		t.Fatal(err)
	}

	handoffAllocs := testing.AllocsPerRun(1000, func() {
		next, err := rb.Handoff(activeSlot)
		if err != nil {
			t.Fatal(err)
		}
		c, err := rb.Consume()
		if err != nil {
			t.Fatal(err)
		}
		rb.Release(c)
		activeSlot = next
	})

	rb.Release(activeSlot)

	if handoffAllocs != 0.0 {
		t.Fatalf("expected 0 allocations in steady-state Handoff rotation, got %f", handoffAllocs)
	}
}

func BenchmarkRingBufferRotation(b *testing.B) {
	rb, err := NewRingBuffer(4, 2, 2, 1024)
	if err != nil {
		b.Fatalf("NewRingBuffer: %v", err)
	}
	defer rb.Close()

	slot, err := rb.Acquire()
	if err != nil {
		b.Fatal(err)
	}

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		next, err := rb.Handoff(slot)
		if err != nil {
			b.Fatal(err)
		}
		c, err := rb.Consume()
		if err != nil {
			b.Fatal(err)
		}
		rb.Release(c)
		slot = next
	}

	b.StopTimer()
	rb.Release(slot)
}

func TestActivityRingBuffer(t *testing.T) {
	const (
		cap  = 3
		size = 256
	)

	rb, err := NewActivityRingBuffer(cap, size)
	if err != nil {
		t.Fatalf("NewActivityRingBuffer: %v", err)
	}
	defer rb.Close()

	slot, err := rb.Acquire()
	if err != nil {
		t.Fatalf("Acquire: %v", err)
	}

	if slot.ActivitySize() != size {
		t.Fatalf("expected activity size %d, got %d", size, slot.ActivitySize())
	}
	if slot.ActivityPointer() == 0 {
		t.Fatal("expected non-zero activity pointer")
	}
	if slot.ActivityPointer()%8 != 0 {
		t.Fatalf("activity pointer 0x%x is not 8-byte aligned", slot.ActivityPointer())
	}

	// Write payload and check roundtrip
	buf := slot.ActivityBuffer()
	copy(buf, []byte("cupti-activity-test-data"))
	slot.SetValidBytes(24)

	next, err := rb.Handoff(slot)
	if err != nil {
		t.Fatalf("Handoff: %v", err)
	}
	if next == nil {
		t.Fatal("expected non-nil next slot")
	}

	consumed, err := rb.Consume()
	if err != nil {
		t.Fatalf("Consume: %v", err)
	}
	if consumed.ValidBytes() != 24 {
		t.Fatalf("expected 24 valid bytes, got %d", consumed.ValidBytes())
	}
	if string(consumed.ActivityBuffer()[:24]) != "cupti-activity-test-data" {
		t.Fatalf("unexpected content: %s", string(consumed.ActivityBuffer()[:24]))
	}

	rb.Release(consumed)
	rb.Release(next)
}

func TestMultiBufferAlias(t *testing.T) {
	mb, err := NewMultiBuffer(2, 2, 2, 1024)
	if err != nil {
		t.Fatalf("NewMultiBuffer: %v", err)
	}
	defer mb.Close()

	if mb.Stats().Capacity != 2 {
		t.Fatalf("expected capacity 2, got %d", mb.Stats().Capacity)
	}
}

func TestSessionRingBufferConcurrent(t *testing.T) {
	for _, version := range []uint32{130001, 130400} {
		t.Run(fmt.Sprintf("v%d", version), func(t *testing.T) {
			cfg := Config{
				Period:             10,
				WorkerSleepMS:      1,
				ReasonIDs:          []uint32{7},
				MaxReasons:         8,
				ScratchBytes:       1024,
				HardwareBytes:      1024,
				MaxNativeBytes:     2048,
				PCsPerBuffer:       1,
				BufferCount:        3, // Ring buffer with 3 drain buffers
				MaxOutputBytes:     608,
				MaxReads:           10,
				MaxNames:           10,
				MaxNameBytes:       20,
				MaxTotalNameBytes:  100,
				CollectionMode:     cupti.PcSamplingCollectionModeContinuous,
				AllowSerialization: false,
			}

			var (
				starts, stops uint64
				queryCalls    uint64
				saved         []cupti.PCSamplingConfigurationInfo
			)
			api := samplingControlAPI{
				device:  func(cupti.Context) (uint32, error) { return 0, nil },
				version: func() (uint32, error) { return version, nil },
				set: func(p *cupti.PCSamplingConfigurationInfoParams) error {
					attrs := unsafe.Slice((*cupti.PCSamplingConfigurationInfo)(nativePointer(p.PPCSamplingConfigurationInfo)), int(p.NumAttributes))
					for i := range attrs {
						attrs[i].AttributeStatus = 0
					}
					saved = append(saved, attrs...)
					return nil
				},
				get: func(p *cupti.PCSamplingConfigurationInfoParams) error {
					attrs := unsafe.Slice((*cupti.PCSamplingConfigurationInfo)(nativePointer(p.PPCSamplingConfigurationInfo)), int(p.NumAttributes))
					for i := range attrs {
						for _, v := range saved {
							if attrs[i].AttributeType == v.AttributeType {
								attrs[i] = v
							}
						}
					}
					return nil
				},
				enable:  func(*cupti.PCSamplingEnableParams) error { return nil },
				start:   func(*cupti.PCSamplingStartParams) error { starts++; return nil },
				stop:    func(*cupti.PCSamplingStopParams) error { stops++; return nil },
				disable: func(*cupti.PCSamplingDisableParams) error { return nil },
			}

			inventory := func(cupti.Context, int) (ReasonInventory, error) {
				return ReasonInventory{CUPTIVersion: version, Reasons: []StallReason{{Index: 7, Name: "warp"}}}, nil
			}

			names := func(entries, nameBytes, totalBytes int) (*samplingNames, error) {
				return newSamplingNames(entries, nameBytes, totalBytes, func(uintptr, int) (string, error) {
					return "kernel_fn", nil
				}, func(uintptr) {})
			}

			query := func(p *cupti.PCSamplingGetDataParams) error {
				queryCalls++
				data := (*cupti.PCSamplingData)(nativePointer(p.PcSamplingData))
				pcs := unsafe.Slice((*cupti.PCSamplingPCData)(nativePointer(data.PPcData)), int(data.CollectNumPcs))
				data.TotalNumPcs = 1
				data.TotalSamples = queryCalls
				pcs[0].FunctionName = 0x1000
				pcs[0].StallReasonCount = 1
				reasons := unsafe.Slice((*cupti.PCSamplingStallReason)(nativePointer(pcs[0].StallReason)), 1)
				reasons[0] = cupti.PCSamplingStallReason{PcSamplingStallReasonIndex: 7, Samples: 1}
				return nil
			}

			s, err := startSession(5, cfg, api, inventory, names, query)
			if err != nil {
				t.Fatalf("startSession: %v", err)
			}
			defer s.Close()

			rb := s.RingBuffer()
			if rb == nil {
				t.Fatal("expected non-nil ring buffer on session with BufferCount=3")
			}
			if rb.Stats().Capacity != 3 {
				t.Fatalf("expected ring buffer capacity 3, got %d", rb.Stats().Capacity)
			}

			var receivedBatches []Batch
			var batchMu sync.Mutex

			stopDrain, err := s.DrainConcurrent(func(b Batch) error {
				batchMu.Lock()
				receivedBatches = append(receivedBatches, b)
				batchMu.Unlock()
				return nil
			})
			if err != nil {
				t.Fatalf("DrainConcurrent: %v", err)
			}

			// Producer loop using Handoff
			const iterations = 5
			for i := 0; i < iterations; i++ {
				if err := s.Handoff(); err != nil {
					t.Fatalf("Handoff iteration %d: %v", i, err)
				}
				time.Sleep(2 * time.Millisecond)
			}

			if err := stopDrain(); err != nil {
				t.Fatalf("stopDrain: %v", err)
			}

			batchMu.Lock()
			count := len(receivedBatches)
			batchMu.Unlock()

			if count != iterations {
				t.Fatalf("expected %d batches, got %d", iterations, count)
			}

			// Clean close
			if err := s.Close(); err != nil {
				t.Fatalf("Close: %v", err)
			}

			info := s.Info()
			if !info.Closed {
				t.Fatal("expected Info.Closed=true")
			}
			if info.RingStats.Capacity != 3 {
				t.Fatalf("expected RingStats.Capacity=3, got %d", info.RingStats.Capacity)
			}
		})
	}
}

func ExampleRingBuffer() {
	// Allocate a zero-copy circular ring buffer with 3 pinned buffers.
	rb, err := NewRingBuffer(3, 10, 4, 4096)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	defer rb.Close()

	// Acquire next free buffer without blocking.
	slot, err := rb.Acquire()
	if err != nil {
		fmt.Println("acquire error:", err)
		return
	}

	// Hand off filled buffer to consumer queue and acquire next free buffer.
	next, err := rb.Handoff(slot)
	if err != nil {
		fmt.Println("handoff error:", err)
		return
	}

	// Consumer receives and processes filled buffer.
	consumed, err := rb.Consume()
	if err != nil {
		fmt.Println("consume error:", err)
		return
	}

	// Release drained buffer back to free queue with zero allocations.
	rb.Release(consumed)
	rb.Release(next)

	fmt.Println("available:", rb.Stats().Available)
	// Output: available: 3
}
