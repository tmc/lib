package cuptipcsampling

import (
	"errors"
	"sync/atomic"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/cupti"
)

func TestConcurrentConsumerRetainsOwnership(t *testing.T) {
	entered, release := make(chan struct{}), make(chan struct{})
	failure := errors.New("consumer rejected batch")
	var disables atomic.Uint32
	api := testPublicControlAPI()
	api.disable = func(*cupti.PCSamplingDisableParams) error { disables.Add(1); return nil }
	cfg := testPublicConfig()
	cfg.BufferCount, cfg.MaxOutputBytes = 2, 456
	s, err := startSession(5, cfg, api, testPublicInventory, testPublicNames, testPublicOnePC)
	if err != nil {
		t.Fatal(err)
	}
	stop, err := s.DrainConcurrent(func(Batch) error {
		close(entered)
		<-release
		return failure
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Handoff(); err != nil {
		t.Fatal(err)
	}
	select {
	case <-entered:
	case <-time.After(time.Second):
		t.Fatal("consumer did not receive the batch")
	}
	if err := s.Close(); err == nil || disables.Load() != 0 || s.Info().Closed {
		t.Fatalf("close released beneath an active consumer: err=%v disables=%d info=%+v", err, disables.Load(), s.Info())
	}
	close(release)
	if err := stop(); !errors.Is(err, failure) {
		t.Fatalf("stop lost consumer error: %v", err)
	}
	if err := stop(); !errors.Is(err, failure) {
		t.Fatalf("repeated stop did not finish with the same error: %v", err)
	}
	if err := s.Close(); !errors.Is(err, failure) || disables.Load() != 1 || !s.Info().Closed || !s.Info().Failed {
		t.Fatalf("cleanup lost sticky failure or ownership: err=%v disables=%d info=%+v", err, disables.Load(), s.Info())
	}
}

func TestRingPressureIsNotSampleLossCount(t *testing.T) {
	cfg := testPublicConfig()
	cfg.BufferCount, cfg.MaxOutputBytes, cfg.MaxReads = 2, 456, 3
	s, err := startSession(5, cfg, testPublicControlAPI(), testPublicInventory, testPublicNames, testPublicEmptyQuery)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	for i := 0; i < 2; i++ {
		if err := s.Handoff(); err != nil {
			t.Fatal(err)
		}
	}
	if err := s.Handoff(); !errors.Is(err, ErrRingBufferFull) {
		t.Fatalf("ring pressure not reported: %v", err)
	}
	stats := s.Info().RingStats
	if stats.Overflows != 1 || stats.Drops != 0 {
		t.Fatalf("failed acquisition mixed events with hardware samples: %+v", stats)
	}
	if err := s.Close(); err == nil || s.Info().Closed {
		t.Fatalf("queued native buffers lost cleanup ownership: err=%v info=%+v", err, s.Info())
	}
	stop, err := s.DrainConcurrent(func(Batch) error { return nil })
	if err != nil {
		t.Fatal(err)
	}
	if err := stop(); err != nil {
		t.Fatal(err)
	}
}
