package main

import (
	"errors"
	"io"
	"sync"
	"sync/atomic"
	"testing"
)

type cudaStreamingTestConsumer struct{ close func() error }

func (c *cudaStreamingTestConsumer) Close() error { return c.close() }

func TestCUDAStreamingRequestOrder(t *testing.T) {
	var calls []string
	o := &cudaStreamingCoordinator{}
	q := o.request()
	consumer := &cudaStreamingTestConsumer{func() error { calls = append(calls, "completion"); return nil }}
	err := q.run(func() (io.Closer, error) { calls = append(calls, "selection"); return consumer, nil }, func(owner io.Closer) error {
		if owner != consumer {
			t.Fatal("selection owner replaced")
		}
		calls = append(calls, "gate", "weighted-swiglu", "down")
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"selection", "gate", "weighted-swiglu", "down", "completion"}
	if len(calls) != len(want) {
		t.Fatal(calls)
	}
	for i := range want {
		if calls[i] != want[i] {
			t.Fatal(calls)
		}
	}
	if q.pending != nil || o.active != nil {
		t.Fatal("successful consumer retained")
	}
}

func TestCUDAStreamingRequestRetainsConsumer(t *testing.T) {
	for _, stage := range []string{"upload", "gate", "down", "completion"} {
		t.Run(stage, func(t *testing.T) {
			cause := errors.New(stage)
			cleanup := errors.New("completion uncertain")
			o := &cudaStreamingCoordinator{}
			q := o.request()
			failClose := true
			consumer := &cudaStreamingTestConsumer{func() error {
				if failClose {
					return cleanup
				}
				return nil
			}}
			consumeCalls := 0
			err := q.run(func() (io.Closer, error) {
				if stage == "upload" {
					return consumer, cause
				}
				return consumer, nil
			}, func(io.Closer) error {
				consumeCalls++
				if stage != "completion" {
					return cause
				}
				return nil
			})
			if !errors.Is(err, cleanup) || (stage != "completion" && !errors.Is(err, cause)) {
				t.Fatal(err)
			}
			if q.pending != consumer || o.active != q {
				t.Fatal("uncertain consumer owner discarded")
			}
			if stage == "upload" && consumeCalls != 0 {
				t.Fatal("failed selection consumed")
			}
			other := o.request()
			if err := other.run(func() (io.Closer, error) { t.Fatal("cache reused before completion"); return nil, nil }, func(io.Closer) error { return nil }); err == nil {
				t.Fatal("concurrent retained-reader admission")
			}
			if err := q.Close(); !errors.Is(err, cleanup) {
				t.Fatal(err)
			}
			if q.pending != consumer || o.active != q {
				t.Fatal("Close lost retryable reader")
			}
			failClose = false
			if err := q.Close(); err != nil {
				t.Fatal(err)
			}
			if q.pending != nil || o.active != nil {
				t.Fatal("completion retry did not release")
			}
			if err := q.run(func() (io.Closer, error) { return consumer, nil }, func(io.Closer) error { return nil }); err == nil {
				t.Fatal("closed request readmitted")
			}
		})
	}
}

func TestCUDAStreamingCoordinatorCloseOrder(t *testing.T) {
	o := &cudaStreamingCoordinator{}
	q := o.request()
	consumerFault := true
	cacheFault := true
	cacheCloses := 0
	consumer := &cudaStreamingTestConsumer{func() error {
		if consumerFault {
			return errors.New("reader pending")
		}
		return nil
	}}
	o.closeCache = func() error {
		cacheCloses++
		if cacheFault {
			return errors.New("cache retained")
		}
		return nil
	}
	if err := q.run(func() (io.Closer, error) { return consumer, nil }, func(io.Closer) error { return nil }); err == nil {
		t.Fatal("consumer fault ignored")
	}
	if err := o.Close(); err == nil || cacheCloses != 0 || o.active != q {
		t.Fatal("cache freed before consumer completion")
	}
	consumerFault = false
	if err := o.Close(); err == nil || cacheCloses != 1 || o.closeCache == nil || o.closed {
		t.Fatal("cache cleanup owner lost")
	}
	cacheFault = false
	if err := o.Close(); err != nil {
		t.Fatal(err)
	}
	if !o.closed || o.closeCache != nil || cacheCloses != 2 {
		t.Fatal("cache close retry incomplete")
	}
}

func TestCUDAStreamingRequestNilSelection(t *testing.T) {
	o := &cudaStreamingCoordinator{}
	q := o.request()
	if err := q.run(func() (io.Closer, error) { return nil, nil }, func(io.Closer) error { t.Fatal("nil selection consumed"); return nil }); err == nil {
		t.Fatal("nil selection accepted")
	}
	if o.active != nil || q.failed == nil {
		t.Fatal("nil selection did not close request admission")
	}
	if err := q.reset(); err != nil {
		t.Fatal(err)
	}
	if q.failed != nil {
		t.Fatal("successful reset did not clear request fault")
	}
}

func TestCUDAStreamingRequestFaultRequiresReset(t *testing.T) {
	o := &cudaStreamingCoordinator{}
	q := o.request()
	cause := errors.New("launch failed")
	consumer := &cudaStreamingTestConsumer{func() error { return nil }}
	if err := q.run(func() (io.Closer, error) { return consumer, nil }, func(io.Closer) error { return cause }); !errors.Is(err, cause) {
		t.Fatal(err)
	}
	if q.pending != nil || o.active != nil {
		t.Fatal("successful drain retained active reader")
	}
	if err := q.run(func() (io.Closer, error) { t.Fatal("faulted request reselected"); return consumer, nil }, func(io.Closer) error { return nil }); err == nil {
		t.Fatal("faulted request admitted")
	}
	if err := q.reset(); err != nil {
		t.Fatal(err)
	}
	if err := q.run(func() (io.Closer, error) { return consumer, nil }, func(io.Closer) error { return nil }); err != nil {
		t.Fatal(err)
	}
}

func TestCUDAStreamingRequestsSerialize(t *testing.T) {
	o := &cudaStreamingCoordinator{}
	var active, high atomic.Int32
	var wg sync.WaitGroup
	for range 20 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			q := o.request()
			err := q.run(func() (io.Closer, error) {
				n := active.Add(1)
				if n > high.Load() {
					high.Store(n)
				}
				return &cudaStreamingTestConsumer{func() error { active.Add(-1); return nil }}, nil
			}, func(io.Closer) error { return nil })
			if err != nil {
				t.Error(err)
			}
		}()
	}
	wg.Wait()
	if high.Load() != 1 || active.Load() != 0 {
		t.Fatal("overlapping cache reader")
	}
}
