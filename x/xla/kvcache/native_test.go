package kvcache

import (
	"context"
	"encoding/binary"
	"errors"
	"math"
	"os"
	"testing"

	"github.com/tmc/lib/x/xla"
)

// cancelOnCompletion cancels at the first post-dispatch context check.
// It exercises cancellation after native execution without a timing race.
type cancelOnCompletion struct {
	context.Context
	cancel context.CancelFunc
	calls  int
}

func (c *cancelOnCompletion) Err() error {
	c.calls++
	if c.calls == 3 {
		c.cancel()
	}
	return c.Context.Err()
}

func TestNativeCache(t *testing.T) {
	path := os.Getenv("XLA_PJRT_CPU_PLUGIN")
	if path == "" {
		t.Skip("set XLA_PJRT_CPU_PLUGIN to a CPU PJRT library")
	}
	engine, err := xla.NewEngine(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := engine.Close(); err != nil {
			t.Error(err)
		}
	})
	devices, err := engine.Devices()
	if err != nil {
		t.Fatal(err)
	}
	if len(devices) == 0 {
		t.Fatalf("got %d devices", len(devices))
	}
	cfg := Config{BatchShape: []int64{2}, Capacity: 6, KeySize: 1, ValueSize: 1, ElementType: xla.F32}
	cache, err := New(engine, cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := cache.Close(); err != nil {
			t.Error(err)
		}
	})
	for _, device := range devices {
		if device == cache.device {
			continue
		}
		foreign, err := engine.NewBufferFromHost(make([]byte, 8), xla.F32, []int64{2, 1, 1}, device)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := foreign.Close(); err != nil {
				t.Error(err)
			}
		})
		if _, err := cache.Append(context.Background(), foreign, foreign, foreign); err == nil {
			t.Fatal("accepted input on another device")
		}
		if cache.Len() != 0 || len(cache.plans) != 0 {
			t.Fatal("device rejection changed cache")
		}
		break
	}
	cfg.BatchShape[0] = 99
	if cache.config.BatchShape[0] != 2 {
		t.Fatal("config aliases caller batch shape")
	}
	upload := func(values []float32, n int64) *xla.Buffer {
		t.Helper()
		data := make([]byte, 4*len(values))
		for i, v := range values {
			binary.LittleEndian.PutUint32(data[4*i:], math.Float32bits(v))
		}
		b, err := engine.NewBufferFromHost(data, xla.F32, []int64{2, n, 1}, cache.device)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := b.Close(); err != nil {
				t.Error(err)
			}
		})
		return b
	}
	// Q=K=0 makes causal attention the arithmetic mean of the cached prefix.
	appendChunk := func(ctx context.Context, values, want []float32, n int64) error {
		t.Helper()
		q := upload(make([]float32, 2*n), n)
		k := upload(make([]float32, 2*n), n)
		v := upload(values, n)
		out, err := cache.Append(ctx, q, k, v)
		if err != nil {
			return err
		}
		defer func() {
			if err := out.Close(); err != nil {
				t.Error(err)
			}
		}()
		data := make([]byte, 4*len(want))
		if err := out.ToHost(data); err != nil {
			t.Fatal(err)
		}
		for i, w := range want {
			got := math.Float32frombits(binary.LittleEndian.Uint32(data[4*i:]))
			if math.IsNaN(float64(got)) || math.Abs(float64(got-w)) > 1e-5 {
				t.Errorf("output %d = %g, want %g", i, got, w)
			}
		}
		return nil
	}
	if err := appendChunk(context.Background(), []float32{1, 3, 10, 30}, []float32{1, 2, 10, 20}, 2); err != nil {
		t.Fatal(err)
	}
	if err := appendChunk(context.Background(), []float32{5, 50}, []float32{3, 30}, 1); err != nil {
		t.Fatal(err)
	}
	decode := cache.plans[1]
	if err := appendChunk(context.Background(), []float32{7, 70}, []float32{4, 40}, 1); err != nil {
		t.Fatal(err)
	}
	if cache.plans[1] != decode || cache.Len() != 4 {
		t.Fatal("decode graph not reused or length incorrect")
	}
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if err := appendChunk(canceled, []float32{999, 999}, nil, 1); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancel: %v", err)
	}
	during, cancel := context.WithCancel(context.Background())
	defer cancel()
	controlled := &cancelOnCompletion{Context: during, cancel: cancel}
	if err := appendChunk(controlled, []float32{999, 999}, nil, 1); !errors.Is(err, context.Canceled) {
		t.Fatalf("post-dispatch cancel: %v", err)
	}
	if cache.Len() != 4 {
		t.Fatal("cancellation advanced history")
	}
	if err := appendChunk(context.Background(), []float32{9, 90}, []float32{5, 50}, 1); err != nil {
		t.Fatal(err)
	}
	if err := appendChunk(context.Background(), []float32{1, 2, 1, 2}, nil, 2); err == nil {
		t.Fatal("accepted capacity overflow")
	}
	if cache.Len() != 5 {
		t.Fatal("capacity error changed history")
	}
	if err := cache.Reset(); err != nil {
		t.Fatal(err)
	}
	if err := appendChunk(context.Background(), []float32{4, 40}, []float32{4, 40}, 1); err != nil {
		t.Fatal(err)
	}
	if cache.plans[1] != decode {
		t.Fatal("reset discarded decode graph")
	}
	if err := cache.Reset(); err != nil {
		t.Fatal(err)
	}
	if err := appendChunk(context.Background(), []float32{2, 4, 6, 20, 40, 60}, []float32{2, 3, 4, 20, 30, 40}, 3); err != nil {
		t.Fatal(err)
	}
	if len(cache.plans) != 2 || cache.plans[2] != nil || cache.plans[3] == nil {
		t.Fatal("prefill program cache is not bounded")
	}
	if err := cache.Close(); err != nil {
		t.Fatal(err)
	}
	if err := cache.Reset(); err == nil {
		t.Fatal("closed cache reset")
	}
}
