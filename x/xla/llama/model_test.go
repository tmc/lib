package llama

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/tmc/lib/x/xla"
)

func TestValidation(t *testing.T) {
	for _, tt := range []struct {
		name   string
		change func(*Config, *Weights)
	}{
		{"batch", func(c *Config, w *Weights) { c.BatchSize = 0 }},
		{"odd head", func(c *Config, w *Weights) { c.HeadSize = 3 }},
		{"heads", func(c *Config, w *Weights) { c.QueryHeads = 3; c.KVHeads = 2 }},
		{"dimension overflow", func(c *Config, w *Weights) { c.HeadSize = math.MaxInt64 - 1 }},
		{"score overflow", func(c *Config, w *Weights) { c.Capacity = math.MaxInt64 / 100 }},
		{"epsilon", func(c *Config, w *Weights) { c.NormEpsilon = 0 }},
		{"theta", func(c *Config, w *Weights) { c.RopeTheta = math.Inf(1) }},
		{"angles", func(c *Config, w *Weights) {
			c.RopeTheta = math.SmallestNonzeroFloat64
			c.HeadSize = 64
			*w = fixture(*c, false)
		}},
		{"angles at zero", func(c *Config, w *Weights) {
			c.Capacity = 1
			c.RopeTheta = math.SmallestNonzeroFloat64
			c.HeadSize = 64
			*w = fixture(*c, false)
		}},
		{"layers", func(c *Config, w *Weights) { w.Layers = nil }},
		{"embedding length", func(c *Config, w *Weights) { w.Embedding = w.Embedding[:1] }},
		{"head empty", func(c *Config, w *Weights) { w.LMHead = []float32{} }},
		{"bias length", func(c *Config, w *Weights) { w.Layers[0].KeyBias = []float32{1} }},
		{"norm nonfinite", func(c *Config, w *Weights) { w.Norm[0] = float32(math.NaN()) }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			c := testConfig()
			w := fixture(c, false)
			tt.change(&c, &w)
			if _, err := describe(c, w); err == nil {
				t.Fatal("invalid model accepted")
			}
		})
	}
	c := testConfig()
	w := fixture(c, true)
	s, err := describe(c, w)
	if err != nil {
		t.Fatal(err)
	}
	if s.head != s.embedding {
		t.Fatal("tied head duplicated")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := New(ctx, nil, c, w); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled new: %v", err)
	}
}

func TestCommitCleanupFailure(t *testing.T) {
	m := &Model{history: []*xla.Buffer{new(xla.Buffer), new(xla.Buffer), new(xla.Buffer), new(xla.Buffer)}, parameters: []*xla.Buffer{new(xla.Buffer)}, lengths: []int64{2, 3}, plans: map[int64]*xla.Executable{1: new(xla.Executable)}}
	outputs := []*xla.Buffer{new(xla.Buffer), new(xla.Buffer), new(xla.Buffer), new(xla.Buffer), new(xla.Buffer)}
	sentinel := errors.New("destroy failure")
	released := 0
	out, err := m.commit(outputs, []int64{1, 1}, func(*xla.Buffer) error {
		released++
		if released == 2 {
			return sentinel
		}
		return nil
	})
	if out != nil || !errors.Is(err, sentinel) || released != 4 {
		t.Fatalf("commit=%v,%v releases=%d", out, err, released)
	}
	if !m.closed || m.history != nil || m.parameters != nil || m.Lengths() != nil || len(m.plans) != 0 {
		t.Fatal("failed commit retained model state")
	}
	if err := m.Reset(); err == nil {
		t.Fatal("reset reopened model")
	}
	if err := m.Close(); err != nil {
		t.Fatal(err)
	}
}
