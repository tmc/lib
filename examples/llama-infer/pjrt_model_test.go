package main

import (
	"context"
	"errors"
	"math"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/x/xla"
)

func pjrtFixture() (*ModelConfig, *ModelWeights) {
	c := DefaultModelConfig()
	c.HiddenSize, c.IntermediateSize = 8, 11
	c.NumAttentionHeads, c.NumKeyValueHeads = 2, 1
	c.VocabSize, c.MaxSeqLen = 13, 6
	w := GenerateDeterministicWeights(&c)
	n := 0
	fill := func(x []float32, norm bool) {
		n++
		for i := range x {
			x[i] = float32(math.Sin(float64(i*17+n*7))) * 0.2
			if norm {
				x[i] += 1
			}
		}
	}
	fill(w.EmbedTokens, false)
	fill(w.Norm, true)
	fill(w.LMHead, false)
	for i := range w.Layers {
		l := &w.Layers[i]
		fill(l.InputNorm, true)
		fill(l.PostNorm, true)
		for _, x := range [][]float32{l.WQ, l.WK, l.WV, l.WO, l.WGate, l.WUp, l.WDown} {
			fill(x, false)
		}
		l.QBias, l.KBias, l.VBias = make([]float32, 8), make([]float32, 4), make([]float32, 4)
		for _, x := range [][]float32{l.QBias, l.KBias, l.VBias} {
			fill(x, false)
		}
	}
	return &c, w
}

func TestPJRTMappingValidation(t *testing.T) {
	for _, tt := range []struct {
		name   string
		change func(*ModelConfig, *ModelWeights)
		want   string
	}{
		{"layers", func(c *ModelConfig, w *ModelWeights) { c.NumHiddenLayers++ }, "layer count"},
		{"head division", func(c *ModelConfig, w *ModelWeights) { c.NumAttentionHeads = 3 }, "head dimensions"},
		{"negative head", func(c *ModelConfig, w *ModelWeights) { c.ExplicitHeadDim = -1 }, "head dimensions"},
		{"untied missing", func(c *ModelConfig, w *ModelWeights) { w.LMHead = nil }, "missing untied LM head"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			c, w := pjrtFixture()
			tt.change(c, w)
			if _, err := newPJRTModel(context.Background(), nil, c, w, float64(c.RopeTheta), xla.F32); err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("got %v, want %s", err, tt.want)
			}
		})
	}
}

func TestNativePJRTModel(t *testing.T) {
	path := os.Getenv("XLA_PJRT_CPU_PLUGIN")
	if path == "" {
		t.Skip("set XLA_PJRT_CPU_PLUGIN to a CPU PJRT library")
	}
	engine, err := xla.NewEngine(path)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := engine.Close(); err != nil {
			t.Error(err)
		}
	}()
	for _, tied := range []bool{false, true} {
		name := "untied"
		if tied {
			name = "tied"
		}
		t.Run(name, func(t *testing.T) {
			c, w := pjrtFixture()
			c.TieWordEmbeddings = tied
			if tied {
				w.LMHead = w.EmbedTokens
			}
			p, err := newPJRTModel(context.Background(), engine, c, w, float64(c.RopeTheta), xla.F32)
			if err != nil {
				t.Fatal(err)
			}
			defer func() {
				if err := p.close(); err != nil {
					t.Error(err)
				}
			}()
			if _, err := p.forward(context.Background(), nil, 0); err == nil || !strings.Contains(err.Error(), "chunk") {
				t.Fatalf("empty input: %v", err)
			}
			if !slices.Equal(p.model.Lengths(), []int64{0}) {
				t.Fatal("empty input changed history")
			}
			if _, err := p.readLogits(); err == nil {
				t.Fatal("initial logits available")
			}
			compare := func(got, want []float32) {
				t.Helper()
				if len(got) != len(want) {
					t.Fatal("logit shape")
				}
				for i, v := range got {
					if math.Abs(float64(v-want[i])) > 2e-5*(1+math.Abs(float64(want[i]))) {
						t.Fatalf("logit %d=%g, want %g", i, v, want[i])
					}
				}
			}
			kv := NewKVCacheState(c.NumHiddenLayers, c.MaxSeqLen, c.NumKeyValueHeads*c.HeadDim())
			var want []float32
			for i, id := range []int32{3, 1, 9} {
				want = Forward(w, c, kv, id, i)
			}
			got, err := p.forward(context.Background(), []int32{3, 1, 9}, 0)
			if err != nil {
				t.Fatal(err)
			}
			compare(got, want)
			got[0] = 123
			saved, err := p.readLogits()
			if err != nil {
				t.Fatal(err)
			}
			compare(saved, want)
			saved[0] = 456
			ctx, cancel := context.WithCancel(context.Background())
			cancel()
			if _, err := p.forward(ctx, []int32{2}, 3); !errors.Is(err, context.Canceled) {
				t.Fatalf("cancel: %v", err)
			}
			if _, err := p.forward(context.Background(), []int32{2}, 2); err == nil {
				t.Fatal("wrong position accepted")
			}
			if !slices.Equal(p.model.Lengths(), []int64{3}) {
				t.Fatal("validation changed history")
			}
			want = Forward(w, c, kv, 2, 3)
			got, err = p.forward(context.Background(), []int32{2}, 3)
			if err != nil {
				t.Fatal(err)
			}
			compare(got, want)
			got[0] = 123
			saved, err = p.readLogits()
			if err != nil {
				t.Fatal(err)
			}
			compare(saved, want)
			saved[0] = 456
			saved, err = p.readLogits()
			if err != nil {
				t.Fatal(err)
			}
			compare(saved, want)
			if err := p.reset(); err != nil {
				t.Fatal(err)
			}
			if _, err := p.readLogits(); err == nil {
				t.Fatal("stale logits after reset")
			}
			kv = NewKVCacheState(c.NumHiddenLayers, c.MaxSeqLen, c.NumKeyValueHeads*c.HeadDim())
			want = Forward(w, c, kv, 5, 0)
			got, err = p.forward(context.Background(), []int32{5}, 0)
			if err != nil {
				t.Fatal(err)
			}
			compare(got, want)
			// Four Err checks precede adapter readback; the fifth reports cancellation
			// after the decoder committed. Retrying this adapter must be disallowed.
			late := &pjrtLateCancel{Context: context.Background()}
			if _, err := p.forward(late, []int32{1}, 1); !errors.Is(err, context.Canceled) {
				t.Fatalf("late cancel: %v", err)
			}
			if p.model != nil {
				t.Fatal("adapter retained advanced history after readback cancellation")
			}
			if _, err := p.readLogits(); err == nil {
				t.Fatal("closed adapter exposed logits")
			}
		})
	}
}

type pjrtLateCancel struct {
	context.Context
	calls int
}

func (c *pjrtLateCancel) Err() error {
	c.calls++
	if c.calls >= 5 {
		return context.Canceled
	}
	return nil
}

func TestNativePJRTLogitShape(t *testing.T) {
	path := os.Getenv("XLA_PJRT_CPU_PLUGIN")
	if path == "" {
		t.Skip("set XLA_PJRT_CPU_PLUGIN to a CPU PJRT library")
	}
	engine, err := xla.NewEngine(path)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := engine.Close(); err != nil {
			t.Error(err)
		}
	}()
	c, w := pjrtFixture()
	p, err := newPJRTModel(context.Background(), engine, c, w, float64(c.RopeTheta), xla.F32)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := p.close(); err != nil {
			t.Error(err)
		}
	}()
	// ToHost permits an oversized destination. The adapter must reject a
	// mismatched output shape rather than accepting an unread zero-filled tail.
	p.vocab++
	if _, err := p.forward(context.Background(), []int32{3}, 0); err == nil || !strings.Contains(err.Error(), "logit shape") {
		t.Fatalf("got %v, want logit shape error", err)
	}
	if p.model != nil {
		t.Fatal("adapter retained history after invalid output")
	}
}
