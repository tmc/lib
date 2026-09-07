package main

import (
	"context"
	"math"
	"os"
	"testing"

	"github.com/tmc/lib/x/xla"
)

func TestPJRTEncodedLoad(t *testing.T) {
	for _, precision := range []string{"f16", "bf16"} {
		t.Run(precision, func(t *testing.T) {
			dir, _ := writeRetainedTestCheckpoint(t, "BF16")
			p, tok, err := preparePJRTModel(Config{ModelPath: dir, MaxSeqLen: 6, Precision: precision})
			if err != nil {
				t.Fatal(err)
			}
			if tok != nil || p.retained == nil || p.weights != nil {
				t.Fatal("half checkpoint widened or acquired a tokenizer")
			}
			if p.retained.EmbedTokens.DType != "BF16" {
				t.Fatal("source dtype changed")
			}
			if p.receipt().Verdict != "accepted" {
				t.Fatal("missing receipt")
			}
		})
	}
}

func TestNativePJRTEncodedModel(t *testing.T) {
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
	for _, tt := range []struct {
		name, source string
		element      xla.ElementType
		tol          float64
	}{
		{"bf16", "BF16", xla.BF16, 0.08}, {"f16", "F16", xla.F16, 0.01},
		{"bf16 from f16", "F16", xla.BF16, 0.08}, {"f16 from f32", "F32", xla.F16, 0.01},
	} {
		t.Run(tt.name, func(t *testing.T) {
			dir, _ := writeRetainedTestCheckpoint(t, tt.source)
			p, err := loadPJRTCheckpoint(dir, 6, tt.element)
			if err != nil {
				t.Fatal(err)
			}
			if p.retained == nil || p.weights != nil {
				t.Fatal("checkpoint widened before upload")
			}
			model, err := p.newModel(context.Background(), engine)
			if err != nil {
				t.Fatal(err)
			}
			defer func() {
				if err := model.close(); err != nil {
					t.Error(err)
				}
			}()
			if model.weightBytes != p.retained.PayloadBytes() {
				t.Fatalf("uploaded bytes %d, want %d", model.weightBytes, p.retained.PayloadBytes())
			}
			// Widen only the independent test reference, after production model creation.
			w := p.retained.Float32()
			c := &p.config
			kv := NewKVCacheState(c.NumHiddenLayers, c.MaxSeqLen, c.NumKeyValueHeads*c.HeadDim())
			for pos, id := range []int32{1, 2, 3, 4} {
				want := Forward(w, c, kv, id, pos)
				got, err := model.forward(context.Background(), []int32{id}, pos)
				if err != nil {
					t.Fatal(err)
				}
				for i, v := range want {
					delta := math.Abs(float64(got[i] - v))
					if math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) || delta > tt.tol*(1+math.Abs(float64(v))) {
						t.Fatalf("position %d logit %d: %g, want %g", pos, i, got[i], v)
					}
				}
			}
		})
	}
}
