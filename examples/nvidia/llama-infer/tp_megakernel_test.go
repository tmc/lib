package main

import (
	"fmt"
	"math/rand/v2"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/egraph/hopper"
	"github.com/tmc/lib/nvidia/x/egraph/tensorparallel"
)

func testTPMegakernelModelConfig() ModelConfig {
	return ModelConfig{
		ModelType:         "llama",
		HiddenSize:        1024,
		IntermediateSize:  1024,
		NumHiddenLayers:   2,
		NumAttentionHeads: 16,
		NumKeyValueHeads:  4,
		ExplicitHeadDim:   64,
		VocabSize:         256,
		RmsNormEps:        1e-5,
		RopeTheta:         10000.0,
		MaxSeqLen:         32,
	}
}

func TestTPMegakernelGroupAndParity(t *testing.T) {
	cfg := testTPMegakernelModelConfig()
	w := GenerateDeterministicWeights(&cfg)
	for i := range w.Layers {
		qDim := cfg.NumAttentionHeads * cfg.HeadDim()
		kvDim := cfg.NumKeyValueHeads * cfg.HeadDim()
		w.Layers[i].QBias = make([]float32, qDim)
		w.Layers[i].KBias = make([]float32, kvDim)
		w.Layers[i].VBias = make([]float32, kvDim)
		for j := range qDim {
			w.Layers[i].QBias[j] = float32(j%11-5) * 0.01
		}
		for j := range kvDim {
			w.Layers[i].KBias[j] = float32(j%7-3) * 0.01
			w.Layers[i].VBias[j] = float32(j%5-2) * 0.01
		}
	}

	rng := rand.New(rand.NewPCG(77, 88))
	x := make([]float32, cfg.HiddenSize)
	for i := range x {
		x[i] = float32(rng.NormFloat64() * 0.25)
	}
	fullAttn := make([]float32, cfg.NumAttentionHeads*cfg.HeadDim())
	for i := range fullAttn {
		fullAttn[i] = float32(rng.NormFloat64() * 0.25)
	}

	for _, tp := range []int{2, 4} {
		for _, fusion := range []tensorparallel.Fusion{
			tensorparallel.FuseReduceAdd,
			tensorparallel.FuseMultimem,
		} {
			for _, qkvBias := range []bool{false, true} {
				name := fmt.Sprintf("tp%d_%s_bias_%v", tp, fusion, qkvBias)
				t.Run(name, func(t *testing.T) {
					if err := verifyTPMegakernelParity(&cfg, &w.Layers[0], &w.Layers[1], x, fullAttn, tp, fusion, qkvBias); err != nil {
						t.Fatalf("verifyTPMegakernelParity: %v", err)
					}

					grp, err := newTPMegakernelGroup(&cfg, tpMegakernelOptions{
						TP:      tp,
						Prefix:  fmt.Sprintf("llama_tp%d", tp),
						Fusion:  fusion,
						Target:  "sm_100a",
						QKVBias: qkvBias,
					})
					if err != nil {
						t.Fatalf("newTPMegakernelGroup: %v", err)
					}
					srcs, err := grp.Sources()
					if err != nil {
						t.Fatalf("Sources(): %v", err)
					}
					if len(srcs) != tp {
						t.Fatalf("len(srcs) = %d, want %d", len(srcs), tp)
					}
					for r, src := range srcs {
						for _, sub := range []string{
							"eg_ready_signal_sys",
							"eg_ready_wait_sys",
							"fence.acq_rel.sys;",
							"ld.acquire.sys.global.u64",
						} {
							if !strings.Contains(src, sub) {
								t.Errorf("rank %d source missing %q", r, sub)
							}
						}
						if fusion == tensorparallel.FuseMultimem {
							for _, sub := range []string{
								"multimem.red.release.sys.global.add.u64",
								"multimem.red.relaxed.sys.global.add.f32",
							} {
								if !strings.Contains(src, sub) {
									t.Errorf("rank %d multimem source missing %q", r, sub)
								}
							}
						}
					}
				})
			}
		}
	}
}

func TestTPMegakernelLlama8BMultiLayer(t *testing.T) {
	cfg := llama3_8B
	cfg.NumHiddenLayers = 4

	for _, tp := range []int{2, 4, 8} {
		grp, err := newTPMegakernelGroup(&cfg, tpMegakernelOptions{
			TP:          tp,
			Fusion:      tensorparallel.FuseMultimem,
			Target:      "sm_90a",
			BarrierMode: hopper.SysBarrierMultimem,
		})
		if err != nil {
			t.Fatalf("TP=%d newTPMegakernelGroup: %v", tp, err)
		}
		if len(grp.ranks) != tp {
			t.Fatalf("TP=%d len(ranks) = %d", tp, len(grp.ranks))
		}
		for r, rp := range grp.ranks {
			if len(rp.Boundaries) != cfg.NumHiddenLayers-1 {
				t.Fatalf("TP=%d rank %d len(Boundaries) = %d, want %d",
					tp, r, len(rp.Boundaries), cfg.NumHiddenLayers-1)
			}
		}
	}
}

func TestTPMegakernelErrors(t *testing.T) {
	cfg := testTPMegakernelModelConfig()
	if _, err := newTPMegakernelGroup(nil, tpMegakernelOptions{TP: 2}); err == nil {
		t.Error("nil config: expected error")
	}
	cfg1 := cfg
	cfg1.NumHiddenLayers = 1
	if _, err := newTPMegakernelGroup(&cfg1, tpMegakernelOptions{TP: 2}); err == nil {
		t.Error("1 layer config: expected error")
	}
	if _, err := newTPMegakernelGroup(&cfg, tpMegakernelOptions{TP: 0}); err == nil {
		t.Error("TP=0: expected error")
	}
	if _, err := newTPMegakernelGroup(&cfg, tpMegakernelOptions{TP: 2, Rows: -1}); err == nil {
		t.Error("negative rows: expected error")
	}

	grp, err := newTPMegakernelGroup(&cfg, tpMegakernelOptions{TP: 2})
	if err != nil {
		t.Fatal(err)
	}
	if _, _, _, _, _, err := grp.evalCrossLayerHost(nil, nil, nil, nil); err == nil {
		t.Error("nil weights: expected error")
	}
	var emptyGrp *tpMegakernelGroup
	if _, err := emptyGrp.Sources(); err == nil {
		t.Error("nil group Sources(): expected error")
	}
}
