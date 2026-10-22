package main

import (
	"fmt"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudakv"
)

const graphDecodeSteps = 10

func graphParityConfigs() []struct {
	name       string
	cfg        ModelConfig
	dtype      weightDType
	fused      bool
	fuseDecode bool
} {
	def := DefaultModelConfig()
	odd := DefaultModelConfig()
	odd.HiddenSize, odd.IntermediateSize, odd.NumAttentionHeads, odd.NumKeyValueHeads, odd.VocabSize = 18, 30, 3, 1, 50

	var cases []struct {
		name       string
		cfg        ModelConfig
		dtype      weightDType
		fused      bool
		fuseDecode bool
	}
	for _, m := range []struct {
		label string
		cfg   ModelConfig
	}{
		{"default", def},
		{"odd_rows", odd},
	} {
		for _, dtype := range []weightDType{weightF32, weightBF16} {
			for _, mode := range []struct {
				suffix     string
				fused      bool
				fuseDecode bool
			}{
				{"unfused", false, false},
				{"fused", true, false},
				{"fuse_decode", false, true},
			} {
				cases = append(cases, struct {
					name       string
					cfg        ModelConfig
					dtype      weightDType
					fused      bool
					fuseDecode bool
				}{
					name:       fmt.Sprintf("%s_%s_%s", m.label, dtype, mode.suffix),
					cfg:        m.cfg,
					dtype:      dtype,
					fused:      mode.fused,
					fuseDecode: mode.fuseDecode,
				})
			}
		}
	}
	return cases
}

// TestDecodeGraphParity runs eager prefill followed by 10 decode steps on one
// KV cache + scratchpad and DecodeGraph from the same prefill state on a
// second KV cache + scratchpad across DefaultModelConfig() and odd_rows, for
// both FP32 and BF16 weights, unfused, fuse-egraph, and fuse-decode. At every
// step it asserts bit-for-bit equality (math.Float32bits) on sp.Logits and
// identical output token IDs. For fuse-decode it also verifies bit-for-bit
// equality between weightBF16 and weightF32 on widened BF16 weights and
// numerical parity within logitTolerance = 1e-4 against CPU Forward.
func TestDecodeGraphParity(t *testing.T) {
	k := newForwardKit(t)
	prompt := []int32{1, 7, 3}
	f32FuseDecodeLogits := make(map[string][][]float32)

	for _, tt := range graphParityConfigs() {
		t.Run(tt.name, func(t *testing.T) {
			if tt.fused && (k.d.gm.fusedQKVFn == 0 || k.d.gm.fusedSwiGLUFn == 0 || k.d.gm.fusedQKVBF16Fn == 0 || k.d.gm.fusedSwiGLUBF16Fn == 0) {
				t.Fatal("fused kernels missing from module; the fused path would silently fall back")
			}
			if tt.fuseDecode && (k.d.gm.fusedQKVRoPEStoreFn == 0 || k.d.gm.fusedOutProjResFn == 0 || k.d.gm.fusedGateUpSwiGLUFn == 0 || k.d.gm.fusedDownProjResFn == 0 || k.d.gm.fusedQKVRoPEStoreBF16Fn == 0 || k.d.gm.fusedOutProjResBF16Fn == 0 || k.d.gm.fusedGateUpSwiGLUBF16Fn == 0 || k.d.gm.fusedDownProjResBF16Fn == 0) {
				t.Fatal("fused decode kernels missing from module; the fuse-decode path would silently fall back")
			}
			saveFlag, saveDecodeFlag, saveQualified := *fuseEGraphFlag, *fuseDecodeFlag, fusedKernelsQualified
			*fuseEGraphFlag, *fuseDecodeFlag, fusedKernelsQualified = tt.fused, tt.fuseDecode, true
			t.Cleanup(func() {
				*fuseEGraphFlag, *fuseDecodeFlag, fusedKernelsQualified = saveFlag, saveDecodeFlag, saveQualified
			})

			cfg := tt.cfg
			w := bf16Model(cfg, false)

			gw, err := uploadModel(k.d.dev, w, &cfg, tt.dtype)
			if gw != nil {
				defer func() {
					if err := gw.Free(); err != nil {
						t.Error(err)
					}
				}()
			}
			if err != nil {
				t.Fatal(err)
			}

			kvEager, err := cudakv.New(k.d.dev, cfg.NumHiddenLayers, cfg.NumKeyValueHeads, cfg.HeadDim(), cfg.MaxSeqLen)
			if kvEager != nil {
				defer func() {
					if err := kvEager.Close(); err != nil {
						t.Error(err)
					}
				}()
			}
			if err != nil {
				t.Fatal(err)
			}

			spEager, err := NewGPUScratchpad(k.d.dev, &cfg)
			if spEager != nil {
				defer func() {
					if err := spEager.Free(); err != nil {
						t.Error(err)
					}
				}()
			}
			if err != nil {
				t.Fatal(err)
			}

			kvGraph, err := cudakv.New(k.d.dev, cfg.NumHiddenLayers, cfg.NumKeyValueHeads, cfg.HeadDim(), cfg.MaxSeqLen)
			if kvGraph != nil {
				defer func() {
					if err := kvGraph.Close(); err != nil {
						t.Error(err)
					}
				}()
			}
			if err != nil {
				t.Fatal(err)
			}

			spGraph, err := NewGPUScratchpad(k.d.dev, &cfg)
			if spGraph != nil {
				defer func() {
					if err := spGraph.Free(); err != nil {
						t.Error(err)
					}
				}()
			}
			if err != nil {
				t.Fatal(err)
			}

			dg, err := NewDecodeGraph(k.d.dev, gw, &cfg, kvGraph, spGraph, k.d.gm, k.attnFn, k.sampleFn)
			if dg != nil {
				defer func() {
					if err := dg.Close(); err != nil {
						t.Error(err)
					}
				}()
			}
			if err != nil {
				t.Fatal(err)
			}

			hostKV := NewKVCacheState(cfg.NumHiddenLayers, cfg.MaxSeqLen, cfg.NumKeyValueHeads*cfg.HeadDim())
			var eagerTok, graphTok int32
			for pos, tok := range prompt {
				want := Forward(w, &cfg, hostKV, tok, pos)
				if err := k.d.dev.WithContext(func() error {
					var err error
					eagerTok, err = GPUForwardPass(gw, &cfg, kvEager, spEager, k.d.gm, k.attnFn, k.sampleFn, tok, pos)
					if err != nil {
						return err
					}
					graphTok, err = GPUForwardPass(gw, &cfg, kvGraph, spGraph, k.d.gm, k.attnFn, k.sampleFn, tok, pos)
					return err
				}); err != nil {
					t.Fatalf("prefill pos %d: %v", pos, err)
				}
				if eagerTok != graphTok {
					t.Fatalf("prefill pos %d: eager tok %d != graph tok %d", pos, eagerTok, graphTok)
				}
				if tt.fuseDecode {
					eagerLogits := k.d.download(spEager.Logits)
					if i := parityMismatch(eagerLogits, want, logitTolerance); i >= 0 {
						t.Fatalf("prefill pos %d logit %d: got %g want %g", pos, i, eagerLogits[i], want[i])
					}
					if hostNext := sampleArgmax(want); eagerTok != hostNext {
						t.Fatalf("prefill pos %d: device token %d != host token %d", pos, eagerTok, hostNext)
					}
				}
			}

			var stepLogits [][]float32
			for step := 0; step < graphDecodeSteps; step++ {
				pos := len(prompt) + step
				inTok := eagerTok
				want := Forward(w, &cfg, hostKV, inTok, pos)
				if err := k.d.dev.WithContext(func() error {
					var err error
					eagerTok, err = GPUForwardPass(gw, &cfg, kvEager, spEager, k.d.gm, k.attnFn, k.sampleFn, inTok, pos)
					if err != nil {
						return err
					}
					graphTok, err = dg.Step(pos, nil)
					return err
				}); err != nil {
					t.Fatalf("decode step %d (pos %d): %v", step, pos, err)
				}
				eagerLogits := k.d.download(spEager.Logits)
				graphLogits := k.d.download(spGraph.Logits)
				if j := bitMismatch(graphLogits, eagerLogits); j >= 0 {
					t.Fatalf("decode step %d (pos %d) logit %d: graph=%g (%#08x) want eager=%g (%#08x)",
						step, pos, j, graphLogits[j], uint32(graphLogits[j]), eagerLogits[j], uint32(eagerLogits[j]))
				}
				if graphTok != eagerTok {
					t.Fatalf("decode step %d (pos %d): graph token %d != eager token %d", step, pos, graphTok, eagerTok)
				}
				if tt.fuseDecode {
					if i := parityMismatch(graphLogits, want, logitTolerance); i >= 0 {
						t.Fatalf("decode step %d (pos %d) CPU logit %d: got %g want %g", step, pos, i, graphLogits[i], want[i])
					}
					if hostNext := sampleArgmax(want); graphTok != hostNext {
						t.Fatalf("decode step %d (pos %d): device token %d != host token %d", step, pos, graphTok, hostNext)
					}
					stepLogits = append(stepLogits, graphLogits)
				}
			}
			if tt.fuseDecode {
				shapeKey := fmt.Sprintf("%d_%d", cfg.HiddenSize, cfg.IntermediateSize)
				if tt.dtype == weightF32 {
					f32FuseDecodeLogits[shapeKey] = stepLogits
				} else if wantSteps, ok := f32FuseDecodeLogits[shapeKey]; ok {
					for step, got := range stepLogits {
						if j := bitMismatch(got, wantSteps[step]); j >= 0 {
							t.Fatalf("decode step %d BF16 vs widened F32 logit %d mismatch", step, j)
						}
					}
				}
			}
		})
	}
}

// TestDecodeGraphStalePosControl is the must-fail control for
// TestDecodeGraphParity: at decode step 2 it skips updating layer 0's RoPE
// position node (leaving it at step 1's position) while updating the other
// 3L-1 position nodes, and verifies that bitMismatch detects the resulting
// logit divergence at step 2.
func TestDecodeGraphStalePosControl(t *testing.T) {
	k := newForwardKit(t)
	cfg := DefaultModelConfig()
	w := bf16Model(cfg, false)
	prompt := []int32{1, 7, 3}

	gw, err := uploadModel(k.d.dev, w, &cfg, weightF32)
	if gw != nil {
		defer func() {
			if err := gw.Free(); err != nil {
				t.Error(err)
			}
		}()
	}
	if err != nil {
		t.Fatal(err)
	}

	kvEager, err := cudakv.New(k.d.dev, cfg.NumHiddenLayers, cfg.NumKeyValueHeads, cfg.HeadDim(), cfg.MaxSeqLen)
	if kvEager != nil {
		defer func() {
			if err := kvEager.Close(); err != nil {
				t.Error(err)
			}
		}()
	}
	if err != nil {
		t.Fatal(err)
	}

	spEager, err := NewGPUScratchpad(k.d.dev, &cfg)
	if spEager != nil {
		defer func() {
			if err := spEager.Free(); err != nil {
				t.Error(err)
			}
		}()
	}
	if err != nil {
		t.Fatal(err)
	}

	kvGraph, err := cudakv.New(k.d.dev, cfg.NumHiddenLayers, cfg.NumKeyValueHeads, cfg.HeadDim(), cfg.MaxSeqLen)
	if kvGraph != nil {
		defer func() {
			if err := kvGraph.Close(); err != nil {
				t.Error(err)
			}
		}()
	}
	if err != nil {
		t.Fatal(err)
	}

	spGraph, err := NewGPUScratchpad(k.d.dev, &cfg)
	if spGraph != nil {
		defer func() {
			if err := spGraph.Free(); err != nil {
				t.Error(err)
			}
		}()
	}
	if err != nil {
		t.Fatal(err)
	}

	dg, err := NewDecodeGraph(k.d.dev, gw, &cfg, kvGraph, spGraph, k.d.gm, k.attnFn, k.sampleFn)
	if dg != nil {
		defer func() {
			if err := dg.Close(); err != nil {
				t.Error(err)
			}
		}()
	}
	if err != nil {
		t.Fatal(err)
	}

	var tok int32
	for pos, ptok := range prompt {
		if err := k.d.dev.WithContext(func() error {
			var err error
			tok, err = GPUForwardPass(gw, &cfg, kvEager, spEager, k.d.gm, k.attnFn, k.sampleFn, ptok, pos)
			if err != nil {
				return err
			}
			_, err = GPUForwardPass(gw, &cfg, kvGraph, spGraph, k.d.gm, k.attnFn, k.sampleFn, ptok, pos)
			return err
		}); err != nil {
			t.Fatalf("prefill pos %d: %v", pos, err)
		}
	}

	// Steps 0 and 1 update all 3L nodes and must match bit-for-bit.
	for step := 0; step < 2; step++ {
		pos := len(prompt) + step
		inTok := tok
		if err := k.d.dev.WithContext(func() error {
			var err error
			tok, err = GPUForwardPass(gw, &cfg, kvEager, spEager, k.d.gm, k.attnFn, k.sampleFn, inTok, pos)
			if err != nil {
				return err
			}
			_, err = dg.Step(pos, nil)
			return err
		}); err != nil {
			t.Fatalf("step %d: %v", step, err)
		}
		if j := bitMismatch(k.d.download(spGraph.Logits), k.d.download(spEager.Logits)); j >= 0 {
			t.Fatalf("step %d matched control unexpectedly differed at logit %d", step, j)
		}
	}

	// Step 2: skip updating layer 0's RoPE pos (leaving it at step 1's pos).
	step2Pos := len(prompt) + 2
	inTok := tok
	if err := k.d.dev.WithContext(func() error {
		var err error
		tok, err = GPUForwardPass(gw, &cfg, kvEager, spEager, k.d.gm, k.attnFn, k.sampleFn, inTok, step2Pos)
		if err != nil {
			return err
		}
		skipped := 0
		for i := range dg.posNodes {
			p := &dg.posNodes[i]
			if p.layer == 0 && p.kind == "rope" {
				skipped++
				continue
			}
			if err := dg.setNodePosition(p, step2Pos); err != nil {
				return err
			}
		}
		if skipped != 1 {
			return fmt.Errorf("skipped %d nodes, want 1", skipped)
		}
		if err := dg.launchExec(nil); err != nil {
			return err
		}
		if err := dg.syncCtx(); err != nil {
			return err
		}
		return dg.copyOut(nil)
	}); err != nil {
		t.Fatalf("step 2 stale launch: %v", err)
	}
	eagerLogits := k.d.download(spEager.Logits)
	staleLogits := k.d.download(spGraph.Logits)
	if bitMismatch(staleLogits, eagerLogits) < 0 {
		t.Fatal("must-fail control: skipping layer 0 RoPE position update at step 2 produced bit-identical logits")
	}
}
