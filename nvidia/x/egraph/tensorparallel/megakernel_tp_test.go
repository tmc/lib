package tensorparallel

import (
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/egraph/hopper"
)

var testTPLayer = Layer{
	Hidden:       512,
	Heads:        32,
	KVHeads:      8,
	HeadDim:      64,
	Intermediate: 2048,
	Eps:          1e-5,
	QKVBias:      true,
	QKVOut:       hopper.BF16,
	AttnDType:    hopper.F32,
}

// crossLayerForward runs Layer[L].OProj + Layer[L].FFN + Layer[L+1].AttnIn
// across all TP ranks in plans, summing the o_proj partials into h (at the
// first .sys barrier) and the down_proj partials into x_out (at the second
// .sys barrier), and returns the concatenated QKV outputs [q_all; k_all; v_all]
// of Layer[L+1].AttnIn along with the reduced h and x_out vectors.
func crossLayerForward(pL, pNext *Plan, wL, wNext weights, x, fullAttn []float64) (h, xOut, qOut, kOut, vOut []float64) {
	hd := pL.Layer.HeadDim
	hidDim := pL.Layer.Hidden

	h = make([]float64, hidDim)
	for _, rp := range pL.Ranks {
		attnSlice := fullAttn[rp.QHeads.Lo*hd : rp.QHeads.Hi*hd]
		oPart := matvec(O(rp, wL.wo), attnSlice)
		if rp.Rank == 0 {
			oPart = add(oPart, x)
		}
		h = add(h, oPart)
	}

	xOut = make([]float64, hidDim)
	for _, rp := range pL.Ranks {
		hn := rmsnorm(h, wL.ffnNorm, float64(pL.Layer.Eps))
		gu := matvec(GateUp(rp, wL.wg, wL.wu), hn)
		n := rp.Inter.Len()
		act := make([]float64, n)
		for i := range act {
			g := gu[i]
			act[i] = g / (1 + math.Exp(-g)) * gu[n+i]
		}
		fPart := matvec(Down(rp, wL.wd), act)
		if rp.Rank == 0 {
			fPart = add(fPart, h)
		}
		xOut = add(xOut, fPart)
	}

	for _, rp := range pNext.Ranks {
		xn := rmsnorm(xOut, wNext.attnNorm, float64(pNext.Layer.Eps))
		qkv := matvec(QKV(rp, wNext.wq, wNext.wk, wNext.wv, hidDim), xn)
		if pNext.Layer.QKVBias {
			qkv = add(qkv, QKV(rp, wNext.bq, wNext.bk, wNext.bv, 1))
		}
		qLen := rp.QHeads.Len() * hd
		kvLen := rp.KVHeads.Len() * hd
		qOut = append(qOut, qkv[:qLen]...)
		kOut = append(kOut, qkv[qLen:qLen+kvLen]...)
		vOut = append(vOut, qkv[qLen+kvLen:qLen+2*kvLen]...)
	}
	return h, xOut, qOut, kOut, vOut
}

func TestTPMegakernelCrossLayer(t *testing.T) {
	for _, tp := range []int{2, 4, 8} {
		for _, mode := range []struct {
			name        string
			fusion      Fusion
			target      string
			barrier     hopper.SysBarrierMode
			wantStore   hopper.GEMMStore
			wantCUDASub []string
		}{
			{
				name:      "peer_atomic",
				fusion:    FuseReduceAdd,
				target:    "sm_90a",
				barrier:   hopper.SysBarrierPeerAtomic,
				wantStore: hopper.StoreReduceAdd,
				wantCUDASub: []string{
					"eg_ready_signal_sys",
					"eg_ready_wait_sys",
					`asm volatile("fence.acq_rel.sys;" ::: "memory");`,
					`atomicAdd_system(slot, (unsigned long long)units);`,
					`asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(slot) : "memory");`,
					`atomicAdd(&out[`,
				},
			},
			{
				name:      "multimem",
				fusion:    FuseMultimem,
				target:    "sm_100a",
				barrier:   hopper.SysBarrierMultimem,
				wantStore: hopper.StoreMultimemReduceAdd,
				wantCUDASub: []string{
					"eg_ready_signal_sys",
					"eg_ready_wait_sys",
					`asm volatile("fence.acq_rel.sys;" ::: "memory");`,
					`multimem.red.release.sys.global.add.u64`,
					`multimem.red.relaxed.sys.global.add.f32`,
					`asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(slot) : "memory");`,
				},
			},
		} {
			t.Run(fmt.Sprintf("tp%d_%s", tp, mode.name), func(t *testing.T) {
				cfg := TPMegakernelConfig{
					TP:          tp,
					Prefix:      fmt.Sprintf("tp%d", tp),
					Fusion:      mode.fusion,
					Target:      mode.target,
					BarrierMode: mode.barrier,
				}
				plans, err := BuildCrossLayerTPMegakernel(cfg, testTPLayer, testTPLayer)
				if err != nil {
					t.Fatalf("BuildCrossLayerTPMegakernel: %v", err)
				}
				if len(plans) != tp {
					t.Fatalf("len(plans) = %d, want %d", len(plans), tp)
				}

				for r, rp := range plans {
					if rp.Rank != r || rp.WorldSize != tp {
						t.Fatalf("rank %d: got Rank=%d WorldSize=%d", r, rp.Rank, rp.WorldSize)
					}
					if rp.BarrierMode != mode.barrier {
						t.Fatalf("rank %d: BarrierMode = %v, want %v", r, rp.BarrierMode, mode.barrier)
					}
					if len(rp.SysBarrierSlots) != 2 || len(rp.SysBarriers) != 2 {
						t.Fatalf("rank %d: SysBarrierSlots=%v SysBarriers=%d, want 2 (.sys after o_proj and down_proj)",
							r, rp.SysBarrierSlots, len(rp.SysBarriers))
					}

					// Check that both reducing stages in Megakernel use the expected GEMMStore.
					oStage := rp.Megakernel.Stages[rp.OProjBarrierSlot]
					dStage := rp.Megakernel.Stages[rp.DownProjBarrierSlot]
					if oStage.Kernel.GEMM == nil || oStage.Kernel.GEMM.Store != mode.wantStore {
						t.Fatalf("rank %d o_proj stage %s: store = %v, want %v", r, oStage.Name, oStage.Kernel.GEMM, mode.wantStore)
					}
					if dStage.Kernel.GEMM == nil || dStage.Kernel.GEMM.Store != mode.wantStore {
						t.Fatalf("rank %d down_proj stage %s: store = %v, want %v", r, dStage.Name, dStage.Kernel.GEMM, mode.wantStore)
					}

					// Only rank 0 takes the residual input l0_x.
					inNames := inputNames(rp.FusedPlan)
					hasResidualX := slices.Contains(inNames, "l0_x")
					if (r == 0) != hasResidualX {
						t.Fatalf("rank %d: has l0_x = %v, want %v (inputs: %v)", r, hasResidualX, r == 0, inNames)
					}

					src, err := rp.Source()
					if err != nil {
						t.Fatalf("rank %d Source(): %v", r, err)
					}
					for _, sub := range mode.wantCUDASub {
						if !strings.Contains(src, sub) {
							t.Errorf("rank %d Source() missing %q", r, sub)
						}
					}

					// Verify that both .sys barrier slots signal &ready[slot] and wait for tp * Blocks.
					for _, bnd := range rp.SysBarriers {
						wantSig := fmt.Sprintf("eg_ready_signal_sys(&ready[%d], epoch, 1);", bnd.Slot)
						wantWait := fmt.Sprintf("eg_ready_wait_sys(&ready[%d], epoch, %d);", bnd.Slot, bnd.TotalNeed())
						if !strings.Contains(src, wantSig) {
							t.Errorf("rank %d Source() missing %q", r, wantSig)
						}
						if !strings.Contains(src, wantWait) {
							t.Errorf("rank %d Source() missing %q", r, wantWait)
						}
					}

					// Verify intra-rank ready slots on rank > 0 are offset to avoid collisions.
					if r > 0 {
						for s := range rp.Megakernel.Stages {
							if slices.Contains(rp.SysBarrierSlots, s) || s == len(rp.Megakernel.Stages)-1 {
								continue
							}
							mapped := rp.ReadySlot(s)
							if mapped == s {
								t.Errorf("rank %d intra-rank slot %d was not offset", r, s)
							}
							wantIntraSig := fmt.Sprintf("eg_ready_signal(&ready[%d], epoch, 1);", mapped)
							if !strings.Contains(src, wantIntraSig) {
								t.Errorf("rank %d Source() missing offset intra-rank signal %q", r, wantIntraSig)
							}
						}
					}
				}
			})
		}
	}
}

func TestTPMegakernelMultiLayerStackAndParity(t *testing.T) {
	layers := []Layer{testTPLayer, testTPLayer, testTPLayer}
	plans1, err := BuildTPMegakernel(TPMegakernelConfig{TP: 1}, layers)
	if err != nil {
		t.Fatalf("BuildTPMegakernel TP=1: %v", err)
	}
	if len(plans1) != 1 || len(plans1[0].Boundaries) != 2 {
		t.Fatalf("TP=1 boundaries = %d, want 2", len(plans1[0].Boundaries))
	}

	rng := rand.New(rand.NewPCG(42, 43))
	w0 := randWeights(rng, testTPLayer)
	w1 := randWeights(rng, testTPLayer)
	x := make([]float64, testTPLayer.Hidden)
	for i := range x {
		x[i] = rng.NormFloat64() * 0.5
	}
	fullAttn := make([]float64, testTPLayer.Heads*testTPLayer.HeadDim)
	for i := range fullAttn {
		fullAttn[i] = rng.NormFloat64() * 0.5
	}

	wantH, wantXOut, wantQ, wantK, wantV := crossLayerForward(plans1[0].ShardL, plans1[0].ShardNext, w0, w1, x, fullAttn)

	for _, tp := range []int{2, 4, 8} {
		plans, err := BuildTPMegakernel(TPMegakernelConfig{
			TP:     tp,
			Fusion: FuseMultimem,
		}, layers)
		if err != nil {
			t.Fatalf("BuildTPMegakernel TP=%d: %v", tp, err)
		}
		if len(plans) != tp {
			t.Fatalf("TP=%d len(plans) = %d", tp, len(plans))
		}
		for r, rp := range plans {
			if len(rp.Boundaries) != 2 {
				t.Fatalf("TP=%d rank %d len(Boundaries) = %d, want 2", tp, r, len(rp.Boundaries))
			}
			if rp.FinalPost == nil || rp.FirstAttnIn == nil {
				t.Fatalf("TP=%d rank %d missing FirstAttnIn or FinalPost", tp, r)
			}
		}

		gotH, gotXOut, gotQ, gotK, gotV := crossLayerForward(plans[0].ShardL, plans[0].ShardNext, w0, w1, x, fullAttn)
		for _, chk := range []struct {
			name      string
			got, want []float64
		}{
			{"h", gotH, wantH},
			{"xOut", gotXOut, wantXOut},
			{"qOut", gotQ, wantQ},
			{"kOut", gotK, wantK},
			{"vOut", gotV, wantV},
		} {
			if len(chk.got) != len(chk.want) {
				t.Fatalf("TP=%d %s len=%d, want %d", tp, chk.name, len(chk.got), len(chk.want))
			}
			for i := range chk.want {
				if d := math.Abs(chk.got[i] - chk.want[i]); d > 1e-9*(1+math.Abs(chk.want[i])) {
					t.Fatalf("TP=%d %s[%d] = %g, want %g (diff %g)", tp, chk.name, i, chk.got[i], chk.want[i], d)
				}
			}
		}
	}
}

func TestTPMegakernelErrors(t *testing.T) {
	tests := []struct {
		name   string
		cfg    TPMegakernelConfig
		layers []Layer
	}{
		{
			name:   "tp_zero",
			cfg:    TPMegakernelConfig{TP: 0},
			layers: []Layer{testTPLayer, testTPLayer},
		},
		{
			name:   "single_layer",
			cfg:    TPMegakernelConfig{TP: 2},
			layers: []Layer{testTPLayer},
		},
		{
			name:   "invalid_barrier_mode",
			cfg:    TPMegakernelConfig{TP: 2, BarrierMode: hopper.SysBarrierMode(99)},
			layers: []Layer{testTPLayer, testTPLayer},
		},
		{
			name:   "invalid_fusion",
			cfg:    TPMegakernelConfig{TP: 2, Fusion: Fusion(99)},
			layers: []Layer{testTPLayer, testTPLayer},
		},
		{
			name:   "multimem_pre_sm90",
			cfg:    TPMegakernelConfig{TP: 2, Fusion: FuseMultimem, Target: "sm_80"},
			layers: []Layer{testTPLayer, testTPLayer},
		},
		{
			name:   "non_divisible_kv_heads",
			cfg:    TPMegakernelConfig{TP: 6},
			layers: []Layer{testTPLayer, testTPLayer},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := BuildTPMegakernel(tc.cfg, tc.layers); err == nil {
				t.Fatalf("BuildTPMegakernel(%+v) succeeded, want error", tc.cfg)
			}
		})
	}
}
