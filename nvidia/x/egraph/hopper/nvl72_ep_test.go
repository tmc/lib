package hopper

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudahostemu"
)

func nvl72BytesU32(b []byte) []uint32 {
	xs := make([]uint32, len(b)/4)
	for i := range xs {
		xs[i] = binary.LittleEndian.Uint32(b[4*i:])
	}
	return xs
}

func TestNVL72FabricHelpersPTX(t *testing.T) {
	helpers := NVL72FabricHelpers()
	for _, want := range []string{
		"eg_nvl72_tray_id",
		"eg_nvl72_tray_local_rank",
		"eg_nvl72_multicast_barrier",
		"fence.acq_rel.sys;",
		"multimem.red.release.sys.global.add.u32 [%0], 1;",
		"ld.acquire.sys.global.u32 %0, [%1];",
		"multimem.st.global.s32 [%0], %1;",
		"eg_nvl72_emu_run",
	} {
		if !strings.Contains(helpers, want) {
			t.Fatalf("NVL72FabricHelpers missing %q", want)
		}
	}
}

func TestNVL72WideEP(t *testing.T) {
	cases := []struct {
		name    string
		ranks   int
		experts int
		topK    int
		tokens  int
		hidden  int
		dtype   DType
		threads int
	}{
		{
			name:    "R8_E16_K2_S4_H32_F32",
			ranks:   8,
			experts: 16,
			topK:    2,
			tokens:  4,
			hidden:  32,
			dtype:   F32,
			threads: 64,
		},
		{
			name:    "R8_E16_K2_S4_H32_BF16",
			ranks:   8,
			experts: 16,
			topK:    2,
			tokens:  4,
			hidden:  32,
			dtype:   BF16,
			threads: 64,
		},
		{
			name:    "R72_FullNVL72_18Trays_E72_K2_S2_H16_F32",
			ranks:   72,
			experts: 72,
			topK:    2,
			tokens:  2,
			hidden:  16,
			dtype:   F32,
			threads: 32,
		},
		{
			name:    "R72_FullNVL72_18Trays_E72_K2_S2_H16_BF16",
			ranks:   72,
			experts: 72,
			topK:    2,
			tokens:  2,
			hidden:  16,
			dtype:   BF16,
			threads: 32,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			pipe := NVL72WideEP{
				Name:          "nvl72_ep",
				TokensPerRank: tc.tokens,
				HiddenSize:    tc.hidden,
				TopK:          tc.topK,
				TotalExperts:  tc.experts,
				Ranks:         tc.ranks,
				Eps:           1e-5,
				DType:         tc.dtype,
				QuantizeScale: 1.0,
				Threads:       tc.threads,
			}
			src, err := pipe.Source()
			if err != nil {
				t.Fatalf("Source: %v", err)
			}

			epn := tc.experts / tc.ranks
			routesPerRank := tc.tokens * tc.topK
			maxSlots := tc.ranks * routesPerRank
			totalTokenElems := tc.ranks * tc.tokens * tc.hidden

			// Input activations x [R, S, H] and residual [R, S, H] using exact dyadic rationals.
			xF64 := make([]float64, totalTokenElems)
			resF64 := make([]float64, totalTokenElems)
			for r := 0; r < tc.ranks; r++ {
				for s := 0; s < tc.tokens; s++ {
					for c := 0; c < tc.hidden; c++ {
						idx := (r*tc.tokens+s)*tc.hidden + c
						xv := float32(((r*3+s*5+c)%11)-5) * 0.25
						rv := float32(((r*7+s*2+c*3)%9)-4) * 0.125
						xF64[idx] = float64(tc.dtype.Round(xv))
						resF64[idx] = float64(tc.dtype.Round(rv))
					}
				}
			}

			// Routing table topk_ids [R, S, K] and topk_weights [R, S, K].
			// Route k=0 across remote trays ((tray + 1) % numTrays) and k=1 across
			// further remote experts so every rank exercises cross-tray NVLink 5
			// and Grace NVLink-C2C cold-expert fallback.
			topkIDs := make([]int32, tc.ranks*routesPerRank)
			topkWeights := make([]float32, tc.ranks*routesPerRank)
			for r := 0; r < tc.ranks; r++ {
				for s := 0; s < tc.tokens; s++ {
					base := (r*tc.tokens + s) * tc.topK
					e0 := (r*epn + s + 5) % tc.experts
					e1 := (r*epn + s*3 + 17) % tc.experts
					if e1 == e0 {
						e1 = (e1 + 1) % tc.experts
					}
					topkIDs[base+0] = int32(e0)
					topkIDs[base+1] = int32(e1)
					topkWeights[base+0] = 0.625
					topkWeights[base+1] = 0.375
				}
			}

			// Flag every 4th expert as cold/offloaded to Grace CPU LPDDR5X over NVLink-C2C.
			coldExpertMask := make([]int32, tc.experts)
			for e := 0; e < tc.experts; e++ {
				if e%4 == 1 {
					coldExpertMask[e] = 1
				}
			}

			expertScalesF64 := make([]float64, tc.experts*tc.hidden)
			for e := 0; e < tc.experts; e++ {
				for c := 0; c < tc.hidden; c++ {
					sv := float32(0.5) + float32((e+c)%5)*0.25
					expertScalesF64[e*tc.hidden+c] = float64(tc.dtype.Round(sv))
				}
			}

			rmsWeightF64 := make([]float64, tc.hidden)
			for c := 0; c < tc.hidden; c++ {
				wv := float32(1.0) + float32((c%5)-2)*0.125
				rmsWeightF64[c] = float64(tc.dtype.Round(wv))
			}

			wantTPE, wantOffsets, wantDestSlots, wantSrcInfo, wantResOut, wantNormOut, err := pipe.Reference(
				xF64, topkIDs, topkWeights, expertScalesF64, coldExpertMask, resF64, rmsWeightF64,
			)
			if err != nil {
				t.Fatalf("Reference: %v", err)
			}

			encodeVals := func(vals []float64) []byte {
				if tc.dtype == F32 {
					f32 := make([]float32, len(vals))
					for i, v := range vals {
						f32[i] = float32(v)
					}
					return f32Bytes(f32)
				}
				u16 := make([]uint16, len(vals))
				for i, v := range vals {
					u16[i] = BF16Bits(float32(v))
				}
				return u16Bytes(u16)
			}

			elemBytes := tc.dtype.Bits() / 8
			ct := tc.dtype.cType()
			initSrcInfo := make([]int32, tc.ranks*maxSlots)
			for i := range initSrcInfo {
				initSrcInfo[i] = -1
			}

			launchCode := fmt.Sprintf(`
    nvl72_ep_plan_launch(topk_ids, tokens_per_expert, expert_offsets, dest_slots, mc_barrier, 0u, nullptr);
    nvl72_ep_dispatch_launch(x, topk_ids, topk_weights, dest_slots, sym_recv_x, src_info, recv_weights, mc_barrier, 3u, nullptr);
    nvl72_ep_expert_scale_launch(sym_recv_x, expert_offsets, expert_scales, cold_expert_mask, expert_out, c2c_cold_expert_out, mc_barrier, 4u, nullptr);
    nvl72_ep_combine_launch(expert_out, c2c_cold_expert_out, cold_expert_mask, topk_ids, topk_weights, dest_slots, residual, rms_weight, residual_out, norm_out, mc_barrier, 5u, nullptr);
`)

			res, err := cudahostemu.Run(context.Background(), t.TempDir(), cudahostemu.Program{
				Source: src,
				Launch: launchCode,
				Buffers: []cudahostemu.Buffer{
					{Name: "x", CType: ct, Data: encodeVals(xF64)},
					{Name: "topk_ids", CType: "int32_t", Data: i32Bytes(topkIDs)},
					{Name: "topk_weights", CType: "float", Data: f32Bytes(topkWeights)},
					{Name: "tokens_per_expert", CType: "int32_t", Data: cudahostemu.Poison(len(wantTPE) * 4)},
					{Name: "expert_offsets", CType: "int32_t", Data: cudahostemu.Poison(len(wantOffsets) * 4)},
					{Name: "dest_slots", CType: "int32_t", Data: cudahostemu.Poison(len(wantDestSlots) * 4)},
					{Name: "sym_recv_x", CType: ct, Data: make([]byte, tc.ranks*maxSlots*tc.hidden*elemBytes)},
					{Name: "src_info", CType: "int32_t", Data: i32Bytes(initSrcInfo)},
					{Name: "recv_weights", CType: "float", Data: make([]byte, tc.ranks*maxSlots*4)},
					{Name: "expert_scales", CType: ct, Data: encodeVals(expertScalesF64)},
					{Name: "cold_expert_mask", CType: "int32_t", Data: i32Bytes(coldExpertMask)},
					{Name: "expert_out", CType: ct, Data: make([]byte, tc.ranks*maxSlots*tc.hidden*elemBytes)},
					{Name: "c2c_cold_expert_out", CType: ct, Data: make([]byte, tc.ranks*maxSlots*tc.hidden*elemBytes)},
					{Name: "residual", CType: ct, Data: encodeVals(resF64)},
					{Name: "rms_weight", CType: ct, Data: encodeVals(rmsWeightF64)},
					{Name: "residual_out", CType: ct, Data: cudahostemu.Poison(totalTokenElems * elemBytes)},
					{Name: "norm_out", CType: ct, Data: cudahostemu.Poison(totalTokenElems * elemBytes)},
					{Name: "mc_barrier", CType: "uint32_t", Data: u32Bytes(make([]uint32, tc.ranks))},
				},
			})
			if errors.Is(err, cudahostemu.ErrNoCompiler) {
				t.Skip(err)
			}
			if err != nil {
				t.Fatalf("cudahostemu.Run: %v", err)
			}

			// Verify O(1) NVSwitch multicast barrier reached 5 epochs * R across all R ranks.
			gotBarrier := nvl72BytesU32(res["mc_barrier"])
			wantBarrierVal := uint32(5 * tc.ranks)
			for r, got := range gotBarrier {
				if got != wantBarrierVal {
					t.Fatalf("mc_barrier[rank=%d] = %d, want %d", r, got, wantBarrierVal)
				}
			}

			// Verify routing plan outputs: tokens_per_expert, expert_offsets, dest_slots, src_info.
			gotTPE := bytesI32(res["tokens_per_expert"])
			for i, w := range wantTPE {
				if gotTPE[i] != w {
					t.Fatalf("tokens_per_expert[%d] = %d, want %d", i, gotTPE[i], w)
				}
			}
			gotOffsets := bytesI32(res["expert_offsets"])
			for i, w := range wantOffsets {
				if gotOffsets[i] != w {
					t.Fatalf("expert_offsets[%d] = %d, want %d", i, gotOffsets[i], w)
				}
			}
			gotDestSlots := bytesI32(res["dest_slots"])
			for i, w := range wantDestSlots {
				if gotDestSlots[i] != w {
					t.Fatalf("dest_slots[%d] = %d, want %d", i, gotDestSlots[i], w)
				}
			}
			gotSrcInfo := bytesI32(res["src_info"])
			for i, w := range wantSrcInfo {
				if gotSrcInfo[i] != w {
					t.Fatalf("src_info[%d] = %d, want %d", i, gotSrcInfo[i], w)
				}
			}

			// Verify Combine + Grace C2C fallback + Residual RMSNorm bit-accurately.
			if tc.dtype == F32 {
				gotRes := bytesF32(res["residual_out"])
				gotNorm := bytesF32(res["norm_out"])
				for i := range wantResOut {
					wResBits := math.Float32bits(float32(wantResOut[i]))
					gResBits := math.Float32bits(gotRes[i])
					if gResBits != wResBits {
						t.Fatalf("residual_out[%d] = %#08x (%v), want %#08x (%v)",
							i, gResBits, gotRes[i], wResBits, float32(wantResOut[i]))
					}
					wNormBits := math.Float32bits(float32(wantNormOut[i]))
					gNormBits := math.Float32bits(gotNorm[i])
					if gNormBits != wNormBits {
						t.Fatalf("norm_out[%d] = %#08x (%v), want %#08x (%v)",
							i, gNormBits, gotNorm[i], wNormBits, float32(wantNormOut[i]))
					}
				}
			} else {
				gotRes := bytesU16(res["residual_out"])
				gotNorm := bytesU16(res["norm_out"])
				for i := range wantResOut {
					wResBits := BF16Bits(float32(wantResOut[i]))
					if gotRes[i] != wResBits {
						t.Fatalf("residual_out[%d] = %#04x (%v), want %#04x (%v)",
							i, gotRes[i], FromBF16Bits(gotRes[i]), wResBits, float32(wantResOut[i]))
					}
					wNormBits := BF16Bits(float32(wantNormOut[i]))
					if gotNorm[i] != wNormBits {
						t.Fatalf("norm_out[%d] = %#04x (%v), want %#04x (%v)",
							i, gotNorm[i], FromBF16Bits(gotNorm[i]), wNormBits, float32(wantNormOut[i]))
					}
				}
			}
		})
	}
}

func TestEstimateNVL72WideEP(t *testing.T) {
	hotOnly := EstimateNVL72WideEP(64, 4096, 8, 288, 72, 0.0)
	if hotOnly.NumRanks != 72 || hotOnly.NumTrays != 18 {
		t.Fatalf("topology = (%d ranks, %d trays), want (72, 18)", hotOnly.NumRanks, hotOnly.NumTrays)
	}
	if hotOnly.CombineC2CBytes != 0 || hotOnly.CombineC2CNs != 0 {
		t.Fatalf("hotOnly C2C traffic = (%d B, %v ns), want 0", hotOnly.CombineC2CBytes, hotOnly.CombineC2CNs)
	}
	if hotOnly.TBOOverlapSavingsNs <= 0 || hotOnly.TBOSpeedup <= 1.0 {
		t.Fatalf("expected positive TBO overlap speedup, got savings=%v ns speedup=%v",
			hotOnly.TBOOverlapSavingsNs, hotOnly.TBOSpeedup)
	}
	if hotOnly.TBOTwoBatchNs >= hotOnly.SequentialTwoBatchNs {
		t.Fatalf("TBOTwoBatchNs (%v) must be < SequentialTwoBatchNs (%v)",
			hotOnly.TBOTwoBatchNs, hotOnly.SequentialTwoBatchNs)
	}

	withCold := EstimateNVL72WideEP(64, 4096, 8, 288, 72, 0.25)
	if withCold.CombineC2CBytes == 0 || withCold.CombineC2CNs <= 0 {
		t.Fatalf("expected non-zero Grace NVLink-C2C traffic for coldFraction=0.25, got %+v", withCold)
	}
	if withCold.CombineNVLinkBytes >= hotOnly.CombineNVLinkBytes {
		t.Fatalf("expected reduced NVLink 5 combine bytes when 25%% of routes use Grace C2C: got %d vs %d",
			withCold.CombineNVLinkBytes, hotOnly.CombineNVLinkBytes)
	}
	// Grace NVLink-C2C is 450 GB/s vs 900 GB/s NVLink 5, so shifting 25% of
	// routes to C2C increases total combine latency by 1.25x on the combine fabric component.
	if withCold.CombineTotalNs <= hotOnly.CombineTotalNs {
		t.Fatalf("expected CombineTotalNs with C2C offload (%v) > hot-only NVLink 5 (%v)",
			withCold.CombineTotalNs, hotOnly.CombineTotalNs)
	}
}
