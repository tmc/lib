package hopper

import (
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"
)

func TestSelectArchDispatchPlanAndOnlineProfile(t *testing.T) {
	tests := []struct {
		target         string
		wantTC         string
		wantDequant    string
		wantCollective string
		wantWorkSteal  string
		wantUsesTMEM   bool
		wantTMEMCols   int
		wantStages     int
		wantSyncRestr  bool
		wantLUTB       bool
	}{
		{
			target:         TargetSM90A,
			wantTC:         "wgmma.mma_async.sync.aligned",
			wantDequant:    "prmt.b32",
			wantCollective: "dsmem.cluster.allreduce",
			wantWorkSteal:  "atomicAdd.grid_queue",
			wantUsesTMEM:   false,
			wantTMEMCols:   0,
			wantStages:     3,
		},
		{
			target:         TargetSM100A,
			wantTC:         "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X",
			wantDequant:    "cvt.rn.f16x2.e2m1x2",
			wantCollective: "multimem.ld_reduce",
			wantWorkSteal:  "clusterlaunchcontrol.try_cancel",
			wantUsesTMEM:   true,
			wantTMEMCols:   512,
			wantStages:     4,
		},
		{
			target:         TargetSM103A,
			wantTC:         "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X",
			wantDequant:    "tcgen05.ld.red.sync.aligned.32x32b.x4.max.f32",
			wantCollective: "multimem.ld_reduce",
			wantWorkSteal:  "clusterlaunchcontrol.try_cancel",
			wantUsesTMEM:   true,
			wantTMEMCols:   512,
			wantStages:     4,
		},
		{
			target:         TargetSM107A,
			wantTC:         "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.decompress::lut::b.block32.sync_restrict::shared::read::mma::a",
			wantDequant:    "tcgen05.ld.red.spcompress.sync.aligned.32x32b.x4.max.sp::2:4.abs.f32.b2",
			wantCollective: "fabric.try_red.async.tensor.2d.multimem",
			wantWorkSteal:  "clusterlaunchcontrol.try_cancel",
			wantUsesTMEM:   true,
			wantTMEMCols:   576,
			wantStages:     4,
			wantSyncRestr:  true,
			wantLUTB:       true,
		},
		{
			target:         TargetSM120A,
			wantTC:         "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale",
			wantDequant:    "cvt.rn.f16x2.e2m1x2",
			wantCollective: "cp.async.bulk.global.to.shared::cta",
			wantWorkSteal:  "clusterlaunchcontrol.try_cancel",
			wantUsesTMEM:   false,
			wantTMEMCols:   0,
			wantStages:     2,
		},
		{
			target:         TargetSM121A,
			wantTC:         "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale",
			wantDequant:    "prmt.b32 (eg_e2m1x4_to_bf16x4)",
			wantCollective: "cp.async.bulk.global.to.shared::cta",
			wantWorkSteal:  "clusterlaunchcontrol.try_cancel",
			wantUsesTMEM:   false,
			wantTMEMCols:   0,
			wantStages:     2,
		},
	}

	for _, tc := range tests {
		t.Run(tc.target, func(t *testing.T) {
			plan, err := SelectArchDispatchPlan(tc.target, nil)
			if err != nil {
				t.Fatalf("SelectArchDispatchPlan(%q, nil): %v", tc.target, err)
			}
			if plan.Target != tc.target {
				t.Errorf("Target = %q, want %q", plan.Target, tc.target)
			}
			if !strings.Contains(plan.TensorCoreMnemonic, tc.wantTC) {
				t.Errorf("TensorCoreMnemonic = %q, want substring %q", plan.TensorCoreMnemonic, tc.wantTC)
			}
			if !strings.Contains(plan.DequantMnemonic, tc.wantDequant) {
				t.Errorf("DequantMnemonic = %q, want substring %q", plan.DequantMnemonic, tc.wantDequant)
			}
			if !strings.Contains(plan.CollectiveMnemonic, tc.wantCollective) {
				t.Errorf("CollectiveMnemonic = %q, want substring %q", plan.CollectiveMnemonic, tc.wantCollective)
			}
			if !strings.Contains(plan.WorkStealMnemonic, tc.wantWorkSteal) {
				t.Errorf("WorkStealMnemonic = %q, want substring %q", plan.WorkStealMnemonic, tc.wantWorkSteal)
			}
			if plan.UsesTMEM != tc.wantUsesTMEM || plan.TMEMColumns != tc.wantTMEMCols {
				t.Errorf("UsesTMEM=%v TMEMColumns=%d, want %v/%d",
					plan.UsesTMEM, plan.TMEMColumns, tc.wantUsesTMEM, tc.wantTMEMCols)
			}
			if plan.PipelineStages != tc.wantStages {
				t.Errorf("PipelineStages = %d, want %d", plan.PipelineStages, tc.wantStages)
			}
			if plan.UsesEarlyReleaseSyncRestrict != tc.wantSyncRestr {
				t.Errorf("UsesEarlyReleaseSyncRestrict = %v, want %v", plan.UsesEarlyReleaseSyncRestrict, tc.wantSyncRestr)
			}
			if plan.UsesLUTDecompressB != tc.wantLUTB {
				t.Errorf("UsesLUTDecompressB = %v, want %v", plan.UsesLUTDecompressB, tc.wantLUTB)
			}

			// Verify online profile override adjusts PipelineStages, register budgets, and UsesLUTDecompressB.
			prof := &OnlineMegakernelProfile{
				DominantStall:    "memory_latency",
				SpillDetected:    true,
				PreferLUTB:       true,
				PreferSparse2to4: true,
			}
			overridden, err := SelectArchDispatchPlan(tc.target, prof)
			if err != nil {
				t.Fatalf("SelectArchDispatchPlan(%q, prof): %v", tc.target, err)
			}
			if overridden.PipelineStages != plan.PipelineStages+1 {
				t.Errorf("overridden PipelineStages = %d, want %d", overridden.PipelineStages, plan.PipelineStages+1)
			}
			if overridden.ProducerMaxRegs >= plan.ProducerMaxRegs {
				t.Errorf("overridden ProducerMaxRegs = %d, want < %d", overridden.ProducerMaxRegs, plan.ProducerMaxRegs)
			}
			if overridden.ConsumerMaxRegs >= plan.ConsumerMaxRegs {
				t.Errorf("overridden ConsumerMaxRegs = %d, want < %d", overridden.ConsumerMaxRegs, plan.ConsumerMaxRegs)
			}
			if !overridden.UsesLUTDecompressB {
				t.Errorf("overridden UsesLUTDecompressB = false, want true")
			}
			if !strings.Contains(overridden.DequantMnemonic, "sp::2:4") {
				t.Errorf("overridden DequantMnemonic = %q, want sp::2:4", overridden.DequantMnemonic)
			}
		})
	}

	if _, err := SelectArchDispatchPlan("sm_80", nil); err == nil {
		t.Fatal("SelectArchDispatchPlan(sm_80) succeeded, want error")
	}
}

func TestMultiArchMegakernelAllPlatforms(t *testing.T) {
	const (
		numTokens = 2
		H         = 16
		I         = 32
		HQ        = 2
		D         = 8
	)

	r := rand.New(rand.NewPCG(1403, 2026))
	randVec := func(n int, scale float32) []float32 {
		out := make([]float32, n)
		for i := range out {
			out[i] = (r.Float32()*2 - 1) * scale
		}
		return out
	}

	x := randVec(numTokens*H, 0.5)
	rmsAttnW := randVec(H, 1.0)
	wQKV := randVec(3*H*H, 0.25)
	wO := randVec(H*H, 0.25)
	rmsFFNW := randVec(H, 1.0)
	wGate := randVec(I*H, 0.25)
	wUp := randVec(I*H, 0.25)
	wDown := randVec(H*I, 0.25)

	base := MultiArchMegakernel{
		Name:         "eg_multiarch_megakernel",
		Hidden:       H,
		Intermediate: I,
		NumHeads:     HQ,
		HeadDim:      D,
		Eps:          1e-5,
	}

	allSources, err := base.EmitAllTargets()
	if err != nil {
		t.Fatalf("EmitAllTargets: %v", err)
	}
	if len(allSources) != len(AllMegakernelArchTargets) {
		t.Fatalf("len(EmitAllTargets) = %d, want %d", len(allSources), len(AllMegakernelArchTargets))
	}

	wantSubstrings := map[string][]string{
		TargetSM90A: {
			"wgmma.mma_async.sync.aligned",
			"dsmem.cluster.allreduce",
			"eg_e2m1x4_to_bf16x4",
		},
		TargetSM100A: {
			"tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X",
			"clusterlaunchcontrol.try_cancel",
			"multimem.ld_reduce",
		},
		TargetSM103A: {
			"tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X",
			"tcgen05.ld.red.sync.aligned.32x32b.x4.max.f32",
			"clusterlaunchcontrol.try_cancel",
		},
		TargetSM107A: {
			"tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.decompress::lut::b.block32",
			".sync_restrict::shared::read::mma::a",
			"tcgen05.ld.red.spcompress.sync.aligned.32x32b.x4.max.sp::2:4.abs.f32.b2",
			"fma.rn.satfinite.e4m3x4.e2m1p4x4.e4m3x4.e4m3x4",
			"add.rn.f32x2",
			"fabric.try_red.async.tensor.2d.multimem",
		},
		TargetSM120A: {
			"mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale",
			"cvt.rn.f16x2.e2m1x2",
			"__CUDA_ARCH_FEAT_SM120_ALL",
		},
		TargetSM121A: {
			"mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale",
			"prmt.b32",
			"eg_e2m1x4_to_bf16x4",
		},
	}

	for _, target := range AllMegakernelArchTargets {
		t.Run(target, func(t *testing.T) {
			mk := base
			mk.Target = target
			if err := mk.Validate(); err != nil {
				t.Fatalf("Validate(%s): %v", target, err)
			}
			cudaSrc, err := mk.CUDA()
			if err != nil {
				t.Fatalf("CUDA(%s): %v", target, err)
			}
			for _, sub := range wantSubstrings[target] {
				if !strings.Contains(cudaSrc, sub) {
					t.Errorf("target %s CUDA missing %q", target, sub)
				}
			}
			if target == TargetSM121A && strings.Contains(cudaSrc, "cvt.rn.f16x2.e2m1x2") {
				t.Errorf("sm_121a must use 3-instruction prmt.b32 LUT instead of cvt.rn.f16x2.e2m1x2")
			}

			want, err := mk.Reference(x, rmsAttnW, wQKV, wO, rmsFFNW, wGate, wUp, wDown)
			if err != nil {
				t.Fatalf("Reference(%s): %v", target, err)
			}

			launchSrc, err := mk.Launcher()
			if err != nil {
				t.Fatalf("Launcher(%s): %v", target, err)
			}
			launchCall := fmt.Sprintf("%s_run(x, rms_attn_w, w_qkv, w_o, rms_ffn_w, w_gate, w_up, w_down, out, ready, 1u, %d)",
				mk.Name, numTokens)
			res := runEmu(t, cudaSrc+"\n"+launchSrc, launchCall,
				emuBuf{"x", "float", f32Bytes(x)},
				emuBuf{"rms_attn_w", "float", f32Bytes(rmsAttnW)},
				emuBuf{"w_qkv", "float", f32Bytes(wQKV)},
				emuBuf{"w_o", "float", f32Bytes(wO)},
				emuBuf{"rms_ffn_w", "float", f32Bytes(rmsFFNW)},
				emuBuf{"w_gate", "float", f32Bytes(wGate)},
				emuBuf{"w_up", "float", f32Bytes(wUp)},
				emuBuf{"w_down", "float", f32Bytes(wDown)},
				emuBuf{"out", "float", poison(4 * numTokens * H)},
				emuBuf{"ready", "unsigned long long", make([]byte, 8*8)},
			)
			got := bytesF32(res["out"])
			var maxDiff float64
			for i := range want {
				if math.IsNaN(float64(got[i])) {
					t.Fatalf("target %s out[%d] is NaN", target, i)
				}
				diff := math.Abs(float64(got[i] - want[i]))
				if diff > maxDiff {
					maxDiff = diff
				}
			}
			if maxDiff >= 1e-4 {
				t.Fatalf("target %s maxAbsDiff = %g, want < 1e-4", target, maxDiff)
			}

			// Verify ready counter words were signaled by the op queue.
			gotReady := bytesU64(res["ready"])
			for slot := range 6 {
				if wantWord := PackReadyWord(1, 1); gotReady[slot] != wantWord {
					t.Fatalf("target %s ready[%d] = %#x, want %#x", target, slot, gotReady[slot], wantWord)
				}
			}

			// Mutant rejection: corrupting wDown must significantly alter the megakernel output.
			corruptDown := slices.Clone(wDown)
			for i := range corruptDown {
				corruptDown[i] = -corruptDown[i] * 3.0
			}
			mutRes := runEmu(t, cudaSrc+"\n"+launchSrc, launchCall,
				emuBuf{"x", "float", f32Bytes(x)},
				emuBuf{"rms_attn_w", "float", f32Bytes(rmsAttnW)},
				emuBuf{"w_qkv", "float", f32Bytes(wQKV)},
				emuBuf{"w_o", "float", f32Bytes(wO)},
				emuBuf{"rms_ffn_w", "float", f32Bytes(rmsFFNW)},
				emuBuf{"w_gate", "float", f32Bytes(wGate)},
				emuBuf{"w_up", "float", f32Bytes(wUp)},
				emuBuf{"w_down", "float", f32Bytes(corruptDown)},
				emuBuf{"out", "float", poison(4 * numTokens * H)},
				emuBuf{"ready", "unsigned long long", make([]byte, 8*8)},
			)
			mutGot := bytesF32(mutRes["out"])
			var maxMutDiff float64
			for i := range want {
				if d := math.Abs(float64(mutGot[i] - want[i])); d > maxMutDiff {
					maxMutDiff = d
				}
			}
			if maxMutDiff < 1e-2 {
				t.Fatalf("target %s mutant wDown not rejected: maxMutDiff = %g", target, maxMutDiff)
			}
		})
	}
}

func ExampleMultiArchMegakernel() {
	mk := MultiArchMegakernel{
		Hidden:       16,
		Intermediate: 32,
		NumHeads:     2,
		HeadDim:      8,
		Target:       TargetSM107A,
		Profile: &OnlineMegakernelProfile{
			DominantStall: "memory_latency",
			PreferLUTB:    true,
		},
	}
	plan, _ := mk.Plan()
	all, _ := mk.EmitAllTargets()
	fmt.Println(plan.Target, plan.UsesTMEM, plan.TMEMColumns, plan.PipelineStages, plan.UsesLUTDecompressB, len(all))
	// Output:
	// sm_107a true 576 5 true 6
}
