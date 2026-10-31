package model

import (
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudahostemu"
	"github.com/tmc/lib/nvidia/x/cudatmem"
	"github.com/tmc/lib/nvidia/x/egraph/hopper"
	"github.com/tmc/lib/nvidia/x/nvl72"
)

func TestCompileExtendedLayer(t *testing.T) {
	t.Run("medium_batch_hopper_pdl_chain", func(t *testing.T) {
		cfg := ArchConfig{
			HiddenSize:        64,
			IntermediateSize:  128,
			NumHiddenLayers:   2,
			NumAttentionHeads: 4,
			NumKeyValueHeads:  2,
			HeadDim:           16,
			RMSNormEps:        1e-5,
			WeightDType:       BF16,
			KVDType:           BF16,
		}
		bundle, err := CompileExtendedLayer(cfg, ExtendedArchOptions{
			Device: "NVIDIA H200",
			BatchM: 16,
			SeqLen: 64,
		})
		if err != nil {
			t.Fatalf("CompileExtendedLayer: %v", err)
		}
		if bundle.Mode != ModePDLChain {
			t.Fatalf("bundle.Mode = %q, want %q", bundle.Mode, ModePDLChain)
		}
		if bundle.PDL == nil {
			t.Fatalf("bundle.PDL is nil, want emitted PDLPlanSource")
		}
		if len(bundle.PDL.Stages) < 2 {
			t.Fatalf("len(bundle.PDL.Stages) = %d, want >= 2", len(bundle.PDL.Stages))
		}
		for _, want := range []string{
			"griddepcontrol.launch_dependents",
			"griddepcontrol.wait",
			"cudaLaunchAttributeProgrammaticStreamSerialization",
			"eg_pdl_trigger(1u);",
			"eg_pdl_wait(1u);",
		} {
			if !strings.Contains(bundle.PDL.Source, want) {
				t.Fatalf("PDL.Source missing %q", want)
			}
		}

		r := rand.New(rand.NewPCG(701, 702))
		env := randEnv(r, bundle.PDL.Inputs...)

		pdlOut, err := runPDLPlanChain(t, bundle.PDL, env)
		if err != nil {
			t.Fatalf("RunPDLChain: %v", err)
		}
		launcherOut := runPDLPlanLauncher(t, bundle.PDL, env)
		agree(t, "hopper_m16_pdl_chain_vs_launcher", bundle.PDL.Output.DType, pdlOut, launcherOut)
	})

	t.Run("speculative_tree_decode_mtp_attention", func(t *testing.T) {
		const (
			batchM        = 2
			qTokensPerSeq = 4
			seqLen        = 64
		)
		treeMask, err := hopper.TreeDraftMask([]int{-1, 0, 1, 0})
		if err != nil {
			t.Fatalf("TreeDraftMask: %v", err)
		}
		var batchDraftMask []uint64
		for range batchM {
			batchDraftMask = append(batchDraftMask, treeMask...)
		}

		cfg := ArchConfig{
			HiddenSize:        256,
			IntermediateSize:  512,
			NumHiddenLayers:   2,
			NumAttentionHeads: 4,
			NumKeyValueHeads:  2,
			HeadDim:           64,
			RMSNormEps:        1e-5,
			AttentionGate:     true,
			WeightDType:       E4M3,
			KVDType:           E4M3,
		}
		bundle, err := CompileExtendedLayer(cfg, ExtendedArchOptions{
			Device:        "NVIDIA H200",
			BatchM:        batchM,
			SeqLen:        seqLen,
			QTokensPerSeq: qTokensPerSeq,
			DraftMask:     batchDraftMask,
		})
		if err != nil {
			t.Fatalf("CompileExtendedLayer: %v", err)
		}
		if bundle.MTPAttention == nil {
			t.Fatalf("bundle.MTPAttention is nil")
		}
		mtp := *bundle.MTPAttention
		if mtp.QTokensPerSeq != qTokensPerSeq || !mtp.KVFP8 || !mtp.Gate || !mtp.Swizzle || !mtp.WGMMA {
			t.Fatalf("unexpected MTPAttention config: %+v", mtp)
		}

		r := rand.New(rand.NewPCG(801, 802))
		seqlens := []int32{40, 28}
		q, cache := buildMTPTestCase(r, mtp, seqlens, batchDraftMask, 0.5)
		want, err := mtp.Reference(q, cache)
		if err != nil {
			t.Fatalf("mtp.Reference: %v", err)
		}
		got := runMTPPaged(t, mtp, q, cache)
		if errRel := relErr(got, want); errRel > 3e-3 || math.IsNaN(errRel) {
			t.Fatalf("MTP relative L2 error %.4e > 3e-3", errRel)
		}
	})

	t.Run("datacenter_blackwell_b200_tcgen05_cta2_clc", func(t *testing.T) {
		cfg := ArchConfig{
			HiddenSize:        512,
			IntermediateSize:  1024,
			NumHiddenLayers:   4,
			NumAttentionHeads: 8,
			NumKeyValueHeads:  2,
			HeadDim:           64,
			RMSNormEps:        1e-5,
			WeightDType:       NVFP4,
			KVDType:           E4M3,
		}
		bundle, err := CompileExtendedLayer(cfg, ExtendedArchOptions{
			Device:    "NVIDIA B200",
			BatchM:    256,
			SeqLen:    1024,
			EnableCLC: true,
		})
		if err != nil {
			t.Fatalf("CompileExtendedLayer B200: %v", err)
		}
		if bundle.Mode != ModeBlackwellCTA2CLC {
			t.Fatalf("bundle.Mode = %q, want %q", bundle.Mode, ModeBlackwellCTA2CLC)
		}
		if len(bundle.TCGen05Kernels) < 3 {
			t.Fatalf("len(TCGen05Kernels) = %d, want >= 3", len(bundle.TCGen05Kernels))
		}
		if len(bundle.TCGen05Traces) != len(bundle.TCGen05Kernels) {
			t.Fatalf("len(TCGen05Traces) = %d != len(TCGen05Kernels) = %d",
				len(bundle.TCGen05Traces), len(bundle.TCGen05Kernels))
		}
		if len(bundle.CLCKernels) == 0 {
			t.Fatalf("expected non-empty CLCKernels on B200 with BatchM=256")
		}

		allTCGen05 := strings.Join(bundle.TCGen05Kernels, "\n")
		for _, want := range []string{
			"tcgen05.mma.cta_group::2",
			"tcgen05.cp.cta_group::1.32x128b",
			"tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale",
			"eg_tcgen05_mma_tmem_a",
		} {
			if !strings.Contains(allTCGen05, want) {
				t.Errorf("TCGen05Kernels missing %q", want)
			}
		}

		allCLC := strings.Join(bundle.CLCKernels, "\n")
		for _, want := range []string{
			"clusterlaunchcontrol.try_cancel",
			"clusterlaunchcontrol.query_cancel",
		} {
			if !strings.Contains(allCLC, want) {
				t.Errorf("CLCKernels missing %q", want)
			}
		}

		var sawCTA2, sawCopies, sawStores bool
		for i, trace := range bundle.TCGen05Traces {
			summary, err := cudatmem.VerifyTrace(cudatmem.SM100A, trace)
			if err != nil {
				t.Fatalf("VerifyTrace[%d]: %v", i, err)
			}
			if summary.CTAGroup == 2 {
				sawCTA2 = true
			}
			if summary.Copies > 0 {
				sawCopies = true
			}
			if summary.Stores > 0 {
				sawStores = true
			}
		}
		if !sawCTA2 || !sawCopies || !sawStores {
			t.Fatalf("trace coverage: sawCTA2=%v sawCopies=%v sawStores=%v, want all true",
				sawCTA2, sawCopies, sawStores)
		}
	})

	t.Run("multi_gpu_tp_ep_collectives", func(t *testing.T) {
		const (
			ranks          = 4
			rows           = 4
			cols           = 64
			topK           = 2
			expertsPerRank = 2
		)
		cfg := ArchConfig{
			HiddenSize:          cols,
			IntermediateSize:    128,
			MoEIntermediateSize: 64,
			NumHiddenLayers:     2,
			NumAttentionHeads:   4,
			NumKeyValueHeads:    2,
			HeadDim:             16,
			NumLocalExperts:     expertsPerRank,
			NumExperts:          ranks * expertsPerRank,
			NumExpertsPerTok:    topK,
			RMSNormEps:          1e-5,
			WeightDType:         BF16,
			KVDType:             BF16,
		}
		bundle, err := CompileExtendedLayer(cfg, ExtendedArchOptions{
			Device:  "NVIDIA H200",
			BatchM:  rows,
			SeqLen:  64,
			TPRanks: ranks,
			EPRanks: ranks,
		})
		if err != nil {
			t.Fatalf("CompileExtendedLayer TP+EP: %v", err)
		}
		if bundle.CollectiveSource == "" {
			t.Fatalf("bundle.CollectiveSource is empty")
		}
		for _, want := range []string{
			"allreduce_residual_rmsnorm",
			"nvlink_moe_combine",
			"multimem.ld_reduce",
			"eg_nvlink_peer_ptr",
		} {
			if !strings.Contains(bundle.CollectiveSource, want) {
				t.Fatalf("CollectiveSource missing %q", want)
			}
		}

		// 1. Execute AllReduceResidualRMSNorm via cudahostemu.RunMultiRank.
		ar := nvl72.AllReduceResidualRMSNorm{
			Name:    "allreduce_residual_rmsnorm",
			Cols:    cols,
			Ranks:   ranks,
			Eps:     cfg.RMSNormEps,
			DType:   nvl72.BF16,
			Threads: 64,
		}
		elemsPerRank := rows * cols
		symInF64 := make([]float64, ranks*elemsPerRank)
		resInF64 := make([]float64, ranks*elemsPerRank)
		weightF64 := make([]float64, cols)
		for r := range ranks {
			for row := range rows {
				for col := range cols {
					idx := r*elemsPerRank + row*cols + col
					symInF64[idx] = float64(hopper.BF16.Round(float32(((r+1)*(row+1)+col)%7-3) * 0.25))
					resInF64[idx] = float64(hopper.BF16.Round(float32((r+row*3+col*5)%9-4) * 0.125))
				}
			}
		}
		for col := range cols {
			weightF64[col] = float64(hopper.BF16.Round(1.0 + float32((col%5)-2)*0.125))
		}
		wantRes, wantNorm, err := ar.Reference(rows, symInF64, resInF64, weightF64)
		if err != nil {
			t.Fatalf("ar.Reference: %v", err)
		}
		encodeBF16 := func(vals []float64) []byte {
			u16 := make([]uint16, len(vals))
			for i, v := range vals {
				u16[i] = hopper.BF16Bits(float32(v))
			}
			return u16Bytes(u16)
		}
		resOutBytes := cudahostemu.Poison(ranks * elemsPerRank * 2)
		normOutBytes := cudahostemu.Poison(ranks * elemsPerRank * 2)
		arGrid, arBlock := ar.Grid(rows)
		err = cudahostemu.RunMultiRank(t.Context(), bundle.CollectiveSource, ranks, cudahostemu.Launch{
			Kernel: ar.Name,
			Grid:   cudahostemu.Dim3{X: arGrid[0], Y: arGrid[1], Z: arGrid[2]},
			Block:  cudahostemu.Dim3{X: arBlock[0], Y: arBlock[1], Z: arBlock[2]},
		},
			cudahostemu.IntArg(rows),
			cudahostemu.BufferArg("sym_in", "uint16_t", encodeBF16(symInF64)),
			cudahostemu.BufferArg("residual", "uint16_t", encodeBF16(resInF64)),
			cudahostemu.BufferArg("weight", "uint16_t", encodeBF16(weightF64)),
			cudahostemu.BufferArg("residual_out", "uint16_t", resOutBytes),
			cudahostemu.BufferArg("norm_out", "uint16_t", normOutBytes),
		)
		if errors.Is(err, cudahostemu.ErrNoCompiler) {
			t.Skip("no host C++ compiler")
		}
		if err != nil {
			t.Fatalf("RunMultiRank AllReduceResidualRMSNorm: %v", err)
		}
		gotRes := bytesU16(resOutBytes)
		gotNorm := bytesU16(normOutBytes)
		for i := range wantRes {
			if gotRes[i] != hopper.BF16Bits(float32(wantRes[i])) {
				t.Fatalf("residual_out[%d] = %#04x, want %#04x", i, gotRes[i], hopper.BF16Bits(float32(wantRes[i])))
			}
			if gotNorm[i] != hopper.BF16Bits(float32(wantNorm[i])) {
				t.Fatalf("norm_out[%d] = %#04x, want %#04x", i, gotNorm[i], hopper.BF16Bits(float32(wantNorm[i])))
			}
		}

		// 2. Execute NVLinkMoECombine via cudahostemu.RunMultiRank.
		ep := nvl72.NVLinkMoECombine{
			Name:    "nvlink_moe_combine",
			Cols:    cols,
			TopK:    topK,
			Ranks:   ranks,
			DType:   nvl72.BF16,
			Threads: 64,
		}
		totalExperts := ranks * expertsPerRank
		sliceElems := expertsPerRank * rows * cols
		symExpertF64 := make([]float64, ranks*sliceElems)
		for i := range symExpertF64 {
			symExpertF64[i] = float64(hopper.BF16.Round(float32((i%13)-6) * 0.25))
		}
		routesPerRank := rows * topK
		expertIDs := make([]int32, ranks*routesPerRank)
		routerWeightsF64 := make([]float64, ranks*routesPerRank)
		routerWeightsF32 := make([]float32, ranks*routesPerRank)
		for r := range ranks {
			for m := range rows {
				e0 := (r*expertsPerRank + m) % totalExperts
				e1 := (e0 + 3) % totalExperts
				idx0 := r*routesPerRank + m*topK + 0
				idx1 := r*routesPerRank + m*topK + 1
				expertIDs[idx0] = int32(e0)
				expertIDs[idx1] = int32(e1)
				routerWeightsF32[idx0] = 0.625
				routerWeightsF32[idx1] = 0.375
				routerWeightsF64[idx0] = 0.625
				routerWeightsF64[idx1] = 0.375
			}
		}
		wantEP, err := ep.Reference(rows, expertsPerRank, symExpertF64, expertIDs, routerWeightsF64)
		if err != nil {
			t.Fatalf("ep.Reference: %v", err)
		}
		epOutBytes := cudahostemu.Poison(ranks * rows * cols * 2)
		epGrid, epBlock := ep.Grid(rows)
		err = cudahostemu.RunMultiRank(t.Context(), bundle.CollectiveSource, ranks, cudahostemu.Launch{
			Kernel: ep.Name,
			Grid:   cudahostemu.Dim3{X: epGrid[0], Y: epGrid[1], Z: epGrid[2]},
			Block:  cudahostemu.Dim3{X: epBlock[0], Y: epBlock[1], Z: epBlock[2]},
		},
			cudahostemu.IntArg(rows),
			cudahostemu.IntArg(expertsPerRank),
			cudahostemu.BufferArg("sym_expert_out", "uint16_t", encodeBF16(symExpertF64)),
			cudahostemu.BufferArg("expert_ids", "int32_t", i32Bytes(expertIDs)),
			cudahostemu.BufferArg("router_weights", "float", f32Bytes(routerWeightsF32)),
			cudahostemu.BufferArg("out", "uint16_t", epOutBytes),
		)
		if err != nil {
			t.Fatalf("RunMultiRank NVLinkMoECombine: %v", err)
		}
		gotEP := bytesU16(epOutBytes)
		for i, w := range wantEP {
			if gotEP[i] != hopper.BF16Bits(float32(w)) {
				t.Fatalf("ep out[%d] = %#04x, want %#04x", i, gotEP[i], hopper.BF16Bits(float32(w)))
			}
		}
	})
}

func ExampleCompileExtendedLayer() {
	cfg := ArchConfig{
		HiddenSize:        512,
		IntermediateSize:  1024,
		NumAttentionHeads: 8,
		NumKeyValueHeads:  2,
		HeadDim:           64,
		WeightDType:       NVFP4,
		KVDType:           E4M3,
	}
	bundle, err := CompileExtendedLayer(cfg, ExtendedArchOptions{
		Device:    "NVIDIA B200",
		BatchM:    256,
		SeqLen:    1024,
		EnableCLC: true,
	})
	if err != nil {
		panic(err)
	}
	fmt.Println(bundle.Mode, len(bundle.TCGen05Traces) > 0, len(bundle.CLCKernels) > 0)
	// Output:
	// blackwell_cta2_clc true true
}

func runPDLPlanChain(t *testing.T, pdl *hopper.PDLPlanSource, env map[string]hopper.Value) ([]float64, error) {
	t.Helper()
	var args []cudahostemu.Arg
	for _, in := range pdl.Inputs {
		args = append(args, cudahostemu.BufferArg(
			in.Name,
			in.DType.CType(),
			encodeTyped(in.DType, env[in.Name].Data),
		))
	}
	for _, tmp := range pdl.Temps {
		nbytes := tensorSize(tmp.Shape) * tmp.DType.Bits() / 8
		args = append(args, cudahostemu.BufferArg(
			tmp.Name,
			tmp.DType.CType(),
			cudahostemu.Poison(nbytes),
		))
	}
	outBytes := cudahostemu.Poison(tensorSize(pdl.Output.Shape) * pdl.Output.DType.Bits() / 8)
	args = append(args, cudahostemu.BufferArg(
		pdl.Output.Name,
		pdl.Output.DType.CType(),
		outBytes,
	))

	stages := make([]cudahostemu.PDLStage, len(pdl.Stages))
	for i, st := range pdl.Stages {
		var bufNames []string
		for _, out := range st.Outputs {
			bufNames = append(bufNames, out.Name)
		}
		sid := st.TriggerStage
		if sid == 0 {
			sid = st.WaitStage
		}
		stages[i] = cudahostemu.PDLStage{
			Kernel:      st.Name,
			Grid:        cudahostemu.Dim3{X: st.Grid[0], Y: st.Grid[1], Z: st.Grid[2]},
			Block:       cudahostemu.Dim3{X: st.Block[0], Y: st.Block[1], Z: st.Block[2]},
			SharedBytes: st.SharedBytes,
			StageID:     sid,
			Buffers:     bufNames,
			Args:        st.Args,
		}
	}

	err := cudahostemu.RunPDLChain(t.Context(), pdl.Source, stages, args...)
	if errors.Is(err, cudahostemu.ErrNoCompiler) {
		t.Skip("no host C++ compiler; emulated PDL chain not run")
	}
	return decodeTyped(pdl.Output.DType, outBytes), err
}

func runPDLPlanLauncher(t *testing.T, pdl *hopper.PDLPlanSource, env map[string]hopper.Value) []float64 {
	t.Helper()
	var cbufs []cudahostemu.Buffer
	var callArgs []string
	for _, in := range pdl.Inputs {
		cbufs = append(cbufs, cudahostemu.Buffer{
			Name:  in.Name,
			CType: in.DType.CType(),
			Data:  encodeTyped(in.DType, env[in.Name].Data),
		})
		callArgs = append(callArgs, in.Name)
	}
	for _, tmp := range pdl.Temps {
		cbufs = append(cbufs, cudahostemu.Buffer{
			Name:  tmp.Name,
			CType: tmp.DType.CType(),
			Data:  cudahostemu.Poison(tensorSize(tmp.Shape) * tmp.DType.Bits() / 8),
		})
		callArgs = append(callArgs, tmp.Name)
	}
	cbufs = append(cbufs, cudahostemu.Buffer{
		Name:  pdl.Output.Name,
		CType: pdl.Output.DType.CType(),
		Data:  cudahostemu.Poison(tensorSize(pdl.Output.Shape) * pdl.Output.DType.Bits() / 8),
	})
	callArgs = append(callArgs, pdl.Output.Name, "nullptr")

	launch := fmt.Sprintf("launch_%s_pdl(%s); if (eg_emu_pdl_violation_count() != 0) std::_Exit(4)",
		pdl.Name, strings.Join(callArgs, ", "))
	res, err := cudahostemu.Run(t.Context(), t.TempDir(), cudahostemu.Program{
		Source:  cudahostemu.PDLHeader() + "\n" + pdl.Source,
		Launch:  launch,
		Buffers: cbufs,
	})
	if errors.Is(err, cudahostemu.ErrNoCompiler) {
		t.Skip("no host C++ compiler; emulated PDL launcher not run")
	}
	if err != nil {
		t.Fatalf("emulated launch_%s_pdl: %v", pdl.Name, err)
	}
	return decodeTyped(pdl.Output.DType, res[pdl.Output.Name])
}

func buildMTPTestCase(r *rand.Rand, a hopper.MTPPagedAttention, seqlens []int32, draftMask []uint64, kvScale float32) ([]uint16, hopper.MTPPagedCache) {
	batch := len(seqlens)
	totalPages := batch * a.MaxPages
	c := hopper.MTPPagedCache{
		PageTable: make([]int32, batch*a.MaxPages),
		SeqLens:   seqlens,
		DraftMask: draftMask,
		KScale:    kvScale,
		VScale:    kvScale,
	}
	perm := r.Perm(totalPages)
	for i := range c.PageTable {
		c.PageTable[i] = int32(perm[i])
	}
	for b, s := range seqlens {
		used := (int(s) + a.PageSize - 1) / a.PageSize
		for p := used; p < a.MaxPages; p++ {
			c.PageTable[b*a.MaxPages+p] = -1
		}
	}
	totalSlots := totalPages * a.PageSize
	live := make([]bool, totalSlots)
	for b, s := range seqlens {
		for t := range int(s) {
			page := int(c.PageTable[b*a.MaxPages+t/a.PageSize])
			live[page*a.PageSize+t%a.PageSize] = true
		}
	}
	kvElems := totalSlots * a.KVHeads * a.HeadDim
	if a.KVFP8 {
		c.KFP8 = make([]uint8, kvElems)
		c.VFP8 = make([]uint8, kvElems)
		for i := range kvElems {
			c.KFP8[i], c.VFP8[i] = 0x7f, 0x7f
			if live[i/(a.KVHeads*a.HeadDim)] {
				c.KFP8[i] = hopper.E4M3Bits(float32(r.NormFloat64()*0.5) / kvScale)
				c.VFP8[i] = hopper.E4M3Bits(float32(r.NormFloat64()*0.5) / kvScale)
			}
		}
	} else {
		c.KBF16 = make([]uint16, kvElems)
		c.VBF16 = make([]uint16, kvElems)
		for i := range kvElems {
			c.KBF16[i], c.VBF16[i] = 0x7fc0, 0x7fc0
			if live[i/(a.KVHeads*a.HeadDim)] {
				c.KBF16[i] = hopper.BF16Bits(float32(r.NormFloat64() * 0.5))
				c.VBF16[i] = hopper.BF16Bits(float32(r.NormFloat64() * 0.5))
			}
		}
	}
	qElems := batch * a.QTokensPerSeq * a.NumHeads * a.HeadDim
	q := make([]uint16, qElems)
	for i := range q {
		v := float32(r.NormFloat64() * 0.5)
		if a.WGMMA {
			v = hopper.FromE4M3Bits(hopper.E4M3Bits(v))
		}
		q[i] = hopper.BF16Bits(v)
	}
	if a.Gate {
		c.Gate = make([]uint16, qElems)
		for i := range c.Gate {
			c.Gate[i] = hopper.BF16Bits(float32(r.NormFloat64()))
		}
	}
	return q, c
}

func runMTPPaged(t *testing.T, a hopper.MTPPagedAttention, q []uint16, c hopper.MTPPagedCache) []float64 {
	t.Helper()
	src, err := a.Source()
	if err != nil {
		t.Fatal(err)
	}
	batch := len(c.SeqLens)
	cbufs := []cudahostemu.Buffer{
		{Name: "q", CType: "uint16_t", Data: u16Bytes(q)},
		{Name: "page_table", CType: "int32_t", Data: i32Bytes(c.PageTable)},
		{Name: "seqlens", CType: "int32_t", Data: i32Bytes(c.SeqLens)},
	}
	if a.KVFP8 {
		cbufs = append(cbufs,
			cudahostemu.Buffer{Name: "k_cache", CType: "uint8_t", Data: c.KFP8},
			cudahostemu.Buffer{Name: "v_cache", CType: "uint8_t", Data: c.VFP8},
			cudahostemu.Buffer{Name: "k_scale", CType: "float", Data: f32Bytes([]float32{c.KScale})},
			cudahostemu.Buffer{Name: "v_scale", CType: "float", Data: f32Bytes([]float32{c.VScale})},
		)
	} else {
		cbufs = append(cbufs,
			cudahostemu.Buffer{Name: "k_cache", CType: "uint16_t", Data: u16Bytes(c.KBF16)},
			cudahostemu.Buffer{Name: "v_cache", CType: "uint16_t", Data: u16Bytes(c.VBF16)},
		)
	}
	maskArg := "nullptr"
	if c.DraftMask != nil {
		cbufs = append(cbufs, cudahostemu.Buffer{Name: "draft_mask", CType: "uint64_t", Data: u64Bytes(c.DraftMask)})
		maskArg = "draft_mask"
	}
	if a.Gate {
		cbufs = append(cbufs, cudahostemu.Buffer{Name: "gate", CType: "uint16_t", Data: u16Bytes(c.Gate)})
	}
	cbufs = append(cbufs, cudahostemu.Buffer{Name: "out", CType: "uint16_t", Data: cudahostemu.Poison(2 * len(q))})

	args := []string{"q", "k_cache", "v_cache", "page_table", "seqlens", maskArg}
	if a.KVFP8 {
		args = append(args, "k_scale", "v_scale")
	}
	if a.Gate {
		args = append(args, "gate")
	}
	args = append(args, "out", "nullptr")
	launch := fmt.Sprintf("%s_launch(%d, %s)", a.Name, batch, strings.Join(args, ", "))

	res, err := cudahostemu.Run(t.Context(), t.TempDir(), cudahostemu.Program{
		Source:  src,
		Launch:  launch,
		Buffers: cbufs,
		Flags:   strings.Fields(os.Getenv("EG_EMU_FLAGS")),
	})
	if errors.Is(err, cudahostemu.ErrNoCompiler) {
		t.Skip("no host C++ compiler; emulated kernel not run")
	}
	if err != nil {
		t.Fatalf("emulated kernel: %v", err)
	}
	return decodeTyped(hopper.BF16, res["out"])
}
