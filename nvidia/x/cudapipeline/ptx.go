package cudapipeline

import (
	"fmt"
	"strings"
)

// EmitMBarrierInit returns a PTX instruction string initializing an mbarrier
// in shared memory with the given arrival count.
func EmitMBarrierInit(mbarAddr string, arrivalCount int) string {
	return fmt.Sprintf("mbarrier.init.shared::cta.b64 [%s], %d;", mbarAddr, arrivalCount)
}

// EmitArriveExpectTx returns a PTX instruction string arriving on an mbarrier
// and registering an expected transaction byte count for TMA.
func EmitArriveExpectTx(mbarAddr string, txBytes int) string {
	return fmt.Sprintf("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%s], %d;", mbarAddr, txBytes)
}

// EmitTMABulkAsync2D returns a PTX instruction string issuing a 2D TMA bulk asynchronous copy
// from global memory to CTA shared memory, signaling completion to an mbarrier.
func EmitTMABulkAsync2D(dstSmem, descAddr, coord0, coord1, mbarAddr string) string {
	return fmt.Sprintf(
		"cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%s], [%s, {%s, %s}], [%s];",
		dstSmem, descAddr, coord0, coord1, mbarAddr,
	)
}

// EmitTMABulkAsync2DCluster returns a PTX instruction string issuing a 2D TMA bulk asynchronous copy
// targeted across a CTA cluster.
func EmitTMABulkAsync2DCluster(dstSmem, descAddr, coord0, coord1, mbarAddr string) string {
	return fmt.Sprintf(
		"cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%s], [%s, {%s, %s}], [%s];",
		dstSmem, descAddr, coord0, coord1, mbarAddr,
	)
}

// EmitTMEMAlloc returns a PTX instruction string allocating nCols of Tensor Memory (TMEM).
func EmitTMEMAlloc(dstSharedAddr string, nCols int) string {
	return fmt.Sprintf("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%s], %d;", dstSharedAddr, nCols)
}

// EmitTMEMDealloc returns a PTX instruction string deallocating nCols of Tensor Memory (TMEM).
func EmitTMEMDealloc(taddrReg string, nCols int) string {
	return fmt.Sprintf("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %s, %d;", taddrReg, nCols)
}

// EmitTMEMRelinquishPermit returns a PTX instruction string relinquishing the TMEM allocation permit.
func EmitTMEMRelinquishPermit() string {
	return "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"
}

// EmitTMABulkAsync2DTMEM returns a PTX instruction string issuing a 2D TMA bulk asynchronous copy
// directly from global memory into TMEM, signaling completion to an mbarrier in shared memory.
func EmitTMABulkAsync2DTMEM(dstTmem, descAddr, coord0, coord1, mbarAddr string) string {
	return fmt.Sprintf(
		"cp.async.bulk.tensor.2d.tmem.global.mbarrier::complete_tx::bytes [%s], [%s, {%s, %s}], [%s];",
		dstTmem, descAddr, coord0, coord1, mbarAddr,
	)
}

// EmitTMABulkAsyncTMEM returns a PTX instruction string issuing a 1D bulk asynchronous copy
// directly from global memory into TMEM, signaling completion to an mbarrier in shared memory.
func EmitTMABulkAsyncTMEM(dstTmem, srcGlobal string, sizeBytes int, mbarAddr string) string {
	return fmt.Sprintf(
		"cp.async.bulk.tmem.global.mbarrier::complete_tx::bytes [%s], [%s], %d, [%s];",
		dstTmem, srcGlobal, sizeBytes, mbarAddr,
	)
}

// EmitWaitParity returns a single PTX try_wait instruction testing completion of a phase parity.
func EmitWaitParity(mbarAddr string, phaseParity int, predReg string) string {
	return fmt.Sprintf("mbarrier.try_wait.parity.shared::cta.b64 %s, [%s], %d;", predReg, mbarAddr, phaseParity)
}

// EmitWaitLoop returns a spin-wait loop in PTX using nanosleep to minimize power/replay overhead.
func EmitWaitLoop(mbarAddr string, phaseParity int, labelPrefix string) string {
	lbl := labelPrefix
	if lbl == "" {
		lbl = "wait"
	}
	return fmt.Sprintf(
		"%s_loop:\n\tmbarrier.try_wait.parity.shared::cta.b64 %%p, [%s], %d;\n\t@!%%p nanosleep.u32 20;\n\t@!%%p bra %s_loop;",
		lbl, mbarAddr, phaseParity, lbl,
	)
}

// EmitInlineAsmInit returns a CUDA C++ inline assembly snippet initializing an mbarrier at mbarOffset.
func EmitInlineAsmInit(mbarOffset, arrivalCount int) string {
	return fmt.Sprintf(
		`asm volatile("mbarrier.init.shared::cta.b64 [%%0], %%1;" :: "r"(smem_base + %d), "r"(%d) : "memory");`,
		mbarOffset, arrivalCount,
	)
}

// EmitInlineAsmArriveExpectTx returns a CUDA C++ inline assembly snippet for arrive-expect-tx.
func EmitInlineAsmArriveExpectTx(mbarOffset, txBytes int) string {
	return fmt.Sprintf(
		`asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%%0], %%1;" :: "r"(smem_base + %d), "r"(%d) : "memory");`,
		mbarOffset, txBytes,
	)
}

// EmitInlineAsmTMA2D returns a CUDA C++ inline assembly snippet issuing a 2D TMA bulk asynchronous copy.
func EmitInlineAsmTMA2D(dstOffset int, descPtrReg, coord0Reg, coord1Reg string, mbarOffset int) string {
	return fmt.Sprintf(
		`asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%%0], [%%1, {%%2, %%3}], [%%4];"`+
			` :: "r"(smem_base + %d), "l"(%s), "r"(%s), "r"(%s), "r"(smem_base + %d) : "memory");`,
		dstOffset, descPtrReg, coord0Reg, coord1Reg, mbarOffset,
	)
}

// EmitInlineAsmTMEMAlloc returns a CUDA C++ inline assembly snippet allocating nCols in TMEM.
func EmitInlineAsmTMEMAlloc(dstSharedOffset int, nCols int) string {
	return fmt.Sprintf(
		`asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%%0], %%1;" :: "r"(smem_base + %d), "r"(%d) : "memory");`,
		dstSharedOffset, nCols,
	)
}

// EmitInlineAsmTMEMDealloc returns a CUDA C++ inline assembly snippet deallocating nCols from TMEM.
func EmitInlineAsmTMEMDealloc(taddrReg string, nCols int) string {
	return fmt.Sprintf(
		`asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %%0, %%1;" :: "r"(%s), "r"(%d) : "memory");`,
		taddrReg, nCols,
	)
}

// EmitInlineAsmTMA2DTMEM returns a CUDA C++ inline assembly snippet issuing a 2D TMA copy into TMEM.
func EmitInlineAsmTMA2DTMEM(dstTmemReg, descPtrReg, coord0Reg, coord1Reg string, mbarOffset int) string {
	return fmt.Sprintf(
		`asm volatile("cp.async.bulk.tensor.2d.tmem.global.mbarrier::complete_tx::bytes [%%0], [%%1, {%%2, %%3}], [%%4];"`+
			` :: "r"(%s), "l"(%s), "r"(%s), "r"(%s), "r"(smem_base + %d) : "memory");`,
		dstTmemReg, descPtrReg, coord0Reg, coord1Reg, mbarOffset,
	)
}

// EmitInlineAsmWait returns a CUDA C++ inline assembly snippet implementing a spin-wait loop on phase parity.
func EmitInlineAsmWait(mbarOffset int, phaseParity int) string {
	return fmt.Sprintf(
		`asm volatile(`+
			`"{\n\t"`+
			`".reg .pred p;\n\t"`+
			`"STAGE_WAIT:\n\t"`+
			`"mbarrier.try_wait.parity.shared::cta.b64 p, [%%0], %%1;\n\t"`+
			`"@!p nanosleep.u32 20;\n\t"`+
			`"@!p bra STAGE_WAIT;\n\t"`+
			`"}\n"`+
			` :: "r"(smem_base + %d), "r"(%d) : "memory");`,
		mbarOffset, phaseParity,
	)
}

// EmitPrologPTX generates PTX sequence to initialize all stage mbarriers in the pipeline
// and, if targeting TMEM, allocates the required TMEM columns.
func (p *Pipeline) EmitPrologPTX() string {
	var b strings.Builder
	b.WriteString("// Pipeline Prolog: initialize stage mbarriers\n")
	if p.Config.MemorySpace == MemorySpaceTMEM {
		cols := p.TotalTMEMColumns()
		fmt.Fprintf(&b, "// Allocate %d TMEM columns (%d bytes) across %d stages\n",
			cols, p.TotalTMEMBytes(), len(p.Stages))
		fmt.Fprintf(&b, "%s\n", EmitTMEMAlloc("smem_base", cols))
	}
	for i, s := range p.Stages {
		fmt.Fprintf(&b, "// Stage %d mbarrier init (offset %d, arrival count %d)\n",
			i, s.Barrier.Offset, s.Barrier.ArrivalCount)
		fmt.Fprintf(&b, "%s\n", EmitMBarrierInit(fmt.Sprintf("smem_base+%d", s.Barrier.Offset), s.Barrier.ArrivalCount))
	}
	return b.String()
}

// EmitStageLoadPTX generates PTX instructions to issue an asynchronous TMA load for the given stage.
func (p *Pipeline) EmitStageLoadPTX(stageIdx int, descAddr, coord0, coord1 string) string {
	if stageIdx < 0 || stageIdx >= len(p.Stages) {
		return "// Invalid stage index\n"
	}
	s := &p.Stages[stageIdx]
	mbarAddr := fmt.Sprintf("smem_base+%d", s.Barrier.Offset)

	var b strings.Builder
	if s.Buffer.Space == MemorySpaceTMEM {
		dstAddr := fmt.Sprintf("tmem_base+%d", s.Buffer.Offset)
		fmt.Fprintf(&b, "// Stage %d load to TMEM (%d bytes via TMA)\n", stageIdx, s.TMABox.PayloadBytes())
		fmt.Fprintf(&b, "%s\n", EmitArriveExpectTx(mbarAddr, s.TMABox.PayloadBytes()))
		fmt.Fprintf(&b, "%s\n", EmitTMABulkAsync2DTMEM(dstAddr, descAddr, coord0, coord1, mbarAddr))
	} else {
		dstAddr := fmt.Sprintf("smem_base+%d", s.Buffer.Offset)
		fmt.Fprintf(&b, "// Stage %d load (%d bytes via TMA)\n", stageIdx, s.TMABox.PayloadBytes())
		fmt.Fprintf(&b, "%s\n", EmitArriveExpectTx(mbarAddr, s.TMABox.PayloadBytes()))
		fmt.Fprintf(&b, "%s\n", EmitTMABulkAsync2D(dstAddr, descAddr, coord0, coord1, mbarAddr))
	}
	return b.String()
}

// EmitStageWaitPTX generates PTX instructions to wait for completion of the given stage.
func (p *Pipeline) EmitStageWaitPTX(stageIdx int, phaseParity int) string {
	if stageIdx < 0 || stageIdx >= len(p.Stages) {
		return "// Invalid stage index\n"
	}
	s := &p.Stages[stageIdx]
	mbarAddr := fmt.Sprintf("smem_base+%d", s.Barrier.Offset)
	label := fmt.Sprintf("stage_%d_wait", stageIdx)

	var b strings.Builder
	fmt.Fprintf(&b, "// Stage %d wait (phase parity %d)\n", stageIdx, phaseParity)
	fmt.Fprintf(&b, "%s\n", EmitWaitLoop(mbarAddr, phaseParity, label))
	return b.String()
}

// EmitEpilogPTX generates PTX sequence to deallocate TMEM columns if the pipeline targeted TMEM.
func (p *Pipeline) EmitEpilogPTX() string {
	var b strings.Builder
	if p.Config.MemorySpace == MemorySpaceTMEM {
		b.WriteString("// Pipeline Epilog: deallocate TMEM columns\n")
		cols := p.TotalTMEMColumns()
		fmt.Fprintf(&b, "%s\n", EmitTMEMDealloc("tmem_base", cols))
		fmt.Fprintf(&b, "%s\n", EmitTMEMRelinquishPermit())
	}
	return b.String()
}
