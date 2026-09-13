package cudapipeline_test

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudapipeline"
	"github.com/tmc/lib/nvidia/x/cudaptx"
)

func TestTMEMPipelineConfigurations(t *testing.T) {
	tests := []struct {
		name          string
		cfg           cudapipeline.PipelineConfig
		wantStages    int
		wantMinTMEM   int
		wantMinShared int
	}{
		{
			name: "double_buffering_sm100_fp16",
			cfg: cudapipeline.PipelineConfig{
				TargetArch:  cudapipeline.ArchSM100,
				NumStages:   2,
				MemorySpace: cudapipeline.MemorySpaceTMEM,
				Box: cudapipeline.TMABox{
					DimX:        64,
					DimY:        16,
					ElementSize: 2, // 2048 bytes per stage
				},
				ArrivalTokensPerStage: 1,
				MemoryLatencyCycles:   600,
				ComputeIntervalCycles: 400,
			},
			wantStages:    2,
			wantMinTMEM:   4096, // 2 stages * 2048 bytes
			wantMinShared: 16,   // 2 mbarriers * 8 bytes
		},
		{
			name: "triple_buffering_sm100_fp8",
			cfg: cudapipeline.PipelineConfig{
				TargetArch:  cudapipeline.ArchSM100,
				NumStages:   3,
				MemorySpace: cudapipeline.MemorySpaceTMEM,
				Box: cudapipeline.TMABox{
					DimX:        128,
					DimY:        32,
					ElementSize: 1, // 4096 bytes per stage
				},
				ArrivalTokensPerStage: 2,
				MemoryLatencyCycles:   800,
				ComputeIntervalCycles: 300,
			},
			wantStages:    3,
			wantMinTMEM:   12288, // 3 stages * 4096 bytes
			wantMinShared: 24,    // 3 mbarriers * 8 bytes
		},
		{
			name: "quad_buffering_sm121a_tf32",
			cfg: cudapipeline.PipelineConfig{
				TargetArch:  cudapipeline.ArchSM121a,
				NumStages:   4,
				MemorySpace: cudapipeline.MemorySpaceTMEM,
				Box: cudapipeline.TMABox{
					DimX:        32,
					DimY:        16,
					ElementSize: 4, // 2048 bytes per stage
				},
				ArrivalTokensPerStage: 1,
				MemoryLatencyCycles:   1000,
				ComputeIntervalCycles: 350,
			},
			wantStages:    4,
			wantMinTMEM:   8192, // 4 stages * 2048 bytes
			wantMinShared: 32,   // 4 mbarriers * 8 bytes
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pipe, err := cudapipeline.NewPipeline(tt.cfg)
			if err != nil {
				t.Fatalf("NewPipeline failed: %v", err)
			}
			if pipe.NumStages() != tt.wantStages {
				t.Errorf("NumStages = %d, want %d", pipe.NumStages(), tt.wantStages)
			}
			if pipe.TotalTMEMBytes() < tt.wantMinTMEM {
				t.Errorf("TotalTMEMBytes = %d, want >= %d", pipe.TotalTMEMBytes(), tt.wantMinTMEM)
			}
			if pipe.TotalSharedBytes() < tt.wantMinShared {
				t.Errorf("TotalSharedBytes = %d, want >= %d", pipe.TotalSharedBytes(), tt.wantMinShared)
			}
			if cols := pipe.TotalTMEMColumns(); cols <= 0 || cols%32 != 0 {
				t.Errorf("TotalTMEMColumns = %d, want positive multiple of 32", cols)
			}

			// Verify all stages reside in TMEM with proper alignment
			for i := 0; i < pipe.NumStages(); i++ {
				s := pipe.Stage(i)
				if s.Buffer.Space != cudapipeline.MemorySpaceTMEM {
					t.Errorf("stage %d Space = %q, want %q", i, s.Buffer.Space, cudapipeline.MemorySpaceTMEM)
				}
				if s.Buffer.Offset%16 != 0 {
					t.Errorf("stage %d buffer offset %d not 16-byte aligned", i, s.Buffer.Offset)
				}
				if s.Barrier.Offset%8 != 0 {
					t.Errorf("stage %d barrier offset %d not 8-byte aligned", i, s.Barrier.Offset)
				}
			}

			report := pipe.Validate()
			if !report.Valid() {
				t.Errorf("expected valid pipeline report, got errors: %v", report.Errors())
			}
		})
	}
}

func TestTMEMValidation_Errors(t *testing.T) {
	tests := []struct {
		name         string
		setup        func() *cudapipeline.Pipeline
		wantSeverity cudapipeline.Severity
		wantCategory cudapipeline.Category
		wantText     string
	}{
		{
			name: "unsupported_arch_sm90",
			setup: func() *cudapipeline.Pipeline {
				arch, _ := cudapipeline.ArchProfileFor(cudapipeline.ArchSM90)
				return &cudapipeline.Pipeline{
					Arch: arch,
					Config: cudapipeline.PipelineConfig{
						TargetArch:  cudapipeline.ArchSM90,
						NumStages:   2,
						MemorySpace: cudapipeline.MemorySpaceTMEM,
					},
					Stages: []cudapipeline.Stage{
						{
							Index: 0,
							Buffer: cudapipeline.Buffer{
								Space:     cudapipeline.MemorySpaceTMEM,
								Offset:    0,
								SizeBytes: 1024,
							},
							TMABox:  cudapipeline.TMABox{DimX: 32, DimY: 16, ElementSize: 2, StrideBytes: 64},
							Barrier: cudapipeline.Barrier{Offset: 0, ArrivalCount: 1, TransactionBytes: 1024},
						},
						{
							Index: 1,
							Buffer: cudapipeline.Buffer{
								Space:     cudapipeline.MemorySpaceTMEM,
								Offset:    1024,
								SizeBytes: 1024,
							},
							TMABox:  cudapipeline.TMABox{DimX: 32, DimY: 16, ElementSize: 2, StrideBytes: 64},
							Barrier: cudapipeline.Barrier{Offset: 8, ArrivalCount: 1, TransactionBytes: 1024},
						},
					},
				}
			},
			wantSeverity: cudapipeline.SeverityError,
			wantCategory: cudapipeline.CatHardwareConstraint,
			wantText:     "does not support Tensor Memory",
		},
		{
			name: "tmem_capacity_exceeded",
			setup: func() *cudapipeline.Pipeline {
				arch, _ := cudapipeline.ArchProfileFor(cudapipeline.ArchSM100)
				// Create 4 stages of 70KB each = 280KB > 256KB capacity
				stageSize := 70 * 1024
				stages := make([]cudapipeline.Stage, 4)
				for i := 0; i < 4; i++ {
					stages[i] = cudapipeline.Stage{
						Index: i,
						Buffer: cudapipeline.Buffer{
							Space:     cudapipeline.MemorySpaceTMEM,
							Offset:    i * stageSize,
							SizeBytes: stageSize,
						},
						TMABox:  cudapipeline.TMABox{DimX: 64, DimY: 560, ElementSize: 2, StrideBytes: 128},
						Barrier: cudapipeline.Barrier{Offset: i * 8, ArrivalCount: 1, TransactionBytes: 64 * 560 * 2},
					}
				}
				return &cudapipeline.Pipeline{
					Arch: arch,
					Config: cudapipeline.PipelineConfig{
						TargetArch:  cudapipeline.ArchSM100,
						NumStages:   4,
						MemorySpace: cudapipeline.MemorySpaceTMEM,
					},
					Stages: stages,
				}
			},
			wantSeverity: cudapipeline.SeverityError,
			wantCategory: cudapipeline.CatBufferCapacity,
			wantText:     "exceeds maximum SM capacity limit",
		},
		{
			name: "stage_buffer_boundary_exceeded",
			setup: func() *cudapipeline.Pipeline {
				arch, _ := cudapipeline.ArchProfileFor(cudapipeline.ArchSM100)
				// Buffer offset near end of TMEM, overflowing 256KB limit
				return &cudapipeline.Pipeline{
					Arch: arch,
					Config: cudapipeline.PipelineConfig{
						TargetArch:  cudapipeline.ArchSM100,
						NumStages:   2,
						MemorySpace: cudapipeline.MemorySpaceTMEM,
					},
					Stages: []cudapipeline.Stage{
						{
							Index: 0,
							Buffer: cudapipeline.Buffer{
								Space:     cudapipeline.MemorySpaceTMEM,
								Offset:    0,
								SizeBytes: 2048,
							},
							TMABox:  cudapipeline.TMABox{DimX: 64, DimY: 16, ElementSize: 2, StrideBytes: 128},
							Barrier: cudapipeline.Barrier{Offset: 0, ArrivalCount: 1, TransactionBytes: 2048},
						},
						{
							Index: 1,
							Buffer: cudapipeline.Buffer{
								Space:     cudapipeline.MemorySpaceTMEM,
								Offset:    261120, // 255 KiB
								SizeBytes: 4096,   // ends at 265216 > 262144
							},
							TMABox:  cudapipeline.TMABox{DimX: 64, DimY: 32, ElementSize: 2, StrideBytes: 128},
							Barrier: cudapipeline.Barrier{Offset: 8, ArrivalCount: 1, TransactionBytes: 4096},
						},
					},
				}
			},
			wantSeverity: cudapipeline.SeverityError,
			wantCategory: cudapipeline.CatBufferCapacity,
			wantText:     "exceeds maximum SM TMEM capacity",
		},
		{
			name: "tmem_buffer_unaligned",
			setup: func() *cudapipeline.Pipeline {
				pipe, _ := cudapipeline.NewPipeline(cudapipeline.PipelineConfig{
					TargetArch:  cudapipeline.ArchSM100,
					NumStages:   2,
					MemorySpace: cudapipeline.MemorySpaceTMEM,
					Box:         cudapipeline.TMABox{DimX: 64, DimY: 16, ElementSize: 2},
				})
				pipe.Stages[0].Buffer.Offset = 13 // not 16-byte aligned
				return pipe
			},
			wantSeverity: cudapipeline.SeverityError,
			wantCategory: cudapipeline.CatAlignmentStride,
			wantText:     "not 16-byte aligned",
		},
		{
			name: "tmem_buffer_suboptimal_alignment_warning",
			setup: func() *cudapipeline.Pipeline {
				pipe, _ := cudapipeline.NewPipeline(cudapipeline.PipelineConfig{
					TargetArch:  cudapipeline.ArchSM100,
					NumStages:   2,
					MemorySpace: cudapipeline.MemorySpaceTMEM,
					Box:         cudapipeline.TMABox{DimX: 64, DimY: 16, ElementSize: 2},
				})
				pipe.Stages[0].Buffer.Offset = 16 // 16-byte aligned, but not 128-byte aligned
				return pipe
			},
			wantSeverity: cudapipeline.SeverityWarning,
			wantCategory: cudapipeline.CatAlignmentStride,
			wantText:     "not 128-byte aligned",
		},
		{
			name: "tmem_buffer_capacity_smaller_than_payload",
			setup: func() *cudapipeline.Pipeline {
				pipe, _ := cudapipeline.NewPipeline(cudapipeline.PipelineConfig{
					TargetArch:  cudapipeline.ArchSM100,
					NumStages:   2,
					MemorySpace: cudapipeline.MemorySpaceTMEM,
					Box:         cudapipeline.TMABox{DimX: 64, DimY: 16, ElementSize: 2},
				})
				pipe.Stages[0].Buffer.SizeBytes = 1024 // payload is 2048
				return pipe
			},
			wantSeverity: cudapipeline.SeverityError,
			wantCategory: cudapipeline.CatBufferCapacity,
			wantText:     "TMEM buffer size (1024 bytes) is smaller than TMA payload",
		},
		{
			name: "barrier_transaction_undercounting",
			setup: func() *cudapipeline.Pipeline {
				pipe, _ := cudapipeline.NewPipeline(cudapipeline.PipelineConfig{
					TargetArch:  cudapipeline.ArchSM100,
					NumStages:   2,
					MemorySpace: cudapipeline.MemorySpaceTMEM,
					Box:         cudapipeline.TMABox{DimX: 64, DimY: 16, ElementSize: 2},
				})
				pipe.Stages[0].Barrier.TransactionBytes = 4096 // payload is 2048; TMA delivers fewer bytes than expected
				return pipe
			},
			wantSeverity: cudapipeline.SeverityError,
			wantCategory: cudapipeline.CatTransactionMismatch,
			wantText:     "under-counting will cause permanent deadlock",
		},
		{
			name: "barrier_transaction_overcounting",
			setup: func() *cudapipeline.Pipeline {
				pipe, _ := cudapipeline.NewPipeline(cudapipeline.PipelineConfig{
					TargetArch:  cudapipeline.ArchSM100,
					NumStages:   2,
					MemorySpace: cudapipeline.MemorySpaceTMEM,
					Box:         cudapipeline.TMABox{DimX: 64, DimY: 16, ElementSize: 2},
				})
				pipe.Stages[0].Barrier.TransactionBytes = 1024 // payload is 2048; TMA delivers more bytes than expected
				return pipe
			},
			wantSeverity: cudapipeline.SeverityError,
			wantCategory: cudapipeline.CatTransactionMismatch,
			wantText:     "over-counting will cause premature release",
		},
		{
			name: "overlapping_tmem_buffers",
			setup: func() *cudapipeline.Pipeline {
				pipe, _ := cudapipeline.NewPipeline(cudapipeline.PipelineConfig{
					TargetArch:  cudapipeline.ArchSM100,
					NumStages:   2,
					MemorySpace: cudapipeline.MemorySpaceTMEM,
					Box:         cudapipeline.TMABox{DimX: 64, DimY: 16, ElementSize: 2},
				})
				// Force stage 1 to overlap with stage 0
				pipe.Stages[1].Buffer.Offset = pipe.Stages[0].Buffer.Offset + 128
				return pipe
			},
			wantSeverity: cudapipeline.SeverityError,
			wantCategory: cudapipeline.CatBufferCapacity,
			wantText:     "overlaps with stage 0 TMEM buffer",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pipe := tt.setup()
			report := pipe.Validate()

			var found bool
			for _, iss := range report.Issues {
				if iss.Severity == tt.wantSeverity && iss.Category == tt.wantCategory && strings.Contains(iss.Message, tt.wantText) {
					found = true
					break
				}
			}
			if !found {
				t.Errorf("issue not found: want severity=%v category=%v containing %q\nAll issues:\n%s",
					tt.wantSeverity, tt.wantCategory, tt.wantText, report.String())
			}
		})
	}
}

func TestNewPipeline_TMEM_Rejection(t *testing.T) {
	// Creating a TMEM pipeline on an architecture without TMEM support (e.g. sm_90) must fail at construction.
	_, err := cudapipeline.NewPipeline(cudapipeline.PipelineConfig{
		TargetArch:  cudapipeline.ArchSM90,
		NumStages:   2,
		MemorySpace: cudapipeline.MemorySpaceTMEM,
		Box:         cudapipeline.TMABox{DimX: 32, DimY: 16, ElementSize: 2},
	})
	if err == nil {
		t.Fatal("expected error creating TMEM pipeline on sm_90, got nil")
	}
	if !strings.Contains(err.Error(), "does not support Tensor Memory") {
		t.Errorf("error %q should mention lack of TMEM support", err.Error())
	}
}

func TestTMEMPTXEmission(t *testing.T) {
	pipe, err := cudapipeline.NewPipeline(cudapipeline.PipelineConfig{
		TargetArch:  cudapipeline.ArchSM100,
		NumStages:   3,
		MemorySpace: cudapipeline.MemorySpaceTMEM,
		Box: cudapipeline.TMABox{
			DimX:        64,
			DimY:        16,
			ElementSize: 2, // 2048 bytes per stage
		},
	})
	if err != nil {
		t.Fatalf("NewPipeline failed: %v", err)
	}

	// 1. Prolog emission should include tcgen05.alloc and mbarrier.init
	prolog := pipe.EmitPrologPTX()
	if !strings.Contains(prolog, "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [smem_base], 32;") {
		t.Errorf("prolog missing expected tcgen05.alloc:\n%s", prolog)
	}
	if !strings.Contains(prolog, "mbarrier.init.shared::cta.b64 [smem_base+0], 1;") {
		t.Errorf("prolog missing mbarrier init for stage 0:\n%s", prolog)
	}

	// 2. Stage load emission should use cp.async.bulk.tensor.2d.tmem.global
	load0 := pipe.EmitStageLoadPTX(0, "%rd_desc", "0", "0")
	if !strings.Contains(load0, "mbarrier.arrive.expect_tx.shared::cta.b64 _, [smem_base+0], 2048;") {
		t.Errorf("load missing expect_tx:\n%s", load0)
	}
	if !strings.Contains(load0, "cp.async.bulk.tensor.2d.tmem.global.mbarrier::complete_tx::bytes [tmem_base+0], [%rd_desc, {0, 0}], [smem_base+0];") {
		t.Errorf("load missing cp.async.bulk.tensor.2d.tmem.global:\n%s", load0)
	}

	// 3. Stage wait emission
	wait0 := pipe.EmitStageWaitPTX(0, 0)
	if !strings.Contains(wait0, "mbarrier.try_wait.parity.shared::cta.b64 %p, [smem_base+0], 0;") {
		t.Errorf("wait missing try_wait:\n%s", wait0)
	}

	// 4. Epilog emission should include tcgen05.dealloc and relinquish permit
	epilog := pipe.EmitEpilogPTX()
	if !strings.Contains(epilog, "tcgen05.dealloc.cta_group::1.sync.aligned.b32 tmem_base, 32;") {
		t.Errorf("epilog missing tcgen05.dealloc:\n%s", epilog)
	}
	if !strings.Contains(epilog, "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;") {
		t.Errorf("epilog missing relinquish permit:\n%s", epilog)
	}

	// 5. Standalone TMEM emitters
	allocStr := cudapipeline.EmitTMEMAlloc("smem_base+128", 64)
	if allocStr != "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [smem_base+128], 64;" {
		t.Errorf("EmitTMEMAlloc = %q", allocStr)
	}

	deallocStr := cudapipeline.EmitTMEMDealloc("%r_taddr", 64)
	if deallocStr != "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %r_taddr, 64;" {
		t.Errorf("EmitTMEMDealloc = %q", deallocStr)
	}

	relinquishStr := cudapipeline.EmitTMEMRelinquishPermit()
	if relinquishStr != "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" {
		t.Errorf("EmitTMEMRelinquishPermit = %q", relinquishStr)
	}

	bulk1dStr := cudapipeline.EmitTMABulkAsyncTMEM("taddr", "gaddr", 1024, "bar")
	if bulk1dStr != "cp.async.bulk.tmem.global.mbarrier::complete_tx::bytes [taddr], [gaddr], 1024, [bar];" {
		t.Errorf("EmitTMABulkAsyncTMEM = %q", bulk1dStr)
	}

	// Inline assembly strings
	asmAlloc := cudapipeline.EmitInlineAsmTMEMAlloc(0, 32)
	if !strings.Contains(asmAlloc, "tcgen05.alloc") {
		t.Errorf("EmitInlineAsmTMEMAlloc = %q", asmAlloc)
	}
	asmDealloc := cudapipeline.EmitInlineAsmTMEMDealloc("taddr", 32)
	if !strings.Contains(asmDealloc, "tcgen05.dealloc") {
		t.Errorf("EmitInlineAsmTMEMDealloc = %q", asmDealloc)
	}
	asmTma := cudapipeline.EmitInlineAsmTMA2DTMEM("taddr", "desc", "0", "0", 0)
	if !strings.Contains(asmTma, "cp.async.bulk.tensor.2d.tmem.global") {
		t.Errorf("EmitInlineAsmTMA2DTMEM = %q", asmTma)
	}
}

func TestValidatePTX_TMEM_Table(t *testing.T) {
	tests := []struct {
		name          string
		ptx           string
		opts          []cudapipeline.Option
		wantViolation string
		wantCount     int
	}{
		{
			name: "valid_2stage_tmem_pipeline",
			ptx: `
.version 8.8
.target sm_100
.address_size 64

.visible .entry valid_tmem_kernel(
    .param .u64 tmap
)
{
    .reg .b64 %rd_tmap;
    .reg .b64 %state;
    .reg .b32 %taddr;
    .reg .pred %p0;
    .shared .align 8 .b64 bar;

    ld.param.u64 %rd_tmap, [tmap];

    // Allocate 32 TMEM columns
    tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%taddr], 32;

    // Initialize mbarrier
    mbarrier.init.shared::cta.b64 [bar], 1;

    // Phase 0: expect 2048 bytes
    mbarrier.arrive.expect_tx.shared::cta.b64 %state, [bar], 2048;

    // TMA copy to TMEM transfers 2048 bytes
    cp.async.bulk.tensor.2d.tmem.global.mbarrier::complete_tx::bytes [0], [%rd_tmap, {0, 0}], [bar];

    // Wait on phase 0 (parity 0)
    mbarrier.try_wait.parity.shared::cta.b64 %p0, [bar], 0;

    // Phase 1: expect 2048 bytes
    mbarrier.arrive.expect_tx.shared::cta.b64 %state, [bar], 2048;

    // TMA copy to TMEM transfers 2048 bytes
    cp.async.bulk.tensor.2d.tmem.global.mbarrier::complete_tx::bytes [2048], [%rd_tmap, {16, 0}], [bar];

    // Wait on phase 1 (parity 1)
    mbarrier.try_wait.parity.shared::cta.b64 %p0, [bar], 1;

    // Deallocate TMEM
    tcgen05.dealloc.cta_group::1.sync.aligned.b32 %taddr, 32;
    tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;

    ret;
}
`,
			opts: []cudapipeline.Option{
				cudapipeline.WithTensorSize("tmap", 2048),
			},
			wantCount: 0,
		},
		{
			name: "tmem_copy_mismatched_tx_bytes",
			ptx: `
.version 8.8
.target sm_100
.address_size 64

.visible .entry mismatched_tmem_kernel(
    .param .u64 tmap
)
{
    .reg .b64 %rd_tmap;
    .reg .b64 %state;
    .reg .pred %p0;
    .shared .align 8 .b64 bar;

    ld.param.u64 %rd_tmap, [tmap];
    mbarrier.init.shared::cta.b64 [bar], 1;

    // Expect 4096 bytes
    mbarrier.arrive.expect_tx.shared::cta.b64 %state, [bar], 4096;

    // TMA copy to TMEM only copies 2048 bytes
    cp.async.bulk.tensor.2d.tmem.global.mbarrier::complete_tx::bytes [0], [%rd_tmap, {0, 0}], [bar];

    // Wait triggers transaction byte mismatch error
    mbarrier.try_wait.parity.shared::cta.b64 %p0, [bar], 0;
    ret;
}
`,
			opts: []cudapipeline.Option{
				cudapipeline.WithTensorSize("tmap", 2048),
			},
			wantViolation: cudapipeline.ErrMismatchedTxBytes,
			wantCount:     1,
		},
		{
			name: "tmem_invalid_column_count",
			ptx: `
.version 8.8
.target sm_100
.address_size 64

.visible .entry invalid_cols_kernel()
{
    .reg .b32 %taddr;
    // 17 columns is not a multiple of 32
    tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%taddr], 17;
    ret;
}
`,
			wantViolation: cudapipeline.ErrInvalidTMEMColumns,
			wantCount:     1,
		},
		{
			name: "tmem_unsupported_on_sm90",
			ptx: `
.version 8.8
.target sm_90
.address_size 64

.visible .entry sm90_tmem_kernel()
{
    .reg .b32 %taddr;
    tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%taddr], 32;
    ret;
}
`,
			opts: []cudapipeline.Option{
				cudapipeline.WithTargetArch(cudapipeline.ArchSM90),
			},
			wantViolation: cudapipeline.ErrUnsupportedTMEM,
			wantCount:     1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mod, err := cudaptx.Parse(strings.NewReader(tt.ptx))
			if err != nil {
				t.Fatalf("Parse failed: %v", err)
			}
			violations, err := cudapipeline.ValidatePTX(mod, tt.opts...)
			if err != nil {
				t.Fatalf("ValidatePTX failed: %v", err)
			}
			if len(violations) != tt.wantCount {
				t.Errorf("got %d violations, want %d", len(violations), tt.wantCount)
				for _, v := range violations {
					t.Logf("  violation: %v", v)
				}
			}
			if tt.wantViolation != "" {
				var found bool
				for _, v := range violations {
					if v.Violation == tt.wantViolation {
						found = true
						break
					}
				}
				if !found {
					t.Errorf("expected violation %q not found in: %v", tt.wantViolation, violations)
				}
			}
		})
	}
}
