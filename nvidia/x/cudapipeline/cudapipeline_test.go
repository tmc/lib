package cudapipeline

import (
	"strings"
	"testing"
)

func TestPipelineConfigurations(t *testing.T) {
	tests := []struct {
		name         string
		cfg          PipelineConfig
		wantStages   int
		wantPayload  int
		wantMinBytes int
	}{
		{
			name: "double-buffering sm_121a fp16",
			cfg: PipelineConfig{
				TargetArch: ArchSM121a,
				NumStages:  2,
				Box: TMABox{
					DimX:        64,
					DimY:        32,
					ElementSize: 2, // 4096 bytes per stage
				},
				MemoryLatencyCycles:   600,
				ComputeIntervalCycles: 600,
			},
			wantStages:   2,
			wantPayload:  4096,
			wantMinBytes: 8192,
		},
		{
			name: "triple-buffering sm_121a fp8",
			cfg: PipelineConfig{
				TargetArch: ArchSM121a,
				NumStages:  3,
				Box: TMABox{
					DimX:        128,
					DimY:        64,
					ElementSize: 1, // 8192 bytes per stage
				},
				MemoryLatencyCycles:   600,
				ComputeIntervalCycles: 300,
			},
			wantStages:   3,
			wantPayload:  8192,
			wantMinBytes: 24576,
		},
		{
			name: "double-buffering sm_90 tf32",
			cfg: PipelineConfig{
				TargetArch: ArchSM90,
				NumStages:  2,
				Box: TMABox{
					DimX:        32,
					DimY:        16,
					ElementSize: 4, // 2048 bytes per stage
				},
			},
			wantStages:   2,
			wantPayload:  2048,
			wantMinBytes: 4096,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pipe, err := NewPipeline(tt.cfg)
			if err != nil {
				t.Fatalf("NewPipeline failed: %v", err)
			}
			if pipe.NumStages() != tt.wantStages {
				t.Errorf("NumStages = %d, want %d", pipe.NumStages(), tt.wantStages)
			}
			for i := 0; i < pipe.NumStages(); i++ {
				s := pipe.Stage(i)
				if s == nil {
					t.Fatalf("Stage(%d) is nil", i)
				}
				if s.TMABox.PayloadBytes() != tt.wantPayload {
					t.Errorf("Stage %d payload = %d, want %d", i, s.TMABox.PayloadBytes(), tt.wantPayload)
				}
				if s.Barrier.TransactionBytes != tt.wantPayload {
					t.Errorf("Stage %d barrier tx bytes = %d, want %d", i, s.Barrier.TransactionBytes, tt.wantPayload)
				}
				if s.Buffer.Offset%128 != 0 {
					t.Errorf("Stage %d buffer offset %d not 128-byte aligned", i, s.Buffer.Offset)
				}
				if s.Barrier.Offset%8 != 0 {
					t.Errorf("Stage %d barrier offset %d not 8-byte aligned", i, s.Barrier.Offset)
				}
			}
			if pipe.TotalSharedBytes() < tt.wantMinBytes {
				t.Errorf("TotalSharedBytes = %d, want >= %d", pipe.TotalSharedBytes(), tt.wantMinBytes)
			}

			report := pipe.Validate()
			if !report.Valid() {
				t.Errorf("expected valid pipeline, got errors: %v", report.Errors())
			}
		})
	}
}

func TestValidationMismatchesAndErrors(t *testing.T) {
	tests := []struct {
		name         string
		mutate       func(p *Pipeline)
		wantCategory Category
		wantSeverity Severity
		wantSubstr   string
	}{
		{
			name: "transaction under-counting",
			mutate: func(p *Pipeline) {
				// Barrier expects more bytes than TMA delivers
				p.Stages[0].Barrier.TransactionBytes = p.Stages[0].TMABox.PayloadBytes() + 128
			},
			wantCategory: CatTransactionMismatch,
			wantSeverity: SeverityError,
			wantSubstr:   "under-counting",
		},
		{
			name: "transaction over-counting",
			mutate: func(p *Pipeline) {
				// Barrier expects fewer bytes than TMA delivers
				p.Stages[0].Barrier.TransactionBytes = p.Stages[0].TMABox.PayloadBytes() - 64
			},
			wantCategory: CatTransactionMismatch,
			wantSeverity: SeverityError,
			wantSubstr:   "over-counting",
		},
		{
			name: "buffer capacity insufficient",
			mutate: func(p *Pipeline) {
				p.Stages[1].Buffer.SizeBytes = p.Stages[1].TMABox.PayloadBytes() - 16
			},
			wantCategory: CatBufferCapacity,
			wantSeverity: SeverityError,
			wantSubstr:   "smaller than TMA payload",
		},
		{
			name: "buffer offset not 16-byte aligned",
			mutate: func(p *Pipeline) {
				p.Stages[0].Buffer.Offset = 18 // unaligned
			},
			wantCategory: CatAlignmentStride,
			wantSeverity: SeverityError,
			wantSubstr:   "not 16-byte aligned",
		},
		{
			name: "buffer offset 16-byte aligned but not 128-byte aligned",
			mutate: func(p *Pipeline) {
				p.Stages[0].Buffer.Offset = 48 // 16-byte aligned, but not 128
			},
			wantCategory: CatAlignmentStride,
			wantSeverity: SeverityWarning,
			wantSubstr:   "not 128-byte aligned",
		},
		{
			name: "barrier offset not 8-byte aligned",
			mutate: func(p *Pipeline) {
				p.Stages[0].Barrier.Offset = 12 // unaligned for b64
			},
			wantCategory: CatAlignmentStride,
			wantSeverity: SeverityError,
			wantSubstr:   "not 8-byte aligned",
		},
		{
			name: "TMA box row not multiple of 16 bytes",
			mutate: func(p *Pipeline) {
				p.Stages[0].TMABox.DimX = 15
				p.Stages[0].TMABox.ElementSize = 1 // 15 bytes
				p.Stages[0].Barrier.TransactionBytes = p.Stages[0].TMABox.PayloadBytes()
			},
			wantCategory: CatAlignmentStride,
			wantSeverity: SeverityError,
			wantSubstr:   "not a multiple of 16 bytes",
		},
		{
			name: "TMA box row exceeds 128 bytes limit",
			mutate: func(p *Pipeline) {
				p.Stages[0].TMABox.DimX = 128
				p.Stages[0].TMABox.ElementSize = 2 // 256 bytes > 128 max box width
				p.Stages[0].Barrier.TransactionBytes = p.Stages[0].TMABox.PayloadBytes()
			},
			wantCategory: CatAlignmentStride,
			wantSeverity: SeverityError,
			wantSubstr:   "exceeds maximum 128 bytes",
		},
		{
			name: "initial barrier phase non-zero",
			mutate: func(p *Pipeline) {
				p.Stages[0].Barrier.Phase = 1
			},
			wantCategory: CatPhaseParity,
			wantSeverity: SeverityError,
			wantSubstr:   "initial mbarrier phase",
		},
		{
			name: "barrier arrival count zero",
			mutate: func(p *Pipeline) {
				p.Stages[0].Barrier.ArrivalCount = 0
			},
			wantCategory: CatHardwareConstraint,
			wantSeverity: SeverityError,
			wantSubstr:   "arrival count",
		},
		{
			name: "overlapping stage buffers",
			mutate: func(p *Pipeline) {
				p.Stages[1].Buffer.Offset = p.Stages[0].Buffer.Offset + 128 // overlap
			},
			wantCategory: CatBufferCapacity,
			wantSeverity: SeverityError,
			wantSubstr:   "overlaps",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pipe, err := NewPipeline(PipelineConfig{
				TargetArch: ArchSM121a,
				NumStages:  2,
				Box: TMABox{
					DimX:        64,
					DimY:        16,
					ElementSize: 2, // 128 bytes row, 2048 payload
				},
			})
			if err != nil {
				t.Fatalf("setup failed: %v", err)
			}

			tt.mutate(pipe)
			report := pipe.Validate()

			var found *Issue
			for i := range report.Issues {
				iss := &report.Issues[i]
				if iss.Category == tt.wantCategory && iss.Severity == tt.wantSeverity &&
					strings.Contains(iss.Message, tt.wantSubstr) {
					found = iss
					break
				}
			}

			if found == nil {
				t.Fatalf("expected issue with category=%s severity=%s substr=%q; report had:\n%s",
					tt.wantCategory, tt.wantSeverity, tt.wantSubstr, report.String())
			}
		})
	}
}

func TestPhaseParityVerification(t *testing.T) {
	tests := []struct {
		name         string
		numStages    int
		prodParities []int
		consParities []int
		wantErrors   int
	}{
		{
			name:         "valid 2-stage 4-step sequence",
			numStages:    2,
			prodParities: []int{0, 0, 1, 1},
			consParities: []int{0, 0, 1, 1},
			wantErrors:   0,
		},
		{
			name:         "valid 3-stage 6-step sequence",
			numStages:    3,
			prodParities: []int{0, 0, 0, 1, 1, 1},
			consParities: []int{0, 0, 0, 1, 1, 1},
			wantErrors:   0,
		},
		{
			name:         "consumer inverted phase parity deadlock",
			numStages:    2,
			prodParities: []int{0, 0, 1, 1},
			consParities: []int{1, 1, 0, 0},
			wantErrors:   8, // Each mismatched iteration flags consP != expected and prodP != consP
		},
		{
			name:         "producer incorrect parity flip",
			numStages:    2,
			prodParities: []int{0, 1, 1, 1}, // step 1 should be 0
			consParities: []int{0, 0, 1, 1},
			wantErrors:   2,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			issues := ValidatePhaseParity(tt.numStages, tt.prodParities, tt.consParities)
			if len(issues) != tt.wantErrors {
				t.Errorf("got %d issues, want %d; issues: %v", len(issues), tt.wantErrors, issues)
			}
		})
	}
}

func TestLatencyHidingAnalysis(t *testing.T) {
	tests := []struct {
		name         string
		stages       int
		memLatency   int
		compute      int
		wantHidden   float64
		wantNeeded   int
		wantWarnings int
	}{
		{
			name:         "double buffering fully hides latency",
			stages:       2,
			memLatency:   600,
			compute:      600,
			wantHidden:   100.0,
			wantNeeded:   2,
			wantWarnings: 0,
		},
		{
			name:         "double buffering insufficient latency hiding",
			stages:       2,
			memLatency:   600,
			compute:      300,
			wantHidden:   50.0,
			wantNeeded:   3,
			wantWarnings: 1,
		},
		{
			name:         "triple buffering fully hides latency",
			stages:       3,
			memLatency:   600,
			compute:      300,
			wantHidden:   100.0,
			wantNeeded:   3,
			wantWarnings: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pipe, err := NewPipeline(PipelineConfig{
				TargetArch:            ArchSM121a,
				NumStages:             tt.stages,
				Box:                   TMABox{DimX: 64, DimY: 16, ElementSize: 2},
				MemoryLatencyCycles:   tt.memLatency,
				ComputeIntervalCycles: tt.compute,
			})
			if err != nil {
				t.Fatalf("setup failed: %v", err)
			}

			report := pipe.Validate()
			if report.LatencyHiddenPercent != tt.wantHidden {
				t.Errorf("LatencyHiddenPercent = %.1f, want %.1f", report.LatencyHiddenPercent, tt.wantHidden)
			}
			if report.StagesNeeded != tt.wantNeeded {
				t.Errorf("StagesNeeded = %d, want %d", report.StagesNeeded, tt.wantNeeded)
			}
			if len(report.Warnings()) != tt.wantWarnings {
				t.Errorf("Warnings count = %d, want %d", len(report.Warnings()), tt.wantWarnings)
			}
		})
	}
}

func TestArchConstraints(t *testing.T) {
	// sm_121a verification
	p121, err := ArchProfileFor(ArchSM121a)
	if err != nil {
		t.Fatalf("ArchProfileFor(sm_121a) error: %v", err)
	}
	if !p121.SupportsTMA {
		t.Errorf("sm_121a should support TMA")
	}
	if !p121.SupportsMBarrier {
		t.Errorf("sm_121a should support mbarrier")
	}
	if !p121.SupportsMultimem {
		t.Errorf("sm_121a should support multimem")
	}
	if !p121.SupportsTMEM {
		t.Errorf("sm_121a should support TMEM")
	}
	if err := p121.ValidateFeature(FeatureTMEM); err != nil {
		t.Errorf("sm_121a should validate TMEM, got: %v", err)
	}
	// Verify sm_121a hardware reality: rejects wgmma and tcgen05
	if err := p121.ValidateFeature(FeatureWGMMA); err == nil {
		t.Errorf("expected sm_121a to reject wgmma, got nil error")
	} else if !strings.Contains(err.Error(), "rejects wgmma") {
		t.Errorf("expected error to explain sm_121a rejects wgmma, got: %v", err)
	}
	if err := p121.ValidateFeature(FeatureTCGen05); err == nil {
		t.Errorf("expected sm_121a to reject tcgen05, got nil error")
	} else if !strings.Contains(err.Error(), "rejects tcgen05") {
		t.Errorf("expected error to explain sm_121a rejects tcgen05, got: %v", err)
	}

	// sm_90 verification
	p90, err := ArchProfileFor(ArchSM90)
	if err != nil {
		t.Fatalf("ArchProfileFor(sm_90) error: %v", err)
	}
	if !p90.SupportsWGMMA {
		t.Errorf("sm_90 should support wgmma")
	}
	if p90.SupportsTCGen05 {
		t.Errorf("sm_90 should not support tcgen05")
	}
	if p90.SupportsTMEM {
		t.Errorf("sm_90 should not support TMEM")
	}
	if err := p90.ValidateFeature(FeatureTMEM); err == nil {
		t.Errorf("expected sm_90 to reject TMEM, got nil error")
	} else if !strings.Contains(err.Error(), "does not support tmem") {
		t.Errorf("expected error to explain sm_90 does not support tmem, got: %v", err)
	}

	// sm_100 verification
	p100, err := ArchProfileFor(ArchSM100)
	if err != nil {
		t.Fatalf("ArchProfileFor(sm_100) error: %v", err)
	}
	if !p100.SupportsTCGen05 {
		t.Errorf("sm_100 should support tcgen05")
	}
	if p100.SupportsWGMMA {
		t.Errorf("sm_100 should not support wgmma")
	}
	if !p100.SupportsTMEM {
		t.Errorf("sm_100 should support TMEM")
	}
	if err := p100.ValidateFeature(FeatureTMEM); err != nil {
		t.Errorf("sm_100 should validate TMEM, got: %v", err)
	}
}

func TestSimulation(t *testing.T) {
	pipe, err := NewPipeline(PipelineConfig{
		TargetArch: ArchSM121a,
		NumStages:  3,
		Box: TMABox{
			DimX:        64,
			DimY:        16,
			ElementSize: 2,
		},
	})
	if err != nil {
		t.Fatalf("setup failed: %v", err)
	}

	trace, err := pipe.Simulate(12)
	if err != nil {
		t.Fatalf("simulation failed: %v", err)
	}
	if trace.StepsCompleted != 12 {
		t.Errorf("StepsCompleted = %d, want 12", trace.StepsCompleted)
	}
	if len(trace.Events) != 24 { // 12 produce + 12 consume
		t.Errorf("Events count = %d, want 24", len(trace.Events))
	}
}

func TestPTEmission(t *testing.T) {
	// Instruction level checks
	initInst := EmitMBarrierInit("mbar_addr", 1)
	if initInst != "mbarrier.init.shared::cta.b64 [mbar_addr], 1;" {
		t.Errorf("unexpected initInst: %s", initInst)
	}

	arriveInst := EmitArriveExpectTx("mbar_addr", 2048)
	if arriveInst != "mbarrier.arrive.expect_tx.shared::cta.b64 _, [mbar_addr], 2048;" {
		t.Errorf("unexpected arriveInst: %s", arriveInst)
	}

	tmaInst := EmitTMABulkAsync2D("dst", "desc", "c0", "c1", "mbar")
	if tmaInst != "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [dst], [desc, {c0, c1}], [mbar];" {
		t.Errorf("unexpected tmaInst: %s", tmaInst)
	}

	waitInst := EmitWaitParity("mbar", 1, "%p")
	if waitInst != "mbarrier.try_wait.parity.shared::cta.b64 %p, [mbar], 1;" {
		t.Errorf("unexpected waitInst: %s", waitInst)
	}

	waitLoop := EmitWaitLoop("mbar", 0, "test")
	if !strings.Contains(waitLoop, "test_loop:") || !strings.Contains(waitLoop, "nanosleep.u32 20;") {
		t.Errorf("unexpected waitLoop: %s", waitLoop)
	}

	// Inline assembly checks
	asmInit := EmitInlineAsmInit(64, 1)
	if !strings.Contains(asmInit, "mbarrier.init.shared::cta.b64") {
		t.Errorf("unexpected asmInit: %s", asmInit)
	}

	// Pipeline level checks
	pipe, err := NewPipeline(PipelineConfig{
		TargetArch: ArchSM121a,
		NumStages:  2,
		Box:        TMABox{DimX: 64, DimY: 16, ElementSize: 2},
	})
	if err != nil {
		t.Fatalf("setup failed: %v", err)
	}

	prolog := pipe.EmitPrologPTX()
	if !strings.Contains(prolog, "mbarrier.init.shared::cta.b64") {
		t.Errorf("prolog missing mbarrier init: %s", prolog)
	}

	loadPTX := pipe.EmitStageLoadPTX(0, "desc_ptr", "c0", "c1")
	if !strings.Contains(loadPTX, "mbarrier.arrive.expect_tx") || !strings.Contains(loadPTX, "cp.async.bulk.tensor.2d") {
		t.Errorf("loadPTX missing arrive or tma: %s", loadPTX)
	}

	waitPTX := pipe.EmitStageWaitPTX(0, 0)
	if !strings.Contains(waitPTX, "mbarrier.try_wait.parity") {
		t.Errorf("waitPTX missing try_wait: %s", waitPTX)
	}
}
