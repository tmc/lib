package cuptipcsampling_test

import (
	"context"
	"io"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaprof"
	"github.com/tmc/lib/nvidia/x/cudaptx"
	"github.com/tmc/lib/nvidia/x/cuptipcsampling"
)

// BenchmarkRecordBatchByResolution measures what a resolved sample costs the
// drain against what an unresolved one costs, because the per-sample cost of
// RecordBatch is a function of how deeply the PC resolved and not of the
// sampling period. A capture whose attribution is switched off is therefore the
// cheapest possible version of the same drain, and a sampling rate measured
// against it is not a rate the resolving drain can sustain.
//
// The arms differ only in whether the resolver knows the kernel. Everything
// else -- the batch, the reason inventory, the session -- is identical, so the
// ratio between them is the cost of attribution alone.
func BenchmarkRecordBatchByResolution(b *testing.B) {
	inv := cuptipcsampling.ReasonInventory{
		CUPTIVersion: 130001,
		Reasons: []cuptipcsampling.StallReason{
			{Index: 1, Name: "selected"},
			{Index: 2, Name: "membar"},
			{Index: 3, Name: "short_scoreboard"},
		},
	}
	batch := cuptipcsampling.Batch{
		Valid:        true,
		TotalSamples: 24,
		PCs: []cuptipcsampling.PC{{
			CRC:    0x12345678,
			Name:   "ds4_q8_0_matvec_q8_1_batch",
			Offset: 0x20,
			Reasons: []cuptipcsampling.StallSamples{
				{Index: 1, Samples: 10},
				{Index: 2, Samples: 8},
				{Index: 3, Samples: 6},
			},
		}},
	}

	// resolved carries every tier the bridge knows how to emit: a SASS leaf, a
	// PTX frame, one inlined frame and the CUDA source line. unresolved is the
	// same resolver with an empty table, which is what a capture that cannot
	// name its kernels hands the bridge.
	resolved := &mockResolver{locations: map[string]cuptipcsampling.CodeLocation{
		"ds4_q8_0_matvec_q8_1_batch": {
			KernelName:  "ds4_q8_0_matvec_q8_1_batch",
			SourceFile:  "cuda_q80_matvec.cu",
			SourceLine:  516,
			PTXFile:     "ds4_math.ptx",
			PTXLine:     105,
			SASSFile:    "ds4_math.sass",
			SASSLine:    30,
			CUBINOffset: 0x1000,
			InlinedAt:   []cudaptx.Location{{File: "cuda_probe.cu", Line: 77}},
		},
	}}
	unresolved := &mockResolver{locations: nil}

	// distinct varies the PC offset per iteration. It matters because a
	// repeated offset interns after the first record: the location table is
	// built once and every later sample is a hit, which measures the cheapest
	// case and not the one a real capture is in. The ds4-min prefill reported
	// 649,032 unique PCs over 6.85M samples, so nearly every record allocates
	// a new location -- and a resolved sample allocates four of them where an
	// unresolved sample allocates one.
	for _, arm := range []struct {
		name     string
		resolver cuptipcsampling.CodeResolver
		distinct bool
	}{
		{"repeated/resolved", resolved, false},
		{"repeated/unresolved", unresolved, false},
		{"distinct/resolved", resolved, true},
		{"distinct/unresolved", unresolved, true},
	} {
		b.Run(arm.name, func(b *testing.B) {
			session := cudaprof.NewSession(io.Discard)
			session.SetMappingFile("kernels.cubin", "")
			bridge := &cuptipcsampling.ProfileBridge{
				Session:   session,
				Inventory: inv,
				Resolver:  arm.resolver,
			}
			ctx := context.Background()
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if arm.distinct {
					batch.PCs[0].Offset = uint64(i) * 16
				}
				if err := bridge.RecordBatch(ctx, batch); err != nil {
					b.Fatalf("RecordBatch: %v", err)
				}
			}
		})
	}
}
