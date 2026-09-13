package cudapipeline_test

import (
	"fmt"
	"log"

	"github.com/tmc/lib/nvidia/x/cudapipeline"
)

func Example() {
	// Configure a 3-stage async pipeline targeting Blackwell sm_121a.
	box := cudapipeline.TMABox{
		DimX:        64,
		DimY:        32,
		ElementSize: 2, // 16-bit fp16 elements (4096 bytes payload)
	}

	pipe, err := cudapipeline.NewPipeline(cudapipeline.PipelineConfig{
		TargetArch:            cudapipeline.ArchSM121a,
		NumStages:             3, // Triple-buffering
		Box:                   box,
		MemoryLatencyCycles:   600,
		ComputeIntervalCycles: 300,
	})
	if err != nil {
		log.Fatal(err)
	}

	// Statically verify the pipeline.
	report := pipe.Validate()
	fmt.Printf("Valid: %t\n", report.Valid())
	fmt.Printf("Stages: %d\n", pipe.NumStages())
	fmt.Printf("Stage 0 Payload: %d bytes\n", pipe.Stage(0).TMABox.PayloadBytes())
	fmt.Printf("Latency Hidden: %.0f%%\n", report.LatencyHiddenPercent)

	// Emit PTX for asynchronous stage loading.
	loadPTX := pipe.EmitStageLoadPTX(0, "%desc", "%c0", "%c1")
	fmt.Print(loadPTX)

	// Output:
	// Valid: true
	// Stages: 3
	// Stage 0 Payload: 4096 bytes
	// Latency Hidden: 100%
	// // Stage 0 load (4096 bytes via TMA)
	// mbarrier.arrive.expect_tx.shared::cta.b64 _, [smem_base+12288], 4096;
	// cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [smem_base+0], [%desc, {%c0, %c1}], [smem_base+12288];
}
