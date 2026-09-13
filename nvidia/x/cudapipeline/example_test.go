package cudapipeline_test

import (
	"fmt"
	"log"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudapipeline"
	"github.com/tmc/lib/nvidia/x/cudaptx"
)

func Example() {
	const src = `
.version 8.8
.target sm_121a
.address_size 64

.visible .entry example_kernel(.param .u64 tmap) {
    .reg .b64 %rd_tmap, %state;
    .reg .pred %p0;
    .shared .align 8 .b64 bar;

    ld.param.u64 %rd_tmap, [tmap];
    mbarrier.init.shared::cta.b64 [bar], 1;
    mbarrier.arrive.expect_tx.shared::cta.b64 %state, [bar], 1024;
    cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [0], [%rd_tmap, {0, 0}], [bar];
    mbarrier.try_wait.parity.shared::cta.b64 %p0, [bar], 0;
    ret;
}
`
	mod, err := cudaptx.Parse(strings.NewReader(src))
	if err != nil {
		panic(err)
	}

	violations, err := cudapipeline.ValidatePTX(mod, cudapipeline.WithTensorSize("tmap", 1024))
	if err != nil {
		panic(err)
	}
	fmt.Println("Violations:", len(violations))
	// Output:
	// Violations: 0
}

func ExampleValidatePTX() {
	const src = `
.version 8.8
.target sm_121a
.address_size 64

.visible .entry buggy_kernel() {
    .reg .pred %p0;
    .shared .align 8 .b64 bar;

    mbarrier.init.shared::cta.b64 [bar], 1;
    // Missing arrival before wait
    mbarrier.try_wait.parity.shared::cta.b64 %p0, [bar], 0;
    ret;
}
`
	mod, err := cudaptx.Parse(strings.NewReader(src))
	if err != nil {
		panic(err)
	}

	violations, err := cudapipeline.ValidatePTX(mod)
	if err != nil {
		panic(err)
	}
	for _, v := range violations {
		fmt.Printf("Violation: %s\n", v.Violation)
	}
	// Output:
	// Violation: missing barrier arrive
}

func ExampleNewBarrierTracker() {
	tracker := cudapipeline.NewBarrierTracker(
		cudapipeline.WithTensorSize("tensor_a", 2048),
	)
	tracker.SetTensorSize("tensor_b", 1024)

	const src = `
.version 8.8
.target sm_121a
.address_size 64

.visible .entry pipelined_kernel(.param .u64 tensor_b) {
    .reg .b64 %rd_b, %state;
    .reg .pred %p0;
    .shared .align 8 .b64 bar;

    ld.param.u64 %rd_b, [tensor_b];
    mbarrier.init.shared::cta.b64 [bar], 1;
    mbarrier.arrive.expect_tx.shared::cta.b64 %state, [bar], 1024;
    cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [0], [%rd_b, {0, 0}], [bar];
    mbarrier.try_wait.parity.shared::cta.b64 %p0, [bar], 0;
    ret;
}
`
	mod, err := cudaptx.Parse(strings.NewReader(src))
	if err != nil {
		panic(err)
	}

	violations, err := tracker.Validate(mod)
	if err != nil {
		panic(err)
	}
	fmt.Printf("Violations: %d, bar phase: %d\n", len(violations), tracker.Barrier("bar").Phase)
	// Output:
	// Violations: 0, bar phase: 1
}

func ExampleNewPipeline() {
	box := cudapipeline.TMABox{
		DimX:        64,
		DimY:        32,
		ElementSize: 2,
	}

	pipe, err := cudapipeline.NewPipeline(cudapipeline.PipelineConfig{
		TargetArch:            cudapipeline.ArchSM121a,
		NumStages:             3,
		Box:                   box,
		MemoryLatencyCycles:   600,
		ComputeIntervalCycles: 300,
	})
	if err != nil {
		log.Fatal(err)
	}

	report := pipe.Validate()
	fmt.Printf("Valid: %t\n", report.Valid())
	fmt.Printf("Stages: %d\n", pipe.NumStages())
	fmt.Printf("Stage 0 Payload: %d bytes\n", pipe.Stage(0).TMABox.PayloadBytes())
	// Output:
	// Valid: true
	// Stages: 3
	// Stage 0 Payload: 4096 bytes
}

func ExampleNewPipeline_tmem() {
	box := cudapipeline.TMABox{
		DimX:        64,
		DimY:        16,
		ElementSize: 2, // 2048 bytes per stage
	}

	pipe, err := cudapipeline.NewPipeline(cudapipeline.PipelineConfig{
		TargetArch:  cudapipeline.ArchSM100,
		NumStages:   3,
		MemorySpace: cudapipeline.MemorySpaceTMEM,
		Box:         box,
	})
	if err != nil {
		log.Fatal(err)
	}

	report := pipe.Validate()
	fmt.Printf("Valid: %t\n", report.Valid())
	fmt.Printf("Stages: %d\n", pipe.NumStages())
	fmt.Printf("Space: %s\n", pipe.Stage(0).Buffer.Space)
	fmt.Printf("Total TMEM: %d bytes\n", pipe.TotalTMEMBytes())
	fmt.Printf("TMEM Columns: %d\n", pipe.TotalTMEMColumns())
	// Output:
	// Valid: true
	// Stages: 3
	// Space: tmem
	// Total TMEM: 6144 bytes
	// TMEM Columns: 32
}
