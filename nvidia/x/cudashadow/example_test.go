package cudashadow_test

import (
	"context"
	"fmt"
	"time"

	"github.com/tmc/lib/nvidia/x/cudashadow"
)

func ExampleEvaluator_Evaluate() {
	eval, err := cudashadow.NewEvaluator(10)
	if err != nil {
		fmt.Printf("new evaluator error: %v\n", err)
		return
	}
	eval.SetClock(func() int64 { return 5000 })

	pol, err := cudashadow.NewFIFOPolicy(cudashadow.PolicyInfo{
		Name:    "gpu_ext-fifo",
		Version: "1.0.0",
		Domain:  cudashadow.DomainEviction,
	})
	if err != nil {
		fmt.Printf("new policy error: %v\n", err)
		return
	}

	ctx := context.Background()

	// 1. Activate chunk 0x1000
	snap1, err := eval.Evaluate(ctx, pol, cudashadow.Input{
		Provenance: cudashadow.Provenance{
			DeviceID:      0,
			DriverVersion: "575.57.08",
			KernelModule:  "nvidia-uvm.ko",
			Hook:          cudashadow.HookBlockActivate,
			Sequence:      1,
			TimestampNano: 1000,
			TTL:           time.Second,
		},
		Domain:   cudashadow.DomainEviction,
		Eviction: &cudashadow.EvictionData{ChunkAddr: 0x1000},
	})
	if err != nil {
		fmt.Printf("eval 1 error: %v\n", err)
		return
	}

	// 2. Access chunk 0x1000
	snap2, err := eval.Evaluate(ctx, pol, cudashadow.Input{
		Provenance: cudashadow.Provenance{
			DeviceID:      0,
			DriverVersion: "575.57.08",
			KernelModule:  "nvidia-uvm.ko",
			Hook:          cudashadow.HookBlockAccess,
			Sequence:      2,
			TimestampNano: 2000,
			TTL:           time.Second,
		},
		Domain:   cudashadow.DomainEviction,
		Eviction: &cudashadow.EvictionData{ChunkAddr: 0x1000},
	})
	if err != nil {
		fmt.Printf("eval 2 error: %v\n", err)
		return
	}

	// 3. Prepare eviction: FIFO targets oldest chunk (0x1000)
	snap3, err := eval.Evaluate(ctx, pol, cudashadow.Input{
		Provenance: cudashadow.Provenance{
			DeviceID:      0,
			DriverVersion: "575.57.08",
			KernelModule:  "nvidia-uvm.ko",
			Hook:          cudashadow.HookEvictPrepare,
			Sequence:      3,
			TimestampNano: 3000,
			TTL:           time.Second,
		},
		Domain:   cudashadow.DomainEviction,
		Eviction: &cudashadow.EvictionData{},
	})
	if err != nil {
		fmt.Printf("eval 3 error: %v\n", err)
		return
	}

	fmt.Printf("activate decision: %s (status: %s)\n", snap1.Decision.Type, snap1.Status)
	fmt.Printf("access decision: %s (status: %s)\n", snap2.Decision.Type, snap2.Status)
	fmt.Printf("evict decision: %s on 0x%x (status: %s)\n",
		snap3.Decision.Type, snap3.Decision.Parameters["target_chunk"], snap3.Status)

	// Output:
	// activate decision: bypass (status: valid)
	// access decision: bypass (status: valid)
	// evict decision: evict on 0x1000 (status: valid)
}

func ExampleActiveBackendSpy() {
	spy := cudashadow.NewActiveBackendSpy()

	// In shadow mode, no mutation methods should be called
	if err := spy.AssertZeroMutations(); err != nil {
		fmt.Printf("mutation error: %v\n", err)
		return
	}

	fmt.Printf("active mutation calls recorded: %d\n", spy.CallCount())

	// Output:
	// active mutation calls recorded: 0
}
