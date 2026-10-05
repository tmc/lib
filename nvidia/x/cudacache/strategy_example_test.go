package cudacache_test

import (
	"bytes"
	"context"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudacache"
	"github.com/tmc/lib/nvidia/x/cudatrial"
)

func ExampleStrategyKnowledgeBase() {
	var k cudacache.StrategyKnowledgeBase
	err := k.Update(cudacache.Strategy{Family: cudatrial.MemoryBandwidthLimited, TargetArch: "sm_121", Technique: "shared_memory_tiling", Confidence: 0.5, PredictedSpeedup: 1})
	fmt.Println(err, k.Query(cudatrial.MemoryBandwidthLimited, "sm_121")[0].Technique)
	// Output: <nil> shared_memory_tiling
}

func ExampleStrategyKnowledgeBase_Update() {
	var k cudacache.StrategyKnowledgeBase
	err := k.Update(cudacache.Strategy{Family: cudatrial.MemoryBandwidthLimited, TargetArch: "sm_121", Technique: "shared_memory_tiling", Confidence: 0.5, PredictedSpeedup: 1})
	fmt.Println(err)
	// Output: <nil>
}

func ExampleStrategyKnowledgeBase_Query() {
	var k cudacache.StrategyKnowledgeBase
	fmt.Println(len(k.Query(cudatrial.MemoryBandwidthLimited, "sm_121")))
	// Output: 0
}

func ExampleStrategyKnowledgeBase_RecordTrajectory() {
	var k cudacache.StrategyKnowledgeBase
	err := k.RecordTrajectory(cudatrial.TrajectoryReceipt{PendingHardware: true}, nil)
	fmt.Println(err != nil)
	// Output: true
}

func ExampleStrategyKnowledgeBase_RecordTrustedTrajectory() {
	var k cudacache.StrategyKnowledgeBase
	fmt.Println(k.RecordTrustedTrajectory(cudatrial.TrajectoryReceipt{PendingHardware: true}, nil) != nil)
	// Output: true
}

func ExampleCache_PutTrustedTrajectory() {
	var cache *cudacache.Cache
	fmt.Println(cache.PutTrustedTrajectory(context.Background(), cudatrial.TrajectoryReceipt{}) != nil)
	// Output: true
}

func ExampleStrategyKnowledgeBase_LoadKernelBlaster() {
	var k cudacache.StrategyKnowledgeBase
	err := k.LoadKernelBlaster(bytes.NewReader(nil), "sm_121")
	fmt.Println(err != nil)
	// Output: true
}

func ExampleStrategyKnowledgeBase_SaveFile() {
	var k cudacache.StrategyKnowledgeBase
	// An explicit destination is required; no default shared temp path is used.
	fmt.Println(k.SaveFile("") != nil)
	// Output: true
}

func ExampleStrategyKnowledgeBase_LoadFile() {
	var k cudacache.StrategyKnowledgeBase
	fmt.Println(k.LoadFile("") != nil)
	// Output: true
}

func ExampleCache_PutTrajectory() {
	var cache *cudacache.Cache
	// An initialized cache and qualified observation are required.
	fmt.Println(cache.PutTrajectory(context.Background(), cudatrial.TrajectoryReceipt{}) != nil)
	// Output: true
}

func ExampleCache_GetTrajectory() {
	var cache *cudacache.Cache
	_, err := cache.GetTrajectory(context.Background(), "not-a-content-key")
	fmt.Println(err != nil)
	// Output: true
}

func ExampleStrategyKnowledgeBase_Write() {
	var k cudacache.StrategyKnowledgeBase
	var snapshot bytes.Buffer
	err := k.Write(&snapshot)
	fmt.Println(err, snapshot.Len() > 0)
	// Output: <nil> true
}

func ExampleStrategyKnowledgeBase_Load() {
	var original, next cudacache.StrategyKnowledgeBase
	var snapshot bytes.Buffer
	original.Write(&snapshot)
	fmt.Println(next.Load(&snapshot), len(next.Query(cudatrial.MemoryBandwidthLimited, "sm_121")))
	// Output: <nil> 0
}

func ExampleKernelBlasterGroups() {
	mapping := cudacache.KernelBlasterGroups()
	fmt.Println(len(mapping), mapping["api_overhead_dominant"][0])
	// Output: 16 api_overhead_dominant
}
