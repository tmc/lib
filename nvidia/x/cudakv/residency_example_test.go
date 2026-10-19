package cudakv_test

import (
	"context"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudakv"
)

func ExamplePagedCache_AllocationGeneration() {
	pc, _ := cudakv.NewTestPagedCache(1, 1, 2, 2, 2)
	defer pc.Close()
	pc.AllocSequence(7, 2)
	generation, ok := pc.AllocationGeneration(7)
	fmt.Println(generation, ok)
	// Output: 1 true
}

func ExamplePagedCache_ValidateSequence() {
	pc, _ := cudakv.NewTestPagedCache(1, 1, 2, 2, 2)
	defer pc.Close()
	pc.AllocSequence(7, 2)
	generation, _ := pc.AllocationGeneration(7)
	binding := cudakv.ResidencyBinding{SequenceID: 7, Generation: generation, Blocks: pc.BlockTable(7), Tokens: 2}
	fmt.Println(pc.ValidateSequence(binding))
	pc.FreeSequence(7)
	fmt.Println(pc.ValidateSequence(binding) != nil)
	// Output:
	// <nil>
	// true
}

func ExampleNewResidency() {
	// Residency needs actual owned CUDA Page storage. A host-test cache cannot
	// silently stand in for pinned transfers; this refusal calls no vendor API.
	pc, _ := cudakv.NewTestPagedCache(1, 1, 2, 2, 2)
	defer pc.Close()
	pc.AllocSequence(7, 2)
	generation, _ := pc.AllocationGeneration(7)
	owner, err := cudakv.NewResidency(pc, 7, generation)
	fmt.Println(owner == nil, err != nil)
	// Output: true true
}

func ExampleResidencyTier() {
	fmt.Println(cudakv.ResidencyEvicting, cudakv.ResidencyHost)
	// Output: evicting pinned_host
}

func ExampleResidencyBinding() {
	binding := cudakv.ResidencyBinding{SequenceID: 7, Generation: 3, Blocks: []int{2, 1}, Tokens: 4}
	fmt.Println(binding.SequenceID, binding.Generation, binding.Blocks)
	// Output: 7 3 [2 1]
}

func ExampleResidency_Tier() {
	var owner *cudakv.Residency
	fmt.Println(owner.Tier())
	// Output: closed
}

func ExampleResidency_Binding() {
	var owner *cudakv.Residency
	_, err := owner.Binding()
	fmt.Println(err != nil)
	// Output: true
}

func ExampleResidency_Evict() {
	var owner *cudakv.Residency
	fmt.Println(owner.Evict(nil) != nil)
	// Output: true
}

func ExampleResidency_Prefetch() {
	var owner *cudakv.Residency
	fmt.Println(owner.Prefetch(nil) != nil)
	// Output: true
}

func ExampleResidency_Wait() {
	var owner *cudakv.Residency
	fmt.Println(owner.Wait(context.Background()) != nil)
	// Output: true
}

func ExampleResidency_Close() {
	var owner *cudakv.Residency
	fmt.Println(owner.Close())
	// Output: <nil>
}
