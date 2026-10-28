package cudamega_test

import (
	"fmt"
	"github.com/tmc/lib/nvidia/x/cudamega"
	"strings"
)

func ExampleNewMoEMixtureProgram() {
	config := cudamega.MoEWeightConfig{Layers: 3, Experts: 2, Intermediate: 8, Hidden: 17}
	p, err := cudamega.NewMoEMixtureProgram(config, cudamega.Grid{CTAs: 1, ThreadsPerCTA: 1, MaxElements: 8, Policy: cudamega.CountedPolicy}, cudamega.SM80)
	if err != nil {
		panic(err)
	}
	invocation, err := p.Invocation(8)
	if err != nil {
		panic(err)
	}
	fmt.Println(len(p.Steps()), invocation.ScratchWords)
	// Output: 3 24
}

func ExampleMoEGateUpOperation() {
	op, err := cudamega.MoEGateUpOperation(cudamega.MoEWeightConfig{Layers: 3, Experts: 2, Intermediate: 8, Hidden: 17}, 1)
	if err != nil {
		panic(err)
	}
	descriptor, ok := op.MoE()
	fmt.Println(ok, descriptor.Expert)
	// Output: true 1
}

func ExampleMoECombineOperation() {
	op, err := cudamega.MoECombineOperation(cudamega.MoEWeightConfig{Layers: 3, Experts: 2, Intermediate: 8, Hidden: 17})
	if err != nil {
		panic(err)
	}
	descriptor, ok := op.MoE()
	fmt.Println(ok, descriptor.Expert)
	// Output: true -1
}

func ExampleBindMoE() {
	config := cudamega.MoEWeightConfig{Layers: 3, Experts: 2, Intermediate: 8, Hidden: 17}
	rows := config.Layers * config.Experts * 2 * config.Intermediate
	binding, err := cudamega.BindMoE(cudamega.MoEInput{Config: config, Weights: make([]byte, rows*config.Hidden), Scales: make([]float32, rows), Hidden: make([]float32, config.Hidden), Routing: [2]float32{0.25, 0.75}})
	if err != nil {
		panic(err)
	}
	input, err := binding.Snapshot()
	if err != nil {
		panic(err)
	}
	fmt.Println(input.Layer, len(input.Hidden), input.Routing)
	// Output: 0 17 [0.25 0.75]
}

func ExampleProgram_MoEOrdinarySource() {
	config := cudamega.MoEWeightConfig{Layers: 3, Experts: 2, Intermediate: 8, Hidden: 17}
	p, err := cudamega.NewMoEMixtureProgram(config, cudamega.Grid{CTAs: 1, ThreadsPerCTA: 1, MaxElements: 8, Policy: cudamega.CountedPolicy}, cudamega.SM80)
	if err != nil {
		panic(err)
	}
	source, err := p.MoEOrdinarySource()
	if err != nil {
		panic(err)
	}
	fmt.Println(strings.Contains(source, "moe_weight_slab("), strings.Contains(source, "moe_mixture_ordinary("))
	// Output: true true
}
