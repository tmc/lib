package cudamegacpu_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudamega"
	"github.com/tmc/lib/nvidia/x/cudamegacpu"
)

func ExampleExecutor_RunMoE() {
	c := cudamega.MoEWeightConfig{Layers: 3, Experts: 2, Intermediate: 8, Hidden: 17}
	in := cudamega.MoEInput{Config: c, Layer: 2, Weights: make([]byte, 96*17), Scales: make([]float32, 96), Hidden: make([]float32, 17), Routing: [2]float32{0.25, 0.75}}
	for i := range in.Weights {
		in.Weights[i] = 0x20
	}
	for i := range in.Scales {
		in.Scales[i] = 1
	}
	for i := range in.Hidden {
		in.Hidden[i] = 0.125
	}
	binding, err := cudamega.BindMoE(in)
	if err != nil {
		panic(err)
	}
	p, err := cudamega.NewMoEMixtureProgram(c, cudamega.Grid{CTAs: 1, ThreadsPerCTA: 1, MaxElements: 8, Policy: cudamega.CountedPolicy}, cudamega.SM121a)
	if err != nil {
		panic(err)
	}
	e, err := cudamegacpu.New(cudamega.Admission{ResidentCTAs: 1, WorkerThreads: 1, SharedBytes: 1})
	if err != nil {
		panic(err)
	}
	result, err := e.RunMoE(p, binding, 1, 0)
	if err != nil {
		panic(err)
	}
	fmt.Println(len(result.Scratch), result.Threads, result.Abort)
	// Output: 24 1 0
}
