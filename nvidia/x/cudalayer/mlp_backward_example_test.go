package cudalayer_test

import (
	"fmt"
	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudalayer"
)

func ExampleFusedMLP_ParameterCount() {
	m, _ := cudalayer.NewFusedMLP(5, 7, 3, 2)
	fmt.Println(m.ParameterCount())
	// Output: 56
}

func ExampleNewFusedMLPWithActivation() {
	m, _ := cudalayer.NewFusedMLPWithActivation(5, 7, 3, 2, "silu")
	fmt.Println(m.ParameterCount())
	// Output: 56
}

func ExampleMLPBackwardOptions() {
	o := cudalayer.MLPBackwardOptions{Samples: 7, Input: true, ParameterMode: cudair.GradientModeOverwrite, MaxBytes: 1 << 20}
	fmt.Println(o.Samples, o.Input, o.ParameterMode)
	// Output: 7 true Overwrite
}

func ExampleFusedMLP_BackwardBatch() {
	m, _ := cudalayer.NewFusedMLP(5, 3, 3, 1)
	o := cudalayer.MLPBackwardOptions{Samples: 7, Input: true, ParameterMode: cudair.GradientModeOverwrite, MaxBytes: 1 << 20}
	// A real CUDA device is required after host buffer validation.
	_, err := m.BackwardBatch(nil, make([]float32, 35), make([]float32, 15), make([]float32, 21), nil, o)
	fmt.Println(err)
	// Output: cudalayer: nil backward device
}
