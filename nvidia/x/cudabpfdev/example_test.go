package cudabpfdev_test

import (
	"context"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudabpfdev"
	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudakernel"
)

func Example() {
	p := cudabpfdev.Program{Version: 1, Registers: 2, Slots: 1, Code: []cudabpfdev.Instruction{
		{Op: cudabpfdev.Const, Dst: 1, Imm: 1},
		{Op: cudabpfdev.MapAdd, B: 1},
		{Op: cudabpfdev.Exit},
	}}
	b := cudabpfdev.Budget{MaxInstructions: 10, MaxSteps: 10, MaxRegisters: 4, MaxSlots: 1}
	r, err := cudabpfdev.Evaluate(context.Background(), p, b, 0xff)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(r.Map, r.Steps)
	// Output: [1] 3
}

func ExampleVerify() {
	p := cudabpfdev.Program{Version: 1, Registers: 1, Slots: 1, Code: []cudabpfdev.Instruction{{Op: cudabpfdev.Exit}}}
	a, err := cudabpfdev.Verify(p, cudabpfdev.Budget{MaxInstructions: 1, MaxSteps: 1, MaxRegisters: 1, MaxSlots: 1})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(a.MaxSteps, a.Registers)
	// Output: 1 1
}

func ExampleTranslate() {
	p := cudabpfdev.Program{Version: 1, Registers: 1, Slots: 1, Code: []cudabpfdev.Instruction{{Op: cudabpfdev.Exit}}}
	b := cudabpfdev.Budget{MaxInstructions: 1, MaxSteps: 1, MaxRegisters: 32, MaxSlots: 1}
	x, err := cudabpfdev.Translate(p, b, "sm_121a")
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(x.Artifact.Target, x.Artifact.EntryPoint, x.Artifact.Schema.TotalParamBytes)
	// Output: sm_121a cudabpfdev_main 32
}

func ExampleAssemble() {
	// An unverified or missing translation must never reach the native linker.
	_, err := cudabpfdev.Assemble(cudabpfdev.Translation{})
	fmt.Println(err != nil)
	// Output: true
}

func ExampleNewArtifactSource() {
	ptx := []byte(".version 9.0\n.target sm_121a\n.address_size 64\n.visible .entry noop() { ret; }\n")
	a, err := cudakernel.NewArtifact(cudakernel.KindPTX, ptx, "noop", cudakernel.Schema{}, "sm_121a")
	if err != nil {
		fmt.Println(err)
		return
	}
	s, err := cudabpfdev.NewArtifactSource(a, cudabpfdev.Geometry{
		Grid: [3]uint32{1, 1, 1}, Block: [3]uint32{32, 1, 1},
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	launch, err := s.DescribeLaunch()
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(launch.Artifact.EntryPoint, launch.Geometry.Block)
	// Output: noop [32 1 1]
}

func ExampleArtifactSource_DescribeLaunch() {
	var source cudabpfdev.ArtifactSource
	_, err := source.DescribeLaunch()
	fmt.Println(err)
	// Output: cudabpfdev: uninitialized artifact source
}

func ExampleArtifactSource_Load() {
	// A device must be explicitly supplied; Load does not select a GPU.
	var source cudabpfdev.ArtifactSource
	_, err := source.Load(nil)
	fmt.Println(err)
	// Output: cudabpfdev: nil device
}

func ExampleExecute() {
	// Native execution requires an explicit device; this host-only example
	// exercises refusal before any linker or CUDA call.
	_, err := cudabpfdev.Execute(context.Background(), nil, cudabpfdev.Translation{}, 0xff)
	fmt.Println(err)
	// Output: cudabpfdev: nil device
}

func ExampleExecution_Snapshot() {
	var execution cudabpfdev.Execution
	_, ready := execution.Snapshot()
	fmt.Println(ready)
	// Output: false
}

func ExampleExecution_Close() {
	// Every non-nil Execute result must be closed, including error results.
	var execution *cudabpfdev.Execution
	fmt.Println(execution.Close())
	// Output: <nil>
}

func ExampleExecution_Record() {
	var execution cudabpfdev.Execution
	_, err := execution.Record()
	fmt.Println(err)
	// Output: cudabpfdev: no completed snapshot
}

func ExampleCompareOutputs() {
	g := cudair.NewGraph()
	x := g.AddInput("x", []int{2}, cudair.Float32)
	y := g.AddUnary("neg", x)
	if err := g.SetOutputs(y); err != nil {
		fmt.Println(err)
		return
	}
	inputs := map[string][]float32{"x": {1, -2}}
	// Synthetic buffers illustrate comparison, not native qualification.
	original := map[string][]float32{y.Name: {-1, 2}}
	instrumented := map[string][]float32{y.Name: {-1, 2}}
	err := cudabpfdev.CompareOutputs(g, inputs, original, instrumented, cudabpfdev.NumericalPolicy{Mode: "exact"})
	fmt.Println(err)
	// Output: <nil>
}

func ExamplePrepareInjection() {
	a, err := cudakernel.NewArtifact(cudakernel.KindPTX, []byte(".version 9.0\n.target sm_121a\n.address_size 64\n.visible .entry noop() { ret; }\n"), "noop", cudakernel.Schema{}, "sm_121a")
	if err != nil {
		fmt.Println(err)
		return
	}
	s, err := cudabpfdev.NewArtifactSource(a, cudabpfdev.Geometry{Grid: [3]uint32{1, 1, 1}, Block: [3]uint32{32, 1, 1}})
	if err != nil {
		fmt.Println(err)
		return
	}
	// Instrumentation is disabled unless explicitly enabled and allowlisted.
	i, err := cudabpfdev.PrepareInjection(s, cudabpfdev.Program{}, cudabpfdev.Budget{}, cudabpfdev.InjectionPolicy{})
	if err != nil {
		fmt.Println(err)
		return
	}
	candidate, reason := i.Candidate()
	original, err := i.Original()
	fmt.Println(candidate == nil, reason, original != nil, err)
	// Output: true injection disabled true <nil>
}

func ExampleInjection_Load() {
	// Loading always requires an explicit device, even on the original path.
	var injection cudabpfdev.Injection
	_, err := injection.Load(nil)
	fmt.Println(err)
	// Output: cudabpfdev: nil device
}

func ExampleLoadedInjection_Selection() {
	var loaded cudabpfdev.LoadedInjection
	_, err := loaded.Selection()
	fmt.Println(err)
	// Output: cudabpfdev: no loaded selection
}

func ExampleLoadedInjection_Close() {
	var loaded *cudabpfdev.LoadedInjection
	fmt.Println(loaded.Close())
	// Output: <nil>
}

func ExampleLoadedInjection_Launch() {
	var loaded cudabpfdev.LoadedInjection
	fmt.Println(loaded.Launch(context.Background()))
	// Output: cudabpfdev: no open loaded selection
}

func ExampleLoadedInjection_Snapshot() {
	var loaded cudabpfdev.LoadedInjection
	_, available := loaded.Snapshot()
	fmt.Println(available)
	// Output: false
}

func ExampleLoadedInjection_Record() {
	var loaded cudabpfdev.LoadedInjection
	_, err := loaded.Record()
	fmt.Println(err)
	// Output: cudabpfdev: no completed invocation
}
