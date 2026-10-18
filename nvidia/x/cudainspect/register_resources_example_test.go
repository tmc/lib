package cudainspect_test

import (
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudainspect"
	"github.com/tmc/lib/nvidia/x/cudareg"
)

func ExampleCompiledResources_RegisterResources() {
	// A caller-supplied report; the conversion does not authenticate its tools.
	allocated := uint64(64)
	r := &cudainspect.CompiledResources{
		SchemaVersion: "compiled-resources/v1", ImageDigest: strings.Repeat("a", 64), SourceDigest: strings.Repeat("b", 64),
		Entry: "kernel", Target: "sm_90a", DecodedTarget: "sm_90", CompilerVersion: "supplied compiler", DisassemblerVersion: "supplied disassembler",
		Allocated: cudainspect.ResourceFact{Value: &allocated, Category: "decoded"},
	}
	resources, err := r.RegisterResources()
	if err != nil {
		fmt.Println(err)
		return
	}
	s, _, _ := cudareg.GenerateFixture()
	c, err := cudareg.PublishCandidate(&resources, s)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(c.Status, resources.Target, resources.DecodedTarget)
	fmt.Println(resources.Allocated.Count, resources.PeakLive.Status)
	// Output:
	// protocol feasible sm_90a sm_90
	// 64 unavailable
}
