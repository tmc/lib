package ebpfhost_test

import (
	"debug/elf"
	"fmt"

	"github.com/tmc/lib/ebpfhost"
)

func ExampleNewFrameResolver() {
	// Ordinary inert metadata from this example's fixture helper, never loaded.
	image := frameImage(frameFixture(elf.ET_DYN, "observed", false, false))
	r, err := ebpfhost.NewFrameResolver(frameContext(), []ebpfhost.MappedFrameImage{image})
	fmt.Println(r != nil, err)
	// Output: true <nil>
}

func ExampleFrameResolver_ResolveStack() {
	image := frameImage(frameFixture(elf.ET_DYN, "observed", false, false))
	r, err := ebpfhost.NewFrameResolver(frameContext(), []ebpfhost.MappedFrameImage{image})
	if err != nil {
		fmt.Println(err)
		return
	}
	report, err := r.ResolveStack(captureFrames(0x700028))
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(report.Frames[0].Symbol, report.Frames[0].SymbolOffset, report.Partial)
	fmt.Println(report.IdentityScope, report.LogicalFrames)
	// Output:
	// observed 8 true
	// caller_mapping_consistency_only unresolved_not_captured
}
