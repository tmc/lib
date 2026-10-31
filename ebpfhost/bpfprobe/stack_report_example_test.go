package bpfprobe_test

import (
	"debug/elf"
	"fmt"

	"github.com/tmc/lib/ebpfhost"
	"github.com/tmc/lib/ebpfhost/bpfprobe"
)

func ExampleReportStack() {
	image := stackImage(stackFixture(elf.ET_DYN, "observed", true, false))
	r, err := ebpfhost.NewFrameResolver(ebpfhost.FrameContext{PID: 10, Generation: 20, ExecID: 30}, []ebpfhost.MappedFrameImage{image})
	if err != nil {
		fmt.Println(err)
		return
	}
	// Same Event fields as a retained Probe.Read result; this is fixture data.
	e := bpfprobe.Event{PIDTGID: uint64(10)<<32 | 11, ExecID: 30, ExecIDKnown: true, UserStackRequested: true, StackID: 7, Stack: []uint64{0x700028}}
	out, err := bpfprobe.ReportStack(e, bpfprobe.StackReportOptions{ProcessGeneration: 20}, r)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(out.Frames[0].Symbol, out.Frames[0].SymbolOffset, out.Partial)
	// Output: observed 8 true
}
