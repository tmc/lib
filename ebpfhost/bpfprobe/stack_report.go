package bpfprobe

import (
	"fmt"

	"github.com/tmc/lib/ebpfhost"
)

// StackReportOptions supplies the caller's process-start snapshot and frame
// pointer declaration. ProcessGeneration is not Event.ThreadGeneration:
// Event does not contain a process-start token. Matching declarations establish
// consistency only, not authentic running-image or unwind coverage.
type StackReportOptions struct {
	ProcessGeneration                 uint64
	FramePointersKnown, FramePointers bool
}

// ReportStack consumes the actual bounded Event.Stack from Probe.Read. It does
// not add InstructionPointer, guess a missing generation or recover interpreter
// frames. Unknown execution identity leaves PCs unresolved. Run after draining;
// this function performs no attachment, map syscall, tool or target-memory read.
func ReportStack(event Event, options StackReportOptions, resolver *ebpfhost.FrameResolver) (ebpfhost.FrameReport, error) {
	if uint32(event.PIDTGID>>32) == 0 || uint32(event.PIDTGID) == 0 {
		return ebpfhost.FrameReport{}, fmt.Errorf("stack report: missing captured process or thread")
	}
	if event.UserStackRequested && event.StackID < 0 && event.StackUnavailable == "" {
		return ebpfhost.FrameReport{}, fmt.Errorf("stack report: negative stack ID without unavailable reason")
	}
	var execID uint64
	if event.ExecIDKnown {
		if event.ExecID == 0 {
			return ebpfhost.FrameReport{}, fmt.Errorf("stack report: contradictory execution identity")
		}
		execID = event.ExecID
	}
	return resolver.ResolveStack(ebpfhost.StackCapture{
		Context: ebpfhost.FrameContext{PID: uint32(event.PIDTGID >> 32), Generation: options.ProcessGeneration, ExecID: execID},
		ID:      event.StackID, PCs: event.Stack, Requested: event.UserStackRequested, Unavailable: event.StackUnavailable,
		FramePointersKnown: options.FramePointersKnown, FramePointers: options.FramePointers,
	})
}
