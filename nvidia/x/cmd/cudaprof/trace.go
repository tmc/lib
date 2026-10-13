package main

import (
	"fmt"
	"os"
)

type traceCommand struct {
	name, purpose, inputs, defaults, examples string
	run                                       func([]string) error
}

func traceCommands() []traceCommand {
	return []traceCommand{
		{"export", "Write a device timeline for direct opening in Perfetto UI.", "native capture", "raw timeline; summary tracks opt in", "cudaprof trace export -o run.pftrace run.pprof\n  cudaprof trace export -summary -o run.pftrace run.pprof", runExportCommand},
		{"compare", "Align independent captures by recorded markers without stretching durations.", "two native captures", "fixed offsets; independent clocks; summary tracks opt in", "cudaprof trace compare -iteration-marker decode -o runs.tar base.pprof target.pprof\n  cudaprof trace compare -summary -iteration-marker decode -o runs.tar base.pprof target.pprof", runExportCompareCommand},
		{"check", "Check independent input and qualification evidence.", "native capture", "integrity and collector completeness; no external backend", "cudaprof trace check run.pprof\n  cudaprof trace check -json -require collector run.pprof", runTraceCheckCommand},
		{"summary", "Summarize retained native timeline evidence.", "native capture", "concise text; qualification dimensions remain separate", "cudaprof trace summary run.pprof\n  cudaprof trace summary -json run.pprof", runTraceSummaryCommand},
		{"query", "Run a named native timeline query.", "native capture", "named queries; -list requires no input", "cudaprof trace query -list\n  cudaprof trace query -json inventory run.pprof", runTraceQueryCommand},
	}
}

func traceHelp(c traceCommand, spelling string) {
	switch c.name {
	case "export":
		fmt.Println("Options: -o FILE, -detail raw|compact|token|operation, -operation-domain driver|runtime, -summary, -summary-marker RE, -submission-links, -graph-lifecycle, -memory, -telemetry, -output-bytes N, -max-tracks N, -max-summary-references N; -go-trace-decoder PATH -go-trace-scratch DIR; existing export flags remain supported")
	case "compare":
		fmt.Println("Options: -o FILE, -iteration-marker RE, -iteration-count N, -detail raw|compact|token|operation, -operation-domain driver|runtime, -summary, -submission-links, -graph-lifecycle, -memory, -telemetry, -output-bytes N, -max-tracks N, -max-summary-references N; -go-trace-decoder PATH -go-trace-scratch DIR; existing alignment flags remain supported")
	case "check":
		fmt.Println("Options: -json, -require CHECK,..., -iteration-marker RE, -expect-markers N, -domain driver|runtime, -perfetto PATH, -shape-receipt FILE, -correctness-receipt FILE")
		fmt.Println("Resource overrides: -input-bytes N, -journal-bytes N, -max-records N, -archive-bytes N, -archive-members N, -backend-timeout D, -backend-output-bytes N, -backend-rows N")
	case "summary":
		fmt.Println("Resource overrides: -output-bytes N, -input-bytes N, -journal-bytes N, -max-records N, -archive-bytes N, -archive-members N, -backend-timeout D, -backend-output-bytes N, -backend-rows N")
		fmt.Println("Options: -json, -html FILE, -view VIEW, -domain driver|runtime, -iteration-marker RE, -perfetto PATH, -shape FILE, -campaign FILE")
	case "query":
		fmt.Println("Resource overrides: -output-bytes N, -input-bytes N, -journal-bytes N, -max-records N, -archive-bytes N, -archive-members N, -backend-timeout D, -backend-output-bytes N, -backend-rows N")
		fmt.Println("Options: -json, -list, -domain driver|runtime, -perfetto PATH, -go-trace-decoder PATH -go-trace-scratch DIR; arguments: NAME INPUT")
	}

	fmt.Fprintf(os.Stdout, "Usage: cudaprof %s [options] INPUT\n\n%s\nInputs: %s\nDefaults: %s\nExit status: 0 diagnostic or successful check; 1 usage, I/O or malformed input; 2 requested qualification failed or unavailable\n\nExamples:\n  %s\n\nPerfetto: open exported .pftrace or comparison TAR directly in ui.perfetto.dev.\nNative checks need no Trace Processor. Import health and collector completeness\nare separate evidence. No browser, download, or installation runs implicitly.\n", spelling, c.purpose, c.inputs, c.defaults, c.examples)
}

func runTraceCommand(args []string) error {
	if len(args) == 0 || (len(args) == 1 && isHelpFlag(args[0])) {
		fmt.Print("Usage: cudaprof trace COMMAND [options]\n\n")
		for _, c := range traceCommands() {
			fmt.Printf("  %-9s %s\n", c.name, c.purpose)
		}
		fmt.Print("\nUse cudaprof help trace COMMAND for command help.\n")
		return nil
	}
	for _, c := range traceCommands() {
		if c.name != args[0] {
			continue
		}
		if len(args) == 2 && isHelpFlag(args[1]) {
			traceHelp(c, "trace "+c.name)
			return nil
		}
		return c.run(args[1:])
	}
	return fmt.Errorf("trace: unknown command %q", args[0])
}

func runTraceAlias(name string, args []string) error {
	for _, c := range traceCommands() {
		if c.name != name {
			continue
		}
		if len(args) == 2 && isHelpFlag(args[1]) {
			traceHelp(c, args[0])
			return nil
		}
		return c.run(args[1:])
	}
	return fmt.Errorf("trace: unknown alias %q", name)
}
