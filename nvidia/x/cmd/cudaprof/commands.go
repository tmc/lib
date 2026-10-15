package main

import (
	"errors"
	"fmt"
	"os"
	"slices"
	"strings"
)

const usage = `Usage: cudaprof [pprof flags] capture.pprof

  -http=127.0.0.1:8087  open interactive pprof
  -no_browser          serve without opening a browser
  -top                 show kernel totals
  -allow_unknown_coverage  rank a profile that carries no device time
  -list=compute        show annotated source
  -disasm=compute      show PTX and SASS
  -cfg[=kernel]        print control flow graph (DOT format)
  -bbcfg[=kernel]      print basic block control flow graph (DOT format)
  -pressure[=kernel]   print live register pressure and stall metrics

To capture a GPU profile from a command:
  cudaprof capture [-o profile.pprof] [--max-records=N] -- command [args...]

To view 5-tier pipeline (Go -> CUDA C++ -> Tile IR -> PTX -> SASS):
  cudaprof view [-http=:8088] [-html=report.html] [kernel] capture.pprof

To explain a PTX instruction, directive, register, or modifier:
  cudaprof explain <ptx-statement-or-keyword>

To diff SASS disassembly between captures or assembly files:
  cudaprof diff [flags] <base.pprof|base.sass> <target.pprof|target.sass>

To embed an existing capture's files:
  cudaprof -pack output.pprof path/to/gpu.pprof

To extract source, assembly, control flow graphs, and Go trace:
  cudaprof -extract new-directory capture.pprof

To view or render control flow graphs:
  cudaprof cfg [kernel] capture.pprof | dot -Tsvg -o graph.svg
  cudaprof bbcfg [kernel] capture.pprof | dot -Tsvg -o graph.svg
  cudaprof inspect --cfg [kernel] capture.pprof

To total samples by the program line each instruction came from:
  cudaprof lines [-sample type] [-format profile|rollup|digraph|dot] [-diff other.pprof] [kernel] capture.pprof

To view register pressure:
  cudaprof pressure [kernel] capture.pprof
  cudaprof inspect --pressure [kernel] capture.pprof

To view automated kernel optimization advice:
  cudaprof advise [-e] [-limit N] capture.pprof
  cudaprof inspect --advise capture.pprof

To analyze GPU idle gaps and duty cycle from the capture journal:
  cudaprof gaps [--warmup-ms=N] [--json] capture.pprof

To diff the GPU idle gaps of two captures (the axis summed kernel time cannot see):
  cudaprof gaps --diff_base=base.pprof [--json] target.pprof

To write the device timeline as a Perfetto trace:
  cudaprof export [-o out.pftrace] capture.pprof
  cudaprof export-compare -iteration-marker RE [-iteration-count N] -o comparison.tar base.pprof target.pprof

To compare explicitly instrumented workload shapes:
  cudaprof shape-compare [-iteration-marker RE] base.pprof target.pprof

To collect separately replayed qualified controlled GEMV counters:
  cudaprof counter-gemv workload|collect -h

To inspect retained memory evidence:
  cudaprof memory capture.pprof

To inspect retained device-to-submission links:
  cudaprof correlate [-domain driver|runtime] capture.pprof

To decompose why target took longer than base, per op or stage, in exact nanoseconds:
  cudaprof compare [-phase i] [-warmup-until=RE | -warmup-ms=N] [-ops ops.txt [-ops-check]] [-stages stages.txt [-stage-entries]] [-by phase|range|module] [-templates] [-allow-unresolved] [-max-records=N] [-json] base.pprof target.pprof

To measure each iteration between launches of a marker kernel, with a gate and advice:
  cudaprof iterations -iteration-marker=RE [-nvtx] [-skip-iterations=1] [-expect-markers=N] [-ops ops.txt] [-json] capture.pprof

To capture a base, a target and a repeated-base control back to back, and compare them:
  cudaprof ab -o dir -base 'command' -target 'command' [-control] [-order base,target,control] [-quiet 'command'] [-compare 'flags'] [-- capture flags...]

To list each kernel's launches, time per launch and launch configurations in one capture:
  cudaprof kernels [-grep RE] [-phase i] [-warmup-until=RE | -warmup-ms=N] [-templates] [-json] capture.pprof

To tabulate the cuBLASLt calls of a capture made with --gemms, or join two by shape:
  cudaprof gemms [-phase i] [-warmup-until=RE | -warmup-ms=N] [-json] capture.pprof
  cudaprof compare -gemms [-phase i] [-warmup-until=RE | -warmup-ms=N] [-json] base.pprof target.pprof

To score an ABBA timing campaign (blocks of C1,G1,G2,C2 captures) against its noise and sign gates:
  cudaprof qualify (-warmup-until=RE | -warmup-ms=N) [-noise-multiplier=3] [-noise-floor-ms=1] [-violations=FILE [-violation-margin=2s] [-min-blocks=N]] [-ops ops.txt] [-stages stages.txt] [-max-records=N] [-json] manifest.txt

To inspect observed tokens and operation hierarchy:
  cudaprof overview [-html overview.html] [-iteration-marker RE] capture.pprof

To summarize qualified operation matching and existing alignment receipts:
  cudaprof comparison-summary -h

To collect an interleaved correctness-gated campaign inside one GPU lock:
  cudaprof campaign -h

To inspect available dependency evidence:
  cudaprof dependency capture.pprof

To run a controlled native capability probe inside a GPU lock:
  cudaprof probe [-um | -metrics names] [-managed]

To validate historical counter selections or compare collected receipts:
  cudaprof counter-receipt select -shape FILE -identity FILE
  cudaprof counter-receipt compare BASE.json TARGET.json

To export, check, summarize, or query a timeline:
  cudaprof trace COMMAND [options]

To retain late GPU isolation evidence in a new capture:
  cudaprof isolation bind -receipt FILE -o NEW.pprof INPUT.pprof

To print the build of this cudaprof:
  cudaprof version

To import a bounded offline Nsight Compute project export:
  cudaprof import-ncu [-json] [-limit N] envelope.json

To summarize supplied compiler members against retained activity:
  cudaprof operator-summary -run-id ID context.json activities.jsonl

To read immutable profiling history or plan an observation-driven bisect:
  cudaprof history snapshot.ndjson
  cudaprof history compare [flags] before.json after.json
  cudaprof bisect request.json

To derive an experiment plan from an admitted incident or verify reproducibility:
  cudaprof regress plan -incident incident.json [-o plan.json]
  cudaprof regress check -plan plan.json -environment observed-environment.json
  cudaprof regress export -plan plan.json -o dir

To print the usage of a command:
  cudaprof help [command]
`

const (
	extractUsage = "usage: cudaprof -extract [--capture-file-bytes=N] [--capture-total-bytes=N] new-directory capture.pprof"
	packUsage    = "usage: cudaprof -pack output.pprof path/to/gpu.pprof"
	helpUsage    = "usage: cudaprof help [command]"
	versionUsage = "usage: cudaprof version"
)

// A command is a cudaprof subcommand.
type command struct {
	name    string
	aliases []string

	// usage is printed for -h and by "cudaprof help name".
	// It is empty for a command that prints its own help for -h.
	usage string

	// run runs the command. args[0] is the name it was invoked by.
	run func(args []string) error
}

// commands returns the subcommands that run dispatches.
func commands() []command {
	return []command{
		{name: "isolation", run: func(args []string) error { return runIsolationCommand(args[1:], os.Stdout) }},
		{name: "trace", run: func(args []string) error { return runTraceCommand(args[1:]) }},
		{name: "counter-gemv", run: func(args []string) error { return runCounterGEMVCommand(args[1:]) }},
		{name: "overview", run: func(args []string) error { return runOverviewCommand(args[1:]) }},
		{name: "comparison-summary", run: func(args []string) error { return runComparisonSummaryCommand(args[1:], os.Stdout) }},
		{name: "campaign", run: func(args []string) error { return runCampaignCommand(args[1:], os.Stdout) }},
		{name: "dependency", run: func(args []string) error { return runDependencyCommand(args[1:]) }},
		{name: "probe", run: func(args []string) error { return runProbeCommand(args[1:]) }},
		{name: "counter-receipt", run: func(args []string) error { return runCounterReceiptCommand(args[1:]) }},
		{name: "capture", run: func(args []string) error { return runCaptureCommand(args[1:]) }},
		{name: "import-ncu", usage: importNCUUsage, run: func(args []string) error { return runImportNCUCommand(args[1:], os.Stdout) }},
		{name: "operator-summary", usage: operatorSummaryUsage, run: func(args []string) error { return runOperatorSummaryCommand(args[1:], os.Stdout) }},
		{name: "history", usage: historyUsage, run: func(args []string) error { return runHistoryCommand(args[1:], os.Stdout) }},
		{name: "bisect", usage: bisectUsage, run: func(args []string) error { return runBisectCommand(args[1:], os.Stdout) }},
		{name: "regress", usage: regressUsage, run: func(args []string) error { return runRegressCommand(args[1:], os.Stdout) }},
		{name: "inspect", usage: inspectUsage, run: func(args []string) error { return runInspectCommand(args[1:]) }},
		{name: "export-compare", run: func(args []string) error { return runTraceAlias("compare", args) }},
		{name: "export", run: func(args []string) error { return runTraceAlias("export", args) }},
		{name: "explain", usage: explainUsage, run: func(args []string) error { return runExplainCommand(args[1:]) }},
		{name: "diff", usage: diffUsage, run: func(args []string) error { return runDiffCommand(args[1:]) }},
		{name: "cfg", usage: cfgUsage, run: runCFGCommand},
		{name: "bbcfg", usage: bbcfgUsage, run: runCFGCommand},
		{name: "view", aliases: []string{"web"}, run: runViewCommand},
		{name: "advise", usage: adviseUsage, run: func(args []string) error { return runAdviseCommand(args[1:]) }},
		{name: "gaps", run: func(args []string) error { return runGapsCommand(args[1:]) }},
		{name: "gemms", usage: gemmsUsage, run: runGEMMsCommand},
		{name: "shape-compare", run: func(args []string) error { return runShapeCompareCommand(args[1:]) }},
		{name: "memory", run: func(args []string) error { return runMemoryCommand(args[1:]) }},
		{name: "correlate", run: func(args []string) error { return runCorrelateCommand(args[1:]) }},
		{name: "compare", run: func(args []string) error { return runCompareCommand(args[1:], os.Stdout) }},
		{name: "iterations", usage: iterationsUsage, run: func(args []string) error { return runIterationsCommand(args[1:], os.Stdout) }},
		{name: "ab", run: func(args []string) error { return runABCommand(args[1:], os.Stdout, os.Stderr) }},
		{name: "kernels", run: func(args []string) error { return runKernelsCommand(args[1:], os.Stdout) }},
		{name: "qualify", run: func(args []string) error { return runQualifyCommand(args[1:], os.Stdout) }},
		{name: "lines", usage: linesUsage, run: runLinesCommand},
		{name: "pressure", usage: pressureUsage, run: runPressureCommand},
		{name: "-extract", usage: extractUsage, run: runExtractCommand},
		{name: "-pack", usage: packUsage, run: runPackCommand},
		{name: "version", usage: versionUsage, run: runVersionCommand},
		{name: "help", usage: helpUsage, run: runHelpCommand},
	}
}

// lookupCommand returns the command named by arg, the first argument.
// Besides the command names, it accepts the pprof-style flag spellings
// -cfg, -bbcfg, -pressure (each optionally =kernel) and -web.
func lookupCommand(arg string) (command, bool) {
	name := arg
	for _, n := range []string{"cfg", "bbcfg", "pressure"} {
		if arg == "-"+n || strings.HasPrefix(arg, "-"+n+"=") || strings.HasPrefix(arg, "--"+n) {
			name = n
		}
	}
	if arg == "-web" || arg == "--web" {
		name = "web"
	}
	for _, c := range commands() {
		if c.name == name || slices.Contains(c.aliases, name) {
			return c, true
		}
	}
	return command{}, false
}

// invoke runs c, printing c.usage instead when args asks only for help.
func (c command) invoke(args []string) error {
	if c.usage != "" && len(args) == 2 && isHelpFlag(args[1]) {
		fmt.Println(c.usage)
		return nil
	}
	return c.run(args)
}

func isHelpFlag(arg string) bool {
	return arg == "-h" || arg == "-help" || arg == "--help"
}

// runHelpCommand implements "cudaprof help [command]".
func runHelpCommand(args []string) error {
	if len(args) == 3 && args[1] == "trace" {
		return runTraceCommand([]string{args[2], "-h"})
	}
	switch len(args) {
	case 1:
		fmt.Print(usage)
		return nil
	case 2:
		c, ok := lookupCommand(args[1])
		if !ok {
			return fmt.Errorf("help: unknown command %q\n%s", args[1], helpUsage)
		}
		return c.invoke([]string{args[1], "-h"})
	}
	return errors.New(helpUsage)
}

// runExtractCommand implements "cudaprof -extract".
func runExtractCommand(args []string) error {
	args, limits, err := parseReadLimits(args[1:])
	if err != nil {
		return err
	}
	if len(args) != 2 {
		return errors.New(extractUsage)
	}
	return extract(args[0], args[1], limits)
}

// runPackCommand implements "cudaprof -pack".
func runPackCommand(args []string) error {
	if len(args) != 3 {
		return errors.New(packUsage)
	}
	return pack(args[1], args[2])
}
