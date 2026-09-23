// Command source-wiring follows kernels written in Go down to the instructions
// the GPU executes.
//
// Four kernels are built with cudatensor, lowered to CUDA C++ by cudafuse,
// and attributed by cudalineinfo, which compiles them with NVRTC and
// disassembles them with nvdisasm. For each Go statement the program reports
// how many generated CUDA lines, PTX locations, and SASS instructions name it,
// so every tier can be checked against the line of Go it came from. The first
// kernel is then listed instruction by instruction.
//
// The last kernel is built through helper functions. Its table names the
// lines inside the helpers, and the roll-up that follows charges those
// instructions to the callers too, the way a profiler's cumulative column
// does, so the two call lines in the kernel body account for all of it.
//
// The first kernel is then recompiled with -use_fast_math. The per-line diff
// shows which instructions the flag removed, and running both builds on the
// same inputs shows what the shortcut cost in accuracy.
//
// The control at the end recompiles with the #line directives withheld. The
// attribution disappears, which is what makes the tables above it evidence
// rather than decoration.
//
// Usage:
//
//	source-wiring [-arch sm_121] [-nvdisasm path] [-pcsampling [-pprof file]] [-sassmetrics] [-units dir]
//
// -units writes each kernel's unit, annotated source, PTX, CUBIN, and
// disassembly to a directory, in the forms the cudalineinfo command reads,
// so the same questions can be asked of the artifacts afterwards without
// rebuilding them.
//
// With -pcsampling or -sassmetrics the first kernel is also run under CUPTI
// and the hardware's PC samples, or its exact executed-instruction counts,
// are reported per Go line next to the static counts. -pprof writes the
// samples as a pprof profile whose stacks run from the SASS instruction up
// through the Go statement and the frames that called it. Each collector
// times its own launches, so the run ends with what the measurement cost
// against the same launches with nothing attached.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/tmc/lib/nvidia/x/cudafuse"
	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudalineinfo"
	"github.com/tmc/lib/nvidia/x/cudaprofagent"
)

func main() {
	arch := flag.String("arch", "sm_121", "physical architecture to compile for")
	nvdisasm := flag.String("nvdisasm", "", "path to nvdisasm; empty searches PATH")
	var cfg profileConfig
	flag.BoolVar(&cfg.pcSampling, "pcsampling", false, "sample warp PCs while the first kernel runs")
	flag.BoolVar(&cfg.sassMetrics, "sassmetrics", false, "count executed instructions while the first kernel runs")
	flag.StringVar(&cfg.pprof, "pprof", "", "with -pcsampling, write the samples as a pprof profile to this `file`")
	flag.UintVar(&cfg.period, "period", 10, "PC sampling period exponent to start at: one sample per 2^n GPU cycles, 5 through 31; a period that overflows the hardware buffer is lengthened")
	flag.IntVar(&cfg.n, "n", 1<<20, "elements per profiled launch")
	flag.IntVar(&cfg.launches, "launches", 20, "launches per profiled run")
	units := flag.String("units", "", "write each kernel's unit and build artifacts to this `dir` for the cudalineinfo command")
	flag.Parse()

	// A Go program leaves through exit_group, which runs none of the exits
	// an attached cudaprof capture agent flushes on; asking is what makes
	// `cudaprof capture -- source-wiring` hold the kernels it ran.
	defer cudaprofagent.Finalize()
	if err := run(*arch, *nvdisasm, *units, cfg); err != nil {
		log.Fatal(err)
	}
}

func run(arch, nvdisasm, units string, cfg profileConfig) error {
	ctx := context.Background()
	opts := cudalineinfo.Options{Arch: arch, Nvdisasm: nvdisasm}
	for i, k := range kernels {
		a, fk, err := attribute(ctx, k, opts)
		if err != nil {
			return err
		}
		if units != "" {
			if err := export(ctx, units, a, nvdisasm); err != nil {
				return err
			}
		}
		// Under cudaprof the same unit goes into the capture, where it is
		// what lets the profile be read in terms of this program's lines
		// rather than the generated CUDA nobody wrote.
		if err := publish(a.Unit); err != nil {
			return err
		}
		fmt.Println()
		cudalineinfo.WriteTable(os.Stdout, a)
		switch i {
		case 0:
			fmt.Println()
			cudalineinfo.WriteListing(os.Stdout, a)
			if err := profile(ctx, cfg, a, fk); err != nil {
				return err
			}
		case len(kernels) - 1:
			fmt.Println()
			cudalineinfo.WriteRollup(os.Stdout, a)
		}
	}
	if err := fastMath(ctx, opts); err != nil {
		return err
	}
	return control(ctx, opts)
}

// attribute builds a kernel's graph, lowers it, and attributes the compiled
// result to the Go that built it. The fused kernel is returned as well for
// callers that launch the result.
func attribute(ctx context.Context, k kernel, opts cudalineinfo.Options) (*cudalineinfo.Attribution, cudafuse.FusedKernel, error) {
	g := cudair.NewGraph()
	k.build(g)
	fk, err := cudafuse.FuseWithSourceMap(g, k.name)
	if err != nil {
		return nil, fk, fmt.Errorf("%s: fuse: %w", k.name, err)
	}
	a, err := cudalineinfo.Compile(ctx, fk.Unit(), opts)
	if err != nil {
		return nil, fk, fmt.Errorf("%s: %w", k.name, err)
	}
	return a, fk, nil
}
