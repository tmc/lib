//go:build !darwin && !tpu

package main

import (
	"flag"
	"fmt"
	"os"
)

var ptxReportPath = flag.String("ptx-report", "", "write a report of the compiled forward-kernel PTX to this path, or - for stderr")

// writeCompiledPTXReport reports what NVRTC produced for the forward kernels.
//
// Two things are written. The per-kernel instruction summary is descriptive.
// The dtype section is an assertion: every weight-bearing entry point is
// checked against the dtype it was bound for, so a module that bound the F16
// names but compiled code that loads float32 weights is visible here rather
// than only in the numbers it produces. A disagreement is reported, not
// returned as an error: this is a diagnostic path and refusing to run on its
// verdict would make an untested check able to stop a working engine.
func writeCompiledPTXReport(path string, gm *GPUModules) error {
	w := os.Stderr
	if path != "-" {
		f, err := os.Create(path)
		if err != nil {
			return fmt.Errorf("create ptx report: %w", err)
		}
		defer f.Close()
		w = f
	}
	ptx := gm.PTX()
	if ptx == "" {
		return fmt.Errorf("no ptx was retained for the compiled module")
	}
	d := gm.WeightDType()
	fmt.Fprintf(w, "%s\n\n", weightKernelReport(d))
	if err := writePTXReport(w, ptx); err != nil {
		return err
	}
	fmt.Fprintf(w, "\nweight dtype evidence (%s):\n", d)
	for _, name := range weightBearingKernelNames(d) {
		if err := checkWeightKernelPTX(ptx, d, name); err != nil {
			fmt.Fprintf(w, "  DISAGREES %v\n", err)
			continue
		}
		fmt.Fprintf(w, "  ok        %s\n", name)
	}
	return nil
}
