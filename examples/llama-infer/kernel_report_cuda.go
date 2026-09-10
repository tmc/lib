//go:build !darwin && !tpu

package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/tmc/lib/nvidia/x/cudasass"
)

var kernelReportPath = flag.String("kernel-report", "", "write a report of the compiled forward kernels (PTX, SASS when nvdisasm is present) to this path, or - for stderr")

// writeKernelReport reports what NVRTC produced for the forward kernels.
//
// Two things are written. The per-kernel instruction summary is descriptive.
// The dtype section is an assertion: every weight-bearing entry point is
// checked against the dtype it was bound for, so a module that bound the F16
// names but compiled code that loads float32 weights is visible here rather
// than only in the numbers it produces. A disagreement is reported, not
// returned as an error: this is a diagnostic path and refusing to run on its
// verdict would make an untested check able to stop a working engine.
func writeKernelReport(path string, gm *GPUModules) error {
	w := os.Stderr
	if path != "-" {
		f, err := os.Create(path)
		if err != nil {
			return fmt.Errorf("create kernel report: %w", err)
		}
		defer f.Close()
		w = f
	}
	ptx, cubin := gm.PTX(), gm.CUBIN()
	if ptx == "" && len(cubin) == 0 {
		return fmt.Errorf("the compiled module retained neither ptx nor a cubin")
	}
	d := gm.WeightDType()
	fmt.Fprintf(w, "%s\n\n", weightKernelReport(d))
	if ptx == "" {
		fmt.Fprintf(w, "no ptx: the compiler returned a cubin only\n")
	} else if err := writePTXReport(w, ptx); err != nil {
		return err
	}
	if len(cubin) > 0 {
		// Disassembly needs nvdisasm on PATH. A host without it still gets the
		// PTX report, since a missing tool says nothing about the kernels.
		if disasm, err := cudasass.DisassembleCUBIN(cubin); err != nil {
			fmt.Fprintf(w, "\nno sass: %v\n", err)
		} else {
			fmt.Fprintf(w, "\nsass (%d bytes of cubin; REGISTERS>= is a floor, not the launch register count):\n", len(cubin))
			if err := writeSASSDisassembly(w, disasm); err != nil {
				return err
			}
		}
	} else {
		fmt.Fprintf(w, "\nno sass: the compiler returned ptx only\n")
	}
	if ptx == "" {
		return nil
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
