package main

import (
	"fmt"
	"io"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/internal/capturefiles"
)

// reportMissingModuleCode says which module images the capture did not keep,
// for each view that reads code: a kernel from one of them has no SASS,
// pressure, control-flow graph or code-based advice, and the view would
// otherwise read as if it had none to show.
func reportMissingModuleCode(w io.Writer, cap *cudacapture.Capture) {
	if s := capturefiles.DescribeMissingCode(capturefiles.MissingModuleCode(cap)); s != "" {
		fmt.Fprintf(w, "cudaprof: code not captured for %s; pressure, cfg, disasm and code-based advice cannot inspect those modules\n", s)
	}
}
