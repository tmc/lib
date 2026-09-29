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
	missing := capturefiles.MissingModuleCode(cap)
	if len(missing) == 0 {
		return
	}
	fmt.Fprintf(w, "cudaprof: code not captured for %s\n", capturefiles.DescribeMissingCode(missing))
	// A capture with no decodable timeline still has the census above; it
	// cannot say which modules ran, so it says nothing about that.
	if launches, err := capturefiles.CountModuleLaunches(cap); err == nil {
		if s := capturefiles.DescribeMissingLaunches(missing, launches); s != "" {
			fmt.Fprintf(w, "cudaprof: %s\n", s)
		}
	}
	fmt.Fprintf(w, "cudaprof: pressure, cfg, disasm and code-based advice cannot inspect those modules\n")
}
