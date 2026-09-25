package main

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudacapture"
)

// sourceFusionCaveat is the one-line statement that must accompany anything
// rendered from a capture loaded through a source-fusion failure. It names the
// consequence rather than the error alone, because the consequence is what a
// reader of an empty file name and a zero line number needs to know.
func sourceFusionCaveat(e *cudacapture.SourceFusionError) string {
	return fmt.Sprintf("source attribution unavailable: %v; file names and line numbers below are empty "+
		"because that binding was refused, not because the compiler located nothing", e)
}

// sourceFusionMarkdownCaveat is the same statement as a report header, in the
// shape sampleOnlyCaveat established: above the findings, where it is read
// before the locations it qualifies.
func sourceFusionMarkdownCaveat(failure string) string {
	return "> **Source attribution unavailable.** " + failure +
		"\n>\n> Findings below carry no source file or line: the PTX and SASS of this capture are not from " +
		"one compilation, so the binding was *refused*, not attempted and lost. Kernel names, PC offsets " +
		"and SASS text are unaffected.\n\n"
}
