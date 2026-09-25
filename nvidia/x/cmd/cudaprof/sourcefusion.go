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
