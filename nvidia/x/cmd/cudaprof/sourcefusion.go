package main

import (
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudacapture"
)

// loadCodeTiered loads a capture's code tiers and separates a source-fusion
// failure from every other load failure.
//
// Since LoadCode began returning tiers rather than all or nothing, a
// source-fusion failure comes back alongside a usable Code whose source
// attribution has been zeroed: every instruction's file is empty and its line
// is 0 because the binding between the PTX and the SASS was refused, not
// because the compiler located nothing. Those two states are indistinguishable
// once rendered, so a caller that discards the error prints the refusal as if
// it were the answer. The failure is returned separately here so each caller
// states it in whatever channel its own output has.
//
// A nonnil error is every other load failure. Callers that already tolerate a
// nil Code keep doing so; the error is returned rather than dropped so they can
// say which tier they are rendering without.
func loadCodeTiered(cap *cudacapture.Capture) (*cudacapture.Code, *cudacapture.SourceFusionError, error) {
	code, err := cudacapture.LoadCode(cap)
	if err == nil {
		return code, nil, nil
	}
	var fusion *cudacapture.SourceFusionError
	if errors.As(err, &fusion) && code != nil {
		return code, fusion, nil
	}
	return code, nil, err
}

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
