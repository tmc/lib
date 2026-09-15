package cuptibench

import (
	"crypto/sha256"
	"fmt"
)

func diagnosticSummary(d *Diagnostics) string {
	summary := fmt.Sprintf("cuptibench: diagnostic-only counter image bytes=%d sha256=%x", len(d.Image), sha256.Sum256(d.Image))
	if d.Trace == nil {
		return summary + "; launch observer unavailable"
	}
	return fmt.Sprintf("%s; observed launches=%d completed trace windows=%d (latest window may be partial)", summary, len(d.Trace.Launches), d.Trace.CompletedPasses)
}
