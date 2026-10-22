//go:build darwin && arm64

package main

import "fmt"

// This is Metal's 11e8cad ds4.c:57069..57104 dispatch predicate, not the
// CUDA/Spark policy. An explicit nonempty override follows the oracle's
// first-byte check; it cannot expand this implementation's format admission.
type metalDSparkPolicy struct {
	m5, streaming, exactSampling, tensorParallel bool
	gateType, downType                           uint32
	seedOverride                                 string
}

func (p metalDSparkPolicy) seedBatchEnabled() bool {
	if p.seedOverride != "" {
		return p.seedOverride[0] != '0'
	}
	return p.m5 && !p.streaming && !p.exactSampling &&
		((p.gateType == 16 && p.downType == 10) || (p.tensorParallel && p.gateType == 39 && p.downType == 39))
}

// All Metal schedules short-fallback before runtime proposal capping. A
// minimum-two preparation is separate from a current proposal's length.
func metalDSparkNeedsShortFallback(prepared, proposal int) (bool, error) {
	if prepared < 2 || proposal < 0 || proposal > 16 {
		return false, fmt.Errorf("metal dspark schedule: invalid prepared block or proposal")
	}
	return proposal < 3, nil
}

func newMetalDSparkSchedulePolicy(seedBatch bool, getenv func(string) string) dsparkSchedulePolicy {
	p := newDSparkSchedulePolicy(seedBatch, false, getenv)
	if seedBatch {
		p.baseSkip = dsparkEnvUint(getenv("DS4_DSPARK_SCHEDULER_SKIP"), 32)
	}
	return p
}
