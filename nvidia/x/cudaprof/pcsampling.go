// Package cudaprof correlates SASS PCs with hardware stall reasons.
package cudaprof

import "fmt"

type StallReason int

const (
	StallMemory StallReason = iota
	StallScoreboard
	StallLgThrottle
)

func CorrelateStall(pc uint64, reason StallReason) string {
	switch reason {
	case StallMemory:
		return fmt.Sprintf("PC 0x%x: LDG.E stall_mem (force 128-bit vectorization uint4)", pc)
	case StallScoreboard:
		return fmt.Sprintf("PC 0x%x: stall_scoreboard (high register pressure)", pc)
	case StallLgThrottle:
		return fmt.Sprintf("PC 0x%x: lg_throttle (local memory spill)", pc)
	default:
		return fmt.Sprintf("PC 0x%x: unknown stall", pc)
	}
}
