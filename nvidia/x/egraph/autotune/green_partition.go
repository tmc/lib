package autotune

import (
	"errors"
	"fmt"
	"math"
	"time"
)

// ErrGreenContextInterference reports that background candidate verification in
// the green-context partition perturbed foreground decode P99 latency beyond
// the allowed jitter budget.
var ErrGreenContextInterference = errors.New("autotune: background green-context perturbed foreground p99 latency")

const (
	defaultTPCClusterAlignment = 2
	minBackgroundAutotuneSMs   = 2
	maxBackgroundAutotuneSMs   = 8
	minForegroundSMFraction    = 0.85
	defaultMaxJitterRatio      = 0.02
)

// GreenContextPartitionSpec describes a Hopper/Blackwell TPC/cluster-aligned
// SM partition between a foreground persistent decode megakernel and a
// background autotune verification context.
type GreenContextPartitionSpec struct {
	TotalSMs              int     `json:"total_sms"`
	ClusterAlignment      int     `json:"cluster_alignment"`
	ForegroundDecodeSMs   int     `json:"foreground_decode_sms"`
	BackgroundAutotuneSMs int     `json:"background_autotune_sms"`
	ForegroundFraction    float64 `json:"foreground_fraction"`
	BackgroundCompileLoad float64 `json:"background_compile_load"`
}

// BuildGreenContextPartitionSpec computes a 2-SM TPC/cluster-aligned
// [GreenContextPartitionSpec] for totalSMs and bgCompileLoad in [0, 1].
func BuildGreenContextPartitionSpec(totalSMs int, bgCompileLoad float64) (GreenContextPartitionSpec, error) {
	fg, bg, err := PlanGreenContextSplit(totalSMs, bgCompileLoad)
	if err != nil {
		return GreenContextPartitionSpec{}, err
	}
	alignedTotal := fg + bg
	return GreenContextPartitionSpec{
		TotalSMs:              totalSMs,
		ClusterAlignment:      defaultTPCClusterAlignment,
		ForegroundDecodeSMs:   fg,
		BackgroundAutotuneSMs: bg,
		ForegroundFraction:    float64(fg) / float64(alignedTotal),
		BackgroundCompileLoad: bgCompileLoad,
	}, nil
}

// PlanGreenContextSplit partitions totalSMs into hardware-aligned foreground
// decode SMs and background autotune SMs along 2-SM TPC/cluster boundaries.
// It reserves 85-95%+ of aligned SMs for the foreground persistent megakernel
// and 2-8 SMs for background candidate verification so background profiling
// cannot perturb foreground tail latency.
func PlanGreenContextSplit(totalSMs int, bgCompileLoad float64) (fgDecodeSMs, bgAutotuneSMs int, err error) {
	if totalSMs < 2*defaultTPCClusterAlignment {
		return 0, 0, fmt.Errorf("green context split: totalSMs %d < minimum %d", totalSMs, 2*defaultTPCClusterAlignment)
	}
	if math.IsNaN(bgCompileLoad) || math.IsInf(bgCompileLoad, 0) || bgCompileLoad < 0.0 || bgCompileLoad > 1.0 {
		return 0, 0, fmt.Errorf("green context split: bgCompileLoad %v outside [0, 1]", bgCompileLoad)
	}

	alignedTotal := (totalSMs / defaultTPCClusterAlignment) * defaultTPCClusterAlignment

	// Scale background SMs between 2 and 8 in 2-SM TPC increments.
	rawBG := float64(minBackgroundAutotuneSMs) + bgCompileLoad*float64(maxBackgroundAutotuneSMs-minBackgroundAutotuneSMs)
	bg := int(math.Round(rawBG/float64(defaultTPCClusterAlignment))) * defaultTPCClusterAlignment
	if bg < minBackgroundAutotuneSMs {
		bg = minBackgroundAutotuneSMs
	}
	if bg > maxBackgroundAutotuneSMs {
		bg = maxBackgroundAutotuneSMs
	}

	// Enforce that foreground decode retains at least 85% of SMs whenever
	// alignedTotal is large enough to accommodate a >=85% / >=2-SM split.
	maxBGFor85Pct := int(math.Floor(float64(alignedTotal)*(1.0-minForegroundSMFraction))) / defaultTPCClusterAlignment * defaultTPCClusterAlignment
	if maxBGFor85Pct >= minBackgroundAutotuneSMs && bg > maxBGFor85Pct {
		bg = maxBGFor85Pct
	}
	if alignedTotal-bg < defaultTPCClusterAlignment {
		bg = defaultTPCClusterAlignment
	}

	fg := alignedTotal - bg
	return fg, bg, nil
}

// ValidateNoInterferenceReceipt verifies that running background green-context
// candidate verification alongside the foreground persistent megakernel does
// not increase foreground P99 tail latency beyond maxJitterRatio (defaulting
// to 0.02 when maxJitterRatio <= 0).
func ValidateNoInterferenceReceipt(fgP99WithBG, fgP99Alone time.Duration, maxJitterRatio float64) error {
	if fgP99Alone <= 0 {
		return fmt.Errorf("validate no-interference receipt: fgP99Alone %v <= 0", fgP99Alone)
	}
	if fgP99WithBG <= 0 {
		return fmt.Errorf("validate no-interference receipt: fgP99WithBG %v <= 0", fgP99WithBG)
	}
	if math.IsNaN(maxJitterRatio) || math.IsInf(maxJitterRatio, 0) {
		return fmt.Errorf("validate no-interference receipt: invalid maxJitterRatio %v", maxJitterRatio)
	}
	if maxJitterRatio <= 0 {
		maxJitterRatio = defaultMaxJitterRatio
	}
	jitter := float64(fgP99WithBG-fgP99Alone) / float64(fgP99Alone)
	if jitter > maxJitterRatio {
		return fmt.Errorf("%w: foreground p99 with bg %v vs alone %v (jitter %.4f > max %.4f)",
			ErrGreenContextInterference, fgP99WithBG, fgP99Alone, jitter, maxJitterRatio)
	}
	return nil
}
