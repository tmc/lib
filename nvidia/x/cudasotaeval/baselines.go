package cudasotaeval

import (
	"errors"
	"fmt"
)

var (
	// ErrRelatedBaseline indicates an attempt to count related projects as independent baselines.
	ErrRelatedBaseline = errors.New("cudasotaeval: related engine cannot be classified as independent baseline")
)

// DefaultBaselines returns candidate alternatives, without asserting versions or
// supported modes. Qualification must supply observed frozen ToolSpec receipts.
func DefaultBaselines() []Baseline {
	return []Baseline{
		{Name: "NVIDIA Nsight Systems", Category: "timeline", Version: "unverified", ComparisonScope: "candidate timeline comparator; capabilities require pinned receipts", Independent: true},
		{Name: "NVIDIA Nsight Compute", Category: "kernel_analysis", Version: "unverified", ComparisonScope: "candidate kernel comparator; capabilities require pinned receipts", Independent: true},
		{Name: "Parca GPU", Category: "continuous_sampling", Version: "unverified", ComparisonScope: "candidate continuous comparator; capabilities require pinned receipts", Independent: true},
		{Name: "Beyla / OpenTelemetry eBPF Instrumentation (OBI)", Category: "host_telemetry", Version: "unverified", ComparisonScope: "related comparator; capabilities require pinned receipts", Independent: false},
	}
}

// ValidateBaselines checks baseline declarations against evaluation rules.
func ValidateBaselines(baselines []Baseline) error {
	hasTimeline := false
	hasKernel := false

	for _, b := range baselines {
		if b.Name == "" || b.Version == "" || b.ComparisonScope == "" {
			return fmt.Errorf("baseline %q has incomplete declaration", b.Name)
		}
		if (b.Name == "Beyla / OpenTelemetry eBPF Instrumentation (OBI)" || b.Category == "host_telemetry") && b.Independent {
			return fmt.Errorf("%w: %s marked independent", ErrRelatedBaseline, b.Name)
		}
		if b.Category == "timeline" && b.Independent {
			hasTimeline = true
		}
		if b.Category == "kernel_analysis" && b.Independent {
			hasKernel = true
		}
	}

	if !hasTimeline {
		return errors.New("baselines: missing independent timeline baseline (Nsight Systems)")
	}
	if !hasKernel {
		return errors.New("baselines: missing independent kernel analysis baseline (Nsight Compute)")
	}
	return nil
}
