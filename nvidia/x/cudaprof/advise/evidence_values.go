package advise

import (
	"fmt"
	"math"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudaevidence"
)

// WithShareEvidence returns a copy with classified profile-share evidence.
// known must reflect the producer's presence information: the legacy numeric
// field alone cannot distinguish a missing value from a measured zero.
// basis must identify the actual numerator, denominator and time channel.
// An observed profile share is Derived from the stated denominator; suspected
// bottleneck attribution remains Inferred. No result is a measured cause.
func (item Item) WithShareEvidence(source cudaevidence.Source, known bool, basis string) (Item, error) {
	var record cudaevidence.Record
	record.Source = source
	record.Unit = "percent"
	if !known || item.EvidenceLevel == EvidenceUnmeasured {
		record.Reason = "profile share unavailable"
	} else {
		if strings.TrimSpace(basis) == "" {
			return Item{}, fmt.Errorf("classify profile share: missing denominator basis")
		}
		if err := source.Identity.Validate(); err != nil {
			return Item{}, fmt.Errorf("classify profile share: %w", err)
		}
		if item.Location.Kernel == "" || item.Location.Kernel != source.Identity.Entry {
			return Item{}, fmt.Errorf("classify profile share: kernel identity mismatch")
		}
		if math.IsNaN(item.ShareOfProfile) || math.IsInf(item.ShareOfProfile, 0) || item.ShareOfProfile < 0 || item.ShareOfProfile > 100 {
			return Item{}, fmt.Errorf("classify profile share: invalid percentage")
		}
		switch item.EvidenceLevel {
		case EvidenceObserved:
			record.Class = cudaevidence.Derived
			record.Basis = basis
		case EvidenceSuspected:
			record.Class = cudaevidence.Inferred
			record.Basis = basis + "; profile attribution interpreted as a possible bottleneck"
		default:
			return Item{}, fmt.Errorf("classify profile share: unknown evidence level")
		}
		number := item.ShareOfProfile
		record.Number = &number
	}
	value, err := cudaevidence.New(record)
	if err != nil {
		return Item{}, fmt.Errorf("classify profile share: %w", err)
	}
	item.ClassifiedShare = &value
	return item, nil
}
