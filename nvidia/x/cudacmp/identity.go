package cudacmp

import (
	"errors"
	"fmt"
	"time"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudatrial"
)

// ErrIneligibleComparison indicates that two arms, receipts, or captures cannot be compared
// due to missing or incompatible identity, incomplete execution, or unmeasured timing kinds.
var ErrIneligibleComparison = errors.New("cudacmp: ineligible comparison")

// ComparisonIdentity provides a versioned identity schema for captures and trial runs.
type ComparisonIdentity = cudacapture.ComparisonIdentity

// CheckCompatibility verifies that two comparison identities are compatible for comparative analysis.
func CheckCompatibility(baseline, candidate ComparisonIdentity) error {
	if err := candidate.Compatible(baseline); err != nil {
		return fmt.Errorf("%w: %v", ErrIneligibleComparison, err)
	}
	return nil
}

// ReceiptComparison holds the differential result of comparing two trial receipts.
type ReceiptComparison struct {
	TimingKind   cudatrial.TimingKind
	CDuration    time.Duration
	GoDuration   time.Duration
	Delta        time.Duration // Go - C
	Speedup      float64       // C / Go
	NullDrift    time.Duration
	ExceedsDrift bool
}

// CompareReceipts compares two trial receipts on an explicit timing kind against a null drift threshold.
// It rejects receipts that are incomplete, lack the requested timing kind, or specify a negative drift.
func CompareReceipts(cReceipt, goReceipt cudatrial.Receipt, kind cudatrial.TimingKind, nullDrift time.Duration) (ReceiptComparison, error) {
	if kind == "" {
		return ReceiptComparison{}, fmt.Errorf("%w: timing kind must be specified", ErrIneligibleComparison)
	}
	if !cReceipt.Complete {
		return ReceiptComparison{}, fmt.Errorf("%w: baseline receipt %q round %d is not complete", ErrIneligibleComparison, cReceipt.Name, cReceipt.Round)
	}
	if !goReceipt.Complete {
		return ReceiptComparison{}, fmt.Errorf("%w: candidate receipt %q round %d is not complete", ErrIneligibleComparison, goReceipt.Name, goReceipt.Round)
	}
	if nullDrift < 0 {
		return ReceiptComparison{}, fmt.Errorf("%w: null drift threshold cannot be negative: %v", ErrIneligibleComparison, nullDrift)
	}

	cDur, err := cReceipt.DurationFor(kind)
	if err != nil {
		return ReceiptComparison{}, fmt.Errorf("%w: baseline: %v", ErrIneligibleComparison, err)
	}
	goDur, err := goReceipt.DurationFor(kind)
	if err != nil {
		return ReceiptComparison{}, fmt.Errorf("%w: candidate: %v", ErrIneligibleComparison, err)
	}

	delta := goDur - cDur
	var speedup float64
	if goDur > 0 {
		speedup = float64(cDur) / float64(goDur)
	}
	absDelta := delta
	if absDelta < 0 {
		absDelta = -absDelta
	}
	exceedsDrift := absDelta > nullDrift

	return ReceiptComparison{
		TimingKind:   kind,
		CDuration:    cDur,
		GoDuration:   goDur,
		Delta:        delta,
		Speedup:      speedup,
		NullDrift:    nullDrift,
		ExceedsDrift: exceedsDrift,
	}, nil
}
