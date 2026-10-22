package dsv4

import (
	"errors"
	"fmt"
	"math"
)

// SpeculativeTrace is the portion of one greedy decode needed by the G1
// identity gate. Strict and Speculative contain the generated token IDs in
// order. Margins contains the strict run's top-1 minus top-2 margin at each
// position. AcceptedDrafts and Cycles are reported by the speculative
// scheduler; both are required to make an identity result non-vacuous.
//
// A trace is host data. It does not retain model or device state.
type SpeculativeTrace struct {
	Strict         []int
	Speculative    []int
	Margins        []float64
	AcceptedDrafts int
	Cycles         int
}

// SpeculativeIdentityReport is the fail-closed result of the G1 greedy
// identity check. Positions before the first margin below the threshold are
// compared strictly. Differences after that boundary are recorded but are
// not evidence against identity because reduction-order drift is allowed
// there.
type SpeculativeIdentityReport struct {
	Cases                 int
	ComparedCases         int
	AcceptedCases         int
	ComparedPositions     int
	EligiblePositions     int
	Mismatches            int
	ToleratedMismatches   int
	FirstMismatchCase     int
	FirstMismatchPosition int
}

// Pass reports whether every complete case exercised speculation and all
// positions before its near-tie boundary matched. A report with no eligible
// position never passes; otherwise a prompt with only near ties could pass
// without testing identity.
func (r SpeculativeIdentityReport) Pass() bool {
	return r.Cases > 0 && r.ComparedCases == r.Cases &&
		r.AcceptedCases == r.Cases && r.EligiblePositions > 0 &&
		r.Mismatches == 0
}

// CheckSpeculativeIdentity compares strict and speculative greedy traces.
// marginThreshold must be finite and non-negative. Each trace must contain
// the same non-empty token and margin counts, and must report at least one
// accepted draft in one or more cycles. Malformed or incomplete traces are
// errors rather than skipped cases, so callers cannot mistake missing data
// for a passing gate.
func CheckSpeculativeIdentity(traces []SpeculativeTrace, marginThreshold float64) (SpeculativeIdentityReport, error) {
	report := SpeculativeIdentityReport{FirstMismatchCase: -1, FirstMismatchPosition: -1}
	if len(traces) == 0 {
		return report, errors.New("speculative identity: no traces")
	}
	if math.IsNaN(marginThreshold) || math.IsInf(marginThreshold, 0) || marginThreshold < 0 {
		return report, errors.New("speculative identity: invalid margin threshold")
	}
	report.Cases = len(traces)
	for caseIndex, trace := range traces {
		if len(trace.Strict) == 0 {
			return report, fmt.Errorf("speculative identity: case %d has no tokens", caseIndex)
		}
		if len(trace.Strict) != len(trace.Speculative) || len(trace.Strict) != len(trace.Margins) {
			return report, fmt.Errorf("speculative identity: case %d has mismatched trace lengths", caseIndex)
		}
		if trace.Cycles <= 0 || trace.AcceptedDrafts <= 0 {
			return report, fmt.Errorf("speculative identity: case %d did not exercise accepted drafts", caseIndex)
		}
		boundary := len(trace.Margins)
		for i, margin := range trace.Margins {
			if math.IsNaN(margin) || math.IsInf(margin, 0) || margin < 0 {
				return report, fmt.Errorf("speculative identity: case %d position %d has invalid margin", caseIndex, i)
			}
			if margin < marginThreshold && boundary == len(trace.Margins) {
				boundary = i
			}
		}
		report.ComparedCases++
		report.AcceptedCases++
		report.ComparedPositions += len(trace.Strict)
		report.EligiblePositions += boundary
		for i := range trace.Strict {
			if trace.Strict[i] == trace.Speculative[i] {
				continue
			}
			if i >= boundary {
				report.ToleratedMismatches++
				continue
			}
			report.Mismatches++
			if report.FirstMismatchCase < 0 {
				report.FirstMismatchCase = caseIndex
				report.FirstMismatchPosition = i
			}
		}
	}
	if report.EligiblePositions == 0 {
		return report, errors.New("speculative identity: no positions precede a near-tie boundary")
	}
	return report, nil
}
