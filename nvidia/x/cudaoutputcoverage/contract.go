package cudaoutputcoverage

import (
	"fmt"
	"math"
	"strings"
)

// Kind identifies a closed write-contract policy.
type Kind string

const (
	ExactlyOnce Kind = "exactly_once"
	AtLeastOnce Kind = "at_least_once"
	WriterCount Kind = "writer_count"
	Accumulated Kind = "accumulated"
)

// Range is a half-open logical element interval [Start, End).
type Range struct{ Start, End int }

// Contract declares a buffer's write domain and writer semantics. Elements
// includes padding; Domain excludes it and must be sorted, nonoverlapping and
// nonempty. Atomic and Reduction require WriterCount or Accumulated.
type Contract struct {
	Buffer            string
	Elements          int
	Domain            []Range
	Kind              Kind
	Atomic, Reduction bool
	ExpectedWriters   uint64
	Deterministic     bool
}

// Validate rejects malformed domains and contradictory writer contracts.
func (c Contract) Validate() error {
	bad := func(s string) error { return fmt.Errorf("output contract: %s", s) }
	if strings.TrimSpace(c.Buffer) == "" || c.Elements <= 0 || c.Elements > 1<<24 || len(c.Domain) == 0 || len(c.Domain) > 1<<20 {
		return bad("invalid buffer or domain extent")
	}
	end := 0
	for _, r := range c.Domain {
		if r.Start < end || r.Start < 0 || r.End <= r.Start || r.End > c.Elements {
			return bad("invalid or overlapping domain range")
		}
		end = r.End
	}
	switch c.Kind {
	case ExactlyOnce, AtLeastOnce:
		if c.Atomic || c.Reduction || c.ExpectedWriters != 0 {
			return bad("ordinary contract applied to counted or accumulated output")
		}
	case WriterCount:
		if c.ExpectedWriters == 0 {
			return bad("writer count must be positive")
		}
	case Accumulated:
		if !c.Atomic && !c.Reduction {
			return bad("accumulated output must declare atomic or reduction semantics")
		}
		if c.ExpectedWriters != 0 {
			return bad("accumulated contract cannot also declare writer count")
		}
	default:
		return bad("unknown contract kind")
	}
	return nil
}

// Geometry records the admitted launch dimensions, rather than deriving them
// from the logical domain.
type Geometry struct{ Grid, Block [3]uint32 }

// TwoFill records reference and readback bits from two executions with the
// same admitted inputs, artifact, launch geometry and contract.
type TwoFill struct {
	Contract                        Contract
	FillA, FillB                    uint32
	Reference, ReadbackA, ReadbackB []uint32
	Geometry                        Geometry
	SourceDigest, InputDigest       string
	Target, MathMode                string
}

// Report separates write-domain failures from nondeterministic readbacks.
// Numerical correctness must be checked independently.
type Report struct {
	Contract                                    Contract
	Arm                                         string
	Geometry                                    Geometry
	Unwritten, Nondeterministic                 []Range
	SourceDigest, InputDigest, Target, MathMode string
}

func finite(bits uint32) bool {
	v := float64(math.Float32frombits(bits))
	return !math.IsNaN(v) && !math.IsInf(v, 0)
}
func digest(s string) bool {
	if len(s) != 64 {
		return false
	}
	for _, c := range s {
		if !(c >= '0' && c <= '9' || c >= 'a' && c <= 'f') {
			return false
		}
	}
	return true
}

func validateTwoFill(t TwoFill) error {
	if err := t.Contract.Validate(); err != nil {
		return err
	}
	if t.Contract.Kind != AtLeastOnce || !t.Contract.Deterministic {
		return fmt.Errorf("two fill: requires deterministic ordinary at-least-once contract")
	}
	if len(t.Reference) != t.Contract.Elements {
		return fmt.Errorf("two fill: reference extent differs from declared buffer")
	}
	if !digest(t.SourceDigest) || !digest(t.InputDigest) || strings.TrimSpace(t.Target) == "" || strings.TrimSpace(t.MathMode) == "" {
		return fmt.Errorf("two fill: missing execution identity")
	}
	for i := 0; i < 3; i++ {
		if t.Geometry.Grid[i] == 0 || t.Geometry.Block[i] == 0 {
			return fmt.Errorf("two fill: invalid launch geometry")
		}
	}
	if !finite(t.FillA) || !finite(t.FillB) || math.Float32frombits(t.FillA) == math.Float32frombits(t.FillB) {
		return fmt.Errorf("two fill: fill values must be finite and numerically distinct")
	}
	for _, r := range t.Contract.Domain {
		for i := r.Start; i < r.End; i++ {
			v := math.Float32frombits(t.Reference[i])
			if !finite(t.Reference[i]) || v == math.Float32frombits(t.FillA) || v == math.Float32frombits(t.FillB) {
				return fmt.Errorf("two fill: reference at element %d conflicts with fills or is nonfinite", i)
			}
		}
	}
	return nil
}

// ClassifyTwoFill checks recorded readbacks without touching hardware. Invalid
// admission is an error, never an empty successful report.
func ClassifyTwoFill(t TwoFill) (Report, error) {
	if err := validateTwoFill(t); err != nil {
		return Report{}, err
	}
	if len(t.ReadbackA) != t.Contract.Elements || len(t.ReadbackB) != t.Contract.Elements {
		return Report{}, fmt.Errorf("two fill: readback extent differs from declared buffer")
	}
	out := Report{Contract: t.Contract, Arm: "two_fill", Geometry: t.Geometry, SourceDigest: t.SourceDigest, InputDigest: t.InputDigest, Target: t.Target, MathMode: t.MathMode}
	out.Contract.Domain = append([]Range(nil), t.Contract.Domain...)
	for _, r := range t.Contract.Domain {
		for i := r.Start; i < r.End; i++ {
			a, b := t.ReadbackA[i], t.ReadbackB[i]
			if a == t.FillA && b == t.FillB {
				out.Unwritten = appendIndex(out.Unwritten, i)
			} else if a != b {
				out.Nondeterministic = appendIndex(out.Nondeterministic, i)
			}
		}
	}
	return out, nil
}

func appendIndex(ranges []Range, i int) []Range {
	if len(ranges) > 0 && ranges[len(ranges)-1].End == i {
		ranges[len(ranges)-1].End++
		return ranges
	}
	return append(ranges, Range{i, i + 1})
}
