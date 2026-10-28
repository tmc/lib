package ncclinspector

import (
	"errors"
	"math"
	"math/big"
	"sync"
	"testing"
)

func clockSpec() ClockTransformSpec {
	return ClockTransformSpec{Source: ClockIdentity{Host: "node-a", Domain: ClockDomainHostMonotonic, BootEpoch: "boot-a", ProcessEpoch: "process-a"}, Reference: ClockIdentity{Host: "reference", Domain: ClockDomainHostMonotonic, BootEpoch: "boot-ref", ProcessEpoch: "reference-1"}, SourceUnit: "nanoseconds", ReferenceUnit: "nanoseconds", Anchors: []ClockAnchor{{SourceNS: 10, ReferenceMinNS: 100, ReferenceMaxNS: 103}}, RateMinPPB: 1_000_000_000, RateMaxPPB: 1_000_000_000, DriftKnown: true, UncertaintyKnown: true, UncertaintyNS: 2, ValidFromNS: 0, ValidUntilNS: 20, Method: "supplied interval calibration", VerificationRef: "constructed-verification.json"}
}
func sourceTime(spec ClockTransformSpec, n, unc int64) BoundedTimestamp {
	return BoundedTimestamp{Nanoseconds: n, UncertaintyNS: unc, Domain: spec.Source.Domain, Epoch: spec.Source.ProcessEpoch, BootEpoch: spec.Source.BootEpoch}
}

func TestClockTransformBounds(t *testing.T) {
	spec := clockSpec()
	transform, err := NewClockTransform(spec)
	if err != nil {
		t.Fatal(err)
	}
	got, err := sourceTime(spec, 11, 1).Aligned(transform.Mapping())
	if err != nil {
		t.Fatal(err)
	}
	if got.Nanoseconds != 102 || got.UncertaintyNS != 5 || got.Epoch != spec.Reference.ProcessEpoch || got.BootEpoch != spec.Reference.BootEpoch {
		t.Fatalf("asymmetric bound: %+v", got)
	}
	spec.RateMinPPB, spec.RateMaxPPB, spec.DriftPPB = 500_000_000, 1_500_000_000, 100_000_000
	transform, err = NewClockTransform(spec)
	if err != nil {
		t.Fatal(err)
	}
	got, err = sourceTime(spec, 9, 2).Aligned(transform.Mapping())
	if err != nil {
		t.Fatal(err)
	}
	if got.Nanoseconds != 100 || got.UncertaintyNS != 8 {
		t.Fatalf("rate/drift bound: %+v", got)
	}
	// Independent rational endpoint enumeration must fit the returned enclosure.
	for _, x := range []int64{7, 8, 9, 10, 11} {
		for _, rate := range []int64{spec.RateMinPPB, 1_000_000_001, spec.RateMaxPPB} {
			for _, ref := range []int64{100, 103} {
				actual := new(big.Rat).SetFrac(new(big.Int).Mul(big.NewInt(x-10), big.NewInt(rate)), big.NewInt(1_000_000_000))
				actual.Add(actual, new(big.Rat).SetInt64(ref))
				lower, upper := new(big.Rat).SetInt64(got.Nanoseconds-got.UncertaintyNS), new(big.Rat).SetInt64(got.Nanoseconds+got.UncertaintyNS)
				if actual.Cmp(lower) < 0 || actual.Cmp(upper) > 0 {
					t.Fatalf("bound excludes exact rational %s: %+v", actual, got)
				}
			}
		}
	}
}

func TestClockTransformAnchorsAndDrift(t *testing.T) {
	spec := clockSpec()
	spec.UncertaintyNS = 0
	spec.Anchors = []ClockAnchor{{0, 100, 104}, {20, 121, 125}}
	transform, err := NewClockTransform(spec)
	if err != nil {
		t.Fatal(err)
	}
	got, err := sourceTime(spec, 10, 0).Aligned(transform.Mapping())
	if err != nil {
		t.Fatal(err)
	}
	if got.Nanoseconds != 112 || got.UncertaintyNS != 2 {
		t.Fatalf("multiple-anchor intersection: %+v", got)
	}
	spec.Anchors = []ClockAnchor{{0, 100, 100}}
	spec.ValidUntilNS = 1_000_000
	spec.DriftPPB = 1_000
	transform, err = NewClockTransform(spec)
	if err != nil {
		t.Fatal(err)
	}
	near, _ := sourceTime(spec, 0, 0).Aligned(transform.Mapping())
	far, err := sourceTime(spec, 1_000_000, 0).Aligned(transform.Mapping())
	if err != nil {
		t.Fatal(err)
	}
	if near.UncertaintyNS != 0 || far.UncertaintyNS != 1 {
		t.Fatalf("drift did not grow: near=%+v far=%+v", near, far)
	}
}

func TestClockTransformRefusals(t *testing.T) {
	for _, tc := range []struct {
		name   string
		change func(*ClockTransformSpec)
	}{
		{"missing anchors", func(s *ClockTransformSpec) { s.Anchors = nil }},
		{"too many anchors", func(s *ClockTransformSpec) {
			s.Anchors = nil
			for i := int64(0); i < 17; i++ {
				s.Anchors = append(s.Anchors, ClockAnchor{i, 100 + i, 100 + i})
			}
		}},
		{"unordered", func(s *ClockTransformSpec) { s.Anchors = append(s.Anchors, s.Anchors[0]) }},
		{"asymmetric reversed", func(s *ClockTransformSpec) { s.Anchors[0].ReferenceMinNS = 104 }},
		{"contradictory anchors", func(s *ClockTransformSpec) { s.Anchors = append(s.Anchors, ClockAnchor{20, 1000, 1000}) }},
		{"non-affine anchors", func(s *ClockTransformSpec) {
			s.Anchors = []ClockAnchor{{0, 0, 0}, {10, 10, 10}, {20, 30, 30}}
			s.RateMinPPB = 500_000_000
			s.RateMaxPPB = 2_000_000_000
		}},
		{"unknown rate", func(s *ClockTransformSpec) { s.RateMinPPB = 0 }},
		{"reversed rate", func(s *ClockTransformSpec) { s.RateMaxPPB = s.RateMinPPB - 1 }},
		{"unknown drift", func(s *ClockTransformSpec) { s.DriftKnown = false }},
		{"negative source PID", func(s *ClockTransformSpec) { s.Source.PID = -1 }},
		{"negative reference PID", func(s *ClockTransformSpec) { s.Reference.PID = -1 }},
		{"negative drift", func(s *ClockTransformSpec) { s.DriftPPB = -1 }},
		{"unknown residual", func(s *ClockTransformSpec) { s.UncertaintyKnown = false }},
		{"negative residual", func(s *ClockTransformSpec) { s.UncertaintyNS = -1 }},
		{"unknown unit", func(s *ClockTransformSpec) { s.SourceUnit = "" }},
		{"source units", func(s *ClockTransformSpec) { s.SourceUnit = "microseconds" }},
		{"reference units", func(s *ClockTransformSpec) { s.ReferenceUnit = "cycles" }},
		{"source host", func(s *ClockTransformSpec) { s.Source.Host = "" }},
		{"source boot", func(s *ClockTransformSpec) { s.Source.BootEpoch = "" }},
		{"reference process", func(s *ClockTransformSpec) { s.Reference.ProcessEpoch = "" }},
		{"wall step", func(s *ClockTransformSpec) { s.WallStepObserved = true }},
		{"unknown wall status", func(s *ClockTransformSpec) { s.Source.Domain = ClockDomainHostRealtime }},
		{"missing method", func(s *ClockTransformSpec) { s.Method = "" }},
		{"missing verification", func(s *ClockTransformSpec) { s.VerificationRef = "" }},
		{"expired anchor", func(s *ClockTransformSpec) { s.ValidUntilNS = 9 }},
		{"negative validity", func(s *ClockTransformSpec) { s.ValidFromNS = -1 }},
		{"overflow", func(s *ClockTransformSpec) {
			s.Anchors = []ClockAnchor{{0, math.MaxInt64, math.MaxInt64}}
			s.ValidUntilNS = 1
			s.UncertaintyNS = 0
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := clockSpec()
			tc.change(&s)
			if _, err := NewClockTransform(s); !errors.Is(err, ErrIncompatibleClocks) {
				t.Fatalf("invalid calibration: %v", err)
			}
		})
	}
	spec := clockSpec()
	transform, err := NewClockTransform(spec)
	if err != nil {
		t.Fatal(err)
	}
	for _, ts := range []BoundedTimestamp{sourceTime(spec, 21, 0), sourceTime(spec, 20, 1), sourceTime(spec, 0, 1), sourceTime(spec, 10, -1), {Nanoseconds: 10, Domain: ClockDomainGPUDevice, Epoch: spec.Source.ProcessEpoch, BootEpoch: spec.Source.BootEpoch}, {Nanoseconds: 10, Domain: spec.Source.Domain, Epoch: "new-process", BootEpoch: spec.Source.BootEpoch}, {Nanoseconds: 10, Domain: spec.Source.Domain, Epoch: spec.Source.ProcessEpoch, BootEpoch: "new-boot"}} {
		if _, err := ts.Aligned(transform.Mapping()); !errors.Is(err, ErrIncompatibleClocks) {
			t.Fatalf("invalid timestamp accepted: %+v %v", ts, err)
		}
	}
	m := transform.Mapping()
	m.UncertaintyNS = 1
	if _, err := sourceTime(spec, 10, 0).Aligned(m); !errors.Is(err, ErrIncompatibleClocks) {
		t.Fatal("mutated mapping admitted")
	}
}

func TestClockTransformComposition(t *testing.T) {
	s := clockSpec()
	s.Anchors = []ClockAnchor{{0, 100, 100}}
	s.ValidUntilNS = 100
	s.RateMinPPB = 2_000_000_000
	s.RateMaxPPB = s.RateMinPPB
	s.UncertaintyNS = 3
	first, err := NewClockTransform(s)
	if err != nil {
		t.Fatal(err)
	}
	secondSpec := clockSpec()
	secondSpec.Source = s.Reference
	secondSpec.Reference = ClockIdentity{Host: "final", Domain: ClockDomainHostRealtime, BootEpoch: "final-boot", ProcessEpoch: "final-process"}
	secondSpec.WallStepKnown = true
	secondSpec.Anchors = []ClockAnchor{{100, 1000, 1000}}
	secondSpec.ValidFromNS = 100
	secondSpec.ValidUntilNS = 400
	secondSpec.RateMinPPB = 3_000_000_000
	secondSpec.RateMaxPPB = secondSpec.RateMinPPB
	secondSpec.UncertaintyNS = 5
	second, err := NewClockTransform(secondSpec)
	if err != nil {
		t.Fatal(err)
	}
	combined, err := ComposeClockTransforms(first, second, "constructed-composition.json")
	if err != nil {
		t.Fatal(err)
	}
	got, err := sourceTime(s, 10, 2).Aligned(combined.Mapping())
	if err != nil {
		t.Fatal(err)
	}
	if got.Nanoseconds != 1060 || got.UncertaintyNS != 26 || got.Epoch != "final-process" || got.BootEpoch != "final-boot" {
		t.Fatalf("composed interval: %+v", got)
	}
	if _, err := sourceTime(s, 0, 0).Aligned(combined.Mapping()); !errors.Is(err, ErrIncompatibleClocks) {
		t.Fatal("intermediate validity crossing accepted")
	}
	steps := combined.Steps()
	steps[0].Anchors[0].ReferenceMinNS = -999
	steps[1].Source.Host = "changed"
	s.Anchors[0].ReferenceMinNS = -999
	again, err := sourceTime(s, 10, 2).Aligned(combined.Mapping())
	if err != nil || again != got {
		t.Fatal("constructor or Steps exposed mutable model")
	}
	for _, change := range []func(*ClockTransformSpec){func(s *ClockTransformSpec) { s.Source.Host = "wrong" }, func(s *ClockTransformSpec) { s.Source.BootEpoch = "wrong" }, func(s *ClockTransformSpec) { s.Source.ProcessEpoch = "wrong" }, func(s *ClockTransformSpec) { s.Source.Domain = ClockDomainGPUDevice }} {
		bad := secondSpec
		change(&bad)
		other, err := NewClockTransform(bad)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := ComposeClockTransforms(first, other, "composition"); !errors.Is(err, ErrIncompatibleClocks) {
			t.Fatal("intermediate identity mismatch accepted")
		}
	}
	if _, err := ComposeClockTransforms(first, second, ""); !errors.Is(err, ErrIncompatibleClocks) {
		t.Fatal("missing composition verification accepted")
	}
}

func TestClockTransformCompositionLimit(t *testing.T) {
	s := clockSpec()
	s.Reference = s.Source
	s.ValidUntilNS = 1
	s.Anchors = []ClockAnchor{{0, 0, 0}}
	s.UncertaintyNS = 0
	identity, err := NewClockTransform(s)
	if err != nil {
		t.Fatal(err)
	}
	combined := identity
	for i := 1; i < 8; i++ {
		combined, err = ComposeClockTransforms(combined, identity, "constructed-composition")
		if err != nil {
			t.Fatal(err)
		}
	}
	if len(combined.Steps()) != 8 {
		t.Fatal("bounded composition lost steps")
	}
	if _, err := ComposeClockTransforms(combined, identity, "constructed-composition"); !errors.Is(err, ErrIncompatibleClocks) {
		t.Fatal("unbounded composition admitted")
	}
}

func TestClockTransformIntegerPrecision(t *testing.T) {
	s := clockSpec()
	s.ValidUntilNS = math.MaxInt64
	s.Anchors = []ClockAnchor{{0, 0, 0}}
	s.UncertaintyNS = 0
	transform, err := NewClockTransform(s)
	if err != nil {
		t.Fatal(err)
	}
	for _, n := range []int64{0, 1<<53 + 1, math.MaxInt64} {
		got, err := sourceTime(s, n, 0).Aligned(transform.Mapping())
		if err != nil || got.Nanoseconds != n || got.UncertaintyNS != 0 {
			t.Fatalf("integer clock rounded: n=%d got=%+v err=%v", n, got, err)
		}
	}
	s.RateMinPPB = math.MaxInt64
	s.RateMaxPPB = math.MaxInt64
	if _, err := NewClockTransform(s); !errors.Is(err, ErrIncompatibleClocks) {
		t.Fatal("wide multiplication overflow admitted")
	}
	s = clockSpec()
	s.RateMinPPB = 500_000_000
	s.RateMaxPPB = s.RateMinPPB
	s.UncertaintyNS = 0
	s.Anchors = []ClockAnchor{{10, 100, 100}}
	transform, err = NewClockTransform(s)
	if err != nil {
		t.Fatal(err)
	}
	for _, n := range []int64{9, 11} {
		got, err := sourceTime(s, n, 0).Aligned(transform.Mapping())
		if err != nil || got.UncertaintyNS != 1 {
			t.Fatalf("fractional rounding narrowed interval: %+v %v", got, err)
		}
	}
}

func TestClockTransformConcurrentMapping(t *testing.T) {
	s := clockSpec()
	transform, err := NewClockTransform(s)
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 25; j++ {
				got, err := sourceTime(s, 11, 1).Aligned(transform.Mapping())
				if err != nil || got.Nanoseconds != 102 || got.UncertaintyNS != 5 {
					t.Errorf("concurrent mapping: %+v %v", got, err)
					return
				}
				copy := transform.Steps()
				copy[0].Anchors[0].ReferenceMinNS = -1
			}
		}()
	}
	wg.Wait()
}
