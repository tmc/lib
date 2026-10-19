package ncclinspector

import (
	"fmt"
	"math/big"
	"strings"
)

// ClockIdentity identifies a timing source, including its host and restart
// generations. These are caller-supplied facts, not inferred from timestamps.
// PID binds a process source when positive; zero is allowed for a generic clock
// provider. Inspector records require a positive matching source PID. PID alone
// does not establish a restart generation. ProcessEpoch can identify a reference
// provider's generation rather than a PID.
type ClockIdentity struct {
	Host         string
	PID          int
	Domain       ClockDomain
	BootEpoch    string
	ProcessEpoch string
}

// ClockAnchor bounds a reference-clock reading at an exact source-clock reading.
// Bounds are inclusive and can be asymmetric; all readings are nanoseconds.
type ClockAnchor struct {
	SourceNS       int64
	ReferenceMinNS int64
	ReferenceMaxNS int64
}

// ClockTransformSpec declares a bounded affine calibration. SourceUnit and
// ReferenceUnit must be "nanoseconds": adapters normalize producer units before
// mapping, retaining the original integers in their source records. RateMinPPB
// and RateMaxPPB bound reference nanoseconds per source nanosecond, multiplied
// by 1e9 (1e9 is unit rate). Anchors must admit one common affine model.
// DriftPPB is additional uncertainty growth per source
// nanosecond from an anchor. UncertaintyNS bounds additional residual error.
// Explicit known flags distinguish zero error from unavailable calibration.
// VerificationRef names independent verification evidence; its presence does
// not authenticate that evidence or establish physical synchronization.
type ClockTransformSpec struct {
	Source           ClockIdentity
	Reference        ClockIdentity
	SourceUnit       string
	ReferenceUnit    string
	Anchors          []ClockAnchor
	RateMinPPB       int64
	RateMaxPPB       int64
	DriftPPB         int64
	DriftKnown       bool
	UncertaintyNS    int64
	UncertaintyKnown bool
	ValidFromNS      int64
	ValidUntilNS     int64
	WallStepKnown    bool
	WallStepObserved bool
	Method           string
	VerificationRef  string
}

// ClockTransform is an immutable bounded transform, possibly composed of up to
// eight steps. Its zero value is unusable. Construct one with NewClockTransform.
// It models supplied calibrations; it does not measure or acquire calibration.
type ClockTransform struct {
	steps       []ClockTransformSpec
	evidenceRef string
}

// NewClockTransform validates and copies a supplied calibration. It accepts one
// to sixteen ordered anchors. Every query must remain inside declared validity;
// extrapolation within that policy propagates rate and drift bounds. A wall
// clock requires explicit no-step evidence for that interval. An observed step
// requires a separately calibrated interval and is refused here.
func NewClockTransform(spec ClockTransformSpec) (*ClockTransform, error) {
	bad := func(reason string) (*ClockTransform, error) { return nil, clockTransformError(reason) }
	if !validClockIdentity(spec.Source) || !validClockIdentity(spec.Reference) {
		return bad("missing clock identity")
	}
	if spec.SourceUnit != "nanoseconds" || spec.ReferenceUnit != "nanoseconds" {
		return bad("clock transform units must be nanoseconds")
	}
	if len(spec.Anchors) == 0 || len(spec.Anchors) > 16 {
		return bad("clock transform requires one to sixteen anchors")
	}
	if spec.RateMinPPB <= 0 || spec.RateMaxPPB < spec.RateMinPPB || !spec.DriftKnown || spec.DriftPPB < 0 {
		return bad("missing or invalid rate/drift bounds")
	}
	if !spec.UncertaintyKnown || spec.UncertaintyNS < 0 {
		return bad("missing or invalid residual uncertainty")
	}
	if spec.ValidFromNS < 0 || spec.ValidUntilNS < spec.ValidFromNS {
		return bad("invalid clock transform validity")
	}
	if spec.WallStepObserved || (spec.Source.Domain == ClockDomainHostRealtime || spec.Reference.Domain == ClockDomainHostRealtime) && !spec.WallStepKnown {
		return bad("wall step observed or status unavailable")
	}
	if !clockText(spec.Method) || !clockText(spec.VerificationRef) {
		return bad("missing method or verification artifact")
	}
	spec.Anchors = append([]ClockAnchor(nil), spec.Anchors...)
	for i, a := range spec.Anchors {
		if a.SourceNS < spec.ValidFromNS || a.SourceNS > spec.ValidUntilNS || a.ReferenceMinNS > a.ReferenceMaxNS || i > 0 && a.SourceNS <= spec.Anchors[i-1].SourceNS {
			return bad("invalid or unordered clock anchors")
		}
	}
	if !affineClockAnchors(spec) {
		return bad("anchors do not admit a common affine clock model")
	}
	// Check model consistency at every anchor and validity endpoint. These checks
	// do not replace independent calibration verification named by the caller.
	points := []int64{spec.ValidFromNS, spec.ValidUntilNS}
	for _, a := range spec.Anchors {
		points = append(points, a.SourceNS)
	}
	for _, n := range points {
		if _, err := mapClockStep(BoundedTimestamp{Nanoseconds: n, Domain: spec.Source.Domain, Epoch: spec.Source.ProcessEpoch, BootEpoch: spec.Source.BootEpoch}, spec); err != nil {
			return nil, err
		}
	}
	return &ClockTransform{steps: []ClockTransformSpec{spec}, evidenceRef: spec.VerificationRef}, nil
}

// ComposeClockTransforms maps first, then second, retaining both calibrations
// and propagating their intervals conservatively. Their intermediate identities
// must match exactly. evidenceRef names the caller's composition verification.
// Query validity is checked at every step; composition does not extend it.
func ComposeClockTransforms(first, second *ClockTransform, evidenceRef string) (*ClockTransform, error) {
	if first == nil || second == nil || len(first.steps) == 0 || len(second.steps) == 0 || len(first.steps)+len(second.steps) > 8 || !clockText(evidenceRef) {
		return nil, clockTransformError("missing or excessive clock composition")
	}
	if first.steps[len(first.steps)-1].Reference != second.steps[0].Source {
		return nil, clockTransformError("clock composition identity mismatch")
	}
	steps := append(first.Steps(), second.Steps()...)
	return &ClockTransform{steps: steps, evidenceRef: evidenceRef}, nil
}

// Steps returns independent copies of the constituent calibration records.
func (t *ClockTransform) Steps() []ClockTransformSpec {
	if t == nil {
		return nil
	}
	steps := append([]ClockTransformSpec(nil), t.steps...)
	for i := range steps {
		steps[i].Anchors = append([]ClockAnchor(nil), steps[i].Anchors...)
	}
	return steps
}

// Mapping adapts t to ObservationFromRecord and BoundedTimestamp.Aligned. The
// returned legacy constant UncertaintyNS is zero: Transform carries the actual
// query-dependent bounds. Mutating other mapping fields makes it invalid.
// A nil or zero transform returns an unusable zero mapping.
func (t *ClockTransform) Mapping() ClockMapping {
	if t == nil || len(t.steps) == 0 {
		return ClockMapping{}
	}
	s, r := t.steps[0], t.steps[len(t.steps)-1]
	return ClockMapping{Domain: s.Source.Domain, SourceBootEpoch: s.Source.BootEpoch, SourceProcessEpoch: s.Source.ProcessEpoch, CalibrationKnown: true, UncertaintyKnown: true, ValidityKnown: true, ValidFromNS: s.ValidFromNS, ValidUntilNS: s.ValidUntilNS, ReferenceDomain: r.Reference.Domain, ReferenceEpoch: r.Reference.ProcessEpoch, EvidenceRef: t.evidenceRef, Transform: t}
}

func validClockIdentity(id ClockIdentity) bool {
	return id.PID >= 0 && clockText(id.Host) && clockText(string(id.Domain)) && clockText(id.BootEpoch) && clockText(id.ProcessEpoch)
}
func clockText(s string) bool { return len(s) <= 4096 && strings.TrimSpace(s) != "" }
func clockTransformError(reason string) error {
	return fmt.Errorf("%w: %s", ErrIncompatibleClocks, reason)
}

func (t *ClockTransform) aligned(bt BoundedTimestamp, mapping ClockMapping) (BoundedTimestamp, error) {
	if t == nil || len(t.steps) == 0 || mapping != t.Mapping() {
		return BoundedTimestamp{}, clockTransformError("transform mapping metadata changed")
	}
	for _, step := range t.steps {
		var err error
		bt, err = mapClockStep(bt, step)
		if err != nil {
			return BoundedTimestamp{}, err
		}
	}
	return bt, nil
}

func mapClockStep(bt BoundedTimestamp, spec ClockTransformSpec) (BoundedTimestamp, error) {
	if bt.Domain != spec.Source.Domain || bt.Epoch != spec.Source.ProcessEpoch || bt.BootEpoch != spec.Source.BootEpoch {
		return BoundedTimestamp{}, clockTransformError("clock transform source epoch or domain mismatch")
	}
	if bt.UncertaintyNS < 0 || bt.Nanoseconds < spec.ValidFromNS || bt.Nanoseconds > spec.ValidUntilNS || bt.UncertaintyNS > bt.Nanoseconds-spec.ValidFromNS || bt.UncertaintyNS > spec.ValidUntilNS-bt.Nanoseconds {
		return BoundedTimestamp{}, clockTransformError("timestamp interval outside transform validity")
	}
	lo, hi := bt.Nanoseconds-bt.UncertaintyNS, bt.Nanoseconds+bt.UncertaintyNS
	var lower, upper *big.Int
	for _, a := range spec.Anchors {
		dl, dh := lo-a.SourceNS, hi-a.SourceNS // all operands are nonnegative int64 readings
		minRate, maxRate := spec.RateMinPPB, spec.RateMaxPPB
		if dl < 0 {
			minRate = spec.RateMaxPPB
		}
		if dh < 0 {
			maxRate = spec.RateMinPPB
		}
		l := scaledClockDelta(dl, minRate, false)
		h := scaledClockDelta(dh, maxRate, true)
		l.Add(l, big.NewInt(a.ReferenceMinNS))
		h.Add(h, big.NewInt(a.ReferenceMaxNS))
		distance := dl
		if distance < 0 {
			distance = -distance
		}
		other := dh
		if other < 0 {
			other = -other
		}
		if other > distance {
			distance = other
		}
		growth := scaledClockDelta(distance, spec.DriftPPB, true)
		growth.Add(growth, big.NewInt(spec.UncertaintyNS))
		l.Sub(l, growth)
		h.Add(h, growth)
		if lower == nil || l.Cmp(lower) > 0 {
			lower = l
		}
		if upper == nil || h.Cmp(upper) < 0 {
			upper = h
		}
	}
	if lower == nil || lower.Cmp(upper) > 0 {
		return BoundedTimestamp{}, clockTransformError("clock anchor bounds conflict")
	}
	// Convert the possibly asymmetric interval to an enclosing symmetric one.
	// Quo's toward-zero rounding is safe because radius covers both endpoints.
	center := new(big.Int).Add(lower, upper)
	center.Quo(center, big.NewInt(2))
	radius := new(big.Int).Sub(center, lower)
	right := new(big.Int).Sub(upper, center)
	if right.Cmp(radius) > 0 {
		radius = right
	}
	if !center.IsInt64() || !radius.IsInt64() || !lower.IsInt64() || !upper.IsInt64() {
		return BoundedTimestamp{}, clockTransformError("clock transform overflow")
	}
	// The symmetric representation itself must also fit, not only the original bounds.
	if !new(big.Int).Sub(center, radius).IsInt64() || !new(big.Int).Add(center, radius).IsInt64() {
		return BoundedTimestamp{}, clockTransformError("clock transform interval overflow")
	}
	return BoundedTimestamp{Nanoseconds: center.Int64(), UncertaintyNS: radius.Int64(), Domain: spec.Reference.Domain, Epoch: spec.Reference.ProcessEpoch, BootEpoch: spec.Reference.BootEpoch}, nil
}

// Eliminate the intercept from L_i <= offset + rate*x_i <= U_i.
// For every ordered pair, the common rate must lie between
// (L_j-U_i)/(x_j-x_i) and (U_j-L_i)/(x_j-x_i). Their intersection
// with the supplied rate interval must be nonempty. Pairwise range checks
// alone would accept three exact points that admit no single affine model.
func affineClockAnchors(spec ClockTransformSpec) bool {
	lower := new(big.Rat).SetFrac(big.NewInt(spec.RateMinPPB), big.NewInt(1_000_000_000))
	upper := new(big.Rat).SetFrac(big.NewInt(spec.RateMaxPPB), big.NewInt(1_000_000_000))
	for i, a := range spec.Anchors {
		for _, b := range spec.Anchors[i+1:] {
			denominator := big.NewInt(b.SourceNS - a.SourceNS)
			lo := new(big.Rat).SetFrac(new(big.Int).Sub(big.NewInt(b.ReferenceMinNS), big.NewInt(a.ReferenceMaxNS)), denominator)
			hi := new(big.Rat).SetFrac(new(big.Int).Sub(big.NewInt(b.ReferenceMaxNS), big.NewInt(a.ReferenceMinNS)), denominator)
			if lo.Cmp(lower) > 0 {
				lower = lo
			}
			if hi.Cmp(upper) < 0 {
				upper = hi
			}
			if lower.Cmp(upper) > 0 {
				return false
			}
		}
	}
	return true
}

// scaledClockDelta computes floor/ceil(delta*rate/1e9), including negative deltas.
func scaledClockDelta(delta, rate int64, up bool) *big.Int {
	n := new(big.Int).Mul(big.NewInt(delta), big.NewInt(rate))
	q, r := new(big.Int), new(big.Int)
	q.QuoRem(n, big.NewInt(1_000_000_000), r)
	if up && r.Sign() > 0 {
		q.Add(q, big.NewInt(1))
	}
	if !up && r.Sign() < 0 {
		q.Sub(q, big.NewInt(1))
	}
	return q
}
