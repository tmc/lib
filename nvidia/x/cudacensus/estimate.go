package cudacensus

import (
	"fmt"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaroofline"
)

// Confidence classes an estimate by what it rests on. The classes are ordered
// by how much of the figure is measurement and how much is assumption, and the
// class is part of the estimate because a consumer that cannot see the
// difference will average them.
type Confidence string

const (
	// ConfidenceIdle means the device reported zero activity on every counter
	// read. Zero achieved FLOPs is then a measurement, and it is the one
	// confident figure this package produces. It is not the same as an
	// unavailable counter.
	ConfidenceIdle Confidence = "idle"

	// ConfidenceStaticMix means the precision came from the kernel's own
	// instruction mix. The activity is measured and the FLOPs-per-cycle
	// multiplier rests on a static property of the artifact.
	ConfidenceStaticMix Confidence = "static_mix"

	// ConfidenceCounterRatio means the precision came from a thresholded ratio
	// of pipe activity. The threshold's calibration receipt bounds how much
	// this class is worth, and the calibration may refute it.
	ConfidenceCounterRatio Confidence = "counter_ratio"

	// ConfidenceUnavailable means no estimate was produced. The Derivation says
	// why, and the FLOPs field is not a number to read.
	ConfidenceUnavailable Confidence = "unavailable"
)

// Derivation records everything an estimate rests on: the counters actually read
// with their absences intact, the identity and peak entry the ceiling came from,
// the precision attribution and how it was decided, and the sampling interval.
//
// It exists so that a FLOPs figure can be argued with. An estimate whose
// derivation is missing is a number with no error mode, which is the thing this
// package is built not to emit.
type Derivation struct {
	Readings []Reading // the counters read, including the ones reported absent

	Architecture string                 // from the step 1 identity
	Residency    cudaroofline.Residency // memory path the bandwidth figure describes

	PeakTFLOPS        float64                 // the peak the activity was scaled by
	PeakProvenance    cudaroofline.Provenance // how that peak was obtained
	PeakMemoryGBps    float64                 // the bandwidth ceiling the byte rate was scaled by
	MemoryProvenance  cudaroofline.Provenance // how that bandwidth was obtained
	Attribution       Attribution             // which precision, and how decided
	Interval          time.Duration           // the sampling interval the activity covers
	Note              string                  // qualifications a consumer must carry forward
	UnavailableReason string                  // why no estimate was produced
}

// Estimate is an achieved-FLOPs figure with its derivation. It is an estimate
// and not a measurement: the counters report that a pipe was active, the FLOPs
// per cycle differ by format, and the format came from Derivation.Attribution.
//
// Read Available before reading any number. When Available is false the fields
// hold no meaning, and when Rejected is true the estimator produced a figure the
// hardware says is impossible, which is a bug in the estimator reported with its
// inputs rather than a discovery.
type Estimate struct {
	AchievedTFLOPS      float64 // estimated achieved throughput
	ArithmeticIntensity float64 // estimated FLOPs per byte of DRAM traffic
	CeilingTFLOPS       float64 // the roofline at that intensity, for comparison

	Available  bool
	Confidence Confidence
	Derivation Derivation

	Rejected     bool   // the estimate exceeded the ceiling and is not reported as a result
	RejectReason string // what exceeded what, with the inputs
}

// Inputs are one sampling tick's evidence.
type Inputs struct {
	Readings    []Reading
	Attribution Attribution
	Interval    time.Duration
}

// NewEstimate estimates achieved FLOPs and arithmetic intensity from one tick of
// counters, bounded by m's ceiling.
//
// It returns an unavailable estimate, rather than an error, for every ordinary
// absence: an unattributed precision, a precision the peak table has no entry
// for, or a missing activity counter. Those are the states the device and the
// table actually produce, and an error would invite a caller to substitute a
// zero for them.
func NewEstimate(m *cudaroofline.Model, in Inputs) Estimate {
	d := Derivation{
		Readings:         in.Readings,
		Architecture:     m.Ceiling.Architecture,
		Residency:        m.Ceiling.Residency,
		PeakMemoryGBps:   m.Ceiling.PeakMemoryGBps,
		MemoryProvenance: m.Ceiling.MemoryProvenance,
		Attribution:      in.Attribution,
		Interval:         in.Interval,
	}

	if in.Interval <= 0 {
		d.UnavailableReason = "no sampling interval: a rate cannot be derived from a tick of unknown length, and host wall time is not a substitute"
		return Estimate{Confidence: ConfidenceUnavailable, Derivation: d}
	}

	// Idle is a real state, and it is decided before precision: a device that
	// answered every counter with zero has told us there were no FLOPs, whatever
	// format it would have run them in.
	switch idle, err := allZero(in.Readings); {
	case err != nil:
		d.UnavailableReason = err.Error()
		return Estimate{Confidence: ConfidenceUnavailable, Derivation: d}
	case idle:
		d.Note = "every counter read reported zero: the device was idle"
		return Estimate{
			AchievedTFLOPS: 0,
			Available:      true,
			Confidence:     ConfidenceIdle,
			Derivation:     d,
		}
	}

	if !in.Attribution.Known {
		d.UnavailableReason = fmt.Sprintf("precision unattributed: %s", in.Attribution.Reason)
		return Estimate{Confidence: ConfidenceUnavailable, Derivation: d}
	}

	peak, prov, err := cudaroofline.PeakTFLOPSFor(m.Ceiling.Architecture, in.Attribution.Precision)
	if err != nil {
		d.UnavailableReason = err.Error()
		return Estimate{Confidence: ConfidenceUnavailable, Derivation: d}
	}
	d.PeakTFLOPS, d.PeakProvenance = peak, prov

	activity, ok := activityFor(in.Readings, in.Attribution.Precision)
	if !ok {
		d.UnavailableReason = fmt.Sprintf("no activity counter available for %s", in.Attribution.Precision)
		return Estimate{Confidence: ConfidenceUnavailable, Derivation: d}
	}

	est := Estimate{
		AchievedTFLOPS: activity * peak,
		Available:      true,
		Confidence:     confidenceFor(in.Attribution.Method),
		Derivation:     d,
	}

	// The byte rate comes from a driver-theoretical utilization fraction scaled
	// by this repository's own bandwidth ceiling. Those two figures are not the
	// same measurement, so the intensity carries the mismatch as a note rather
	// than pretending the product is one number.
	if util, ok := readingValue(in.Readings, DRAMBandwidthUtil); ok && util > 0 && m.Ceiling.PeakMemoryGBps > 0 {
		gbps := util * m.Ceiling.PeakMemoryGBps
		est.ArithmeticIntensity = (est.AchievedTFLOPS * 1000.0) / gbps
		est.CeilingTFLOPS = m.CeilingAt(est.ArithmeticIntensity)
		est.Derivation.Note = fmt.Sprintf("intensity uses %s (%.4f), a fraction of the driver's theoretical bandwidth, scaled by this model's %s ceiling of %.1f GB/s: the two figures are not one measurement",
			DRAMBandwidthUtil, util, m.Ceiling.MemoryProvenance, m.Ceiling.PeakMemoryGBps)
	} else {
		est.Derivation.Note = fmt.Sprintf("%s unavailable: no arithmetic intensity, and the estimate is bounded by the compute plateau alone", DRAMBandwidthUtil)
		est.CeilingTFLOPS = m.Ceiling.PeakTFLOPS
	}

	if est.CeilingTFLOPS > 0 && est.AchievedTFLOPS > est.CeilingTFLOPS {
		est.Rejected = true
		est.Available = false
		est.RejectReason = fmt.Sprintf("estimated %.4f TFLOPS exceeds the %.4f TFLOPS ceiling at intensity %.4f on %s/%s: activity %.4f, peak %.4f (%s), precision %s by %s",
			est.AchievedTFLOPS, est.CeilingTFLOPS, est.ArithmeticIntensity,
			m.Ceiling.Architecture, m.Ceiling.Residency,
			activity, peak, prov, in.Attribution.Precision, in.Attribution.Method)
	}
	return est
}

// confidenceFor maps an attribution method to the confidence class an estimate
// resting on it can carry.
func confidenceFor(method Method) Confidence {
	switch method {
	case FromInstructionMix:
		return ConfidenceStaticMix
	case FromCounterRatio:
		return ConfidenceCounterRatio
	}
	return ConfidenceUnavailable
}

// precisionActivity names the counter whose activity scales the peak for a
// precision. A precision absent here has no activity counter, which is a
// different absence from a missing peak and is reported as such.
var precisionActivity = map[cudaroofline.Precision]Counter{
	cudaroofline.FP64:       FP64Active,
	cudaroofline.FP32:       FP32Active,
	cudaroofline.FP16:       FP16Active,
	cudaroofline.FP16Tensor: HMMATensorActive,
	cudaroofline.INT8Tensor: IMMATensorActive,
}

func activityFor(readings []Reading, p cudaroofline.Precision) (float64, bool) {
	c, ok := precisionActivity[p]
	if !ok {
		return 0, false
	}
	return readingValue(readings, c)
}

func readingValue(readings []Reading, c Counter) (float64, bool) {
	for _, r := range readings {
		if r.Counter == c {
			return r.Get()
		}
	}
	return 0, false
}

// allZero reports whether every available reading is zero. It refuses a set in
// which nothing at all was available, because a device that answered nothing is
// not an idle device and the difference is the point.
func allZero(readings []Reading) (bool, error) {
	var available int
	for _, r := range readings {
		v, ok := r.Get()
		if !ok {
			continue
		}
		available++
		if v != 0 {
			return false, nil
		}
	}
	if available == 0 {
		return false, fmt.Errorf("cudacensus: no counter was available: the device answered nothing, which is not an idle device")
	}
	return true, nil
}
