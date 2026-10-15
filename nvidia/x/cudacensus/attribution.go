package cudacensus

import (
	"fmt"
	"math"
	"sort"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaroofline"
)

// Method says how a precision attribution was decided. It travels with every
// Attribution because the two methods are not equally good and a consumer that
// cannot tell them apart cannot tell a static fact from a thresholded guess.
type Method string

const (
	// FromInstructionMix decided the precision by counting the tensor
	// instructions in the kernel's own disassembly. The count is a static fact
	// about the artifact that ran.
	FromInstructionMix Method = "instruction_mix"

	// FromCounterRatio decided the precision by comparing per-format pipe
	// activity against a threshold. The threshold is a judgement, and it is
	// only as good as the calibration receipt it carries.
	FromCounterRatio Method = "counter_ratio"

	// Unattributed decided nothing. The evidence was absent or it was
	// ambiguous, and the Reason says which.
	Unattributed Method = "unattributed"
)

// InstructionMix counts a kernel's tensor and math instructions by the
// precision they operate on, as read from the kernel's disassembly.
//
// This package does not produce a mix and does not import cudasass: a count is
// a number, and an adapter that turns a CUBIN into one belongs with the
// disassembler, not with the counter reader. Build the map from cudasass's
// instruction counts and pass it here.
type InstructionMix struct {
	Counts map[cudaroofline.Precision]uint64
}

// Total returns the sum of the instruction counts modulo 2^64.
// It cannot report overflow; use Dominant or AttributeFromMix when a validated
// total is needed for attribution.
func (m InstructionMix) Total() uint64 {
	var n uint64
	for _, c := range m.Counts {
		n += c
	}
	return n
}

// Dominant returns the precision holding the largest share of the mix, and that
// share in (0, 1]. It reports an error if the total overflows uint64, the mix
// is empty, or the largest counts tie. A tie must not be resolved by map
// iteration order.
func (m InstructionMix) Dominant() (cudaroofline.Precision, float64, error) {
	var total uint64
	for _, c := range m.Counts {
		if c > math.MaxUint64-total {
			return "", 0, fmt.Errorf("cudacensus: instruction mix total overflows uint64")
		}
		total += c
	}
	if total == 0 {
		return "", 0, fmt.Errorf("cudacensus: empty instruction mix")
	}
	keys := make([]string, 0, len(m.Counts))
	for p := range m.Counts {
		keys = append(keys, string(p))
	}
	sort.Strings(keys)

	var best cudaroofline.Precision
	var bestCount uint64
	var tied bool
	for _, k := range keys {
		p := cudaroofline.Precision(k)
		switch c := m.Counts[p]; {
		case c > bestCount:
			best, bestCount, tied = p, c, false
		case c == bestCount && c > 0:
			tied = true
		}
	}
	if tied {
		return "", 0, fmt.Errorf("cudacensus: instruction mix has no dominant precision: %d instructions tie at %d", len(m.Counts), bestCount)
	}
	return best, float64(bestCount) / float64(total), nil
}

// CalibrationReceipt records the workload a threshold was calibrated against
// and what that calibration showed. A threshold without one is a constant
// somebody liked, and this package refuses to apply it.
//
// FailsWhen is required and may not be empty. A calibration that found no
// failure mode did not look, and the honest content of the field is the case
// the calibration did not cover.
type CalibrationReceipt struct {
	Workload   string    // the named workload, reproducible by a reader
	Device     string    // the identity the calibration ran on
	Separates  string    // what the threshold distinguishes, e.g. "fp16_tensor from fp8_tensor"
	FailsWhen  string    // where it is known or suspected to fail
	Samples    int       // how many samples the separation rests on
	Error      float64   // relative error observed against the analytic value
	RecordedAt time.Time // when the calibration ran
}

// Validate reports whether the receipt records enough to be checked by someone
// who was not there.
func (r CalibrationReceipt) Validate() error {
	switch {
	case r.Workload == "":
		return fmt.Errorf("cudacensus: calibration receipt has no workload")
	case r.Device == "":
		return fmt.Errorf("cudacensus: calibration receipt for %q has no device identity", r.Workload)
	case r.Separates == "":
		return fmt.Errorf("cudacensus: calibration receipt for %q does not say what the threshold separates", r.Workload)
	case r.FailsWhen == "":
		return fmt.Errorf("cudacensus: calibration receipt for %q does not say where the threshold fails", r.Workload)
	case r.Samples <= 0:
		return fmt.Errorf("cudacensus: calibration receipt for %q rests on %d samples", r.Workload, r.Samples)
	case r.Error < 0 || math.IsNaN(r.Error) || math.IsInf(r.Error, 0):
		return fmt.Errorf("cudacensus: calibration receipt for %q reports an invalid error %v", r.Workload, r.Error)
	case r.RecordedAt.IsZero():
		return fmt.Errorf("cudacensus: calibration receipt for %q has no timestamp", r.Workload)
	}
	return nil
}

// Threshold is a ratio constant that cannot exist without its receipt. The
// zero Threshold is unusable and AttributeFromCounters refuses it, which is the
// mechanism enforcing that no threshold enters this package uncalibrated.
type Threshold struct {
	value   float64
	receipt CalibrationReceipt
}

// NewThreshold returns a threshold carrying the calibration that justifies it.
// It refuses a value outside (0, 1], since the value is a share of tensor-pipe
// activity, and it refuses a receipt that does not validate.
func NewThreshold(value float64, r CalibrationReceipt) (Threshold, error) {
	if value <= 0 || value > 1 || math.IsNaN(value) {
		return Threshold{}, fmt.Errorf("cudacensus: threshold %v is not a share in (0, 1]", value)
	}
	if err := r.Validate(); err != nil {
		return Threshold{}, err
	}
	return Threshold{value: value, receipt: r}, nil
}

// Value returns the threshold and whether one was set. A zero Threshold reports
// false, and callers must not read its value as zero.
func (t Threshold) Value() (float64, bool) {
	return t.value, t.value > 0
}

// Receipt returns the calibration behind the threshold.
func (t Threshold) Receipt() CalibrationReceipt { return t.receipt }

// Attribution is a decision about which precision the math pipes were running.
// It is not a measurement: the counters say a pipe was active, and the FLOPs per
// cycle differ by format, so every FLOPs figure downstream is the product of a
// measurement and this decision.
type Attribution struct {
	Precision cudaroofline.Precision // the attributed format; empty when Known is false
	Method    Method                 // how it was decided
	Share     float64                // the share of the evidence supporting it
	Known     bool                   // false when nothing was attributed

	// Threshold and Receipt are set only for FromCounterRatio, and the receipt
	// travels so a consumer can see what the threshold was calibrated against
	// without holding the Threshold value.
	Threshold float64
	Receipt   CalibrationReceipt

	// Reason says why Known is false, or which ambiguity a known attribution
	// had to resolve.
	Reason string
}

// AttributeFromMix attributes the precision from a kernel's static instruction
// mix. This is the preferred method: the mix is a property of the artifact that
// ran, and no threshold is involved.
func AttributeFromMix(mix InstructionMix) (Attribution, error) {
	p, share, err := mix.Dominant()
	if err != nil {
		return Attribution{
			Method: Unattributed,
			Reason: err.Error(),
		}, err
	}
	return Attribution{
		Precision: p,
		Method:    FromInstructionMix,
		Share:     share,
		Known:     true,
	}, nil
}

// tensorFormats maps each per-format tensor counter to the precision its
// activity implies. AnyTensorActive is deliberately absent: it says the pipes
// were busy and not what they were busy with, which is the ambiguity this
// function exists to report rather than resolve.
var tensorFormats = map[Counter]cudaroofline.Precision{
	HMMATensorActive: cudaroofline.FP16Tensor,
	IMMATensorActive: cudaroofline.INT8Tensor,
	DMMATensorActive: cudaroofline.FP64,
}

// AttributeFromCounters attributes the precision from per-format pipe activity,
// applying t to the largest format's share of the total format activity. It is
// the fallback for a running kernel whose CUBIN is not available.
//
// It returns an unattributed result, and no error, in the two cases that are
// ordinary rather than exceptional: when the device reports only
// AnyTensorActive, which names no format, and when the dominant format's share
// falls below the threshold, which is what a mixed-precision kernel looks like.
// Both are the negative controls for this method, and a caller that treats them
// as failures will hide them.
func AttributeFromCounters(readings []Reading, t Threshold) (Attribution, error) {
	cut, ok := t.Value()
	if !ok {
		return Attribution{Method: Unattributed, Reason: "no calibrated threshold"},
			fmt.Errorf("cudacensus: counter-ratio attribution requires a threshold with a calibration receipt")
	}

	var total float64
	var best Counter
	var bestValue float64
	var anyTensor Reading
	var haveAny bool
	for _, r := range readings {
		if r.Counter == AnyTensorActive {
			anyTensor, haveAny = r, true
			continue
		}
		if _, ok := tensorFormats[r.Counter]; !ok {
			continue
		}
		v, ok := r.Get()
		if !ok {
			continue
		}
		total += v
		if v > bestValue {
			best, bestValue = r.Counter, v
		}
	}

	if total == 0 {
		reason := "no per-format tensor counter is available"
		if haveAny {
			if v, ok := anyTensor.Get(); ok && v > 0 {
				reason = fmt.Sprintf("only %s is available (%.4f): it reports that the tensor pipes were busy, not what they were busy with", AnyTensorActive, v)
			}
		}
		return Attribution{Method: Unattributed, Reason: reason, Threshold: cut, Receipt: t.Receipt()}, nil
	}

	share := bestValue / total
	if share < cut {
		return Attribution{
			Method:    Unattributed,
			Share:     share,
			Threshold: cut,
			Receipt:   t.Receipt(),
			Reason: fmt.Sprintf("%s holds %.4f of format activity, below the %.4f threshold: the pipes ran more than one format",
				best, share, cut),
		}, nil
	}
	return Attribution{
		Precision: tensorFormats[best],
		Method:    FromCounterRatio,
		Share:     share,
		Known:     true,
		Threshold: cut,
		Receipt:   t.Receipt(),
	}, nil
}
