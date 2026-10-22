package dsv4

import (
	"errors"
	"fmt"
	"math"
)

// DecodeConsistencyReport describes a live decode compared with fresh
// re-prefills of the same prefix. Values are accumulated over every logit
// element at every compared step.
type DecodeConsistencyReport struct {
	Steps          int
	Compared       int
	Top1Mismatches int
	MaxAbs         float64
	RMS            float64
}

// Pass reports whether the report satisfies finite, non-negative tolerances.
// A report with no compared steps never passes.
func (r DecodeConsistencyReport) Pass(maxAbs, maxRMS float64) bool {
	return r.Steps > 0 && r.Steps == r.Compared &&
		finiteNonNegative(maxAbs) && finiteNonNegative(maxRMS) &&
		r.Top1Mismatches == 0 && r.MaxAbs <= maxAbs && r.RMS <= maxRMS
}

// CheckDecodeConsistency compares one live session with a fresh session after
// each generated token. It exercises prefix reuse: the live session accepts
// each token incrementally, while the comparison session is newly created and
// evaluates prompt plus the complete generated prefix. The function returns a
// report even when the values disagree; callers choose tolerances with Pass.
//
// generated must not contain a stop token, because a stopped token has no
// resulting logits in Session.Accept. The backend is evaluated sequentially;
// it must provide independent State values from NewState.
func CheckDecodeConsistency(backend Backend, context int, stop StopSet, prompt, generated []int) (DecodeConsistencyReport, error) {
	if backend == nil {
		return DecodeConsistencyReport{}, errors.New("decode consistency: missing backend")
	}
	if len(prompt) == 0 {
		return DecodeConsistencyReport{}, errors.New("decode consistency: empty prompt")
	}
	if len(generated) == 0 {
		return DecodeConsistencyReport{}, errors.New("decode consistency: empty generation")
	}
	if context <= len(prompt)+len(generated) {
		return DecodeConsistencyReport{}, fmt.Errorf("decode consistency: context %d cannot hold %d tokens", context, len(prompt)+len(generated))
	}
	if len(stop) == 0 {
		return DecodeConsistencyReport{}, errors.New("decode consistency: empty stop-token set")
	}

	live, err := NewSession(backend, context, stop)
	if err != nil {
		return DecodeConsistencyReport{}, fmt.Errorf("decode consistency: live session: %w", err)
	}
	defer live.Close()
	if err := live.Sync(prompt); err != nil {
		return DecodeConsistencyReport{}, fmt.Errorf("decode consistency: live prompt: %w", err)
	}

	var report DecodeConsistencyReport
	comparedValues := 0
	full := append([]int(nil), prompt...)
	for step, token := range generated {
		if stop.contains(token) {
			return report, fmt.Errorf("decode consistency: generated token %d is a stop token", step)
		}
		stopped, err := live.Accept(token)
		if err != nil {
			return report, fmt.Errorf("decode consistency: live step %d: %w", step, err)
		}
		if stopped {
			return report, fmt.Errorf("decode consistency: live stopped at step %d", step)
		}

		full = append(full, token)
		fresh, err := NewSession(backend, context, stop)
		if err != nil {
			return report, fmt.Errorf("decode consistency: fresh step %d: %w", step, err)
		}
		if err := fresh.Sync(full); err != nil {
			fresh.Close()
			return report, fmt.Errorf("decode consistency: fresh step %d: %w", step, err)
		}
		liveLogits, freshLogits := live.Logits(), fresh.Logits()
		fresh.Close()

		report.Steps++
		if len(liveLogits) != len(freshLogits) || len(liveLogits) == 0 {
			return report, fmt.Errorf("decode consistency: step %d logits have lengths %d and %d", step, len(liveLogits), len(freshLogits))
		}
		report.Compared++
		if argmax(liveLogits) != argmax(freshLogits) {
			report.Top1Mismatches++
		}
		for i := range liveLogits {
			if !finiteFloat32(liveLogits[i]) || !finiteFloat32(freshLogits[i]) {
				return report, fmt.Errorf("decode consistency: step %d logit %d is non-finite", step, i)
			}
			delta := math.Abs(float64(liveLogits[i]) - float64(freshLogits[i]))
			if delta > report.MaxAbs {
				report.MaxAbs = delta
			}
			report.RMS += delta * delta
			comparedValues++
		}
	}
	report.RMS = math.Sqrt(report.RMS / float64(comparedValues))
	return report, nil
}

func finiteFloat32(v float32) bool { return !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0) }

func finiteNonNegative(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) && v >= 0 }
