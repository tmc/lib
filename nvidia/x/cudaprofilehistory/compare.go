package cudaprofilehistory

import (
	"fmt"
	"math"
	"sort"

	"github.com/tmc/lib/nvidia/x/cudaevidence"
)

// Policy selects one non-additive timing channel, a fractional regression
// threshold, absolute noise in nanoseconds and a minimum observation count.
// Equality at the noise boundary is unknown. A zero baseline is unknown.
type Policy struct {
	Timing            string  `json:"timing"`
	RelativeThreshold float64 `json:"relative_threshold"`
	NoiseNanos        float64 `json:"noise_nanos"`
	MinSamples        int     `json:"min_samples"`
}

func (p Policy) validate() error {
	switch p.Timing {
	case "wall_elapsed", "summed_kernel", "gpu_busy", "api_call":
	default:
		return fmt.Errorf("history: unsupported timing channel")
	}
	if p.RelativeThreshold < 0 || p.NoiseNanos < 0 || p.MinSamples <= 0 || math.IsNaN(p.RelativeThreshold) || math.IsInf(p.RelativeThreshold, 0) || math.IsNaN(p.NoiseNanos) || math.IsInf(p.NoiseNanos, 0) {
		return fmt.Errorf("history: invalid comparison policy")
	}
	return nil
}

// Change retains both classified facts, including missing and unread values.
type Change struct {
	Name   string `json:"name"`
	Before *Fact  `json:"before"`
	After  *Fact  `json:"after"`
}

// Comparison separates code generation, resource facts, correctness and timing.
// Performance is Unknown unless both observations qualify under Policy.
type Comparison struct {
	BeforeKey         string      `json:"before_key"`
	AfterKey          string      `json:"after_key"`
	CodeChanged       bool        `json:"code_changed"`
	CompilerChanged   bool        `json:"compiler_changed"`
	Codegen           []Change    `json:"codegen"`
	Resources         []Change    `json:"resources"`
	Timings           []Change    `json:"timings"`
	BeforeCorrectness Correctness `json:"before_correctness"`
	AfterCorrectness  Correctness `json:"after_correctness"`
	Performance       Verdict     `json:"performance"`
	Reason            string      `json:"reason"`
	Policy            Policy      `json:"policy"`
}

// Compare reports independent axes without treating code changes as failures.
func Compare(a, b Record, p Policy) (Comparison, error) {
	out := Comparison{Performance: Unknown, BeforeCorrectness: a.Correctness, AfterCorrectness: b.Correctness, Policy: p}
	if err := p.validate(); err != nil {
		return out, err
	}
	var err error
	out.BeforeKey, err = ObservationKey(a)
	if err != nil {
		return out, err
	}
	out.AfterKey, err = ObservationKey(b)
	if err != nil {
		return out, err
	}
	out.CompilerChanged = a.CompilerKey != b.CompilerKey
	out.CodeChanged = (a.Artifact == nil) != (b.Artifact == nil) || (a.Artifact != nil && b.Artifact != nil && a.Artifact.Digest != b.Artifact.Digest)
	facts := func(r Record, axis string) map[string]Fact {
		m := map[string]Fact{}
		for _, f := range r.Facts {
			if f.Axis == axis {
				m[f.Name] = f
			}
		}
		return m
	}
	for _, axis := range []string{"codegen", "resource", "performance"} {
		left, right := facts(a, axis), facts(b, axis)
		names := map[string]bool{}
		for n := range left {
			names[n] = true
		}
		for n := range right {
			names[n] = true
		}
		sorted := make([]string, 0, len(names))
		for n := range names {
			sorted = append(sorted, n)
		}
		sort.Strings(sorted)
		for _, n := range sorted {
			c := Change{Name: n}
			if f, ok := left[n]; ok {
				x := f
				c.Before = &x
			}
			if f, ok := right[n]; ok {
				x := f
				c.After = &x
			}
			switch axis {
			case "codegen":
				out.Codegen = append(out.Codegen, c)
			case "resource":
				out.Resources = append(out.Resources, c)
			case "performance":
				out.Timings = append(out.Timings, c)
			}
		}
	}
	unknown := func(reason string) (Comparison, error) { out.Reason = reason; return out, nil }
	if a.Status != "complete" || b.Status != "complete" || a.Correctness.Verdict != Pass || b.Correctness.Verdict != Pass {
		return unknown("execution or independent correctness not qualified")
	}
	if a.Correctness.Oracle != b.Correctness.Oracle || a.Correctness.Method != b.Correctness.Method || a.Correctness.ExpectedDigest != b.Correctness.ExpectedDigest {
		return unknown("correctness oracle, method or expected output differs")
	}
	if a.InputDigest != b.InputDigest || a.WorkloadKey != b.WorkloadKey || a.NumericalMode != b.NumericalMode || a.Target != b.Target || a.Entry != b.Entry || a.PlanDigest != b.PlanDigest || !a.Environment.complete() || a.Environment != b.Environment {
		return unknown("workload, input, target, plan or environment differs")
	}
	left, right := facts(a, "performance"), facts(b, "performance")
	af, aok := left[p.Timing]
	bf, bok := right[p.Timing]
	if !aok || !bok {
		return unknown("timing channel unavailable")
	}
	av, bv := af.Value.Record(), bf.Value.Record()
	usable := func(f Fact, v cudaevidence.Record) bool {
		return f.Qualified && f.Samples >= p.MinSamples && v.Class == cudaevidence.Measured && v.Number != nil && *v.Number >= 0 && v.Unit == "ns" && v.Loss != nil && v.Loss.Total > 0 && v.Loss.Dropped == 0
	}
	if !usable(af, av) || !usable(bf, bv) {
		return unknown("timing is unread, unqualified, lossy or undersampled")
	}
	if *av.Number == 0 {
		return unknown("zero baseline cannot establish a relative regression")
	}
	boundary := *av.Number * (1 + p.RelativeThreshold)
	noise := p.NoiseNanos
	if av.Uncertainty != nil {
		noise += *av.Uncertainty * (1 + p.RelativeThreshold)
	}
	if bv.Uncertainty != nil {
		noise += *bv.Uncertainty
	}
	if math.IsInf(boundary, 0) || math.IsInf(noise, 0) {
		return unknown("comparison arithmetic exceeds finite range")
	}
	if math.Abs(*bv.Number-boundary) <= noise {
		return unknown("difference is within declared noise")
	}
	if *bv.Number > boundary+noise {
		out.Performance = Fail
		out.Reason = "qualified timing exceeds regression threshold"
	} else {
		out.Performance = Pass
		out.Reason = "qualified timing below regression threshold"
	}
	// Inputs are copied by encoding in Store; Compare itself never mutates them.
	return out, nil
}
