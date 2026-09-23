package cudabpfdev

import (
	"fmt"
	"math"
	"slices"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudaref"
)

// NumericalPolicy declares float32 output comparison semantics. Mode must be
// "exact" (identical bits) or "tolerance". Tolerance mode accepts finite values
// when abs(got-want) <= AbsoluteTolerance + RelativeTolerance*abs(want).
// Tolerances must be finite and nonnegative; exact mode requires both to be zero.
// Infinities must match in sign. NaNs match only when EqualNaNs is true, regardless
// of payload. Signed zeros must match unless IgnoreZeroSign is explicitly set
// in tolerance mode. The zero policy is invalid, requiring an explicit choice.
type NumericalPolicy struct {
	Mode                                 string
	AbsoluteTolerance, RelativeTolerance float64
	EqualNaNs, IgnoreZeroSign            bool
}

// CompareOutputs evaluates graph with cudaref and compares both original and
// instrumented outputs to that independent reference, then to each other.
// Output names and lengths must match exactly. Only float32 graphs are accepted.
// The first mismatch is returned in sorted output-name order, with its pair and
// element index. Graph and buffers must not be mutated concurrently.
//
// The caller must supply completed readbacks for the same inputs and identify
// the original and instrumented artifacts in its qualification receipt. This
// function establishes numerical agreement of supplied buffers, not that either
// kernel ran, that injection occurred, or that instrumentation overhead is small.
func CompareOutputs(graph *cudair.Graph, inputs, original, instrumented map[string][]float32, policy NumericalPolicy) error {
	if policy.Mode != "exact" && policy.Mode != "tolerance" {
		return fmt.Errorf("cudabpfdev: numerical mode must be exact or tolerance")
	}
	for _, v := range []float64{policy.AbsoluteTolerance, policy.RelativeTolerance} {
		if v < 0 || math.IsNaN(v) || math.IsInf(v, 0) {
			return fmt.Errorf("cudabpfdev: invalid numerical tolerance")
		}
	}
	if policy.Mode == "exact" && (policy.AbsoluteTolerance != 0 || policy.RelativeTolerance != 0 || policy.IgnoreZeroSign) {
		return fmt.Errorf("cudabpfdev: exact comparison cannot relax tolerance or zero sign")
	}
	if graph == nil {
		return fmt.Errorf("cudabpfdev: nil reference graph")
	}
	snap, err := graph.Snapshot()
	if err != nil {
		return fmt.Errorf("cudabpfdev: reference graph: %w", err)
	}
	if len(snap.Outputs()) == 0 {
		return fmt.Errorf("cudabpfdev: reference graph has no outputs")
	}
	for _, n := range snap.Nodes() {
		if n.Type != cudair.Float32 {
			return fmt.Errorf("cudabpfdev: reference node %q is not float32", n.Name)
		}
	}
	reference, err := cudaref.EvaluateSnapshot(snap, inputs)
	if err != nil {
		return fmt.Errorf("cudabpfdev: reference evaluation: %w", err)
	}
	names := make([]string, 0, len(reference))
	for name := range reference {
		names = append(names, name)
	}
	slices.Sort(names)
	for _, pair := range []struct {
		name      string
		got, want map[string][]float32
	}{
		{"original/reference", original, reference},
		{"instrumented/reference", instrumented, reference},
		{"instrumented/original", instrumented, original},
	} {
		if len(pair.got) != len(reference) {
			return fmt.Errorf("cudabpfdev: %s output count mismatch", pair.name)
		}
		for _, name := range names {
			got, ok := pair.got[name]
			want := pair.want[name]
			if !ok || len(got) != len(want) {
				return fmt.Errorf("cudabpfdev: %s output %q missing or length mismatch", pair.name, name)
			}
			for i, v := range got {
				if !policy.matches(v, want[i]) {
					return fmt.Errorf("cudabpfdev: %s output %q[%d]: got %g (0x%08x), want %g (0x%08x)", pair.name, name, i, v, math.Float32bits(v), want[i], math.Float32bits(want[i]))
				}
			}
		}
	}
	return nil
}

func (p NumericalPolicy) matches(got, want float32) bool {
	g, w := float64(got), float64(want)
	if math.IsNaN(g) || math.IsNaN(w) {
		return p.EqualNaNs && math.IsNaN(g) && math.IsNaN(w)
	}
	if p.Mode == "exact" {
		return math.Float32bits(got) == math.Float32bits(want)
	}
	if math.IsInf(g, 0) || math.IsInf(w, 0) {
		return g == w
	}
	if g == 0 && w == 0 && !p.IgnoreZeroSign {
		return math.Signbit(g) == math.Signbit(w)
	}
	return math.Abs(g-w) <= p.AbsoluteTolerance+p.RelativeTolerance*math.Abs(w)
}
