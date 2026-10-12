package cudamegacpu

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudamega"
)

// Reference computes the scratch buffer an ordinary launch per task produces:
// the tasks in rank order, each reading its dependency's finished output.
//
// It is the oracle, and it is written to share as little as possible with the
// thing it judges. It uses no thread, no barrier, no completion word and none
// of the primitives; it does not call the driver; and it applies the two
// operations through its own switch rather than the backend's, so that an
// operation implemented wrongly in one is not implemented wrongly in both.
// What it does share is [cudamega.Step], which is the statement of the
// problem rather than a step of the solution.
//
// Agreement between this and the CPU backend is evidence about the CPU
// backend. It is not evidence about the emitted CUDA: see the package
// comment.
// Extents and storage sizes use [cudamega.Program.Invocation] before allocation;
// operation evaluation remains independent of the worker.
func Reference(p *cudamega.Program, input []uint32) ([]uint32, error) {
	if p == nil {
		return nil, fmt.Errorf("cudamegacpu: nil program")
	}
	inv, err := p.Invocation(uint64(len(input)))
	if err != nil {
		return nil, fmt.Errorf("cudamegacpu: invocation: %w", err)
	}
	steps := p.Steps()
	n := len(input)
	out := make([]uint32, inv.ScratchWords)
	for i, s := range steps {
		from := input
		if len(s.Deps) > 0 {
			d := s.Deps[0]
			from = out[d*n : d*n+n]
		}
		for e := 0; e < n; e++ {
			switch s.Operation.Kind {
			case cudamega.Add:
				out[i*n+e] = from[e] + s.Operation.Scalar
			case cudamega.Multiply:
				out[i*n+e] = from[e] * s.Operation.Scalar
			default:
				return nil, fmt.Errorf("cudamegacpu: task %q has unsupported operation %d", s.Task.ID, s.Operation.Kind)
			}
		}
	}
	return out, nil
}
