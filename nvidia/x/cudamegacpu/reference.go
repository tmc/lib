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
	layout, err := p.Storage(uint64(len(input)))
	if err != nil {
		return nil, err
	}
	out := make([]uint32, inv.ScratchWords)
	for i, s := range steps {
		// Resolve declared operand names independently of the Runner.
		ids := s.Task.Inputs
		if len(ids) == 0 && len(s.Deps) != 0 {
			ids = []string{steps[s.Deps[0]].Task.ID}
		}
		operands := [][]uint32{input}
		if len(ids) != 0 {
			operands = nil
			for _, id := range ids {
				for d := range steps[:i] {
					if steps[d].Task.ID == id {
						r := layout[d]
						operands = append(operands, out[r.Offset:r.Offset+r.Elements])
						break
					}
				}
			}
		}
		r := layout[i]
		values := make([]uint32, len(operands))
		for e := 0; e < r.Elements; e++ {
			for j := range operands {
				values[j] = operands[j][e]
			}
			if s.Operation.Body != nil {
				out[r.Offset+e] = s.Operation.Body.Evaluate(values)
				continue
			}
			switch s.Operation.Kind {
			case cudamega.Add:
				out[r.Offset+e] = values[0] + s.Operation.Scalar
			case cudamega.Multiply:
				out[r.Offset+e] = values[0] * s.Operation.Scalar
			default:
				return nil, fmt.Errorf("cudamegacpu: task %q has unsupported operation %d", s.Task.ID, s.Operation.Kind)
			}
		}
	}
	return out, nil
}
