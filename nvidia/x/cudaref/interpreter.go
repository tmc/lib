// Package cudaref provides a CPU reference interpreter for cudair graphs.
package cudaref

import (
	"fmt"
	"math"

	"github.com/tmc/lib/nvidia/x/cudair"
)

type Interpreter struct{}

func NewInterpreter() *Interpreter {
	return &Interpreter{}
}

func (ip *Interpreter) Evaluate(g *cudair.Graph, inputs map[int][]float32) (map[int][]float32, error) {
	results := make(map[int][]float32)
	for k, v := range inputs {
		results[k] = v
	}

	for _, n := range g.Nodes() {
		if n.Kind == cudair.KindInput {
			if _, ok := results[n.ID]; !ok {
				return nil, fmt.Errorf("cudaref: missing input for node %d", n.ID)
			}
			continue
		}
		if n.Kind == cudair.KindConstant {
			results[n.ID] = append([]float32(nil), n.ConstValues...)
			continue
		}

		if len(n.Inputs) == 0 {
			return nil, fmt.Errorf("cudaref: node %d has no input", n.ID)
		}
		in0 := results[n.Inputs[0].ID]
		if in0 == nil {
			return nil, fmt.Errorf("cudaref: missing input for node %d", n.ID)
		}
		out := make([]float32, len(in0))

		switch n.Op {
		case "add":
			if len(n.Inputs) != 2 {
				return nil, fmt.Errorf("cudaref: add node %d has %d inputs", n.ID, len(n.Inputs))
			}
			in1 := results[n.Inputs[1].ID]
			for i := range out {
				out[i] = in0[i] + in1[i]
			}
		case "multiply", "mul":
			if len(n.Inputs) != 2 {
				return nil, fmt.Errorf("cudaref: multiply node %d has %d inputs", n.ID, len(n.Inputs))
			}
			in1 := results[n.Inputs[1].ID]
			for i := range out {
				out[i] = in0[i] * in1[i]
			}
		case "relu":
			for i := range out {
				if in0[i] > 0 {
					out[i] = in0[i]
				} else {
					out[i] = 0
				}
			}
		case "exp":
			for i := range out {
				out[i] = float32(math.Exp(float64(in0[i])))
			}
		default:
			return nil, fmt.Errorf("cudaref: unsupported op %v", n.Op)
		}
		results[n.ID] = out
	}

	outputs := make(map[int][]float32)
	for _, outNode := range g.Outputs() {
		outputs[outNode.ID] = results[outNode.ID]
	}
	return outputs, nil
}
