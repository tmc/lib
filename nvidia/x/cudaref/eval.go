package cudaref

import (
	"errors"
	"fmt"
	"math"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudashape"
)

// Evaluate runs a computation graph on the CPU using reference arithmetic.
// The inputs map supplies floating-point buffers keyed by input node name.
// Returns a map of tensor buffers for all graph output nodes.
func Evaluate(g *cudair.Graph, inputs map[string][]float32) (map[string][]float32, error) {
	snap, err := g.Snapshot()
	if err != nil {
		return nil, fmt.Errorf("cudaref.Evaluate: snapshot validation failed: %w", err)
	}
	return EvaluateSnapshot(snap, inputs)
}

// EvaluateSnapshot executes a validated immutable snapshot on the CPU.
func EvaluateSnapshot(snap *cudair.Snapshot, inputs map[string][]float32) (map[string][]float32, error) {
	values := make(map[int][]float32)

	// Validate and load inputs
	for _, inNode := range snap.Inputs() {
		buf, ok := inputs[inNode.Name]
		if !ok {
			return nil, fmt.Errorf("cudaref: missing input buffer for %q", inNode.Name)
		}
		expectedElems := inNode.NumElements()
		if len(buf) != expectedElems {
			return nil, fmt.Errorf("cudaref: input %q buffer length %d != expected %d",
				inNode.Name, len(buf), expectedElems)
		}
		// Defensive copy to prevent external mutation
		cp := make([]float32, len(buf))
		copy(cp, buf)
		values[inNode.ID] = cp
	}

	// Sequential execution along topological order
	for _, n := range snap.Order() {
		if n.Kind == cudair.KindInput {
			continue // Already populated
		}

		switch n.Kind {
		case cudair.KindConstant:
			if len(n.ConstValues) > 0 {
				cp := make([]float32, len(n.ConstValues))
				copy(cp, n.ConstValues)
				values[n.ID] = cp
			} else {
				values[n.ID] = make([]float32, n.NumElements())
			}

		case cudair.KindUnary:
			inData := values[n.Inputs[0].ID]
			out, err := ElementwiseUnary(inData, n.Op)
			if err != nil {
				return nil, fmt.Errorf("cudaref: node %s: %w", n.Name, err)
			}
			values[n.ID] = out

		case cudair.KindBinary:
			aData := values[n.Inputs[0].ID]
			bData := values[n.Inputs[1].ID]
			out, err := ElementwiseBinary(expandScalar(aData, n.NumElements()), expandScalar(bData, n.NumElements()), n.Op)
			if err != nil {
				return nil, fmt.Errorf("cudaref: node %s: %w", n.Name, err)
			}
			values[n.ID] = out

		case cudair.KindMatMul:
			aData := values[n.Inputs[0].ID]
			bData := values[n.Inputs[1].ID]
			M := n.Inputs[0].Shape[0]
			K := n.Inputs[0].Shape[1]
			N := n.Inputs[1].Shape[1]
			out, err := MatMul(aData, bData, M, N, K)
			if err != nil {
				return nil, fmt.Errorf("cudaref: node %s: %w", n.Name, err)
			}
			values[n.ID] = out

		case cudair.KindReduce:
			inData := values[n.Inputs[0].ID]
			inShape := n.Inputs[0].Shape
			axis := n.Axes[0]
			out, err := evalReduce(inData, inShape, axis, n.Op)
			if err != nil {
				return nil, fmt.Errorf("cudaref: node %s: %w", n.Name, err)
			}
			values[n.ID] = out

		case cudair.KindReshape:
			inData := values[n.Inputs[0].ID]
			out := make([]float32, len(inData))
			copy(out, inData)
			values[n.ID] = out

		case cudair.KindTranspose:
			inData := values[n.Inputs[0].ID]
			out, err := evalTranspose(inData, n.Inputs[0].Shape, n.Perm)
			if err != nil {
				return nil, fmt.Errorf("cudaref: node %s: %w", n.Name, err)
			}
			values[n.ID] = out

		case cudair.KindSlice:
			inData := values[n.Inputs[0].ID]
			out, err := evalSlice(inData, n.Inputs[0].Shape, n.SliceStarts, n.SliceEnds, n.SliceSteps)
			if err != nil {
				return nil, fmt.Errorf("cudaref: node %s: %w", n.Name, err)
			}
			values[n.ID] = out

		case cudair.KindConcat:
			var inDataSlice [][]float32
			var inShapes [][]int
			for _, inp := range n.Inputs {
				inDataSlice = append(inDataSlice, values[inp.ID])
				inShapes = append(inShapes, inp.Shape)
			}
			out, err := evalConcat(inDataSlice, inShapes, n.Axis)
			if err != nil {
				return nil, fmt.Errorf("cudaref: node %s: %w", n.Name, err)
			}
			values[n.ID] = out

		default:
			return nil, fmt.Errorf("cudaref: unsupported op kind %v", n.Kind)
		}
	}

	outputs := make(map[string][]float32, len(snap.Outputs()))
	for _, outNode := range snap.Outputs() {
		outputs[outNode.Name] = values[outNode.ID]
	}
	return outputs, nil
}

func evalReduce(in []float32, inShape []int, axis int, op string) ([]float32, error) {
	outerSize := 1
	for i := 0; i < axis; i++ {
		outerSize *= inShape[i]
	}
	reduceSize := inShape[axis]
	innerSize := 1
	for i := axis + 1; i < len(inShape); i++ {
		innerSize *= inShape[i]
	}

	outSize := outerSize * innerSize
	out := make([]float32, outSize)

	// Policy for empty reduction set:
	if reduceSize == 0 {
		switch op {
		case "sum":
			// Empty sum yields 0.0 (additive identity)
			for i := range out {
				out[i] = 0.0
			}
			return out, nil
		case "max":
			// Empty max yields -Infinity (maximum identity)
			for i := range out {
				out[i] = float32(math.Inf(-1))
			}
			return out, nil
		default:
			return nil, fmt.Errorf("unsupported empty reduction op %q", op)
		}
	}

	for outer := 0; outer < outerSize; outer++ {
		for inner := 0; inner < innerSize; inner++ {
			outIdx := outer*innerSize + inner
			switch op {
			case "sum":
				var acc float64
				for r := 0; r < reduceSize; r++ {
					inIdx := outer*(reduceSize*innerSize) + r*innerSize + inner
					acc += float64(in[inIdx])
				}
				out[outIdx] = float32(acc)
			case "max":
				firstIdx := outer*(reduceSize*innerSize) + inner
				maxVal := in[firstIdx]
				for r := 1; r < reduceSize; r++ {
					inIdx := outer*(reduceSize*innerSize) + r*innerSize + inner
					val := in[inIdx]
					if math.IsNaN(float64(val)) {
						maxVal = float32(math.NaN())
						break
					}
					if val > maxVal {
						maxVal = val
					}
				}
				out[outIdx] = maxVal
			default:
				return nil, fmt.Errorf("unsupported reduce op %q", op)
			}
		}
	}
	return out, nil
}

// expandScalar applies the IR's bounded scalar broadcast contract.
func expandScalar(values []float32, size int) []float32 {
	if len(values) != 1 || size == 1 {
		return values
	}
	out := make([]float32, size)
	for i := range out {
		out[i] = values[0]
	}
	return out
}

func evalTranspose(in []float32, inShape []int, perm []int) ([]float32, error) {
	rank := len(inShape)
	inStrides, err := cudashape.RowMajorStrides(inShape)
	if err != nil {
		return nil, err
	}

	outShape := make([]int, rank)
	for i, p := range perm {
		outShape[i] = inShape[p]
	}
	outStrides, err := cudashape.RowMajorStrides(outShape)
	if err != nil {
		return nil, err
	}

	out := make([]float32, len(in))
	coords := make([]int, rank)
	for inIdx := range in {
		// Recover coords from inIdx
		rem := inIdx
		for d := 0; d < rank; d++ {
			if inStrides[d] > 0 {
				coords[d] = rem / inStrides[d]
				rem %= inStrides[d]
			}
		}
		// Map coords through perm
		outIdx := 0
		for d := 0; d < rank; d++ {
			outIdx += coords[perm[d]] * outStrides[d]
		}
		out[outIdx] = in[inIdx]
	}
	return out, nil
}

func evalSlice(in []float32, inShape []int, starts, ends, steps []int) ([]float32, error) {
	rank := len(inShape)
	outShape := make([]int, rank)
	for d := 0; d < rank; d++ {
		outShape[d] = (ends[d] - starts[d] + steps[d] - 1) / steps[d]
	}
	inStrides, _ := cudashape.RowMajorStrides(inShape)
	outStrides, _ := cudashape.RowMajorStrides(outShape)

	totalOut := 1
	for _, d := range outShape {
		totalOut *= d
	}
	out := make([]float32, totalOut)

	outCoords := make([]int, rank)
	for outIdx := 0; outIdx < totalOut; outIdx++ {
		rem := outIdx
		for d := 0; d < rank; d++ {
			if outStrides[d] > 0 {
				outCoords[d] = rem / outStrides[d]
				rem %= outStrides[d]
			}
		}
		inIdx := 0
		for d := 0; d < rank; d++ {
			inIdx += (starts[d] + outCoords[d]*steps[d]) * inStrides[d]
		}
		out[outIdx] = in[inIdx]
	}
	return out, nil
}

func evalConcat(inList [][]float32, inShapes [][]int, axis int) ([]float32, error) {
	if len(inList) == 0 {
		return nil, errors.New("empty concat input")
	}
	rank := len(inShapes[0])
	outShape := append([]int(nil), inShapes[0]...)
	outShape[axis] = 0
	for _, s := range inShapes {
		outShape[axis] += s[axis]
	}
	outStrides, _ := cudashape.RowMajorStrides(outShape)

	totalOut := 1
	for _, d := range outShape {
		totalOut *= d
	}
	out := make([]float32, totalOut)

	axisOffset := 0
	for itemIdx, inData := range inList {
		inShape := inShapes[itemIdx]
		inStrides, _ := cudashape.RowMajorStrides(inShape)
		inTotal := len(inData)
		inCoords := make([]int, rank)
		for inIdx := 0; inIdx < inTotal; inIdx++ {
			rem := inIdx
			for d := 0; d < rank; d++ {
				if inStrides[d] > 0 {
					inCoords[d] = rem / inStrides[d]
					rem %= inStrides[d]
				}
			}
			outIdx := 0
			for d := 0; d < rank; d++ {
				coord := inCoords[d]
				if d == axis {
					coord += axisOffset
				}
				outIdx += coord * outStrides[d]
			}
			out[outIdx] = inData[inIdx]
		}
		axisOffset += inShape[axis]
	}
	return out, nil
}
