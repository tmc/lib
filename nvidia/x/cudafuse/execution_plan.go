package cudafuse

import (
	"fmt"
	"math"
	"slices"

	"github.com/tmc/lib/nvidia/x/cudair"
)

// executionPlan snapshots the caller's graph before borrowing storage or
// generating source. Each step has its own iteration and output domains.
type executionPlan struct {
	nodes     []*LazyNode
	root      *LazyNode
	originals map[*LazyNode]*LazyNode
	steps     []executionStep
}

type executionStep struct {
	inputs, outputs []*LazyNode
	count           int
	sum             bool
	scan            bool
	filter          bool
	expression      *LazyNode
	predicate       *LazyNode
}

func snapshotExecution(p *MaterializedPlan, eager bool) (*executionPlan, error) {
	if p == nil || p.RootNode == nil || p.RootNode.Op == "input" {
		return nil, fmt.Errorf("cudafuse: execution requires a compute output")
	}
	plan := &executionPlan{originals: make(map[*LazyNode]*LazyNode)}
	state := make(map[*LazyNode]int)
	names := make(map[string]bool)
	var originalOrder []*LazyNode
	filters := 0
	var visit func(*LazyNode) (*LazyNode, error)
	visit = func(n *LazyNode) (*LazyNode, error) {
		if n == nil || state[n] == 1 {
			return nil, fmt.Errorf("cudafuse: nil or cyclic execution node")
		}
		if state[n] == 2 {
			return plan.originals[n], nil
		}
		if !identifier(n.Name()) || names[n.Name()] {
			return nil, fmt.Errorf("cudafuse: invalid or duplicate execution node name")
		}
		names[n.Name()] = true
		count, err := indexedElements(n)
		if err != nil && !unresolvedShape(n.Shape) {
			return nil, err
		}
		if count > uint64(math.MaxInt-2) {
			return nil, fmt.Errorf("cudafuse: guarded execution extent overflows int")
		}
		if n.DType != cudair.Float32 {
			return nil, fmt.Errorf("cudafuse: execution requires float32: %w", cudair.ErrTypeMismatch)
		}
		clone := *n
		clone.Shape = slices.Clone(n.Shape)
		clone.Inputs = nil
		clone.Attrs = make(map[string]any, len(n.Attrs))
		for key, value := range n.Attrs {
			if ints, ok := value.([]int); ok {
				value = slices.Clone(ints)
			}
			clone.Attrs[key] = value
		}
		state[n] = 1
		for _, in := range n.Inputs {
			v, err := visit(in)
			if err != nil {
				return nil, err
			}
			clone.Inputs = append(clone.Inputs, v)
		}
		if n.Op == "input" {
			if len(n.Inputs) != 0 || len(n.Attrs) != 0 || unresolvedShape(n.Shape) {
				return nil, fmt.Errorf("cudafuse: invalid execution input")
			}
		} else if n.Op == "filter" {
			filters++
			if filters > 1 {
				return nil, fmt.Errorf("cudafuse: execution supports one Filter")
			}
			if err := validateFilterShape(&clone); err != nil {
				return nil, err
			}
		} else if n.Op == "sum" {
			axis, ok := clone.Attrs["axis"].(int)
			if !ok || axis != 0 || len(clone.Attrs) != 1 || len(clone.Inputs) != 1 || len(clone.Inputs[0].Shape) != 1 || !slices.Equal(clone.Shape, []int{1}) {
				return nil, fmt.Errorf("cudafuse: sum requires rank-one input, axis zero and output [1]")
			}
		} else if n.Op == "scan" {
			op, ok := clone.Attrs["scan_op"].(string)
			if !ok || op != "sum" || len(clone.Attrs) != 1 || len(clone.Inputs) != 1 || len(clone.Inputs[0].Shape) != 1 || !slices.Equal(clone.Shape, clone.Inputs[0].Shape) {
				return nil, fmt.Errorf("cudafuse: scan requires rank-one sum and matching output shape")
			}
		} else if err := validateIndexedOperator(&clone); err != nil {
			return nil, err
		}
		if clone.Op == "reshape" || clone.Op == "transpose" {
			a, _ := indexedElements(&clone)
			b, _ := indexedElements(clone.Inputs[0])
			if a != b {
				return nil, fmt.Errorf("cudafuse: view changes execution extent")
			}
		}
		state[n] = 2
		plan.originals[n] = &clone
		plan.nodes = append(plan.nodes, &clone)
		originalOrder = append(originalOrder, n)
		return &clone, nil
	}
	root, err := visit(p.RootNode)
	if err != nil {
		return nil, err
	}
	plan.root = root
	canonical, err := partitionIntoRegions(originalOrder)
	if err != nil {
		return nil, err
	}
	if len(canonical) != len(p.Regions) {
		return nil, fmt.Errorf("cudafuse: incomplete execution regions")
	}
	for i, want := range canonical {
		got := p.Regions[i]
		if got == nil || got.Compaction != want.Compaction || got.Boundary != want.Boundary || !sameExecutionNodes(got.Nodes, want.Nodes) || !sameExecutionNodes(got.Inputs, want.Inputs) || !sameExecutionNodes(got.Outputs, want.Outputs) {
			return nil, fmt.Errorf("cudafuse: stale execution region %d", i)
		}
	}
	regions, _ := partitionIntoRegions(plan.nodes)
	if eager {
		for _, n := range plan.nodes {
			if n.Op == "input" {
				continue
			}
			step := executionStep{inputs: uniqueExecutionNodes(n.Inputs), outputs: []*LazyNode{n}, count: executionCount(n)}
			if n.Op == "filter" {
				step.filter, step.expression, step.predicate = true, n.Inputs[0], n.Inputs[1]
			}
			if n.Op == "sum" {
				step.sum, step.expression = true, n.Inputs[0]
			}
			if n.Op == "scan" {
				step.scan, step.expression = true, n.Inputs[0]
			}
			plan.steps = append(plan.steps, step)
		}
		return plan, nil
	}
	for i := 0; i < len(regions); i++ {
		r := regions[i]
		var outputs []*LazyNode
		for _, n := range r.Outputs {
			if n.Op != "input" {
				outputs = append(outputs, n)
			}
		}
		if len(outputs) == 0 {
			continue
		}
		if r.Boundary == "filter" {
			n := outputs[0]
			plan.steps = append(plan.steps, executionStep{inputs: uniqueExecutionNodes(r.Inputs), outputs: outputs, count: executionCount(n.Inputs[0]), filter: true, expression: n.Inputs[0], predicate: n.Inputs[1]})
			continue
		}
		if r.Boundary == "" && !r.Compaction && i+1 < len(regions) && regions[i+1].Boundary == "filter" && filterCanFuse(outputs, regions[i+1].Outputs[0], plan.nodes) {
			n := regions[i+1].Outputs[0]
			plan.steps = append(plan.steps, executionStep{inputs: filterRegionInputs(n, r.Nodes), outputs: []*LazyNode{n}, count: executionCount(n.Inputs[0]), filter: true, expression: n.Inputs[0], predicate: n.Inputs[1]})
			i++
			continue
		}
		if r.Boundary == "scan" {
			plan.steps = append(plan.steps, executionStep{inputs: uniqueExecutionNodes(r.Inputs), outputs: outputs, count: executionCount(outputs[0]), scan: true, expression: outputs[0].Inputs[0]})
			continue
		}
		if r.Boundary == "sum" {
			plan.steps = append(plan.steps, executionStep{inputs: uniqueExecutionNodes(r.Inputs), outputs: outputs, count: 1, sum: true, expression: outputs[0].Inputs[0]})
			continue
		}
		if len(outputs) == 1 && i+1 < len(regions) && regions[i+1].Boundary == "scan" && regions[i+1].Outputs[0].Inputs[0] == outputs[0] && otherUsersCount(outputs[0], plan.nodes) == 1 {
			scan := regions[i+1].Outputs[0]
			plan.steps = append(plan.steps, executionStep{inputs: uniqueExecutionNodes(r.Inputs), outputs: []*LazyNode{scan}, count: executionCount(scan), scan: true, expression: outputs[0]})
			i++
			continue
		}
		// Only pull an entire region into its sole sum consumer. Other
		// consumers, including requested secondary outputs, retain storage.
		if len(outputs) == 1 && i+1 < len(regions) && regions[i+1].Boundary == "sum" && regions[i+1].Outputs[0].Inputs[0] == outputs[0] && otherUsersCount(outputs[0], plan.nodes) == 1 {
			sum := regions[i+1].Outputs[0]
			plan.steps = append(plan.steps, executionStep{inputs: uniqueExecutionNodes(r.Inputs), outputs: []*LazyNode{sum}, count: 1, sum: true, expression: outputs[0]})
			i++
			continue
		}
		plan.steps = append(plan.steps, executionStep{inputs: uniqueExecutionNodes(r.Inputs), outputs: outputs, count: executionCount(outputs[0])})
	}
	return plan, nil
}

func filterCanFuse(outputs []*LazyNode, filter *LazyNode, nodes []*LazyNode) bool {
	for _, out := range outputs {
		uses := 0
		for _, in := range filter.Inputs {
			if in == out {
				uses++
			}
		}
		if uses == 0 || otherUsersCount(out, nodes) != uses {
			return false
		}
	}
	return true
}

func filterRegionInputs(filter *LazyNode, fused []*LazyNode) []*LazyNode {
	var inputs []*LazyNode
	var visit func(*LazyNode)
	visit = func(n *LazyNode) {
		if n.Op == "input" || !slices.Contains(fused, n) {
			if !slices.Contains(inputs, n) {
				inputs = append(inputs, n)
			}
			return
		}
		for _, in := range n.Inputs {
			visit(in)
		}
	}
	for _, in := range filter.Inputs {
		visit(in)
	}
	return inputs
}

func uniqueExecutionNodes(nodes []*LazyNode) []*LazyNode {
	var out []*LazyNode
	for _, n := range nodes {
		if !slices.Contains(out, n) {
			out = append(out, n)
		}
	}
	return out
}

func sameExecutionNodes(got, want []*LazyNode) bool {
	if len(got) != len(want) || len(uniqueExecutionNodes(got)) != len(got) {
		return false
	}
	for _, n := range want {
		if !slices.Contains(got, n) {
			return false
		}
	}
	return true
}

func (s executionStep) source(name string) (FusedKernel, error) {
	if s.scan {
		return compileScanKernel(s.inputs, s.expression, s.outputs[0], name)
	}
	if s.sum {
		return compileSumKernel(s.inputs, s.expression, s.outputs[0], name)
	}
	// Preserve the existing source/receipt contract for an ordinary single
	// region rooted entirely in caller inputs. Boundary inputs use FuseLazy's
	// explicit external-node contract instead of traversing into their producer.
	plain := len(s.outputs) == 1
	for _, in := range s.inputs {
		plain = plain && in.Op == "input"
	}
	if plain {
		nodes, err := checkedTopologicalOrder(s.outputs[0])
		if err != nil {
			return FusedKernel{}, err
		}
		return executionSource(nodes, s.outputs[0], name, false)
	}
	return compileIndexedLazyRegion(s.inputs, s.outputs, name)
}
