package cudafuse

import (
	"context"
	"errors"
	"fmt"
	"io"
	"slices"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudair"
)

func unresolvedShape(shape []int) bool { return len(shape) == 1 && shape[0] == -1 }

func validateFilterShape(n *LazyNode) error {
	if len(n.Inputs) != 2 || len(n.Attrs) != 0 || !unresolvedShape(n.Shape) || n.DType != cudair.Float32 {
		return fmt.Errorf("cudafuse: Filter requires two FP32 inputs and unresolved rank-one output")
	}
	a, p := n.Inputs[0], n.Inputs[1]
	if a == nil || p == nil || a.DType != cudair.Float32 || p.DType != cudair.Float32 || len(a.Shape) != 1 || !slices.Equal(a.Shape, p.Shape) {
		return fmt.Errorf("cudafuse: Filter requires matching rank-one FP32 data and predicate")
	}
	count, err := elementCount(a.Shape)
	if err != nil {
		return err
	}
	if count > maxFilterElements {
		return fmt.Errorf("cudafuse: Filter count exceeds exact FP32 prefix domain: %w", ErrUnsupportedOperator)
	}
	return nil
}

func executionCount(n *LazyNode) int {
	if unresolvedShape(n.Shape) {
		return -1
	}
	return int(mustExecutionElements(n))
}

func resolveEmptyFilter(plan *executionPlan) error {
	for _, step := range plan.steps {
		if step.filter && executionCount(step.expression) == 0 {
			return resolveFilterShapes(plan, 0)
		}
	}
	return nil
}

func resolveFilterShapes(plan *executionPlan, selected int) error {
	for _, n := range plan.nodes {
		if !unresolvedShape(n.Shape) {
			continue
		}
		switch n.Op {
		case "filter":
			n.Shape = []int{selected}
		case "add", "sub", "mul", "div":
			shape, err := binaryShape(n.Inputs[0].Shape, n.Inputs[1].Shape)
			if err != nil || unresolvedShape(shape) {
				return fmt.Errorf("cudafuse: unresolved Filter consumer shape")
			}
			n.Shape = shape
		default:
			n.Shape = slices.Clone(n.Inputs[0].Shape)
		}
	}
	for i := range plan.steps {
		if !plan.steps[i].filter {
			plan.steps[i].count = executionCount(plan.steps[i].outputs[0])
		}
	}
	return nil
}

func filterCount(predicate, prefix, metadata []float32) (int, error) {
	if len(predicate) != len(prefix) || len(metadata) != 2 || len(prefix) > maxFilterElements {
		return 0, fmt.Errorf("cudafuse: missing Filter count evidence")
	}
	count := 0
	for i, p := range predicate {
		if p != 0 && p != 1 {
			return 0, fmt.Errorf("cudafuse: invalid Filter predicate at %d", i)
		}
		if p == 1 {
			count++
		}
		if prefix[i] != float32(count) {
			return 0, fmt.Errorf("cudafuse: Filter prefix/count disagreement at %d", i)
		}
	}
	if metadata[0] != 0 || metadata[1] != float32(count) {
		return 0, fmt.Errorf("cudafuse: Filter status/count disagreement")
	}
	return count, nil
}

func (e *Execution) allocate(n int) (*executionBuffer, error) {
	b, err := e.ops.alloc(n)
	if b != nil {
		e.resources = append(e.resources, b)
		e.stats.Allocations++
	}
	if err != nil {
		return b, fmt.Errorf("cudafuse: allocate: %w", err)
	}
	if b == nil || b.ptr == 0 {
		return b, errors.New("cudafuse: missing execution buffer")
	}
	return b, nil
}

func (e *Execution) compile(source FusedKernel) (*executionKernel, error) {
	k, err := e.ops.compile(source)
	if k != nil {
		e.kernels = append(e.kernels, k)
	}
	if err != nil {
		var pending io.Closer
		if errors.As(err, &pending) {
			e.pending = append(e.pending, pending)
		}
		return k, fmt.Errorf("cudafuse: compile/load: %w", err)
	}
	if k == nil || k.launch == nil {
		return k, errors.New("cudafuse: missing execution kernel")
	}
	e.receipts = append(e.receipts, k.receipt)
	return k, nil
}

func (e *Execution) launch(k *executionKernel, pointers []driver.DevicePtr, n int) error {
	e.attempted, e.complete = true, false
	if err := k.launch(pointers, n); err != nil {
		return fmt.Errorf("cudafuse: launch: %w", err)
	}
	e.stats.Launches++
	return nil
}

func (e *Execution) filter(ctx context.Context, plan *executionPlan, step executionStep, values map[*LazyNode]driver.DevicePtr, index int) error {
	output := step.outputs[0]
	n := executionCount(step.expression)
	selected := 0
	var scratch []*executionBuffer
	if n > 0 {
		source, err := compileFilterCount(step.inputs, step.expression, step.predicate, output, fmt.Sprintf("lazy_filter_count_%d", index))
		if err != nil {
			return err
		}
		kernel, err := e.compile(source)
		if err != nil {
			return err
		}
		var pointers []driver.DevicePtr
		for _, in := range step.inputs {
			ptr := values[in]
			if ptr == 0 && executionCount(in) != 0 {
				return errors.New("cudafuse: missing Filter operand")
			}
			pointers = append(pointers, ptr)
		}
		for _, length := range []int{n, n, n, 2} {
			if ctx != nil && ctx.Err() != nil {
				return ctx.Err()
			}
			b, err := e.allocate(length)
			if err != nil {
				return err
			}
			scratch = append(scratch, b)
			pointers = append(pointers, b.ptr)
		}
		if err := e.launch(kernel, pointers, n); err != nil {
			return err
		}
		if err := e.Wait(ctx); err != nil {
			return err
		}
		predicate, prefix, metadata := make([]float32, n), make([]float32, n), make([]float32, 2)
		for i, out := range [][]float32{predicate, prefix, metadata} {
			if err := scratch[i+1].read(out); err != nil {
				return fmt.Errorf("cudafuse: read Filter count: %w", err)
			}
		}
		selected, err = filterCount(predicate, prefix, metadata)
		if err != nil {
			return err
		}
	}
	if err := resolveFilterShapes(plan, selected); err != nil {
		return err
	}
	e.n, e.shape = executionCount(plan.root), slices.Clone(plan.root.Shape)
	if selected == 0 {
		values[output] = 0
		return nil
	}
	if ctx != nil && ctx.Err() != nil {
		return ctx.Err()
	}
	b, err := e.allocate(selected)
	if err != nil {
		return err
	}
	values[output] = b.ptr
	if output == plan.root {
		e.output = b
	}
	source := compileFilterScatter(output, n, selected, fmt.Sprintf("lazy_filter_scatter_%d", index))
	kernel, err := e.compile(source)
	if err != nil {
		return err
	}
	if ctx != nil && ctx.Err() != nil {
		return ctx.Err()
	}
	return e.launch(kernel, []driver.DevicePtr{scratch[0].ptr, scratch[1].ptr, scratch[2].ptr, b.ptr}, n)
}
