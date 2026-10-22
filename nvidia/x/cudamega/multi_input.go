package cudamega

import (
	"fmt"
	"math"
	"strings"
)

// Body describes one element of a task. Inputs reports its operand count.
// Source receives that many CUDA C++ expression names and a writable output
// name, and returns the statements that compute the output. Evaluate computes
// the same value on the CPU. Values are uint32 words; a body may explicitly
// interpret their bits as another type in both implementations.
//
// Bodies are trusted caller code, not a sandbox for candidate code. A body must
// be immutable while any Program refers to it. Evaluate must be deterministic,
// total, safe for concurrent calls, and must not retain or change its operands.
// Source must write only its output, without synchronization or hidden effects.
// The caller owns the numerical equivalence of Source and Evaluate.
type Body interface {
	Inputs() int
	Source(inputs []string, output string) (string, error)
	Evaluate(inputs []uint32) uint32
}

// TaskStorage is one task's exact range in the invocation's scratch buffer.
// Offset and Elements count uint32 words. Ranges are disjoint and packed in
// task order; an extent is never used as another task's scratch stride.
type TaskStorage struct {
	Offset   int
	Elements int
}

// Storage returns a detached layout for the checked invocation. elements is
// the input buffer's length. A declared task extent is fixed; an undeclared
// extent uses elements. Every consumed source must cover the output extent.
func (p *Program) Storage(elements uint64) ([]TaskStorage, error) {
	if _, err := p.Invocation(elements); err != nil {
		return nil, err
	}
	layout, _, err := p.storage(elements)
	return layout, err
}

func (p *Program) inputSources(i int) []int {
	s := p.steps[i]
	if len(s.Task.Inputs) == 0 {
		if len(s.Deps) == 0 {
			return []int{-1}
		}
		return []int{s.Deps[0]}
	}
	out := make([]int, len(s.Task.Inputs))
	for j, name := range s.Task.Inputs {
		for _, d := range s.Deps {
			if p.steps[d].Task.ID == name {
				out[j] = d
				break
			}
		}
	}
	return out
}

func operationInputs(op Operation) int {
	if op.Body != nil {
		return op.Body.Inputs()
	}
	return 1
}

func (p *Program) storage(elements uint64) ([]TaskStorage, int, error) {
	maxWords := uint64(^uint(0)>>1) / 4
	var total uint64
	out := make([]TaskStorage, len(p.steps))
	for i, s := range p.steps {
		n := s.Task.Extent
		if n == 0 {
			n = elements
		}
		if n == 0 || n > math.MaxUint32 || total > maxWords || n > maxWords-total {
			return nil, 0, fmt.Errorf("task %q scratch extent exceeds addressable storage", s.Task.ID)
		}
		if p.grid.MaxElements != 0 && n > uint64(p.grid.MaxElements) {
			return nil, 0, fmt.Errorf("task %q extent %d exceeds maximum %d", s.Task.ID, n, p.grid.MaxElements)
		}
		out[i] = TaskStorage{Offset: int(total), Elements: int(n)}
		total += n
	}
	for i, s := range p.steps {
		for _, d := range p.inputSources(i) {
			n := elements
			if d >= 0 {
				n = uint64(out[d].Elements)
			}
			if n < uint64(out[i].Elements) {
				return nil, 0, fmt.Errorf("task %q requires %d input elements, source has %d: %w", s.Task.ID, out[i].Elements, n, ErrUnsupportedInputs)
			}
		}
	}
	return out, int(total), nil
}

func (p *Program) extentExpression(i int) string {
	if n := p.steps[i].Task.Extent; n != 0 {
		return fmt.Sprintf("%du", n)
	}
	return "n"
}

func (p *Program) offsetExpression(i int) string {
	var fixed uint64
	dynamic := 0
	for _, s := range p.steps[:i] {
		if s.Task.Extent == 0 {
			dynamic++
		} else {
			fixed += s.Task.Extent
		}
	}
	return fmt.Sprintf("(%dull + (size_t)%du*n)", fixed, dynamic)
}

func (c *cudaSource) boundChecks() string {
	var b strings.Builder
	for i, s := range c.p.steps {
		for _, d := range c.p.inputSources(i) {
			n := "n"
			if d >= 0 {
				n = c.p.extentExpression(d)
			}
			fmt.Fprintf(&b, "  if ((size_t)(%s) < (size_t)(%s)) { if (threadIdx.x == 0) atomicCAS((unsigned *)abort_code, 0u, %du); return; } // task %q input extent\n", n, c.p.extentExpression(i), AbortBadExtent, s.Task.ID)
		}
	}
	return b.String()
}

func (c *cudaSource) boundReference() error {
	// The ordinary kernel uses exactly the same operand binding and task
	// storage ranges, with one host-ordered launch for each task.
	c.defs.WriteString("extern \"C\" __global__ void ordinary_task(const unsigned *input, unsigned *scratch, unsigned n, unsigned task) {\n  if (!n) return;\n")
	if c.p.grid.MaxElements != 0 {
		fmt.Fprintf(&c.defs, "  if (n > %du) return;\n", c.p.grid.MaxElements)
	}
	c.defs.WriteString("  size_t e = (size_t)blockIdx.x*blockDim.x + threadIdx.x;\n  switch (task) {\n")
	for i := range c.p.steps {
		fmt.Fprintf(&c.defs, "  case %d: {\n", i)
		args := make([]string, 0)
		for j, d := range c.p.inputSources(i) {
			base, extent := "input", "n"
			if d >= 0 {
				base = "scratch + " + c.p.offsetExpression(d)
				extent = c.p.extentExpression(d)
			}
			fmt.Fprintf(&c.defs, "    if ((size_t)(%s) < (size_t)(%s)) return;\n", extent, c.p.extentExpression(i))
			fmt.Fprintf(&c.defs, "    const unsigned *x%d = %s;\n", j, base)
			args = append(args, fmt.Sprintf("x%d[e]", j))
		}
		fmt.Fprintf(&c.defs, "    if (e < %s) scratch[%s + e] = op_%d(%s);\n    return;\n  }\n", c.p.extentExpression(i), c.p.offsetExpression(i), i, strings.Join(args, ", "))
	}
	c.defs.WriteString("  default: return;\n  }\n}\n")
	return nil
}
