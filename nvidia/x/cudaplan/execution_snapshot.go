package cudaplan

import (
	"fmt"
	"math"

	"github.com/tmc/lib/nvidia/x/cudair"
)

type nodeSnapshot interface {
	Order() []*cudair.Node
	Inputs() []*cudair.Node
	Outputs() []*cudair.Node
	NodeByID(int) *cudair.Node
}

// executionSnapshot owns its metadata. The public IR snapshot exposes mutable
// node pointers, so retaining it would let later edits change submitted work.
type executionSnapshot struct {
	order, inputs, outputs []*cudair.Node
	byID                   map[int]*cudair.Node
	arena                  *cudair.ArenaPlan
}

func (s *executionSnapshot) Order() []*cudair.Node        { return s.order }
func (s *executionSnapshot) Inputs() []*cudair.Node       { return s.inputs }
func (s *executionSnapshot) Outputs() []*cudair.Node      { return s.outputs }
func (s *executionSnapshot) NodeByID(id int) *cudair.Node { return s.byID[id] }
func (s *executionSnapshot) ArenaPlan() *cudair.ArenaPlan { return s.arena }

func freezeExecution(source *cudair.Snapshot) (*executionSnapshot, error) {
	reachable := make(map[*cudair.Node]bool)
	var visit func(*cudair.Node)
	visit = func(n *cudair.Node) {
		if n == nil || reachable[n] {
			return
		}
		reachable[n] = true
		for _, in := range n.Inputs {
			visit(in)
		}
		visit(n.StorageAlias)
	}
	for _, out := range source.Outputs() {
		visit(out)
	}
	clones := make(map[*cudair.Node]*cudair.Node)
	s := &executionSnapshot{byID: make(map[int]*cudair.Node)}
	for _, n := range source.Order() {
		if !reachable[n] {
			continue
		}
		if n == nil || s.byID[n.ID] != nil {
			return nil, fmt.Errorf("cudaplan: invalid execution node identity")
		}
		c := *n
		c.Inputs = append([]*cudair.Node(nil), n.Inputs...)
		c.Shape = append([]int(nil), n.Shape...)
		c.Strides = append([]int(nil), n.Strides...)
		c.ConstValues = append([]float32(nil), n.ConstValues...)
		c.Axes = append([]int(nil), n.Axes...)
		c.Perm = append([]int(nil), n.Perm...)
		c.SliceStarts = append([]int(nil), n.SliceStarts...)
		c.SliceEnds = append([]int(nil), n.SliceEnds...)
		c.SliceSteps = append([]int(nil), n.SliceSteps...)
		c.SourceLoc.Callers = append([]cudair.Frame(nil), n.SourceLoc.Callers...)
		clones[n] = &c
		s.byID[c.ID] = &c
		s.order = append(s.order, &c)
		if c.Kind == cudair.KindInput {
			s.inputs = append(s.inputs, &c)
		}
	}
	for original, c := range clones {
		for i, in := range original.Inputs {
			c.Inputs[i] = clones[in]
			if c.Inputs[i] == nil {
				return nil, fmt.Errorf("cudaplan: missing execution dependency")
			}
		}
		c.StorageAlias = clones[original.StorageAlias]
	}
	for _, out := range source.Outputs() {
		c := clones[out]
		if c == nil {
			return nil, fmt.Errorf("cudaplan: missing execution output")
		}
		s.outputs = append(s.outputs, c)
	}
	if len(s.outputs) == 0 {
		return nil, fmt.Errorf("cudaplan: missing execution outputs")
	}
	var err error
	s.arena, err = executionArena(s.order)
	return s, err
}

// Reuse starts only after the last consumer has finished. Reusing at that
// consumer's step can overlap a parallel reduction's reads with its writes.
func executionArena(order []*cudair.Node) (*cudair.ArenaPlan, error) {
	last := make(map[*cudair.Node]int)
	for i, n := range order {
		last[n] = i
		for _, in := range n.Inputs {
			last[in] = i
		}
	}
	type block struct{ offset, size, until int }
	var blocks []block
	p := &cudair.ArenaPlan{Offsets: make(map[*cudair.Node]int)}
	for i, n := range order {
		if n.Kind == cudair.KindInput {
			continue
		}
		size := n.SizeBytes()
		if size < 0 || size > math.MaxInt-255 {
			return nil, ErrWorkspaceBudgetExceeded
		}
		size = (size + 255) &^ 255
		found := -1
		for j, b := range blocks {
			if b.until < i && b.size >= size {
				found = j
				break
			}
		}
		if found >= 0 {
			p.Offsets[n] = blocks[found].offset
			blocks[found].until = last[n]
		} else {
			if p.TotalBytes > math.MaxInt-size {
				return nil, ErrWorkspaceBudgetExceeded
			}
			p.Offsets[n] = p.TotalBytes
			blocks = append(blocks, block{p.TotalBytes, size, last[n]})
			p.TotalBytes += size
		}
	}
	return p, nil
}
