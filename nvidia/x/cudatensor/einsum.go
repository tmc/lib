package cudatensor

import (
	"errors"
	"fmt"
	"sort"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudashape"
)

var (
	// ErrCyclicGraph reports a rank graph that cannot be traversed from its inputs.
	ErrCyclicGraph = errors.New("cudatensor: cyclic einsum graph")
	// ErrNotCanonicalized reports an export attempted before rank canonicalization.
	ErrNotCanonicalized = errors.New("cudatensor: einsum graph is not canonicalized")
)

// EinsumNode describes one operation in an einsum DAG. Inputs are predecessor
// node IDs in operand order. Start nodes describe external tensors and have no
// equation; their output rank count is OutputRank.
type EinsumNode struct {
	ID         string
	Equation   string
	Inputs     []string
	OutputRank int
}

// Graph is an ordered einsum graph. Node order is significant and supplies the
// stable order used to resolve otherwise equivalent traversals.
type Graph struct {
	Nodes []EinsumNode
	// Shape maps dimension labels to Timeloop bound expressions.
	Shape map[string]string
}

// CanonicalNode is the rank-renamed representation of one graph node.
type CanonicalNode struct {
	ID       string
	Equation string
	Inputs   []string
	Start    bool
	Output   []cudashape.DimToken
}

// CanonicalGraph is an einsum graph whose ranks have stable generated names.
type CanonicalGraph struct {
	Nodes         []CanonicalNode
	Shape         map[string]string
	canonicalized bool
}

// CanonicalizeRanks renames dimension ranks in predecessor-ready breadth-first
// order. Start nodes are retained as external inputs and do not consume labels.
func CanonicalizeRanks(g *Graph) (*CanonicalGraph, error) {
	if g == nil {
		return nil, errors.New("cudatensor: nil einsum graph")
	}
	byID := make(map[string]int, len(g.Nodes))
	for i, n := range g.Nodes {
		if n.ID == "" {
			return nil, errors.New("cudatensor: empty einsum node ID")
		}
		if _, ok := byID[n.ID]; ok {
			return nil, fmt.Errorf("cudatensor: duplicate einsum node %q", n.ID)
		}
		byID[n.ID] = i
	}
	indegree := make([]int, len(g.Nodes))
	successors := make([][]int, len(g.Nodes))
	for i, n := range g.Nodes {
		for _, id := range n.Inputs {
			p, ok := byID[id]
			if !ok {
				return nil, fmt.Errorf("cudatensor: node %q has unknown input %q", n.ID, id)
			}
			indegree[i]++
			successors[p] = append(successors[p], i)
		}
	}
	queue := make([]int, 0, len(g.Nodes))
	for i, d := range indegree {
		if d == 0 {
			queue = append(queue, i)
		}
	}
	processed := make([]bool, len(g.Nodes))
	outputs := make(map[string][]cudashape.DimToken, len(g.Nodes))
	result := &CanonicalGraph{Nodes: make([]CanonicalNode, len(g.Nodes)), Shape: cloneStringMap(g.Shape), canonicalized: true}
	nextLabel := 0
	stalled := 0
	for len(queue) != 0 {
		i := queue[0]
		queue = queue[1:]
		if processed[i] {
			continue
		}
		ready := true
		for _, id := range g.Nodes[i].Inputs {
			if !processed[byID[id]] {
				ready = false
				break
			}
		}
		if !ready {
			queue = append(queue, i)
			stalled++
			if stalled >= len(queue) {
				return nil, ErrCyclicGraph
			}
			continue
		}
		stalled = 0
		n := g.Nodes[i]
		cn := CanonicalNode{ID: n.ID, Inputs: append([]string(nil), n.Inputs...), Start: len(n.Inputs) == 0}
		if cn.Start {
			if n.Equation != "" {
				return nil, fmt.Errorf("cudatensor: start node %q must not have an equation", n.ID)
			}
			if n.OutputRank < 0 {
				return nil, fmt.Errorf("cudatensor: node %q has negative output rank", n.ID)
			}
			outputs[n.ID] = make([]cudashape.DimToken, n.OutputRank)
			for j := range outputs[n.ID] {
				outputs[n.ID][j] = labelToken(nextLabel)
				nextLabel++
			}
			cn.Output = append([]cudashape.DimToken(nil), outputs[n.ID]...)
		} else {
			inputs, out, err := cudashape.ParseEinsumEquation(n.Equation)
			if err != nil {
				return nil, fmt.Errorf("cudatensor: node %q: %w", n.ID, err)
			}
			if len(inputs) != len(n.Inputs) {
				return nil, fmt.Errorf("cudatensor: node %q has %d operands for %d inputs", n.ID, len(inputs), len(n.Inputs))
			}
			mapping := make(map[cudashape.DimToken]cudashape.DimToken)
			used := make(map[cudashape.DimToken]bool)
			for oi, operand := range inputs {
				pred := outputs[n.Inputs[oi]]
				if len(operand) != len(pred) {
					return nil, fmt.Errorf("cudatensor: node %q input %q rank mismatch", n.ID, n.Inputs[oi])
				}
				for j, token := range operand {
					if _, ok := mapping[token]; ok {
						continue
					}
					candidate := pred[j]
					if !used[candidate] {
						mapping[token] = candidate
						used[candidate] = true
					}
				}
			}
			all := append([]cudashape.DimToken(nil), out...)
			for _, operand := range inputs {
				all = append(all, operand...)
			}
			for _, token := range all {
				if _, ok := mapping[token]; !ok {
					mapping[token] = labelToken(nextLabel)
					nextLabel++
					used[mapping[token]] = true
				}
			}
			newInputs := make([][]cudashape.DimToken, len(inputs))
			for oi, operand := range inputs {
				for _, token := range operand {
					newInputs[oi] = append(newInputs[oi], mapping[token])
				}
			}
			newOut := make([]cudashape.DimToken, len(out))
			for j, token := range out {
				newOut[j] = mapping[token]
			}
			cn.Equation = buildEinsumEquation(newInputs, newOut)
			outputs[n.ID] = newOut
			cn.Output = append([]cudashape.DimToken(nil), newOut...)
		}
		result.Nodes[i] = cn
		processed[i] = true
		for _, s := range successors[i] {
			if !processed[s] {
				queue = append(queue, s)
			}
		}
	}
	for _, done := range processed {
		if !done {
			return nil, ErrCyclicGraph
		}
	}
	return result, nil
}

// CanonicalKey returns a stable serialization suitable for graph comparisons.
func CanonicalKey(g *Graph) string {
	c, err := CanonicalizeRanks(g)
	if err != nil {
		return ""
	}
	var b strings.Builder
	for _, n := range c.Nodes {
		fmt.Fprintf(&b, "%q|%t|%q|", n.ID, n.Start, n.Equation)
		for _, rank := range n.Output {
			fmt.Fprintf(&b, "%q,", rank)
		}
		b.WriteByte('|')
		for _, in := range n.Inputs {
			fmt.Fprintf(&b, "%q,", in)
		}
		b.WriteByte('\n')
	}
	keys := make([]string, 0, len(c.Shape))
	for k := range c.Shape {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		fmt.Fprintf(&b, "%q=%q\n", k, c.Shape[k])
	}
	return b.String()
}

func labelToken(index int) cudashape.DimToken {
	letter := index % 26
	suffix := ""
	if index >= 26 {
		suffix = fmt.Sprint((index - 26) / 26)
	}
	return cudashape.DimToken(string(rune('A'+letter)) + suffix)
}

func buildEinsumEquation(inputs [][]cudashape.DimToken, output []cudashape.DimToken) string {
	parts := make([]string, len(inputs))
	for i, operand := range inputs {
		parts[i] = joinDimTokens(operand)
	}
	return strings.Join(parts, ",") + "->" + joinDimTokens(output)
}

func joinDimTokens(tokens []cudashape.DimToken) string {
	var b strings.Builder
	for _, t := range tokens {
		b.WriteString(string(t))
	}
	return b.String()
}

func cloneStringMap(in map[string]string) map[string]string {
	if in == nil {
		return nil
	}
	out := make(map[string]string, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}
