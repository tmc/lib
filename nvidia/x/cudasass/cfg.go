package cudasass

import (
	"cmp"
	"fmt"
	"slices"
	"strings"
)

// A FlowKind names how control passes from one basic block to another.
type FlowKind int

const (
	FallThrough FlowKind = iota // control continues to the next block
	Taken                       // a direct branch transfers to the block
	Call                        // a call enters the block; control returns after the caller
)

func (k FlowKind) String() string {
	return [...]string{"fallthrough", "taken", "call"}[k]
}

// A FlowEdge is a control transfer between two blocks of a [CFG].
type FlowEdge struct {
	From, To int // indices into CFG.Blocks
	Kind     FlowKind
}

// A CFG is the control flow graph of one function: its basic blocks and the
// edges between them. It corresponds to NVBit's CFG_t, whose blocks are the
// same partition and whose is_degenerate flag is IsDegenerate.
type CFG struct {
	Function string
	Blocks   []BasicBlock // exactly BasicBlocks of the function
	Edges    []FlowEdge   // sorted by From, then To, then Kind

	// Indirect lists the blocks that end in an indirect branch (BRX, BRXU,
	// JMX, JMXU), in order. Their targets come from a register at run time,
	// so Edges has no edge for them, only the fall-through of a guarded one.
	Indirect []int

	// IsDegenerate reports that Indirect is not empty: the graph is missing
	// the edges of at least one branch, so it is not the complete set of
	// paths through the function.
	IsDegenerate bool
}

// NewCFG builds the control flow graph of fn over [BasicBlocks].
//
// Edges come from the last instruction of each block. A branch or exit whose
// guard is absent or @PT is unconditional; any other guard, or a predicate
// operand other than PT, also allows the fall-through. A guard of @!PT, or a
// predicate operand of !PT, never branches. A call has a Call edge to its
// target when the target is inside fn, and always falls through.
//
// A direct branch whose target names no block of fn is an error rather than
// a missing edge.
func NewCFG(fn *Function) (*CFG, error) {
	if fn == nil || len(fn.Instructions) == 0 {
		return nil, fmt.Errorf("control flow graph: no instructions")
	}
	g := &CFG{Function: fn.Name, Blocks: BasicBlocks(fn)}
	block := make(map[uint64]int, len(g.Blocks))
	for i, b := range g.Blocks {
		block[b.Start] = i
	}
	for i, b := range g.Blocks {
		add := func(to int, kind FlowKind) {
			if to < len(g.Blocks) {
				g.Edges = append(g.Edges, FlowEdge{From: i, To: to, Kind: kind})
			}
		}
		last := b.Instructions[len(b.Instructions)-1]
		class, _ := Classify(last.Opcode)
		always, never := guard(last.Predicate)
		switch base := BaseOpcode(last.Opcode); {
		case class.Indirect:
			g.Indirect = append(g.Indirect, i)
			if !always {
				add(i+1, FallThrough)
			}
		case base == "BRA" || base == "JMP":
			ops := splitOperands(last.Operands)
			if len(ops) == 0 {
				return nil, fmt.Errorf("control flow graph of %s: %#x: %s has no target", fn.Name, last.Address, last.Opcode)
			}
			if len(ops) > 1 {
				a, n := guard("@" + strings.TrimSpace(ops[0]))
				always, never = always && a, never || n
			}
			tok := ops[len(ops)-1]
			a, ok := fn.target(tok)
			if !ok {
				return nil, fmt.Errorf("control flow graph of %s: %#x: branch target %s does not resolve", fn.Name, last.Address, strings.TrimSpace(tok))
			}
			to, ok := block[a]
			if !ok {
				return nil, fmt.Errorf("control flow graph of %s: %#x: branch target %#x is not a block of the function", fn.Name, last.Address, a)
			}
			if !never {
				add(to, Taken)
			}
			if !always || never {
				add(i+1, FallThrough)
			}
		case base == "EXIT" || base == "RET" || base == "KILL":
			if !always {
				add(i+1, FallThrough)
			}
		case base == "CALL":
			if ops := splitOperands(last.Operands); len(ops) > 0 {
				if a, ok := fn.target(ops[len(ops)-1]); ok {
					if to, ok := block[a]; ok {
						add(to, Call)
					}
				}
			}
			add(i+1, FallThrough)
		default:
			add(i+1, FallThrough)
		}
	}
	slices.SortFunc(g.Edges, func(a, b FlowEdge) int {
		return cmp.Or(cmp.Compare(a.From, b.From), cmp.Compare(a.To, b.To), cmp.Compare(a.Kind, b.Kind))
	})
	g.IsDegenerate = len(g.Indirect) > 0
	return g, nil
}

// guard reports whether a predicate such as "@P0", "@!PT" or "" always holds
// and whether it never does.
func guard(pred string) (always, never bool) {
	switch strings.TrimSpace(pred) {
	case "", "@PT", "@UPT":
		return true, false
	case "@!PT", "@!UPT":
		return false, true
	}
	return false, false
}

// Successors returns the blocks control can reach directly from block b, in
// increasing order and without repeats.
func (g *CFG) Successors(b int) []int {
	var out []int
	for _, e := range g.Edges {
		if e.From == b && !slices.Contains(out, e.To) {
			out = append(out, e.To)
		}
	}
	slices.Sort(out)
	return out
}

// DOT renders g in Graphviz DOT. Each block is a node named BBn, for its index
// in Blocks, labelled with one line per instruction beginning with its offset
// in four or more hex digits, as nvdisasm's --print-instr-offsets-cfg prints
// them. Edges are labelled with their [FlowKind]. A degenerate graph says so
// in the graph label, and its indirect blocks are drawn dashed.
//
// [DisassembleCFG] is nvdisasm's rendering of the same function, and
// [ListDOTGraphs] reads the name of either.
func (g *CFG) DOT() string {
	esc := strings.NewReplacer(`\`, `\\`, `"`, `\"`)
	var b strings.Builder
	fmt.Fprintf(&b, "digraph %s {\n", dotID(g.Function))
	b.WriteString("\tnode [shape=box, fontname=\"monospace\"];\n")
	if g.IsDegenerate {
		names := make([]string, len(g.Indirect))
		for i, n := range g.Indirect {
			names[i] = fmt.Sprintf("BB%d", n)
		}
		fmt.Fprintf(&b, "\tlabel=\"degenerate: indirect branch in %s\";\n", strings.Join(names, ", "))
	}
	for i, blk := range g.Blocks {
		fmt.Fprintf(&b, "\t\"BB%d\" [", i)
		if slices.Contains(g.Indirect, i) {
			b.WriteString("style=dashed, ")
		}
		b.WriteString(`label="`)
		for _, inst := range blk.Instructions {
			text := inst.Opcode
			if inst.Predicate != "" {
				text = inst.Predicate + " " + text
			}
			if inst.Operands != "" {
				text += " " + inst.Operands
			}
			fmt.Fprintf(&b, `%04x: %s ;\l`, inst.Address, esc.Replace(text))
		}
		b.WriteString("\"];\n")
	}
	for _, e := range g.Edges {
		fmt.Fprintf(&b, "\t\"BB%d\" -> \"BB%d\" [label=\"%s\"];\n", e.From, e.To, e.Kind)
	}
	b.WriteString("}\n")
	return b.String()
}
