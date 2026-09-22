package cudalineinfo

import (
	"bufio"
	"fmt"
	"io"
	"sort"
	"strconv"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudasass"
)

// An Edge is one step of the path from a program's code to the device's.
type Edge struct{ From, To string }

// Edges returns the graph of a unit as caller-to-callee edges, without
// duplicates and in sorted order. Program frames point at the frames they
// called, down to the statement that built each site; a statement's function
// points at the site; the site points at the kernel it was lowered into and
// at any subroutine an instruction on it calls; and the kernel points at
// subroutines called from instructions that name no site.
//
// The frames come from the locations the unit recorded, so the graph is of
// the paths that built the unit, not of every path the program has. It is
// written by [WriteDigraph] in the format golang.org/x/tools/cmd/digraph
// reads, so that tool's queries apply: somepath from a function to the
// division slowpath answers which line pulled it in.
func (a *Attribution) Edges() []Edge {
	seen := map[Edge]bool{}
	var edges []Edge
	add := func(from, to string) {
		if from == "" || to == "" || from == to {
			return
		}
		e := Edge{from, to}
		if !seen[e] {
			seen[e] = true
			edges = append(edges, e)
		}
	}
	kernel := a.Unit.Name
	for _, s := range a.Sites() {
		loc, _ := a.Unit.Location(s)
		callee := loc.Function
		for _, c := range loc.Callers {
			add(c.Function, callee)
			callee = c.Function
		}
		add(loc.Function, s.String())
		add(s.String(), kernel)
		for _, in := range a.SASS[s] {
			add(s.String(), callTarget(in))
		}
	}
	if a.Disassembly != nil {
		for _, fn := range a.Disassembly.Functions {
			for _, in := range fn.Instructions {
				if _, ok := a.SiteAt(fn.Name, in.Address); ok {
					continue
				}
				add(fn.SymbolAt(in.Address), callTarget(in))
			}
		}
	}
	sort.Slice(edges, func(i, j int) bool {
		if edges[i].From != edges[j].From {
			return edges[i].From < edges[j].From
		}
		return edges[i].To < edges[j].To
	})
	return edges
}

// callTarget returns the function a CALL instruction's operand names, as
// nvdisasm prints it: `($__internal_0_$__cuda_sm3x_div_rn_noftz_f32_slowpath).
// A RET names its caller the same way and is not a call.
func callTarget(in cudasass.Instruction) string {
	if cudasass.BaseOpcode(in.Opcode) != "CALL" {
		return ""
	}
	_, rest, ok := strings.Cut(in.Operands, "`(")
	if !ok {
		return ""
	}
	name, _, ok := strings.Cut(rest, ")")
	if !ok || strings.HasPrefix(name, ".L_") {
		return ""
	}
	return name
}

// WriteDigraph writes the edges as golang.org/x/tools/cmd/digraph reads
// them: one line per node, the node followed by its successors.
func WriteDigraph(w io.Writer, a *Attribution) error {
	bw := bufio.NewWriter(w)
	var from string
	for _, e := range a.Edges() {
		if e.From != from {
			if from != "" {
				bw.WriteByte('\n')
			}
			from = e.From
			bw.WriteString(digraphWord(from))
		}
		bw.WriteByte(' ')
		bw.WriteString(digraphWord(e.To))
	}
	if from != "" {
		bw.WriteByte('\n')
	}
	return bw.Flush()
}

// digraphWord quotes a node name that whitespace or quotes would split.
func digraphWord(s string) string {
	if strings.ContainsAny(s, " \t\"") {
		return strconv.Quote(s)
	}
	return s
}

// WriteDot writes the graph in Graphviz format. Sites carry their static
// instruction counts, and subroutines the instructions they hold.
func WriteDot(w io.Writer, a *Attribution) error {
	bw := bufio.NewWriter(w)
	fmt.Fprintf(bw, "digraph %s {\n\trankdir=LR;\n\tnode [shape=box, fontname=\"monospace\"];\n", strconv.Quote(a.Unit.Name))
	for _, s := range a.Sites() {
		fmt.Fprintf(bw, "\t%s [label=%s];\n", strconv.Quote(s.String()), strconv.Quote(fmt.Sprintf("%s\n%d SASS", s, len(a.SASS[s]))))
	}
	fmt.Fprintf(bw, "\t%s [label=%s, shape=component];\n", strconv.Quote(a.Unit.Name), strconv.Quote(fmt.Sprintf("%s\n%d SASS", a.Unit.Name, a.Instructions-a.Remainder.subroutines())))
	for _, name := range sortedKeys(a.Remainder.Subroutines) {
		fmt.Fprintf(bw, "\t%s [label=%s, shape=component];\n", strconv.Quote(name), strconv.Quote(fmt.Sprintf("%s\n%d SASS", name, a.Remainder.Subroutines[name])))
	}
	for _, e := range a.Edges() {
		fmt.Fprintf(bw, "\t%s -> %s;\n", strconv.Quote(e.From), strconv.Quote(e.To))
	}
	fmt.Fprintln(bw, "}")
	return bw.Flush()
}

func (r Remainder) subroutines() int {
	n := 0
	for _, v := range r.Subroutines {
		n += v
	}
	return n
}
