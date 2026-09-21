package cudafuse

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudair"
)

// TestLineMapIndexesTheRightLine gives every node a distinct source line and
// checks each mapping lands on a generated line that mentions that node. A
// chunk holding an embedded newline once shifted every later mapping by one,
// which a test comparing only line counts cannot see.
func TestLineMapIndexesTheRightLine(t *testing.T) {
	g := cudair.NewGraph()
	names := map[int]string{}
	mark := func(n *cudair.Node, line int) *cudair.Node {
		n.SourceLoc = cudair.SourceLocation{File: "m.go", Line: line}
		names[line] = n.Name
		return n
	}
	x := mark(g.AddInput("x", []int{8}, cudair.Float32), 10)
	y := mark(g.AddInput("y", []int{8}, cudair.Float32), 20)
	s := mark(g.AddBinary("add", x, y), 30)
	mark(g.AddUnary("relu", s), 40)

	fk, err := FuseWithSourceMap(g, "k")
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(fk.Source, "\n")
	if len(fk.LineMap) == 0 {
		t.Fatal("no mappings recorded")
	}
	for n, loc := range fk.LineMap {
		if n < 1 || n > len(lines) {
			t.Errorf("mapping for %v indexes line %d, outside 1..%d", loc, n, len(lines))
			continue
		}
		name, ok := names[loc.Line]
		if !ok {
			t.Errorf("line %d mapped to unexpected %v", n, loc)
			continue
		}
		if !strings.Contains(lines[n-1], name) {
			t.Errorf("line %d maps to %v (node %q) but reads %q", n, loc, name, lines[n-1])
		}
	}
}
