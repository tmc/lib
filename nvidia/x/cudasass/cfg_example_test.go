package cudasass_test

import (
	"fmt"
	"log"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudasass"
)

const loopSASS = `
.section .text.loop,"ax",@progbits
.text.loop:
        /*0000*/                   ISETP.GE.AND P0, PT, R0, 0x10, PT ;
.L_x_0:
        /*0010*/                   IADD R0, R0, 0x1 ;
        /*0020*/               @!P0 BRA ` + "`" + `(.L_x_0) ;
        /*0030*/                   EXIT ;
`

func ExampleNewCFG() {
	d, err := cudasass.ParseString(loopSASS)
	if err != nil {
		log.Fatal(err)
	}
	g, err := cudasass.NewCFG(d.Function("loop"))
	if err != nil {
		log.Fatal(err)
	}
	for i, b := range g.Blocks {
		fmt.Printf("BB%d %#x-%#x successors %v\n", i, b.Start, b.End, g.Successors(i))
	}
	fmt.Println("degenerate:", g.IsDegenerate)
	// Output:
	// BB0 0x0-0x10 successors [1]
	// BB1 0x10-0x30 successors [1 2]
	// BB2 0x30-0x40 successors []
	// degenerate: false
}

func ExampleCFG_DOT() {
	d, err := cudasass.ParseString(loopSASS)
	if err != nil {
		log.Fatal(err)
	}
	g, err := cudasass.NewCFG(d.Function("loop"))
	if err != nil {
		log.Fatal(err)
	}
	for _, line := range strings.Split(g.DOT(), "\n") {
		if strings.Contains(line, "->") {
			fmt.Println(strings.TrimSpace(line))
		}
	}
	// Output:
	// "BB0" -> "BB1" [label="fallthrough"];
	// "BB1" -> "BB1" [label="taken"];
	// "BB1" -> "BB2" [label="fallthrough"];
}

func ExampleEdgeIndex() {
	// The same two blocks in opposite orders are different edges.
	_, prev := cudasass.EdgeIndex(0, 5)
	ab, _ := cudasass.EdgeIndex(prev, 9)
	_, prev = cudasass.EdgeIndex(0, 9)
	ba, _ := cudasass.EdgeIndex(prev, 5)
	fmt.Println(ab, ba)
	// Output: 32779 32769
}

func ExampleEdgeCoverage() {
	var c cudasass.EdgeCoverage
	for thread := range uint64(3) {
		c.Record(thread, 5)
		c.Record(thread, 9)
	}
	i, _ := cudasass.EdgeIndex(2, 9)
	fmt.Println(c.Count(i), c.Bitmap()[i])
	// Output: 3 3
}
