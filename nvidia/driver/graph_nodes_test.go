package driver

import (
	"fmt"
	"testing"
	"unsafe"
)

func TestGraphNodesBounded(t *testing.T) {
	calls := 0
	got, err := graphNodes(1, 3, func(_ Graph, p *GraphNode, n *uint) Result {
		calls++
		if p == nil {
			*n = 3
			return Success
		}
		copy(unsafe.Slice(p, int(*n)), []GraphNode{9, 4, 7})
		return Success
	})
	if err != nil || len(got) != 3 || got[2] != 7 || calls != 2 {
		t.Fatal(got, err, calls)
	}
	calls = 0
	_, err = graphNodes(1, 2, func(_ Graph, p *GraphNode, n *uint) Result { calls++; *n = 3; return Success })
	if err == nil || calls != 1 {
		t.Fatal("unbounded allocation", err, calls)
	}
	_, err = graphNodes(1, 3, func(_ Graph, p *GraphNode, n *uint) Result {
		if p == nil {
			*n = 3
		} else {
			*n = 4
		}
		return Success
	})
	if err == nil {
		t.Fatal("changed count accepted")
	}
}

func ExampleGraph_NodesBounded() {
	_, err := Graph(0).NodesBounded(0)
	fmt.Println(err)
	// Output: driver: invalid graph node limit 0
}
