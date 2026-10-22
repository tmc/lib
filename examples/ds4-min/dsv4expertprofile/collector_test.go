package dsv4expertprofile

import (
	"bytes"
	"fmt"
	"testing"
)

func TestLocalityValuesAndMutation(t *testing.T) {
	c, err := New(1, 4, []int{1, 2, 4})
	if err != nil {
		t.Fatal(err)
	}
	for pos, ids := range [][]int{{0, 1}, {1, 2}, {0, 2}} {
		if err := c.Observe(0, pos, ids, []float32{0.25, 0.75}); err != nil {
			t.Fatal(err)
		}
	}
	l := c.layers[0]
	if l.Records != 3 || l.AdjacentPairs != 2 || l.OverlapSum != 1 || l.JaccardSum != 2.0/3 || l.Hits[0] != 2 || l.Weights[0] != 0.5 {
		t.Fatalf("wrong values: %+v", l)
	}
	if l.CacheHits[0] != 1 || l.CacheHits[2] != 3 {
		t.Fatalf("wrong LRU hits: %v", l.CacheHits)
	}
	var before, after bytes.Buffer
	_ = c.Write(&before)
	if err := c.Observe(0, 3, []int{1, 1}, []float32{1, 1}); err == nil {
		t.Fatal("duplicate accepted")
	}
	_ = c.Write(&after)
	if !bytes.Equal(before.Bytes(), after.Bytes()) {
		t.Fatal("failed admission mutated profile")
	}
	m, _ := New(1, 4, nil)
	_ = m.Observe(0, 0, []int{0, 1}, []float32{0.25, 0.75})
	_ = m.Observe(0, 2, []int{1, 2}, []float32{0.25, 0.75})
	if m.layers[0].AdjacentPairs != 0 {
		t.Fatal("position-gap mutation escaped oracle")
	}
}

func ExampleCollector() {
	c, _ := New(1, 4, []int{2})
	_ = c.Observe(0, 0, []int{1}, []float32{1})
	var out bytes.Buffer
	_ = c.Write(&out)
	fmt.Println(out.Len() > 0)
	// Output: true
}
