package dsv4_test

import (
	"fmt"
	"github.com/tmc/lib/examples/ds4-min/dsv4"
)

func ExampleContextMemory_Total() {
	m := dsv4.ContextMemory{Raw: 32, Compressed: 64, Scratch: 16}
	n, err := m.Total()
	fmt.Println(n, err)
	// Output: 112 <nil>
}
