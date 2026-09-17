package cudatma_test

import (
	"fmt"
	"github.com/tmc/lib/nvidia/x/cudatma"
)

func Example() {
	m, _ := cudatma.NewMulticast(0, []uint32{1, 2}, 128)
	fmt.Println(m.Mask, m.CompleteAll())
	// Output: 6 false
}
