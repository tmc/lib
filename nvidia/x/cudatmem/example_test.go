package cudatmem_test

import (
	"fmt"
	"github.com/tmc/lib/nvidia/x/cudatmem"
)

func Example() {
	a, err := cudatmem.New(cudatmem.SM100A)
	if err != nil {
		fmt.Println(err)
		return
	}
	p, err := a.AcquirePermit(32)
	if err != nil {
		fmt.Println(err)
		return
	}
	x, err := a.Allocate(p, 32, 32, 1)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(x.Columns(), x.Rows())
	_ = a.Release(p, x, 32)
	_ = a.Relinquish(p)
	_ = a.Close()
	// Output: 32 1
}
