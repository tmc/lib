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
	x, err := a.Allocate(p, 32, 64, 1)
	if err != nil {
		fmt.Println(err)
		return
	}
	y, err := a.Allocate(p, 32, 32, 1)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(x.Columns(), y.Columns())

	// Relinquishing ends allocation, not ownership of live allocations.
	fmt.Println(a.Relinquish(p))
	_, err = a.Allocate(p, 32, 32, 1)
	fmt.Println(err)
	fmt.Println(a.Release(p, x, 32), a.Release(p, y, 32), a.Close())
	// Output:
	// 64 32
	// <nil>
	// cudatmem: allocation permission required: permit relinquished
	// <nil> <nil> <nil>
}
