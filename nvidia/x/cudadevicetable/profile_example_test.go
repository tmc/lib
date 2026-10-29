package cudadevicetable_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudadevicetable"
)

func ExampleLookupProfile() {
	p, err := cudadevicetable.LookupProfile("sm_103a")
	if err != nil {
		panic(err)
	}
	tcgen05, _ := p.Supports("tcgen05")
	fmt.Println(p.Product, p.SMs.Value, p.SMs.Source)
	fmt.Println(p.SharedPerBlockOptin.Value, tcgen05)
	// Output:
	// GB300 160 unverified
	// 232448 true
}
