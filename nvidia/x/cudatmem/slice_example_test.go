package cudatmem_test

import (
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudalayout"
	"github.com/tmc/lib/nvidia/x/cudatmem"
)

func ExampleAllocation_Slice() {
	a, ids, p, x, err := exampleAllocation()
	if err != nil {
		fmt.Println(err)
		return
	}
	// The hardware allocation is 32 columns; a logical view can be narrower.
	s, err := x.Slice(0, 4, 32, 2)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(s.Address(31, 1))
	fmt.Println(a.ReleaseCollective(p, x, ids, columnArgs(32)))
	_, err = s.Address(0, 0)
	fmt.Println(errors.Is(err, cudatmem.ErrClosed))
	fmt.Println(a.RelinquishCollective(p, ids), a.Close())
	// Output:
	// {31 5} <nil>
	// <nil>
	// true
	// <nil> <nil>
}

func ExampleSlice_Sub() {
	a, ids, p, x, err := exampleAllocation()
	if err != nil {
		fmt.Println(err)
		return
	}
	s, err := x.Slice(0, 0, 32, 32)
	if err != nil {
		fmt.Println(err)
		return
	}
	child, err := s.Sub(0, 7, 32, 1)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(child.Address(31, 0))
	fmt.Println(a.ReleaseCollective(p, x, ids, columnArgs(32)))
	fmt.Println(a.RelinquishCollective(p, ids), a.Close())
	// Output:
	// {31 7} <nil>
	// <nil>
	// <nil> <nil>
}

func ExampleSlice_Address() {
	a, ids, p, x, err := exampleAllocation()
	if err != nil {
		fmt.Println(err)
		return
	}
	s, err := x.Slice(cudatmem.Row(0), cudatmem.Column(3), 32, 2)
	if err != nil {
		fmt.Println(err)
		return
	}
	addr, err := s.Address(2, 1)
	fmt.Println(addr.Row, addr.Column, err) // coordinates, not native pointer bits
	fmt.Println(a.ReleaseCollective(p, x, ids, columnArgs(32)))
	fmt.Println(a.RelinquishCollective(p, ids), a.Close())
	// Output:
	// 2 4 <nil>
	// <nil>
	// <nil> <nil>
}

func ExampleSlice_BeginAccess() {
	a, ids, p, x, err := exampleAllocation()
	if err != nil {
		fmt.Println(err)
		return
	}
	s, err := x.Slice(0, 0, 32, 2)
	if err != nil {
		fmt.Println(err)
		return
	}
	layout, err := cudalayout.NewTile(32, 2, 4)
	if err != nil {
		fmt.Println(err)
		return
	}
	base, err := s.Address(0, 0)
	if err != nil {
		fmt.Println(err)
		return
	}
	bases := make([]cudatmem.Address, 32)
	for i := range bases {
		bases[i] = base
	}
	use, err := s.BeginAccess(0, layout, ids, bases)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(errors.Is(a.ReleaseCollective(p, x, ids, columnArgs(32)), cudatmem.ErrBusy))
	// A real caller waits for its operation before declaring completion.
	fmt.Println(use.Complete())
	fmt.Println(a.ReleaseCollective(p, x, ids, columnArgs(32)))
	fmt.Println(a.RelinquishCollective(p, ids), a.Close())
	// Output:
	// true
	// <nil>
	// <nil>
	// <nil> <nil>
}

func ExampleAccess_Complete() {
	a, ids, p, x, err := exampleAllocation()
	if err != nil {
		fmt.Println(err)
		return
	}
	s, err := x.Slice(0, 0, 32, 1)
	if err != nil {
		fmt.Println(err)
		return
	}
	layout, err := cudalayout.NewTile(32, 1, 4)
	if err != nil {
		fmt.Println(err)
		return
	}
	base, err := s.Address(0, 0)
	if err != nil {
		fmt.Println(err)
		return
	}
	bases := make([]cudatmem.Address, 32)
	for i := range bases {
		bases[i] = base
	}
	use, err := s.BeginAccess(0, layout, ids, bases)
	if err != nil {
		fmt.Println(err)
		return
	}
	alias := *use
	fmt.Println(use.Complete(), alias.Complete())
	fmt.Println(a.ReleaseCollective(p, x, ids, columnArgs(32)))
	fmt.Println(a.RelinquishCollective(p, ids), a.Close())
	// Output:
	// <nil> <nil>
	// <nil>
	// <nil> <nil>
}
