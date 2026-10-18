package cudatmem_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudagroup"
	"github.com/tmc/lib/nvidia/x/cudatmem"
)

func exampleCollective() (*cudatmem.Allocator, []cudagroup.Issuer, error) {
	one := driver.Dim3{X: 1, Y: 1, Z: 1}
	block := driver.Dim3{X: 32, Y: 1, Z: 1}
	cluster, err := cudagroup.New(cudagroup.Cluster, one)
	if err != nil {
		return nil, nil, err
	}
	c, err := cudagroup.NewTMEMCollective("sm_100a", cudagroup.TMEMGroup1, cluster, block, 0, 0)
	if err != nil {
		return nil, nil, err
	}
	// Cluster remains zero: this describes an ordinary one-CTA launch.
	a, err := cudatmem.NewCollective(cudatmem.SM100A, c, cudagroup.LaunchRequirements{Grid: one, Block: block})
	return a, issuingWarp(0), err
}

func ExampleNewCollective() {
	a, _, err := exampleCollective()
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(a.Capability().MaxRows, a.Capability().ColumnUnit)
	fmt.Println(a.Close())
	// Output:
	// 128 32
	// <nil>
}

func ExampleAllocator_AcquireCollectivePermit() {
	a, ids, err := exampleCollective()
	if err != nil {
		fmt.Println(err)
		return
	}
	p, err := a.AcquireCollectivePermit(ids)
	fmt.Println(p != nil, err)
	fmt.Println(a.RelinquishCollective(p, ids), a.Close())
	// Output:
	// true <nil>
	// <nil> <nil>
}

func ExampleAllocator_AllocateCollective() {
	a, ids, err := exampleCollective()
	if err != nil {
		fmt.Println(err)
		return
	}
	p, err := a.AcquireCollectivePermit(ids)
	if err != nil {
		fmt.Println(err)
		return
	}
	x, err := a.AllocateCollective(p, ids, columnArgs(32), 32)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(x.Columns(), x.Rows())
	fmt.Println(a.ReleaseCollective(p, x, ids, columnArgs(32)))
	fmt.Println(a.RelinquishCollective(p, ids), a.Close())
	// Output:
	// 32 32
	// <nil>
	// <nil> <nil>
}

func exampleAllocation() (*cudatmem.Allocator, []cudagroup.Issuer, *cudatmem.Permit, *cudatmem.Allocation, error) {
	a, ids, err := exampleCollective()
	if err != nil {
		return nil, nil, nil, nil, err
	}
	p, err := a.AcquireCollectivePermit(ids)
	if err != nil {
		return a, ids, nil, nil, err
	}
	x, err := a.AllocateCollective(p, ids, columnArgs(32), 32)
	return a, ids, p, x, err
}

func ExampleAllocator_ReleaseCollective() {
	a, ids, p, x, err := exampleAllocation()
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(a.RelinquishCollective(p, ids))
	// Relinquishment forbids new allocations but retains owning deallocation.
	fmt.Println(a.ReleaseCollective(p, x, ids, columnArgs(32)), a.Close())
	// Output:
	// <nil>
	// <nil> <nil>
}

func ExampleAllocator_RelinquishCollective() {
	a, ids, err := exampleCollective()
	if err != nil {
		fmt.Println(err)
		return
	}
	p, err := a.AcquireCollectivePermit(ids)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(a.RelinquishCollective(p, ids))
	_, err = a.AcquireCollectivePermit(ids)
	fmt.Println(err)
	fmt.Println(a.Close())
	// Output:
	// <nil>
	// cudatmem: allocation permission required
	// <nil>
}
