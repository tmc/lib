package hipmega_test

import (
	"fmt"

	"github.com/tmc/lib/rocm/driver"
	"github.com/tmc/lib/rocm/x/hipmega"
)

func Example() {
	// Host callbacks model the contract. They do not execute HIP or a kernel.
	ops := hipmega.Operations{
		Limits: func(driver.Function) (hipmega.Limits, error) {
			return hipmega.Limits{
				Device: 0, CooperativeLaunch: true, Multiprocessors: 2,
				MaxGrid: [3]int64{1024, 1, 1}, MaxBlock: [3]int64{1024, 1, 1},
				MaxThreadsPerBlock: 1024, MaxSharedMemPerBlock: 65536,
			}, nil
		},
		Occupancy: func(driver.Function, int32, uint) (int32, error) { return 2, nil },
		CooperativeLaunch: func(r hipmega.Request) error {
			fmt.Println("host launch callback:", r.Grid)
			return nil
		},
	}
	a, err := hipmega.Launch(ops, hipmega.Request{
		Function: 1, Grid: [3]int64{4, 1, 1}, Block: [3]int64{64, 1, 1},
		Progress: hipmega.CoResident,
	})
	fmt.Println(a.ResidentBlocks, err)
	// Output:
	// host launch callback: [4 1 1]
	// 4 <nil>
}

func ExampleHIPOperations() {
	// Construction makes no HIP calls. Do not invoke these operations without
	// an initialized runtime and the appropriate current context/OS thread.
	ops := hipmega.HIPOperations()
	fmt.Println(ops.Limits != nil && ops.Occupancy != nil && ops.CooperativeLaunch != nil)
	// Output: true
}
