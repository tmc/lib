//go:build !windows

package moonep_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/moonep"
)

func ExampleSymmBufferManager() {
	mgr := moonep.NewSymmBufferManager()
	planned, err := mgr.PlanAllocations([]moonep.SymmBufferKey{
		{GroupName: "ep_decode", MaxBytes: 16384, AlignBytes: 256},
		{GroupName: "ep_prefill", MaxBytes: 65536, AlignBytes: 256},
	})
	if err != nil {
		panic(err)
	}
	view, err := mgr.GetSymmView("ep_decode", []int{32, 128}, 2)
	if err != nil {
		panic(err)
	}
	comm, comp, err := moonep.PartitionSMBudget(148, 24)
	if err != nil {
		panic(err)
	}
	fmt.Printf("largest=%s:%d viewOffset=%d sm=%d/%d\n",
		planned[0].GroupName, planned[0].MaxBytes, view.ByteOffset, comm, comp)
	// Output:
	// largest=ep_prefill:65536 viewOffset=0 sm=24/124
}
