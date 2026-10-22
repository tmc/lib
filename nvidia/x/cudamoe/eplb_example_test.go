// SPDX-License-Identifier: Apache-2.0

package cudamoe_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudamoe"
)

func ExampleRebalanceExpertsGlobal() {
	numPhys, err := cudamoe.ResolveNumPhysicalExperts(4, 2, 2)
	if err != nil {
		panic(err)
	}
	placement, err := cudamoe.RebalanceExpertsGlobal(
		[]float64{90, 30, 20, 10},
		cudamoe.EPLBConfig{
			NumRoutedExperts:   4,
			NumPhysicalExperts: numPhys,
			EPSize:             2,
		},
	)
	if err != nil {
		panic(err)
	}
	fmt.Printf("physical=%d replicas=%v rankLoads=[%.0f %.0f]\n",
		len(placement.PhysicalToLogical),
		placement.ExpertReplicaCount,
		placement.RankLoads[0],
		placement.RankLoads[1],
	)
	// Output:
	// physical=6 replicas=[3 1 1 1] rankLoads=[80 70]
}

func ExamplePlanExpertShuffle() {
	oldMap := []int32{0, 1, 2, 3}
	newMap := []int32{1, 2, 2, 0}
	plan, err := cudamoe.PlanExpertShuffle(oldMap, newMap, 2)
	if err != nil {
		panic(err)
	}
	fmt.Printf("local=%d waves=%d fills=%d updated=%v\n",
		len(plan.LocalCopies),
		len(plan.P2PWaves),
		len(plan.ReplicaFills),
		plan.UpdatedMapping,
	)
	// Output:
	// local=1 waves=1 fills=0 updated=[1 2 2 0]
}
