package cudacluster_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudacluster"
	"github.com/tmc/lib/nvidia/x/cudashared"
)

// Example shows the all-to-one shape: every CTA publishes its rank into its
// own window, the cluster synchronizes, and rank 0 reads each peer in turn.
// The synchronization is written by the caller because only the caller knows
// the dependency it is enforcing.
func Example() {
	limits := cudashared.Limits{StaticMax: 49152, DynamicMax: 102400, TotalMax: 102400, Source: "example target"}
	plan, err := cudashared.NewPlan(limits)
	if err != nil {
		panic(err)
	}
	window, err := plan.Add(cudashared.Request{Name: "window", Size: 4, Alignment: 4})
	if err != nil {
		panic(err)
	}

	c, err := cudacluster.Open("sm_121a")
	if err != nil {
		panic(err)
	}
	fmt.Println(cudacluster.EmitBlockRank("%r1"))
	for _, s := range cudacluster.EmitClusterSync() {
		fmt.Println(s)
	}
	addr, err := cudacluster.EmitSharedAddr("%r9", "smem", plan, window)
	if err != nil {
		panic(err)
	}
	fmt.Println(addr)
	read, err := c.EmitReadU32("%r4", "%r3", "%r9", plan, window, "%r2")
	if err != nil {
		panic(err)
	}
	for _, s := range read {
		fmt.Println(s)
	}

	// Output:
	// mov.u32 %r1, %cluster_ctarank;
	// barrier.cluster.arrive;
	// barrier.cluster.wait;
	// mov.u32 %r9, smem;
	// mapa.shared::cluster.u32 %r3, %r9, %r2;
	// ld.shared::cluster.u32 %r4, [%r3];
}

// ExampleCheckDSMEM shows that a target with clusters is not thereby a target
// with a verified remote window.
func ExampleCheckDSMEM() {
	fmt.Println(cudacluster.Check("sm_121a"))
	fmt.Println(cudacluster.CheckDSMEM("sm_121a") != nil)

	// Output:
	// <nil>
	// true
}
