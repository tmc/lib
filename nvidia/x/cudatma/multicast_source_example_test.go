package cudatma_test

import (
	"fmt"
	"github.com/tmc/lib/nvidia/x/cudatma"
	"strings"
)

func ExampleNewMulticastPlan() {
	d := cudatma.Descriptor{Target: cudatma.TargetSM100a, Rank: 1, Base: 0x1000, Dims: [5]uint64{32}, Box: [5]uint32{32}, ElementBytes: 4}
	c := cudatma.MulticastConfig{Descriptor: d, Cluster: [3]uint32{2, 1, 1}, Destinations: []uint32{0, 1}, AvailableSharedBytes: 136, SharedLimitProvenance: "requested fixture allocation"}
	for rank := uint32(0); rank < 2; rank++ {
		c.Windows = append(c.Windows, cudatma.SharedWindow{Rank: rank, Descriptor: d, BarrierOffset: 128, ReservedBytes: 136})
	}
	p, err := cudatma.NewMulticastPlan(c)
	if err != nil {
		fmt.Println(err)
		return
	}
	s, _ := p.Source()
	r, _ := p.Requirements()
	snapshot, _ := p.Descriptor()
	fmt.Println(strings.Contains(s, ".multicast::cluster"), r.CTAs, r.TileBytes, r.SharedBytes, snapshot.Rank)
	// Output: true 2 128 136 1
}

func examplePlan() *cudatma.MulticastPlan {
	d := cudatma.Descriptor{Target: cudatma.TargetSM100a, Rank: 1, Base: 0x1000, Dims: [5]uint64{32}, Box: [5]uint32{32}, ElementBytes: 4}
	c := cudatma.MulticastConfig{Descriptor: d, Cluster: [3]uint32{1, 1, 1}, Destinations: []uint32{0}, AvailableSharedBytes: 136, SharedLimitProvenance: "requested fixture allocation", Windows: []cudatma.SharedWindow{{Rank: 0, Descriptor: d, BarrierOffset: 128, ReservedBytes: 136}}}
	p, _ := cudatma.NewMulticastPlan(c)
	return p
}

func ExampleMulticastPlan_Source() {
	s, err := examplePlan().Source()
	fmt.Println(err, strings.Contains(s, ".multicast::cluster"))
	// Output: <nil> true
}

func ExampleMulticastPlan_Requirements() {
	r, err := examplePlan().Requirements()
	fmt.Println(err, r.Cluster, r.Mask, r.SharedBytes)
	// Output: <nil> [1 1 1] 1 136
}

func ExampleMulticastPlan_Descriptor() {
	d, err := examplePlan().Descriptor()
	fmt.Println(err, d.Rank, d.ElementBytes)
	// Output: <nil> 1 4
}
