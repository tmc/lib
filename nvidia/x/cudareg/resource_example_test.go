package cudareg_test

import (
	"fmt"
	"github.com/tmc/lib/nvidia/x/cudareg"
)

func ExamplePublishCandidate() {
	r := registerResources()
	for _, split := range [][2]uint32{{32, 96}, {40, 88}} {
		s, _, _ := cudareg.GenerateFixture()
		s.Steps[0].Count, s.Steps[1].Count = split[0], split[1]
		c, err := cudareg.PublishCandidate(&r, s)
		if err != nil {
			fmt.Println(err)
			return
		}
		fmt.Println(c.Status, split, c.Resources.Allocated.Count, c.Resources.PeakLive.Status)
	}
	// Output:
	// protocol feasible [32 96] 64 unavailable
	// protocol feasible [40 88] 64 unavailable
}

func ExampleResources_Validate() {
	r := registerResources()
	fmt.Println(r.Validate()) // Output: <nil>
}
func ExampleCandidate() {
	s, _, _ := cudareg.GenerateFixture()
	c, _ := cudareg.PublishCandidate(nil, s)
	fmt.Println(c.Status, c.Resources == nil) // Output: unverified true
}
func ExampleFact() {
	f := cudareg.Fact{Status: cudareg.FactUnavailable, Reason: "not recorded"}
	fmt.Println(f.Status, f.Reason) // Output: unavailable not recorded
}
func ExampleFactStatus() {
	fmt.Println(cudareg.FactUnavailable, cudareg.FactUnverified, cudareg.FactAvailable) // Output: unavailable unverified available
}
func ExamplePublicationStatus() {
	fmt.Println(cudareg.PublicationUnverified, cudareg.ProtocolFeasible) // Output: unverified protocol feasible
}
