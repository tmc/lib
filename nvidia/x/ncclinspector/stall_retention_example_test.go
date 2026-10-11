package ncclinspector_test

import (
	"fmt"
	"github.com/tmc/lib/nvidia/x/ncclinspector"
)

func ExampleTwoRankAttributor_Forget() {
	a := ncclinspector.NewTwoRankAttributor()
	obs := ncclinspector.CollectiveObservation{CommunicatorHash: "0x1234567890abcdef", Rank: 0, TotalRanks: 2, HostPID: 1, Sequence: 1, Operation: "AllReduce"}
	if err := a.AddObservation(obs); err != nil {
		fmt.Println(err)
		return
	}
	a.Forget(obs.CommunicatorHash, obs.Sequence)
	_, err := a.Attribute(obs.CommunicatorHash, obs.Sequence)
	fmt.Println(err != nil)
	// Output: true
}
