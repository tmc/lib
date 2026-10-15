package egraph_test

import (
	"context"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudacollective"
	"github.com/tmc/lib/nvidia/x/egraph"
)

// ExampleCollectiveSchedule_Plan lowers one two-rank allreduce and replays it
// on host inputs.
func ExampleCollectiveSchedule_Plan() {
	m, _ := cudacollective.NewMembership([]cudacollective.Rank{{ID: 0}, {ID: 1, DeviceID: 1}})
	tr := cudacollective.NewDeterministicTestTransport(2)
	var s egraph.CollectiveSchedule
	s.Quarantine = new(egraph.CollectiveQuarantine)
	for r := range 2 {
		comm, err := cudacollective.NewCommunicator(cudacollective.CommunicatorConfig{Rank: r, Membership: m, Generation: 1, Transport: tr})
		if err != nil {
			fmt.Println(err)
			return
		}
		defer comm.Close()
		s.Communicators = append(s.Communicators, comm)
		buf := egraph.CollectiveBuffer{Address: uint64(0x1000 * (r + 1)), Bytes: 12}
		s.Calls = append(s.Calls, []egraph.CollectiveCall{{
			CollectiveIntent: egraph.CollectiveIntent{
				Communicator: "tp", Generation: 1, Rank: r, WorldSize: 2,
				Operation: "allreduce", DType: "f32", Count: 3, InPlace: true, EffectToken: 1,
			},
			Reduction: "sum", Layout: "contiguous", Partition: 1,
			Send: buf, Recv: buf,
			Completion:  1,
			StreamState: egraph.CollectiveStreamOrdered,
		}})
	}
	p, err := s.Plan()
	if err != nil {
		fmt.Println(err)
		return
	}
	defer p.Close()
	rec, err := p.Replay(context.Background(), [][][]float32{{{1, 2, 3}, {10, 20, 30}}})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(rec.Transport, rec.Steps[0].Outputs)
	// Output:
	// controlplanetest [[11 22 33] [11 22 33]]
}
