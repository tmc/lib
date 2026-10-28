package cudadiag_test

import (
	"context"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudadiag"
)

func ExampleSummarizeMovement() {
	m := newMovementCase()
	// Swap the destination peer: the path claim loses its topology edge.
	// The peer copy has no supplied operation, so no activity link is made.
	m.context.Transfers[0].Destination.Entity = "GPU-2"
	in, err := m.build()
	if err != nil {
		fmt.Println(err)
		return
	}
	s, err := cudadiag.SummarizeMovement(context.Background(), in)
	if err != nil {
		fmt.Println(err)
		return
	}
	for _, f := range s.Transfers[0].Findings {
		fmt.Println(f.Rule, f.Class)
	}
	fmt.Println(*s.Transfers[0].Bytes)
	// Output:
	// movement_bytes associated
	// movement_direction associated
	// movement_completion associated
	// movement_duration associated
	// movement_path unavailable
	// movement_activity_link unavailable
	// movement_attempt_link associated
	// movement_congestion unavailable
	// 9007199254740993
}
