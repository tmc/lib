package cudadiag_test

import (
	"bytes"
	"context"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudadiag"
	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func ExampleSummarizeServing() {
	c := newServingCase()
	// The second attempt's batch membership lost its request identity.
	c.serving.Batches[0].Members[1].RequestID = ""
	var b bytes.Buffer
	if err := nvtelemetry.WriteServingContextV2(&b, c.serving, nvtelemetry.WorkloadLimits{}); err != nil {
		fmt.Println(err)
		return
	}
	movement, err := c.movement.build()
	if err != nil {
		fmt.Println(err)
		return
	}
	s, err := cudadiag.SummarizeServing(context.Background(), cudadiag.ServingInput{
		Serving:  &b,
		Source:   nvtelemetry.WorkloadSource{Provider: "application", Version: "serving-app-1", RunID: "run"},
		Movement: &movement,
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	for _, a := range s.Attempts {
		fmt.Println(a.Attempt.RequestID, a.Outcome, a.Batches, a.Transfers)
	}
	for _, e := range s.Edges {
		fmt.Println(e.Kind, e.Target, e.Finding.Class)
	}
	// Output:
	// r1 accepted [b1] [t1]
	// r2 failed [] [t2]
	// route fe-0 observed
	// batch_member b1 associated
	// batch_member b1 unavailable
	// batch_operation b1/9 associated
	// cache_use kv/g1 associated
	// transfer t1 associated
	// transfer t2 associated
}
