package cudashared_test

import (
	"fmt"
	"github.com/tmc/lib/nvidia/x/cudashared"
)

func Example() {
	p, err := cudashared.NewPlan(cudashared.Limits{StaticMax: 49152, DynamicMax: 101376, TotalMax: 101376, Source: "sm_121a device record"})
	if err != nil {
		panic(err)
	}
	r, err := p.Add(cudashared.Request{Name: "stage0", Size: 4096, Alignment: 128, FirstStage: 0, LastStage: 0, Dynamic: true})
	if err != nil {
		panic(err)
	}
	fmt.Println(r.Offset, p.FootprintBytes())
	// Output: 0 4096
}
