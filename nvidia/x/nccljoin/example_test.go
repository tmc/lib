package nccljoin_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/ncclinspector"
	"github.com/tmc/lib/nvidia/x/nccljoin"
)

func ExampleJoin() {
	r := ncclinspector.Record{
		Header:     ncclinspector.Header{ID: "comm", Rank: 0, Ranks: 2, Nodes: 1},
		Metadata:   ncclinspector.Metadata{Version: "v4.1", Mechanism: "ring", Hostname: "host", PID: 42, TimestampMicros: 100},
		Collective: &ncclinspector.Collective{Name: "AllReduce", Sequence: 7, Micros: 20},
	}
	result, err := nccljoin.Join([]nccljoin.Capture{{ID: "dump", Records: []ncclinspector.Record{r}}}, nccljoin.Options{Scope: nccljoin.Scope{Capture: "selected", Job: "job", Epoch: "epoch"}})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(result.Collectives[0].Present, result.Collectives[0].Unobserved)
	// Output: [0] [1]
}
