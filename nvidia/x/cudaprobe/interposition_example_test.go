package cudaprobe_test

import (
	"fmt"
	"github.com/tmc/lib/nvidia/x/cudaprobe"
	"time"
)

func ExampleInterposition() {
	declaration := cudaprobe.Interposition{Status: "declared_active", Evidence: "operator run manifest"}
	event := cudaprobe.Event{Identity: cudaprobe.Identity{Interposition: declaration}}
	record, err := event.Record(time.Unix(100, 0))
	fmt.Println(err, record.Labels["probe.interposition"], record.Labels["probe.interposition_source"])
	// Output: <nil> declared_active caller_reported
}

func ExampleInterposition_Validate() {
	fmt.Println((cudaprobe.Interposition{}).Validate())
	// Output: <nil>
}
