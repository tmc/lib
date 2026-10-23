package cudacapture_test

import (
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudacapture"
)

func ExampleCorrelationTracker_ExternalCallstack() {
	t := cudacapture.NewCorrelationTracker()
	_ = t.ReadExternalStackTable(strings.NewReader(`{"version":1,"stacks":[{"process":101,"external_kind":3,"external_id":7,"frames":["main.Do"]}]}`))
	_ = t.RecordActivity(cudacapture.ActivityRecord{Process: 101, Subtype: "external_correlation", Correlation: 77, ExternalKind: 3, ExternalID: 7})
	id, frames, err := t.ExternalCallstack(101, 77, 3)
	fmt.Println(id, frames, err) // Sidecar attribution, not a runtime launch.
	// Output: 7 [main.Do] <nil>
}

func ExampleCorrelationTracker_JoinKind() {
	t := cudacapture.NewCorrelationTracker()
	_ = t.RecordRuntime(cudacapture.RuntimeLaunch{CorrelationID: 77, HostThreadID: 9})
	_ = t.RecordActivity(cudacapture.ActivityRecord{Process: 101, Subtype: "kernel", Correlation: 77, Start: 100, End: 150, Valid: true})
	joins, _, _, _ := t.JoinKind(3)
	fmt.Println(len(joins), joins[0].HostThreadID, joins[0].HasExternalID)
	// Output: 1 9 false
}
