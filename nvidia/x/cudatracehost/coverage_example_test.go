package cudatracehost_test

import (
	"encoding/json"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudatracehost"
)

func ExampleSnapshotCoverage() {
	// This is a serialized attachment manifest, not a native collection receipt.
	data := []byte(`{"ThreadCoverage":{"Status":"generation_verified_attachment_snapshot","Clock":"elapsed_since_discovery","Threads":[{"TID":7,"Generation":107}],"FutureThreads":"unobserved","BlindInterval":"discovery_through_attachment"},"Complete":false}`)
	var manifest cudatracehost.Manifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		fmt.Println(err)
		return
	}
	coverage := manifest.ThreadCoverage
	fmt.Println(coverage.Threads[0].TID, coverage.FutureThreads, manifest.Complete)
	// Output: 7 unobserved false
}

func ExampleThreadSnapshot() {
	task := cudatracehost.ThreadSnapshot{TID: 7, Generation: 107}
	fmt.Println(task.TID, task.Generation)
	// Output: 7 107
}
