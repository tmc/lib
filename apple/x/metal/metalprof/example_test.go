package metalprof_test

import (
	"context"
	"fmt"
	"runtime/pprof"
	"time"

	"github.com/tmc/lib/apple/x/metal/metalprof"
)

func Example() {
	// 1. Initialize continuous in-memory flight recorder
	fr := metalprof.NewFlightRecorder(128)

	// 2. Propagate pprof context labels into GPU dispatches
	labels := pprof.Labels("model", "llama3", "layer", "0")
	metalprof.Do(context.Background(), labels, func(ctx context.Context) {
		ctxLabels := metalprof.LabelsFromContext(ctx)

		// Record a simulated GPU dispatch
		fr.Record(
			"EmbeddingLookupH",
			1450*time.Nanosecond,
			[3]int{2048, 1, 1},
			[3]int{128, 1, 1},
			ctxLabels,
		)
	})

	events := fr.Events()
	fmt.Printf("Recorded %d GPU dispatches: %s (%v)\n", len(events), events[0].Name, events[0].Duration)

	// Output:
	// Recorded 1 GPU dispatches: EmbeddingLookupH (1.45µs)
}
