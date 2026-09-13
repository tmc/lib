package cudaflightrecorder_test

import (
	"bytes"
	"context"
	"fmt"
	"log"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaflightrecorder"
)

func Example() {
	rec, err := cudaflightrecorder.New(cudaflightrecorder.Config{
		Window:   5 * time.Second,
		Capacity: 1024,
	})
	if err != nil {
		log.Fatal(err)
	}

	if err := rec.Start(); err != nil {
		log.Fatal(err)
	}
	defer rec.Stop()

	// Record an execution observation
	rec.Push(cudaflightrecorder.Record{
		TimestampNs:   time.Now().UnixNano(),
		KernelName:    "ds4_moe_expert_dispatch",
		CorrelationID: 42,
		DurationNs:    8500,
	})

	// Take a zero-device crash dump snapshot to a buffer
	var buf bytes.Buffer
	manifest, err := rec.Snapshot(context.Background(), &buf, "watchdog_timeout")
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Snapshot reason: %s (records: %d)\n", manifest.Reason, manifest.GPURecords)

	// Output:
	// Snapshot reason: watchdog_timeout (records: 1)
}
