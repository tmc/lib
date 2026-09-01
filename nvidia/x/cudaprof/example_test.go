package cudaprof_test

import (
	"bytes"
	"context"
	"fmt"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaprof"
)

func ExampleDo() {
	ctx := context.Background()

	// Wrap execution in cudaprof.Do to propagate labels to Go context,
	// NVTX ranges, and CUPTI hardware activity correlations.
	labels := cudaprof.Labels("layer", "0", "phase", "decode")
	cudaprof.Do(ctx, labels, func(ctx context.Context) {
		// GPU kernel launches and SM PC samples inherit these labels.
		fmt.Println("executing kernel with labels")
	})

	// Output:
	// executing kernel with labels
}

func ExampleFlightRecorder() {
	var buf bytes.Buffer

	// Create and start continuous in-memory GPU flight recorder
	fr := cudaprof.NewFlightRecorder(
		cudaprof.WithWindowSize(1024),
		cudaprof.WithWindowDuration(10*time.Second),
	)
	_ = fr.Start()
	defer fr.Stop()

	// Record sample into ring buffer
	fr.Record(cudaprof.SampleRecord{
		KernelName: "LinearGEMV",
		CuFile:     "forward.cu",
		CuLine:     90,
		PtxFile:    "forward.ptx",
		PtxLine:    26,
		SassPC:     0x7f001040,
		Values: cudaprof.SampleValues{
			Samples:       512,
			DRAMReadBytes: 2048,
		},
	})

	// Snapshot ring buffer on demand without stopping execution
	_ = fr.WriteTo(&buf)
	fmt.Printf("snapshot written: %v\n", buf.Len() > 0)

	// Output:
	// snapshot written: true
}

