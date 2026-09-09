package cudaprof_test

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaprof"
)

func ExampleSession_SetSampleTypes() {
	s := cudaprof.NewSession(io.Discard)
	fmt.Println(s.SetSampleTypes("gpu_time"))
	// Output: <nil>
}

func ExampleBuilder_SetSampleTypes() {
	b := cudaprof.NewBuilder()
	fmt.Println(b.SetSampleTypes("dram_read", "gpu_time"))
	// Output: <nil>
}

func ExampleDo() {
	ctx := context.Background()

	// Wrap execution in cudaprof.Do to attach Go labels and an NVTX range.
	labels := cudaprof.Labels("layer", "0", "phase", "decode")
	cudaprof.Do(ctx, labels, func(ctx context.Context) {
		// Pass ctx to recording methods to attach these labels to observations.
		fmt.Println("executing kernel with labels")
	})

	// Output:
	// executing kernel with labels
}

func ExampleFlightRecorder() {
	var buf bytes.Buffer

	// Create and start a bounded in-memory recorder.
	fr := cudaprof.NewFlightRecorder(
		cudaprof.WithWindowSize(1024),
		cudaprof.WithWindowDuration(10*time.Second),
	)
	_ = fr.Start()
	defer fr.Stop()

	// Record a caller-supplied sample into the ring buffer.
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
	_, _ = fr.WriteTo(&buf)
	fmt.Printf("snapshot written: %v\n", buf.Len() > 0)

	// Output:
	// snapshot written: true
}
