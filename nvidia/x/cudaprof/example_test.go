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

func ExampleExportFoldedStacks() {
	var buf bytes.Buffer
	executions := []cudaprof.KernelExecution{
		{
			KernelName: "fused_attention",
			Callstack:  []string{"main", "runInference"},
			StartNs:    1000,
			EndNs:      2500,
		},
	}
	_ = cudaprof.ExportFoldedStacks(&buf, executions)
	fmt.Print(buf.String())

	// Output:
	// main;runInference;[GPU]fused_attention 1500
}

func ExampleWriteExternalStackTable() {
	cudaprof.ResetExternalStacks()
	defer cudaprof.ResetExternalStacks()
	cudaprof.RecordExternalStackKind(3, 7, []string{"main", "run"})
	var buf bytes.Buffer
	_ = cudaprof.WriteExternalStackTable(&buf)
	fmt.Print(buf.String())

	// Output:
	// {"version":1,"losses":0,"stacks":[{"external_kind":3,"external_id":7,"frames":["main","run"]}]}
}

func ExampleExternalStackKind() {
	cudaprof.ResetExternalStacks()
	cudaprof.RecordExternalStackKind(3, 42, []string{"main", "launch"})
	stack, ok := cudaprof.ExternalStackKind(3, 42)
	fmt.Println(stack, ok)
	// Output: [main launch] true
}

func ExampleDoID() {
	cudaprof.Correlate(0)
	id := cudaprof.DoID(context.Background(), cudaprof.Labels(), func(context.Context) {})
	fmt.Println(id)
	// Output: 0
}
