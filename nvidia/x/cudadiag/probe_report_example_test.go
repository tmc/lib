package cudadiag_test

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"time"

	"github.com/tmc/lib/nvidia/x/cudadiag"
	"github.com/tmc/lib/nvidia/x/cudaprobe"
	"github.com/tmc/lib/nvidia/x/cudatracehost"
)

func ExampleReadProbeReport() {
	process := cudaprobe.Process{PID: 12, Generation: 30}
	// Recorded input preserves the live collector's unknown address space.
	capture := &cudaprobe.Capture{Identity: cudaprobe.Identity{Process: process, ThreadID: 14, ThreadGeneration: 44, ThreadGenerationKnown: true, Library: "/recorded/libcuda.so", BuildID: "recorded-build", Version: "fixture-v1", Symbol: "cuLaunchKernel", ABI: "linux-amd64-sysv-v1", Provenance: "example-recorded-input"}}
	binary.LittleEndian.PutUint16(capture.Data[:], 1)
	binary.LittleEndian.PutUint16(capture.Data[2:], 3)
	binary.LittleEndian.PutUint64(capture.Data[8:], 1)
	binary.LittleEndian.PutUint64(capture.Data[16:], 100)
	binary.LittleEndian.PutUint64(capture.Data[32:], 99)
	raw := cudatracehost.Result{Capture: capture, Call: &cudatracehost.CallInterval{Process: process, TID: 14, ThreadGeneration: 44, ThreadGenerationKnown: true, ExecID: 9, PairingStatus: "verified_thread_lifetime", EntryTimestamp: 100, ReturnTimestamp: 250, Clock: "linux_monotonic", Epoch: "recorded-boot"}}
	var input bytes.Buffer
	if err := json.NewEncoder(&input).Encode(raw); err != nil {
		fmt.Println(err)
		return
	}
	r, err := cudadiag.ReadProbeReport(context.Background(), &input, cudadiag.ProbeReportOptions{
		Manifest: cudatracehost.Manifest{Process: process, Backend: "bpf", Clock: "linux_monotonic", Epoch: "recorded-boot"},
		ThreadID: 14, ThreadGeneration: 44, ExecID: 9, AddressSpace: 1, ModuleGeneration: 7,
		FirstSequence: 1, LastSequence: 1, LossKnown: true,
		Received: time.Unix(1700000000, 0), Threshold: 100 * time.Nanosecond,
		MaxBytes: 64 << 10, MaxLineBytes: 16 << 10, MaxRecords: 8, MaxOutliers: 4,
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(len(r.Outliers), r.Outliers[0].HostDurationNanos, r.Partial)
	fmt.Println(r.Outliers[0].Record.Labels["kernel_identity_reason"])
	// Output:
	// 1 150 true
	// capture address space unavailable
}
