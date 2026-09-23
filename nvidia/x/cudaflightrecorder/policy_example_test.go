package cudaflightrecorder_test

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudabpf"
	"github.com/tmc/lib/nvidia/x/cudaflightrecorder"
)

func ExampleRecorder_PushPolicyRecord() {
	r, err := cudaflightrecorder.New(cudaflightrecorder.Config{Capacity: 8})
	if err != nil {
		fmt.Println(err)
		return
	}
	if err := r.Start(); err != nil {
		fmt.Println(err)
		return
	}
	defer r.Stop()
	observer := cudabpf.SinkObserver{Sink: r}
	observer.Observe(cudabpf.Outcome{Hook: "on_bind", Fallback: true, Reason: "hook unavailable"})
	var out bytes.Buffer
	if _, err := r.Snapshot(context.Background(), &out, "policy"); err != nil {
		fmt.Println(err)
		return
	}
	bundle, err := cudaflightrecorder.ReadBundle(&out)
	if err != nil {
		fmt.Println(err)
		return
	}
	var records []cudaflightrecorder.Record
	if err := json.Unmarshal(bundle.Files["gpu_records.json"], &records); err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(records[0].Labels["hook"], records[0].Labels["reason"])
	// Output: on_bind hook unavailable
}
