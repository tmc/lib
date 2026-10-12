package nvtelemetry_test

import (
	"context"
	"fmt"
	"os"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaworkload"
	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func exampleAttributionSource() nvtelemetry.SlurmAttributionSource {
	return nvtelemetry.SlurmAttributionSource{
		Squeue:       "squeue.json",
		Mappings:     "mappings.json",
		Executions:   "executions.json",
		ProcDir:      "proc",
		PIDs:         []uint32{1234},
		BootID:       "boot-1",
		NodeID:       "node-1",
		PIDNamespace: "pid:[4026531836]",
		Snapshot: nvtelemetry.EntitySnapshot{
			SourceRevision: "fixture-squeue-1",
			Authoritative:  true,
			NodeID:         "node-1",
			ClockDomain:    "synthetic_unix_utc",
			ClockEpoch:     "fixture-epoch-1",
			StartNanos:     100,
			EndNanos:       1_000_000_000_000,
		},
	}
}

func ExampleReadSlurmAttributionSnapshot() {
	fsys := os.DirFS("testdata/entity/b690-refresh")
	read, err := nvtelemetry.ReadSlurmAttributionSnapshot(fsys, exampleAttributionSource())
	if err != nil {
		fmt.Println(err)
		return
	}
	s := read.Snapshot
	fmt.Println(len(s.Allocations), len(s.Mappings), len(s.Processes), len(s.Executions), len(read.Withheld))
	// Output: 1 1 1 1 0
}

func ExampleAttributionRefresher() {
	fsys := os.DirFS("testdata/entity/b690-refresh")
	table, _ := nvtelemetry.NewJobAttributionTable(16, int64(time.Hour))
	clock := nvtelemetry.NewFakeClock(0, 200)
	read := func(context.Context) (nvtelemetry.AttributionRead, error) {
		return nvtelemetry.ReadSlurmAttributionSnapshot(fsys, exampleAttributionSource())
	}
	r, err := nvtelemetry.NewAttributionRefresher(table, read, clock, nvtelemetry.RefreshConfig{
		Interval: time.Second, Timeout: time.Second, MaxAge: 5 * time.Second,
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	defer r.Close()
	if err := r.Step(context.Background()); err != nil {
		fmt.Println(err)
		return
	}
	proc := nvtelemetry.EntityProcess{
		Process:           cudaworkload.ProcessGenID{PID: 1234, Generation: 1000},
		BootID:            "boot-1",
		NodeID:            "node-1",
		PIDNamespace:      "pid:[4026531836]",
		ExecGeneration:    1,
		HasExecGeneration: true,
	}
	const gpu = "GPU-11111111-2222-3333-4444-555555555555"
	res := r.Resolve(proc, gpu, 200)
	fmt.Println(res.Assignment.State, res.Assignment.Owner.JobID)

	clock.Advance(6 * time.Second)
	res = r.Resolve(proc, gpu, 200)
	fmt.Println(res.Assignment.State, res.Reason, r.Ready())
	// Output:
	// assigned 42
	// unknown source_stale false
}
