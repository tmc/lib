package nvtelemetry_test

import (
	"context"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudaworkload"
	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func ExampleFakeSampler() {
	sampler := nvtelemetry.NewFakeSampler(nvtelemetry.DeviceMetadata{
		UUID:  "GPU-EXAMPLE-1",
		Index: 0,
		Model: "NVIDIA L4",
	})

	sampler.Enqueue([]nvtelemetry.Reading{
		{
			Name:       nvtelemetry.MetricGPUUtilization,
			Unit:       nvtelemetry.UnitPercent,
			Backend:    "fake",
			DeviceUUID: "GPU-EXAMPLE-1",
			Evidence:   nvtelemetry.EvidenceMeasured,
			Value:      78.5,
		},
	})

	readings, err := sampler.Sample(context.Background())
	if err != nil {
		fmt.Printf("error: %v\n", err)
		return
	}

	for _, r := range readings {
		fmt.Printf("%s: %.1f %s (%s)\n", r.Name, r.Value, r.Unit, r.Evidence)
	}

	// Output:
	// gpu_utilization_pct: 78.5 percent (measured)
}

func ExampleParseSlurmCgroupPath() {
	id, err := nvtelemetry.ParseSlurmCgroupPath("/sys/fs/cgroup/system.slice/slurmstepd.scope/job_12345/step_0/task_1")
	if err != nil {
		panic(err)
	}
	fmt.Printf("job=%s step=%s task=%s\n", id.JobID, id.StepID, id.TaskID)
	// Output:
	// job=12345 step=0 task=1
}

func ExampleJobAttributionTable() {
	table, err := nvtelemetry.NewJobAttributionTable(10, 1000)
	if err != nil {
		panic(err)
	}

	snap := nvtelemetry.JobAttributionSnapshot{
		Allocations: []nvtelemetry.SlurmAllocation{
			{
				JobID:     "42",
				TRESAlloc: "gres/gpu=1",
				State:     []string{"RUNNING"},
				Snapshot:  nvtelemetry.EntitySnapshot{SourceRevision: "v1", ClockDomain: "monotonic", ClockEpoch: "boot-1", StartNanos: 10, EndNanos: 500},
			},
		},
		Mappings: []nvtelemetry.SlurmGPUMapping{
			{
				JobID:         "42",
				Device:        nvtelemetry.DeviceMetadata{UUID: "GPU-1"},
				Sharing:       "exclusive",
				Authoritative: true,
				ClockDomain:   "monotonic",
				ClockEpoch:    "boot-1",
				StartNanos:    10,
				EndNanos:      500,
			},
		},
		Processes: []nvtelemetry.ProcEntityFacts{
			{
				Process: nvtelemetry.EntityProcess{
					Process:      cudaworkload.ProcessGenID{PID: 100, Generation: 20},
					BootID:       "boot-1",
					NodeID:       "node-1",
					PIDNamespace: "pid:[1]",
				},
				Cgroups: []string{"0::/system.slice/slurmstepd.scope/job_42/step_0/task_0"},
			},
		},
		Executions: []nvtelemetry.EntityAssignment{
			{
				Process: nvtelemetry.EntityProcess{
					Process:           cudaworkload.ProcessGenID{PID: 100, Generation: 20},
					BootID:            "boot-1",
					NodeID:            "node-1",
					PIDNamespace:      "pid:[1]",
					ExecGeneration:    1,
					HasExecGeneration: true,
				},
				Device:         nvtelemetry.DeviceMetadata{UUID: "GPU-1"},
				Owner:          nvtelemetry.EntityOwner{JobID: "42"},
				Source:         "process_proof",
				SourceRevision: "v1",
				Authoritative:  true,
				State:          "assigned",
				ClockDomain:    "monotonic",
				ClockEpoch:     "boot-1",
				StartNanos:     10,
				EndNanos:       500,
				Sharing:        "exclusive",
			},
		},
	}

	results, err := table.Refresh(snap, 50)
	if err != nil {
		panic(err)
	}

	fmt.Printf("%d state=%s job=%s device=%s\n",
		len(results),
		results[0].Assignment.State,
		results[0].Assignment.Owner.JobID,
		results[0].Assignment.Device.UUID,
	)
	// Output:
	// 1 state=assigned job=42 device=GPU-1
}
