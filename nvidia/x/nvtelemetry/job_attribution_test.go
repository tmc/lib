package nvtelemetry_test

import (
	"fmt"
	"sync"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaworkload"
	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func TestParseSlurmCgroupPath_Positive(t *testing.T) {
	tests := []struct {
		name       string
		path       string
		wantJob    string
		wantStep   string
		wantTask   string
		wantCgroup string
	}{
		{
			name:       "full systemd scope",
			path:       "/sys/fs/cgroup/system.slice/slurmstepd.scope/job_12345/step_0/task_2",
			wantJob:    "12345",
			wantStep:   "0",
			wantTask:   "2",
			wantCgroup: "/sys/fs/cgroup/system.slice/slurmstepd.scope/job_12345/step_0/task_2",
		},
		{
			name:       "proc cgroup line batch step",
			path:       "0::/system.slice/slurmstepd.scope/job_42/step_batch/task_0",
			wantJob:    "42",
			wantStep:   "batch",
			wantTask:   "0",
			wantCgroup: "0::/system.slice/slurmstepd.scope/job_42/step_batch/task_0",
		},
		{
			name:       "step extern without task",
			path:       "/sys/fs/cgroup/slurm/job_101/step_extern",
			wantJob:    "101",
			wantStep:   "extern",
			wantTask:   "",
			wantCgroup: "/sys/fs/cgroup/slurm/job_101/step_extern",
		},
		{
			name:       "interactive step with task",
			path:       "/job_777/step_interactive/task_1",
			wantJob:    "777",
			wantStep:   "interactive",
			wantTask:   "1",
			wantCgroup: "/job_777/step_interactive/task_1",
		},
		{
			name:       "job only without step",
			path:       "/job_999",
			wantJob:    "999",
			wantStep:   "",
			wantTask:   "",
			wantCgroup: "/job_999",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := nvtelemetry.ParseSlurmCgroupPath(tt.path)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got.JobID != tt.wantJob {
				t.Errorf("JobID = %q; want %q", got.JobID, tt.wantJob)
			}
			if got.StepID != tt.wantStep {
				t.Errorf("StepID = %q; want %q", got.StepID, tt.wantStep)
			}
			if got.TaskID != tt.wantTask {
				t.Errorf("TaskID = %q; want %q", got.TaskID, tt.wantTask)
			}
			if got.CgroupPath != tt.wantCgroup {
				t.Errorf("CgroupPath = %q; want %q", got.CgroupPath, tt.wantCgroup)
			}
		})
	}
}

func TestParseSlurmCgroupPath_Negative(t *testing.T) {
	tests := []struct {
		name string
		path string
	}{
		{"empty path", ""},
		{"cgroup v1 hierarchy", "1:devices:/job_123"},
		{"cgroup v1 controllers", "2:cpu,cpuacct:/job_123"},
		{"missing job segment", "/sys/fs/cgroup/system.slice/sshd.service"},
		{"empty job id", "/job_/step_0"},
		{"non-numeric job id", "/job_abc/step_0"},
		{"job id zero", "/job_0/step_0"},
		{"multiple jobs", "/job_123/job_456"},
		{"step before job", "/step_0/job_123"},
		{"empty step id", "/job_123/step_"},
		{"invalid step id chars", "/job_123/step_@!/task_0"},
		{"task without step", "/job_123/task_0"},
		{"empty task id", "/job_123/step_0/task_"},
		{"non-numeric task id", "/job_123/step_0/task_xyz"},
		{"multiple steps", "/job_123/step_0/step_1"},
		{"multiple tasks", "/job_123/step_0/task_0/task_1"},
		{"relative path component", "/job_123/../etc"},
		{"non-absolute path", "job_123/step_0"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := nvtelemetry.ParseSlurmCgroupPath(tt.path)
			if err == nil {
				t.Errorf("ParseSlurmCgroupPath(%q) succeeded; want error", tt.path)
			}
		})
	}
}

func TestResolveSlurmProcessCgroup(t *testing.T) {
	validProc := nvtelemetry.ProcEntityFacts{
		Process: nvtelemetry.EntityProcess{
			Process:      cudaworkload.ProcessGenID{PID: 100, Generation: 500},
			BootID:       "boot-1",
			NodeID:       "node-1",
			PIDNamespace: "pid:[1]",
		},
		StartTimeTicks: 500,
		Cgroups: []string{
			"1:devices:/system.slice/docker.service",
			"0::/system.slice/slurmstepd.scope/job_42/step_0/task_1",
		},
	}

	id, err := nvtelemetry.ResolveSlurmProcessCgroup(validProc)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if id.JobID != "42" || id.StepID != "0" || id.TaskID != "1" {
		t.Fatalf("unexpected identity: %+v", id)
	}

	// Incomplete identity
	badProc := validProc
	badProc.Process.BootID = ""
	if _, err := nvtelemetry.ResolveSlurmProcessCgroup(badProc); err == nil {
		t.Errorf("expected error for empty boot id")
	}

	// No Slurm cgroup
	noSlurm := validProc
	noSlurm.Cgroups = []string{"0::/system.slice/docker-123.scope"}
	if _, err := nvtelemetry.ResolveSlurmProcessCgroup(noSlurm); err == nil {
		t.Errorf("expected error for non-slurm cgroup")
	}

	// Contradictory Slurm cgroups
	conflict := validProc
	conflict.Cgroups = []string{
		"0::/system.slice/slurmstepd.scope/job_42/step_0/task_1",
		"0::/system.slice/slurmstepd.scope/job_99/step_0/task_1",
	}
	if _, err := nvtelemetry.ResolveSlurmProcessCgroup(conflict); err == nil {
		t.Errorf("expected error for contradictory slurm cgroups")
	}
}

func TestParseSlurmGRES(t *testing.T) {
	// Simple count
	g, err := nvtelemetry.ParseSlurmGRES("cpu=4,mem=8G,gres/gpu=2")
	if err != nil || g.Count != 2 || g.GPUType != "" {
		t.Fatalf("gres count: %+v %v", g, err)
	}

	// Type and count
	g, err = nvtelemetry.ParseSlurmGRES("gres/gpu:a100=4")
	if err != nil || g.Count != 4 || g.GPUType != "a100" {
		t.Fatalf("gres type: %+v %v", g, err)
	}

	// Explicit device index
	g, err = nvtelemetry.ParseSlurmGRES("gres/gpu:0")
	if err != nil || g.Count != 1 || len(g.DeviceIDs) != 1 || g.DeviceIDs[0] != "0" {
		t.Fatalf("gres device index: %+v %v", g, err)
	}

	// Type with device index
	g, err = nvtelemetry.ParseSlurmGRES("gres/gpu:h100:0")
	if err != nil || g.Count != 1 || g.GPUType != "h100" || len(g.DeviceIDs) != 1 || g.DeviceIDs[0] != "0" {
		t.Fatalf("gres type and index: %+v %v", g, err)
	}

	// No GRES
	g, err = nvtelemetry.ParseSlurmGRES("cpu=4,mem=8G")
	if err != nil || g.Count != 0 {
		t.Fatalf("empty gres: %+v %v", g, err)
	}

	// Conflicting TRES
	if _, err := nvtelemetry.ParseSlurmGRES("cpu=4,gres/gpu=1,gres/gpu=2"); err == nil {
		t.Errorf("expected error for conflicting TRES")
	}

	// Invalid count
	if _, err := nvtelemetry.ParseSlurmGRES("gres/gpu=-1"); err == nil {
		t.Errorf("expected error for negative GPU count")
	}
}

func TestProcessToGPUAttribution_Positive(t *testing.T) {
	s := nvtelemetry.EntitySnapshot{
		SourceRevision: "slurm-25.05",
		NodeID:         "node-1",
		ClockDomain:    "monotonic",
		ClockEpoch:     "boot-1",
		StartNanos:     100,
		EndNanos:       500,
	}

	job := nvtelemetry.SlurmAllocation{
		JobID:     "42",
		Nodes:     "node-1",
		TRESAlloc: "cpu=4,gres/gpu=1",
		State:     []string{"RUNNING"},
		Snapshot:  s,
	}

	dev := nvtelemetry.DeviceMetadata{
		UUID:  "GPU-TEST-UUID-1",
		Index: 0,
		Model: "NVIDIA H100",
	}

	mapping := nvtelemetry.SlurmGPUMapping{
		JobID:         "42",
		StepID:        "0",
		Device:        dev,
		Sharing:       "exclusive",
		Authoritative: true,
		AllocationID:  "alloc-42",
		Attempt:       "attempt-1",
		ClockDomain:   "monotonic",
		ClockEpoch:    "boot-1",
		StartNanos:    100,
		EndNanos:      500,
	}

	procFacts := nvtelemetry.ProcEntityFacts{
		Process: nvtelemetry.EntityProcess{
			Process:      cudaworkload.ProcessGenID{PID: 1234, Generation: 1000},
			BootID:       "boot-1",
			NodeID:       "node-1",
			PIDNamespace: "pid:[10]",
		},
		StartTimeTicks: 1000,
		Cgroups: []string{
			"0::/system.slice/slurmstepd.scope/job_42/step_0/task_0",
		},
	}

	exec := nvtelemetry.EntityAssignment{
		Process: nvtelemetry.EntityProcess{
			Process:           cudaworkload.ProcessGenID{PID: 1234, Generation: 1000},
			BootID:            "boot-1",
			NodeID:            "node-1",
			PIDNamespace:      "pid:[10]",
			ExecGeneration:    1,
			HasExecGeneration: true,
		},
		Device:         dev,
		Owner:          nvtelemetry.EntityOwner{JobID: "42", AllocationID: "alloc-42", Attempt: "attempt-1"},
		Source:         "process_proof",
		SourceRevision: "v1",
		Authoritative:  true,
		State:          "assigned",
		ClockDomain:    "monotonic",
		ClockEpoch:     "boot-1",
		StartNanos:     100,
		EndNanos:       500,
		Sharing:        "exclusive",
	}

	results, err := nvtelemetry.JoinSlurmAttribution(
		[]nvtelemetry.SlurmAllocation{job},
		[]nvtelemetry.SlurmGPUMapping{mapping},
		[]nvtelemetry.ProcEntityFacts{procFacts},
		[]nvtelemetry.EntityAssignment{exec},
		200,
	)
	if err != nil {
		t.Fatalf("unexpected join error: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("expected 1 result; got %d", len(results))
	}

	r := results[0]
	if r.Assignment.State != "assigned" {
		t.Fatalf("expected assigned state; got %q (reason: %q)", r.Assignment.State, r.Reason)
	}
	if r.Reason != "" {
		t.Errorf("unexpected reason %q", r.Reason)
	}
	if r.Cgroup.JobID != "42" || r.Cgroup.StepID != "0" || r.Cgroup.TaskID != "0" {
		t.Errorf("cgroup mismatch: %+v", r.Cgroup)
	}
	if r.Assignment.Owner.JobID != "42" || r.Assignment.Device.UUID != "GPU-TEST-UUID-1" {
		t.Errorf("assignment mismatch: owner=%+v device=%+v", r.Assignment.Owner, r.Assignment.Device)
	}
	if len(r.Assignment.Provenance) != 3 {
		t.Errorf("expected 3 provenance records; got %d", len(r.Assignment.Provenance))
	}
}

func TestAttribution_NegativeControls(t *testing.T) {
	s := nvtelemetry.EntitySnapshot{
		SourceRevision: "slurm-25.05",
		NodeID:         "node-1",
		ClockDomain:    "monotonic",
		ClockEpoch:     "boot-1",
		StartNanos:     100,
		EndNanos:       500,
	}

	dev := nvtelemetry.DeviceMetadata{
		UUID:  "GPU-TEST-UUID-1",
		Index: 0,
		Model: "NVIDIA H100",
	}

	baseJob := nvtelemetry.SlurmAllocation{
		JobID:     "42",
		Nodes:     "node-1",
		TRESAlloc: "cpu=4,gres/gpu=1",
		State:     []string{"RUNNING"},
		Snapshot:  s,
	}

	baseMapping := nvtelemetry.SlurmGPUMapping{
		JobID:         "42",
		StepID:        "0",
		Device:        dev,
		Sharing:       "exclusive",
		Authoritative: true,
		AllocationID:  "alloc-42",
		Attempt:       "attempt-1",
		ClockDomain:   "monotonic",
		ClockEpoch:    "boot-1",
		StartNanos:    100,
		EndNanos:      500,
	}

	baseProc := nvtelemetry.ProcEntityFacts{
		Process: nvtelemetry.EntityProcess{
			Process:      cudaworkload.ProcessGenID{PID: 1234, Generation: 1000},
			BootID:       "boot-1",
			NodeID:       "node-1",
			PIDNamespace: "pid:[10]",
		},
		StartTimeTicks: 1000,
		Cgroups: []string{
			"0::/system.slice/slurmstepd.scope/job_42/step_0/task_0",
		},
	}

	baseExec := nvtelemetry.EntityAssignment{
		Process: nvtelemetry.EntityProcess{
			Process:           cudaworkload.ProcessGenID{PID: 1234, Generation: 1000},
			BootID:            "boot-1",
			NodeID:            "node-1",
			PIDNamespace:      "pid:[10]",
			ExecGeneration:    1,
			HasExecGeneration: true,
		},
		Device:         dev,
		Owner:          nvtelemetry.EntityOwner{JobID: "42", AllocationID: "alloc-42", Attempt: "attempt-1"},
		Source:         "process_proof",
		SourceRevision: "v1",
		Authoritative:  true,
		State:          "assigned",
		ClockDomain:    "monotonic",
		ClockEpoch:     "boot-1",
		StartNanos:     100,
		EndNanos:       500,
		Sharing:        "exclusive",
	}

	// 1. Stale allocation (job completed)
	t.Run("stale allocation completed", func(t *testing.T) {
		completedJob := baseJob
		completedJob.State = []string{"COMPLETED"}
		res, err := nvtelemetry.JoinSlurmAttribution(
			[]nvtelemetry.SlurmAllocation{completedJob},
			[]nvtelemetry.SlurmGPUMapping{baseMapping},
			[]nvtelemetry.ProcEntityFacts{baseProc},
			[]nvtelemetry.EntityAssignment{baseExec},
			200,
		)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(res) != 1 || res[0].Assignment.State != "unknown" || res[0].Reason != "stale_allocation" {
			t.Fatalf("expected stale_allocation unknown state; got %+v", res[0])
		}
	})

	// Stale allocation (time after EndNanos)
	t.Run("stale allocation past expiry", func(t *testing.T) {
		res, err := nvtelemetry.JoinSlurmAttribution(
			[]nvtelemetry.SlurmAllocation{baseJob},
			[]nvtelemetry.SlurmGPUMapping{baseMapping},
			[]nvtelemetry.ProcEntityFacts{baseProc},
			[]nvtelemetry.EntityAssignment{baseExec},
			600, // past EndNanos=500
		)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(res) != 1 || res[0].Assignment.State != "unknown" || res[0].Reason != "stale_allocation" {
			t.Fatalf("expected stale_allocation unknown state; got %+v", res[0])
		}
	})

	// 2. PID reuse across generations
	t.Run("pid reuse across generations", func(t *testing.T) {
		reusedExec := baseExec
		reusedExec.Process.Process.Generation = 9999 // Different generation!
		res, err := nvtelemetry.JoinSlurmAttribution(
			[]nvtelemetry.SlurmAllocation{baseJob},
			[]nvtelemetry.SlurmGPUMapping{baseMapping},
			[]nvtelemetry.ProcEntityFacts{baseProc},
			[]nvtelemetry.EntityAssignment{reusedExec},
			200,
		)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(res) != 1 || res[0].Assignment.State != "unknown" || res[0].Reason != "reused_pid" {
			t.Fatalf("expected reused_pid unknown state; got %+v", res[0])
		}
	})

	// 3. Conflicting TRES
	t.Run("conflicting tres count", func(t *testing.T) {
		conflictJob := baseJob
		conflictJob.TRESAlloc = "gres/gpu=1,gres/gpu=2" // syntax conflict
		res, err := nvtelemetry.JoinSlurmAttribution(
			[]nvtelemetry.SlurmAllocation{conflictJob},
			[]nvtelemetry.SlurmGPUMapping{baseMapping},
			[]nvtelemetry.ProcEntityFacts{baseProc},
			[]nvtelemetry.EntityAssignment{baseExec},
			200,
		)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(res) != 1 || res[0].Assignment.State != "contradictory" || res[0].Reason != "conflicting_tres" {
			t.Fatalf("expected conflicting_tres state; got %+v", res[0])
		}
	})

	t.Run("conflicting tres mapped count exceeds allocated", func(t *testing.T) {
		dev2 := dev
		dev2.UUID = "GPU-TEST-UUID-2"
		dev2.Index = 1
		mapping2 := baseMapping
		mapping2.Device = dev2

		// TRES allocated only 1 GPU, but 2 distinct GPUs mapped to job
		res, err := nvtelemetry.JoinSlurmAttribution(
			[]nvtelemetry.SlurmAllocation{baseJob},
			[]nvtelemetry.SlurmGPUMapping{baseMapping, mapping2},
			[]nvtelemetry.ProcEntityFacts{baseProc},
			[]nvtelemetry.EntityAssignment{baseExec},
			200,
		)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(res) != 1 || res[0].Assignment.State != "contradictory" || res[0].Reason != "conflicting_tres" {
			t.Fatalf("expected conflicting_tres state; got %+v", res[0])
		}
	})

	// 4. Unidentifiable MIG sharing mode
	t.Run("unidentifiable mig sharing mode", func(t *testing.T) {
		migMapping := baseMapping
		migMapping.Sharing = "mig"
		// Missing MIGScopeMIGInstance and MIGParentUUID
		migMapping.Device.MIGScope = nvtelemetry.MIGScopeUnknown
		migMapping.Device.MIGParentUUID = ""
		migMapping.MIGConfigurationEpoch = ""

		res, err := nvtelemetry.JoinSlurmAttribution(
			[]nvtelemetry.SlurmAllocation{baseJob},
			[]nvtelemetry.SlurmGPUMapping{migMapping},
			[]nvtelemetry.ProcEntityFacts{baseProc},
			[]nvtelemetry.EntityAssignment{baseExec},
			200,
		)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(res) != 1 || res[0].Assignment.State != "unknown" || res[0].Reason != "unidentifiable_mig_sharing" {
			t.Fatalf("expected unidentifiable_mig_sharing; got %+v", res[0])
		}
	})

	// 5. Invalid cgroup paths
	t.Run("invalid cgroup path in process", func(t *testing.T) {
		invalidProc := baseProc
		invalidProc.Cgroups = []string{"1:devices:/system.slice/non-slurm"}
		res, err := nvtelemetry.JoinSlurmAttribution(
			[]nvtelemetry.SlurmAllocation{baseJob},
			[]nvtelemetry.SlurmGPUMapping{baseMapping},
			[]nvtelemetry.ProcEntityFacts{invalidProc},
			[]nvtelemetry.EntityAssignment{baseExec},
			200,
		)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(res) != 1 || res[0].Assignment.State != "unknown" || res[0].Reason != "missing_slurm_cgroup" {
			t.Fatalf("expected missing_slurm_cgroup; got %+v", res[0])
		}
	})
}

func TestJobAttributionTable_LifecycleAndBounds(t *testing.T) {
	s := nvtelemetry.EntitySnapshot{
		SourceRevision: "slurm-25.05",
		NodeID:         "node-1",
		ClockDomain:    "monotonic",
		ClockEpoch:     "boot-1",
		StartNanos:     100,
		EndNanos:       500,
	}

	dev := nvtelemetry.DeviceMetadata{
		UUID:  "GPU-TEST-UUID-1",
		Index: 0,
		Model: "NVIDIA H100",
	}

	job := nvtelemetry.SlurmAllocation{
		JobID:     "42",
		Nodes:     "node-1",
		TRESAlloc: "cpu=4,gres/gpu=1",
		State:     []string{"RUNNING"},
		Snapshot:  s,
	}

	mapping := nvtelemetry.SlurmGPUMapping{
		JobID:         "42",
		StepID:        "0",
		Device:        dev,
		Sharing:       "exclusive",
		Authoritative: true,
		AllocationID:  "alloc-42",
		Attempt:       "attempt-1",
		ClockDomain:   "monotonic",
		ClockEpoch:    "boot-1",
		StartNanos:    100,
		EndNanos:      500,
	}

	procFacts := nvtelemetry.ProcEntityFacts{
		Process: nvtelemetry.EntityProcess{
			Process:      cudaworkload.ProcessGenID{PID: 1234, Generation: 1000},
			BootID:       "boot-1",
			NodeID:       "node-1",
			PIDNamespace: "pid:[10]",
		},
		StartTimeTicks: 1000,
		Cgroups: []string{
			"0::/system.slice/slurmstepd.scope/job_42/step_0/task_0",
		},
	}

	exec := nvtelemetry.EntityAssignment{
		Process: nvtelemetry.EntityProcess{
			Process:           cudaworkload.ProcessGenID{PID: 1234, Generation: 1000},
			BootID:            "boot-1",
			NodeID:            "node-1",
			PIDNamespace:      "pid:[10]",
			ExecGeneration:    1,
			HasExecGeneration: true,
		},
		Device:         dev,
		Owner:          nvtelemetry.EntityOwner{JobID: "42", AllocationID: "alloc-42", Attempt: "attempt-1"},
		Source:         "process_proof",
		SourceRevision: "v1",
		Authoritative:  true,
		State:          "assigned",
		ClockDomain:    "monotonic",
		ClockEpoch:     "boot-1",
		StartNanos:     100,
		EndNanos:       500,
		Sharing:        "exclusive",
	}

	// Create table with capacity 2 and TTL of 200ns
	table, err := nvtelemetry.NewJobAttributionTable(2, 200)
	if err != nil {
		t.Fatalf("failed to create table: %v", err)
	}

	snap := nvtelemetry.JobAttributionSnapshot{
		Allocations: []nvtelemetry.SlurmAllocation{job},
		Mappings:    []nvtelemetry.SlurmGPUMapping{mapping},
		Processes:   []nvtelemetry.ProcEntityFacts{procFacts},
		Executions:  []nvtelemetry.EntityAssignment{exec},
	}

	// Initial refresh at t=150
	res, err := table.Refresh(snap, 150)
	if err != nil {
		t.Fatalf("refresh error: %v", err)
	}
	if len(res) != 1 || res[0].Assignment.State != "assigned" {
		t.Fatalf("unexpected refresh result: %+v", res)
	}

	entries, lastRefresh, fails := table.Stats()
	if entries != 1 || lastRefresh != 150 || fails != 0 {
		t.Fatalf("unexpected stats: entries=%d lastRefresh=%d fails=%d", entries, lastRefresh, fails)
	}

	// Query at t=200 (within TTL 150+200=350 and before EndNanos=500)
	queryProc := procFacts.Process
	queryProc.ExecGeneration = 1
	resolved := table.Resolve(queryProc, dev.UUID, 200)
	if resolved.Assignment.State != "assigned" {
		t.Fatalf("resolve failed at t=200: %+v", resolved)
	}

	// Query at t=360 (past TTL 350) -> expired
	expired := table.Resolve(queryProc, dev.UUID, 360)
	if expired.Assignment.State != "unknown" || expired.Reason != "expired_entry" {
		t.Fatalf("expected expired entry at t=360: %+v", expired)
	}

	// Bounds testing: fill capacity
	table2, err := nvtelemetry.NewJobAttributionTable(1, 1000)
	if err != nil {
		t.Fatalf("table create: %v", err)
	}
	_, err = table2.Refresh(snap, 150)
	if err != nil {
		t.Fatalf("initial refresh: %v", err)
	}

	// Add second distinct process to exceed capacity of 1
	procFacts2 := procFacts
	procFacts2.Process.Process.PID = 5678
	exec2 := exec
	exec2.Process.Process.PID = 5678
	snap2 := nvtelemetry.JobAttributionSnapshot{
		Allocations: []nvtelemetry.SlurmAllocation{job},
		Mappings:    []nvtelemetry.SlurmGPUMapping{mapping},
		Processes:   []nvtelemetry.ProcEntityFacts{procFacts, procFacts2},
		Executions:  []nvtelemetry.EntityAssignment{exec, exec2},
	}

	// Refreshing with 2 entries exceeding limit 1 should report capacity error
	_, err = table2.Refresh(snap2, 160)
	if err == nil {
		t.Errorf("expected capacity exceeded error when table is full")
	}

	// Clear table
	table.Clear()
	entries, _, _ = table.Stats()
	if entries != 0 {
		t.Errorf("expected 0 entries after clear; got %d", entries)
	}
}

func TestJobAttributionTable_Race(t *testing.T) {
	table, err := nvtelemetry.NewJobAttributionTable(100, 10000)
	if err != nil {
		t.Fatalf("create table: %v", err)
	}

	s := nvtelemetry.EntitySnapshot{
		SourceRevision: "slurm-25.05",
		NodeID:         "node-1",
		ClockDomain:    "monotonic",
		ClockEpoch:     "boot-1",
		StartNanos:     100,
		EndNanos:       100000,
	}

	var wg sync.WaitGroup
	for i := range 10 {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			jobID := fmt.Sprintf("%d", 1000+workerID)
			pid := uint32(2000 + workerID)
			devUUID := fmt.Sprintf("GPU-RACE-%d", workerID)

			job := nvtelemetry.SlurmAllocation{
				JobID:     jobID,
				Nodes:     "node-1",
				TRESAlloc: "gres/gpu=1",
				State:     []string{"RUNNING"},
				Snapshot:  s,
			}
			dev := nvtelemetry.DeviceMetadata{UUID: devUUID}
			mapping := nvtelemetry.SlurmGPUMapping{
				JobID:         jobID,
				Device:        dev,
				Sharing:       "exclusive",
				Authoritative: true,
				StartNanos:    100,
				EndNanos:      100000,
			}
			proc := nvtelemetry.ProcEntityFacts{
				Process: nvtelemetry.EntityProcess{
					Process:      cudaworkload.ProcessGenID{PID: pid, Generation: 500},
					BootID:       "boot-1",
					NodeID:       "node-1",
					PIDNamespace: "pid:[1]",
				},
				StartTimeTicks: 500,
				Cgroups: []string{
					fmt.Sprintf("0::/slurm/job_%s/step_0/task_0", jobID),
				},
			}
			exec := nvtelemetry.EntityAssignment{
				Process: nvtelemetry.EntityProcess{
					Process:           cudaworkload.ProcessGenID{PID: pid, Generation: 500},
					BootID:            "boot-1",
					NodeID:            "node-1",
					PIDNamespace:      "pid:[1]",
					ExecGeneration:    1,
					HasExecGeneration: true,
				},
				Device:         dev,
				Owner:          nvtelemetry.EntityOwner{JobID: jobID},
				Source:         "proof",
				SourceRevision: "v1",
				Authoritative:  true,
				State:          "assigned",
				ClockDomain:    "monotonic",
				ClockEpoch:     "boot-1",
				StartNanos:     100,
				EndNanos:       100000,
				Sharing:        "exclusive",
			}

			snap := nvtelemetry.JobAttributionSnapshot{
				Allocations: []nvtelemetry.SlurmAllocation{job},
				Mappings:    []nvtelemetry.SlurmGPUMapping{mapping},
				Processes:   []nvtelemetry.ProcEntityFacts{proc},
				Executions:  []nvtelemetry.EntityAssignment{exec},
			}

			for step := range 10 {
				now := int64(100 + step*10)
				_, _ = table.Refresh(snap, now)
				qp := proc.Process
				qp.ExecGeneration = 1
				_ = table.Resolve(qp, devUUID, now+5)
				_, _, _ = table.Stats()
			}
		}(i)
	}

	wg.Wait()
}
