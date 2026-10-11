package nvtelemetry_test

import (
	w "github.com/tmc/lib/nvidia/x/cudaworkload"
	n "github.com/tmc/lib/nvidia/x/nvtelemetry"
	"math"
	"sync"
	"testing"
)

func independentSnapshot() n.JobAttributionSnapshot {
	p := n.EntityProcess{Process: w.ProcessGenID{PID: 21, Generation: 17}, BootID: "boot", NodeID: "node", PIDNamespace: "ns", ExecGeneration: 2, HasExecGeneration: true, NamespacePIDs: []uint32{21}}
	d := n.DeviceMetadata{UUID: "GPU-1", Index: 0}
	return n.JobAttributionSnapshot{
		Allocations: []n.SlurmAllocation{{JobID: "42", State: []string{"RUNNING"}, TRESAlloc: "gres/gpu=1", Snapshot: n.EntitySnapshot{SourceRevision: "synthetic-control", Authoritative: true, NodeID: "node", ClockDomain: "mono", ClockEpoch: "epoch", StartNanos: 100, EndNanos: 1000}}},
		Mappings:    []n.SlurmGPUMapping{{JobID: "42", StepID: "0", TaskID: "1", Device: d, Sharing: "exclusive", Authoritative: true, AllocationID: "alloc-1", Attempt: "1", ClockDomain: "mono", ClockEpoch: "epoch", StartNanos: 100, EndNanos: 1000}},
		Processes:   []n.ProcEntityFacts{{Process: p, StartTimeTicks: 17, Cgroups: []string{"0::/system.slice/slurmstepd.scope/job_42/step_0/task_1"}}},
		Executions:  []n.EntityAssignment{{Process: p, Device: d, Owner: n.EntityOwner{JobID: "42", AllocationID: "alloc-1", Attempt: "1"}, Source: "observed-execution-control", SourceRevision: "1", Authoritative: true, State: "assigned", Sharing: "exclusive", ClockDomain: "mono", ClockEpoch: "epoch", StartNanos: 100, EndNanos: 1000}},
	}
}
func independentJoin(t *testing.T, s n.JobAttributionSnapshot, at int64) []n.JobAttributionResult {
	t.Helper()
	r, e := n.JoinSlurmAttribution(s.Allocations, s.Mappings, s.Processes, s.Executions, at)
	if e != nil {
		t.Logf("refused: %v", e)
		return nil
	}
	return r
}
func independentAssigned(r []n.JobAttributionResult) bool {
	for _, v := range r {
		if v.Assignment.State == "assigned" {
			return true
		}
	}
	return false
}
func TestIndependentPositiveAndSharing(t *testing.T) {
	for _, sharing := range []string{"exclusive", "time_slice", "mps", "unknown", "mig"} {
		t.Run(sharing, func(t *testing.T) {
			s := independentSnapshot()
			s.Mappings[0].Sharing = sharing
			s.Executions[0].Sharing = sharing
			if sharing == "mig" {
				s.Mappings[0].Device.MIGScope = n.MIGScopeMIGInstance
				s.Mappings[0].Device.MIGParentUUID = "GPU-parent"
				s.Mappings[0].MIGConfigurationEpoch = "mig-epoch"
				s.Executions[0].Device = s.Mappings[0].Device
				s.Executions[0].MIGConfigurationEpoch = "mig-epoch"
			}
			r := independentJoin(t, s, 200)
			if len(r) != 1 || r[0].Assignment.State != "assigned" || r[0].Assignment.Sharing != sharing {
				t.Fatalf("valid supplied-input relationship lost: %+v", r)
			}
			if e := r[0].Assignment.Validate(); e != nil {
				t.Fatal(e)
			}
		})
	}
}
func TestIndependentWithholdMutations(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*n.JobAttributionSnapshot)
		at     int64
	}{
		{"known_exec_generation_conflict", func(s *n.JobAttributionSnapshot) { s.Processes[0].Process.ExecGeneration = 3 }, 200},
		{"missing_proc_namespace", func(s *n.JobAttributionSnapshot) { s.Processes[0].Process.PIDNamespace = "" }, 200},
		{"mapping_epoch", func(s *n.JobAttributionSnapshot) { s.Mappings[0].ClockEpoch = "other" }, 200},
		{"execution_epoch", func(s *n.JobAttributionSnapshot) { s.Executions[0].ClockEpoch = "other" }, 200},
		{"mapping_domain", func(s *n.JobAttributionSnapshot) { s.Mappings[0].ClockDomain = "wall" }, 200},
		{"execution_namespace", func(s *n.JobAttributionSnapshot) { s.Executions[0].Process.PIDNamespace = "other" }, 200},
		{"unknown_execution_state", func(s *n.JobAttributionSnapshot) { s.Executions[0].State = "unknown" }, 200},
		{"contradictory_execution_state", func(s *n.JobAttributionSnapshot) { s.Executions[0].State = "contradictory" }, 200},
		{"reused_allocation", func(s *n.JobAttributionSnapshot) { s.Mappings[0].AllocationID = "alloc-2" }, 200},
		{"reused_job_attempt", func(s *n.JobAttributionSnapshot) { s.Mappings[0].Attempt = "2" }, 200},
		{"wrong_execution_owner", func(s *n.JobAttributionSnapshot) { s.Executions[0].Owner.JobID = "99" }, 200},
		{"expired_mapping", func(s *n.JobAttributionSnapshot) { s.Mappings[0].EndNanos = 150 }, 200},
		{"disjoint_intervals", func(s *n.JobAttributionSnapshot) {
			s.Allocations[0].Snapshot.StartNanos = 300
			s.Mappings[0].EndNanos = 200
		}, 500},
		{"zero_gpu_count", func(s *n.JobAttributionSnapshot) { s.Allocations[0].TRESAlloc = "gres/gpu=0" }, 200},
		{"missing_step_identity", func(s *n.JobAttributionSnapshot) { s.Processes[0].Cgroups = []string{"0::/slurm/job_42"} }, 200},
		{"allocation_node_conflict", func(s *n.JobAttributionSnapshot) { s.Allocations[0].Snapshot.NodeID = "other" }, 200},
		{"proc_birth_conflict", func(s *n.JobAttributionSnapshot) { s.Processes[0].StartTimeTicks = 999 }, 200},
		{"mig_epoch_conflict", func(s *n.JobAttributionSnapshot) {
			s.Mappings[0].Sharing = "mig"
			s.Mappings[0].Device.MIGScope = n.MIGScopeMIGInstance
			s.Mappings[0].Device.MIGParentUUID = "GPU-parent"
			s.Mappings[0].MIGConfigurationEpoch = "new"
			s.Executions[0].Sharing = "mig"
			s.Executions[0].Device = s.Mappings[0].Device
			s.Executions[0].MIGConfigurationEpoch = "old"
		}, 200},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			s := independentSnapshot()
			c.mutate(&s)
			r := independentJoin(t, s, c.at)
			if independentAssigned(r) {
				t.Fatalf("unsafe mutation accepted as assigned: %+v", r)
			}
		})
	}
}
func TestIndependentParserBoundaries(t *testing.T) {
	for _, p := range []string{"0::/system.slice/slurmstepd.scope/job_42/step_batch/task_0", "/slurm/job_42/step_extern", "/slurm/job_42/step_0/task_1"} {
		if _, e := n.ParseSlurmCgroupPath(p); e != nil {
			t.Error(e)
		}
	}
	for _, p := range []string{"0::/slurm/job_42/../job_99", "0::/slurm/job_42/step_0/task_1\n0::/slurm/job_99/step_0/task_1", "/sys/fs/cgroup/not-a-slurm/job_42/step_0/task_1\x00", "1:devices:/slurm/job_42"} {
		if _, e := n.ParseSlurmCgroupPath(p); e == nil {
			t.Errorf("invalid cgroup accepted %q", p)
		}
	}
	if _, e := n.ParseSlurmGRES("cpu=4,gres/gpu=2,gres/gpu:a100=2"); e != nil {
		t.Errorf("aggregate plus typed GRES refused: %v", e)
	}
}
func TestIndependentCacheInvalidation(t *testing.T) {
	t.Run("successful_empty_refresh", func(t *testing.T) {
		s := independentSnapshot()
		tab, _ := n.NewJobAttributionTable(4, 500)
		_, e := tab.Refresh(s, 200)
		if e != nil {
			t.Fatal(e)
		}
		_, e = tab.Refresh(n.JobAttributionSnapshot{}, 210)
		if e != nil {
			t.Fatal(e)
		}
		if tab.Resolve(s.Processes[0].Process, "GPU-1", 220).Assignment.State == "assigned" {
			t.Fatal("successful empty snapshot retained old assignment")
		}
	})
	t.Run("contradiction_refresh", func(t *testing.T) {
		s := independentSnapshot()
		tab, _ := n.NewJobAttributionTable(4, 500)
		_, _ = tab.Refresh(s, 200)
		s.Executions = append(s.Executions, s.Executions[0])
		r, e := tab.Refresh(s, 210)
		if e != nil {
			t.Fatal(e)
		}
		if independentAssigned(r) {
			t.Fatal("join control failed")
		}
		if tab.Resolve(s.Processes[0].Process, "GPU-1", 220).Assignment.State == "assigned" {
			t.Fatal("contradictory refresh kept assigned cache")
		}
	})
	t.Run("capacity_failure_atomic", func(t *testing.T) {
		s := independentSnapshot()
		tab, _ := n.NewJobAttributionTable(1, 500)
		_, _ = tab.Refresh(s, 200)
		s.Mappings[0].AllocationID = "new"
		s.Executions[0].Owner.AllocationID = "new"
		s.Processes = append(s.Processes, s.Processes[0])
		s.Processes[1].Process.Process.PID = 22
		s.Executions = append(s.Executions, s.Executions[0])
		s.Executions[1].Process.Process.PID = 22
		_, e := tab.Refresh(s, 210)
		if e == nil {
			t.Fatal("expected capacity failure")
		}
		if got := tab.Resolve(s.Processes[0].Process, "GPU-1", 220); got.Assignment.Owner.AllocationID != "alloc-1" {
			t.Fatalf("failed refresh partially mutated cache: %+v", got)
		}
	})
	t.Run("returned_result_alias", func(t *testing.T) {
		s := independentSnapshot()
		tab, _ := n.NewJobAttributionTable(1, 500)
		r, e := tab.Refresh(s, 200)
		if e != nil {
			t.Fatal(e)
		}
		r[0].Assignment.Process.NamespacePIDs[0] = 999
		r[0].Assignment.Provenance[0].Revision = "corrupt"
		got := tab.Resolve(s.Processes[0].Process, "GPU-1", 220)
		if got.Assignment.Process.NamespacePIDs[0] == 999 || got.Assignment.Provenance[0].Revision == "corrupt" {
			t.Fatalf("public result mutated cache: %+v", got)
		}
	})
	t.Run("ttl_overflow", func(t *testing.T) {
		s := independentSnapshot()
		s.Allocations[0].Snapshot.StartNanos = math.MaxInt64 - 100
		s.Allocations[0].Snapshot.EndNanos = math.MaxInt64
		s.Mappings[0].StartNanos = s.Allocations[0].Snapshot.StartNanos
		s.Mappings[0].EndNanos = math.MaxInt64
		s.Executions[0].StartNanos = s.Mappings[0].StartNanos
		s.Executions[0].EndNanos = math.MaxInt64
		tab, _ := n.NewJobAttributionTable(1, 500)
		_, e := tab.Refresh(s, math.MaxInt64-90)
		if e != nil {
			t.Fatal(e)
		}
		if tab.Resolve(s.Processes[0].Process, "GPU-1", math.MaxInt64-80).Assignment.State != "assigned" {
			t.Fatal("ttl arithmetic overflow expired valid entry")
		}
	})
}
func TestIndependentHealthyConcurrency(t *testing.T) {
	s := independentSnapshot()
	tab, _ := n.NewJobAttributionTable(4, 500)
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 10; j++ {
				if _, e := tab.Refresh(s, 200); e != nil {
					t.Error(e)
				}
				_ = tab.Resolve(s.Processes[0].Process, "GPU-1", 210)
				tab.Stats()
			}
		}()
	}
	wg.Wait()
}

func TestIndependentTrustAndSharing(t *testing.T) {
	t.Run("allocation_trust", func(t *testing.T) {
		s := independentSnapshot()
		s.Allocations[0].Snapshot.Authoritative = false
		r := independentJoin(t, s, 200)
		for _, v := range r {
			if v.Assignment.State == "assigned" && v.Assignment.Authoritative {
				t.Fatal("allocation trust upgraded")
			}
		}
	})
	t.Run("execution_trust", func(t *testing.T) {
		s := independentSnapshot()
		s.Executions[0].Authoritative = false
		r := independentJoin(t, s, 200)
		for _, v := range r {
			if v.Assignment.State == "assigned" && v.Assignment.Authoritative {
				t.Fatal("execution trust upgraded")
			}
		}
	})
	t.Run("shared_execution_not_exclusive", func(t *testing.T) {
		s := independentSnapshot()
		s.Executions[0].Sharing = "mps"
		r := independentJoin(t, s, 200)
		for _, v := range r {
			if v.Assignment.State == "assigned" && v.Assignment.Sharing == "exclusive" {
				t.Fatal("shared execution upgraded to exclusive")
			}
		}
	})
	t.Run("pid_generation", func(t *testing.T) {
		s := independentSnapshot()
		s.Executions[0].Process.Process.Generation++
		if independentAssigned(independentJoin(t, s, 200)) {
			t.Fatal("pid birth reuse accepted")
		}
	})
	t.Run("expiry", func(t *testing.T) {
		s := independentSnapshot()
		tab, _ := n.NewJobAttributionTable(1, 10)
		_, _ = tab.Refresh(s, 200)
		if tab.Resolve(s.Processes[0].Process, "GPU-1", 211).Assignment.State == "assigned" {
			t.Fatal("expired cache accepted")
		}
	})
}

func TestIndependentConflictingMappings(t *testing.T) {
	t.Run("same_device_two_owners", func(t *testing.T) {
		s := independentSnapshot()
		m := s.Mappings[0]
		m.AllocationID = "alloc-2"
		s.Mappings = append(s.Mappings, m)
		if independentAssigned(independentJoin(t, s, 200)) {
			t.Fatal("conflicting assignments for same process/device returned assigned")
		}
	})
	t.Run("allocation_conflict_then_duplicate", func(t *testing.T) {
		s := independentSnapshot()
		a := s.Allocations[0]
		a.TRESAlloc = "gres/gpu=2"
		s.Allocations = append(s.Allocations, a, s.Allocations[0])
		if independentAssigned(independentJoin(t, s, 200)) {
			t.Fatal("later duplicate cleared prior allocation contradiction")
		}
	})
	t.Run("mps_server_contradiction", func(t *testing.T) {
		s := independentSnapshot()
		s.Mappings[0].Sharing = "mps"
		s.Mappings[0].MPSRole = "client"
		server := s.Processes[0].Process
		server.Process.PID = 80
		s.Mappings[0].MPSServer = &server
		s.Executions[0].Sharing = "mps"
		s.Executions[0].MPSRole = "client"
		other := server
		other.Process.PID = 90
		s.Executions[0].MPSServer = &other
		if independentAssigned(independentJoin(t, s, 200)) {
			t.Fatal("conflicting MPS server accepted")
		}
	})
	t.Run("vgpu_unsupported_refusal", func(t *testing.T) {
		s := independentSnapshot()
		s.Mappings[0].Sharing = "vgpu"
		if independentAssigned(independentJoin(t, s, 200)) {
			t.Fatal("unknown virtualization precision asserted")
		}
	})
}
