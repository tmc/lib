package nvtelemetry

// The resolver fixtures use public core/v1 and PodResources field spellings,
// but synthetic identities. They do not measure orchestrator accuracy.

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"strings"
	"sync"
	"testing"

	"github.com/tmc/lib/nvidia/nvml"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
)

func entityAssignment() EntityAssignment {
	return EntityAssignment{Process: EntityProcess{Process: cudaworkload.ProcessGenID{PID: 12, Generation: 400}, BootID: "boot", NodeID: "node", PIDNamespace: "pid:[1]", NamespacePIDs: []uint32{12, 2}, HasExecGeneration: true}, Device: DeviceMetadata{UUID: "GPU-a"}, Owner: EntityOwner{PodUID: "uid-a"}, Source: "allocation-authority", SourceRevision: "fixture-v1", Authoritative: true, State: "assigned", ClockDomain: "monotonic", ClockEpoch: "boot", StartNanos: 0, EndNanos: 100, Sharing: "time_slice"}
}

func TestEntityPublicConsumer(t *testing.T) {
	a := entityAssignment()
	b := a
	b.Owner.PodUID = "uid-b"
	var buf bytes.Buffer
	if err := WriteEntityExport(&buf, EntityExport{1, []EntityAssignment{a, b}}); err != nil {
		t.Fatal(err)
	}
	e, err := ReadEntityExport(&buf)
	if err != nil {
		t.Fatal(err)
	}
	r, err := EntityRelationships(e.Assignments, "GPU-a", "monotonic", "boot", 50)
	if err != nil || len(r) != 2 {
		t.Fatalf("overlapping relationships: %v %v", r, err)
	}
	for _, at := range []int64{-1, 100} {
		r, err := EntityRelationships(e.Assignments, "GPU-a", "monotonic", "boot", at)
		if err != nil || len(r) != 0 {
			t.Fatal("half-open interval")
		}
	}
	r, err = EntityRelationships(e.Assignments, "GPU-a", "monotonic", "other-boot", 50)
	if err != nil || len(r) != 0 {
		t.Fatal("cross epoch join")
	}
	a.Authoritative = false
	r, err = EntityRelationships([]EntityAssignment{a}, "GPU-a", "monotonic", "boot", 50)
	if err != nil || len(r) != 0 {
		t.Fatal("self report promoted")
	}
}

func TestEntityReaderNegativeControls(t *testing.T) {
	for _, s := range []string{`{"version":1,"version":1,"assignments":[]}`, `{"version":1,"Version":1,"assignments":[]}`, `{"version":2,"assignments":[]}`, `{"version":1,"assignments":[],"extra":true}`, `{"version":1,"assignments":[]} {}`, strings.Repeat("[", 33) + strings.Repeat("]", 33)} {
		if _, err := ReadEntityExport(strings.NewReader(s)); err == nil {
			t.Fatalf("accepted %s", s)
		}
	}
	a := entityAssignment()
	a.EndNanos = 0
	var b bytes.Buffer
	if err := WriteEntityExport(&b, EntityExport{1, []EntityAssignment{a}}); err == nil || b.Len() != 0 {
		t.Fatal("invalid assignment emitted")
	}
	if _, err := ReadEntityExport(io.LimitReader(strings.NewReader(strings.Repeat(" ", entityMaxBytes+1)), entityMaxBytes+1)); err == nil {
		t.Fatal("oversize accepted")
	}
}

func TestEntityValidation(t *testing.T) {
	for _, tt := range []struct {
		name   string
		mutate func(*EntityAssignment)
	}{
		{"zero-start containment", func(a *EntityAssignment) { a.EndNanos = 0 }},
		{"missing provenance", func(a *EntityAssignment) { a.SourceRevision = "" }},
		{"unknown exec with value", func(a *EntityAssignment) { a.Process.HasExecGeneration = false; a.Process.ExecGeneration = 1 }},
		{"mig missing epoch", func(a *EntityAssignment) { a.Sharing = "mig" }},
		{"mps role without sharing", func(a *EntityAssignment) { a.MPSRole = "client" }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			a := entityAssignment()
			tt.mutate(&a)
			if err := a.Validate(); err == nil {
				t.Fatal("invalid assignment accepted")
			}
		})
	}
	a := entityAssignment()
	a.Sharing = "mig"
	a.MIGConfigurationEpoch = "config-1"
	a.Device.MIGScope = MIGScopeMIGInstance
	a.Device.MIGParentUUID = "GPU-parent"
	if err := a.Validate(); err != nil {
		t.Fatal(err)
	}
	a = entityAssignment()
	a.Sharing = "mps"
	a.MPSRole = "client"
	a.MPSServer = &a.Process
	if err := a.Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestEntityCompositionConsumer(t *testing.T) {
	allocation := entityAssignment()
	allocation.Process = EntityProcess{NodeID: "node", BootID: "boot"}
	allocation.Source = "authenticated_allocation_export"
	allocation.SourceRevision = "fixture-a"
	allocation.StartNanos = 10
	allocation.EndNanos = 90
	execution := entityAssignment()
	execution.Source = "independent_process_gpu_binding"
	execution.SourceRevision = "fixture-b"
	execution.StartNanos = 20
	execution.EndNanos = 100
	composed, err := ComposeEntityAssignment(allocation, execution)
	if err != nil {
		t.Fatal(err)
	}
	if composed.StartNanos != 20 || composed.EndNanos != 90 || composed.Process.Process.PID != 12 || composed.Owner.PodUID != "uid-a" || len(composed.Provenance) != 2 || composed.Provenance[0].Source != allocation.Source || composed.Provenance[1].Revision != execution.SourceRevision {
		t.Fatalf("composition=%+v", composed)
	}
	var wire bytes.Buffer
	if err := WriteEntityExport(&wire, EntityExport{1, []EntityAssignment{composed}}); err != nil {
		t.Fatal(err)
	}
	decoded, err := ReadEntityExport(&wire)
	if err != nil {
		t.Fatal(err)
	}
	relations, err := EntityRelationships(decoded.Assignments, "GPU-a", "monotonic", "boot", 30)
	if err != nil || len(relations) != 1 || len(relations[0].Provenance) != 2 {
		t.Fatalf("consumer=%+v %v", relations, err)
	}
	for _, tt := range []struct {
		name   string
		mutate func(*EntityAssignment)
	}{
		{"self report", func(a *EntityAssignment) { a.Authoritative = false }},
		{"display name is not bridge", func(a *EntityAssignment) { a.Owner = EntityOwner{JobID: "worker"} }},
		{"pod restart", func(a *EntityAssignment) { a.Owner.PodUID = "uid-new" }},
		{"clock epoch", func(a *EntityAssignment) { a.ClockEpoch = "new" }},
		{"expired interval", func(a *EntityAssignment) { a.StartNanos = 100; a.EndNanos = 150 }},
		{"wrong device", func(a *EntityAssignment) { a.Device.UUID = "GPU-b" }},
		{"node mismatch", func(a *EntityAssignment) { a.Process.NodeID = "new-node" }},
		{"boot mismatch", func(a *EntityAssignment) { a.Process.BootID = "new-boot" }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			bad := cloneAssignment(allocation)
			tt.mutate(&bad)
			if _, err := ComposeEntityAssignment(bad, execution); err == nil {
				t.Fatal("incorrect exact identity composition")
			}
		})
	}
	// A conflict is meaningful only when both authorities supply a container ID.
	allocation.Owner.ContainerID = "container-a"
	execution.Owner.ContainerID = "container-a"
	bad := cloneAssignment(execution)
	bad.Owner.ContainerID = "container-b"
	if _, err := ComposeEntityAssignment(allocation, bad); err == nil {
		t.Fatal("container contradiction accepted")
	}
	bad = cloneAssignment(execution)
	bad.Process.HasExecGeneration = false
	if _, err := ComposeEntityAssignment(allocation, bad); err == nil {
		t.Fatal("unknown exec composed")
	}
	allocation.Owner = EntityOwner{JobID: "42"}
	execution.Owner = EntityOwner{JobID: "42"}
	if _, err := ComposeEntityAssignment(allocation, execution); err == nil {
		t.Fatal("bare reusable job id composed")
	}
}

func TestEntityMPSCompositionAndTransition(t *testing.T) {
	server := entityAssignment().Process
	server.Process = cudaworkload.ProcessGenID{PID: 99, Generation: 100}
	server.NamespacePIDs = []uint32{99}
	var assignments []EntityAssignment
	for i := range 2 {
		allocation := entityAssignment()
		allocation.Source = "mps_allocation"
		allocation.Sharing = "mps"
		allocation.MPSRole = "client"
		allocation.MPSServer = &server
		allocation.Owner.ContainerID = fmt.Sprintf("client-%d", i)
		allocation.Process = EntityProcess{NodeID: "node", BootID: "boot"}
		execution := cloneAssignment(allocation)
		execution.Source = "process_binding"
		execution.Process = entityAssignment().Process
		execution.Process.Process.PID += uint32(i)
		execution.Process.NamespacePIDs[0] += uint32(i)
		composed, err := ComposeEntityAssignment(allocation, execution)
		if err != nil {
			t.Fatal(err)
		}
		assignments = append(assignments, composed)
	}
	relationships, err := EntityRelationships(assignments, "GPU-a", "monotonic", "boot", 20)
	if err != nil || len(relationships) != 2 || relationships[0].MPSServer.Process.PID != 99 {
		t.Fatalf("mps=%+v %v", relationships, err)
	}
	bad := cloneAssignment(assignments[0])
	bad.MPSServer.HasExecGeneration = false
	if _, err := ComposeEntityAssignment(assignments[0], bad); err == nil {
		t.Fatal("unverified mps execution accepted")
	}
	previous := entityAssignment()
	next := cloneAssignment(previous)
	next.Process.ExecGeneration++
	next.Owner.PodUID = "uid-new"
	next.StartNanos = 20
	next.EndNanos = 200
	old, current, err := TransitionEntityAssignment(previous, next, 50)
	if err != nil {
		t.Fatal(err)
	}
	if old.EndNanos != 50 || current.StartNanos != 50 || current.Process.ExecGeneration != 1 {
		t.Fatal("transition did not split intervals")
	}
	for _, at := range []int64{49, 50} {
		relations, err := EntityRelationships([]EntityAssignment{old, current}, "GPU-a", "monotonic", "boot", at)
		if err != nil || len(relations) != 1 {
			t.Fatal("transition overlap")
		}
		if at == 49 && relations[0].Owner.PodUID != "uid-a" || at == 50 && relations[0].Owner.PodUID != "uid-new" {
			t.Fatal("identity boundary wrong")
		}
	}
	if _, _, err := TransitionEntityAssignment(previous, next, 0); err == nil {
		t.Fatal("empty historical interval accepted")
	}
	next.ClockEpoch = "new"
	if _, _, err := TransitionEntityAssignment(previous, next, 50); err == nil {
		t.Fatal("cross-epoch transition accepted")
	}
}

func TestEntityMIGComposition(t *testing.T) {
	allocation := entityAssignment()
	allocation.Sharing = "mig"
	allocation.MIGConfigurationEpoch = "config-1"
	allocation.Device.MIGScope = MIGScopeMIGInstance
	allocation.Device.MIGParentUUID = "GPU-parent"
	allocation.Device.HasMIGInstanceID = true
	allocation.Device.HasMIGComputeInstanceID = true
	execution := cloneAssignment(allocation)
	execution.Source = "process_binding"
	composed, err := ComposeEntityAssignment(allocation, execution)
	if err != nil || !composed.Device.HasMIGInstanceID || composed.Device.MIGInstanceID != 0 {
		t.Fatalf("zero MIG ID lost: %+v %v", composed, err)
	}
	for _, mutate := range []func(*EntityAssignment){func(a *EntityAssignment) { a.MIGConfigurationEpoch = "config-2" }, func(a *EntityAssignment) { a.Device.MIGParentUUID = "GPU-other" }, func(a *EntityAssignment) { a.Device.MIGInstanceID = 1 }, func(a *EntityAssignment) { a.Device.MIGComputeInstanceID = 1 }} {
		bad := cloneAssignment(execution)
		mutate(&bad)
		if _, err := ComposeEntityAssignment(allocation, bad); err == nil {
			t.Fatal("MIG identity mutation accepted")
		}
	}
	next := cloneAssignment(execution)
	next.MIGConfigurationEpoch = "config-2"
	old, current, err := TransitionEntityAssignment(allocation, next, 50)
	if err != nil || old.EndNanos != 50 || current.StartNanos != 50 || current.MIGConfigurationEpoch != "config-2" {
		t.Fatal("explicit MIG transition failed")
	}
}

func TestEntityGenerationCache(t *testing.T) {
	c, err := NewEntityCache(1, 10)
	if err != nil {
		t.Fatal(err)
	}
	a := entityAssignment()
	if err := c.Put([]EntityAssignment{a}, 0); err != nil {
		t.Fatal(err)
	}
	for _, tt := range []struct {
		name   string
		mutate func(*EntityAssignment)
	}{
		{"pid reuse", func(a *EntityAssignment) { a.Process.Process.Generation++ }},
		{"exec", func(a *EntityAssignment) { a.Process.ExecGeneration++ }},
		{"restart", func(a *EntityAssignment) { a.Process.BootID = "new" }},
		{"namespace", func(a *EntityAssignment) { a.Process.PIDNamespace = "pid:[2]" }},
		{"mig reconfiguration", func(a *EntityAssignment) { a.MIGConfigurationEpoch = "new" }},
		{"unknown exec", func(a *EntityAssignment) { a.Process.HasExecGeneration = false }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			q := a
			tt.mutate(&q)
			if len(c.Get(q, 1)) != 0 {
				t.Fatal("stale exact join")
			}
		})
	}
	got := c.Get(a, 1)
	got[0].Process.NamespacePIDs[0] = 99
	if c.Get(a, 1)[0].Process.NamespacePIDs[0] != 12 {
		t.Fatal("cache aliased")
	}
	if len(c.Get(a, 10)) != 0 {
		t.Fatal("ttl not enforced")
	}
	a.EndNanos = 3
	if err := c.Put([]EntityAssignment{a}, 0); err != nil {
		t.Fatal(err)
	}
	if len(c.Get(a, 3)) != 0 {
		t.Fatal("completion not expired")
	}
	if err := (&EntityCache{}).Put([]EntityAssignment{a}, 0); err == nil {
		t.Fatal("zero cache accepted")
	}
	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 30 {
				_ = c.Put([]EntityAssignment{a}, 0)
				c.Get(a, 1)
			}
		}()
	}
	wg.Wait()
	c.Clear()
	if len(c.Get(a, 1)) != 0 {
		t.Fatal("clear failed")
	}
}

func TestProcEntity(t *testing.T) {
	stat := "12 (worker ) nested) S 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 400"
	f, err := ReadProcEntity(strings.NewReader(stat), strings.NewReader("NSpid:\t12\t2\t1\n"), strings.NewReader("0::/pod/spoofed-name\n"), "boot", "node", "pid:[1]")
	if err != nil {
		t.Fatal(err)
	}
	if f.StartTimeTicks != 400 || len(f.Process.NamespacePIDs) != 3 || f.Process.HasExecGeneration {
		t.Fatalf("facts=%+v", f)
	}
	for _, bad := range []string{"NSpid: 99 2\n", "NSpid: 12 bad\n", "NSpid: 12\nNSpid: 12\n"} {
		if _, err := ReadProcEntity(strings.NewReader(stat), strings.NewReader(bad), strings.NewReader(""), "boot", "node", "pid:[1]"); err == nil {
			t.Fatal("invalid namespace accepted")
		}
	}
}

func TestVendorExportReaders(t *testing.T) {
	s := EntitySnapshot{SourceRevision: "fixture-revision", NodeID: "node", ClockDomain: "monotonic", ClockEpoch: "boot", StartNanos: 0, EndNanos: 100}
	p := `{"podResources":[{"namespace":"ns-a","name":"same","containers":[{"name":"app","devices":[{"resourceName":"nvidia.com/gpu","deviceIds":["opaque-id"]}]}]},{"namespace":"ns-b","name":"same","containers":[{"name":"app","devices":[{"resourceName":"nvidia.com/gpu","deviceIds":["opaque-id"]}]}]}]}`
	allocations, err := ReadPodResourcesJSON(strings.NewReader(p), s)
	if err != nil || len(allocations) != 2 {
		t.Fatalf("allocations %v %v", allocations, err)
	}
	if allocations[0].Namespace == allocations[1].Namespace {
		t.Fatal("duplicate names collapsed")
	}
	for _, bad := range []string{`{}`, `{"podResources":null}`, strings.ReplaceAll(p, `["opaque-id"]`, `["opaque-id","opaque-id"]`), `{"podResources":[],"podResources":[]}`} {
		if _, err := ReadPodResourcesJSON(strings.NewReader(bad), s); err == nil {
			t.Fatal("invalid pod export accepted")
		}
	}
	j := `{"jobs":[{"job_id":42,"job_state":["RUNNING"],"nodes":"node[1-2]","tres_alloc_str":"gres/gpu=2"},{"job_id":43,"job_state":["COMPLETED"]}]}`
	jobs, err := ReadSlurmSqueueJSON(strings.NewReader(j), s)
	if err != nil || len(jobs) != 1 || jobs[0].JobID != "42" {
		t.Fatalf("jobs %v %v", jobs, err)
	}
	for _, bad := range []string{`{"SLURM_JOB_ID":"42"}`, `{"jobs":[{"job_id":0}]}`, `{"jobs":[{"job_id":1.5}]}`, `{"jobs":[{"job_id":4294967296}]}`} {
		if _, err := ReadSlurmSqueueJSON(strings.NewReader(bad), s); err == nil {
			t.Fatal("spoofed or invalid scheduler accepted")
		}
	}
	for _, bad := range []string{
		`{"jobs":[{"job_id":42,"job_state":["RUNNING","COMPLETED"],"nodes":"node1","tres_alloc_str":"gres/gpu=1"}]}`,
		`{"jobs":[{"job_id":42,"job_state":["RUNNING"],"nodes":"node1","tres_alloc_str":"gres/gpu=1"},{"job_id":42,"job_state":["COMPLETED"]}]}`,
		`{"jobs":[{"job_id":42,"job_state":["RUNNING","RUNNING"],"nodes":"node1","tres_alloc_str":"gres/gpu=1"}]}`,
	} {
		if _, err := ReadSlurmSqueueJSON(strings.NewReader(bad), s); err == nil {
			t.Fatal("contradictory scheduler allocation admitted")
		}
	}
}

func TestEntityDocumentedFixtures(t *testing.T) {
	s := EntitySnapshot{SourceRevision: "synthetic-schema-fixtures-v1", NodeID: "node", ClockDomain: "monotonic", ClockEpoch: "boot", EndNanos: 100}
	p, err := os.Open("testdata/entity/podresources-v1.json")
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	allocations, err := ReadPodResourcesJSON(p, s)
	if err != nil || len(allocations) != 2 || allocations[0].Namespace == allocations[1].Namespace {
		t.Fatalf("pod fixture: %v %v", allocations, err)
	}
	j, err := os.Open("testdata/entity/squeue-v0.0.43.json")
	if err != nil {
		t.Fatal(err)
	}
	defer j.Close()
	jobs, err := ReadSlurmSqueueJSON(j, s)
	if err != nil || len(jobs) != 1 || jobs[0].JobID != "42" {
		t.Fatalf("slurm fixture: %v %v", jobs, err)
	}
}

func TestNVMLProcessFacts(t *testing.T) {
	s := EntitySnapshot{SourceRevision: "nvml-v3-fixture", ClockDomain: "monotonic", ClockEpoch: "boot", StartNanos: 0, EndNanos: 100}
	processes := []nvml.ProcessInfo{{Pid: 12, UsedGpuMemory: ^uint64(0), GpuInstanceId: ^uint32(0), ComputeInstanceId: ^uint32(0)}}
	f, err := AdaptNVMLProcesses(DeviceMetadata{UUID: "GPU-a"}, processes, "nvml_mps_compute_v3", s)
	if err != nil {
		t.Fatal(err)
	}
	if f[0].HasUsedMemory || f[0].HasGPUInstanceID || f[0].HasComputeInstanceID {
		t.Fatal("sentinel promoted")
	}
	processes[0].UsedGpuMemory = 0
	processes[0].GpuInstanceId = 0
	f, err = AdaptNVMLProcesses(DeviceMetadata{UUID: "GPU-a"}, processes, "nvml_compute_v3", s)
	if err != nil || !f[0].HasUsedMemory || !f[0].HasGPUInstanceID {
		t.Fatal("measured zero lost")
	}
	p := processes[0]
	p.GpuInstanceId = 1
	f, err = AdaptNVMLProcesses(DeviceMetadata{UUID: "GPU-a"}, append(processes, p), "nvml_compute_v3", s)
	if err != nil || len(f) != 2 {
		t.Fatalf("process across MIG instances: %v %v", f, err)
	}
}

func FuzzReadEntityExport(f *testing.F) {
	f.Add(`{"version":1,"assignments":[]}`)
	f.Add(`{"version":1,"version":1}`)
	f.Fuzz(func(t *testing.T, s string) {
		if len(s) > entityMaxBytes+1 {
			t.Skip()
		}
		e, err := ReadEntityExport(strings.NewReader(s))
		if err == nil {
			var b bytes.Buffer
			if err := WriteEntityExport(&b, e); err != nil {
				t.Fatal(err)
			}
			if _, err := ReadEntityExport(&b); err != nil {
				t.Fatal(err)
			}
		}
	})
}

func ExampleAdaptNVMLProcesses() {
	s := EntitySnapshot{SourceRevision: "synthetic-v3", ClockDomain: "monotonic", ClockEpoch: "boot", EndNanos: 1}
	f, err := AdaptNVMLProcesses(DeviceMetadata{UUID: "GPU-a"}, []nvml.ProcessInfo{{Pid: 12, UsedGpuMemory: ^uint64(0), GpuInstanceId: ^uint32(0), ComputeInstanceId: ^uint32(0)}}, "nvml_compute_v3", s)
	if err != nil {
		panic(err)
	}
	fmt.Println(f[0].PID, f[0].HasUsedMemory)
	// Output: 12 false
}

func TestAuthoritativeEntityResolvers(t *testing.T) {
	s := EntitySnapshot{SourceRevision: "fixture-v1", Authoritative: true, NodeID: "node", ClockDomain: "monotonic", ClockEpoch: "boot", StartNanos: 0, EndNanos: 100}
	pods, err := ReadKubernetesPodsJSON(strings.NewReader(`{"apiVersion":"v1","kind":"PodList","items":[{"metadata":{"name":"p","namespace":"ns","uid":"uid-a"},"spec":{"nodeName":"node"},"status":{"containerStatuses":[{"name":"c","containerID":"containerd://abc","state":{"running":{}}}]}}]}`), s)
	if err != nil || len(pods) != 1 {
		t.Fatalf("pods: %v %v", pods, err)
	}
	resources := []PodResourceAllocation{{Namespace: "ns", PodName: "p", ContainerName: "c", ResourceName: "nvidia.com/gpu", DeviceIDs: []string{"opaque"}, Snapshot: s}}
	exec := entityAssignment()
	exec.Owner.ContainerID = "containerd://abc"
	device := s.assignment("authenticated_plugin")
	device.Device = exec.Device
	bindings := []PluginDeviceBinding{{"nvidia.com/gpu", "opaque", device}}
	got, err := ResolvePodResourceAssignments(resources, pods, bindings, []EntityAssignment{exec})
	if err != nil || len(got) != 1 || got[0].Assignment.State != "assigned" || len(got[0].Assignment.Provenance) != 4 {
		t.Fatalf("resolve: %+v %v", got, err)
	}
	score, err := EvaluateEntityResolutions([]EntityAssignment{exec}, got)
	if err != nil || score.Matched != 1 || score.Precision != 1 || score.Recall != 1 {
		t.Fatalf("score: %+v %v", score, err)
	}
	for _, name := range []string{"untrusted", "uid_missing", "ambiguous_map", "unknown_exec", "expired", "wrong_epoch"} {
		t.Run(name, func(t *testing.T) {
			pp := append([]PodEntityMetadata(nil), pods...)
			bb := append([]PluginDeviceBinding(nil), bindings...)
			ee := cloneAssignment(exec)
			switch name {
			case "untrusted":
				bb[0].Proof.Authoritative = false
			case "uid_missing":
				pp[0].Owner.PodUID = ""
			case "ambiguous_map":
				bb = append(bb, bb[0])
			case "unknown_exec":
				ee.Process.HasExecGeneration = false
			case "expired":
				bb[0].Proof.StartNanos = 100
				bb[0].Proof.EndNanos = 200
			case "wrong_epoch":
				bb[0].Proof.ClockEpoch = "other"
			}
			r, err := ResolvePodResourceAssignments(resources, pp, bb, []EntityAssignment{ee})
			if err != nil {
				t.Fatal(err)
			}
			if r[0].Assignment.State == "assigned" {
				t.Fatalf("promoted: %+v", r)
			}
		})
	}
	wrong := got[0]
	wrong.Assignment.Process.ExecGeneration++
	score, err = EvaluateEntityResolutions([]EntityAssignment{exec}, []EntityResolution{wrong})
	if err != ErrIdentityMismatch || score.Wrong != 1 {
		t.Fatalf("wrong join: %+v %v", score, err)
	}
	job := SlurmAllocation{JobID: "42", Nodes: "node[1-2]", TRESAlloc: "gres/gpu=1", State: []string{"RUNNING"}, Snapshot: s}
	mapping := cloneAssignment(exec)
	mapping.Process = EntityProcess{NodeID: "node"}
	mapping.Owner = EntityOwner{JobID: "42", AllocationID: "allocation42", Attempt: "attempt1"}
	ee := cloneAssignment(exec)
	ee.Owner = mapping.Owner
	r, err := ResolveSlurmAssignments([]SlurmAllocation{job}, []EntityAssignment{mapping}, []EntityAssignment{ee})
	if err != nil || r[0].Assignment.State != "assigned" {
		t.Fatalf("slurm: %+v %v", r, err)
	}
	mapping.Owner.Attempt = ""
	r, err = ResolveSlurmAssignments([]SlurmAllocation{job}, []EntityAssignment{mapping}, []EntityAssignment{ee})
	if err != nil || r[0].Assignment.State != "unknown" {
		t.Fatalf("bare job: %+v %v", r, err)
	}
}

func TestKubernetesPodReaderRefusal(t *testing.T) {
	s := EntitySnapshot{SourceRevision: "v1", ClockDomain: "mono", ClockEpoch: "boot", StartNanos: 0, EndNanos: 1}
	for _, raw := range []string{`{"apiVersion":"v1","kind":"PodList"}`, `{"apiVersion":"v2","kind":"PodList","items":[]}`, `{"apiVersion":"v1","kind":"Pod","metadata":{"namespace":"ns","name":"p"},"status":{"containerStatuses":[{"name":"c"},{"name":"c"}]}}`} {
		if _, err := ReadKubernetesPodsJSON(strings.NewReader(raw), s); err == nil {
			t.Fatalf("accepted %s", raw)
		}
	}
}
