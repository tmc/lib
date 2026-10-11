package nvtelemetry_test

import (
	"bytes"
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudaworkload"
	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func ExampleReadEntityExport() {
	a := nvtelemetry.EntityAssignment{Device: nvtelemetry.DeviceMetadata{UUID: "GPU-a"}, Owner: nvtelemetry.EntityOwner{PodUID: "uid-a"}, Source: "allocation_authority", SourceRevision: "fixture-v1", Authoritative: true, State: "assigned", ClockDomain: "monotonic", ClockEpoch: "boot-a", StartNanos: 10, EndNanos: 30, Sharing: "time_slice"}
	var b bytes.Buffer
	if err := nvtelemetry.WriteEntityExport(&b, nvtelemetry.EntityExport{Version: 1, Assignments: []nvtelemetry.EntityAssignment{a}}); err != nil {
		panic(err)
	}
	export, err := nvtelemetry.ReadEntityExport(&b)
	if err != nil {
		panic(err)
	}
	relationships, err := nvtelemetry.EntityRelationships(export.Assignments, "GPU-a", "monotonic", "boot-a", 20)
	if err != nil {
		panic(err)
	}
	fmt.Println(len(relationships), relationships[0].Owner.PodUID)
	// Output: 1 uid-a
}

func ExampleReadPodResourcesJSON() {
	s := nvtelemetry.EntitySnapshot{SourceRevision: "kubernetes-v1.32.0", NodeID: "node-a", ClockDomain: "monotonic", ClockEpoch: "boot-a", StartNanos: 10, EndNanos: 20}
	r := strings.NewReader(`{"podResources":[{"namespace":"default","name":"worker","containers":[{"name":"app","devices":[{"resourceName":"nvidia.com/gpu","deviceIds":["GPU-a"]}]}]}]}`)
	allocations, err := nvtelemetry.ReadPodResourcesJSON(r, s)
	if err != nil {
		panic(err)
	}
	fmt.Println(allocations[0].PodName, allocations[0].DeviceIDs[0])
	// Output: worker GPU-a
}

func ExampleReadProcEntity() {
	stat := strings.NewReader("12 (worker ) task) S 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 400")
	facts, err := nvtelemetry.ReadProcEntity(stat, strings.NewReader("NSpid:\t12\t2\n"), strings.NewReader("0::/tenant/worker\n"), "boot-a", "node-a", "pid:[100]")
	if err != nil {
		panic(err)
	}
	fmt.Println(facts.Process.Process.PID, facts.StartTimeTicks, facts.Process.HasExecGeneration)
	// Output: 12 400 false
}

func ExampleNewEntityCache() {
	c, err := nvtelemetry.NewEntityCache(2, 10)
	if err != nil {
		panic(err)
	}
	a := nvtelemetry.EntityAssignment{Process: nvtelemetry.EntityProcess{Process: cudaworkload.ProcessGenID{PID: 12, Generation: 400}, BootID: "boot-a", NodeID: "node-a", PIDNamespace: "pid:[100]", HasExecGeneration: true}, Device: nvtelemetry.DeviceMetadata{UUID: "GPU-a"}, Owner: nvtelemetry.EntityOwner{JobID: "42"}, Source: "scheduler", SourceRevision: "fixture-v1", Authoritative: true, State: "assigned", ClockDomain: "monotonic", ClockEpoch: "boot-a", StartNanos: 10, EndNanos: 30, Sharing: "time_slice"}
	if err := c.Put([]nvtelemetry.EntityAssignment{a}, 10); err != nil {
		panic(err)
	}
	fmt.Println(len(c.Get(a, 19)), len(c.Get(a, 20)))
	// Output: 1 0
}

func ExampleComposeEntityAssignment() {
	allocation := nvtelemetry.EntityAssignment{Device: nvtelemetry.DeviceMetadata{UUID: "GPU-a"}, Owner: nvtelemetry.EntityOwner{PodUID: "verified-uid"}, Source: "allocation_export", SourceRevision: "fixture-a", Authoritative: true, State: "assigned", ClockDomain: "monotonic", ClockEpoch: "boot-a", StartNanos: 10, EndNanos: 50, Sharing: "unknown"}
	execution := allocation
	execution.Source = "process_binding"
	execution.SourceRevision = "fixture-b"
	execution.StartNanos = 20
	execution.Process = nvtelemetry.EntityProcess{Process: cudaworkload.ProcessGenID{PID: 12, Generation: 400}, BootID: "boot-a", NodeID: "node-a", PIDNamespace: "pid:[100]", HasExecGeneration: true}
	joined, err := nvtelemetry.ComposeEntityAssignment(allocation, execution)
	if err != nil {
		panic(err)
	}
	fmt.Println(joined.StartNanos, joined.EndNanos, len(joined.Provenance), joined.Sharing)
	// Output: 20 50 2 unknown
}

func ExampleTransitionEntityAssignment() {
	previous := nvtelemetry.EntityAssignment{Device: nvtelemetry.DeviceMetadata{UUID: "GPU-a"}, Owner: nvtelemetry.EntityOwner{PodUID: "uid-a"}, Source: "allocation_export", SourceRevision: "fixture-v1", Authoritative: true, State: "assigned", ClockDomain: "monotonic", ClockEpoch: "boot-a", EndNanos: 100, Sharing: "unknown"}
	next := previous
	next.Owner.PodUID = "uid-b"
	next.EndNanos = 200
	old, current, err := nvtelemetry.TransitionEntityAssignment(previous, next, 50)
	if err != nil {
		panic(err)
	}
	fmt.Println(old.EndNanos, current.StartNanos, current.Owner.PodUID)
	// Output: 50 50 uid-b
}

func ExampleReadKubernetesPodsJSON() {
	s := nvtelemetry.EntitySnapshot{SourceRevision: "v1.32.0", ClockDomain: "mono", ClockEpoch: "boot", StartNanos: 0, EndNanos: 10}
	pods, err := nvtelemetry.ReadKubernetesPodsJSON(strings.NewReader(`{"apiVersion":"v1","kind":"Pod","metadata":{"name":"worker","namespace":"default","uid":"pod-uid"},"spec":{"nodeName":"node"},"status":{"containerStatuses":[{"name":"worker","containerID":"containerd://abc","state":{"running":{}}}]}}`), s)
	fmt.Println(len(pods), pods[0].Owner.PodUID, err)
	// Output: 1 pod-uid <nil>
}

func ExampleResolvePodResourceAssignments() {
	s := nvtelemetry.EntitySnapshot{SourceRevision: "v1.32.0", ClockDomain: "mono", ClockEpoch: "boot", StartNanos: 0, EndNanos: 10}
	r := nvtelemetry.PodResourceAllocation{Namespace: "default", PodName: "worker", ContainerName: "worker", ResourceName: "nvidia.com/gpu", DeviceIDs: []string{"opaque"}, Snapshot: s}
	result, err := nvtelemetry.ResolvePodResourceAssignments([]nvtelemetry.PodResourceAllocation{r}, nil, nil, nil)
	fmt.Println(result[0].Assignment.State, result[0].Reason, err)
	// Output: unknown missing_authoritative_mapping <nil>
}

func ExampleResolveSlurmAssignments() {
	s := nvtelemetry.EntitySnapshot{SourceRevision: "25.05", ClockDomain: "mono", ClockEpoch: "boot", StartNanos: 0, EndNanos: 10}
	result, err := nvtelemetry.ResolveSlurmAssignments([]nvtelemetry.SlurmAllocation{{JobID: "42", State: []string{"RUNNING"}, Snapshot: s}}, nil, nil)
	fmt.Println(result[0].Assignment.State, result[0].Reason, err)
	// Output: unknown missing_allocation_attempt <nil>
}

func ExampleEvaluateEntityResolutions() {
	result, err := nvtelemetry.EvaluateEntityResolutions(nil, nil)
	fmt.Println(result.HasPrecision, result.HasRecall, err)
	// Output: false false <nil>
}
