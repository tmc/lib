package nvtelemetry

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"strconv"
	"strings"
)

const entityMaxBytes = 4 << 20
const entityMaxRecords = 4096

// EntityExport is the version 1 allocation relationship interchange document.
// It is not a Kubernetes, Slurm, or NVML native response.
type EntityExport struct {
	Version     int                `json:"version"`
	Assignments []EntityAssignment `json:"assignments"`
}

// ReadEntityExport reads a bounded strict version 1 interchange document.
func ReadEntityExport(r io.Reader) (EntityExport, error) {
	var e EntityExport
	if err := readEntityJSON(r, &e, true); err != nil {
		return e, err
	}
	if e.Version != 1 || len(e.Assignments) > entityMaxRecords {
		return EntityExport{}, fmt.Errorf("entity export: unsupported version or too many assignments")
	}
	for _, a := range e.Assignments {
		if err := a.Validate(); err != nil {
			return EntityExport{}, err
		}
	}
	return e, nil
}

// WriteEntityExport validates and writes a bounded version 1 document.
func WriteEntityExport(w io.Writer, e EntityExport) error {
	if e.Version != 1 || len(e.Assignments) > entityMaxRecords {
		return fmt.Errorf("entity export: unsupported version or too many assignments")
	}
	for _, a := range e.Assignments {
		if err := a.Validate(); err != nil {
			return err
		}
	}
	b, err := json.Marshal(e)
	if err != nil {
		return fmt.Errorf("entity export: %w", err)
	}
	if len(b)+1 > entityMaxBytes {
		return fmt.Errorf("entity export: document too large")
	}
	b = append(b, '\n')
	n, err := w.Write(b)
	if err != nil {
		return err
	}
	if n != len(b) {
		return io.ErrShortWrite
	}
	return nil
}

func readEntityJSON(r io.Reader, value any, strict bool) error {
	b, err := io.ReadAll(io.LimitReader(r, entityMaxBytes+1))
	if err != nil {
		return fmt.Errorf("entity export: %w", err)
	}
	if len(b) > entityMaxBytes {
		return fmt.Errorf("entity export: document too large")
	}
	if err := checkEntityJSON(json.NewDecoder(bytes.NewReader(b)), 0); err != nil {
		return err
	}
	d := json.NewDecoder(bytes.NewReader(b))
	if strict {
		d.DisallowUnknownFields()
	}
	if err := d.Decode(value); err != nil {
		return fmt.Errorf("entity export: %w", err)
	}
	var extra any
	if err := d.Decode(&extra); err != io.EOF {
		return fmt.Errorf("entity export: trailing document")
	}
	return nil
}

func checkEntityJSON(d *json.Decoder, depth int) error {
	if depth > 32 {
		return fmt.Errorf("entity export: json nesting too deep")
	}
	t, err := d.Token()
	if err != nil {
		return fmt.Errorf("entity export: %w", err)
	}
	delim, ok := t.(json.Delim)
	if !ok {
		return nil
	}
	switch delim {
	case '{':
		seen := make(map[string]bool)
		for d.More() {
			k, err := d.Token()
			if err != nil {
				return err
			}
			name, ok := k.(string)
			name = strings.ToLower(name)
			if !ok || seen[name] {
				return fmt.Errorf("entity export: duplicate or invalid json key")
			}
			seen[name] = true
			if err := checkEntityJSON(d, depth+1); err != nil {
				return err
			}
		}
	case '[':
		for d.More() {
			if err := checkEntityJSON(d, depth+1); err != nil {
				return err
			}
		}
	default:
		return fmt.Errorf("entity export: invalid json delimiter")
	}
	_, err = d.Token()
	return err
}

// EntitySnapshot supplies independently collected provenance and observation
// validity to vendor exports, whose content does not authenticate the collector.
type EntitySnapshot struct {
	SourceRevision string
	Authoritative  bool
	NodeID         string
	ClockDomain    string
	ClockEpoch     string
	StartNanos     int64
	EndNanos       int64
}

func (s EntitySnapshot) assignment(source string) EntityAssignment {
	return EntityAssignment{Process: EntityProcess{NodeID: s.NodeID}, Source: source, SourceRevision: s.SourceRevision, Authoritative: s.Authoritative, State: "unknown", ClockDomain: s.ClockDomain, ClockEpoch: s.ClockEpoch, StartNanos: s.StartNanos, EndNanos: s.EndNanos, Sharing: "unknown"}
}

// PodResourceAllocation preserves PodResources v1 List response fields.
// The API has pod names, not pod UIDs, and device-plugin IDs are opaque.
// Consequently this reader never turns a display name into an exact identity.
type PodResourceAllocation struct {
	Namespace     string
	PodName       string
	ContainerName string
	ResourceName  string
	DeviceIDs     []string
	Snapshot      EntitySnapshot
}

// ReadPodResourcesJSON reads an external protobuf-JSON rendering of Kubernetes
// v1 ListPodResourcesResponse using its lowerCamelCase field names. It accepts
// unconsumed CPU, memory and topology fields but does not interpret them.
// It does not read the gRPC socket or establish a pod UID or physical GPU UUID.
func ReadPodResourcesJSON(r io.Reader, snapshot EntitySnapshot) ([]PodResourceAllocation, error) {
	if err := snapshot.assignment("kubelet_podresources_v1").Validate(); err != nil {
		return nil, err
	}
	var response struct {
		Pods []struct {
			Name       string `json:"name"`
			Namespace  string `json:"namespace"`
			Containers []struct {
				Name    string `json:"name"`
				Devices []struct {
					ResourceName string   `json:"resourceName"`
					DeviceIDs    []string `json:"deviceIds"`
				} `json:"devices"`
			} `json:"containers"`
		} `json:"podResources"`
	}
	if err := readEntityJSON(r, &response, false); err != nil {
		return nil, err
	}
	if response.Pods == nil {
		return nil, fmt.Errorf("podresources: missing podResources array")
	}
	if len(response.Pods) > entityMaxRecords {
		return nil, fmt.Errorf("podresources: too many pods")
	}
	var out []PodResourceAllocation
	for _, p := range response.Pods {
		if len(p.Containers) > entityMaxRecords {
			return nil, fmt.Errorf("podresources: too many containers")
		}
		for _, c := range p.Containers {
			if len(c.Devices) > entityMaxRecords {
				return nil, fmt.Errorf("podresources: too many devices")
			}
			for _, d := range c.Devices {
				if p.Name == "" || p.Namespace == "" || c.Name == "" || d.ResourceName == "" || len(d.DeviceIDs) == 0 {
					return nil, fmt.Errorf("podresources: missing allocation identity")
				}
				if len(out) >= entityMaxRecords || len(d.DeviceIDs) > entityMaxRecords {
					return nil, fmt.Errorf("podresources: too many allocations")
				}
				seen := make(map[string]bool)
				for _, id := range d.DeviceIDs {
					if id == "" || seen[id] {
						return nil, fmt.Errorf("podresources: missing or duplicate device id")
					}
					seen[id] = true
				}
				out = append(out, PodResourceAllocation{p.Namespace, p.Name, c.Name, d.ResourceName, append([]string(nil), d.DeviceIDs...), snapshot})
			}
		}
	}
	return out, nil
}

// SlurmAllocation preserves a bounded squeue JSON allocation observation.
// TRESAlloc describes counts/types, not GPU UUIDs or CUDA process identity.
// Job IDs from process environment variables are not accepted by this reader.
type SlurmAllocation struct {
	JobID     string
	Nodes     string
	TRESAlloc string
	State     []string
	Snapshot  EntitySnapshot
}

// ReadSlurmSqueueJSON reads jobs[].job_id, nodes, tres_alloc_str and job_state
// from externally supplied squeue --json data_parser v0.0.43 output. Other
// fields remain unconsumed. Only running jobs carry an active allocation.
// The caller must pin the supplying scheduler revision in snapshot.
func ReadSlurmSqueueJSON(r io.Reader, snapshot EntitySnapshot) ([]SlurmAllocation, error) {
	if err := snapshot.assignment("slurm_squeue_v0.0.43").Validate(); err != nil {
		return nil, err
	}
	var response struct {
		Jobs []struct {
			JobID json.Number `json:"job_id"`
			Nodes string      `json:"nodes"`
			TRES  string      `json:"tres_alloc_str"`
			State []string    `json:"job_state"`
		} `json:"jobs"`
	}
	if err := readEntityJSON(r, &response, false); err != nil {
		return nil, err
	}
	if response.Jobs == nil {
		return nil, fmt.Errorf("squeue: missing jobs array")
	}
	if len(response.Jobs) > entityMaxRecords {
		return nil, fmt.Errorf("squeue: too many jobs")
	}
	var out []SlurmAllocation
	seenJobs := make(map[uint64]bool)
	for _, j := range response.Jobs {
		id, err := strconv.ParseUint(string(j.JobID), 10, 32)
		if err != nil || id == 0 {
			return nil, fmt.Errorf("squeue: invalid job id")
		}
		if seenJobs[id] {
			return nil, fmt.Errorf("squeue: duplicate job id")
		}
		seenJobs[id] = true
		running := false
		baseStates := 0
		seenStates := make(map[string]bool)
		for _, state := range j.State {
			if seenStates[state] {
				return nil, fmt.Errorf("squeue: duplicate job state")
			}
			seenStates[state] = true
			switch state {
			case "PENDING", "RUNNING", "SUSPENDED", "COMPLETED", "CANCELLED", "FAILED", "TIMEOUT", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "OUT_OF_MEMORY":
				baseStates++
			}
			if state == "RUNNING" {
				running = true
			}
		}
		if baseStates > 1 {
			return nil, fmt.Errorf("squeue: contradictory job states")
		}
		if !running {
			continue
		}
		if j.Nodes == "" || j.TRES == "" {
			return nil, fmt.Errorf("squeue: missing allocation fields")
		}
		out = append(out, SlurmAllocation{strconv.FormatUint(id, 10), j.Nodes, j.TRES, append([]string(nil), j.State...), snapshot})
	}
	return out, nil
}

// PodEntityMetadata preserves core/v1 Pod identity and container status. A
// container name alone does not establish an execution identity.
type PodEntityMetadata struct {
	Namespace, PodName, ContainerName, NodeName string
	Owner                                       EntityOwner
	Running                                     bool
	Snapshot                                    EntitySnapshot
}

// ReadKubernetesPodsJSON reads a caller-supplied core/v1 Pod or PodList.
// Credentials, freshness and authority are supplied by the caller. Unknown
// fields are ignored; omitted UID, container ID and running status stay unknown.
func ReadKubernetesPodsJSON(r io.Reader, snapshot EntitySnapshot) ([]PodEntityMetadata, error) {
	if err := snapshot.assignment("kubernetes_core_v1").Validate(); err != nil {
		return nil, err
	}
	type pod struct {
		Metadata struct{ UID, Name, Namespace string } `json:"metadata"`
		Spec     struct {
			NodeName string `json:"nodeName"`
		} `json:"spec"`
		Status struct {
			Containers []struct {
				Name        string `json:"name"`
				ContainerID string `json:"containerID"`
				State       struct {
					Running json.RawMessage `json:"running"`
				} `json:"state"`
			} `json:"containerStatuses"`
		} `json:"status"`
	}
	var raw struct {
		APIVersion string `json:"apiVersion"`
		Kind       string `json:"kind"`
		Items      []pod  `json:"items"`
		pod
	}
	if err := readEntityJSON(r, &raw, false); err != nil {
		return nil, err
	}
	if raw.APIVersion != "v1" {
		return nil, fmt.Errorf("kubernetes pods: unsupported api version")
	}
	var pods []pod
	switch raw.Kind {
	case "Pod":
		pods = []pod{raw.pod}
	case "PodList":
		if raw.Items == nil {
			return nil, fmt.Errorf("kubernetes pods: missing items")
		}
		pods = raw.Items
	default:
		return nil, fmt.Errorf("kubernetes pods: unsupported kind")
	}
	if len(pods) > entityMaxRecords {
		return nil, fmt.Errorf("kubernetes pods: too many pods")
	}
	var out []PodEntityMetadata
	for _, p := range pods {
		if p.Metadata.Name == "" || p.Metadata.Namespace == "" {
			return nil, fmt.Errorf("kubernetes pods: missing name or namespace")
		}
		seen := make(map[string]bool)
		for _, c := range p.Status.Containers {
			if c.Name == "" || seen[c.Name] {
				return nil, fmt.Errorf("kubernetes pods: missing or duplicate container")
			}
			seen[c.Name] = true
			if len(out) >= entityMaxRecords {
				return nil, fmt.Errorf("kubernetes pods: too many containers")
			}
			running := len(c.State.Running) != 0 && string(c.State.Running) != "null"
			if running && c.State.Running[0] != '{' {
				return nil, fmt.Errorf("kubernetes pods: invalid running state")
			}
			out = append(out, PodEntityMetadata{p.Metadata.Namespace, p.Metadata.Name, c.Name, p.Spec.NodeName, EntityOwner{PodUID: p.Metadata.UID, ContainerID: c.ContainerID}, running, snapshot})
		}
	}
	return out, nil
}
