package nvtelemetry

import (
	"fmt"
	"slices"

	"github.com/tmc/lib/nvidia/x/cudaworkload"
)

// EntityProcess extends the existing workload process generation with its
// host namespace and an independently observed execution generation. A proc
// start time establishes process birth, not an exec generation.
type EntityProcess struct {
	Process           cudaworkload.ProcessGenID `json:"process"`
	BootID            string                    `json:"boot_id"`
	NodeID            string                    `json:"node_id"`
	PIDNamespace      string                    `json:"pid_namespace,omitempty"`
	NamespacePIDs     []uint32                  `json:"namespace_pids,omitempty"`
	ExecGeneration    uint64                    `json:"exec_generation,omitempty"`
	HasExecGeneration bool                      `json:"has_exec_generation"`
}

// EntityOwner identifies an allocation recipient by stable identifiers rather
// than display names. Empty identifiers remain unknown.
type EntityOwner struct {
	PodUID       string `json:"pod_uid,omitempty"`
	ContainerID  string `json:"container_id,omitempty"`
	JobID        string `json:"job_id,omitempty"`
	AllocationID string `json:"allocation_id,omitempty"`
	Attempt      string `json:"attempt,omitempty"`
}

// EntityProvenance identifies a supplying assignment observation and its original
// validity interval in the containing assignment's clock domain and epoch.
type EntityProvenance struct {
	Source     string `json:"source"`
	Revision   string `json:"revision"`
	StartNanos int64  `json:"start_nanos"`
	EndNanos   int64  `json:"end_nanos"`
}

// EntityAssignment records a relationship valid in [StartNanos, EndNanos).
// It does not assign device consumption, power, or a kernel to its owner.
// Authoritative means the supplying metadata authority asserted the relationship;
// reading JSON does not authenticate that authority.
type EntityAssignment struct {
	Process               EntityProcess      `json:"process"`
	Device                DeviceMetadata     `json:"device"`
	Owner                 EntityOwner        `json:"owner"`
	Source                string             `json:"source"`
	SourceRevision        string             `json:"source_revision"`
	Provenance            []EntityProvenance `json:"provenance,omitempty"`
	Authoritative         bool               `json:"authoritative"`
	State                 string             `json:"state"` // assigned, unknown, or contradictory
	ClockDomain           string             `json:"clock_domain"`
	ClockEpoch            string             `json:"clock_epoch"`
	StartNanos            int64              `json:"start_nanos"`
	EndNanos              int64              `json:"end_nanos"`
	Sharing               string             `json:"sharing"` // unknown, exclusive, time_slice, mig, or mps
	MIGConfigurationEpoch string             `json:"mig_configuration_epoch,omitempty"`
	MPSRole               string             `json:"mps_role,omitempty"` // unknown, server, or client
	MPSServer             *EntityProcess     `json:"mps_server,omitempty"`
}

// Validate checks structural consistency, without upgrading metadata trust.
func (a EntityAssignment) Validate() error {
	if a.Source == "" || a.SourceRevision == "" || a.ClockDomain == "" || a.ClockEpoch == "" || a.EndNanos <= a.StartNanos {
		return fmt.Errorf("entity assignment: missing provenance or invalid interval")
	}
	if a.State != "assigned" && a.State != "unknown" && a.State != "contradictory" {
		return fmt.Errorf("entity assignment: invalid state %q", a.State)
	}
	if len(a.Provenance) > 8 {
		return fmt.Errorf("entity assignment: too many provenance observations")
	}
	for _, p := range a.Provenance {
		if p.Source == "" || p.Revision == "" || p.StartNanos > a.StartNanos || p.EndNanos < a.EndNanos || p.EndNanos <= p.StartNanos {
			return fmt.Errorf("entity assignment: invalid provenance interval")
		}
	}
	if a.State == "assigned" && (a.Device.UUID == "" || a.Owner == (EntityOwner{})) {
		return fmt.Errorf("entity assignment: missing device or owner")
	}
	switch a.Sharing {
	case "unknown", "exclusive", "time_slice", "mig", "mps":
	default:
		return fmt.Errorf("entity assignment: invalid sharing %q", a.Sharing)
	}
	if a.Sharing == "mig" && (a.Device.MIGScope != MIGScopeMIGInstance || a.Device.MIGParentUUID == "" || a.MIGConfigurationEpoch == "") {
		return fmt.Errorf("entity assignment: missing mig scope or configuration epoch")
	}
	if a.MPSRole != "" && a.MPSRole != "unknown" && a.MPSRole != "server" && a.MPSRole != "client" {
		return fmt.Errorf("entity assignment: invalid mps role")
	}
	if a.Sharing != "mps" && (a.MPSRole == "server" || a.MPSRole == "client" || a.MPSServer != nil) {
		return fmt.Errorf("entity assignment: mps metadata without mps sharing")
	}
	if !a.Process.HasExecGeneration && a.Process.ExecGeneration != 0 {
		return fmt.Errorf("entity assignment: unknown exec generation has value")
	}
	return nil
}

// EntityRelationships returns all asserted, authoritative relationships at a
// timestamp in the supplied clock epoch. Multiple results mean shared or
// ambiguous allocation, never multiple copies of whole-device consumption.
func EntityRelationships(assignments []EntityAssignment, deviceUUID, clockDomain, clockEpoch string, at int64) ([]EntityAssignment, error) {
	var result []EntityAssignment
	for _, a := range assignments {
		if err := a.Validate(); err != nil {
			return nil, err
		}
		if a.Device.UUID == deviceUUID && a.ClockDomain == clockDomain && a.ClockEpoch == clockEpoch && a.StartNanos <= at && at < a.EndNanos && a.State == "assigned" && a.Authoritative {
			result = append(result, cloneAssignment(a))
		}
	}
	return result, nil
}

func cloneAssignment(a EntityAssignment) EntityAssignment {
	a.Provenance = append([]EntityProvenance(nil), a.Provenance...)
	a.Process.NamespacePIDs = append([]uint32(nil), a.Process.NamespacePIDs...)
	if a.MPSServer != nil {
		p := *a.MPSServer
		p.NamespacePIDs = append([]uint32(nil), p.NamespacePIDs...)
		a.MPSServer = &p
	}
	a.Device.FieldCapabilities = append([]string(nil), a.Device.FieldCapabilities...)
	a.Device.Warnings = append([]string(nil), a.Device.Warnings...)
	a.Device.MetadataFailures = append([]MetadataFailure(nil), a.Device.MetadataFailures...)
	if a.Device.FieldStatuses != nil {
		m := make(map[string]string, len(a.Device.FieldStatuses))
		for k, v := range a.Device.FieldStatuses {
			m[k] = v
		}
		a.Device.FieldStatuses = m
	}
	return a
}

// ComposeEntityAssignment intersects independently supplied allocation and
// execution proofs. Both proofs must assert an authoritative assigned relation
// with the same device and an explicit stable owner ID in common. Display names,
// bare scheduler job IDs and raw opaque plugin IDs cannot establish that bridge.
// Execution identity must include independently known birth, namespace and exec
// generation. The caller, not this function, authenticates supplying authorities.
// Neither allocation nor process presence assigns device consumption or kernels.
func ComposeEntityAssignment(allocation, execution EntityAssignment) (EntityAssignment, error) {
	for _, a := range []EntityAssignment{allocation, execution} {
		if err := a.Validate(); err != nil {
			return EntityAssignment{}, err
		}
		if !a.Authoritative || a.State != "assigned" {
			return EntityAssignment{}, fmt.Errorf("entity composition: unverified relationship")
		}
	}
	if allocation.Device.UUID != execution.Device.UUID || allocation.ClockDomain != execution.ClockDomain || allocation.ClockEpoch != execution.ClockEpoch {
		return EntityAssignment{}, fmt.Errorf("entity composition: device or clock mismatch")
	}
	left, right := allocation.Owner, execution.Owner
	bridge := left.PodUID != "" && left.PodUID == right.PodUID || left.ContainerID != "" && left.ContainerID == right.ContainerID || left.AllocationID != "" && left.AllocationID == right.AllocationID && left.Attempt != "" && left.Attempt == right.Attempt
	if !bridge {
		return EntityAssignment{}, fmt.Errorf("entity composition: missing stable owner bridge")
	}
	if !completeEntityProcess(execution.Process) {
		return EntityAssignment{}, fmt.Errorf("entity composition: incomplete execution identity")
	}
	if allocation.Process.NodeID != "" && allocation.Process.NodeID != execution.Process.NodeID || allocation.Process.BootID != "" && allocation.Process.BootID != execution.Process.BootID {
		return EntityAssignment{}, fmt.Errorf("entity composition: node or boot mismatch")
	}
	if allocation.Process.Process.PID != 0 && !sameEntityProcess(allocation.Process, execution.Process) {
		return EntityAssignment{}, fmt.Errorf("entity composition: process generation mismatch")
	}
	out := cloneAssignment(execution)
	for _, pair := range [][2]*string{{&out.Owner.PodUID, &left.PodUID}, {&out.Owner.ContainerID, &left.ContainerID}, {&out.Owner.JobID, &left.JobID}, {&out.Owner.AllocationID, &left.AllocationID}, {&out.Owner.Attempt, &left.Attempt}, {&out.Device.MIGParentUUID, &allocation.Device.MIGParentUUID}, {&out.MIGConfigurationEpoch, &allocation.MIGConfigurationEpoch}} {
		if *pair[0] != "" && *pair[1] != "" && *pair[0] != *pair[1] {
			return EntityAssignment{}, fmt.Errorf("entity composition: conflicting identity")
		}
		if *pair[0] == "" {
			*pair[0] = *pair[1]
		}
	}
	if allocation.Device.HasMIGInstanceID {
		if out.Device.HasMIGInstanceID && out.Device.MIGInstanceID != allocation.Device.MIGInstanceID {
			return EntityAssignment{}, fmt.Errorf("entity composition: conflicting mig instance")
		}
		out.Device.HasMIGInstanceID = true
		out.Device.MIGInstanceID = allocation.Device.MIGInstanceID
	}
	if allocation.Device.HasMIGComputeInstanceID {
		if out.Device.HasMIGComputeInstanceID && out.Device.MIGComputeInstanceID != allocation.Device.MIGComputeInstanceID {
			return EntityAssignment{}, fmt.Errorf("entity composition: conflicting mig compute instance")
		}
		out.Device.HasMIGComputeInstanceID = true
		out.Device.MIGComputeInstanceID = allocation.Device.MIGComputeInstanceID
	}
	for _, pair := range [][2]*string{{&out.Sharing, &allocation.Sharing}, {&out.MPSRole, &allocation.MPSRole}, {&out.Device.MIGScope, &allocation.Device.MIGScope}} {
		if *pair[0] != "" && *pair[0] != "unknown" && *pair[1] != "" && *pair[1] != "unknown" && *pair[0] != *pair[1] {
			return EntityAssignment{}, fmt.Errorf("entity composition: conflicting sharing scope")
		}
		if (*pair[0] == "" || *pair[0] == "unknown") && *pair[1] != "" {
			*pair[0] = *pair[1]
		}
	}
	if allocation.MPSServer != nil {
		if out.MPSServer != nil && !sameEntityProcess(*out.MPSServer, *allocation.MPSServer) {
			return EntityAssignment{}, fmt.Errorf("entity composition: conflicting mps server")
		}
		p := *allocation.MPSServer
		p.NamespacePIDs = append([]uint32(nil), p.NamespacePIDs...)
		out.MPSServer = &p
	}
	if out.MPSServer != nil && (!completeEntityProcess(*out.MPSServer) || out.MPSServer.BootID != out.Process.BootID || out.MPSServer.NodeID != out.Process.NodeID) {
		return EntityAssignment{}, fmt.Errorf("entity composition: incomplete or mismatched mps server identity")
	}
	out.StartNanos = max(allocation.StartNanos, execution.StartNanos)
	out.EndNanos = min(allocation.EndNanos, execution.EndNanos)
	out.Source = "nvtelemetry_entity_composition"
	out.SourceRevision = "1"
	out.Provenance = nil
	for _, a := range []EntityAssignment{allocation, execution} {
		if len(a.Provenance) == 0 {
			out.Provenance = append(out.Provenance, EntityProvenance{a.Source, a.SourceRevision, a.StartNanos, a.EndNanos})
		} else {
			out.Provenance = append(out.Provenance, a.Provenance...)
		}
	}
	if err := out.Validate(); err != nil {
		return EntityAssignment{}, err
	}
	return out, nil
}

func completeEntityProcess(p EntityProcess) bool {
	return p.Process.PID != 0 && p.Process.Generation != 0 && p.BootID != "" && p.NodeID != "" && p.PIDNamespace != "" && p.HasExecGeneration
}
func sameEntityProcess(a, b EntityProcess) bool {
	return a.Process == b.Process && a.BootID == b.BootID && a.NodeID == b.NodeID && a.PIDNamespace == b.PIDNamespace && a.HasExecGeneration == b.HasExecGeneration && a.ExecGeneration == b.ExecGeneration && slices.Equal(a.NamespacePIDs, b.NamespacePIDs)
}

// TransitionEntityAssignment splits a relationship at an explicitly observed
// reassignment, process restart/exec or sharing-configuration change. The caller
// supplies the observation; a new owner or PID alone does not invent its time.
// The device and clock must be unchanged. Observations at the prior start cannot
// produce a nonempty historical interval and are refused.
func TransitionEntityAssignment(previous, next EntityAssignment, at int64) (EntityAssignment, EntityAssignment, error) {
	for _, a := range []EntityAssignment{previous, next} {
		if err := a.Validate(); err != nil {
			return EntityAssignment{}, EntityAssignment{}, err
		}
		if !a.Authoritative {
			return EntityAssignment{}, EntityAssignment{}, fmt.Errorf("entity transition: unverified observation")
		}
	}
	if previous.Device.UUID != next.Device.UUID || previous.ClockDomain != next.ClockDomain || previous.ClockEpoch != next.ClockEpoch || at <= previous.StartNanos || at > previous.EndNanos || at < next.StartNanos || at >= next.EndNanos {
		return EntityAssignment{}, EntityAssignment{}, fmt.Errorf("entity transition: incompatible observation interval")
	}
	previous = cloneAssignment(previous)
	next = cloneAssignment(next)
	previous.EndNanos = at
	next.StartNanos = at
	return previous, next, nil
}

// PluginDeviceBinding maps an opaque device-plugin ID to an explicitly supplied
// device proof. Proof must identify the node and supplying plugin revision.
// Authentication is the caller's responsibility; parsing never creates trust.
type PluginDeviceBinding struct {
	ResourceName, DeviceID string
	Proof                  EntityAssignment
}

// EntityResolution retains an unresolved observation instead of guessing a
// relationship. Reason is empty only for an assigned relationship.
type EntityResolution struct {
	Assignment EntityAssignment
	Reason     string
}

func intersectEntityProof(a, b EntityAssignment) (EntityAssignment, bool) {
	if !a.Authoritative || !b.Authoritative || a.ClockDomain != b.ClockDomain || a.ClockEpoch != b.ClockEpoch || a.Process.NodeID == "" || a.Process.NodeID != b.Process.NodeID {
		return a, false
	}
	a = cloneAssignment(a)
	if len(a.Provenance) == 0 {
		a.Provenance = []EntityProvenance{{a.Source, a.SourceRevision, a.StartNanos, a.EndNanos}}
	}
	a.StartNanos = max(a.StartNanos, b.StartNanos)
	a.EndNanos = min(a.EndNanos, b.EndNanos)
	if a.StartNanos >= a.EndNanos {
		return a, false
	}
	for _, p := range []EntityAssignment{b} {
		if len(p.Provenance) != 0 {
			a.Provenance = append(a.Provenance, p.Provenance...)
		} else {
			a.Provenance = append(a.Provenance, EntityProvenance{p.Source, p.SourceRevision, p.StartNanos, p.EndNanos})
		}
	}
	return a, a.Validate() == nil
}

func resolveEntityExecution(a EntityAssignment, proofs []EntityAssignment) EntityResolution {
	var matches []EntityAssignment
	for _, p := range proofs {
		if c, err := ComposeEntityAssignment(a, p); err == nil {
			matches = append(matches, c)
		}
	}
	if len(matches) == 1 {
		return EntityResolution{Assignment: matches[0]}
	}
	a.State = "unknown"
	reason := "missing_execution_proof"
	if len(matches) > 1 {
		a.State = "contradictory"
		reason = "ambiguous_execution_proof"
	}
	return EntityResolution{a, reason}
}

// ResolvePodResourceAssignments joins actual PodResources names with core/v1
// stable UID/container identities, authenticated plugin mappings and execution
// proofs. It returns one result per opaque device ID. Overlapping metadata or
// mappings are ambiguous even when they happen to name the same GPU.
func ResolvePodResourceAssignments(resources []PodResourceAllocation, pods []PodEntityMetadata, bindings []PluginDeviceBinding, execution []EntityAssignment) ([]EntityResolution, error) {
	if len(resources) > entityMaxRecords || len(pods) > entityMaxRecords || len(bindings) > entityMaxRecords || len(execution) > entityMaxRecords {
		return nil, fmt.Errorf("entity resolver: too many inputs")
	}
	for _, b := range bindings {
		if b.ResourceName == "" || b.DeviceID == "" || b.Proof.Device.UUID == "" {
			return nil, fmt.Errorf("entity resolver: incomplete plugin binding")
		}
		if err := b.Proof.Validate(); err != nil {
			return nil, err
		}
	}
	for _, p := range execution {
		if err := p.Validate(); err != nil {
			return nil, err
		}
	}
	for _, p := range pods {
		if err := p.Snapshot.assignment("kubernetes_core_v1").Validate(); err != nil {
			return nil, err
		}
	}
	var out []EntityResolution
	comparisons := 0
	for _, r := range resources {
		if r.Namespace == "" || r.PodName == "" || r.ContainerName == "" || r.ResourceName == "" || len(r.DeviceIDs) == 0 {
			return nil, fmt.Errorf("entity resolver: incomplete podresources observation")
		}
		base := r.Snapshot.assignment("kubelet_podresources_v1")
		if err := base.Validate(); err != nil {
			return nil, err
		}
		for _, id := range r.DeviceIDs {
			if len(out) >= entityMaxRecords || id == "" {
				return nil, fmt.Errorf("entity resolver: invalid or too many device ids")
			}
			var candidates []EntityAssignment
			for _, p := range pods {
				comparisons++
				if comparisons > 1<<20 {
					return nil, fmt.Errorf("entity resolver: comparison budget exceeded")
				}
				if p.Namespace != r.Namespace || p.PodName != r.PodName || p.ContainerName != r.ContainerName || !p.Running || p.Owner.PodUID == "" || p.Owner.ContainerID == "" || p.NodeName != base.Process.NodeID {
					continue
				}
				joined, ok := intersectEntityProof(base, p.Snapshot.assignment("kubernetes_core_v1"))
				if !ok {
					continue
				}
				joined.Owner = p.Owner
				for _, b := range bindings {
					comparisons++
					if comparisons > 1<<20 {
						return nil, fmt.Errorf("entity resolver: comparison budget exceeded")
					}
					if b.ResourceName != r.ResourceName || b.DeviceID != id || b.Proof.State == "contradictory" {
						continue
					}
					a, ok := intersectEntityProof(joined, b.Proof)
					if !ok {
						continue
					}
					a.Device = cloneAssignment(b.Proof).Device
					a.Sharing = b.Proof.Sharing
					a.MIGConfigurationEpoch = b.Proof.MIGConfigurationEpoch
					a.MPSRole = b.Proof.MPSRole
					a.MPSServer = cloneAssignment(b.Proof).MPSServer
					a.State = "assigned"
					if err := a.Validate(); err != nil {
						return nil, err
					}
					candidates = append(candidates, a)
				}
			}
			switch len(candidates) {
			case 0:
				out = append(out, EntityResolution{base, "missing_authoritative_mapping"})
			case 1:
				out = append(out, resolveEntityExecution(candidates[0], execution))
			default:
				base.State = "contradictory"
				out = append(out, EntityResolution{base, "ambiguous_authoritative_mapping"})
			}
		}
	}
	return out, nil
}

// ResolveSlurmAssignments requires a separate authoritative allocation attempt
// and device mapping. Job ID, TRES counts, environment variables and node-list
// spelling do not establish GPU or process identity. Multiple matching mappings
// remain ambiguous; callers should split observations by device when necessary.
func ResolveSlurmAssignments(jobs []SlurmAllocation, mappings, execution []EntityAssignment) ([]EntityResolution, error) {
	if len(jobs) > entityMaxRecords || len(mappings) > entityMaxRecords || len(execution) > entityMaxRecords {
		return nil, fmt.Errorf("entity resolver: too many inputs")
	}
	for _, list := range [][]EntityAssignment{mappings, execution} {
		for _, a := range list {
			if err := a.Validate(); err != nil {
				return nil, err
			}
		}
	}
	var out []EntityResolution
	for _, j := range jobs {
		base := j.Snapshot.assignment("slurm_squeue_v0.0.43")
		if err := base.Validate(); err != nil {
			return nil, err
		}
		base.Owner.JobID = j.JobID
		validState := false
		for _, state := range j.State {
			if state == "RUNNING" {
				validState = true
			}
			switch state {
			case "PENDING", "SUSPENDED", "COMPLETED", "CANCELLED", "FAILED", "TIMEOUT", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "OUT_OF_MEMORY":
				return nil, fmt.Errorf("entity resolver: inactive or contradictory scheduler state")
			}
		}
		if j.JobID == "" || !validState {
			out = append(out, EntityResolution{base, "missing_running_allocation"})
			continue
		}
		var candidates []EntityAssignment
		for _, m := range mappings {
			if m.State != "assigned" || m.Owner.JobID != j.JobID || m.Owner.AllocationID == "" || m.Owner.Attempt == "" {
				continue
			}
			a, ok := intersectEntityProof(m, base)
			if ok {
				candidates = append(candidates, a)
			}
		}
		switch len(candidates) {
		case 0:
			out = append(out, EntityResolution{base, "missing_allocation_attempt"})
		case 1:
			out = append(out, resolveEntityExecution(candidates[0], execution))
		default:
			base.State = "contradictory"
			out = append(out, EntityResolution{base, "ambiguous_allocation_attempt"})
		}
	}
	return out, nil
}

// EntityEvaluation reports results against caller-provided positive ground
// truth in corresponding slice order. No score establishes native accuracy.
type EntityEvaluation struct {
	Matched, Wrong, Missing, Ambiguous int
	Precision, Recall                  float64
	HasPrecision, HasRecall            bool
}

// EvaluateEntityResolutions evaluates bounded, independently supplied ground
// truth. Wrong exact identities return ErrIdentityMismatch with the counts.
func EvaluateEntityResolutions(truth []EntityAssignment, observed []EntityResolution) (EntityEvaluation, error) {
	var e EntityEvaluation
	if len(truth) != len(observed) || len(truth) > entityMaxRecords {
		return e, fmt.Errorf("entity evaluation: invalid dataset size")
	}
	for i, want := range truth {
		if err := want.Validate(); err != nil {
			return e, err
		}
		if !want.Authoritative || want.State != "assigned" || !completeEntityProcess(want.Process) {
			return e, fmt.Errorf("entity evaluation: incomplete ground truth")
		}
		got := observed[i].Assignment
		if err := got.Validate(); err != nil {
			return e, err
		}
		switch got.State {
		case "contradictory":
			e.Ambiguous++
		case "unknown":
			e.Missing++
		case "assigned":
			serverMatch := got.MPSServer == nil && want.MPSServer == nil || got.MPSServer != nil && want.MPSServer != nil && sameEntityProcess(*got.MPSServer, *want.MPSServer)
			if got.Authoritative && sameEntityProcess(got.Process, want.Process) && got.Owner == want.Owner && got.Device.UUID == want.Device.UUID && got.Device.MIGScope == want.Device.MIGScope && got.Device.MIGParentUUID == want.Device.MIGParentUUID && got.Device.HasMIGInstanceID == want.Device.HasMIGInstanceID && got.Device.MIGInstanceID == want.Device.MIGInstanceID && got.Device.HasMIGComputeInstanceID == want.Device.HasMIGComputeInstanceID && got.Device.MIGComputeInstanceID == want.Device.MIGComputeInstanceID && got.MIGConfigurationEpoch == want.MIGConfigurationEpoch && got.Sharing == want.Sharing && got.MPSRole == want.MPSRole && serverMatch && got.ClockDomain == want.ClockDomain && got.ClockEpoch == want.ClockEpoch && got.StartNanos >= want.StartNanos && got.EndNanos <= want.EndNanos {
				e.Matched++
			} else {
				e.Wrong++
			}
		}
	}
	if n := e.Matched + e.Wrong; n > 0 {
		e.HasPrecision = true
		e.Precision = float64(e.Matched) / float64(n)
	}
	if len(truth) > 0 {
		e.HasRecall = true
		e.Recall = float64(e.Matched) / float64(len(truth))
	}
	if e.Wrong > 0 {
		return e, ErrIdentityMismatch
	}
	return e, nil
}
