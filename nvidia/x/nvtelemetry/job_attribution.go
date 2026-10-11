package nvtelemetry

import (
	"fmt"
	"math"
	"slices"
	"strconv"
	"strings"
	"sync"
)

// SlurmCgroupIdentity records the parsed job, step, and task IDs extracted
// from a Slurm cgroup v2 hierarchy path.
type SlurmCgroupIdentity struct {
	JobID      string `json:"job_id"`
	StepID     string `json:"step_id,omitempty"`
	TaskID     string `json:"task_id,omitempty"`
	CgroupPath string `json:"cgroup_path"`
}

// ParseSlurmCgroupPath parses a unified cgroup v2 path or /proc/$pid/cgroup
// line into a SlurmCgroupIdentity. It requires a valid job_<jobid> component,
// enforces strict hierarchy ordering (job before step, step before task), and
// rejects cgroup v1 entries or contradictory paths.
func ParseSlurmCgroupPath(path string) (SlurmCgroupIdentity, error) {
	if len(path) == 0 {
		return SlurmCgroupIdentity{}, fmt.Errorf("slurm cgroup: empty path")
	}
	if len(path) > entityMaxBytes {
		return SlurmCgroupIdentity{}, fmt.Errorf("slurm cgroup: path exceeds maximum length")
	}
	if strings.ContainsAny(path, "\x00\r\n") {
		return SlurmCgroupIdentity{}, fmt.Errorf("slurm cgroup: invalid control characters in path")
	}

	raw := strings.TrimSpace(path)
	toParse := raw

	// Check for /proc/$pid/cgroup format (hierarchy:subsystems:path).
	if strings.Contains(toParse, ":") {
		parts := strings.SplitN(toParse, ":", 3)
		if len(parts) != 3 {
			return SlurmCgroupIdentity{}, fmt.Errorf("slurm cgroup: invalid cgroup format %q", path)
		}
		// In cgroup v2 unified hierarchy, the hierarchy ID is 0 and controller name is empty.
		if parts[0] != "0" || parts[1] != "" {
			return SlurmCgroupIdentity{}, fmt.Errorf("slurm cgroup: not a cgroup v2 path (hierarchy=%q, controllers=%q)", parts[0], parts[1])
		}
		toParse = parts[2]
	}

	toParse = strings.TrimPrefix(toParse, "/sys/fs/cgroup")
	if !strings.HasPrefix(toParse, "/") {
		return SlurmCgroupIdentity{}, fmt.Errorf("slurm cgroup: path must be absolute")
	}

	segments := strings.Split(toParse, "/")
	var jobID, stepID, taskID string
	var seenJob, seenStep, seenTask bool

	for _, seg := range segments {
		if seg == "" {
			continue
		}
		if seg == "." || seg == ".." {
			return SlurmCgroupIdentity{}, fmt.Errorf("slurm cgroup: relative path components not allowed")
		}

		if strings.HasPrefix(seg, "job_") {
			if seenJob {
				return SlurmCgroupIdentity{}, fmt.Errorf("slurm cgroup: multiple job segments in path")
			}
			id := strings.TrimPrefix(seg, "job_")
			if id == "" {
				return SlurmCgroupIdentity{}, fmt.Errorf("slurm cgroup: empty job id")
			}
			num, err := strconv.ParseUint(id, 10, 64)
			if err != nil || num == 0 {
				return SlurmCgroupIdentity{}, fmt.Errorf("slurm cgroup: invalid job id %q", id)
			}
			jobID = id
			seenJob = true
			continue
		}

		if strings.HasPrefix(seg, "step_") {
			if !seenJob {
				return SlurmCgroupIdentity{}, fmt.Errorf("slurm cgroup: step segment before job segment")
			}
			if seenStep {
				return SlurmCgroupIdentity{}, fmt.Errorf("slurm cgroup: multiple step segments in path")
			}
			sid := strings.TrimPrefix(seg, "step_")
			if sid == "" {
				return SlurmCgroupIdentity{}, fmt.Errorf("slurm cgroup: empty step id")
			}
			for _, r := range sid {
				if !((r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '_') {
					return SlurmCgroupIdentity{}, fmt.Errorf("slurm cgroup: invalid step id %q", sid)
				}
			}
			stepID = sid
			seenStep = true
			continue
		}

		if strings.HasPrefix(seg, "task_") {
			if !seenStep {
				return SlurmCgroupIdentity{}, fmt.Errorf("slurm cgroup: task segment without step segment")
			}
			if seenTask {
				return SlurmCgroupIdentity{}, fmt.Errorf("slurm cgroup: multiple task segments in path")
			}
			tid := strings.TrimPrefix(seg, "task_")
			if tid == "" {
				return SlurmCgroupIdentity{}, fmt.Errorf("slurm cgroup: empty task id")
			}
			if _, err := strconv.ParseUint(tid, 10, 64); err != nil {
				return SlurmCgroupIdentity{}, fmt.Errorf("slurm cgroup: invalid task id %q", tid)
			}
			taskID = tid
			seenTask = true
			continue
		}
	}

	if !seenJob {
		return SlurmCgroupIdentity{}, fmt.Errorf("slurm cgroup: missing job segment")
	}

	return SlurmCgroupIdentity{
		JobID:      jobID,
		StepID:     stepID,
		TaskID:     taskID,
		CgroupPath: raw,
	}, nil
}

// ResolveSlurmProcessCgroup extracts and validates Slurm cgroup v2 membership
// from a process's ProcEntityFacts. It returns an error if no valid Slurm cgroup
// is found or if contradictory memberships are listed.
func ResolveSlurmProcessCgroup(facts ProcEntityFacts) (SlurmCgroupIdentity, error) {
	if facts.Process.Process.PID == 0 || facts.Process.BootID == "" || facts.Process.NodeID == "" {
		return SlurmCgroupIdentity{}, fmt.Errorf("slurm cgroup: incomplete process identity")
	}
	var resolved SlurmCgroupIdentity
	var found bool

	for _, cg := range facts.Cgroups {
		id, err := ParseSlurmCgroupPath(cg)
		if err != nil {
			continue
		}
		if found {
			if id.JobID != resolved.JobID || id.StepID != resolved.StepID || id.TaskID != resolved.TaskID {
				return SlurmCgroupIdentity{}, fmt.Errorf("slurm cgroup: contradictory slurm cgroup memberships")
			}
		} else {
			resolved = id
			found = true
		}
	}

	if !found {
		return SlurmCgroupIdentity{}, fmt.Errorf("slurm cgroup: no slurm cgroup v2 membership")
	}
	return resolved, nil
}

// SlurmGRESAllocation represents parsed GRES GPU resources from Slurm TRESAlloc.
type SlurmGRESAllocation struct {
	Count     int      `json:"count"`
	GPUType   string   `json:"gpu_type,omitempty"`
	DeviceIDs []string `json:"device_ids,omitempty"`
	Raw       string   `json:"raw"`
}

// ParseSlurmGRES parses the GRES GPU component from a Slurm tres_alloc_str.
// Contradictory, duplicate, or malformed GPU allocations return an error. Aggregate
// counts and typed counts that specialize or match the aggregate count are reconciled.
func ParseSlurmGRES(tresAlloc string) (SlurmGRESAllocation, error) {
	if len(tresAlloc) == 0 {
		return SlurmGRESAllocation{}, nil
	}
	if len(tresAlloc) > entityMaxBytes {
		return SlurmGRESAllocation{}, fmt.Errorf("slurm gres: tres string too large")
	}

	fields := strings.Split(tresAlloc, ",")
	type parsedItem struct {
		gpuType   string
		count     int
		deviceIDs []string
		raw       string
	}
	var items []parsedItem

	for _, field := range fields {
		field = strings.TrimSpace(field)
		if !strings.HasPrefix(field, "gres/gpu") {
			continue
		}

		rem := strings.TrimPrefix(field, "gres/gpu")
		var gpuType string
		var countStr string
		var deviceIDs []string

		if strings.HasPrefix(rem, "=") {
			countStr = strings.TrimPrefix(rem, "=")
		} else if strings.HasPrefix(rem, ":") {
			colonPart := strings.TrimPrefix(rem, ":")
			if eqIdx := strings.IndexByte(colonPart, '='); eqIdx >= 0 {
				gpuType = colonPart[:eqIdx]
				countStr = colonPart[eqIdx+1:]
			} else {
				// Format gres/gpu:0 or gres/gpu:type:0
				parts := strings.Split(colonPart, ":")
				if len(parts) == 1 {
					deviceIDs = []string{parts[0]}
					countStr = "1"
				} else if len(parts) == 2 {
					gpuType = parts[0]
					deviceIDs = []string{parts[1]}
					countStr = "1"
				} else {
					return SlurmGRESAllocation{}, fmt.Errorf("slurm gres: invalid gres format %q", field)
				}
			}
		} else {
			return SlurmGRESAllocation{}, fmt.Errorf("slurm gres: invalid gres format %q", field)
		}

		c, err := strconv.Atoi(countStr)
		if err != nil || c < 0 {
			return SlurmGRESAllocation{}, fmt.Errorf("slurm gres: invalid gpu count in %q", field)
		}

		items = append(items, parsedItem{
			gpuType:   gpuType,
			count:     c,
			deviceIDs: deviceIDs,
			raw:       field,
		})
	}

	if len(items) == 0 {
		return SlurmGRESAllocation{}, nil
	}
	if len(items) == 1 {
		it := items[0]
		return SlurmGRESAllocation{
			Count:     it.count,
			GPUType:   it.gpuType,
			DeviceIDs: it.deviceIDs,
			Raw:       it.raw,
		}, nil
	}

	// Multiple GRES fields: check for duplicate keys, integer overflow, and aggregate specialization.
	var aggCount int
	var hasAgg bool
	var typedCount int
	var typedType string
	var allDeviceIDs []string
	seenKeys := make(map[string]bool)

	for _, it := range items {
		allDeviceIDs = append(allDeviceIDs, it.deviceIDs...)
		if it.gpuType == "" && len(it.deviceIDs) == 0 {
			if hasAgg {
				return SlurmGRESAllocation{}, fmt.Errorf("slurm gres: conflicting tres allocation %q", tresAlloc)
			}
			aggCount = it.count
			hasAgg = true
		} else {
			key := it.gpuType
			if len(it.deviceIDs) > 0 {
				key += ":" + strings.Join(it.deviceIDs, ",")
			}
			if seenKeys[key] {
				return SlurmGRESAllocation{}, fmt.Errorf("slurm gres: duplicate typed resource %q", key)
			}
			seenKeys[key] = true

			if it.count > math.MaxInt-typedCount {
				return SlurmGRESAllocation{}, fmt.Errorf("slurm gres: integer overflow in gpu count")
			}
			typedCount += it.count
			if typedType == "" {
				typedType = it.gpuType
			} else if typedType != it.gpuType {
				typedType = "mixed"
			}
		}
	}

	if hasAgg {
		if typedCount > 0 && typedCount != aggCount {
			return SlurmGRESAllocation{}, fmt.Errorf("slurm gres: conflicting tres allocation %q", tresAlloc)
		}
		resType := typedType
		if resType == "mixed" {
			resType = ""
		}
		return SlurmGRESAllocation{
			Count:     aggCount,
			GPUType:   resType,
			DeviceIDs: allDeviceIDs,
			Raw:       tresAlloc,
		}, nil
	}

	return SlurmGRESAllocation{
		Count:     typedCount,
		GPUType:   typedType,
		DeviceIDs: allDeviceIDs,
		Raw:       tresAlloc,
	}, nil
}

// SlurmGPUMapping specifies an explicit device assignment for a Slurm job
// or step. Authentication and freshness are the supplying authority's responsibility.
type SlurmGPUMapping struct {
	JobID                 string         `json:"job_id"`
	StepID                string         `json:"step_id,omitempty"`
	TaskID                string         `json:"task_id,omitempty"`
	Device                DeviceMetadata `json:"device"`
	Sharing               string         `json:"sharing,omitempty"` // exclusive, time_slice, mig, mps, unknown
	MIGConfigurationEpoch string         `json:"mig_configuration_epoch,omitempty"`
	MPSRole               string         `json:"mps_role,omitempty"`
	MPSServer             *EntityProcess `json:"mps_server,omitempty"`
	AllocationID          string         `json:"allocation_id,omitempty"`
	Attempt               string         `json:"attempt,omitempty"`
	Authoritative         bool           `json:"authoritative"`
	ClockDomain           string         `json:"clock_domain,omitempty"`
	ClockEpoch            string         `json:"clock_epoch,omitempty"`
	StartNanos            int64          `json:"start_nanos,omitempty"`
	EndNanos              int64          `json:"end_nanos,omitempty"`
}

// Validate verifies structural validity of a SlurmGPUMapping.
func (m SlurmGPUMapping) Validate() error {
	if m.JobID == "" || m.Device.UUID == "" {
		return fmt.Errorf("slurm gpu mapping: missing job id or device uuid")
	}
	switch m.Sharing {
	case "", "unknown", "exclusive", "time_slice", "mig", "mps", "vgpu":
	default:
		return fmt.Errorf("slurm gpu mapping: invalid sharing %q", m.Sharing)
	}
	if m.MPSRole != "" && m.MPSRole != "unknown" && m.MPSRole != "server" && m.MPSRole != "client" {
		return fmt.Errorf("slurm gpu mapping: invalid mps role")
	}
	if m.Sharing != "mps" && (m.MPSRole == "server" || m.MPSRole == "client" || m.MPSServer != nil) {
		return fmt.Errorf("slurm gpu mapping: mps metadata without mps sharing")
	}
	return nil
}

// JobAttributionResult captures the outcome of joining a process/cgroup
// to an allocation. The Reason field describes why attribution was withheld
// whenever Assignment.State is not "assigned".
type JobAttributionResult struct {
	Assignment EntityAssignment    `json:"assignment"`
	Reason     string              `json:"reason,omitempty"`
	Cgroup     SlurmCgroupIdentity `json:"cgroup,omitempty"`
}

// JobAttributionSnapshot contains the inputs for an attribution refresh cycle.
type JobAttributionSnapshot struct {
	Allocations []SlurmAllocation  `json:"allocations,omitempty"`
	Mappings    []SlurmGPUMapping  `json:"mappings,omitempty"`
	Processes   []ProcEntityFacts  `json:"processes,omitempty"`
	Executions  []EntityAssignment `json:"executions,omitempty"`
}

type jobAllocationFacts struct {
	alloc       SlurmAllocation
	gres        SlurmGRESAllocation
	hasGRES     bool
	zeroGPU     bool
	conflicting bool
	stale       bool
}

// JoinSlurmAttribution performs a pure, replayable process/cgroup-to-allocation join
// across the supplied allocations, device mappings, process facts, and execution proofs
// at timestamp at.
func JoinSlurmAttribution(allocations []SlurmAllocation, mappings []SlurmGPUMapping, procs []ProcEntityFacts, executions []EntityAssignment, at int64) ([]JobAttributionResult, error) {
	if len(allocations) > entityMaxRecords || len(mappings) > entityMaxRecords || len(procs) > entityMaxRecords || len(executions) > entityMaxRecords {
		return nil, fmt.Errorf("job attribution: too many records")
	}

	for _, e := range executions {
		if err := e.Validate(); err != nil {
			return nil, err
		}
	}
	for _, m := range mappings {
		if err := m.Validate(); err != nil {
			return nil, err
		}
	}

	// Index allocations by JobID.
	allocMap := make(map[string]jobAllocationFacts, len(allocations))
	for _, a := range allocations {
		if a.JobID == "" {
			return nil, fmt.Errorf("job attribution: missing allocation job id")
		}
		if err := a.Snapshot.assignment("slurm_squeue_v0.0.43").Validate(); err != nil {
			return nil, err
		}

		gres, err := ParseSlurmGRES(a.TRESAlloc)
		hasGRES := (err == nil && gres.Count > 0)
		zeroGPU := (err == nil && strings.Contains(a.TRESAlloc, "gres/gpu") && gres.Count == 0)
		hasConflict := (err != nil)

		running := false
		stale := false
		for _, s := range a.State {
			if s == "RUNNING" {
				running = true
			}
			switch s {
			case "PENDING", "SUSPENDED", "COMPLETED", "CANCELLED", "FAILED", "TIMEOUT", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "OUT_OF_MEMORY":
				stale = true
			}
		}
		if !running || (at < a.Snapshot.StartNanos) || (a.Snapshot.EndNanos > 0 && at >= a.Snapshot.EndNanos) {
			stale = true
		}

		if existing, ok := allocMap[a.JobID]; ok {
			conflicting := existing.conflicting || hasConflict ||
				existing.alloc.TRESAlloc != a.TRESAlloc ||
				!slices.Equal(existing.alloc.State, a.State) ||
				existing.alloc.Snapshot.ClockDomain != a.Snapshot.ClockDomain ||
				existing.alloc.Snapshot.ClockEpoch != a.Snapshot.ClockEpoch ||
				existing.alloc.Snapshot.NodeID != a.Snapshot.NodeID ||
				existing.alloc.Snapshot.StartNanos != a.Snapshot.StartNanos ||
				existing.alloc.Snapshot.EndNanos != a.Snapshot.EndNanos
			existing.conflicting = conflicting
			allocMap[a.JobID] = existing
			continue
		}

		allocMap[a.JobID] = jobAllocationFacts{
			alloc:       a,
			gres:        gres,
			hasGRES:     hasGRES,
			zeroGPU:     zeroGPU,
			conflicting: hasConflict,
			stale:       stale,
		}
	}

	var results []JobAttributionResult

	for _, proc := range procs {
		baseProc := proc.Process

		// Check for birth conflict or missing namespace.
		if proc.StartTimeTicks != 0 && baseProc.Process.Generation != 0 && proc.StartTimeTicks != baseProc.Process.Generation {
			results = append(results, JobAttributionResult{
				Assignment: EntityAssignment{Process: baseProc, State: "unknown"},
				Reason:     "generation_mismatch",
			})
			continue
		}
		if baseProc.PIDNamespace == "" {
			results = append(results, JobAttributionResult{
				Assignment: EntityAssignment{Process: baseProc, State: "unknown"},
				Reason:     "incomplete_process_identity",
			})
			continue
		}

		cg, err := ResolveSlurmProcessCgroup(proc)
		if err != nil {
			results = append(results, JobAttributionResult{
				Assignment: EntityAssignment{Process: baseProc, State: "unknown"},
				Reason:     "missing_slurm_cgroup",
			})
			continue
		}

		af, ok := allocMap[cg.JobID]
		if !ok {
			results = append(results, JobAttributionResult{
				Assignment: EntityAssignment{Process: baseProc, State: "unknown"},
				Reason:     "missing_allocation",
				Cgroup:     cg,
			})
			continue
		}

		if af.conflicting || af.zeroGPU {
			results = append(results, JobAttributionResult{
				Assignment: EntityAssignment{Process: baseProc, State: "contradictory"},
				Reason:     "conflicting_tres",
				Cgroup:     cg,
			})
			continue
		}

		if af.stale {
			results = append(results, JobAttributionResult{
				Assignment: EntityAssignment{Process: baseProc, State: "unknown"},
				Reason:     "stale_allocation",
				Cgroup:     cg,
			})
			continue
		}

		if af.alloc.Snapshot.NodeID != "" && baseProc.NodeID != "" && af.alloc.Snapshot.NodeID != baseProc.NodeID {
			results = append(results, JobAttributionResult{
				Assignment: EntityAssignment{Process: baseProc, State: "unknown"},
				Reason:     "node_mismatch",
				Cgroup:     cg,
			})
			continue
		}

		// Find device mappings for this job/step.
		var matchingMappings []SlurmGPUMapping
		var unidentifiableMIG bool
		var unsupportedSharing bool
		var clockMismatch bool
		var clockMissing bool

		for _, m := range mappings {
			if m.JobID != cg.JobID {
				continue
			}
			// Mapping scoped to step must match cgroup step.
			if m.StepID != "" && m.StepID != cg.StepID {
				continue
			}
			// Mapping scoped to task must match cgroup task.
			if m.TaskID != "" && m.TaskID != cg.TaskID {
				continue
			}

			if m.Sharing == "vgpu" {
				unsupportedSharing = true
				continue
			}

			// Mapping interval timestamps require verified non-empty ClockDomain and ClockEpoch.
			if m.StartNanos != 0 || m.EndNanos != 0 {
				if m.ClockDomain == "" || m.ClockEpoch == "" {
					clockMissing = true
					continue
				}
			}

			if m.ClockDomain != "" && af.alloc.Snapshot.ClockDomain != "" && m.ClockDomain != af.alloc.Snapshot.ClockDomain {
				clockMismatch = true
				continue
			}
			if m.ClockEpoch != "" && af.alloc.Snapshot.ClockEpoch != "" && m.ClockEpoch != af.alloc.Snapshot.ClockEpoch {
				clockMismatch = true
				continue
			}

			if m.Sharing == "mig" {
				if m.Device.MIGScope != MIGScopeMIGInstance || m.Device.MIGParentUUID == "" || m.MIGConfigurationEpoch == "" {
					unidentifiableMIG = true
					continue
				}
			}
			matchingMappings = append(matchingMappings, m)
		}

		if clockMissing {
			results = append(results, JobAttributionResult{
				Assignment: EntityAssignment{Process: baseProc, State: "unknown"},
				Reason:     "missing_clock_provenance",
				Cgroup:     cg,
			})
			continue
		}

		if clockMismatch {
			results = append(results, JobAttributionResult{
				Assignment: EntityAssignment{Process: baseProc, State: "unknown"},
				Reason:     "clock_mismatch",
				Cgroup:     cg,
			})
			continue
		}

		if unsupportedSharing {
			results = append(results, JobAttributionResult{
				Assignment: EntityAssignment{Process: baseProc, State: "unknown"},
				Reason:     "unsupported_sharing",
				Cgroup:     cg,
			})
			continue
		}

		if unidentifiableMIG {
			results = append(results, JobAttributionResult{
				Assignment: EntityAssignment{Process: baseProc, State: "unknown"},
				Reason:     "unidentifiable_mig_sharing",
				Cgroup:     cg,
			})
			continue
		}

		if len(matchingMappings) == 0 {
			results = append(results, JobAttributionResult{
				Assignment: EntityAssignment{Process: baseProc, State: "unknown"},
				Reason:     "missing_device_mapping",
				Cgroup:     cg,
			})
			continue
		}

		// Reconcile duplicate mappings per device UUID.
		// If multiple mappings for the same owner/device exist and one indicates sharing,
		// do not emit exclusive assignment; conservatively emit shared.
		mappingConflict := false
		mappingsByDev := make(map[string][]SlurmGPUMapping)
		for _, m := range matchingMappings {
			mappingsByDev[m.Device.UUID] = append(mappingsByDev[m.Device.UUID], m)
		}
		var reconciledMappings []SlurmGPUMapping
		for _, devMappings := range mappingsByDev {
			base := devMappings[0]
			for _, other := range devMappings[1:] {
				if other.AllocationID != "" && base.AllocationID != "" && other.AllocationID != base.AllocationID {
					mappingConflict = true
					break
				}
				if other.Attempt != "" && base.Attempt != "" && other.Attempt != base.Attempt {
					mappingConflict = true
					break
				}
				if other.ClockDomain != "" && base.ClockDomain != "" && other.ClockDomain != base.ClockDomain {
					mappingConflict = true
					break
				}
				if other.ClockEpoch != "" && base.ClockEpoch != "" && other.ClockEpoch != base.ClockEpoch {
					mappingConflict = true
					break
				}
				if other.Device.MIGScope != "" && base.Device.MIGScope != "" && other.Device.MIGScope != base.Device.MIGScope {
					mappingConflict = true
					break
				}
				if other.Device.MIGParentUUID != "" && base.Device.MIGParentUUID != "" && other.Device.MIGParentUUID != base.Device.MIGParentUUID {
					mappingConflict = true
					break
				}
				if other.Device.HasMIGInstanceID && base.Device.HasMIGInstanceID && other.Device.MIGInstanceID != base.Device.MIGInstanceID {
					mappingConflict = true
					break
				}
				if other.Device.HasMIGComputeInstanceID && base.Device.HasMIGComputeInstanceID && other.Device.MIGComputeInstanceID != base.Device.MIGComputeInstanceID {
					mappingConflict = true
					break
				}
				if other.MIGConfigurationEpoch != "" && base.MIGConfigurationEpoch != "" && other.MIGConfigurationEpoch != base.MIGConfigurationEpoch {
					mappingConflict = true
					break
				}
				if other.MPSRole != "" && base.MPSRole != "" && other.MPSRole != base.MPSRole {
					mappingConflict = true
					break
				}
				if other.MPSServer != nil && base.MPSServer != nil && !sameEntityProcess(*other.MPSServer, *base.MPSServer) {
					mappingConflict = true
					break
				}

				if base.ClockDomain == "" && other.ClockDomain != "" {
					base.ClockDomain = other.ClockDomain
				}
				if base.ClockEpoch == "" && other.ClockEpoch != "" {
					base.ClockEpoch = other.ClockEpoch
				}
				if base.Device.MIGScope == "" && other.Device.MIGScope != "" {
					base.Device.MIGScope = other.Device.MIGScope
				}
				if base.Device.MIGParentUUID == "" && other.Device.MIGParentUUID != "" {
					base.Device.MIGParentUUID = other.Device.MIGParentUUID
				}
				if !base.Device.HasMIGInstanceID && other.Device.HasMIGInstanceID {
					base.Device.MIGInstanceID = other.Device.MIGInstanceID
					base.Device.HasMIGInstanceID = true
				}
				if !base.Device.HasMIGComputeInstanceID && other.Device.HasMIGComputeInstanceID {
					base.Device.MIGComputeInstanceID = other.Device.MIGComputeInstanceID
					base.Device.HasMIGComputeInstanceID = true
				}
				if base.MIGConfigurationEpoch == "" && other.MIGConfigurationEpoch != "" {
					base.MIGConfigurationEpoch = other.MIGConfigurationEpoch
				}
				if base.MPSRole == "" && other.MPSRole != "" {
					base.MPSRole = other.MPSRole
				}
				if base.MPSServer == nil && other.MPSServer != nil {
					p := *other.MPSServer
					p.NamespacePIDs = append([]uint32(nil), p.NamespacePIDs...)
					base.MPSServer = &p
				}

				// Reconcile sharing mode conservatively.
				if other.Sharing == "time_slice" || base.Sharing == "time_slice" {
					base.Sharing = "time_slice"
				} else if other.Sharing == "mps" || base.Sharing == "mps" {
					base.Sharing = "mps"
				} else if other.Sharing == "mig" || base.Sharing == "mig" {
					base.Sharing = "mig"
				}
			}
			if mappingConflict {
				break
			}
			reconciledMappings = append(reconciledMappings, base)
		}

		if mappingConflict {
			results = append(results, JobAttributionResult{
				Assignment: EntityAssignment{Process: baseProc, State: "contradictory"},
				Reason:     "ambiguous_mapping",
				Cgroup:     cg,
			})
			continue
		}

		// Check TRES consistency against matched mappings.
		distinctDevices := make(map[string]bool)
		for _, m := range reconciledMappings {
			distinctDevices[m.Device.UUID] = true
		}
		if af.hasGRES {
			if len(distinctDevices) > af.gres.Count {
				results = append(results, JobAttributionResult{
					Assignment: EntityAssignment{Process: baseProc, State: "contradictory"},
					Reason:     "conflicting_tres",
					Cgroup:     cg,
				})
				continue
			}
			if len(af.gres.DeviceIDs) > 0 {
				mismatch := false
				for _, m := range reconciledMappings {
					matchedID := false
					for _, d := range af.gres.DeviceIDs {
						if d == m.Device.UUID || d == strconv.Itoa(m.Device.Index) {
							matchedID = true
							break
						}
					}
					if !matchedID {
						mismatch = true
						break
					}
				}
				if mismatch {
					results = append(results, JobAttributionResult{
						Assignment: EntityAssignment{Process: baseProc, State: "contradictory"},
						Reason:     "conflicting_tres",
						Cgroup:     cg,
					})
					continue
				}
			}
		}

		// Attempt join with execution proofs for each matched mapping.
		for _, m := range reconciledMappings {
			var matchingExecutions []EntityAssignment
			var pidReuse bool
			var mpsContradiction bool

			for _, e := range executions {
				if e.State != "assigned" {
					continue
				}
				if e.Device.UUID != m.Device.UUID {
					continue
				}
				if e.Process.Process.PID != baseProc.Process.PID {
					continue
				}
				// Check for PID reuse across generations.
				if e.Process.Process.Generation != baseProc.Process.Generation {
					pidReuse = true
					continue
				}
				// Check for ExecGeneration conflict.
				if baseProc.HasExecGeneration && baseProc.ExecGeneration != e.Process.ExecGeneration {
					continue
				}
				// Check PIDNamespace consistency.
				if e.Process.PIDNamespace != "" && baseProc.PIDNamespace != "" && e.Process.PIDNamespace != baseProc.PIDNamespace {
					continue
				}
				// Check known NamespacePIDs coordinates: if both are non-empty and differ, withhold.
				if len(baseProc.NamespacePIDs) > 0 && len(e.Process.NamespacePIDs) > 0 && !slices.Equal(baseProc.NamespacePIDs, e.Process.NamespacePIDs) {
					continue
				}
				if e.Process.NodeID != "" && baseProc.NodeID != "" && e.Process.NodeID != baseProc.NodeID {
					continue
				}
				if e.Process.BootID != "" && baseProc.BootID != "" && e.Process.BootID != baseProc.BootID {
					continue
				}
				// Check ClockDomain and ClockEpoch.
				if e.ClockDomain != "" && af.alloc.Snapshot.ClockDomain != "" && e.ClockDomain != af.alloc.Snapshot.ClockDomain {
					continue
				}
				if e.ClockEpoch != "" && af.alloc.Snapshot.ClockEpoch != "" && e.ClockEpoch != af.alloc.Snapshot.ClockEpoch {
					continue
				}
				if m.ClockDomain != "" && e.ClockDomain != "" && m.ClockDomain != e.ClockDomain {
					continue
				}
				if m.ClockEpoch != "" && e.ClockEpoch != "" && m.ClockEpoch != e.ClockEpoch {
					continue
				}
				// Check owner bridge.
				if e.Owner.JobID != "" && e.Owner.JobID != cg.JobID {
					continue
				}
				if m.AllocationID != "" && e.Owner.AllocationID != "" && m.AllocationID != e.Owner.AllocationID {
					continue
				}
				if m.Attempt != "" && e.Owner.Attempt != "" && m.Attempt != e.Owner.Attempt {
					continue
				}
				// Check MIG parent UUID and configuration epoch consistency.
				if m.Device.MIGParentUUID != "" && e.Device.MIGParentUUID != "" && m.Device.MIGParentUUID != e.Device.MIGParentUUID {
					continue
				}
				if m.Device.MIGScope != "" && e.Device.MIGScope != "" && m.Device.MIGScope != e.Device.MIGScope {
					continue
				}
				if m.Device.HasMIGInstanceID && e.Device.HasMIGInstanceID && m.Device.MIGInstanceID != e.Device.MIGInstanceID {
					continue
				}
				if m.Device.HasMIGComputeInstanceID && e.Device.HasMIGComputeInstanceID && m.Device.MIGComputeInstanceID != e.Device.MIGComputeInstanceID {
					continue
				}
				if m.MIGConfigurationEpoch != "" && e.MIGConfigurationEpoch != "" && m.MIGConfigurationEpoch != e.MIGConfigurationEpoch {
					continue
				}
				// Check MPS role consistency (e.g. server vs client).
				if m.MPSRole != "" && e.MPSRole != "" && m.MPSRole != e.MPSRole {
					mpsContradiction = true
					continue
				}
				// Check MPS server consistency.
				if m.MPSServer != nil && e.MPSServer != nil && !sameEntityProcess(*m.MPSServer, *e.MPSServer) {
					mpsContradiction = true
					continue
				}
				if !completeEntityProcess(e.Process) {
					continue
				}
				matchingExecutions = append(matchingExecutions, e)
			}

			if mpsContradiction && len(matchingExecutions) == 0 {
				results = append(results, JobAttributionResult{
					Assignment: EntityAssignment{Process: baseProc, Device: m.Device, State: "contradictory"},
					Reason:     "conflicting_mps_server",
					Cgroup:     cg,
				})
				continue
			}

			if pidReuse && len(matchingExecutions) == 0 {
				results = append(results, JobAttributionResult{
					Assignment: EntityAssignment{Process: baseProc, Device: m.Device, State: "unknown"},
					Reason:     "reused_pid",
					Cgroup:     cg,
				})
				continue
			}

			switch len(matchingExecutions) {
			case 0:
				results = append(results, JobAttributionResult{
					Assignment: EntityAssignment{Process: baseProc, Device: m.Device, State: "unknown"},
					Reason:     "missing_execution_proof",
					Cgroup:     cg,
				})
			case 1:
				matchedExec := matchingExecutions[0]

				// Reconcile sharing mode without upgrading trust.
				sharing := m.Sharing
				if sharing == "" {
					sharing = "unknown"
				}
				if matchedExec.Sharing == "mps" || m.Sharing == "mps" {
					sharing = "mps"
				} else if matchedExec.Sharing == "time_slice" || m.Sharing == "time_slice" {
					sharing = "time_slice"
				} else if matchedExec.Sharing == "mig" || m.Sharing == "mig" {
					sharing = "mig"
				} else if matchedExec.Sharing == "exclusive" && m.Sharing == "exclusive" {
					sharing = "exclusive"
				} else if matchedExec.Sharing != "" && matchedExec.Sharing != "exclusive" {
					sharing = matchedExec.Sharing
				}

				clockDomain := af.alloc.Snapshot.ClockDomain
				if clockDomain == "" {
					clockDomain = matchedExec.ClockDomain
				}
				clockEpoch := af.alloc.Snapshot.ClockEpoch
				if clockEpoch == "" {
					clockEpoch = matchedExec.ClockEpoch
				}

				startNanos := max(af.alloc.Snapshot.StartNanos, m.StartNanos, matchedExec.StartNanos)
				endNanos := min(af.alloc.Snapshot.EndNanos, m.EndNanos, matchedExec.EndNanos)
				if endNanos <= startNanos {
					results = append(results, JobAttributionResult{
						Assignment: EntityAssignment{Process: baseProc, Device: m.Device, State: "unknown"},
						Reason:     "stale_allocation",
						Cgroup:     cg,
					})
					continue
				}

				state := "assigned"
				reason := ""
				if at < startNanos || at >= endNanos {
					state = "unknown"
					reason = "stale_allocation"
				}

				procWithExec := baseProc
				procWithExec.ExecGeneration = matchedExec.Process.ExecGeneration
				procWithExec.HasExecGeneration = true
				if len(baseProc.NamespacePIDs) > 0 {
					procWithExec.NamespacePIDs = append([]uint32(nil), baseProc.NamespacePIDs...)
				} else if len(matchedExec.Process.NamespacePIDs) > 0 {
					procWithExec.NamespacePIDs = append([]uint32(nil), matchedExec.Process.NamespacePIDs...)
				}

				authoritative := af.alloc.Snapshot.Authoritative && m.Authoritative && matchedExec.Authoritative

				// Preserve validated observed MPS execution metadata if mapping left it absent.
				mpsRole := m.MPSRole
				if mpsRole == "" {
					mpsRole = matchedExec.MPSRole
				}
				mpsServer := m.MPSServer
				if mpsServer == nil && matchedExec.MPSServer != nil {
					p := *matchedExec.MPSServer
					p.NamespacePIDs = append([]uint32(nil), p.NamespacePIDs...)
					mpsServer = &p
				}

				assigned := EntityAssignment{
					Process:               procWithExec,
					Device:                cloneAssignment(EntityAssignment{Device: m.Device}).Device,
					Owner:                 EntityOwner{JobID: cg.JobID, AllocationID: m.AllocationID, Attempt: m.Attempt},
					Source:                "slurm_cgroup_v2_attribution",
					SourceRevision:        "1",
					Authoritative:         authoritative,
					State:                 state,
					ClockDomain:           clockDomain,
					ClockEpoch:            clockEpoch,
					StartNanos:            startNanos,
					EndNanos:              endNanos,
					Sharing:               sharing,
					MIGConfigurationEpoch: m.MIGConfigurationEpoch,
					MPSRole:               mpsRole,
					MPSServer:             mpsServer,
					Provenance: []EntityProvenance{
						{Source: "slurm_cgroup_v2", Revision: cg.CgroupPath, StartNanos: startNanos, EndNanos: endNanos},
						{Source: "slurm_squeue_v0.0.43", Revision: af.alloc.Snapshot.SourceRevision, StartNanos: af.alloc.Snapshot.StartNanos, EndNanos: af.alloc.Snapshot.EndNanos},
						{Source: matchedExec.Source, Revision: matchedExec.SourceRevision, StartNanos: matchedExec.StartNanos, EndNanos: matchedExec.EndNanos},
					},
				}
				results = append(results, JobAttributionResult{
					Assignment: cloneAssignment(assigned),
					Reason:     reason,
					Cgroup:     cg,
				})
			default:
				results = append(results, JobAttributionResult{
					Assignment: EntityAssignment{Process: baseProc, Device: m.Device, State: "contradictory"},
					Reason:     "ambiguous_execution_proof",
					Cgroup:     cg,
				})
			}
		}
	}

	return results, nil
}

type jobAttributionKey struct {
	bootID         string
	nodeID         string
	pidNamespace   string
	deviceUUID     string
	pid            uint32
	generation     uint64
	execGeneration uint64
}

type jobAttributionEntry struct {
	result   JobAttributionResult
	expires  int64
	cachedAt int64
}

// JobAttributionTable is a bounded relationship table and cache providing
// replayable process/cgroup-to-allocation joins with TTL expiration. The
// zero value is not usable; use NewJobAttributionTable.
type JobAttributionTable struct {
	mu           sync.Mutex
	limit        int
	ttl          int64
	entries      map[jobAttributionKey]jobAttributionEntry
	lastRefresh  int64
	refreshFails int
}

// NewJobAttributionTable creates a table with positive capacity and TTL in nanoseconds.
func NewJobAttributionTable(capacity int, ttlNanos int64) (*JobAttributionTable, error) {
	if capacity <= 0 || capacity > entityMaxRecords || ttlNanos <= 0 {
		return nil, fmt.Errorf("job attribution table: invalid capacity or ttl")
	}
	return &JobAttributionTable{
		limit:   capacity,
		ttl:     ttlNanos,
		entries: make(map[jobAttributionKey]jobAttributionEntry),
	}, nil
}

// Refresh replaces all cached relationships with the newly evaluated snapshot
// at timestamp now. It enforces atomic replacement semantics: if evaluation or
// capacity validation fails, the existing cache is left completely unmutated.
// Refreshes with an empty snapshot clear previous assignments.
func (t *JobAttributionTable) Refresh(snap JobAttributionSnapshot, now int64) ([]JobAttributionResult, error) {
	if t == nil {
		return nil, fmt.Errorf("job attribution table: nil table")
	}
	t.mu.Lock()
	defer t.mu.Unlock()

	results, err := JoinSlurmAttribution(snap.Allocations, snap.Mappings, snap.Processes, snap.Executions, now)
	if err != nil {
		t.refreshFails++
		return nil, err
	}

	// Build new entries map atomically without mutating existing cache on failure.
	newEntries := make(map[jobAttributionKey]jobAttributionEntry)

	for _, r := range results {
		if r.Assignment.State != "assigned" {
			continue
		}
		p := r.Assignment.Process
		key := jobAttributionKey{
			bootID:         p.BootID,
			nodeID:         p.NodeID,
			pidNamespace:   p.PIDNamespace,
			deviceUUID:     r.Assignment.Device.UUID,
			pid:            p.Process.PID,
			generation:     p.Process.Generation,
			execGeneration: p.ExecGeneration,
		}

		expires := int64(math.MaxInt64)
		if t.ttl > 0 && now <= math.MaxInt64-t.ttl {
			expires = now + t.ttl
		}
		if r.Assignment.EndNanos > 0 && r.Assignment.EndNanos < expires {
			expires = r.Assignment.EndNanos
		}

		if _, ok := newEntries[key]; !ok && len(newEntries) >= t.limit {
			t.refreshFails++
			return nil, fmt.Errorf("job attribution table: capacity exceeded")
		}

		// Store deeply-cloned result so cache is isolated.
		clonedRes := r
		clonedRes.Assignment = cloneAssignment(r.Assignment)
		newEntries[key] = jobAttributionEntry{
			result:   clonedRes,
			expires:  expires,
			cachedAt: now,
		}
	}

	t.entries = newEntries
	t.lastRefresh = now

	// Return deeply cloned results to caller.
	out := make([]JobAttributionResult, len(results))
	for i, res := range results {
		out[i] = res
		out[i].Assignment = cloneAssignment(res.Assignment)
	}
	return out, nil
}

// Resolve queries attribution for a specific process on a device at timestamp at.
// Misses, expired entries, and generation mismatches return an unresolved result.
func (t *JobAttributionTable) Resolve(proc EntityProcess, deviceUUID string, at int64) JobAttributionResult {
	if t == nil {
		return JobAttributionResult{
			Assignment: EntityAssignment{State: "unknown"},
			Reason:     "nil_table",
		}
	}
	t.mu.Lock()
	defer t.mu.Unlock()

	key := jobAttributionKey{
		bootID:         proc.BootID,
		nodeID:         proc.NodeID,
		pidNamespace:   proc.PIDNamespace,
		deviceUUID:     deviceUUID,
		pid:            proc.Process.PID,
		generation:     proc.Process.Generation,
		execGeneration: proc.ExecGeneration,
	}

	entry, ok := t.entries[key]
	if !ok {
		return JobAttributionResult{
			Assignment: EntityAssignment{Process: proc, State: "unknown"},
			Reason:     "missing_attribution",
		}
	}

	if at >= entry.expires || (entry.result.Assignment.EndNanos > 0 && at >= entry.result.Assignment.EndNanos) {
		delete(t.entries, key)
		return JobAttributionResult{
			Assignment: EntityAssignment{Process: proc, State: "unknown"},
			Reason:     "expired_entry",
			Cgroup:     entry.result.Cgroup,
		}
	}

	if at < entry.result.Assignment.StartNanos {
		return JobAttributionResult{
			Assignment: EntityAssignment{Process: proc, State: "unknown"},
			Reason:     "stale_allocation",
			Cgroup:     entry.result.Cgroup,
		}
	}

	res := entry.result
	res.Assignment = cloneAssignment(res.Assignment)
	return res
}

// Stats returns table operational statistics.
func (t *JobAttributionTable) Stats() (entries int, lastRefresh int64, refreshFailures int) {
	if t == nil {
		return 0, 0, 0
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	return len(t.entries), t.lastRefresh, t.refreshFails
}

// Clear removes all cached relationships.
func (t *JobAttributionTable) Clear() {
	if t == nil {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	clear(t.entries)
}
