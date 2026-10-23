package nvtelemetry

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"slices"

	"github.com/tmc/lib/nvidia/nvml"
)

// nvmlProcessMaxRecords bounds the per-call and aggregate record caps of an
// NVMLProcessSelection.
const nvmlProcessMaxRecords = 65536

// NVMLProcessQuery is the bounded running-process query of one device.
// Each method makes one query with capacity len(buf), stores the entries in
// buf and returns their count, or refuses a list that does not fit.
// [nvml.Device] implements it.
type NVMLProcessQuery interface {
	ComputeRunningProcessesInto(buf []nvml.ProcessInfo) (int, error)
	GraphicsRunningProcessesInto(buf []nvml.ProcessInfo) (int, error)
	MPSComputeRunningProcessesInto(buf []nvml.ProcessInfo) (int, error)
}

var _ NVMLProcessQuery = nvml.Device(0)

// NVMLProcessDevice is an opened device: its metadata and its query.
type NVMLProcessDevice struct {
	Device DeviceMetadata
	Query  NVMLProcessQuery
}

// NVMLProcessSelection selects live NVML process observations.
//
// UUIDs lists the devices to query, in order; the caller opens each one by
// UUID and verifies the UUID it reports. Sources lists the queried APIs:
// nvml_compute_v3, nvml_graphics_v3 or nvml_mps_compute_v3. Neither list may
// be empty or repeat an entry. MaxPerCall is the capacity of each query and
// MaxRecords bounds the records of all queries; both must be in [1, 65536].
// Snapshot supplies the observation provenance, and BootID, Snapshot.NodeID
// and PIDNamespace identify the host the queries observe, which must be the
// host whose proc files are read.
type NVMLProcessSelection struct {
	UUIDs        []string
	Sources      []string
	MaxPerCall   int
	MaxRecords   int
	BootID       string
	PIDNamespace string
	Snapshot     EntitySnapshot
}

func (s NVMLProcessSelection) validate() error {
	if len(s.UUIDs) == 0 || len(s.Sources) == 0 {
		return errors.New("nvml processes: no devices or sources selected")
	}
	seen := make(map[string]bool)
	for _, uuid := range s.UUIDs {
		if uuid == "" || seen[uuid] {
			return errors.New("nvml processes: empty or duplicate device uuid")
		}
		seen[uuid] = true
	}
	clear(seen)
	for _, source := range s.Sources {
		switch source {
		case "nvml_compute_v3", "nvml_graphics_v3", "nvml_mps_compute_v3":
		default:
			return fmt.Errorf("nvml processes: unsupported source %q", source)
		}
		if seen[source] {
			return errors.New("nvml processes: duplicate source")
		}
		seen[source] = true
		if err := s.Snapshot.assignment(source).Validate(); err != nil {
			return fmt.Errorf("nvml processes: unpinned snapshot: %w", err)
		}
	}
	if s.MaxPerCall < 1 || s.MaxPerCall > nvmlProcessMaxRecords || s.MaxRecords < 1 || s.MaxRecords > nvmlProcessMaxRecords {
		return fmt.Errorf("nvml processes: record caps must be in [1, %d]", nvmlProcessMaxRecords)
	}
	if s.BootID == "" || s.Snapshot.NodeID == "" || s.PIDNamespace == "" {
		return errors.New("nvml processes: missing boot, node or pid namespace identity")
	}
	return nil
}

// Validate reports whether s can select the processes read by src.
// src must be valid and configure no PIDs, and the boot, node and PID
// namespace identities of s and src must agree. Validate makes no native
// call, so a caller can validate before initializing NVML.
func (s NVMLProcessSelection) Validate(src SlurmAttributionSource) error {
	if err := s.validate(); err != nil {
		return err
	}
	if len(src.PIDs) != 0 {
		return errors.New("nvml processes: live selection excludes configured pids")
	}
	if err := src.validate(); err != nil {
		return err
	}
	if s.BootID != src.BootID || s.Snapshot.NodeID != src.NodeID || s.PIDNamespace != src.PIDNamespace {
		return errors.New("nvml processes: boot, node or pid namespace differs from proc source")
	}
	return nil
}

// nvmlProcessCall returns q's query method for source.
func nvmlProcessCall(q NVMLProcessQuery, source string) (func([]nvml.ProcessInfo) (int, error), error) {
	switch source {
	case "nvml_compute_v3":
		return q.ComputeRunningProcessesInto, nil
	case "nvml_graphics_v3":
		return q.GraphicsRunningProcessesInto, nil
	case "nvml_mps_compute_v3":
		return q.MPSComputeRunningProcessesInto, nil
	}
	return nil, fmt.Errorf("nvml processes: unsupported source %q", source)
}

// QueryNVMLProcesses makes one bounded query per selected source on each
// device and returns the observations, in device and then source order.
// devices must be the selected devices in selection order.
//
// The selection and devices are validated before any query. Every query uses
// one buffer of MaxPerCall entries, allocated once. A failed, unsupported or
// permission-denied query, a list that does not fit, or more than MaxRecords
// records in all fails the whole call; there is no partial or truncated result,
// and a failure is never an empty success. Facts keep their device, API and
// GPU and compute instance; the same PID may appear in several facts. A fact
// observes that a context existed during the query. It does not establish
// process identity, exec, allocation or MPS client ownership.
//
// A canceled ctx stops further queries but cannot interrupt one in progress.
func QueryNVMLProcesses(ctx context.Context, sel NVMLProcessSelection, devices []NVMLProcessDevice) ([]NVMLProcessFact, error) {
	if err := sel.validate(); err != nil {
		return nil, err
	}
	if len(devices) != len(sel.UUIDs) {
		return nil, errors.New("nvml processes: devices differ from selection")
	}
	for i, d := range devices {
		if d.Query == nil || d.Device.UUID != sel.UUIDs[i] {
			return nil, fmt.Errorf("nvml processes: device %d differs from selection", i)
		}
	}
	buf := make([]nvml.ProcessInfo, sel.MaxPerCall)
	var facts []NVMLProcessFact
	for _, d := range devices {
		for _, source := range sel.Sources {
			if err := ctx.Err(); err != nil {
				return nil, fmt.Errorf("nvml processes: %w", err)
			}
			query, err := nvmlProcessCall(d.Query, source)
			if err != nil {
				return nil, err
			}
			n, err := query(buf)
			if err != nil {
				return nil, fmt.Errorf("nvml processes: %s %s: %w", d.Device.UUID, source, err)
			}
			if n < 0 || n > len(buf) {
				return nil, fmt.Errorf("nvml processes: %s %s: count %d exceeds buffer %d", d.Device.UUID, source, n, len(buf))
			}
			if len(facts)+n > sel.MaxRecords {
				return nil, fmt.Errorf("nvml processes: more than %d records", sel.MaxRecords)
			}
			f, err := AdaptNVMLProcesses(d.Device, buf[:n], source, sel.Snapshot)
			if err != nil {
				return nil, fmt.Errorf("nvml processes: %s %s: %w", d.Device.UUID, source, err)
			}
			facts = append(facts, f...)
		}
	}
	return facts, nil
}

// ReadNVMLAttributionSnapshot queries the selected devices and then reads src
// with the observed PIDs, as ReadSlurmAttributionSnapshot does. src must
// configure no PIDs; the caller's src is not modified.
//
// Everything is validated before the first query, and a failed query fails
// the read before any file is opened. Each observed PID is read once, in
// increasing order, however many facts name it. A successful query that
// observes no process reads no proc files, so a refresh with it clears earlier
// attribution; it does not mean that a job completed. Attribution still
// requires the scheduler and execution proofs in src. The proc files are read
// after the queries, so their stat start ticks are a later observation: they
// do not show that the PID was not reused in between.
func ReadNVMLAttributionSnapshot(ctx context.Context, fsys fs.FS, src SlurmAttributionSource, sel NVMLProcessSelection, devices []NVMLProcessDevice) (AttributionRead, error) {
	if fsys == nil {
		return AttributionRead{}, errors.New("slurm attribution: nil file system")
	}
	if err := sel.Validate(src); err != nil {
		return AttributionRead{}, err
	}
	facts, err := QueryNVMLProcesses(ctx, sel, devices)
	if err != nil {
		return AttributionRead{}, err
	}
	src.PIDs = nvmlProcessPIDs(facts)
	return ReadSlurmAttributionSnapshot(fsys, src)
}

// nvmlProcessPIDs returns the distinct PIDs of facts in increasing order.
func nvmlProcessPIDs(facts []NVMLProcessFact) []uint32 {
	var pids []uint32
	for _, f := range facts {
		pids = append(pids, f.PID)
	}
	slices.Sort(pids)
	return slices.Compact(pids)
}
