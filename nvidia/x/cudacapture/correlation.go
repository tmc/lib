package cudacapture

import (
	"errors"
	"fmt"
	"io"
	"sort"
	"sync"

	"github.com/tmc/lib/nvidia/x/cudaprof"
)

// ErrAmbiguousCorrelation indicates that a correlation ID matched multiple runtime or activity records.
var ErrAmbiguousCorrelation = errors.New("cudacapture: ambiguous correlation join")

// RuntimeLaunch records host-side kernel launch invocation context.
// Callstack holds the host Go call stack at the Do call site granularity (not per-launch).
// If the launch has no external correlation ID, Callstack is nil (no stack).
type RuntimeLaunch struct {
	CorrelationID uint64
	KernelName    string
	HostThreadID  uint64
	Callstack     []string
}

// CorrelationJoin represents an unambiguous native correlation between a host runtime launch
// and device kernel execution activity.
// Callstack holds the host Go call stack of the Do call site (not per-launch).
// If the kernel has no external correlation ID, Callstack is nil (no stack).
type CorrelationJoin struct {
	CorrelationID    uint64
	KernelName       string
	HostThreadID     uint64
	Callstack        []string
	DeviceStart      uint64
	DeviceEnd        uint64
	DeviceDurationNs uint64
	Stream           uint32
	Device           uint32
}

// AmbiguousCorrelation records an ambiguous join where a correlation ID maps to multiple
// runtime launches, multiple device activity records, or multiple conflicting external
// correlation IDs / call stacks.
type AmbiguousCorrelation struct {
	CorrelationID uint64
	RuntimeCount  int
	ActivityCount int
	KernelNames   []string
	Reason        string
}

// CorrelationTracker coordinates native CUPTI runtime-to-activity correlation.
// It correlates external correlation IDs from Do/Correlate with CUDA runtime/driver
// correlation IDs and GPU device activities by exact ID match, never by timestamp.
// Host call stacks are attached at the Do call site granularity (not per-launch).
type CorrelationTracker struct {
	mu             sync.Mutex
	runtimes       map[uint64][]RuntimeLaunch
	activities     map[uint64][]ActivityRecord
	externalStacks map[uint64][][]string // externalID -> distinct stacks recorded
	cudaToExternal map[uint64][]uint64   // cudaCorrID -> list of externalIDs
	externalToCUDA map[uint64][]uint64   // externalID -> list of cudaCorrIDs
}

// NewCorrelationTracker creates a new correlation tracker.
func NewCorrelationTracker() *CorrelationTracker {
	return &CorrelationTracker{
		runtimes:       make(map[uint64][]RuntimeLaunch),
		activities:     make(map[uint64][]ActivityRecord),
		externalStacks: make(map[uint64][][]string),
		cudaToExternal: make(map[uint64][]uint64),
		externalToCUDA: make(map[uint64][]uint64),
	}
}

// RecordRuntime records a host-side runtime launch event.
func (ct *CorrelationTracker) RecordRuntime(launch RuntimeLaunch) error {
	if launch.CorrelationID == 0 {
		return errors.New("cudacapture: invalid zero correlation ID")
	}
	ct.mu.Lock()
	defer ct.mu.Unlock()
	ct.runtimes[launch.CorrelationID] = append(ct.runtimes[launch.CorrelationID], launch)
	return nil
}

// RecordExternalCorrelation records an association between an external correlation ID
// (pushed by Do or Correlate) and a CUDA runtime/driver correlation ID.
func (ct *CorrelationTracker) RecordExternalCorrelation(externalID uint64, correlationID uint64) error {
	if externalID == 0 {
		return errors.New("cudacapture: invalid zero external correlation ID")
	}
	if correlationID == 0 {
		return errors.New("cudacapture: invalid zero CUDA correlation ID")
	}
	ct.mu.Lock()
	defer ct.mu.Unlock()
	if ct.cudaToExternal == nil {
		ct.cudaToExternal = make(map[uint64][]uint64)
	}
	if ct.externalToCUDA == nil {
		ct.externalToCUDA = make(map[uint64][]uint64)
	}
	ct.cudaToExternal[correlationID] = appendUniqueUint64(ct.cudaToExternal[correlationID], externalID)
	ct.externalToCUDA[externalID] = appendUniqueUint64(ct.externalToCUDA[externalID], correlationID)
	return nil
}

// RecordExternalStack records a host call stack associated with an external correlation ID.
// Granularity: the stack is the stack of the Do call site (not per-launch).
func (ct *CorrelationTracker) RecordExternalStack(externalID uint64, stack []string) error {
	if externalID == 0 {
		return errors.New("cudacapture: invalid zero external correlation ID")
	}
	ct.mu.Lock()
	defer ct.mu.Unlock()
	if ct.externalStacks == nil {
		ct.externalStacks = make(map[uint64][][]string)
	}
	existing := ct.externalStacks[externalID]
	for _, s := range existing {
		if stringSlicesEqual(s, stack) {
			return nil
		}
	}
	st := make([]string, len(stack))
	copy(st, stack)
	ct.externalStacks[externalID] = append(ct.externalStacks[externalID], st)
	return nil
}

// RecordActivity records a GPU device activity event. If rec is an external_correlation record
// (Subtype == "external_correlation"), it is routed to external correlation tracking instead
// of being treated as device work.
func (ct *CorrelationTracker) RecordActivity(rec ActivityRecord) error {
	if rec.Subtype == "external_correlation" {
		corrID := uint64(rec.Correlation)
		if corrID == 0 {
			corrID = uint64(rec.RuntimeCorrelation)
		}
		if corrID == 0 {
			return errors.New("cudacapture: external correlation has zero correlation ID")
		}
		if rec.ExternalID == 0 {
			return errors.New("cudacapture: external correlation has zero external ID")
		}
		return ct.RecordExternalCorrelation(rec.ExternalID, corrID)
	}

	corrID := uint64(rec.Correlation)
	if corrID == 0 {
		corrID = uint64(rec.RuntimeCorrelation)
	}
	if corrID == 0 {
		return errors.New("cudacapture: activity has zero correlation ID")
	}
	ct.mu.Lock()
	defer ct.mu.Unlock()
	ct.activities[corrID] = append(ct.activities[corrID], rec)
	return nil
}

// Join correlates runtime launches with device activities by exact ID matching.
// It partitions results into unambiguous matches, ambiguous joins, unmapped runtime launches,
// and unmapped device activities.
//
// Host call stacks are attached by exact external correlation ID match:
// agent external_correlation record -> CUDA correlation ID -> kernel activity.
// Timestamps are never used to infer correlation. A kernel with no external correlation
// ID has no stack (reported as nil, not an empty slice).
func (ct *CorrelationTracker) Join() (matched []CorrelationJoin, ambiguous []AmbiguousCorrelation, unmappedRuntime []uint64, unmappedActivity []uint64) {
	ct.mu.Lock()
	defer ct.mu.Unlock()

	allIDs := make(map[uint64]bool)
	for id := range ct.runtimes {
		allIDs[id] = true
	}
	for id := range ct.activities {
		allIDs[id] = true
	}

	sortedIDs := make([]uint64, 0, len(allIDs))
	for id := range allIDs {
		sortedIDs = append(sortedIDs, id)
	}
	sort.Slice(sortedIDs, func(i, j int) bool { return sortedIDs[i] < sortedIDs[j] })

	for _, id := range sortedIDs {
		rList := ct.runtimes[id]
		aList := ct.activities[id]

		switch {
		case len(rList) == 1 && len(aList) == 1:
			r := rList[0]
			a := aList[0]

			// Check external correlation ambiguity
			extIDs := ct.cudaToExternal[id]
			if len(extIDs) > 1 {
				name := r.KernelName
				if name == "" {
					name = a.Name
				}
				var names []string
				if name != "" {
					names = []string{name}
				}
				ambiguous = append(ambiguous, AmbiguousCorrelation{
					CorrelationID: id,
					RuntimeCount:  len(rList),
					ActivityCount: len(aList),
					KernelNames:   names,
					Reason:        fmt.Sprintf("correlation ID %d maps to multiple external correlation IDs: %v", id, extIDs),
				})
				continue
			}

			if len(extIDs) == 1 {
				extID := extIDs[0]
				stacks := ct.externalStacks[extID]
				if len(stacks) > 1 {
					name := r.KernelName
					if name == "" {
						name = a.Name
					}
					var names []string
					if name != "" {
						names = []string{name}
					}
					ambiguous = append(ambiguous, AmbiguousCorrelation{
						CorrelationID: id,
						RuntimeCount:  len(rList),
						ActivityCount: len(aList),
						KernelNames:   names,
						Reason:        fmt.Sprintf("external correlation ID %d maps to multiple conflicting callstacks", extID),
					})
					continue
				}
				if len(stacks) == 1 {
					// Unambiguous exact ID match: put the stack in RuntimeLaunch.Callstack
					// through the existing CorrelationTracker.
					r.Callstack = stacks[0]
					ct.runtimes[id][0].Callstack = stacks[0]
				} else {
					// External ID exists but no stack was recorded.
					r.Callstack = nil
					ct.runtimes[id][0].Callstack = nil
				}
			} else {
				// No external ID: kernel with no external ID has no stack (reported as nil, not empty stack).
				if len(ct.cudaToExternal) > 0 || len(ct.externalStacks) > 0 {
					r.Callstack = nil
					ct.runtimes[id][0].Callstack = nil
				}
			}

			var dur uint64
			if a.End >= a.Start {
				dur = a.End - a.Start
			}
			matched = append(matched, CorrelationJoin{
				CorrelationID:    id,
				KernelName:       r.KernelName,
				HostThreadID:     r.HostThreadID,
				Callstack:        r.Callstack,
				DeviceStart:      a.Start,
				DeviceEnd:        a.End,
				DeviceDurationNs: dur,
				Stream:           a.Stream,
				Device:           a.Device,
			})

		case len(rList) > 1 || len(aList) > 1:
			nameSet := make(map[string]bool)
			for _, r := range rList {
				nameSet[r.KernelName] = true
			}
			for _, a := range aList {
				nameSet[a.Name] = true
			}
			names := make([]string, 0, len(nameSet))
			for name := range nameSet {
				if name != "" {
					names = append(names, name)
				}
			}
			sort.Strings(names)

			var reason string
			if len(rList) > 1 && len(aList) > 1 {
				reason = fmt.Sprintf("multiple runtime launches (%d) and multiple activities (%d) for correlation ID %d", len(rList), len(aList), id)
			} else if len(rList) > 1 {
				reason = fmt.Sprintf("multiple runtime launches (%d) for correlation ID %d", len(rList), id)
			} else {
				reason = fmt.Sprintf("multiple activities (%d) for correlation ID %d", len(aList), id)
			}

			ambiguous = append(ambiguous, AmbiguousCorrelation{
				CorrelationID: id,
				RuntimeCount:  len(rList),
				ActivityCount: len(aList),
				KernelNames:   names,
				Reason:        reason,
			})

		case len(rList) == 1 && len(aList) == 0:
			unmappedRuntime = append(unmappedRuntime, id)

		case len(rList) == 0 && len(aList) == 1:
			unmappedActivity = append(unmappedActivity, id)
		}
	}

	return matched, ambiguous, unmappedRuntime, unmappedActivity
}

// ExportFoldedStacks exports joined correlation records as folded stacks in flamegraph.pl format,
// weighted by device busy time from interval unions.
func ExportFoldedStacks(w io.Writer, joins []CorrelationJoin) error {
	executions := make([]cudaprof.KernelExecution, len(joins))
	for i, j := range joins {
		executions[i] = cudaprof.KernelExecution{
			KernelName: j.KernelName,
			Callstack:  j.Callstack,
			StartNs:    j.DeviceStart,
			EndNs:      j.DeviceEnd,
		}
	}
	return cudaprof.ExportFoldedStacks(w, executions)
}

func appendUniqueUint64(slice []uint64, val uint64) []uint64 {
	for _, v := range slice {
		if v == val {
			return slice
		}
	}
	return append(slice, val)
}

func stringSlicesEqual(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
