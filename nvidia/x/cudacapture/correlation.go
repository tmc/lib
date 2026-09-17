package cudacapture

import (
	"errors"
	"fmt"
	"sort"
	"sync"
)

// ErrAmbiguousCorrelation indicates that a correlation ID matched multiple runtime or activity records.
var ErrAmbiguousCorrelation = errors.New("cudacapture: ambiguous correlation join")

// RuntimeLaunch records host-side kernel launch invocation context.
type RuntimeLaunch struct {
	CorrelationID uint64
	KernelName    string
	HostThreadID  uint64
	Callstack     []string
}

// CorrelationJoin represents an unambiguous native correlation between a host runtime launch
// and device kernel execution activity.
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
// runtime launches or multiple device activity records (e.g. from scrambled correlation IDs
// or stream collisions).
type AmbiguousCorrelation struct {
	CorrelationID uint64
	RuntimeCount  int
	ActivityCount int
	KernelNames   []string
	Reason        string
}

// CorrelationTracker coordinates native CUPTI runtime-to-activity correlation.
// It detects ambiguous correlations, dropped records, and missing mappings.
type CorrelationTracker struct {
	mu         sync.Mutex
	runtimes   map[uint64][]RuntimeLaunch
	activities map[uint64][]ActivityRecord
}

// NewCorrelationTracker creates a new correlation tracker.
func NewCorrelationTracker() *CorrelationTracker {
	return &CorrelationTracker{
		runtimes:   make(map[uint64][]RuntimeLaunch),
		activities: make(map[uint64][]ActivityRecord),
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

// RecordActivity records a GPU device activity event.
func (ct *CorrelationTracker) RecordActivity(rec ActivityRecord) error {
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

// Join correlates runtime launches with device activities.
// It partitions results into unambiguous matches, ambiguous joins, unmapped runtime launches,
// and unmapped device activities.
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
