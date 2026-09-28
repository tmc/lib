package cudacapture

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sort"
	"strconv"
	"sync"

	"github.com/tmc/lib/nvidia/x/cudaprof"
)

// ErrAmbiguousCorrelation indicates that a correlation ID matched multiple runtime or activity records.
var ErrAmbiguousCorrelation = errors.New("cudacapture: ambiguous correlation join")

// ErrNoExternalStacks reports that a capture has no stack sidecar member.
var ErrNoExternalStacks = errors.New("cudacapture: no external stacks")

// RuntimeLaunch records host-side kernel launch invocation context.
// Callstack is supplied host attribution. Join replaces it when an external
// correlation ID is present, and otherwise preserves it for this launch.
type RuntimeLaunch struct {
	CorrelationID uint64
	KernelName    string
	HostThreadID  uint64
	Callstack     []string
}

// CorrelationJoin represents an unambiguous native correlation between a host runtime launch
// and device kernel execution activity.
// Callstack is the launch's host attribution after Join. CallstackMissing is
// true when an external ID was joined but its stack was not retained.
type CorrelationJoin struct {
	CorrelationID uint64
	// ExternalID is set when CUPTI linked this kernel to a framework ID.
	ExternalID uint64
	// HasExternalID distinguishes an absent link from a zero-valued field.
	HasExternalID bool
	// CallstackMissing reports an external ID without a retained stack.
	CallstackMissing bool
	KernelName       string
	HostThreadID     uint64
	Callstack        []string
	DeviceStart      uint64
	DeviceEnd        uint64
	DeviceDurationNs uint64
	Stream           uint32
	Device           uint32
}

type externalStackKey struct {
	kind uint32
	id   uint64
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
	mu                  sync.Mutex
	runtimes            map[uint64][]RuntimeLaunch
	activities          map[uint64][]ActivityRecord
	externalStacks      map[externalStackKey][][]string // (kind, externalID) -> distinct stacks
	cudaToExternal      map[uint64][]externalStackKey   // cudaCorrID -> external IDs
	externalStackLosses uint64
}

// ExternalStacksName is the capture member containing the cudaprof stack sidecar.
const ExternalStacksName = "external-stacks.json"

// AttachExternalStackTable adds the bounded cudaprof stack table to the capture.
func (c *Capture) AttachExternalStackTable() error {
	if c == nil {
		return errors.New("cudacapture: nil capture")
	}
	if c.Files == nil {
		c.Files = make(map[string][]byte)
	}
	if _, exists := c.Files[ExternalStacksName]; exists {
		return fmt.Errorf("cudacapture: capture already has %q", ExternalStacksName)
	}
	var buf bytes.Buffer
	if err := cudaprof.WriteExternalStackTable(&buf); err != nil {
		return err
	}
	c.Files[ExternalStacksName] = buf.Bytes()
	return nil
}

// NewCorrelationTracker creates a new correlation tracker.
func NewCorrelationTracker() *CorrelationTracker {
	return &CorrelationTracker{
		runtimes:       make(map[uint64][]RuntimeLaunch),
		activities:     make(map[uint64][]ActivityRecord),
		externalStacks: make(map[externalStackKey][][]string),
		cudaToExternal: make(map[uint64][]externalStackKey),
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
func (ct *CorrelationTracker) RecordExternalCorrelation(kind uint32, externalID uint64, correlationID uint64) error {
	if externalID == 0 {
		return errors.New("cudacapture: invalid zero external correlation ID")
	}
	if correlationID == 0 {
		return errors.New("cudacapture: invalid zero CUDA correlation ID")
	}
	ct.mu.Lock()
	defer ct.mu.Unlock()
	if ct.cudaToExternal == nil {
		ct.cudaToExternal = make(map[uint64][]externalStackKey)
	}
	key := externalStackKey{kind: kind, id: externalID}
	ct.cudaToExternal[correlationID] = appendUniqueExternal(ct.cudaToExternal[correlationID], key)
	return nil
}

// RecordExternalStack records a host call stack associated with an external correlation ID.
// Granularity: the stack is the stack of the Do call site (not per-launch).
func (ct *CorrelationTracker) RecordExternalStack(kind uint32, externalID uint64, stack []string) error {
	if externalID == 0 {
		return errors.New("cudacapture: invalid zero external correlation ID")
	}
	ct.mu.Lock()
	defer ct.mu.Unlock()
	if ct.externalStacks == nil {
		ct.externalStacks = make(map[externalStackKey][][]string)
	}
	key := externalStackKey{kind: kind, id: externalID}
	existing := ct.externalStacks[key]
	for _, s := range existing {
		if stringSlicesEqual(s, stack) {
			return nil
		}
	}
	st := make([]string, len(stack))
	copy(st, stack)
	ct.externalStacks[key] = append(ct.externalStacks[key], st)
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
		return ct.RecordExternalCorrelation(rec.ExternalKind, rec.ExternalID, corrID)
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

// ReadExternalStackTable imports the sidecar written by cudaprof into this tracker.
func (ct *CorrelationTracker) ReadExternalStackTable(r io.Reader) error {
	if r == nil {
		return errors.New("cudacapture: nil external stack table reader")
	}
	data, err := io.ReadAll(r)
	if err != nil {
		return fmt.Errorf("cudacapture: read external stack table: %w", err)
	}
	table, err := decodeExternalStackTable(data)
	if err != nil {
		return fmt.Errorf("cudacapture: decode external stack table: %w", err)
	}
	if table.Version != 1 {
		return fmt.Errorf("cudacapture: unsupported external stack table version %d", table.Version)
	}
	ct.mu.Lock()
	ct.externalStackLosses += table.Losses
	ct.mu.Unlock()
	for _, record := range table.Stacks {
		if err := ct.RecordExternalStack(record.ExternalKind, record.ExternalID, record.Frames); err != nil {
			return err
		}
	}
	return nil
}

func decodeExternalStackTable(data []byte) (cudaprof.ExternalStackTable, error) {
	var table cudaprof.ExternalStackTable
	if err := json.Unmarshal(data, &table); err == nil {
		return table, nil
	}
	lines := bytes.Split(data, []byte{'\n'})
	if len(lines) < 2 {
		return table, errors.New("invalid sidecar records")
	}
	var header struct {
		Version    int             `json:"version"`
		Kind       string          `json:"kind"`
		Losses     json.RawMessage `json:"losses"`
		LowerBound bool            `json:"losses_lower_bound"`
	}
	if err := json.Unmarshal(lines[0], &header); err != nil {
		return table, err
	}
	if header.Kind != "header" {
		return table, errors.New("missing sidecar header")
	}
	table.Version = header.Version
	table.LossesLowerBound = header.LowerBound
	if len(header.Losses) != 0 {
		var text string
		if header.Losses[0] == '"' {
			if err := json.Unmarshal(header.Losses, &text); err != nil {
				return table, err
			}
		} else {
			text = string(header.Losses)
		}
		losses, err := strconv.ParseUint(text, 10, 64)
		if err != nil {
			return table, fmt.Errorf("invalid loss count: %w", err)
		}
		table.Losses = losses
	}
	for _, line := range lines[1:] {
		if len(bytes.TrimSpace(line)) == 0 {
			continue
		}
		var record struct {
			Version      int      `json:"version"`
			Kind         string   `json:"kind"`
			Process      uint64   `json:"process"`
			ExternalKind uint32   `json:"external_kind"`
			ExternalID   uint64   `json:"external_id"`
			Frames       []string `json:"frames"`
		}
		if err := json.Unmarshal(line, &record); err != nil {
			return table, err
		}
		if record.Kind != "stack" || record.Version != 1 {
			return table, fmt.Errorf("invalid sidecar record kind/version %q/%d", record.Kind, record.Version)
		}
		table.Stacks = append(table.Stacks, cudaprof.ExternalStackRecord{Process: record.Process, ExternalKind: record.ExternalKind, ExternalID: record.ExternalID, Frames: record.Frames})
	}
	return table, nil
}

// ReadCaptureExternalStackTable imports the stack sidecar attached to c.
func (ct *CorrelationTracker) ReadCaptureExternalStackTable(c *Capture) error {
	if c == nil {
		return errors.New("cudacapture: nil capture")
	}
	data, ok := c.Files[ExternalStacksName]
	if !ok {
		return fmt.Errorf("%w: capture has no %q member", ErrNoExternalStacks, ExternalStacksName)
	}
	return ct.ReadExternalStackTable(bytes.NewReader(data))
}

// ExternalStackLosses reports stack records omitted by the producer's budget.
func (ct *CorrelationTracker) ExternalStackLosses() uint64 {
	ct.mu.Lock()
	defer ct.mu.Unlock()
	return ct.externalStackLosses
}

// Join correlates runtime launches with device activities by exact ID matching.
// It partitions results into unambiguous matches, ambiguous joins, unmapped runtime launches,
// and unmapped device activities.
//
// Host call stacks are attached by exact external correlation ID match:
// agent external_correlation record -> CUDA correlation ID -> kernel activity.
// Timestamps are never used to infer correlation. External stacks replace
// RuntimeLaunch.Callstack only for the kernel with the matching external ID;
// otherwise a caller-supplied stack is preserved.
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
	runtimeExternal := make(map[uint64]uint64)
	runtimeMissingStack := make(map[uint64]bool)

	for _, id := range sortedIDs {
		rList := ct.runtimes[id]
		aList := ct.activities[id]

		switch {
		case len(rList) == 1 && len(aList) == 1:
			r := rList[0]
			a := aList[0]

			// CUPTI can report more than one framework namespace for a
			// correlation. Use the lowest kind, matching MatchExternal.
			extIDs := ct.cudaToExternal[id]
			var selected []externalStackKey
			for _, key := range extIDs {
				if len(selected) == 0 || key.kind < selected[0].kind {
					selected = []externalStackKey{key}
				} else if key.kind == selected[0].kind {
					selected = appendUniqueExternal(selected, key)
				}
			}
			if len(selected) > 1 {
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
					Reason:        fmt.Sprintf("correlation ID %d maps to multiple external IDs of kind %d", id, selected[0].kind),
				})
				continue
			}

			if len(selected) == 1 {
				ext := selected[0]
				extID := ext.id
				stacks := ct.externalStacks[ext]
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
						Reason:        fmt.Sprintf("external correlation ID %d of kind %d maps to multiple conflicting callstacks", extID, ext.kind),
					})
					continue
				}
				if len(stacks) == 1 {
					r.Callstack = append([]string(nil), stacks[0]...)
					ct.runtimes[id][0].Callstack = append([]string(nil), stacks[0]...)
				} else {
					r.Callstack = nil
					ct.runtimes[id][0].Callstack = nil
					runtimeMissingStack[id] = true
				}
				runtimeExternal[id] = extID
			}

			var dur uint64
			if a.End >= a.Start {
				dur = a.End - a.Start
			}
			matched = append(matched, CorrelationJoin{
				CorrelationID:    id,
				ExternalID:       runtimeExternal[id],
				HasExternalID:    runtimeExternal[id] != 0,
				CallstackMissing: runtimeMissingStack[id],
				KernelName:       r.KernelName,
				HostThreadID:     r.HostThreadID,
				Callstack:        append([]string(nil), r.Callstack...),
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
			Device:     j.Device,
		}
	}
	return cudaprof.ExportFoldedStacks(w, executions)
}

func appendUniqueExternal(slice []externalStackKey, val externalStackKey) []externalStackKey {
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
