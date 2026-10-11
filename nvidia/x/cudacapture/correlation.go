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

	"github.com/tmc/lib/nvidia/cupti"
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
	process uint64
	kind    uint32
	id      uint64
}

type externalCorrelationKey struct {
	process uint64
	id      uint64
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
// Imported stacks and activity links must have the same explicit process ID.
// Process zero and the processless RecordExternal methods use an unknown scope;
// they never join a known process. RuntimeLaunch has no process identifier, so
// reused CUDA IDs with multiple runtime or activity records remain ambiguous.
type CorrelationTracker struct {
	mu                            sync.Mutex
	runtimes                      map[uint64][]RuntimeLaunch
	activities                    map[uint64][]ActivityRecord
	externalStacks                map[externalStackKey][][]string // (kind, externalID) -> distinct stacks
	cudaToExternal                map[externalCorrelationKey][]externalStackKey
	externalStackLosses           uint64
	externalStackLossesLowerBound bool
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
		cudaToExternal: make(map[externalCorrelationKey][]externalStackKey),
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

// RecordExternalCorrelation associates an ID in the custom0 namespace with a CUDA ID.
func (ct *CorrelationTracker) RecordExternalCorrelation(externalID, correlationID uint64) error {
	return ct.RecordExternalCorrelationKind(uint32(cupti.ExternalCorrelationKindCustom0), externalID, correlationID)
}

// RecordExternalStack records a host stack in the custom0 correlation namespace.
func (ct *CorrelationTracker) RecordExternalStack(externalID uint64, stack []string) error {
	return ct.RecordExternalStackKind(uint32(cupti.ExternalCorrelationKindCustom0), externalID, stack)
}

// RecordExternalCorrelationKind records an association between an external correlation ID
// (pushed by Do or Correlate) and a CUDA runtime/driver correlation ID.
func (ct *CorrelationTracker) RecordExternalCorrelationKind(kind uint32, externalID uint64, correlationID uint64) error {
	return ct.recordExternalCorrelation(0, kind, externalID, correlationID)
}

func (ct *CorrelationTracker) recordExternalCorrelation(process uint64, kind uint32, externalID, correlationID uint64) error {
	if externalID == 0 {
		return errors.New("cudacapture: invalid zero external correlation ID")
	}
	if correlationID == 0 {
		return errors.New("cudacapture: invalid zero CUDA correlation ID")
	}
	ct.mu.Lock()
	defer ct.mu.Unlock()
	if ct.cudaToExternal == nil {
		ct.cudaToExternal = make(map[externalCorrelationKey][]externalStackKey)
	}
	key := externalStackKey{process: process, kind: kind, id: externalID}
	cuda := externalCorrelationKey{process: process, id: correlationID}
	ct.cudaToExternal[cuda] = appendUniqueExternal(ct.cudaToExternal[cuda], key)
	return nil
}

// RecordExternalStackKind records a host call stack associated with an external correlation ID.
// Granularity: the stack is the stack of the Do call site (not per-launch).
func (ct *CorrelationTracker) RecordExternalStackKind(kind uint32, externalID uint64, stack []string) error {
	if externalID == 0 {
		return errors.New("cudacapture: invalid zero external correlation ID")
	}
	ct.mu.Lock()
	defer ct.mu.Unlock()
	if ct.externalStacks == nil {
		ct.externalStacks = make(map[externalStackKey][][]string)
	}
	ct.recordExternalStackLocked(externalStackKey{kind: kind, id: externalID}, stack)
	return nil
}

func (ct *CorrelationTracker) recordExternalStackLocked(key externalStackKey, stack []string) {
	existing := ct.externalStacks[key]
	for _, s := range existing {
		if stringSlicesEqual(s, stack) {
			return
		}
	}
	st := make([]string, len(stack))
	copy(st, stack)
	ct.externalStacks[key] = append(ct.externalStacks[key], st)
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
		return ct.recordExternalCorrelation(rec.Process, rec.ExternalKind, rec.ExternalID, corrID)
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
	data, err := io.ReadAll(io.LimitReader(r, maxExternalStackBytes+1))
	if err != nil {
		return fmt.Errorf("cudacapture: read external stack table: %w", err)
	}
	if len(data) > maxExternalStackBytes {
		return errors.New("cudacapture: external stack table exceeds byte budget")
	}
	table, err := decodeExternalStackTable(data)
	if err != nil {
		return fmt.Errorf("cudacapture: decode external stack table: %w", err)
	}
	if table.Version != 1 {
		return fmt.Errorf("cudacapture: unsupported external stack table version %d", table.Version)
	}
	ct.mu.Lock()
	defer ct.mu.Unlock()
	if ^uint64(0)-ct.externalStackLosses < table.Losses {
		ct.externalStackLosses = ^uint64(0)
		ct.externalStackLossesLowerBound = true
	} else {
		ct.externalStackLosses += table.Losses
	}
	ct.externalStackLossesLowerBound = ct.externalStackLossesLowerBound || table.LossesLowerBound
	if ct.externalStacks == nil {
		ct.externalStacks = make(map[externalStackKey][][]string)
	}
	for _, record := range table.Stacks {
		ct.recordExternalStackLocked(externalStackKey{process: record.Process, kind: record.ExternalKind, id: record.ExternalID}, record.Frames)
	}
	return nil
}

const (
	maxExternalStackBytes    = 16 << 20
	maxExternalStackRows     = 16384
	maxExternalStackRowBytes = 64 << 10
)

// decodeExternalStackTable validates the entire import before any tracker mutation.
// Legacy process zero is an unknown scope; it is never promoted to a known PID.
func decodeExternalStackTable(data []byte) (cudaprof.ExternalStackTable, error) {
	var table cudaprof.ExternalStackTable
	var object map[string]json.RawMessage
	if json.Unmarshal(data, &object) == nil {
		if _, journal := object["kind"]; !journal {
			fields, err := externalStackObject(data, "version", "losses", "losses_lower_bound", "discarded_records", "discarded_records_lower_bound", "stacks")
			if err != nil {
				return table, err
			}
			if err := json.Unmarshal(data, &table); err != nil {
				return table, err
			}
			var rows []json.RawMessage
			if err := json.Unmarshal(fields["stacks"], &rows); err != nil {
				return table, err
			}
			if len(rows) > maxExternalStackRows {
				return table, errors.New("too many stack records")
			}
			for _, row := range rows {
				if _, err := externalStackObject(row, "process", "external_kind", "external_id", "frames"); err != nil {
					return table, err
				}
			}
			return validateExternalStackTable(table)
		}
	}
	if len(data) == 0 || data[len(data)-1] != '\n' {
		return table, errors.New("incomplete sidecar journal")
	}
	lines := bytes.Split(data[:len(data)-1], []byte{'\n'})
	type sidecarRow struct {
		Version      int             `json:"version"`
		Kind         string          `json:"kind"`
		Losses       json.RawMessage `json:"losses"`
		LowerBound   bool            `json:"losses_lower_bound"`
		Process      uint64          `json:"process"`
		Epoch        string          `json:"epoch"`
		ExternalKind uint32          `json:"external_kind"`
		ExternalID   uint64          `json:"external_id"`
		Frames       []string        `json:"frames"`
	}
	var header sidecarRow
	for n, line := range lines {
		if len(line) > maxExternalStackRowBytes {
			return table, errors.New("sidecar row exceeds byte budget")
		}
		fields, err := externalStackObject(line, "version", "kind", "losses", "losses_lower_bound", "process", "epoch", "external_kind", "external_id", "frames")
		if err != nil {
			return table, err
		}
		var row sidecarRow
		if err := json.Unmarshal(line, &row); err != nil {
			return table, err
		}
		if row.Version != 1 {
			return table, fmt.Errorf("unsupported external stack table version %d", row.Version)
		}
		if len(row.Epoch) > 128 {
			return table, errors.New("sidecar epoch exceeds byte budget")
		}
		if n == 0 {
			if row.Kind != "header" {
				return table, errors.New("missing sidecar header")
			}
			header = row
			table.Version = 1
		} else if row.Kind == "header" || row.Process != header.Process || row.Epoch != header.Epoch {
			return table, errors.New("sidecar record differs from header scope")
		}
		switch row.Kind {
		case "header", "losses":
			if _, ok := fields["frames"]; ok {
				return table, errors.New("count record contains frames")
			}
			if _, ok := fields["external_id"]; ok {
				return table, errors.New("count record contains external ID")
			}
			if _, ok := fields["external_kind"]; ok {
				return table, errors.New("count record contains external kind")
			}
			if n == 0 && len(row.Losses) == 0 {
				row.Losses = json.RawMessage("0")
			}
			losses, err := externalStackCount(row.Losses)
			if err != nil {
				return table, err
			}
			if losses < table.Losses {
				return table, errors.New("decreasing sidecar loss count")
			}
			table.Losses = losses
			table.LossesLowerBound = table.LossesLowerBound || row.LowerBound
		case "stack":
			if _, ok := fields["losses"]; ok {
				return table, errors.New("stack record contains loss count")
			}
			if row.LowerBound {
				return table, errors.New("stack record contains loss qualification")
			}
			if len(table.Stacks) >= maxExternalStackRows {
				return table, errors.New("too many stack records")
			}
			table.Stacks = append(table.Stacks, cudaprof.ExternalStackRecord{Process: row.Process, ExternalKind: row.ExternalKind, ExternalID: row.ExternalID, Frames: row.Frames})
		default:
			return table, fmt.Errorf("invalid sidecar record kind %q", row.Kind)
		}
	}
	return validateExternalStackTable(table)
}

func externalStackCount(raw json.RawMessage) (uint64, error) {
	text := string(raw)
	if len(raw) > 0 && raw[0] == '"' {
		if err := json.Unmarshal(raw, &text); err != nil {
			return 0, err
		}
	}
	if len(text) == 0 || len(text) > 20 {
		return 0, errors.New("invalid sidecar loss count")
	}
	for _, c := range text {
		if c < '0' || c > '9' {
			return 0, errors.New("invalid sidecar loss count")
		}
	}
	count, err := strconv.ParseUint(text, 10, 64)
	if err != nil {
		return 0, fmt.Errorf("invalid sidecar loss count: %w", err)
	}
	return count, nil
}

func externalStackObject(data []byte, allowed ...string) (map[string]json.RawMessage, error) {
	d := json.NewDecoder(bytes.NewReader(data))
	token, err := d.Token()
	if err != nil || token != json.Delim('{') {
		return nil, errors.New("invalid sidecar object")
	}
	fields := make(map[string]json.RawMessage)
	for d.More() {
		token, err := d.Token()
		if err != nil {
			return nil, err
		}
		key, ok := token.(string)
		if !ok {
			return nil, errors.New("invalid sidecar key")
		}
		if _, seen := fields[key]; seen {
			return nil, errors.New("duplicate sidecar key")
		}
		valid := false
		for _, name := range allowed {
			valid = valid || key == name
		}
		if !valid {
			return nil, fmt.Errorf("unknown sidecar key %q", key)
		}
		var value json.RawMessage
		if err := d.Decode(&value); err != nil {
			return nil, err
		}
		if bytes.Equal(bytes.TrimSpace(value), []byte("null")) {
			return nil, fmt.Errorf("null sidecar field %q", key)
		}
		fields[key] = value
	}
	if _, err := d.Token(); err != nil {
		return nil, err
	}
	if _, err := d.Token(); err != io.EOF {
		return nil, errors.New("trailing sidecar object data")
	}
	return fields, nil
}

func validateExternalStackTable(table cudaprof.ExternalStackTable) (cudaprof.ExternalStackTable, error) {
	if table.Version != 1 {
		return table, fmt.Errorf("unsupported external stack table version %d", table.Version)
	}
	if len(table.Stacks) > maxExternalStackRows {
		return table, errors.New("too many stack records")
	}
	for _, row := range table.Stacks {
		if row.ExternalID == 0 || len(row.Frames) == 0 || len(row.Frames) > 65 {
			return table, errors.New("invalid stack record")
		}
		total := 0
		for _, frame := range row.Frames {
			total += len(frame)
			if len(frame) == 0 || len(frame) > 4096 || total > 8192 {
				return table, errors.New("stack frames exceed byte budget")
			}
		}
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

// ExternalStackLossesLowerBound reports whether the loss count is a lower bound.
// Producer journal checkpoints and a saturated total are lower bounds.
func (ct *CorrelationTracker) ExternalStackLossesLowerBound() bool {
	ct.mu.Lock()
	defer ct.mu.Unlock()
	return ct.externalStackLossesLowerBound
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
			extIDs := ct.cudaToExternal[externalCorrelationKey{process: a.Process, id: id}]
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
