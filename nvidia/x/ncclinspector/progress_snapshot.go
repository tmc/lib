package ncclinspector

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"slices"
)

// ProgressSchema selects the task-owned monitor sidecar. Stock NCCL and the
// example profiler do not emit this format. The patch is retained in testdata.
const ProgressSchema = "nccl.progress-monitor.diagnostic.v1"
const progressCommit = "12df1a11afad322be5a204a2db890161cbf8131d"

// ProgressCapture supplies independent expected identity and the raw artifact
// digest. These strings are caller assertions, not discovered producer facts.
// Worker is the producer's local worker incarnation, not a process-wide epoch.
type ProgressCapture struct {
	SHA256, BuildID, Host, BootEpoch, ProcessEpoch, CaptureID string
	PID, Worker                                               uint64
	Device                                                    int
}

// ProgressSnapshot preserves exact diagnostic mirror arrays. These are not an
// atomic snapshot or an operation-completion or GPU-failure observation.
type ProgressSnapshot struct {
	Counts   [16]uint64 `json:"completed_work_count"`
	GPUTimes [16]uint64 `json:"completed_time_ns"`
	Active   [64]uint64 `json:"channel_active_slot_mask"`
}

// ProgressRecord is a detached raw sidecar row. Integer times are nanoseconds
// in their declared, separate host steady-clock and GPU globaltimer domains.
// Optional communicator and mirror fields are absent on worker-terminal rows.
// Raw preserves the original line, including its newline.
type ProgressRecord struct {
	Schema         string            `json:"schema"`
	SourceCommit   string            `json:"source_commit"`
	BuildID        string            `json:"build_id"`
	Host           string            `json:"host"`
	BootEpoch      string            `json:"boot_epoch"`
	ProcessEpoch   string            `json:"process_epoch"`
	CaptureID      string            `json:"capture_id"`
	PID            uint64            `json:"pid"`
	Device         int               `json:"device"`
	Worker         uint64            `json:"worker"`
	Sequence       uint64            `json:"sequence"`
	Event          string            `json:"event"`
	ObservedNS     uint64            `json:"observed_ns"`
	HostClock      string            `json:"host_clock"`
	GPUClock       string            `json:"gpu_clock"`
	TimeUnit       string            `json:"time_unit"`
	CounterUnit    string            `json:"counter_unit"`
	SnapshotAtomic bool              `json:"snapshot_atomic"`
	UpstreamLoss   string            `json:"upstream_loss"`
	PollMS         uint64            `json:"poll_ms"`
	StaleMS        uint64            `json:"stale_ms"`
	MaxRecords     uint64            `json:"max_records"`
	MaxBytes       uint64            `json:"max_bytes"`
	Terminal       bool              `json:"terminal"`
	RetainedBefore uint64            `json:"exporter_retained_before"`
	DroppedBefore  uint64            `json:"exporter_dropped_before"`
	Untracked      uint64            `json:"untracked_registrations"`
	Code           int32             `json:"code"`
	MirrorPresent  bool              `json:"mirror_present"`
	Communicator   string            `json:"comm_hash,omitempty"`
	Rank           int               `json:"rank,omitempty"`
	Registration   uint64            `json:"registration,omitempty"`
	Generation     uint64            `json:"generation,omitempty"`
	LaunchStartNS  uint64            `json:"launch_start_ns,omitempty"`
	LaunchEndNS    uint64            `json:"launch_end_ns,omitempty"`
	LaunchCode     int32             `json:"launch_code,omitempty"`
	CopyPending    bool              `json:"copy_pending,omitempty"`
	EventRecorded  bool              `json:"event_recorded,omitempty"`
	CopyStale      bool              `json:"copy_stale,omitempty"`
	CounterReset   string            `json:"counter_reset,omitempty"`
	SuccessMinNS   uint64            `json:"copy_success_min_ns,omitempty"`
	SuccessMaxNS   uint64            `json:"copy_success_max_ns,omitempty"`
	Raw            []byte            `json:"-"`
	Mirror         *ProgressSnapshot `json:"-"`
	copyAdmitted   bool
}

// ProgressTrace retains a bounded custom sidecar and its checked private rows.
// Its zero value cannot be bound. Records returns detached values; changing
// them cannot change a later summary. A terminal row closes only this exporter.
type ProgressTrace struct {
	capture  ProgressCapture
	rows     []ProgressRecord
	raw      []byte
	complete bool
	read     bool
}

// Records returns detached reported rows, without authenticating the producer.
func (t *ProgressTrace) Records() []ProgressRecord {
	if t == nil {
		return nil
	}
	rows := slices.Clone(t.rows)
	for i := range rows {
		rows[i].Raw = slices.Clone(rows[i].Raw)
		if rows[i].Mirror != nil {
			m := *rows[i].Mirror
			rows[i].Mirror = &m
		}
	}
	return rows
}

func progressToken(s string) bool {
	if len(s) == 0 || len(s) > 128 {
		return false
	}
	for _, c := range s {
		if !(c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c == '_' || c == '-' || c == '.' || c == ':') {
			return false
		}
	}
	return true
}

func checkProgressCapture(c ProgressCapture) error {
	b, err := hex.DecodeString(c.SHA256)
	if err != nil || len(b) != sha256.Size || hex.EncodeToString(b) != c.SHA256 || c.PID == 0 || c.Worker == 0 || c.Worker > 64 || c.Device < 0 || c.Device >= 64 {
		return errors.New("progress: invalid capture digest or process identity")
	}
	for _, s := range []string{c.BuildID, c.Host, c.BootEpoch, c.ProcessEpoch, c.CaptureID} {
		if !progressToken(s) {
			return errors.New("progress: invalid capture identity token")
		}
	}
	return nil
}

type progressContextReader struct {
	ctx context.Context
	r   io.Reader
}

func (r progressContextReader) Read(p []byte) (int, error) {
	if err := r.ctx.Err(); err != nil {
		return 0, err
	}
	return r.r.Read(p)
}

// ReadProgressSnapshots reads the separately versioned task-owned sidecar,
// checks exact wire presence, integer domains, lifecycle and output counters,
// and retains incomplete output with dependent coverage withheld. Limits also
// have producer caps:16 MiB,16 KiB/line and4097 retained rows. Missing terminal,
// reported omissions or missing retained rows cannot establish complete export.
// The caller must arrange interruption of a blocking Reader; cancellation is
// checked before reads and between records. It makes no native/provider calls.
func ReadProgressSnapshots(ctx context.Context, r io.Reader, capture ProgressCapture, limits Limits) (*ProgressTrace, error) {
	if ctx == nil || r == nil {
		return nil, errors.New("progress: nil context or reader")
	}
	if err := checkProgressCapture(capture); err != nil {
		return nil, err
	}
	l, err := limits.checked()
	if err != nil {
		return nil, err
	}
	l.Bytes, l.Line, l.Records = min(l.Bytes, 16<<20), min(l.Line, 16<<10), min(l.Records, 4097)
	raw, err := io.ReadAll(io.LimitReader(progressContextReader{ctx, r}, int64(l.Bytes)+1))
	if err != nil {
		return nil, fmt.Errorf("progress read: %w", err)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(raw) == 0 || len(raw) > l.Bytes || raw[len(raw)-1] != '\n' {
		return nil, errors.New("progress: empty, incomplete or oversized JSONL artifact")
	}
	if fmt.Sprintf("%x", sha256.Sum256(raw)) != capture.SHA256 {
		return nil, errors.New("progress: artifact digest differs")
	}
	t := &ProgressTrace{capture: capture, raw: raw, complete: true, read: true}
	var previous *ProgressRecord
	entries := make(map[uint64]ProgressRecord)
	for remaining := raw; len(remaining) != 0; {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		i := bytes.IndexByte(remaining, '\n')
		line := remaining[:i+1]
		remaining = remaining[i+1:]
		if len(line) > l.Line || len(t.rows) >= l.Records {
			return nil, errors.New("progress: line or record limit exceeded")
		}
		row, err := readProgressRow(line, capture)
		if err != nil {
			return nil, fmt.Errorf("progress row%d: %w", len(t.rows), err)
		}
		if previous == nil {
			if row.Sequence != 1 || row.RetainedBefore != 0 {
				t.complete = false
			}
		} else {
			p := previous
			if p.Terminal || row.Sequence <= p.Sequence || row.RetainedBefore <= p.RetainedBefore || row.DroppedBefore < p.DroppedBefore || row.Untracked < p.Untracked || row.ObservedNS < p.ObservedNS {
				return nil, errors.New("progress: contradictory order, terminal or output counts")
			}
			if row.PollMS != p.PollMS || row.StaleMS != p.StaleMS || row.MaxRecords != p.MaxRecords || row.MaxBytes != p.MaxBytes {
				return nil, errors.New("progress: configuration changed within worker")
			}
			if row.RetainedBefore != p.RetainedBefore+1 || row.DroppedBefore != p.DroppedBefore {
				t.complete = false
				clear(entries)
			}
		}
		if row.DroppedBefore != 0 || row.Untracked != 0 {
			t.complete = false
		}
		if err := checkProgressLifecycle(&row, entries, !t.complete); err != nil {
			return nil, err
		}
		t.rows = append(t.rows, row)
		previous = &t.rows[len(t.rows)-1]
	}
	last := t.rows[len(t.rows)-1]
	t.complete = t.complete && last.Terminal && last.Event == "worker_closed"
	if uint64(len(raw)) > last.MaxBytes || uint64(len(t.rows)) > last.MaxRecords+1 {
		return nil, errors.New("progress: declared output budget exceeded")
	}
	return t, nil
}

var progressCommonKeys = []string{"schema", "source_commit", "build_id", "host", "boot_epoch", "process_epoch", "capture_id", "pid", "device", "worker", "sequence", "event", "observed_ns", "host_clock", "gpu_clock", "time_unit", "counter_unit", "snapshot_atomic", "upstream_loss", "poll_ms", "stale_ms", "max_records", "max_bytes", "terminal", "exporter_retained_before", "exporter_dropped_before", "untracked_registrations", "code", "mirror_present"}
var progressEntryKeys = []string{"comm_hash", "rank", "registration", "generation", "launch_start_ns", "launch_end_ns", "launch_code", "copy_pending", "event_recorded", "copy_stale", "counter_reset"}
var progressMirrorKeys = []string{"copy_success_min_ns", "copy_success_max_ns", "completed_work_count", "completed_time_ns", "channel_active_slot_mask"}

func readProgressRow(line []byte, c ProgressCapture) (ProgressRecord, error) {
	var row ProgressRecord
	if err := uniqueJSON(line); err != nil {
		return row, err
	}
	m, err := requiredObject(line, progressCommonKeys...)
	if err != nil {
		return row, err
	}
	// Struct integer decoding refuses fractional/exponential/string/overflow arms.
	if err := json.Unmarshal(line, &row); err != nil {
		return row, err
	}
	if row.Schema != ProgressSchema || row.SourceCommit != progressCommit || row.BuildID != c.BuildID || row.Host != c.Host || row.BootEpoch != c.BootEpoch || row.ProcessEpoch != c.ProcessEpoch || row.CaptureID != c.CaptureID || row.PID != c.PID || row.Device != c.Device || row.Worker != c.Worker {
		return row, errors.New("progress: unsupported source or capture identity mismatch")
	}
	if row.HostClock != "std.chrono.steady_clock" || row.GPUClock != "globaltimer" || row.TimeUnit != "nanoseconds" || row.CounterUnit != "owned_completed_work" || row.SnapshotAtomic || row.UpstreamLoss != "unknown" || row.PollMS < 50 || row.PollMS > 2147483647 || row.StaleMS < 1000 || row.StaleMS > 2147483647 || row.MaxRecords < 1 || row.MaxRecords > 4096 || row.MaxBytes < 32768 || row.MaxBytes > 16<<20 {
		return row, errors.New("progress: unsupported units, validity or budgets")
	}
	// sequence = retained-before + dropped-before + current row, without wrap.
	if row.Code < 0 || row.LaunchCode < 0 || row.RetainedBefore > row.MaxRecords || !row.Terminal && row.RetainedBefore == row.MaxRecords {
		return row, errors.New("progress: invalid code or retained-row budget")
	}
	if row.RetainedBefore == ^uint64(0) || row.DroppedBefore > ^uint64(0)-row.RetainedBefore-1 || row.Sequence != row.RetainedBefore+row.DroppedBefore+1 {
		return row, errors.New("progress: invalid output accounting")
	}
	allowed := slices.Clone(progressCommonKeys)
	workerTerminal := row.Event == "worker_closed" || row.Event == "initialization_failed"
	if row.Terminal != workerTerminal || row.Terminal && (row.Code != 0 || row.MirrorPresent) {
		return row, errors.New("progress: invalid terminal")
	}
	if !workerTerminal {
		if _, err := requiredObject(line, progressEntryKeys...); err != nil {
			return row, err
		}
		allowed = append(allowed, progressEntryKeys...)
		if len(row.Communicator) != 16 || row.Registration == 0 || row.Rank < 0 || row.Rank > 2147483647 || row.CounterReset != "unknown" || row.LaunchEndNS < row.LaunchStartNS || row.ObservedNS < row.LaunchEndNS {
			return row, errors.New("progress: invalid communicator, generation or time bounds")
		}
		b, err := hex.DecodeString(row.Communicator)
		if err != nil || len(b) != 8 || hex.EncodeToString(b) != row.Communicator {
			return row, errors.New("progress: noncanonical communicator hash")
		}
	}
	if row.MirrorPresent {
		if _, err := requiredObject(line, progressMirrorKeys...); err != nil {
			return row, err
		}
		allowed = append(allowed, progressMirrorKeys...)
		counts, err := readProgressArray(m["completed_work_count"], 16)
		if err != nil {
			return row, err
		}
		times, err := readProgressArray(m["completed_time_ns"], 16)
		if err != nil {
			return row, err
		}
		active, err := readProgressArray(m["channel_active_slot_mask"], 64)
		if err != nil {
			return row, err
		}
		row.Mirror = &ProgressSnapshot{}
		copy(row.Mirror.Counts[:], counts)
		copy(row.Mirror.GPUTimes[:], times)
		copy(row.Mirror.Active[:], active)
		if row.Event != "copy_complete" || row.Code != 0 || row.LaunchCode != 0 || row.CopyPending || !row.EventRecorded || row.CopyStale || row.Generation == 0 || row.ObservedNS-row.LaunchStartNS >= row.StaleMS*1000000 || row.SuccessMinNS != row.LaunchStartNS || row.SuccessMaxNS != row.ObservedNS {
			return row, errors.New("progress: mirror lacks successful nonstale copy bounds")
		}
	} else if row.Event == "copy_complete" {
		return row, errors.New("progress: complete copy lacks mirror")
	}
	for k := range m {
		if !slices.Contains(allowed, k) {
			return row, fmt.Errorf("progress: unexpected field %s", k)
		}
	}
	row.Raw = slices.Clone(line)
	return row, nil
}

func checkProgressLifecycle(row *ProgressRecord, entries map[uint64]ProgressRecord, partial bool) error {
	if row.Terminal {
		return nil
	}
	old, known := entries[row.Registration]
	if known && (old.Event == "unregistered_drained" || old.Event == "unregister_drain_failed" || old.Event == "unregister_requested" && row.Event != "unregistered_drained" && row.Event != "unregister_drain_failed") {
		return errors.New("progress: event after registration retirement")
	}
	if !known && !partial && row.Event != "registered" && row.Event != "registration_untracked" {
		return errors.New("progress: registration record is missing")
	}
	if known && (old.Communicator != row.Communicator || old.Rank != row.Rank) {
		return errors.New("progress: registration identity changed")
	}
	sameCopy := known && old.Generation == row.Generation && old.LaunchStartNS == row.LaunchStartNS && old.LaunchEndNS == row.LaunchEndNS && old.LaunchCode == row.LaunchCode
	switch row.Event {
	case "registered", "registration_untracked":
		if known || row.Generation != 0 || row.LaunchStartNS != 0 || row.LaunchEndNS != 0 || row.LaunchCode != 0 || row.Code != 0 || row.CopyPending || row.EventRecorded || row.CopyStale {
			return errors.New("progress: invalid registration")
		}
		if row.Event == "registration_untracked" && row.Untracked == 0 {
			return errors.New("progress: missing untracked count")
		}
	case "copy_launched", "copy_launch_failed":
		if row.Generation == 0 || row.Code != row.LaunchCode || row.CopyPending != (row.Code == 0) || row.EventRecorded || row.CopyStale || row.Event == "copy_launched" && row.Code != 0 || row.Event == "copy_launch_failed" && row.Code == 0 || known && (old.CopyPending || row.Generation <= old.Generation || old.Event == "registration_untracked" || old.Event == "unregister_requested" || old.Event == "unregistered_drained" || old.Event == "unregister_drain_failed") {
			return errors.New("progress: invalid copy launch")
		}
	case "copy_pending", "copy_pending_stale":
		if row.Generation == 0 || row.Code != 0 || row.LaunchCode != 0 || !row.CopyPending || !row.EventRecorded || row.CopyStale != (row.Event == "copy_pending_stale") {
			return errors.New("progress: invalid pending copy")
		}
		if known && (!sameCopy || !old.CopyPending || old.CopyStale && !row.CopyStale) {
			return errors.New("progress: pending generation differs")
		}
	case "event_record_failed":
		if row.Code == 0 || row.CopyPending || row.EventRecorded || row.LaunchCode != 0 || known && (!sameCopy || !old.CopyPending) {
			return errors.New("progress: invalid event-record failure")
		}
	case "copy_complete", "copy_completion_unavailable":
		if row.Generation == 0 || row.CopyPending || row.LaunchCode != 0 || known && (!sameCopy || !old.CopyPending || old.CopyStale && !row.CopyStale) {
			return errors.New("progress: copy completion lacks admitted launch")
		}
		row.copyAdmitted = known && sameCopy && old.CopyPending && row.EventRecorded && !row.CopyStale && row.Event == "copy_complete"
	case "unregister_requested":
		if row.Code != 0 || known && !sameCopy {
			return errors.New("progress: invalid unregister state")
		}
	case "unregistered_drained", "unregister_drain_failed":
		if row.Event == "unregistered_drained" && row.Code != 0 || row.Event == "unregister_drain_failed" && row.Code == 0 || known && (!sameCopy || old.Event != "unregister_requested") {
			return errors.New("progress: invalid unregister drain")
		}
	default:
		return fmt.Errorf("progress: unsupported event %q", row.Event)
	}
	entries[row.Registration] = *row
	return nil
}

// A null JSON number must not acquire Go's zero value inside an integer slice.
func readProgressArray(raw json.RawMessage, size int) ([]uint64, error) {
	var cells []json.RawMessage
	if err := json.Unmarshal(raw, &cells); err != nil {
		return nil, err
	}
	if len(cells) != size {
		return nil, errors.New("progress: wrong mirror dimensions")
	}
	values := make([]uint64, size)
	for i, cell := range cells {
		if bytes.Equal(bytes.TrimSpace(cell), []byte("null")) {
			return nil, errors.New("progress: absent mirror integer")
		}
		if err := json.Unmarshal(cell, &values[i]); err != nil {
			return nil, err
		}
	}
	return values, nil
}
