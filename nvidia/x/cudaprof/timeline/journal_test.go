package timeline

import (
	"errors"
	"io"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/internal/cudainject/wire"
	"github.com/tmc/lib/nvidia/x/cudaprof/internal/journaltest"
)

func TestJournalsRefusals(t *testing.T) {
	journal := journaltest.New().Kernel(10, 20, 1).Finalize()
	tests := []struct {
		name   string
		files  map[string][]byte
		status map[string]any
		want   string
		kind   error
	}{
		{
			name:  "journal omitted",
			files: map[string][]byte{},
			status: map[string]any{
				"run_completeness": "complete",
				"omitted_files": []map[string]any{{
					"name": "capture.journal", "bytes": 812345678,
					"sha256": "abc123", "reason": "exceeds capture file budget",
				}},
			},
			want: "reconstructed: capture.journal (812345678 bytes, sha256 abc123: exceeds capture file budget)",
			kind: ErrJournalOmitted,
		},
		{
			name:   "no journal",
			files:  map[string][]byte{},
			status: nil,
			want:   "capture-status.json is absent",
		},
		{
			name:   "records dropped",
			files:  map[string][]byte{"capture.journal": journal},
			status: map[string]any{"dropped_records": 17, "activity_records": 3},
			want:   "capture dropped 17 record(s) and retained 3",
			kind:   ErrRecordsDropped,
		},
		{
			name:   "partial run",
			files:  map[string][]byte{"capture.journal": journal},
			status: map[string]any{"run_completeness": "partial", "unqualified_generations": 2},
			want:   "with 2 unqualified generation(s)",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := Journals(journaltest.Capture(tt.files, tt.status))
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("Journals error = %v, want %q", err, tt.want)
			}
			for _, kind := range []error{ErrJournalOmitted, ErrRecordsDropped} {
				if got, want := errors.Is(err, kind), kind == tt.kind; got != want {
					t.Errorf("errors.Is(err, %v) = %v, want %v", kind, got, want)
				}
			}
		})
	}

	got, err := Journals(journaltest.Capture(map[string][]byte{"capture.journal": journal, "kernels.ptx": nil}, journaltest.CompleteStatus()))
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got["capture.journal"] == nil {
		t.Errorf("Journals = %v, want only capture.journal", got)
	}
}

func TestOpenJournalLimit(t *testing.T) {
	raw := []byte("12345678")
	compressed := journaltest.Gzip(raw)
	for _, tt := range []struct {
		name    string
		data    []byte
		limit   int64
		want    string
		wantErr string
	}{
		{name: "plain exact", data: raw, limit: int64(len(raw)), want: string(raw)},
		{name: "gzip exact", data: compressed, limit: int64(len(raw)), want: string(raw)},
		{name: "plain over", data: raw, limit: int64(len(raw) - 1), wantErr: "journal exceeds"},
		{name: "gzip over", data: compressed, limit: int64(len(raw) - 1), wantErr: "journal exceeds"},
		{name: "invalid", data: raw, limit: 0, wantErr: "invalid journal byte limit"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			r, err := OpenJournal(tt.data, tt.limit)
			var got []byte
			if err == nil {
				got, err = io.ReadAll(r)
				_ = r.Close()
			}
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("error = %v, want %q", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if string(got) != tt.want {
				t.Errorf("journal = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestDecodeNoCUDAJournal(t *testing.T) {
	noted := map[string]any{"kind": "capability", "device_active_note": true}
	worker := func(capability map[string]any, body func(*journaltest.Builder)) []byte {
		b := journaltest.NewWithCapability(capability)
		if body != nil {
			body(b)
		}
		return b.Unfinalized()
	}
	tests := []struct {
		name    string
		journal []byte
		wantErr string // "" admits the journal as no-CUDA
	}{
		{name: "bootstrap and capability only", journal: worker(noted, nil)},
		{name: "host api only", journal: worker(noted, func(b *journaltest.Builder) { b.API(1, 2, 99) })},
		{name: "capture range edge only", journal: worker(noted, func(b *journaltest.Builder) {
			b.Append(wire.KindActivity, map[string]any{"kind": "activity", "subtype": "capture_range", "valid": true, "start": 1, "end": 1})
		})},
		{name: "producer without the note", journal: worker(map[string]any{"kind": "capability"}, nil), wantErr: "never finalized"},
		{name: "device active", journal: worker(noted, func(b *journaltest.Builder) {
			b.Append(wire.KindActivity, map[string]any{"kind": "activity", "subtype": "device_active"})
		}), wantErr: "never finalized"},
		{name: "device work", journal: worker(noted, func(b *journaltest.Builder) { b.Kernel(10, 20, 1) }), wantErr: "never finalized"},
		{name: "producer failure", journal: worker(noted, func(b *journaltest.Builder) {
			b.Append(wire.KindFailure, map[string]any{"kind": "failure", "state": "producer_failure"})
		}), wantErr: "never finalized"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			journals := map[string][]byte{
				"worker-a-capture.journal": tt.journal,
				"worker-b-capture.journal": journaltest.New().Kernel(100, 200, 1).API(90, 95, 1).Finalize(),
			}
			got, err := Decode(journals, MaxRecords, 1<<20)
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Errorf("error = %v, want %q", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("refused a no-CUDA journal: %v", err)
			}
			if len(got.Journals) != 2 || !got.Journals[0].NoCUDA || got.Journals[1].NoCUDA {
				t.Errorf("journals = %+v, want worker-a marked no-CUDA and worker-b not", got.Journals)
			}
			if len(got.Activities) != 1 {
				t.Errorf("activities = %d, want 1: the no-CUDA journal contributes none", len(got.Activities))
			}
			if _, ok := got.APIs[99]; ok || len(got.APIs) != 1 {
				t.Errorf("apis = %v, want only worker-b's: a set-aside journal leaves nothing behind", got.APIs)
			}
		})
	}
}

func TestMeasure(t *testing.T) {
	small := journaltest.New().Kernel(1, 2, 1).Finalize()
	large := journaltest.New().Kernel(1, 2, 1).Kernel(3, 4, 2).Kernel(5, 6, 3).Finalize()
	journals := map[string][]byte{"a.journal": small, "b.journal.gz": journaltest.Gzip(large)}
	_, err := Decode(journals, 2, int64(len(large)))
	if !errors.Is(err, wire.ErrRecordLimit) {
		t.Fatalf("Decode under 2 records: err = %v, want %v", err, wire.ErrRecordLimit)
	}
	records, size, err := Measure(journals)
	if err != nil {
		t.Fatal(err)
	}
	if size != int64(len(large)) {
		t.Errorf("size = %d, want %d", size, len(large))
	}
	if _, err := Decode(journals, records, size); err != nil {
		t.Errorf("Decode under measured bounds %d records, %d bytes: %v", records, size, err)
	}
	if _, err := Decode(journals, records, size-1); !errors.Is(err, ErrJournalBytes) {
		t.Errorf("Decode under %d bytes: err = %v, want %v", size-1, err, ErrJournalBytes)
	}
	if _, err := Decode(journals, records-1, size); !errors.Is(err, wire.ErrRecordLimit) {
		t.Errorf("Decode under %d records: err = %v, want %v", records-1, err, wire.ErrRecordLimit)
	}
}

func TestDecodeRanges(t *testing.T) {
	marker := func(name string, id, flags uint32, thread uint32, at uint64) map[string]any {
		return map[string]any{"kind": "activity", "subtype": "marker", "valid": true, "name": name,
			"marker_id": id, "marker_flags": flags, "thread_id": thread, "process_id": 7, "start": at, "end": at}
	}
	b := journaltest.New().Kernel(100, 200, 1)
	b.Append(wire.KindAPI, map[string]any{"kind": "api", "start": 90, "end": 95, "correlation": 1, "thread_id": 3})
	// Nested ranges sharing an id close innermost first. The END is
	// appended before its START to check that pairing sorts by time.
	b.Append(wire.KindActivity, marker("", 5, MarkerEnd, 3, 300))
	b.Append(wire.KindActivity, marker("outer", 5, MarkerStart, 3, 10))
	b.Append(wire.KindActivity, marker("inner", 5, MarkerStart, 3, 20))
	b.Append(wire.KindActivity, marker("", 5, MarkerEnd, 3, 250))
	b.Append(wire.KindActivity, marker("open", 6, MarkerStart, 4, 30))
	got, err := Decode(map[string][]byte{"capture.journal": b.Finalize()}, MaxRecords, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	want := []Range{
		{Journal: "capture.journal", Name: "inner", Process: 7, Thread: 3, Start: 20, End: 250},
		{Journal: "capture.journal", Name: "outer", Process: 7, Thread: 3, Start: 10, End: 300},
	}
	if !slices.Equal(got.Ranges, want) {
		t.Errorf("ranges = %+v, want %+v", got.Ranges, want)
	}
	if got.UnpairedMarkers != 1 {
		t.Errorf("unpaired markers = %d, want 1", got.UnpairedMarkers)
	}
	if len(got.Activities) != 1 {
		t.Errorf("activities = %d, want 1: markers are not device work", len(got.Activities))
	}
	if a := got.APIs[1]; a.Thread != 3 {
		t.Errorf("api thread = %d, want 3", a.Thread)
	}
}

func TestMarkerUnknownTimestamp(t *testing.T) {
	b := journaltest.New().Kernel(2, 3, 1)
	for _, at := range []int{0, 1} {
		b.Append(wire.KindActivity, map[string]any{"kind": "activity", "subtype": "marker", "valid": true, "name": "endpoint", "marker_flags": MarkerInstantaneous, "process_id": 7, "thread_id": 2, "start": at, "end": at})
	}
	records, err := Decode(map[string][]byte{"capture.journal": b.Finalize()}, MaxRecords, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	if len(records.Markers) != 1 || records.Markers[0].Time != 1 {
		t.Fatalf("unknown CUPTI timestamp retained: %+v", records.Markers)
	}
}
