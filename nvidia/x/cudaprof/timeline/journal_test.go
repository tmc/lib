package timeline

import (
	"errors"
	"io"
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
