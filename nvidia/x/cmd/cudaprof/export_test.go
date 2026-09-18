package main

import (
	"bytes"
	"strings"
	"testing"
)

// traceActivity appends one activity with an explicit stream and name, which
// the gaps fixtures do not carry because a duty cycle does not draw them.
func (b *journalBuilder) traceActivity(subtype, name string, device, stream uint32, start, end uint64, correlation uint32) *journalBuilder {
	b.append(recordActivity, map[string]any{
		"kind": "activity", "subtype": subtype, "valid": true, "name": name,
		"start": start, "end": end, "device": device, "stream": stream, "correlation": correlation,
	})
	return b
}

func decodeFixture(t *testing.T, data []byte) []traceRecord {
	t.Helper()
	records, err := decodeTraceJournals(map[string][]byte{"capture.journal": data}, controllerMaxJournalRecords, 1<<20)
	if err != nil {
		t.Fatalf("decodeTraceJournals: %v", err)
	}
	return records
}

// TestExportLanesSplitOverlappingActivities is the property the whole track
// assignment exists for. A Perfetto track is a stack, so two overlapping
// slices on one track are drawn as one nested inside the other -- a claim that
// one activity contained the other, which CUPTI never reported. Concurrent
// kernels carrying the same stream id are exactly that case.
func TestExportLanesSplitOverlappingActivities(t *testing.T) {
	b := newJournalBuilder()
	b.traceActivity("kernel", "overlapped_a", 0, 7, 1_000, 5_000, 1)
	b.traceActivity("kernel", "overlapped_b", 0, 7, 2_000, 6_000, 2)
	b.traceActivity("kernel", "after_both", 0, 7, 9_000, 10_000, 3)
	records := decodeFixture(t, b.finalize())

	uuids, names := assignTracks(records)
	if len(records) != 3 {
		t.Fatalf("records = %d, want 3", len(records))
	}
	if uuids[0] == uuids[1] {
		t.Errorf("overlapping activities share track %d; Perfetto would draw one inside the other", uuids[0])
	}
	// The third starts after both end, so it must reuse the first lane rather
	// than open a third. Without this, every activity would get its own track
	// and the overlap assertion above would pass vacuously.
	if uuids[2] != uuids[0] {
		t.Errorf("non-overlapping activity opened track %d instead of reusing %d", uuids[2], uuids[0])
	}
	if len(names) != 2 {
		t.Fatalf("tracks = %v, want 2", names)
	}
	if names[0] != "device 0 stream 7" || names[1] != "device 0 stream 7 (overlap 1)" {
		t.Errorf("track names = %v", names)
	}
}

// TestExportGroupsHostAndDeviceSeparately pins that a stream id and a thread id
// are different namespaces. Both are small integers, so a single map keyed on
// the pair alone would silently draw a kernel and an API call on one track.
func TestExportGroupsHostAndDeviceSeparately(t *testing.T) {
	b := newJournalBuilder()
	// The two must NOT overlap in time. If they did, the lane splitter would
	// put them on separate tracks even under a shared namespace, and the test
	// would pass on the defect it is meant to catch -- which it did until this
	// fixture was fixed.
	b.traceActivity("kernel", "k", 0, 1, 5_000, 6_000, 1)
	b.append(recordAPI, map[string]any{
		"kind": "api", "domain": "driver", "start": 7_000, "end": 8_000,
		"process_id": 0, "thread_id": 1, "correlation": 1,
	})
	records := decodeFixture(t, b.finalize())

	uuids, names := assignTracks(records)
	if len(records) != 2 {
		t.Fatalf("records = %d, want 2", len(records))
	}
	if uuids[0] == uuids[1] {
		t.Fatalf("device 0 stream 1 and pid 0 thread 1 share track %d", uuids[0])
	}
	joined := strings.Join(names, " | ")
	if !strings.Contains(joined, "device 0 stream 1") || !strings.Contains(joined, "pid 0 thread 1") {
		t.Errorf("track names = %v, want one device track and one thread track", names)
	}
}

// TestExportRefusesIncompleteJournals keeps the export on the same footing as
// gaps. A dropped record leaves a span the viewer draws as device idle, which
// is a wrong answer rather than a partial one, and nothing in the rendered
// trace says a record was missing.
func TestExportRefusesIncompleteJournals(t *testing.T) {
	tests := []struct {
		name string
		data func() []byte
		want string
	}{
		{
			name: "dropped activity records",
			data: func() []byte {
				b := newJournalBuilder()
				b.traceActivity("kernel", "k", 0, 1, 1_000, 2_000, 1)
				b.append(recordActivity, map[string]any{"kind": "activity", "dropped": 3})
				return b.finalize()
			},
			want: "dropped",
		},
		{
			name: "never finalized",
			data: func() []byte {
				b := newJournalBuilder()
				b.traceActivity("kernel", "k", 0, 1, 1_000, 2_000, 1)
				return append([]byte(nil), b.buf.Bytes()...)
			},
			want: "never finalized",
		},
		{
			name: "finalized with drops",
			data: func() []byte {
				b := newJournalBuilder()
				b.traceActivity("kernel", "k", 0, 1, 1_000, 2_000, 1)
				b.append(recordFinalized, map[string]any{"kind": "finalized", "state": "collected", "dropped": 2})
				return append([]byte(nil), b.buf.Bytes()...)
			},
			want: "dropped",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := decodeTraceJournals(map[string][]byte{"capture.journal": tt.data()}, controllerMaxJournalRecords, 1<<20)
			if err == nil {
				t.Fatal("exported a journal that cannot draw a complete timeline")
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Errorf("error = %q, want it to name %q", err, tt.want)
			}
		})
	}
}

// TestExportWritesATrace checks the whole path end to end and that kernel
// detail survives the journal. The name is what a reader looks for first, and
// it is the one field the gaps payload never decoded.
func TestExportWritesATrace(t *testing.T) {
	b := newJournalBuilder()
	b.traceActivity("kernel", "ds4_q8_0_matvec_q8_1_batch", 0, 7, 1_000, 5_000, 1)
	b.traceActivity("memcpy", "", 0, 7, 6_000, 7_000, 2)
	records := decodeFixture(t, b.finalize())

	var buf bytes.Buffer
	tracks, err := writeTrace(&buf, records)
	if err != nil {
		t.Fatalf("writeTrace: %v", err)
	}
	if tracks != 1 {
		t.Errorf("tracks = %d, want 1: the two activities do not overlap", tracks)
	}
	if buf.Len() == 0 {
		t.Fatal("wrote an empty trace")
	}
	// Names reach Perfetto as plain strings in the packet stream, so their
	// presence is checkable without decoding protobuf. Their absence would
	// mean the export drew unnamed slices, which is what happens when the
	// payload's name field is not decoded.
	if !bytes.Contains(buf.Bytes(), []byte("ds4_q8_0_matvec_q8_1_batch")) {
		t.Error("trace does not carry the kernel name")
	}
	if !bytes.Contains(buf.Bytes(), []byte("device 0 stream 7")) {
		t.Error("trace does not carry the track name")
	}
}

func TestParseExportArgs(t *testing.T) {
	opts, err := parseExportArgs([]string{"run.pprof"})
	if err != nil {
		t.Fatal(err)
	}
	if opts.outPath != "run.pftrace" {
		t.Errorf("default out = %q, want run.pftrace", opts.outPath)
	}
	if opts, err = parseExportArgs([]string{"-o", "x.pftrace", "run.pprof"}); err != nil || opts.outPath != "x.pftrace" {
		t.Errorf("-o = %q, %v", opts.outPath, err)
	}
	if _, err := parseExportArgs([]string{"--nope", "run.pprof"}); err == nil {
		t.Error("accepted an unknown flag")
	}
	if _, err := parseExportArgs([]string{}); err == nil {
		t.Error("accepted no capture path")
	}
}
