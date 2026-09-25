package main

import (
	"bytes"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/internal/cudainject/wire"
)

// traceActivity appends one activity with an explicit stream and name, which
// the gaps fixtures do not carry because a duty cycle does not draw them.
func (b *journalBuilder) traceActivity(subtype, name string, device, stream uint32, start, end uint64, correlation uint32) *journalBuilder {
	p := map[string]any{
		"kind": "activity", "subtype": subtype, "valid": true, "name": name,
		"start": start, "end": end, "device": device, "stream": stream, "correlation": correlation,
	}
	switch subtype {
	case "sync":
		p["sync_type"] = 3
		p["cuda_event_id"] = 3
	case "graph_trace":
		p["graph_id"] = 5
	}
	b.append(wire.KindActivity, p)
	return b
}

// traceMarker appends one NVTX marker record. Markers are appended one at a
// time, never as a pair, because that is how CUPTI delivers them: the range is
// something a consumer reconstructs, not something the journal holds.
func (b *journalBuilder) traceMarker(name, domain string, id, flags, pid, tid uint32, timestamp uint64) *journalBuilder {
	b.append(wire.KindActivity, map[string]any{
		"kind": "activity", "subtype": "marker", "valid": true, "name": name,
		"marker_domain": domain, "marker_id": id, "marker_flags": flags,
		"process_id": pid, "thread_id": tid,
		"start": timestamp, "end": timestamp,
	})
	return b
}

func decodeFixture(t *testing.T, data []byte) []traceRecord {
	t.Helper()
	records, _ := decodeFixtureNVTX(t, data)
	return records
}

func decodeFixtureNVTX(t *testing.T, data []byte) ([]traceRecord, nvtxSummary) {
	t.Helper()
	records, nvtx, _ := decodeFixtureExternal(t, data)
	return records, nvtx
}

func decodeFixtureExternal(t *testing.T, data []byte) ([]traceRecord, nvtxSummary, externalSummary) {
	t.Helper()
	records, nvtx, ext, err := decodeTraceJournals(map[string][]byte{"capture.journal": data}, controllerMaxJournalRecords, 1<<20)
	if err != nil {
		t.Fatalf("decodeTraceJournals: %v", err)
	}
	return records, nvtx, ext
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
	b.append(wire.KindAPI, map[string]any{
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
				b.append(wire.KindActivity, map[string]any{"kind": "activity", "dropped": 3})
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
				b.append(wire.KindFinalized, map[string]any{"kind": "finalized", "state": "collected", "dropped": 2})
				return append([]byte(nil), b.buf.Bytes()...)
			},
			want: "dropped",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, _, _, err := decodeTraceJournals(map[string][]byte{"capture.journal": tt.data()}, controllerMaxJournalRecords, 1<<20)
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

// TestExportPairsNVTXMarkersIntoRanges is the property the NVTX tier exists
// for. CUPTI reports a range as two records sharing an id, each with one
// timestamp, so the duration is a property of the pair. A consumer that drew
// each record where it found it would draw two zero-length ticks and no range
// at all.
func TestExportPairsNVTXMarkersIntoRanges(t *testing.T) {
	b := newJournalBuilder()
	b.traceActivity("kernel", "k", 0, 1, 5000, 6000, 1)
	b.traceMarker("prefill", "ds4", 1, markerFlagStart, 100, 200, 4000)
	b.traceMarker("prefill", "ds4", 1, markerFlagEnd, 100, 200, 7000)
	records, nvtx := decodeFixtureNVTX(t, b.finalize())

	if nvtx.ranges != 1 || nvtx.unpaired != 0 {
		t.Fatalf("nvtx = %+v, want exactly one paired range", nvtx)
	}
	var found int
	for _, r := range records {
		if r.kind != trackNVTX {
			continue
		}
		found++
		if r.rec.Start != 4000 || r.rec.End != 7000 {
			t.Errorf("range spans %d..%d, want 4000..7000 from the two markers", r.rec.Start, r.rec.End)
		}
		if r.rec.Name != "ds4/prefill" {
			t.Errorf("range name = %q, want the domain-qualified name", r.rec.Name)
		}
		if r.group != [2]uint32{100, 200} {
			t.Errorf("range group = %v, want the marker's pid/thread", r.group)
		}
	}
	if found != 1 {
		t.Fatalf("got %d NVTX slices, want 1", found)
	}
}

// TestExportKeepsNVTXOffTheHostAPITrack covers the namespace collision. An
// NVTX range and a runtime API call carry the same pid and thread, so one key
// namespace would merge them onto a single track -- and because a Perfetto
// track is a stack, a range that merely overlaps a call would be drawn as its
// parent.
//
// The API call is disjoint from the range, and earlier than it, on purpose.
// Overlapping records are split onto numbered lanes, which would separate them
// whether or not the namespaces collided; the test would then pass on the
// collapsed key, measuring lane assignment rather than namespacing. Earlier
// and disjoint is the one arrangement in which a collapsed key puts both on
// one lane, so the two records share a track uuid if and only if the
// namespaces merged.
func TestExportKeepsNVTXOffTheHostAPITrack(t *testing.T) {
	b := newJournalBuilder()
	b.traceActivity("kernel", "k", 0, 1, 5000, 6000, 1)
	b.append(wire.KindAPI, map[string]any{
		"kind": "api", "domain": "runtime", "start": 1000, "end": 1100,
		"process_id": 100, "thread_id": 200, "correlation": 1,
	})
	b.traceMarker("prefill", "ds4", 1, markerFlagStart, 100, 200, 4000)
	b.traceMarker("prefill", "ds4", 1, markerFlagEnd, 100, 200, 7000)
	records := decodeFixture(t, b.finalize())

	uuids, names := assignTracks(records)
	var nvtxUUID, hostUUID uint64
	for i, r := range records {
		switch r.kind {
		case trackNVTX:
			nvtxUUID = uuids[i]
		case trackHost:
			hostUUID = uuids[i]
		}
	}
	if nvtxUUID == 0 || hostUUID == 0 {
		t.Fatalf("fixture did not produce both an NVTX and a host record (nvtx=%d host=%d)", nvtxUUID, hostUUID)
	}
	if nvtxUUID == hostUUID {
		t.Errorf("NVTX and host API share track %d of %v: pid 100 thread 200 collided across namespaces", nvtxUUID, names)
	}
	var sawNVTXName bool
	for _, n := range names {
		if n == "nvtx pid 100 thread 200" {
			sawNVTXName = true
		}
	}
	if !sawNVTXName {
		t.Errorf("track names = %v, want one naming the NVTX thread", names)
	}
}

// TestExportCountsUnpairedNVTXMarkers covers the loss that would otherwise be
// invisible. A START with no END is a range still open when the capture ended;
// drawing it to any chosen end states a duration nobody measured, so it is
// drawn as nothing -- which is only honest if the count says so.
func TestExportCountsUnpairedNVTXMarkers(t *testing.T) {
	b := newJournalBuilder()
	b.traceActivity("kernel", "k", 0, 1, 5000, 6000, 1)
	b.traceMarker("open", "ds4", 1, markerFlagStart, 100, 200, 4000)
	b.traceMarker("closed-first", "ds4", 2, markerFlagEnd, 100, 200, 4500)
	b.traceMarker("point", "ds4", 3, markerFlagInstantaneous, 100, 200, 4800)
	records, nvtx := decodeFixtureNVTX(t, b.finalize())

	if nvtx.ranges != 0 {
		t.Errorf("nvtx.ranges = %d, want 0: neither half had a partner", nvtx.ranges)
	}
	if nvtx.unpaired != 2 {
		t.Errorf("nvtx.unpaired = %d, want 2 (one orphan START, one orphan END)", nvtx.unpaired)
	}
	if nvtx.points != 1 {
		t.Errorf("nvtx.points = %d, want 1", nvtx.points)
	}
	var nvtxRecords int
	for _, r := range records {
		if r.kind == trackNVTX {
			nvtxRecords++
		}
	}
	if nvtxRecords != 1 {
		t.Fatalf("drew %d NVTX slices, want 1: only the instantaneous marker is drawable", nvtxRecords)
	}
}

// TestExportNestsNVTXRangesByInnermost covers the LIFO rule. NVTX ranges nest,
// and a capture that reuses one id for an inner and an outer range would, under
// first-match pairing, close the outer range at the inner one's end and leave
// the outer end to pair with nothing.
func TestExportNestsNVTXRangesByInnermost(t *testing.T) {
	b := newJournalBuilder()
	b.traceActivity("kernel", "k", 0, 1, 5000, 6000, 1)
	b.traceMarker("outer", "ds4", 1, markerFlagStart, 100, 200, 1000)
	b.traceMarker("inner", "ds4", 1, markerFlagStart, 100, 200, 2000)
	b.traceMarker("inner", "ds4", 1, markerFlagEnd, 100, 200, 3000)
	b.traceMarker("outer", "ds4", 1, markerFlagEnd, 100, 200, 9000)
	records, nvtx := decodeFixtureNVTX(t, b.finalize())

	if nvtx.ranges != 2 || nvtx.unpaired != 0 {
		t.Fatalf("nvtx = %+v, want two paired ranges", nvtx)
	}
	spans := map[string][2]uint64{}
	for _, r := range records {
		if r.kind == trackNVTX {
			spans[r.rec.Name] = [2]uint64{r.rec.Start, r.rec.End}
		}
	}
	if got := spans["ds4/inner"]; got != [2]uint64{2000, 3000} {
		t.Errorf("inner range spans %v, want 2000..3000", got)
	}
	if got := spans["ds4/outer"]; got != [2]uint64{1000, 9000} {
		t.Errorf("outer range spans %v, want 1000..9000: the END at 3000 closed the innermost open range, not this one", got)
	}
}

// TestExportKeepsCaptureRangeOutOfTheNVTXPairing is the gate for a mistake the
// rest of this suite cannot see. A capture-range edge is a producer record
// that shares the activity stream with NVTX markers, and the natural way to
// filter it -- adding its subtype to the marker branch -- puts it into the
// slice that feeds pairMarkers instead of dropping it. Nothing else here
// notices: the edge carries no marker id, so it pairs with nothing, and the
// only visible symptom is the unpaired count, which no other test reads on a
// fixture that also contains range edges.
//
// The fixture pairs one real NVTX range alongside the edges. Without it the
// assertions would hold on an exporter that dropped NVTX entirely.
func TestExportKeepsCaptureRangeOutOfTheNVTXPairing(t *testing.T) {
	b := newJournalBuilder()
	b.traceActivity("kernel", "k", 0, 7, 5_000, 6_000, 1)
	b.traceMarker("prefill", "ds4", 1, markerFlagStart, 100, 200, 4_000)
	b.traceMarker("prefill", "ds4", 1, markerFlagEnd, 100, 200, 7_000)
	for _, e := range []struct {
		edge string
		ts   uint64
	}{{"start", 3_000}, {"stop", 8_000}} {
		b.append(wire.KindActivity, map[string]any{
			"kind": "activity", "subtype": "capture_range", "valid": true,
			"edge": e.edge, "api": "cudaProfilerStart", "index": 0,
			"toggle_failures": 0, "start": e.ts, "end": e.ts,
		})
	}

	records, nvtx := decodeFixtureNVTX(t, b.finalize())
	if nvtx.ranges != 1 {
		t.Errorf("nvtx ranges = %d, want 1: the one real marker pair", nvtx.ranges)
	}
	if nvtx.unpaired != 0 || nvtx.points != 0 || nvtx.unknown != 0 {
		t.Errorf("capture-range edges reached the NVTX pairing: unpaired=%d points=%d unknown=%d", nvtx.unpaired, nvtx.points, nvtx.unknown)
	}
	for _, r := range records {
		if got := r.rec.Name; got == "cudaProfilerStart" || strings.Contains(got, "capture_range") {
			t.Errorf("a capture-range edge became a trace slice named %q", got)
		}
	}
	// One kernel plus one NVTX range, and nothing else.
	if len(records) != 2 {
		t.Errorf("got %d trace records, want 2 (one kernel, one NVTX range)", len(records))
	}
}

// TestExportKeepsMemsetsOutOfTheKernelKind covers the arm in activityRecord
// that claims the memset subtype.
//
// The switch there has a permissive default -- the kernel arm -- so a subtype
// it does not name is not rejected, it is relabelled. Before the memset case
// existed, a memset would have been exported as a CONCURRENT_KERNEL carrying a
// zero grid and block, and the export would have been complete, well formed,
// and wrong in a way no count would show. The assertion is therefore on the
// kind, not on the record's presence.
func TestExportKeepsMemsetsOutOfTheKernelKind(t *testing.T) {
	b := newJournalBuilder()
	b.traceActivity("kernel", "k", 0, 7, 1_000, 2_000, 1)
	b.traceActivity("memset", "", 0, 7, 3_000, 4_000, 2)
	records := decodeFixture(t, b.finalize())

	var kinds []cupti.ActivityKind
	for _, r := range records {
		if r.kind != trackDevice {
			continue
		}
		kinds = append(kinds, r.rec.Kind)
	}
	if len(kinds) != 2 {
		t.Fatalf("device records = %d (%v), want 2: the memset never reached the trace", len(kinds), kinds)
	}
	var memsets, kernels int
	for _, k := range kinds {
		switch k {
		case cupti.ActivityKindMemset:
			memsets++
		case cupti.ActivityKindConcurrentKernel:
			kernels++
		}
	}
	if memsets != 1 || kernels != 1 {
		t.Errorf("kinds = %v, want one MEMSET and one CONCURRENT_KERNEL: the memset was exported under another activity kind", kinds)
	}
}

// traceExternal appends one external-correlation record: the producer stating
// that externalID was the framework scope on the stack while correlation was
// the live native ID.
func (b *journalBuilder) traceExternal(kind uint32, externalID uint64, correlation uint32) *journalBuilder {
	b.append(wire.KindExternal, map[string]any{
		"kind": "external_correlation", "external_kind": kind, "external_id": externalID, "correlation": correlation,
	})
	return b
}

// TestExportReportsExternalCoverage covers the export half of the external
// tier. Before this the frame kind had no case at all in decodeTraceJournal:
// the records were on disk in every capture, the join that reads them was
// written and tested, and the two never met.
//
// The second kernel is the control. Its correlation is one nothing pushed a
// scope for, so it must lower coverage rather than inherit the first kernel's.
func TestExportReportsExternalCoverage(t *testing.T) {
	b := newJournalBuilder()
	b.traceExternal(3, 770, 1)
	b.traceActivity("kernel", "attributed", 0, 7, 1_000, 3_000, 1)
	b.traceActivity("kernel", "unattributed", 0, 7, 4_000, 5_000, 2)
	_, _, ext := decodeFixtureExternal(t, b.finalize())

	if ext.records != 2 || ext.matched != 1 {
		t.Errorf("coverage = %d of %d device record(s), want 1 of 2", ext.matched, ext.records)
	}
	if ext.kind != cupti.ExternalCorrelationKindCustom0 {
		t.Errorf("reported namespace = %v, want the kind the producer recorded", ext.kind)
	}
	// 2000ns of 3000ns: coverage by time is not coverage by count.
	if ext.matchedNanos != 2_000 || ext.duration != 3_000 {
		t.Errorf("coverage by time = %d of %d ns, want 2000 of 3000", ext.matchedNanos, ext.duration)
	}
	if ext.ambiguous != 0 {
		t.Errorf("ambiguous = %d, want 0", ext.ambiguous)
	}
	// Independence from the test below: one namespace, so no omission note.
	for _, l := range externalSummaryLines(ext) {
		if strings.Contains(l, "namespaces present") {
			t.Errorf("single-namespace capture reports the omission note %q", l)
		}
	}
}

// TestExportStatesTheNamespaceItDidNotJoin is the export half of the omission.
// Coverage that names the namespace it used and stays silent about the one it
// discarded reports a collection gap as an absence, which is the shape this
// tier exists to remove.
func TestExportStatesTheNamespaceItDidNotJoin(t *testing.T) {
	b := newJournalBuilder()
	b.traceExternal(3, 770, 1)
	b.traceExternal(4, 880, 1)
	b.traceExternal(4, 881, 2)
	b.traceActivity("kernel", "attributed", 0, 7, 1_000, 3_000, 1)
	b.traceActivity("kernel", "other_namespace", 0, 7, 4_000, 5_000, 2)
	_, _, ext := decodeFixtureExternal(t, b.finalize())

	if ext.kind != cupti.ExternalCorrelationKindCustom0 {
		t.Errorf("joined namespace = %v, want Custom0, the lowest present", ext.kind)
	}
	// The assertion is on the lines a user reads, not on note() alone: a note
	// that is computed and never printed is worth nothing.
	want := "external: 2 external namespaces present (Custom0, Custom1); only Custom0 was joined, 2 record(s) in Custom1 left unattributed"
	lines := externalSummaryLines(ext)
	found := false
	for _, l := range lines {
		if l == want {
			found = true
		}
	}
	if !found {
		t.Errorf("reported lines = %q, want one reading %q", lines, want)
	}
	// The second kernel has a scope, in the namespace that was not joined. It
	// still counts as unattributed -- the note is what keeps that honest.
	if ext.matched != 1 || ext.records != 2 {
		t.Errorf("coverage = %d of %d, want 1 of 2", ext.matched, ext.records)
	}
}

// TestExportWithoutExternalRecordsReportsNothing is the negative control: a
// capture whose target pushed no scopes must report no coverage rather than
// zero-of-zero or a namespace it invented.
func TestExportWithoutExternalRecordsReportsNothing(t *testing.T) {
	b := newJournalBuilder()
	b.traceActivity("kernel", "plain", 0, 7, 1_000, 3_000, 1)
	if _, _, ext := decodeFixtureExternal(t, b.finalize()); ext != (externalSummary{}) {
		t.Errorf("coverage = %+v, want the zero summary", ext)
	}
}

// TestExportKeepsTheIdleExplainingTiersOutOfTheKernelKind covers the three new
// arms in activityRecord.
//
// The switch there has a permissive default -- the kernel arm -- so a subtype
// it does not name is not rejected, it is relabelled. Before these arms
// existed, a CUPTI overhead span, a graph replay envelope and a
// cudaStreamSynchronize would each have been exported as a CONCURRENT_KERNEL
// carrying a zero grid and block. The trace would open, render, and show three
// kernels the program never launched, and no count anywhere would be wrong.
// The assertion is therefore on the kind of each record, not on their number.
func TestExportKeepsTheIdleExplainingTiersOutOfTheKernelKind(t *testing.T) {
	b := newJournalBuilder()
	b.traceActivity("kernel", "k", 0, 7, 1_000, 2_000, 1)
	b.traceActivity("graph_trace", "", 0, 7, 3_000, 4_000, 2)
	b.traceActivity("sync", "", 0, 7, 5_000, 6_000, 3)
	b.append(wire.KindActivity, map[string]any{
		"kind": "activity", "subtype": "overhead", "valid": true,
		"start": 7_000, "end": 8_000, "overhead_kind": 1 << 16, "object_kind": 2,
		"process_id": 4242, "thread_id": 99,
	})
	records := decodeFixture(t, b.finalize())
	if len(records) != 4 {
		t.Fatalf("records = %d, want 4", len(records))
	}

	byKind := map[cupti.ActivityKind]traceRecord{}
	for _, r := range records {
		byKind[r.rec.Kind] = r
	}
	for _, want := range []cupti.ActivityKind{cupti.ActivityKindConcurrentKernel, cupti.ActivityKindOverhead, cupti.ActivityKindGraphTrace, cupti.ActivityKindSynchronization} {
		if _, ok := byKind[want]; !ok {
			var got []cupti.ActivityKind
			for _, r := range records {
				got = append(got, r.rec.Kind)
			}
			t.Fatalf("no %v record in the trace; kinds = %v: a subtype fell through to the permissive default and was exported as a kernel", want, got)
		}
	}

	// CUPTI's own cost is paid on a host thread, and the record says so. A
	// device track is a claim the device ran it.
	overhead := byKind[cupti.ActivityKindOverhead]
	if overhead.kind != trackHost {
		t.Errorf("overhead is on the device track: the trace would draw CUPTI's own cost beside the kernels the device actually ran")
	}
	if overhead.group != [2]uint32{4242, 99} {
		t.Errorf("overhead host track = %v, want {4242, 99} from its object identity", overhead.group)
	}
	// A record whose name is the numeric enum, or empty, is one a reader
	// cannot act on: "an overhead happened" is not a finding.
	if overhead.rec.Name != cupti.ActivityOverheadCuptiBufferFlush.String() {
		t.Errorf("overhead name = %q, want %q", overhead.rec.Name, cupti.ActivityOverheadCuptiBufferFlush.String())
	}

	sync := byKind[cupti.ActivityKindSynchronization]
	if sync.rec.Synchronization != cupti.ActivitySynchronizationTypeStreamSynchronize {
		t.Errorf("sync type = %v, want stream synchronize: the span survived and the reason for it did not", sync.rec.Synchronization)
	}
	if sync.rec.Name != cupti.ActivitySynchronizationTypeStreamSynchronize.String() {
		t.Errorf("sync name = %q, want %q", sync.rec.Name, cupti.ActivitySynchronizationTypeStreamSynchronize.String())
	}
	if sync.rec.CudaEventID != 3 {
		t.Errorf("sync cuda event id = %d, want 3", sync.rec.CudaEventID)
	}
}
