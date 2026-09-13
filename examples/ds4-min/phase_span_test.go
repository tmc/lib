package main

import (
	"path/filepath"
	"testing"
)

// TestPhaseSpansChainBetweenMarks holds the property the fine sidecar
// depends on: a phase is the interval between two marks, so each mark
// closes the previous span and opens the next, and the group boundary
// closes the last one. A phase left open would be counted as begun and
// never ended, which is exactly what the streaming accounting reports.
func TestPhaseSpansChainBetweenMarks(t *testing.T) {
	path := filepath.Join(t.TempDir(), "spans.jsonl")
	if err := withStreamRecorder(t, levelOp, path, func() {
		markPhaseSpan("dense-front")
		markPhaseSpan("attention")
		markPhaseSpan("compression")
		closePhaseSpan()
	}); err != nil {
		t.Fatalf("close: %v", err)
	}
	got := readSidecar(t, path)
	want := []string{"dense-front", "attention", "compression"}
	if len(got) != len(want) {
		t.Fatalf("wrote %d records, want %d", len(got), len(want))
	}
	for i, w := range want {
		if got[i].Name != w {
			t.Errorf("record %d name = %q, want %q", i, got[i].Name, w)
		}
	}
	acc := readAccounting(t, path)
	if !acc.Complete || acc.Unended != 0 {
		t.Errorf("accounting complete=%v unended=%d, want true/0", acc.Complete, acc.Unended)
	}
	if acc.OpenPeak != 1 {
		t.Errorf("open_peak = %d, want 1: phases chain, they do not nest", acc.OpenPeak)
	}
}

// TestPhaseSpansReportAnUnclosedGroup is the negative: drop the group
// close and the receipt must say a span never ended rather than quietly
// write one fewer record.
func TestPhaseSpansReportAnUnclosedGroup(t *testing.T) {
	path := filepath.Join(t.TempDir(), "spans.jsonl")
	if err := withStreamRecorder(t, levelOp, path, func() {
		markPhaseSpan("dense-front")
		markPhaseSpan("attention")
		// No closePhaseSpan: "attention" is still open.
		closeOpenPhaseForTest()
	}); err != nil {
		t.Fatalf("close: %v", err)
	}
	acc := readAccounting(t, path)
	if acc.Begun != 2 || acc.Ended != 1 || acc.Unended != 1 {
		t.Errorf("begun/ended/unended = %d/%d/%d, want 2/1/1", acc.Begun, acc.Ended, acc.Unended)
	}
	if acc.Complete {
		t.Error("complete = true with a phase left open, want false")
	}
}

// closeOpenPhaseForTest drops the open phase without ending it, which is
// what a run that stopped mid-group leaves behind.
func closeOpenPhaseForTest() { openPhase = tracedEnd{} }

// TestPhaseSpansCostNothingWhenOff guards the ordinary run: with no
// recorder installed the marks must not allocate a span or leave state
// behind for the next run to trip over.
func TestPhaseSpansCostNothingWhenOff(t *testing.T) {
	prevSpans, prevStream := spans, streamSpans
	spans, streamSpans = nil, nil
	defer func() { spans, streamSpans = prevSpans, prevStream }()
	markPhaseSpan("attention")
	if openPhase.done != nil {
		t.Error("a phase span opened with no recorder installed")
	}
	closePhaseSpan()
}
