package main

import (
	"encoding/hex"
	"encoding/json"
	"path/filepath"
	"testing"

	"github.com/tmc/lib/nvidia/internal/cudainject/wire"
)

func TestCapturePlanJournalRetainsKindLossInventory(t *testing.T) {
	const session = "00112233445566778899aabbccddeeff"
	dir := t.TempDir()
	var sid [16]byte
	raw, err := hex.DecodeString(session)
	if err != nil {
		t.Fatal(err)
	}
	copy(sid[:], raw)
	writeHandshake(t, dir, session, 41, 2)
	kinds := map[string]map[string]uint64{}
	for _, name := range []string{"activity", "api", "external", "failure", "host", "module", "pcsample", "gemm"} {
		kinds[name] = map[string]uint64{"offered": 0, "retained": 0, "dropped": 0}
	}
	kinds["activity"] = map[string]uint64{"offered": 2, "retained": 1, "dropped": 1}
	kinds["api"] = map[string]uint64{"offered": 2, "retained": 1, "dropped": 1}
	finalized, err := json.Marshal(map[string]any{"state": "collected", "dropped": 2, "queue_dropped": 2, "offered": 4, "retained": 2, "by_kind": kinds})
	if err != nil {
		t.Fatal(err)
	}
	appendRecords(t, filepath.Join(dir, "capture.journal"), sid, 41, 2, []wire.Record{
		{Kind: wire.KindBootstrap, Payload: []byte(`{"kind":"bootstrap"}`)},
		{Kind: wire.KindCapability, Payload: []byte(`{"kind":"capability"}`)},
		{Kind: wire.KindActivity, Payload: []byte(`{"kind":"activity","subtype":"kernel","valid":true,"start":10,"end":20}`)},
		{Kind: wire.KindAPI, Payload: []byte(`{"kind":"api","domain":"runtime","callback_id":211,"start":10,"end":20,"process_id":41,"thread_id":7,"correlation":1}`)},
		{Kind: wire.KindFinalized, Payload: finalized},
	})
	got := observeCaptureSession(captureResult{TargetPID: 41}, dir, session, observeOptions{injectionRequested: true})
	if got.RecordLoss == nil {
		t.Fatal("native inventory discarded")
	}
	offered, dropped, known, err := got.RecordLoss.activityLoss(got.ActivityRecords)
	if err != nil || !known || offered != 2 || dropped != 1 {
		t.Fatalf("native activity admission=%d/%d known=%v err=%v", offered, dropped, known, err)
	}
}

func TestCapturePlanActivityLossUsesNativeKindInventory(t *testing.T) {
	decode := func() captureLoss {
		t.Helper()
		var kinds captureKindLosses
		err := json.Unmarshal([]byte(`{
			"activity":{"offered":3,"retained":2,"dropped":1},
			"api":{"offered":97,"retained":87,"dropped":10},
			"external":{"offered":0,"retained":0,"dropped":0},
			"failure":{"offered":0,"retained":0,"dropped":0},
			"host":{"offered":0,"retained":0,"dropped":0},
			"module":{"offered":0,"retained":0,"dropped":0},
			"pcsample":{"offered":0,"retained":0,"dropped":0},
			"gemm":{"offered":0,"retained":0,"dropped":0}}`), &kinds)
		if err != nil {
			t.Fatal(err)
		}
		return captureLoss{byKind: &kinds, Offered: 100, Retained: 89, Queue: 11}
	}
	for _, tc := range []struct {
		name   string
		change func(*captureLoss)
		known  bool
		bad    bool
	}{
		{"separate populations", func(*captureLoss) {}, true, false},
		{"missing inventory", func(l *captureLoss) { l.byKind = nil }, false, false},
		{"missing zero field", func(l *captureLoss) { c := (*l.byKind)["host"]; c.Dropped = nil; (*l.byKind)["host"] = c }, false, false},
		{"native unclassified loss", func(l *captureLoss) { l.Native = 1 }, false, false},
		{"forged activity retained", func(l *captureLoss) { *(*l.byKind)["activity"].Retained = 3 }, false, true},
		{"coherent forged decoded count", func(l *captureLoss) {
			*(*l.byKind)["activity"].Offered = 4
			*(*l.byKind)["activity"].Retained = 3
			l.Offered, l.Retained = 101, 90
		}, false, true},
		{"coherent wrapped totals", func(l *captureLoss) {
			*(*l.byKind)["api"].Offered = ^uint64(0)
			*(*l.byKind)["api"].Retained = ^uint64(0)
			*(*l.byKind)["api"].Dropped = 0
			l.Offered, l.Retained, l.Queue = 2, 1, 1
		}, false, true},
		{"aggregate mismatch", func(l *captureLoss) { l.Queue++ }, false, true},
		{"unknown kind", func(l *captureLoss) { (*l.byKind)["other"] = captureKindLoss{} }, false, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			l := decode()
			tc.change(&l)
			offered, dropped, known, err := l.activityLoss(2)
			if known != tc.known || (err != nil) != tc.bad {
				t.Fatalf("known=%v err=%v", known, err)
			}
			if known && (offered != 3 || dropped != 1) {
				t.Fatalf("used global population: offered=%d dropped=%d", offered, dropped)
			}
		})
	}
}
