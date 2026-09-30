package main

import (
	"encoding/hex"
	"os"
	"path/filepath"
	"testing"

	"github.com/tmc/lib/nvidia/internal/cudainject/wire"
)

func TestObserveJournalDependencyMetadata(t *testing.T) {
	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatal(err)
	}
	dir, err := os.MkdirTemp(filepath.Join(home, "tmp"), "dependency-observer-")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)
	for _, tc := range []struct {
		name, payload string
		invalid       bool
	}{
		{"exit", `{"kind":"dependency","domain":"driver","dependency_event":"event_wait","site":"exit","ordinal":2,"timestamp_domain":"monotonic","start":10,"end":10,"success":true}`, false},
		{"unknown ordinal", `{"kind":"dependency","domain":"driver","dependency_event":"event_wait","site":"exit","ordinal":0,"start":10,"end":10}`, true},
		{"unknown site", `{"kind":"dependency","domain":"driver","dependency_event":"launch","site":"unknown","ordinal":2,"start":10,"end":10}`, true},
		{"missing event", `{"kind":"dependency","domain":"driver","site":"exit","ordinal":2,"start":10,"end":10}`, true},
		{"reversed interval", `{"kind":"dependency","domain":"driver","dependency_event":"launch","site":"exit","ordinal":2,"start":10,"end":5}`, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(dir, tc.name+".journal")
			var sid [16]byte
			sid[0] = 1
			appendRecords(t, path, sid, 1234, 1, []wire.Record{{Kind: wire.KindBootstrap, Payload: []byte(`{"kind":"bootstrap"}`)}, {Kind: wire.KindCapability, Payload: []byte(`{"kind":"capability","dependencies":"collected","dependency_coverage":"driver_partial"}`)}, {Kind: wire.KindAPI, Payload: []byte(tc.payload)}, {Kind: wire.KindActivity, Payload: []byte(`{"kind":"activity","subtype":"kernel","name":"work","valid":true,"start":20,"end":30,"correlation":1}`)}, {Kind: wire.KindFinalized, Payload: []byte(`{"kind":"finalized","state":"collected","activity":1}`)}})
			_, _, _, _, apis, _, err := observeJournal(path, hex.EncodeToString(sid[:]), 1234, 0)
			if (err != nil) != tc.invalid {
				t.Fatalf("err=%v, want invalid=%t", err, tc.invalid)
			}
			if !tc.invalid && len(apis) != 0 {
				t.Fatalf("metadata projected as timing calls: %+v", apis)
			}
		})
	}
}
