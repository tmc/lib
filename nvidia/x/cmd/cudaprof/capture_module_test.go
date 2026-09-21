package main

import (
	"encoding/hex"
	"path/filepath"
	"testing"

	"github.com/tmc/lib/nvidia/internal/cudainject/wire"
)

// moduleSession is the session every test here writes journals under.
const moduleSession = "0123456789abcdef0123456789abcdef"

func moduleSessionID(t *testing.T) [16]byte {
	t.Helper()
	raw, err := hex.DecodeString(moduleSession)
	if err != nil {
		t.Fatal(err)
	}
	var sid [16]byte
	copy(sid[:], raw)
	return sid
}

// TestModuleRecordsDecode pins the module record kind end to end through the
// journal reader. The kind is closed -- an unknown one costs the whole capture
// -- so this is also the check that a journal carrying modules still parses.
func TestModuleRecordsDecode(t *testing.T) {
	dir := t.TempDir()
	journalPath := filepath.Join(dir, "capture.journal")
	appendRecords(t, journalPath, moduleSessionID(t), 1234, 1, []wire.Record{
		{Kind: recordBootstrap, Payload: []byte(`{"kind":"bootstrap"}`)},
		{Kind: recordCapability, Payload: []byte(`{"kind":"capability"}`)},
		{Kind: recordModule, Payload: []byte(`{"kind":"module","module_id":22,"module_size":4096,"module_file":"module-0000-id22.cubin","module_stored":true,"module_reason":""}`)},
		{Kind: recordModule, Payload: []byte(`{"kind":"module","module_id":23,"module_size":99999999,"module_file":"","module_stored":false,"module_reason":"file_budget"}`)},
		{Kind: recordActivity, Payload: []byte(`{"kind":"activity","subtype":"kernel","name":"k","valid":true,"start":100,"end":200}`)},
		{Kind: recordFinalized, Payload: []byte(`{"state":"collected","activity":1}`)},
	})

	_, _, metrics, activities, _, _, err := observeJournal(journalPath, moduleSession, 1234, 0)
	if err != nil {
		t.Fatalf("a module record cost the whole capture: %v", err)
	}
	if len(activities) != 1 {
		t.Fatalf("got %d activities, want 1: module records must not enter the activity totals", len(activities))
	}
	if len(metrics.modules) != 2 {
		t.Fatalf("got %d modules, want 2", len(metrics.modules))
	}
	if got := metrics.modules[0]; got.ID != 22 || got.Size != 4096 || got.File != "module-0000-id22.cubin" || !got.Stored {
		t.Errorf("stored module = %+v", got)
	}
	// The module the producer declined to keep must survive as a record of
	// its own. Dropping it would make a module too large to store
	// indistinguishable from one the target never loaded.
	if got := metrics.modules[1]; got.ID != 23 || got.Stored || got.Reason != "file_budget" || got.Size != 99999999 {
		t.Errorf("omitted module = %+v", got)
	}
}
