package main

import (
	"bytes"
	"encoding/hex"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/internal/cudainject/wire"
)

// A missing frame is independent of the producer's reported drop counter.
// Never certify the remaining intervals as a complete device timeline.
func TestCaptureMissingSequence(t *testing.T) {
	for _, missing := range []bool{false, true} {
		name := "complete"
		if missing {
			name = "missing kernel"
		}
		t.Run(name, func(t *testing.T) {
			var session [16]byte
			session[0] = 1
			records := []wire.Record{
				{Kind: wire.KindBootstrap, Payload: []byte(`{"kind":"bootstrap"}`)},
				{Kind: wire.KindCapability, Payload: []byte(`{"kind":"capability"}`)},
				{Kind: wire.KindActivity, Payload: []byte(`{"kind":"activity","subtype":"kernel","valid":true,"start":10,"end":20}`)},
				{Kind: wire.KindFinalized, Payload: []byte(`{"state":"collected","activity":1,"dropped":0}`)},
			}
			var journal bytes.Buffer
			for i, r := range records {
				if missing && i == 2 {
					continue
				}
				r.Session = session
				r.Process = 1
				r.Generation = 1
				r.Sequence = uint64(i + 1)
				if err := wire.Write(&journal, r, wire.Limits{}); err != nil {
					t.Fatal(err)
				}
			}
			path := filepath.Join(t.TempDir(), "capture.journal")
			if err := os.WriteFile(path, journal.Bytes(), 0600); err != nil {
				t.Fatal(err)
			}
			_, _, _, _, _, _, err := observeJournal(path, hex.EncodeToString(session[:]), 1, 0)
			if missing {
				if err == nil || !strings.Contains(err.Error(), "missing sequence") {
					t.Fatalf("missing frame: %v", err)
				}
			} else if err != nil {
				t.Fatal(err)
			}
		})
	}
}
