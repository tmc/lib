package kernelagent

import (
	"context"
	"encoding/json"
	"os"
	"reflect"
	"strings"
	"sync"
	"testing"
)

func TestArchiveResumeAcrossEpochs(t *testing.T) {
	dir := t.TempDir()
	f, err := newFixture(dir)
	if err != nil {
		t.Fatal(err)
	}
	// A complete durable begin record represents an interrupted epoch. No
	// candidate/executable identity is guessed from the last proposed edit.
	id, err := f.archive.append(Record{Epoch: "interrupted", Kind: "attempt", Inputs: bitEvidence(f.cfg.Screen.Inputs), WorkloadKey: f.cfg.Screen.Workload.Key()})
	if err != nil {
		t.Fatal(err)
	}
	if err := f.archive.Close(); err != nil {
		t.Fatal(err)
	}
	a, err := OpenArchive(dir, 1<<20, 1000)
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	f.archive = a
	records := a.Records()
	if len(records) != 2 || records[1].Attempt != id || records[1].Outcome != "quarantined" || !records[1].FaultCauseUnknown || records[1].CandidateID != "" || !records[1].Unmeasured {
		t.Fatalf("recovery=%+v", records)
	}
	l, err := NewLoop(a, f.cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()
	result, err := l.RunEpoch(context.Background(), "next-epoch", f.state, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	result.Close()
	for i, r := range a.Records() {
		if r.Sequence != uint64(i+1) {
			t.Fatal("nonmonotonic append")
		}
	}
	if _, err := l.RunEpoch(context.Background(), "next-epoch", f.state, nil, nil); err == nil {
		t.Fatal("epoch reused")
	}
}

func TestArchiveAppendBoundsCopiesAndConcurrency(t *testing.T) {
	dir := t.TempDir()
	a, err := OpenArchive(dir, 4096, 32)
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := a.append(Record{Epoch: "concurrent", Kind: "diagnostic"}); err != nil {
				t.Error(err)
			}
		}()
	}
	wg.Wait()
	records := a.Records()
	if len(records) != 8 {
		t.Fatal("lost appends")
	}
	records[0].Epoch = "mutated"
	if a.Records()[0].Epoch == "mutated" {
		t.Fatal("archive aliases caller")
	}
	if _, err := a.append(Record{Epoch: "bound", Kind: "diagnostic", Reason: strings.Repeat("x", 4096)}); err == nil {
		t.Fatal("oversized append accepted")
	}
	if len(a.Records()) != 8 {
		t.Fatal("oversized append partially retained")
	}
	data, err := os.ReadFile(dir + "/archive.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	if len(strings.Split(strings.TrimSpace(string(data)), "\n")) != 8 {
		t.Fatal("disk and retained counts differ")
	}
}

func TestArchiveCorruptionAndTornTailRefused(t *testing.T) {
	for _, body := range []string{`{"Version":1`, "{bad}\n", `{"Version":1,"Sequence":1,"Epoch":"x"}` + "\n", `{"Version":2,"Sequence":2,"Epoch":"x"}` + "\n", `{"Version":2,"Sequence":1,"Epoch":"x","Kind":"outcome","Attempt":1}` + "\n"} {
		t.Run(body, func(t *testing.T) {
			dir := t.TempDir()
			if err := os.WriteFile(dir+"/archive.jsonl", []byte(body), 0600); err != nil {
				t.Fatal(err)
			}
			if a, err := OpenArchive(dir, 4096, 32); err == nil {
				a.Close()
				t.Fatal("corrupted archive resumed")
			}
			after, err := os.ReadFile(dir + "/archive.jsonl")
			if err != nil || string(after) != body {
				t.Fatal("corruption silently repaired")
			}
		})
	}
}

func TestArchiveOutsideSymlinkRefused(t *testing.T) {
	dir := t.TempDir()
	candidate := dir + "/candidate"
	os.Mkdir(candidate, 0700)
	outside := dir + "/outside"
	if err := os.WriteFile(outside, []byte("untouched"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink("../outside", candidate+"/archive.jsonl"); err != nil {
		t.Fatal(err)
	}
	if a, err := OpenArchive(candidate, 4096, 32); err == nil {
		a.Close()
		t.Fatal("archive followed escaping symlink")
	}
	data, _ := os.ReadFile(outside)
	if string(data) != "untouched" {
		t.Fatal("outside file modified")
	}
}

func TestArchiveRecordsOwnerIdentityAndExactInputBits(t *testing.T) {
	f, l := setup(t)
	run(t, l, f)
	for _, r := range f.archive.Records() {
		if r.Kind == "attempt" {
			if r.Workload == nil || r.Workload.DType != "float32" || r.Workload.Layout != "row_major" || len(r.Workload.Shape) != 1 || r.Workload.Shape[0] != 2 || r.Workload.InputIdentities["x"] == "" || r.Roofline == nil || r.Roofline.PeakMemoryGBps != 1 || r.Operation == nil || r.Operation.Bytes != 24 {
				t.Fatalf("binding/authority facts missing: %+v", r)
			}
			if !reflect.DeepEqual(r.Inputs, bitEvidence(f.cfg.Screen.Inputs)) && r.Phase == "screen" {
				t.Fatal("bound inputs changed")
			}
			if r.CandidateID != "" {
				t.Fatal("precompile identity invented")
			}
		}
	}
	bits := map[string][]uint32{"special": {0, 0x80000000, 0x7fc01234}}
	if _, err := f.archive.append(Record{Epoch: "special", Kind: "diagnostic", Inputs: bits}); err != nil {
		t.Fatal(err)
	}
	data, _ := json.Marshal(f.archive.Records())
	if !strings.Contains(string(data), "2147483648") {
		t.Fatal("signed zero bits lost")
	}
}
