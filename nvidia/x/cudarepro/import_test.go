package cudarepro

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaschedule"
)

func findingRecord() (cudaschedule.SeedResult, []byte) {
	source := []byte(".version 8.0\n.target sm_80\n.address_size 64\n.visible .entry fixture() { ret; }\n")
	digest := sha256.Sum256(source)
	run := mismatchRecord(0)
	run.PTXDigest = hex.EncodeToString(digest[:])
	run.ExpectedPTXDigest = run.PTXDigest
	second := run
	second.Mismatch = &cudaschedule.MismatchIdentity{Index: 0, Policy: "exact-uint64"}
	return cudaschedule.SeedResult{Seed: 17, Status: cudaschedule.SeedExecuted, Verdict: cudaschedule.VerdictMismatch,
		Run: &run, PristineSourceDigest: hex.EncodeToString(digest[:]),
		Report: cudaschedule.RewriteReport{Seed: 17, Decisions: []cudaschedule.InjectionDecision{}},
		Confirmation: &cudaschedule.ConfirmationSummary{Attempts: 2, Findings: 2, Confirmed: true,
			Runs: []cudaschedule.RunRecord{run, second}, Outcomes: []cudaschedule.Verdict{cudaschedule.VerdictMismatch, cudaschedule.VerdictMismatch}},
	}, source
}

func TestReadFindingEvidence(t *testing.T) {
	for _, tt := range []struct {
		name      string
		change    func(*cudaschedule.SeedResult)
		wantError bool
	}{
		{"valid", func(*cudaschedule.SeedResult) {}, false},
		{"legacy missing mismatch", func(r *cudaschedule.SeedResult) { r.Run.Mismatch = nil }, true},
		{"legacy missing runs", func(r *cudaschedule.SeedResult) { r.Confirmation.Runs = nil }, true},
		{"missing signature in confirmation", func(r *cudaschedule.SeedResult) { r.Confirmation.Runs[1].Mismatch = nil }, true},
		{"signature drift", func(r *cudaschedule.SeedResult) { r.Confirmation.Runs[1].Mismatch.Index = 1 }, true},
		{"invented count", func(r *cudaschedule.SeedResult) { r.Confirmation.Findings = 3 }, true},
		{"different initial run", func(r *cudaschedule.SeedResult) { r.Confirmation.Runs[0].Stderr = "different" }, true},
		{"other candidate", func(r *cudaschedule.SeedResult) {
			r.Confirmation.Runs[1].ExpectedPTXDigest = strings.Repeat("b", 64)
			r.Confirmation.Runs[1].PTXDigest = strings.Repeat("b", 64)
		}, true},
		{"other seed", func(r *cudaschedule.SeedResult) { r.Report.Seed++ }, true},
		{"forged loaded artifact", func(r *cudaschedule.SeedResult) {
			r.Run.PTXDigest = strings.Repeat("b", 64)
			r.Run.ExpectedPTXDigest = r.Run.PTXDigest
			r.Confirmation.Runs = []cudaschedule.RunRecord{*r.Run, *r.Run}
		}, true},
		{"forged report totals", func(r *cudaschedule.SeedResult) { r.Report.SitesInjected++ }, true},
		{"wrong source", func(r *cudaschedule.SeedResult) { r.PristineSourceDigest = strings.Repeat("b", 64) }, true},
		{"never executed", func(r *cudaschedule.SeedResult) { r.Status = cudaschedule.SeedEmptyDraw }, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			r, source := findingRecord()
			tt.change(&r)
			data, err := json.Marshal(r)
			if err != nil {
				t.Fatal(err)
			}
			_, err = ReadFinding(bytes.NewReader(data), source)
			if (err != nil) != tt.wantError {
				t.Fatalf("ReadFinding=%v; want error %v", err, tt.wantError)
			}
		})
	}
}

func TestReadFindingRejectsAmbiguousJSON(t *testing.T) {
	r, source := findingRecord()
	data, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	for _, s := range []string{
		string(data) + " {}",
		strings.Replace(string(data), `"seed":17`, `"seed":17,"seed":17`, 1),
		strings.Replace(string(data), `"seed":17`, `"seed":17,"Seed":17`, 1),
		strings.Replace(string(data), `"seed":17`, `"seed":17,"unknown":1`, 1),
		strings.Repeat(" ", (1<<20)+1),
		strings.Repeat("[", 34) + "0" + strings.Repeat("]", 34),
	} {
		if _, err := ReadFinding(strings.NewReader(s), source); err == nil {
			t.Fatal("accepted ambiguous json")
		}
	}
}

func FuzzReadFinding(f *testing.F) {
	r, source := findingRecord()
	data, _ := json.Marshal(r)
	f.Add(data)
	f.Add([]byte(`{"seed":17,"seed":18}`))
	f.Fuzz(func(t *testing.T, data []byte) { _, _ = ReadFinding(bytes.NewReader(data), source) })
}

func TestReadFindingRecordedLane59AndOneOff(t *testing.T) {
	source, err := os.ReadFile("testdata/lane59-pristine.ptx")
	if err != nil {
		t.Fatal(err)
	}
	for _, seed := range []string{"17", "39"} {
		t.Run(seed, func(t *testing.T) {
			data, err := os.ReadFile("testdata/lane59-seed-" + seed + ".json")
			if err != nil {
				t.Fatal(err)
			}
			_, err = ReadFinding(bytes.NewReader(data), source)
			if err == nil || !strings.Contains(err.Error(), "missing comparison identity") {
				t.Fatalf("legacy record must refuse missing runner signature: %v", err)
			}
		})
	}
}
