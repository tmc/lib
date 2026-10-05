package cudarepro

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaschedule"
)

func mismatchRecord(index uint64) cudaschedule.RunRecord {
	digest := strings.Repeat("a", 64)
	return cudaschedule.RunRecord{
		Exited: true, Stdout: "MISMATCH\n",
		PTXDigest: digest, ExpectedPTXDigest: digest,
		PTXDigestSource: "runner_receipt",
		Mismatch:        &cudaschedule.MismatchIdentity{Index: index, Policy: "exact-uint64"},
	}
}

func TestIdentityRejectsDrift(t *testing.T) {
	i, err := Identify(mismatchRecord(0))
	if err != nil {
		t.Fatal(err)
	}
	for _, tt := range []struct {
		name      string
		change    func(*cudaschedule.RunRecord)
		want      bool
		wantError bool
	}{
		{"same", func(*cudaschedule.RunRecord) {}, true, false},
		{"element", func(r *cudaschedule.RunRecord) { r.Mismatch.Index = 1 }, false, false},
		{"crash", func(r *cudaschedule.RunRecord) { r.ExitCode = 1 }, false, false},
		{"pass", func(r *cudaschedule.RunRecord) { r.Stdout = "PASS" }, false, false},
		{"missing signature", func(r *cudaschedule.RunRecord) { r.Mismatch = nil }, false, true},
		{"different policy", func(r *cudaschedule.RunRecord) { r.Mismatch.Policy = "fp32" }, false, true},
		{"missing load receipt", func(r *cudaschedule.RunRecord) { r.PTXDigest = "" }, false, true},
		{"wrong loaded image", func(r *cudaschedule.RunRecord) { r.PTXDigest = strings.Repeat("b", 64) }, false, true},
		{"unexecuted", func(r *cudaschedule.RunRecord) { r.HarnessErr = "not run" }, false, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			r := mismatchRecord(0)
			tt.change(&r)
			got, err := i.Matches(r)
			if got != tt.want || (err != nil) != tt.wantError {
				t.Fatalf("Matches = %v, %v; want %v, error %v", got, err, tt.want, tt.wantError)
			}
		})
	}
}

func TestIdentifyRefusesMissingEvidence(t *testing.T) {
	r := mismatchRecord(0)
	r.Mismatch = nil
	if _, err := Identify(r); err == nil {
		t.Fatal("accepted legacy mismatch without identity")
	}
	if _, err := Identify(cudaschedule.RunRecord{}); err == nil {
		t.Fatal("accepted absent execution")
	}
	if _, err := (Identity{}).Matches(mismatchRecord(0)); err == nil {
		t.Fatal("accepted zero identity")
	}
}
