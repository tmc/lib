package cudasanitizer

import (
	"crypto/sha256"
	"errors"
	"reflect"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudatrial"
)

func crashFixture(t *testing.T) (*Result, CompletedCrashTrace, CrashIdentity) {
	t.Helper()
	m, err := cudatrial.NewThreadEdgeMap(32, 1)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := m.Record(0, 3); err != nil {
		t.Fatal(err)
	}
	p := CrashIdentity{Kernel: "kernel", Image: sha256.Sum256([]byte("image")), Epoch: "epoch", Trial: "previous", Input: sha256.Sum256([]byte("previous input")), Ordinal: 7}
	f := p
	f.Trial, f.Input, f.Ordinal = "failure", sha256.Sum256([]byte("failing input")), 8
	r := &Result{Config: Config{Tool: Memcheck}, Outcome: OutcomeSanitizerErrors, Log: ParseLog("========= Invalid __global__ write of size 4 bytes\n=========     at kernel\n========= ERROR SUMMARY: 1 error\n")}
	return r, CompletedCrashTrace{Identity: p, Complete: true, MapSize: 32, Bitmap: m.Snapshot()}, f
}

func TestCrashSignaturePrecedingTraceAndDeterminism(t *testing.T) {
	r, p, f := crashFixture(t)
	a, err := CrashSignature(r, p, f)
	if err != nil {
		t.Fatal(err)
	}
	b, err := CrashSignature(r, p, f)
	if err != nil || !reflect.DeepEqual(a, b) {
		t.Fatalf("repeat = %+v, %v", b, err)
	}
	if a.Failure != f || a.Preceding != p.Identity || a.TraceDigest != sha256.Sum256(p.Bitmap.Bytes()) || a.Signature == [32]byte{} {
		t.Fatalf("lost association: %+v", a)
	}
	// Trial and input identities are retained independently of the grouping key.
	f.Trial, f.Input = "another failure", sha256.Sum256([]byte("another input"))
	p.Identity.Trial, p.Identity.Input = "another previous", sha256.Sum256([]byte("other previous input"))
	b, err = CrashSignature(r, p, f)
	if err != nil || b.Signature != a.Signature || b.Failure == a.Failure {
		t.Fatalf("dedup identity: %+v %v", b, err)
	}
	// Detail addresses, order, repetition and error counts do not define classes.
	r.Log.Findings = append(r.Log.Findings, r.Log.Findings[0])
	r.Log.Findings[0].Detail = []string{"different address"}
	r.Log.Errors = 2
	b, err = CrashSignature(r, p, f)
	if err != nil || b.Signature != a.Signature {
		t.Fatalf("duplicate class changed key: %v", err)
	}
	if len(a.Headlines) != 1 {
		t.Fatal(a.Headlines)
	}
}

func TestCrashSignatureRejectsInvalidEvidence(t *testing.T) {
	tests := []struct {
		name   string
		change func(*Result, *CompletedCrashTrace, *CrashIdentity)
	}{
		{"tool failure", func(r *Result, p *CompletedCrashTrace, f *CrashIdentity) { r.Outcome = OutcomeToolFailure }},
		{"application failure", func(r *Result, p *CompletedCrashTrace, f *CrashIdentity) { r.Outcome = OutcomeApplicationFailed }},
		{"clean", func(r *Result, p *CompletedCrashTrace, f *CrashIdentity) { r.Outcome = OutcomeClean }},
		{"error", func(r *Result, p *CompletedCrashTrace, f *CrashIdentity) { r.Err = errors.New("failure") }},
		{"nil log", func(r *Result, p *CompletedCrashTrace, f *CrashIdentity) { r.Log = nil }},
		{"no summary", func(r *Result, p *CompletedCrashTrace, f *CrashIdentity) { r.Log.SummaryFound = false }},
		{"no summaries", func(r *Result, p *CompletedCrashTrace, f *CrashIdentity) { r.Log.Summaries = 0 }},
		{"zero defects", func(r *Result, p *CompletedCrashTrace, f *CrashIdentity) { r.Log.Errors = 0 }},
		{"unknown tool", func(r *Result, p *CompletedCrashTrace, f *CrashIdentity) { r.Config.Tool = "unknown" }},
		{"negative defects", func(r *Result, p *CompletedCrashTrace, f *CrashIdentity) { r.Log.Errors = -1; r.Log.Hazards = 1 }},
		{"unknown image", func(r *Result, p *CompletedCrashTrace, f *CrashIdentity) { f.Image = [32]byte{} }},
		{"unknown input", func(r *Result, p *CompletedCrashTrace, f *CrashIdentity) { f.Input = [32]byte{} }},
		{"unknown previous input", func(r *Result, p *CompletedCrashTrace, f *CrashIdentity) { p.Identity.Input = [32]byte{} }},
		{"partial", func(r *Result, p *CompletedCrashTrace, f *CrashIdentity) { p.Complete = false }},
		{"stale ordinal", func(r *Result, p *CompletedCrashTrace, f *CrashIdentity) { p.Identity.Ordinal-- }},
		{"ordinal overflow", func(r *Result, p *CompletedCrashTrace, f *CrashIdentity) {
			p.Identity.Ordinal = ^uint64(0)
			f.Ordinal = 0
		}},
		{"foreign epoch", func(r *Result, p *CompletedCrashTrace, f *CrashIdentity) { p.Identity.Epoch = "other" }},
		{"same trial", func(r *Result, p *CompletedCrashTrace, f *CrashIdentity) { p.Identity.Trial = f.Trial }},
		{"foreign image", func(r *Result, p *CompletedCrashTrace, f *CrashIdentity) { p.Identity.Image[0] ^= 1 }},
		{"foreign kernel", func(r *Result, p *CompletedCrashTrace, f *CrashIdentity) { p.Identity.Kernel = "other" }},
		{"absent trace", func(r *Result, p *CompletedCrashTrace, f *CrashIdentity) { p.Bitmap = nil }},
		{"zero trace", func(r *Result, p *CompletedCrashTrace, f *CrashIdentity) { p.Bitmap.Reset() }},
		{"size mismatch", func(r *Result, p *CompletedCrashTrace, f *CrashIdentity) { p.MapSize = 34 }},
		{"odd size", func(r *Result, p *CompletedCrashTrace, f *CrashIdentity) { p.MapSize = 31 }},
		{"no findings", func(r *Result, p *CompletedCrashTrace, f *CrashIdentity) { r.Log.Findings = nil }},
		{"invalid headline", func(r *Result, p *CompletedCrashTrace, f *CrashIdentity) { r.Log.Findings[0].Headline = "\n" }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r, p, f := crashFixture(t)
			tt.change(r, &p, &f)
			got, err := CrashSignature(r, p, f)
			if err == nil || !reflect.DeepEqual(got, CrashObservation{}) {
				t.Fatalf("accepted %+v, %v", got, err)
			}
		})
	}
	_, p, f := crashFixture(t)
	if _, err := CrashSignature(nil, p, f); err == nil {
		t.Fatal("nil result accepted")
	}
}

func TestCrashSignatureBindsRawTraceAndClassification(t *testing.T) {
	r, p, f := crashFixture(t)
	a, err := CrashSignature(r, p, f)
	if err != nil {
		t.Fatal(err)
	}
	changes := []struct {
		name   string
		change func(*Result, *CompletedCrashTrace, *CrashIdentity)
	}{
		{"trace", func(r *Result, p *CompletedCrashTrace, f *CrashIdentity) {
			m, _ := cudatrial.NewThreadEdgeMap(32, 1)
			m.Record(0, 4)
			p.Bitmap = m.Snapshot()
		}},
		{"raw counts", func(r *Result, p *CompletedCrashTrace, f *CrashIdentity) {
			m, _ := cudatrial.NewThreadEdgeMap(32, 1)
			m.Record(0, 3)
			m.Reset()
			m.Record(0, 3)
			m.Record(0, 0)
			p.Bitmap = m.Snapshot()
		}},
		{"headline", func(r *Result, p *CompletedCrashTrace, f *CrashIdentity) {
			r.Log.Findings[0].Headline = "Invalid __global__ read of size 4 bytes"
		}},
		{"tool", func(r *Result, p *CompletedCrashTrace, f *CrashIdentity) { r.Config.Tool = Racecheck }},
		{"kernel", func(r *Result, p *CompletedCrashTrace, f *CrashIdentity) {
			p.Identity.Kernel = "other"
			f.Kernel = "other"
		}},
		{"image", func(r *Result, p *CompletedCrashTrace, f *CrashIdentity) {
			p.Identity.Image[0] ^= 1
			f.Image = p.Identity.Image
		}},
	}
	for _, tt := range changes {
		t.Run(tt.name, func(t *testing.T) {
			r, p, f := crashFixture(t)
			tt.change(r, &p, &f)
			b, err := CrashSignature(r, p, f)
			if err != nil || b.Signature == a.Signature {
				t.Fatalf("unbound %s: %v", tt.name, err)
			}
		})
	}
}
