package cudaprof_test

import (
	"bytes"
	"encoding/json"
	"io"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudacache"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudaprof"
)

func publicationFixture(t *testing.T) (cudaprof.CapturePlan, cudaprof.CapturePublication) {
	t.Helper()
	schema, err := cudakernel.NewSchema()
	if err != nil {
		t.Fatal(err)
	}
	a, err := cudakernel.NewArtifact(cudakernel.KindPTX, []byte(".version 8.0\n.target sm_70\n.address_size 64\n.visible .entry kernel() { ret; }\n"), "kernel", schema, "sm_70")
	if err != nil {
		t.Fatal(err)
	}
	w := cudacache.WorkloadIdentity{ProgramID: "synthetic workload", ShapeDims: []int{64}, NumericalMode: "fp32"}
	id, err := cudaprof.BindEvidence(a, w, strings.Repeat("2", 64))
	if err != nil {
		t.Fatal(err)
	}
	receipt := ownerReceipt()
	receipt.Identity = id
	r, err := cudaprof.RangeCaptureProviderResult(receipt, rangeOwner(), []byte{7}, []string{"metric"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	plan := cudaprof.CapturePlan{Version: 1, Identity: id, Environment: r.Environment, Budget: cudaprof.CaptureBudget{MaxRecords: 100, MaxBytes: 1 << 20, MaxPasses: 4, MaxWallNanos: 1000}, Passes: []cudaprof.CapturePass{{ID: r.PassID, Replay: r.Replay, Providers: []cudaprof.CaptureProvider{"range"}, Configs: map[cudaprof.CaptureProvider]cudaprof.CaptureProviderConfig{"range": r.Config}}}}
	return plan, cudaprof.CapturePublication{Version: 1, Result: r, Artifact: a, Workload: w, CompilerKey: id.CompilerKey}
}
func TestCapturePublicationRoundTrip(t *testing.T) {
	plan, p := publicationFixture(t)
	var b bytes.Buffer
	if err := cudaprof.WriteCapturePublication(&b, plan, p); err != nil {
		t.Fatal(err)
	}
	got, err := cudaprof.ReadCapturePublication(bytes.NewReader(b.Bytes()), plan, plan.Budget.MaxBytes)
	if err != nil {
		t.Fatal(err)
	}
	if got.Result.Identity != p.Result.Identity || !bytes.Equal(got.Artifact.Bytes(), p.Artifact.Bytes()) || !bytes.Equal(got.Result.Raw, p.Result.Raw) || got.Workload.Key() != p.Workload.Key() {
		t.Fatal("publication changed identities or raw bytes")
	}
}
func TestCapturePublicationArtifactBinding(t *testing.T) {
	for _, tc := range []struct {
		name   string
		change func(*cudaprof.CapturePublication)
	}{
		{"invalid artifact", func(p *cudaprof.CapturePublication) { p.Artifact = &cudakernel.Artifact{} }},
		{"workload", func(p *cudaprof.CapturePublication) { p.Workload.ProgramID = "swapped" }},
		{"compiler", func(p *cudaprof.CapturePublication) { p.CompilerKey = strings.Repeat("5", 64) }},
		{"missing artifact", func(p *cudaprof.CapturePublication) { p.Artifact = nil }},
		{"version", func(p *cudaprof.CapturePublication) { p.Version = 99 }},
		{"entry", func(p *cudaprof.CapturePublication) { p.Result.Identity.Entry = "other" }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			plan, p := publicationFixture(t)
			tc.change(&p)
			var b bytes.Buffer
			if cudaprof.WriteCapturePublication(&b, plan, p) == nil || b.Len() != 0 {
				t.Fatal("invalid publication wrote bytes")
			}
		})
	}
}
func TestCapturePublicationPartialDiagnostics(t *testing.T) {
	plan, p := publicationFixture(t)
	p.Result.Status = "partial"
	p.Result.Reason = "collection failed"
	p.Result.CleanupComplete = false
	// Actual receipt differs from expected plan but is preserved as diagnostic.
	p.Workload.ProgramID = "actual-other-workload"
	id, err := cudaprof.BindEvidence(p.Artifact, p.Workload, p.CompilerKey)
	if err != nil {
		t.Fatal(err)
	}
	p.Result.Identity = id
	var b bytes.Buffer
	if err := cudaprof.WriteCapturePublication(&b, plan, p); err != nil {
		t.Fatal(err)
	}
	got, err := cudaprof.ReadCapturePublication(&b, plan, plan.Budget.MaxBytes)
	if err != nil || got.Result.Identity != id || got.Result.Reason != "collection failed" {
		t.Fatalf("partial = %+v, %v", got, err)
	}
	p.Artifact = nil
	b.Reset()
	if err := cudaprof.WriteCapturePublication(&b, plan, p); err != nil {
		t.Fatal(err)
	}
}
func TestCapturePublicationJSONAndBudgets(t *testing.T) {
	plan, p := publicationFixture(t)
	var b bytes.Buffer
	if err := cudaprof.WriteCapturePublication(&b, plan, p); err != nil {
		t.Fatal(err)
	}
	data := b.String()
	for _, s := range []string{data + ` {}`, strings.Replace(data, `"version":1`, `"Version":1`, 1), strings.Replace(data, `"version":1`, `"version":1,"version":1`, 1), strings.Replace(data, `"artifact":{`, `"artifact":{"unknown":1,`, 1), strings.Replace(data, `"result":{`, `"result":{"unknown":1,`, 1)} {
		if _, err := cudaprof.ReadCapturePublication(strings.NewReader(s), plan, plan.Budget.MaxBytes); err == nil {
			t.Fatal("non-strict JSON accepted")
		}
	}
	if _, err := cudaprof.ReadCapturePublication(strings.NewReader(data), plan, len(data)-1); err == nil {
		t.Fatal("reader ignored limit")
	}
	small := plan
	small.Budget.MaxBytes = len(data) - 1
	if _, err := cudaprof.ReadCapturePublication(strings.NewReader(data), small, plan.Budget.MaxBytes); err == nil {
		t.Fatal("plan byte limit ignored")
	}
	whitespace := data + strings.Repeat(" ", plan.Budget.MaxBytes)
	if _, err := cudaprof.ReadCapturePublication(strings.NewReader(whitespace), plan, 64<<20); err == nil {
		t.Fatal("original whitespace not budgeted")
	}
	spec := p.Artifact.Spec()
	spec.Raw = make([]byte, plan.Budget.MaxBytes+1)
	big, err := cudakernel.NewArtifactFromSpec(spec)
	if err != nil {
		t.Fatal(err)
	}
	p.Artifact = big
	b.Reset()
	if err := cudaprof.WriteCapturePublication(&b, plan, p); err == nil || !strings.Contains(err.Error(), "artifact byte budget") || b.Len() != 0 {
		t.Fatalf("oversized artifact = %v, wrote %d bytes", err, b.Len())
	}
}

// TestCapturePublicationCorruptArtifactWire edits the artifact inside a
// written publication. The unedited re-encoding is read back, so only the
// edit can cause the rejection.
func TestCapturePublicationCorruptArtifactWire(t *testing.T) {
	plan, p := publicationFixture(t)
	var b bytes.Buffer
	if err := cudaprof.WriteCapturePublication(&b, plan, p); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name string
		edit func(artifact map[string]any)
	}{
		{"unchanged", func(map[string]any) {}},
		{"raw", func(a map[string]any) {
			raw := p.Artifact.Bytes()
			raw[0] ^= 1
			a["raw"] = raw
		}},
		{"digest", func(a map[string]any) { a["digest"] = strings.Repeat("4", 64) }},
		{"missing digest", func(a map[string]any) { delete(a, "digest") }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var m map[string]any
			if err := json.Unmarshal(b.Bytes(), &m); err != nil {
				t.Fatal(err)
			}
			tc.edit(m["artifact"].(map[string]any))
			data, err := json.Marshal(m)
			if err != nil {
				t.Fatal(err)
			}
			_, err = cudaprof.ReadCapturePublication(bytes.NewReader(data), plan, plan.Budget.MaxBytes)
			if (err == nil) != (tc.name == "unchanged") {
				t.Fatalf("ReadCapturePublication = %v", err)
			}
		})
	}
}

type shortPublicationWriter struct{}

func (shortPublicationWriter) Write(b []byte) (int, error) { return len(b) - 1, nil }
func TestCapturePublicationShortWrite(t *testing.T) {
	plan, p := publicationFixture(t)
	if err := cudaprof.WriteCapturePublication(shortPublicationWriter{}, plan, p); err != io.ErrShortWrite {
		t.Fatalf("short write = %v", err)
	}
}
func TestCapturePublicationStrictArtifactNestedJSON(t *testing.T) {
	plan, p := publicationFixture(t)
	b, err := json.Marshal(p)
	if err != nil {
		t.Fatal(err)
	}
	s := strings.Replace(string(b), `"schema":{`, `"schema":{"unknown":1,`, 1)
	if _, err := cudaprof.ReadCapturePublication(strings.NewReader(s), plan, plan.Budget.MaxBytes); err == nil {
		t.Fatal("nested artifact unknown field accepted")
	}
}
