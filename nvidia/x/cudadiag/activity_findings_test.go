package cudadiag_test

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"testing"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cudadiag"
	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
	"github.com/tmc/lib/nvidia/x/cuptiactivity"
)

func activityExport(t *testing.T) ([]byte, cudaworkload.ActivityBatch, []cudaworkload.HostSubmission, tracecontract.Receipt) {
	t.Helper()
	p := cudaworkload.ProcessGenID{PID: 1, Generation: 2}
	sub := cudaworkload.HostSubmission{Process: p, OpID: 9, Symbol: "cuLaunchKernel", Workload: cudaworkload.WorkloadIdentity{Process: p, RunID: "run"}, Provenance: cudaworkload.ProvenanceCUPTI, Context: cudaworkload.ContextStreamID{Context: 3, Stream: 4}, EvidenceRef: "application-annotation"}
	records := []cuptiactivity.Record{{Kind: cupti.ActivityKindConcurrentKernel, Name: "kernel", CorrelationID: 7, ContextID: 3, StreamID: 4, Start: 10, End: 20, Complete: true}, {Kind: cupti.ActivityKindExternalCorrelation, CorrelationID: 7, ExternalID: 9, ExternalKind: cupti.ExternalCorrelationKindCustom0}}
	var raw bytes.Buffer
	for _, r := range records {
		if err := json.NewEncoder(&raw).Encode(r); err != nil {
			t.Fatal(err)
		}
	}
	hash := sha256.Sum256(raw.Bytes())
	receipt := cudadiag.BuildSyntheticFixture(cudadiag.InjectionNone).Input.Receipt
	receipt.InputSHA256 = hex.EncodeToString(hash[:])
	receipt.ClockDomain = "cupti"
	receipt.ClockProvenance = "producer fixture epoch"
	receipt.Partial = true
	batch := cudaworkload.ActivityBatch{Session: "session", Run: "run", ClockEpoch: "producer-epoch", Process: p, EvidenceRef: receipt.InputSHA256, ExternalKind: cupti.ExternalCorrelationKindCustom0}
	return raw.Bytes(), batch, []cudaworkload.HostSubmission{sub}, receipt
}

func TestAnnotationExportToFinding(t *testing.T) {
	raw, batch, annotations, receipt := activityExport(t)
	got, err := cudadiag.AnalyzeActivityJSONL(bytes.NewReader(raw), batch, annotations, receipt)
	if err != nil || len(got.Findings) != 1 || !got.Findings[0].Admission.Admitted || *got.Findings[0].Value != 10 {
		t.Fatalf("bridge %+v %v", got, err)
	}
	if got.Joins[0].Record.CorrelationID != 7 || got.Joins[0].ClockEpoch != batch.ClockEpoch || got.Findings[0].Artifacts[0] != receipt.InputSHA256 {
		t.Fatal("raw provenance lost")
	}
	if got.Findings[0].Class != tracecontract.FindingObserved || !receipt.Partial {
		t.Fatal("evidence promoted")
	}
	receipt.ClockDomain = "host"
	got, err = cudadiag.AnalyzeActivityJSONL(bytes.NewReader(raw), batch, annotations, receipt)
	if err != nil || got.Findings[0].Admission.Admitted {
		t.Fatal("clock contradiction admitted")
	}
	receipt.ClockDomain = "cupti"
	annotations = nil
	got, err = cudadiag.AnalyzeActivityJSONL(bytes.NewReader(raw), batch, annotations, receipt)
	if err != nil || got.Findings[0].Admission.Admitted {
		t.Fatal("missing operation identity admitted")
	}
	if _, err = cudadiag.AnalyzeActivityJSONL(bytes.NewReader(raw[:len(raw)-1]), batch, annotations, receipt); err == nil {
		t.Fatal("truncated source accepted")
	}
	receipt.InputSHA256 = fmt.Sprintf("%064x", 1)
	if _, err = cudadiag.AnalyzeActivityJSONL(bytes.NewReader(raw), batch, annotations, receipt); err == nil {
		t.Fatal("changed raw hash accepted")
	}
}

func ExampleAnalyzeActivities() {
	_, err := cudadiag.AnalyzeActivities(cudaworkload.ActivityBatch{}, nil, tracecontract.Receipt{})
	fmt.Println(err != nil)
	// Output: true
}

func ExampleAnalyzeActivityJSONL() {
	_, err := cudadiag.AnalyzeActivityJSONL(bytes.NewReader(nil), cudaworkload.ActivityBatch{}, nil, tracecontract.Receipt{})
	fmt.Println(err != nil)
	// Output: true
}

func TestPublicActivityFixture(t *testing.T) {
	_, batch, annotations, receipt := activityExport(t)
	raw, err := os.ReadFile("testdata/activity-v1.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	h := sha256.Sum256(raw)
	receipt.InputSHA256 = hex.EncodeToString(h[:])
	got, err := cudadiag.AnalyzeActivityJSONL(bytes.NewReader(raw), batch, annotations, receipt)
	if err != nil || len(got.Findings) != 1 || !got.Findings[0].Admission.Admitted {
		t.Fatalf("public fixture %+v %v", got, err)
	}
}

func TestActivityExportDuplicateIdentityRefusal(t *testing.T) {
	raw, batch, annotations, receipt := activityExport(t)
	for _, replacement := range []string{`"CorrelationID":7,"CorrelationID":7`, `"correlationid":7`} {
		changed := bytes.Replace(raw, []byte(`"CorrelationID":7`), []byte(replacement), 1)
		h := sha256.Sum256(changed)
		receipt.InputSHA256 = hex.EncodeToString(h[:])
		if _, err := cudadiag.AnalyzeActivityJSONL(bytes.NewReader(changed), batch, annotations, receipt); err == nil {
			t.Fatal("ambiguous raw identity accepted")
		}
	}
}
