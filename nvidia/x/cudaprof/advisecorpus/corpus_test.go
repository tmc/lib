package advisecorpus

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/google/pprof/profile"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaevidence"
	"github.com/tmc/lib/nvidia/x/cudaprof/advise"
)

type fixture struct {
	root     string
	r        Record
	link     Link
	decision map[string]any
}

func (f *fixture) file(t *testing.T, name string, raw []byte) Artifact {
	t.Helper()
	p := filepath.Join(f.root, name)
	if err := os.WriteFile(p, raw, 0600); err != nil {
		t.Fatal(err)
	}
	return Artifact{Path: name, SHA256: digest(raw), Bytes: int64(len(raw))}
}
func (f *fixture) object(t *testing.T, name string, v any) Artifact {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return f.file(t, name, b)
}
func newFixture(t *testing.T) *fixture {
	t.Helper()
	f := &fixture{root: t.TempDir()}
	a := f.file(t, "baseline.cu", []byte("// CPU MOCK baseline"))
	b := f.file(t, "candidate.cu", []byte("// CPU MOCK candidate"))
	raw, _ := json.Marshal(map[string]any{"id": "CPU_MOCK_ONLY", "kind": "STATIC_COMPILE_CLOSURE", "source": map[string]Artifact{"baseline": a, "candidate": b}, "mechanism": map[string]string{"hypothesis": "reduce producer issue instructions", "status": "STATIC_ONLY"}, "comparison": map[string]any{"boundary": "STATIC_PRODUCER_LOOP_CFG_UNION", "same_run": false, "paired_delta_qualified": false}, "context": map[string]string{"gpu": "CPU_MOCK_ONLY"}, "case": map[string]int{"BM": 128}, "observation": "STATIC_STOP_NO_NATIVE", "stages": map[string]string{"unprofiled_timing": "NOT_RUN"}, "advise": map[string]string{"status": "NOT_EVALUATED"}, "adoption": "NOT_ADOPTED"})
	records, err := Decode(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	f.r = records[0]
	i := 0
	f.link = Link{Schema: "corpus-advise-link/v1", RecordID: "CPU_MOCK_ONLY", SourcePair: map[string]string{"baseline": a.SHA256, "candidate": b.SHA256}, Origin: "NONBLIND_CALIBRATION", Scope: "STATIC_ONLY", Outcome: "LEAD", Mechanism: f.r["mechanism"], InputArm: "baseline", InputSourceSHA256: a.SHA256, ItemIndex: &i}
	f.link.ToolSourceReceipt = f.file(t, "tool-sources.json", []byte(`{"CPU_MOCK_ONLY":true}`))
	f.link.ToolSourceSHA256 = f.link.ToolSourceReceipt.SHA256
	f.link.ToolBinary = f.file(t, "tool-not-an-executable", []byte("CPU mock tool bytes; not native execution"))
	f.link.InputCapture = f.file(t, "capture.json", []byte(`{"CPU_MOCK_ONLY":true}`))
	f.link.AdviseReport = f.object(t, "advice.json", advise.Report{Items: []advise.Item{{Finding: "elected producer copy loop", ActionableRecommendation: "compare producer issue schedule", Location: advise.Location{Kernel: "cpu_fixture"}, EvidenceLevel: advise.EvidenceSuspected, NoRuntimeAttribution: true}}})
	f.decision = map[string]any{"record_id": f.link.RecordID, "source_pair": f.link.SourcePair, "mechanism": f.link.Mechanism, "advice_report_sha256": f.link.AdviseReport.SHA256, "outcome": f.link.Outcome, "advice_sequence": 1, "decision_sequence": 2}
	f.link.DecisionReceipt = f.object(t, "decision.json", f.decision)
	return f
}
func (f *fixture) runtime(t *testing.T) {
	t.Helper()
	delete(f.r, "kind")
	f.r["observation"] = json.RawMessage(`"CPU_MOCK_PAIRED_TIMING"`)
	f.r["stages"] = json.RawMessage(`{"native_numerical":{"status":"PASS_CPU_MOCK"},"unprofiled_timing":{"status":"PASS_CPU_MOCK"}}`)
	f.r["comparison"] = json.RawMessage(`{"boundary":"CPU_MOCK_PAIRED","same_run":true,"paired_delta_qualified":true}`)
	f.link.Scope = "RUNTIME_DIAGNOSTIC"
	context, _ := json.Marshal(map[string]json.RawMessage{"context": f.r["context"], "case": f.r["case"], "boundary": mustJSON("CPU_MOCK_PAIRED")})
	context, _ = canonical(context)
	validator := f.file(t, "validator.py", []byte("# CPU MOCK validator; not a native producer"))
	f.link.CaptureProofs = map[string]Artifact{}
	for _, arm := range []string{"baseline", "candidate"} {
		capture := f.file(t, arm+"-capture.json", []byte(fmt.Sprintf(`{"CPU_MOCK_ONLY":"%s"}`, arm)))
		receipt := map[string]any{"schema": "corpus-advise-validator-receipt/v1", "status": "PASS", "validator_source_sha256": validator.SHA256, "source_sha256": f.link.SourcePair[arm], "context_sha256": digest(context), "raw_capture": capture, "run_completeness": "complete", "dropped_records": 0, "postprofile_numerical_state": "PASS"}
		proof := map[string]any{"schema": "corpus-advise-capture-proof/v1", "source_sha256": f.link.SourcePair[arm], "context_sha256": digest(context), "capture_success": true, "run_completeness": "complete", "dropped_records": 0, "postprofile_numerical_state": "PASS", "raw_capture": capture, "validator_source_sha256": validator.SHA256, "validator_source": validator, "validator_receipt": f.object(t, arm+"-validator.json", receipt), "metrics": []any{map[string]any{"name": "CPU_MOCK_counter", "unit": "count", "value": 1, "quantity_class": "measured"}}}
		f.link.CaptureProofs[arm] = f.object(t, arm+"-proof.json", proof)
		if arm == f.link.InputArm {
			f.link.InputCapture = capture
		}
	}
}
func (f *fixture) mutateProof(t *testing.T, arm string, fn func(map[string]any)) {
	t.Helper()
	raw, err := artifact(f.root, f.link.CaptureProofs[arm])
	if err != nil {
		t.Fatal(err)
	}
	var p map[string]any
	json.Unmarshal(raw, &p)
	fn(p)
	f.link.CaptureProofs[arm] = f.object(t, arm+"-proof.json", p)
}
func TestNoLinkDoesNotInferAdviceFromRatioOrStaticStop(t *testing.T) {
	f := newFixture(t)
	r, err := Consume(f.root, f.r, nil)
	if err != nil {
		t.Fatal(err)
	}
	if r["classification"] != "NOT_EVALUATED" || r["native_revalidation"] != "NOT_RUN" {
		t.Fatalf("invented evaluation: %v", r)
	}
}
func TestLoggedStaticOutcomes(t *testing.T) {
	for _, outcome := range []string{"LEAD", "MISS", "MISLEAD"} {
		t.Run(outcome, func(t *testing.T) {
			f := newFixture(t)
			f.link.Outcome = outcome
			f.decision["outcome"] = outcome
			if outcome == "MISS" {
				f.link.ItemIndex = nil
			}
			f.link.DecisionReceipt = f.object(t, "decision.json", f.decision)
			r, err := Consume(f.root, f.r, &f.link)
			if err != nil {
				t.Fatal(err)
			}
			if r["classification"] != "STATIC_"+outcome+"_LOGGED_NOT_RUNTIME_WIN" || r["mechanism_causality"] != "NOT_ESTABLISHED" {
				t.Fatal(r)
			}
		})
	}
}
func TestRejectStaticAndChronologyMutations(t *testing.T) {
	tests := []struct {
		name   string
		change func(*fixture)
	}{
		{"source pair", func(f *fixture) { f.link.SourcePair["baseline"] = strings.Repeat("0", 64) }},
		{"input arm", func(f *fixture) { f.link.InputSourceSHA256 = strings.Repeat("0", 64) }},
		{"mechanism", func(f *fixture) { f.link.Mechanism = json.RawMessage(`"different mechanism"`) }},
		{"blind", func(f *fixture) { f.link.Origin = "BLIND_HELD_OUT" }},
		{"late advice", func(f *fixture) {
			f.decision["advice_sequence"] = 3
			f.link.DecisionReceipt = f.object(t, "decision.json", f.decision)
		}},
		{"stale report", func(f *fixture) {
			os.WriteFile(filepath.Join(f.root, f.link.AdviseReport.Path), []byte(`{"items":[]}`), 0600)
		}},
		{"static hardware", func(f *fixture) { f.link.HardwareMetrics = []json.RawMessage{json.RawMessage(`{"value":1}`)} }},
		{"static as runtime", func(f *fixture) { f.link.Scope = "RUNTIME_DIAGNOSTIC" }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := newFixture(t)
			tt.change(f)
			if _, err := Consume(f.root, f.r, &f.link); err == nil {
				t.Fatal("accepted mismatched evidence")
			}
		})
	}
}
func TestRuntimeDTOPositiveIsCPUFixtureOnly(t *testing.T) {
	f := newFixture(t)
	f.runtime(t)
	r, err := Consume(f.root, f.r, &f.link)
	if err != nil {
		t.Fatal(err)
	}
	if r["classification"] != "RUNTIME_LEAD_LOGGED_NOT_CAUSAL_PROOF" || r["blind_success_claim"] != false {
		t.Fatal(r)
	}
}
func TestRejectRuntimeFaults(t *testing.T) {
	tests := []struct {
		name   string
		change func(*fixture)
	}{
		{"static stop with kind removed", func(f *fixture) { f.r["observation"] = json.RawMessage(`"STATIC_STOP_NO_NATIVE"`) }},
		{"native timing unrun", func(f *fixture) {
			f.r["stages"] = json.RawMessage(`{"native_numerical":"PASS","unprofiled_timing":"NOT_RUN"}`)
		}},
		{"case absent", func(f *fixture) { delete(f.r, "case") }},
		{"device absent", func(f *fixture) { f.r["context"] = json.RawMessage(`{}`) }},
		{"partial", func(f *fixture) {
			f.mutateProof(t, "candidate", func(p map[string]any) { p["run_completeness"] = "partial" })
		}},
		{"dropped", func(f *fixture) { f.mutateProof(t, "candidate", func(p map[string]any) { p["dropped_records"] = 1 }) }},
		{"postprofile failure", func(f *fixture) {
			f.mutateProof(t, "candidate", func(p map[string]any) { p["postprofile_numerical_state"] = "FAIL" })
		}},
		{"wrong case device", func(f *fixture) {
			f.mutateProof(t, "candidate", func(p map[string]any) { p["context_sha256"] = strings.Repeat("0", 64) })
		}},
		{"wrong candidate", func(f *fixture) {
			f.mutateProof(t, "candidate", func(p map[string]any) { p["source_sha256"] = f.link.SourcePair["baseline"] })
		}},
		{"malformed validator hash", func(f *fixture) {
			f.mutateProof(t, "candidate", func(p map[string]any) { p["validator_source_sha256"] = "not-a-digest" })
		}},
		{"empty validator receipt", func(f *fixture) {
			f.mutateProof(t, "candidate", func(p map[string]any) { p["validator_receipt"] = f.object(t, "empty-validator.json", map[string]any{}) })
		}},
		{"unrelated input capture", func(f *fixture) {
			f.link.InputCapture = f.file(t, "unrelated.json", []byte(`{"CPU_MOCK_ONLY":"unrelated"}`))
		}},
		{"validator byte drift", func(f *fixture) {
			os.WriteFile(filepath.Join(f.root, "validator.py"), []byte("changed validator"), 0600)
		}},
		{"missing candidate proof", func(f *fixture) { delete(f.link.CaptureProofs, "candidate") }},
		{"unpaired timing", func(f *fixture) {
			f.r["comparison"] = json.RawMessage(`{"boundary":"FULL","same_run":false,"paired_delta_qualified":false}`)
		}},
		{"quoted hardware number", func(f *fixture) {
			f.mutateProof(t, "candidate", func(p map[string]any) {
				p["metrics"] = []any{map[string]any{"name": "counter", "unit": "count", "value": "1", "quantity_class": "measured"}}
			})
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := newFixture(t)
			f.runtime(t)
			tt.change(f)
			if _, err := Consume(f.root, f.r, &f.link); err == nil {
				t.Fatal("accepted invalid runtime proof")
			}
		})
	}
}
func TestEveryValidatorReceiptFieldIsBound(t *testing.T) {
	for _, key := range []string{"schema", "status", "validator_source_sha256", "source_sha256", "context_sha256", "raw_capture", "run_completeness", "dropped_records", "postprofile_numerical_state"} {
		t.Run(key, func(t *testing.T) {
			f := newFixture(t)
			f.runtime(t)
			f.mutateProof(t, "candidate", func(p map[string]any) {
				b, _ := json.Marshal(p["validator_receipt"])
				var ref Artifact
				json.Unmarshal(b, &ref)
				raw, err := artifact(f.root, ref)
				if err != nil {
					t.Fatal(err)
				}
				var receipt map[string]any
				json.Unmarshal(raw, &receipt)
				delete(receipt, key)
				p["validator_receipt"] = f.object(t, "candidate-validator.json", receipt)
			})
			if _, err := Consume(f.root, f.r, &f.link); err == nil {
				t.Fatal("accepted missing validator binding")
			}
		})
	}
}
func TestStrictJSON(t *testing.T) {
	for _, raw := range []string{`{"id":"x","id":"y"}`, `{"a":{"x":1,"x":2}}`, `{"value":NaN}`, `{"value":1e9999}`, `{} {}`} {
		if _, err := canonical([]byte(raw)); err == nil {
			t.Fatalf("accepted %s", raw)
		}
	}
	raw := `{"id":"x","source":{"baseline":{"path":"b","sha256":"` + strings.Repeat("0", 64) + `","bytes":1},"candidate":{"path":"c","sha256":"` + strings.Repeat("1", 64) + `","bytes":1}}}`
	if _, err := Decode(strings.NewReader(raw + "\n" + raw)); err == nil {
		t.Fatal("accepted duplicate record id")
	}
}
func TestPythonCanonicalIdentity(t *testing.T) {
	raw := []byte(`{"z":"µ<\u007f😀","f":[1.0,1e-8,1e15,1e16,-0.0,0.0001]}`)
	got, err := canonical(raw)
	if err != nil {
		t.Fatal(err)
	}
	want := `{"f":[1.0,1e-08,1000000000000000.0,1e+16,-0.0,0.0001],"z":"\u00b5<\u007f\ud83d\ude00"}`
	if string(got) != want {
		t.Fatalf("got %s\nwant %s", got, want)
	}
}
func TestExactMechanismMatcher(t *testing.T) {
	r := &advise.Report{Items: []advise.Item{{Finding: "producer loop", ActionableRecommendation: "A/B publication schedule", Location: advise.Location{Kernel: "same_kernel"}, EvidenceLevel: advise.EvidenceSuspected}}}
	ids, err := Match(r, "producer loop", "A/B publication schedule", "same_kernel")
	if err != nil || len(ids) != 1 {
		t.Fatalf("%v %v", ids, err)
	}
	ids, err = Match(r, "producer", "A/B publication schedule", "same_kernel")
	if err != nil || len(ids) != 0 {
		t.Fatal("substring matched")
	}
	if _, err := Match(r, "", "", "same_kernel"); err == nil {
		t.Fatal("empty criterion matched")
	}
}
func TestAnalyzeCallsRealAPIWithoutNativeClaim(t *testing.T) {
	f := newFixture(t)
	var capture bytes.Buffer
	cap := &cudacapture.Capture{CodeUnavailable: map[string]string{"ptx": "CPU fixture, no compiler run", "cubin": "CPU fixture, no native binary", "sass": "CPU fixture, no native code"}, Profile: &profile.Profile{SampleType: []*profile.ValueType{{Type: "samples", Unit: "count"}}}, Files: map[string][]byte{"baseline.cu": []byte("// CPU MOCK baseline")}, Code: cudacapture.CodeFiles{Source: "baseline.cu"}}
	if err := cudacapture.Write(&capture, cap); err != nil {
		t.Fatal(err)
	}
	ref := f.file(t, "compiler-capture.pprof", capture.Bytes())
	evidence := cudaevidence.Source{Identity: cudaevidence.Identity{ArtifactDigest: strings.Repeat("1", 64), CompilerKey: strings.Repeat("2", 64), WorkloadKey: strings.Repeat("3", 64), Entry: "CPU_MOCK_ONLY", Target: "sm_90a"}, Producer: "CPU_MOCK_ONLY", Locator: "compiler fixture"}
	report, err := Analyze(f.root, f.r, "baseline", ref, evidence)
	if err != nil {
		t.Fatal(err)
	}
	if !report.NoProfileSamples {
		t.Fatal("invented native samples")
	}
	if _, err := Analyze(f.root, f.r, "candidate", ref, evidence); err == nil {
		t.Fatal("joined wrong source by symbol")
	}
	result, err := Consume(f.root, f.r, nil)
	if err != nil || result["classification"] != "NOT_EVALUATED" {
		t.Fatal("analysis invented outcome")
	}
}

func TestDecodeLinkRejectsAmbiguity(t *testing.T) {
	for _, raw := range []string{`{"schema":"corpus-advise-link/v1","record_id":"x","record_id":"y"}`, `{"schema":"other","record_id":"x"}`, `{"schema":"corpus-advise-link/v1","record_id":"x"} {}`} {
		if _, err := DecodeLink(strings.NewReader(raw)); err == nil {
			t.Fatal("accepted ambiguous link")
		}
	}
}

func TestRejectInvalidUnicodeIdentity(t *testing.T) {
	tests := []struct {
		name string
		raw  []byte
	}{
		{"reviewer lone high surrogate", []byte(`{"s":"\ud800"}`)},
		{"lone low surrogate", []byte(`{"s":"\udc00"}`)},
		{"high followed by scalar", []byte(`{"s":"\ud800\u0041"}`)},
		{"reversed pair", []byte(`{"s":"\udc00\ud800"}`)},
		{"surrogate object key", []byte(`{"\ud800":"value"}`)},
		{"high followed by escaped backslash", []byte(`{"s":"\ud800\\udc00"}`)},
		{"invalid UTF8", append([]byte(`{"s":"`), 0xff, '"', '}')},
		{"encoded surrogate", append([]byte(`{"s":"`), 0xed, 0xa0, 0x80, '"', '}')},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got, err := canonical(tt.raw); err == nil {
				t.Fatalf("silently normalized to %s", got)
			}
		})
	}
}

func TestValidUnicodeIdentityRoundTrips(t *testing.T) {
	tests := []struct{ name, raw, want string }{
		{"paired surrogate", `{"s":"\ud83d\ude00"}`, `{"s":"\ud83d\ude00"}`},
		{"literal multibyte", `{"s":"µ<😀"}`, `{"s":"\u00b5<\ud83d\ude00"}`},
		{"escaped surrogate text", `{"s":"\\ud800"}`, `{"s":"\\ud800"}`},
		{"replacement scalar", `{"s":"\ufffd"}`, `{"s":"\ufffd"}`},
		{"paired object key", `{"\ud83d\ude00":"µ"}`, `{"\ud83d\ude00":"\u00b5"}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := canonical([]byte(tt.raw))
			if err != nil || string(got) != tt.want {
				t.Fatalf("got %s, %v; want %s", got, err, tt.want)
			}
		})
	}
}

func TestCorpusAndLinkRejectInvalidUnicode(t *testing.T) {
	if _, err := Decode(strings.NewReader(`{"id":"\ud800"}`)); err == nil || !strings.Contains(err.Error(), "unpaired Unicode surrogate") {
		t.Fatalf("corpus did not reject Unicode before decoding: %v", err)
	}
	if _, err := DecodeLink(strings.NewReader(`{"schema":"corpus-advise-link/v1","record_id":"\ud800"}`)); err == nil || !strings.Contains(err.Error(), "unpaired Unicode surrogate") {
		t.Fatalf("link did not reject Unicode before decoding: %v", err)
	}
}

// A measured quantity must keep the arm whose source and capture were validated.
func TestRejectMetricArmOverride(t *testing.T) {
	for _, arm := range []string{"baseline", "candidate"} {
		for _, claimed := range []any{"baseline", "candidate", nil, 0} {
			t.Run(fmt.Sprintf("%s/%v", arm, claimed), func(t *testing.T) {
				f := newFixture(t)
				f.runtime(t)
				f.mutateProof(t, arm, func(p map[string]any) {
					metrics := p["metrics"].([]any)
					metrics[0].(map[string]any)["arm"] = claimed
				})
				if _, err := Consume(f.root, f.r, &f.link); err == nil || !strings.Contains(err.Error(), "reserved arm") {
					t.Fatalf("Consume error = %v, want reserved arm rejection", err)
				}
			})
		}
	}
}

func TestRuntimeMetricArmComesFromProof(t *testing.T) {
	f := newFixture(t)
	f.runtime(t)
	r, err := Consume(f.root, f.r, &f.link)
	if err != nil {
		t.Fatal(err)
	}
	metrics := r["hardware_metrics"].([]any)
	if len(metrics) != 2 {
		t.Fatalf("got %d metrics, want 2", len(metrics))
	}
	for i, arm := range []string{"baseline", "candidate"} {
		if got := metrics[i].(map[string]any)["arm"]; got != arm {
			t.Fatalf("metric %d arm = %v, want %s", i, got, arm)
		}
	}
}
