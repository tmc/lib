package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/internal/cudainject/wire"
	"github.com/tmc/lib/nvidia/x/cudaprof/marker"
	"github.com/tmc/lib/nvidia/x/cudaprof/shape"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
)

func componentJournal(t *testing.T, batches, roots int) []byte {
	t.Helper()
	b := newJournalBuilder()
	appendMarker := func(name string, id, flags uint32, at int) {
		b.append(wire.KindActivity, map[string]any{"kind": "activity", "subtype": "marker", "valid": true, "name": name, "marker_id": id, "marker_flags": flags, "thread_id": 2, "process_id": 7, "start": at, "end": at})
	}
	contract := &marker.ComponentContract{Name: "gelu", BankSHA256: strings.Repeat("a", 64), RootSetSHA256: strings.Repeat("b", 64), Shape: []int{256}, Strides: []int{1}, DType: "bfloat16", MathForm: "direct-fused", ScalarPolicy: "weak-float", ExecutionPolicy: "synchronized_eval", RootPolicy: "distinct_banks"}
	invocation := 0
	for batch := 0; batch < batches; batch++ {
		count := batch
		m := marker.Metadata{Version: 3, RequestID: "r", Role: "measured", Phase: "evaluate", Boundary: "submission", ExecutionUnit: "component_batch", Executions: &count, Invocation: &invocation, Batch: &batch, RootCount: &roots, Operation: "gelu", Component: contract}
		encode := func() string {
			text, err := marker.Encode(m)
			if err != nil {
				t.Fatal(err)
			}
			return text
		}
		start := batch*10000 + 100
		batchID := uint32(batch*1000 + 1)
		m.Event = "batch_start"
		appendMarker(encode(), batchID+900, timeline.MarkerInstantaneous, start)
		m.Event = ""
		appendMarker(encode(), batchID, timeline.MarkerStart, start+1)
		for root := 0; root < roots; root++ {
			m.Root = &root
			id := batchID + uint32(root) + 1
			rs := start + 10 + root*10
			appendMarker(encode(), id, timeline.MarkerStart, rs)
			for k := 0; k < 2; k++ {
				corr := batch*1000 + root*2 + k + 1
				b.append(wire.KindAPI, map[string]any{"kind": "api", "domain": "driver", "correlation": corr, "thread_id": 2, "process_id": 7, "start": rs + 1 + k*2, "end": rs + 2 + k*2})
				b.append(wire.KindActivity, map[string]any{"kind": "activity", "subtype": "kernel", "valid": true, "name": "OMultiply", "correlation": corr, "start": start + 6000 + root*4 + k*2, "end": start + 6001 + root*4 + k*2, "grid_x": 1, "grid_y": 1, "grid_z": 1, "block_x": 32, "block_y": 1, "block_z": 1, "dynamic_shared_memory": 0})
			}
			appendMarker("", id, timeline.MarkerEnd, rs+8)
		}
		appendMarker("", batchID, timeline.MarkerEnd, start+5000)
		m.Root = nil
		count = batch + 1
		m.Event = "batch_end"
		m.Boundary = "completion"
		appendMarker(encode(), batchID+901, timeline.MarkerInstantaneous, start+8000)
	}
	return b.finalize()
}
func TestComponentShapeCommand(t *testing.T) {
	dir := t.TempDir()
	capture := filepath.Join(dir, "component.pprof")
	pairs := filepath.Join(dir, "pairs.json")
	writeTestCaptureFile(t, capture, testCapture(map[string][]byte{"capture.journal": componentJournal(t, 6, 3)}, completeStatus()))
	data, _ := json.Marshal([]shape.RequestPair{{BaseJournal: "capture.journal", BaseProcess: 7, BaseID: "r", TargetJournal: "capture.journal", TargetProcess: 7, TargetID: "r", Role: "measured"}})
	if err := os.WriteFile(pairs, data, 0600); err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	if err := runShapeCompare([]string{"-scope=component", "-request-pairs", pairs, "-iteration-marker=OMultiply", capture, capture}, &out); err != nil {
		t.Fatal(err)
	}
	var report struct {
		Version   int                            `json:"version"`
		Schema    string                         `json:"schema"`
		Component shape.ComponentReport          `json:"component_qualification"`
		Regex     [2]componentBoundaryDiagnostic `json:"inferred_boundary_diagnostic"`
		Arms      []struct {
			Isolation struct {
				State string `json:"state"`
			} `json:"gpu_isolation"`
		} `json:"arms"`
	}
	if err := json.Unmarshal(out.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	if report.Version != 2 || !report.Component.Pass || report.Component.MatchedRoots != 18 || report.Component.Arms[0].ExplicitIterations != 6 || report.Component.Arms[0].Kernels != 36 {
		t.Fatalf("%s", out.Bytes())
	}
	for _, d := range report.Regex {
		if d.TotalMatches != 36 || d.AmbiguousBatches != 6 || len(d.Samples) != 4 || d.OmittedSamples != 2 || d.BatchMatches[6] != 6 || !strings.HasPrefix(d.State, "ambiguous:") {
			t.Fatalf("%+v", d)
		}
	}
	if report.Arms[0].Isolation.State != "unchecked" || report.Component.Qualifications["native_completion"] != "unknown" {
		t.Fatal("promoted native unknown")
	}
	receipt := filepath.Join(dir, "receipt.json")
	if err := os.WriteFile(receipt, out.Bytes(), 0600); err != nil {
		t.Fatal(err)
	}
	alignment := alignmentReceipt{}
	alignment.Arms[0].SHA256 = "base"
	alignment.Arms[1].SHA256 = "target"
	encoded, _ := json.Marshal(alignment)
	session := &traceSession{Kind: "comparison_archive", Members: map[string][]byte{".cudaprof-bindings.json": {}, ".alignment.json": encoded}}
	if _, _, _, err := validatedTraceShape(session, receipt, defaultTraceInputLimits()); err == nil || !strings.Contains(err.Error(), "unsupported schema") {
		t.Fatalf("model consumer accepted component receipt: %v", err)
	}
	out.Reset()
	if err := runShapeCompare([]string{"-scope=geometry", capture, capture}, &out); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), "cudaprof.kernel-geometry-comparison.v1") || strings.Contains(out.String(), "component_qualification") {
		t.Fatal("geometry promoted component contract")
	}
	out.Reset()
	if err := runShapeCompare([]string{capture, capture}, &out); err != nil {
		t.Fatal(err)
	}
	var model map[string]json.RawMessage
	if err := json.Unmarshal(out.Bytes(), &model); err != nil {
		t.Fatal(err)
	}
	if string(model["version"]) != "1" || model["shape_qualification"] == nil || model["component_qualification"] != nil {
		t.Fatal("changed default model envelope")
	}
}
func TestComponentShapeCommandArgs(t *testing.T) {
	for _, args := range [][]string{{"-scope=other", "a", "b"}, {"-scope=component", "a", "b"}} {
		if err := runShapeCompare(args, new(bytes.Buffer)); err == nil {
			t.Fatalf("accepted %v", args)
		}
	}
	dir := t.TempDir()
	pairs := filepath.Join(dir, "pairs.json")
	if err := os.WriteFile(pairs, []byte(`[{"extra":true}]`), 0600); err != nil {
		t.Fatal(err)
	}
	if err := runShapeCompare([]string{"-scope=component", "-request-pairs", pairs, "a", "b"}, new(bytes.Buffer)); err == nil {
		t.Fatal("accepted malformed mapping")
	}
}

func TestComponentDocumentationExamples(t *testing.T) {
	data, err := os.ReadFile("trace.md")
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, line := range strings.Split(string(data), "\n") {
		if strings.HasPrefix(line, "cudaprof:v3 ") {
			m, err := marker.Parse(line)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := marker.Encode(m); err != nil {
				t.Fatal(err)
			}
			if m.Version != 3 || *m.RootCount != 45 || m.Root != nil {
				t.Fatal("invalid batch example")
			}
			found = true
		}
	}
	if !found {
		t.Fatal("missing complete component marker example")
	}
	const example = `[{"base_journal":"base.journal","base_process":101,"base_id":"geglu","target_journal":"target.journal","target_process":202,"target_id":"geglu","role":"measured"}]`
	if !strings.Contains(string(data), example) {
		t.Fatal("missing request mapping example")
	}
	path := filepath.Join(t.TempDir(), "pairs.json")
	if err := os.WriteFile(path, []byte(example), 0600); err != nil {
		t.Fatal(err)
	}
	pairs, err := readShapeRequestPairs(path)
	if err != nil || len(pairs) != 1 || pairs[0].BaseProcess != 101 || pairs[0].TargetProcess != 202 {
		t.Fatalf("%v %v", pairs, err)
	}
}
