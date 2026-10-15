package cudadiag_test

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
	"sync"
	"testing"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cudadiag"
	"github.com/tmc/lib/nvidia/x/cudaevidence"
	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
	"github.com/tmc/lib/nvidia/x/cuptiactivity"
	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func summaryDigest(raw []byte) string {
	sum := sha256.Sum256(raw)
	return hex.EncodeToString(sum[:])
}

// Entirely constructed inputs: these bytes are not compiler/framework output.
func operatorSummaryInputs() ([]byte, nvtelemetry.OperatorContext, nvtelemetry.WorkloadSource) {
	p := cudaworkload.ProcessGenID{PID: 1, Generation: 2}
	cs := cudaworkload.ContextStreamID{Context: 3, Stream: 4}
	records := []cuptiactivity.Record{{Kind: cupti.ActivityKindConcurrentKernel, Name: "same kernel", CorrelationID: 7, ContextID: 3, StreamID: 4, Start: 10, End: 20, Complete: true}, {Kind: cupti.ActivityKindExternalCorrelation, CorrelationID: 7, ExternalID: 9, ExternalKind: cupti.ExternalCorrelationKindCustom0}}
	var raw bytes.Buffer
	for _, r := range records {
		if err := json.NewEncoder(&raw).Encode(r); err != nil {
			panic(err)
		}
	}
	zero := int64(0)
	receipt := tracecontract.Receipt{Version: 1, Command: "constructed", InputPath: "activities.jsonl", InputSHA256: summaryDigest(raw.Bytes()), Kind: "capture", ClockDomain: "cupti", ClockProvenance: "constructed producer clock", ClockUncertaintyNS: &zero, Partial: true, Backends: []tracecontract.Backend{{Name: "synthetic"}}, Gates: map[string]tracecontract.Gate{}, LostRecords: map[string]tracecontract.Loss{"activity": {Total: 1}, "external": {Total: 1}, "metadata": {Total: 2}}}
	for _, name := range []string{"collector", "import", "workload", "stationarity", "shape", "correctness"} {
		receipt.Gates[name] = tracecontract.Gate{Disposition: "unavailable", Reason: "constructed host control", Provenance: "synthetic"}
	}
	metadata, code := []byte(`{"constructed":["fx:1","fx:2"]}`), []byte("constructed code image")
	ref := nvtelemetry.OperatorReference{SHA256: summaryDigest(metadata), Locator: "nodes/0"}
	members := []nvtelemetry.OperatorMember{{ID: "fx:1", Name: "same", Schema: "constructed-v1", Shapes: [][]int64{{2}}, Source: ref}, {ID: "fx:2", Name: "same", Schema: "constructed-v1", Source: nvtelemetry.OperatorReference{SHA256: ref.SHA256, Locator: "nodes/1"}}}
	binding := nvtelemetry.OperatorBinding{Process: p, OpID: 9, Context: cs, CompilerBuild: "constructed-compiler", CompilerOptions: []string{"declared-option"}, Code: &cudaevidence.Identity{ArtifactDigest: summaryDigest(code), Entry: "k", Target: "sm_80", CompilerKey: strings.Repeat("a", 64), WorkloadKey: strings.Repeat("b", 64)}, ModuleID: "module", ModuleGeneration: 1, Pass: "p0", Replay: "normal"}
	inventory, err := json.Marshal(nvtelemetry.OperatorInventory{Version: 1, RunID: "run", Binding: binding, Members: members})
	if err != nil {
		panic(err)
	}
	c := nvtelemetry.OperatorContext{Version: 1, RunID: "run", Producer: "constructed", ProducerVersion: "1", ProducerSchema: "constructed-v1", SourceClass: "synthetic", Receipt: receipt, ActivityFamily: "activity", CorrelationFamily: "external", MetadataFamily: "metadata", Batch: cudaworkload.ActivityBatch{Session: "session", Run: "run", ClockEpoch: "epoch", EvidenceRef: receipt.InputSHA256, Process: p, ExternalKind: cupti.ExternalCorrelationKindCustom0}, Annotations: []cudaworkload.HostSubmission{{Process: p, OpID: 9, Symbol: "cuLaunchKernel", Workload: cudaworkload.WorkloadIdentity{Process: p, WorkloadName: "constructed", RunID: "run"}, Context: cs, Provenance: cudaworkload.ProvenanceCUPTI, EvidenceRef: "constructed annotation"}}, Artifacts: map[string][]byte{ref.SHA256: metadata, summaryDigest(code): code, summaryDigest(inventory): inventory}, Groups: []nvtelemetry.OperatorGroup{{ID: "fusion", Iteration: "iteration:1", Binding: binding, Members: members, InventorySource: &nvtelemetry.OperatorReference{SHA256: summaryDigest(inventory), Locator: "inventory/0"}}}}
	s := nvtelemetry.WorkloadSource{Provider: "application", Version: "operator-context-v1", RunID: "run", ClockDomain: "cupti", ClockEpochID: "epoch"}
	return raw.Bytes(), c, s
}

func analyzeOperatorFixture(t *testing.T, raw []byte, c nvtelemetry.OperatorContext, source nvtelemetry.WorkloadSource, limits nvtelemetry.WorkloadLimits) cudadiag.OperatorSummary {
	t.Helper()
	var metadata bytes.Buffer
	if err := nvtelemetry.WriteOperatorContext(&metadata, c, nvtelemetry.WorkloadLimits{}); err != nil {
		t.Fatal(err)
	}
	out, err := cudadiag.AnalyzeOperatorJSONL(context.Background(), bytes.NewReader(raw), &metadata, source, limits)
	if err != nil {
		t.Fatal(err)
	}
	return out
}

func TestOperatorSummaryPublicFusion(t *testing.T) {
	raw, c, source := operatorSummaryInputs()
	out := analyzeOperatorFixture(t, raw, c, source, nvtelemetry.WorkloadLimits{})
	if out.KernelPopulation.Total != 1 || out.KernelPopulation.Exact != 1 || out.MemberEdgePopulation.Total != 2 || out.MemberEdgePopulation.Associated != 2 || out.MemberEdgePopulation.Exact != 0 || len(out.Activities.Joins) != 1 || len(out.OrphanGroups) != 0 {
		t.Fatal(out)
	}
	if out.Membership[0].Status != "complete_relative_to_inventory" || len(out.Context.Artifacts) != 0 || len(out.RawArtifactSHA256) != 3 || out.Context.SourceClass != "synthetic" {
		t.Fatal(out)
	}
	if out.Context.Groups[0].Members[0].Name != out.Context.Groups[0].Members[1].Name || out.Edges[0].MemberID == out.Edges[1].MemberID {
		t.Fatal("names were used as identity")
	}
	data, err := json.Marshal(out)
	if err != nil {
		t.Fatal(err)
	}
	var replay cudadiag.OperatorSummary
	if err := json.Unmarshal(data, &replay); err != nil || replay.MemberEdgePopulation != out.MemberEdgePopulation || replay.Edges[0].Source != out.Edges[0].Source || replay.ContextSHA256 != out.ContextSHA256 {
		t.Fatal(replay, err)
	}
	for _, edge := range replay.Edges {
		if edge.Basis != "supplied_compiler_metadata" || edge.Outcome == cudaworkload.OutcomeExact {
			t.Fatal(edge)
		}
	}
}

func TestOperatorSummaryMissingExpectedMember(t *testing.T) {
	raw, c, source := operatorSummaryInputs()
	c.Groups[0].Members = c.Groups[0].Members[:1]
	out := analyzeOperatorFixture(t, raw, c, source, nvtelemetry.WorkloadLimits{})
	if out.Membership[0].Status != "incomplete" || out.MemberEdgePopulation.Total != 2 || out.MemberEdgePopulation.Associated != 1 || out.MemberEdgePopulation.Missing != 1 || !out.Edges[1].ExpectedOnly || out.Edges[1].MemberID != "fx:2" {
		t.Fatal(out)
	}
}

func TestOperatorSummaryBindingContradictions(t *testing.T) {
	for _, tc := range []struct {
		name   string
		change func(*nvtelemetry.OperatorBinding)
	}{
		{"process", func(b *nvtelemetry.OperatorBinding) { b.Process.Generation++ }},
		{"context", func(b *nvtelemetry.OperatorBinding) { b.Context.Context++ }},
		{"module", func(b *nvtelemetry.OperatorBinding) { b.ModuleGeneration++ }},
		{"pass", func(b *nvtelemetry.OperatorBinding) { b.Pass = "other" }},
		{"replay", func(b *nvtelemetry.OperatorBinding) { b.Replay = "other" }},
		{"compiler", func(b *nvtelemetry.OperatorBinding) { b.CompilerBuild = "other" }},
		{"code", func(b *nvtelemetry.OperatorBinding) { b.Code.ArtifactDigest = strings.Repeat("c", 64) }},
		{"graph", func(b *nvtelemetry.OperatorBinding) { b.GraphReplay.GraphID = 1 }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			raw, c, source := operatorSummaryInputs()
			tc.change(&c.Groups[0].Binding)
			out := analyzeOperatorFixture(t, raw, c, source, nvtelemetry.WorkloadLimits{})
			if out.MemberEdgePopulation.Associated != 0 || out.MemberEdgePopulation.Ambiguous != 2 || out.Membership[0].Status != "contradictory" || out.KernelPopulation.Exact != 1 {
				t.Fatal(out)
			}
		})
	}
}

func TestOperatorSummaryUnknownOrphansAndLoss(t *testing.T) {
	for _, mode := range []string{"inventory", "schema", "compiler", "raw", "loss", "clock", "correlation", "duplicate group"} {
		t.Run(mode, func(t *testing.T) {
			raw, c, source := operatorSummaryInputs()
			switch mode {
			case "inventory":
				c.Groups[0].InventorySource = nil
			case "schema":
				c.Groups[0].Members[0].Schema = ""
			case "compiler":
				c.Groups[0].Binding.CompilerBuild = ""
			case "raw":
				delete(c.Artifacts, c.Groups[0].Members[0].Source.SHA256)
			case "loss":
				c.Receipt.LostRecords["metadata"] = tracecontract.Loss{Unknown: true}
			case "clock":
				source.ClockEpochID = "other"
			case "correlation":
				raw = raw[:bytes.IndexByte(raw, '\n')+1]
				c.Receipt.InputSHA256 = summaryDigest(raw)
				c.Batch.EvidenceRef = c.Receipt.InputSHA256
			case "duplicate group":
				copy := c.Groups[0]
				copy.ID = "another-group"
				c.Groups = append(c.Groups, copy)
			}
			out := analyzeOperatorFixture(t, raw, c, source, nvtelemetry.WorkloadLimits{})
			switch mode {
			case "inventory":
				if out.Membership[0].Status != "unknown" || out.MemberEdgePopulation.Associated != 2 {
					t.Fatal(out)
				}
			case "schema", "compiler", "raw":
				if out.MemberEdgePopulation.Missing == 0 {
					t.Fatal(out)
				}
			case "loss":
				if !strings.Contains(strings.Join(out.PopulationLimits, " "), `loss for family "metadata"`) || !out.Context.Receipt.LostRecords["metadata"].Unknown {
					t.Fatal(out)
				}
			case "clock":
				if !strings.Contains(strings.Join(out.PopulationLimits, " "), "clock mapping") {
					t.Fatal(out)
				}
			case "correlation":
				if out.KernelPopulation.Missing != 1 || out.MemberEdgePopulation.Missing != 2 || len(out.OrphanGroups) != 1 || len(out.UnattributedKernelIndices) != 1 {
					t.Fatal(out)
				}
			case "duplicate group":
				if out.MemberEdgePopulation.Ambiguous != 4 {
					t.Fatal(out)
				}
			}
		})
	}
}

func TestOperatorSummaryLimitsCancellationAndConcurrentRead(t *testing.T) {
	raw, c, source := operatorSummaryInputs()
	var metadata bytes.Buffer
	if err := nvtelemetry.WriteOperatorContext(&metadata, c, nvtelemetry.WorkloadLimits{}); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := cudadiag.AnalyzeOperatorJSONL(ctx, bytes.NewReader(raw), bytes.NewReader(metadata.Bytes()), source, nvtelemetry.WorkloadLimits{}); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if _, err := cudadiag.AnalyzeOperatorJSONL(context.Background(), bytes.NewReader(raw[:len(raw)-1]), bytes.NewReader(metadata.Bytes()), source, nvtelemetry.WorkloadLimits{}); err == nil {
		t.Fatal("truncated activity admitted")
	}
	if _, err := cudadiag.AnalyzeOperatorJSONL(context.Background(), bytes.NewReader(append(raw, '\n')), bytes.NewReader(metadata.Bytes()), source, nvtelemetry.WorkloadLimits{}); err == nil {
		t.Fatal("changed raw activity admitted")
	}
	var wg sync.WaitGroup
	failures := make(chan error, 4)
	for range 4 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			got, err := cudadiag.AnalyzeOperatorJSONL(context.Background(), bytes.NewReader(raw), bytes.NewReader(metadata.Bytes()), source, nvtelemetry.WorkloadLimits{})
			if err != nil || got.MemberEdgePopulation.Associated != 2 {
				failures <- fmt.Errorf("concurrent read: %v", err)
			}
		}()
	}
	wg.Wait()
	close(failures)
	for err := range failures {
		t.Error(err)
	}
	// Three kernels and three members require nine edges, exceeding the
	// caller's seven-record budget even though the activity batch fits it.
	third := c.Groups[0].Members[1]
	third.ID, third.Source.Locator = "fx:3", "nodes/2"
	c.Groups[0].Members = append(c.Groups[0].Members, third)
	inv, err := json.Marshal(nvtelemetry.OperatorInventory{Version: 1, RunID: "run", Binding: c.Groups[0].Binding, Members: c.Groups[0].Members})
	if err != nil {
		t.Fatal(err)
	}
	hash := summaryDigest(inv)
	c.Artifacts[hash] = inv
	c.Groups[0].InventorySource.SHA256 = hash
	var many bytes.Buffer
	for i := uint32(0); i < 3; i++ {
		if err := json.NewEncoder(&many).Encode(cuptiactivity.Record{Kind: cupti.ActivityKindConcurrentKernel, CorrelationID: 7 + i, ContextID: 3, StreamID: 4}); err != nil {
			t.Fatal(err)
		}
		if err := json.NewEncoder(&many).Encode(cuptiactivity.Record{Kind: cupti.ActivityKindExternalCorrelation, CorrelationID: 7 + i, ExternalID: 9, ExternalKind: cupti.ExternalCorrelationKindCustom0}); err != nil {
			t.Fatal(err)
		}
	}
	c.Receipt.InputSHA256 = summaryDigest(many.Bytes())
	c.Batch.EvidenceRef = c.Receipt.InputSHA256
	metadata.Reset()
	if err := nvtelemetry.WriteOperatorContext(&metadata, c, nvtelemetry.WorkloadLimits{}); err != nil {
		t.Fatal(err)
	}
	if _, err := cudadiag.AnalyzeOperatorJSONL(context.Background(), &many, &metadata, source, nvtelemetry.WorkloadLimits{MaxMetrics: 7}); err == nil || !strings.Contains(err.Error(), "member edge limit") {
		t.Fatal("edge budget did not refuse explicitly", err)
	}
}

func TestOperatorSummaryCallerRecordCap(t *testing.T) {
	_, c, source := operatorSummaryInputs()
	var many bytes.Buffer
	for i := uint32(0); i < 4; i++ {
		if err := json.NewEncoder(&many).Encode(cuptiactivity.Record{Kind: cupti.ActivityKindConcurrentKernel, CorrelationID: 7 + i, ContextID: 3, StreamID: 4}); err != nil {
			t.Fatal(err)
		}
		if err := json.NewEncoder(&many).Encode(cuptiactivity.Record{Kind: cupti.ActivityKindExternalCorrelation, CorrelationID: 7 + i, ExternalID: 9, ExternalKind: cupti.ExternalCorrelationKindCustom0}); err != nil {
			t.Fatal(err)
		}
	}
	c.Receipt.InputSHA256 = summaryDigest(many.Bytes())
	c.Batch.EvidenceRef = c.Receipt.InputSHA256
	c.Batch.MaxRecords = 65536 // Input declarations cannot raise the caller cap.
	c.Groups = []nvtelemetry.OperatorGroup{}
	var metadata bytes.Buffer
	if err := nvtelemetry.WriteOperatorContext(&metadata, c, nvtelemetry.WorkloadLimits{}); err != nil {
		t.Fatal(err)
	}
	if _, err := cudadiag.AnalyzeOperatorJSONL(context.Background(), &many, &metadata, source, nvtelemetry.WorkloadLimits{MaxMetrics: 6}); err == nil {
		t.Fatal("input declaration raised caller activity cap")
	}
}

type operatorCancelReader struct {
	r      io.Reader
	cancel context.CancelFunc
}

func (r operatorCancelReader) Read(p []byte) (int, error) {
	n, err := r.r.Read(p)
	r.cancel()
	return n, err
}

func TestOperatorSummaryIndependentInventoryAndUnknownPopulation(t *testing.T) {
	for _, mode := range []string{"shortened without expectation", "self inventory", "unmatched same-name kernel", "inventory string bound", "mid-read cancel"} {
		t.Run(mode, func(t *testing.T) {
			raw, c, source := operatorSummaryInputs()
			switch mode {
			case "shortened without expectation":
				c.Groups[0].Members = c.Groups[0].Members[:1]
				c.Groups[0].InventorySource = nil
			case "self inventory":
				for i := range c.Groups[0].Members {
					c.Groups[0].Members[i].Source.SHA256 = c.Groups[0].InventorySource.SHA256
				}
			case "unmatched same-name kernel":
				var more bytes.Buffer
				more.Write(raw)
				if err := json.NewEncoder(&more).Encode(cuptiactivity.Record{Kind: cupti.ActivityKindConcurrentKernel, Name: "same kernel", CorrelationID: 99, ContextID: 3, StreamID: 4}); err != nil {
					t.Fatal(err)
				}
				raw = more.Bytes()
				c.Receipt.InputSHA256 = summaryDigest(raw)
				c.Batch.EvidenceRef = c.Receipt.InputSHA256
			case "inventory string bound":
				members := append([]nvtelemetry.OperatorMember(nil), c.Groups[0].Members...)
				members[1].Source.Locator = strings.Repeat("x", 4097)
				inv, err := json.Marshal(nvtelemetry.OperatorInventory{Version: 1, RunID: "run", Binding: c.Groups[0].Binding, Members: members})
				if err != nil {
					t.Fatal(err)
				}
				hash := summaryDigest(inv)
				c.Artifacts[hash] = inv
				c.Groups[0].InventorySource.SHA256 = hash
			case "mid-read cancel":
				var metadata bytes.Buffer
				if err := nvtelemetry.WriteOperatorContext(&metadata, c, nvtelemetry.WorkloadLimits{}); err != nil {
					t.Fatal(err)
				}
				ctx, cancel := context.WithCancel(context.Background())
				defer cancel()
				_, err := cudadiag.AnalyzeOperatorJSONL(ctx, bytes.NewReader(raw), operatorCancelReader{bytes.NewReader(metadata.Bytes()), cancel}, source, nvtelemetry.WorkloadLimits{})
				if !errors.Is(err, context.Canceled) {
					t.Fatal(err)
				}
				return
			}
			out := analyzeOperatorFixture(t, raw, c, source, nvtelemetry.WorkloadLimits{})
			if mode == "unmatched same-name kernel" {
				if out.KernelPopulation.Total != 2 || out.KernelPopulation.Missing != 1 || len(out.UnattributedKernelIndices) != 1 || out.MemberEdgePopulation.Total != 2 {
					t.Fatal(out)
				}
			} else if out.Membership[0].Status != "unknown" {
				t.Fatal(out)
			}
		})
	}
}
