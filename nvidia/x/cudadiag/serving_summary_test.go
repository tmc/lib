package cudadiag_test

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"reflect"
	"sort"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudadiag"
	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

type servingCase struct {
	serving  nvtelemetry.ServingContextV2
	movement *movementCase // nil omits the movement capture
	window   bool
}

func newServingCase() *servingCase {
	tokens := uint64(7)
	a1 := nvtelemetry.ServingAttemptRef{RequestID: "r1", Attempt: "a1", Worker: "decode-0", WorkerEpoch: "e1"}
	a2 := nvtelemetry.ServingAttemptRef{RequestID: "r2", Attempt: "a1", Worker: "decode-0", WorkerEpoch: "e1"}
	return &servingCase{
		serving: nvtelemetry.ServingContextV2{
			Version: 2, RunID: "run",
			Attempts: []nvtelemetry.ServingAttempt{
				{RequestID: "r1", Attempt: "a1", Worker: "decode-0", WorkerEpoch: "e1", Role: "decode", Outcome: "accepted", OutputTokens: &tokens, OutputPolicy: "slo-v1", Evidence: "log-1"},
				{RequestID: "r2", Attempt: "a1", Worker: "decode-0", WorkerEpoch: "e1", Role: "decode", Outcome: "failed", Evidence: "log-2"},
			},
			Routes:    []nvtelemetry.ServingRoute{{Attempt: a1, Frontend: "fe-0", Decision: "routed", Basis: "native_id", Evidence: movementSHA([]byte("router log"))}},
			Batches:   []nvtelemetry.ServingBatch{{ID: "b1", Worker: "decode-0", WorkerEpoch: "e1", Role: "decode", Members: []nvtelemetry.ServingAttemptRef{a1, a2}, OpIDs: []cudaworkload.OpID{9}, Process: &cudaworkload.ProcessGenID{PID: 1, Generation: 2}, Basis: "annotation", Evidence: "batch-log"}},
			Caches:    []nvtelemetry.ServingCache{{Namespace: "kv", Generation: "g1", Worker: "decode-0", WorkerEpoch: "e1", Evidence: "cache-log"}},
			CacheUses: []nvtelemetry.ServingCacheUse{{Attempt: a1, Namespace: "kv", Generation: "g1", Event: "hit", Basis: "annotation", Evidence: "cache-log"}},
			Transfers: []nvtelemetry.ServingTransferRef{
				{Attempt: a1, TransferID: "t1", Basis: "annotation", Evidence: "nixl-log"},
				{Attempt: a2, TransferID: "t2", Basis: "annotation", Evidence: "nixl-log"},
			},
		},
		movement: newMovementCase(),
		window:   true,
	}
}

func servingExport(t *testing.T, name, run string, at int64) *nvtelemetry.WorkloadExport {
	t.Helper()
	raw, err := os.ReadFile("../nvtelemetry/testdata/workload/" + name)
	if err != nil {
		t.Fatal(err)
	}
	s := nvtelemetry.WorkloadSource{Provider: "triton", Version: "25.06", RunID: run}
	e, err := nvtelemetry.ReadServingExposition(context.Background(), bytes.NewReader(raw), s, nvtelemetry.WorkloadLimits{})
	if err != nil {
		t.Fatal(err)
	}
	e, err = nvtelemetry.StampWorkloadSnapshot(e, at, "test_monotonic", "test_epoch")
	if err != nil {
		t.Fatal(err)
	}
	return &e
}

func (c *servingCase) input(t *testing.T) cudadiag.ServingInput {
	t.Helper()
	var b bytes.Buffer
	if err := nvtelemetry.WriteServingContextV2(&b, c.serving, nvtelemetry.WorkloadLimits{}); err != nil {
		t.Fatal(err)
	}
	in := cudadiag.ServingInput{Serving: &b, Source: nvtelemetry.WorkloadSource{Provider: "application", Version: "serving-app-1", RunID: "run"}}
	if c.movement != nil {
		m := c.movement.input(t)
		in.Movement = &m
	}
	if c.window {
		in.Before, in.After = servingExport(t, "triton-before.prom", "run", 0), servingExport(t, "triton-after.prom", "run", 100)
		in.WindowStartNanos, in.WindowEndNanos, in.WindowClockDomain, in.WindowClockEpoch = 0, 100, "test_monotonic", "test_epoch"
	}
	return in
}

func summarizeServing(t *testing.T, c *servingCase) cudadiag.ServingSummary {
	t.Helper()
	s, err := cudadiag.SummarizeServing(context.Background(), c.input(t))
	if err != nil {
		t.Fatal(err)
	}
	return s
}

func withheldEdges(t *testing.T, s cudadiag.ServingSummary) []string {
	t.Helper()
	var out []string
	for _, e := range s.Edges {
		if _, err := e.Finding.JSON(); err != nil {
			t.Fatalf("edge %s: %v", e.Finding.ID, err)
		}
		if !e.Finding.Admission.Admitted {
			out = append(out, e.Finding.ID)
		}
	}
	sort.Strings(out)
	return out
}

func servingEdge(s cudadiag.ServingSummary, id string) cudadiag.ServingEdge {
	for _, e := range s.Edges {
		if e.Finding.ID == id {
			return e
		}
	}
	return cudadiag.ServingEdge{}
}

func TestServingSummarySharedBatchAndTransfers(t *testing.T) {
	s := summarizeServing(t, newServingCase())
	if got := withheldEdges(t, s); len(got) != 0 {
		t.Fatalf("withheld %v", got)
	}
	if s.Outcomes.Accepted != 1 || s.Outcomes.Failed != 1 || s.Attempts[0].Outcome != "accepted" || s.Attempts[1].Outcome != "failed" || *s.Attempts[0].OutputTokens != 7 {
		t.Fatalf("per-attempt outcomes lost: %+v", s.Attempts)
	}
	for i, want := range [][]string{{"t1"}, {"t2"}} {
		if !reflect.DeepEqual(s.Attempts[i].Batches, []string{"b1"}) || !reflect.DeepEqual(s.Attempts[i].Transfers, want) {
			t.Fatalf("attempt %d edges %+v", i, s.Attempts[i])
		}
	}
	for id, want := range map[string]tracecontract.FindingClass{
		"serving-route-0":          tracecontract.FindingObserved,
		"serving-batch-0-member-0": tracecontract.FindingAssociated,
		"serving-batch-0-op-0":     tracecontract.FindingAssociated,
		"serving-cache-0":          tracecontract.FindingAssociated,
		"serving-transfer-1":       tracecontract.FindingAssociated,
	} {
		if e := servingEdge(s, id); e.Finding.Class != want {
			t.Errorf("%s class %q, want %q", id, e.Finding.Class, want)
		}
	}
	if len(s.Aggregate) == 0 || len(s.AggregateRaw) != 2 || s.AggregateRaw[0] == s.AggregateRaw[1] {
		t.Fatal("aggregate window or raw digests not retained")
	}
	if s.ContextSHA256 == "" || s.Movement == nil || s.Movement.ContextSHA256 == "" || len(s.Movement.Activities.Joins) == 0 {
		t.Fatal("consumed movement and activity analyses not retained")
	}
	if e := servingEdge(s, "serving-transfer-1"); len(e.Finding.Artifacts) != 1 || e.Finding.Artifacts[0] != s.Movement.Context.Transfers[1].RawSHA256 || e.Evidence != "nixl-log" {
		t.Fatalf("transfer edge lineage %+v", e)
	}
	if e := servingEdge(s, "serving-batch-0-op-0"); len(e.Finding.Artifacts) != 1 || e.Finding.Artifacts[0] != s.Movement.Activities.Joins[0].EvidenceRef {
		t.Fatalf("operation edge lineage %+v", e)
	}
	// Serving edges and attempt summaries carry no numeric device share.
	for _, e := range s.Edges {
		if e.Finding.Value != nil {
			t.Fatalf("edge %s assigns a value", e.Finding.ID)
		}
	}
}

func TestServingSummaryWithholdsOnlyAffectedEdges(t *testing.T) {
	tests := []struct {
		name string
		edit func(*servingCase)
		want []string
	}{
		{"missing member request", func(c *servingCase) { c.serving.Batches[0].Members[1].RequestID = "" }, []string{"serving-batch-0-member-1"}},
		{"unresolved member", func(c *servingCase) { c.serving.Batches[0].Members[1].Attempt = "a2" }, []string{"serving-batch-0-member-1"}},
		{"swapped batch worker epoch", func(c *servingCase) { c.serving.Batches[0].WorkerEpoch = "e2" }, []string{"serving-batch-0-member-0", "serving-batch-0-member-1"}},
		{"role mismatch", func(c *servingCase) { c.serving.Batches[0].Role = "prefill" }, []string{"serving-batch-0-member-0", "serving-batch-0-member-1"}},
		{"swapped cache owner", func(c *servingCase) { c.serving.Caches[0].WorkerEpoch = "e2" }, []string{"serving-cache-0"}},
		{"undeclared cache generation", func(c *servingCase) { c.serving.CacheUses[0].Generation = "g2" }, []string{"serving-cache-0"}},
		{"transfer names other attempt", func(c *servingCase) { c.serving.Transfers[0].TransferID = "t2" }, []string{"serving-transfer-0"}},
		{"unknown transfer", func(c *servingCase) { c.serving.Transfers[1].TransferID = "t9" }, []string{"serving-transfer-1"}},
		{"missing transfer epoch", func(c *servingCase) { c.serving.Transfers[1].Attempt.WorkerEpoch = "" }, []string{"serving-transfer-1"}},
		{"unknown route basis", func(c *servingCase) { c.serving.Routes[0].Basis, c.serving.Routes[0].Evidence = "unknown", "" }, []string{"serving-route-0"}},
		{"missing batch activity", func(c *servingCase) { c.movement.annotations = nil }, []string{"serving-batch-0-op-0"}},
		{"missing batch process", func(c *servingCase) { c.serving.Batches[0].Process = nil }, []string{"serving-batch-0-op-0"}},
		{"other batch process", func(c *servingCase) { c.serving.Batches[0].Process = &cudaworkload.ProcessGenID{PID: 1, Generation: 3} }, []string{"serving-batch-0-op-0"}},
		{"request-only movement record", func(c *servingCase) { c.movement.context.Transfers[1].Attempt = "" }, nil},
		{"movement record other request", func(c *servingCase) {
			c.movement.context.Transfers[1].RequestID, c.movement.context.Transfers[1].Attempt = "r9", ""
		}, []string{"serving-transfer-1"}},
		{"ambiguous batch activity", func(c *servingCase) { c.movement.annotations[0].Context.Stream = 5 }, []string{"serving-batch-0-op-0"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := newServingCase()
			tt.edit(c)
			s := summarizeServing(t, c)
			if got := withheldEdges(t, s); !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("withheld %v, want %v", got, tt.want)
			}
			if s.Outcomes.Accepted != 1 || s.Outcomes.Failed != 1 || len(s.Aggregate) == 0 {
				t.Fatal("edge withholding changed outcomes or aggregate")
			}
		})
	}
}

func TestServingSummaryTransferStrength(t *testing.T) {
	for _, source := range []string{"imported", "native"} {
		c := newServingCase()
		c.movement.context.SourceClass = source
		c.serving.Transfers[1].Basis, c.serving.Transfers[1].Evidence = "native_id", movementSHA([]byte("nixl log"))
		s := summarizeServing(t, c)
		if e := servingEdge(s, "serving-transfer-1"); !e.Finding.Admission.Admitted || e.Finding.Class != tracecontract.FindingAssociated {
			t.Errorf("%s: transfer edge %s; movement request identity is an annotation", source, e.Finding.Class)
		}
		if e := servingEdge(s, "serving-batch-0-op-0"); e.Finding.Class != tracecontract.FindingAssociated {
			t.Errorf("%s: batch operation edge %s", source, e.Finding.Class)
		}
	}
	c := newServingCase()
	c.movement.context.SourceClass = "modeled"
	s := summarizeServing(t, c)
	for _, id := range []string{"serving-transfer-0", "serving-transfer-1"} {
		if e := servingEdge(s, id); e.Finding.Admission.Admitted {
			t.Errorf("modeled movement admitted %s", id)
		}
	}
	if len(s.Attempts[1].Transfers) != 0 {
		t.Error("modeled transfer attached to attempt")
	}
}

func TestServingSummaryWithoutMovement(t *testing.T) {
	c := newServingCase()
	c.movement = nil
	s := summarizeServing(t, c)
	want := []string{"serving-batch-0-op-0", "serving-transfer-0", "serving-transfer-1"}
	if got := withheldEdges(t, s); !reflect.DeepEqual(got, want) {
		t.Fatalf("withheld %v, want %v", got, want)
	}
	if s.Movement != nil || s.Outcomes.Accepted != 1 || len(s.Aggregate) == 0 {
		t.Fatal("summary without movement capture lost unrelated evidence")
	}
}

func TestServingSummaryAggregateOnly(t *testing.T) {
	c := newServingCase()
	c.serving.Routes, c.serving.Batches, c.serving.Caches, c.serving.CacheUses, c.serving.Transfers = []nvtelemetry.ServingRoute{}, []nvtelemetry.ServingBatch{}, []nvtelemetry.ServingCache{}, []nvtelemetry.ServingCacheUse{}, []nvtelemetry.ServingTransferRef{}
	s := summarizeServing(t, c)
	if len(s.Edges) != 0 || len(s.Aggregate) == 0 || s.Outcomes.Accepted != 1 {
		t.Fatal("aggregate summary lost without request edges")
	}
	for _, a := range s.Attempts {
		if len(a.Batches) != 0 || len(a.Transfers) != 0 {
			t.Fatal("request edges synthesized")
		}
	}
}

func TestServingSummaryRefusesInputs(t *testing.T) {
	foreignBefore, foreignAfter := servingExport(t, "triton-before.prom", "serving1", 0), servingExport(t, "triton-after.prom", "serving1", 100)
	tests := []struct {
		name string
		edit func(*cudadiag.ServingInput)
	}{
		{"nil serving", func(in *cudadiag.ServingInput) { in.Serving = nil }},
		{"run mismatch", func(in *cudadiag.ServingInput) { in.Movement.Source.RunID = "other" }},
		{"aggregate run", func(in *cudadiag.ServingInput) { in.Before, in.After = foreignBefore, foreignAfter }},
		{"half window", func(in *cudadiag.ServingInput) { in.After = nil }},
		{"window clock", func(in *cudadiag.ServingInput) { in.WindowClockEpoch = "other" }},
		{"movement", func(in *cudadiag.ServingInput) { in.Movement.Activities = nil }},
	}
	for _, tt := range tests {
		in := newServingCase().input(t)
		tt.edit(&in)
		if _, err := cudadiag.SummarizeServing(context.Background(), in); err == nil {
			t.Errorf("%s accepted", tt.name)
		}
	}
	c := newServingCase()
	var v1 bytes.Buffer
	if err := nvtelemetry.WriteServingContext(&v1, c.serving.V1(), nvtelemetry.WorkloadLimits{}); err != nil {
		t.Fatal(err)
	}
	in := c.input(t)
	in.Serving = &v1
	if _, err := cudadiag.SummarizeServing(context.Background(), in); err == nil {
		t.Error("version 1 context accepted as version 2")
	}
}

func TestServingSummaryArtifactLineage(t *testing.T) {
	native, owner, nixl := movementSHA([]byte("native log")), movementSHA([]byte("cache owner log")), movementSHA([]byte("nixl log"))
	tests := []struct {
		name string
		edit func(*servingCase)
		id   string
		want func(cudadiag.ServingSummary) []string
	}{
		{"native route", func(*servingCase) {}, "serving-route-0", func(s cudadiag.ServingSummary) []string { return []string{s.Context.Routes[0].Evidence} }},
		{"withheld native route", func(c *servingCase) { c.serving.Routes[0].Attempt.Attempt = "a9" }, "serving-route-0", func(cudadiag.ServingSummary) []string { return []string{} }},
		{"annotation member", func(*servingCase) {}, "serving-batch-0-member-0", func(cudadiag.ServingSummary) []string { return []string{} }},
		{"native member", func(c *servingCase) { c.serving.Batches[0].Basis, c.serving.Batches[0].Evidence = "native_id", native }, "serving-batch-0-member-1", func(cudadiag.ServingSummary) []string { return []string{native} }},
		{"native operation", func(c *servingCase) { c.serving.Batches[0].Basis, c.serving.Batches[0].Evidence = "native_id", native }, "serving-batch-0-op-0", func(s cudadiag.ServingSummary) []string {
			return []string{native, s.Movement.Activities.Joins[0].EvidenceRef}
		}},
		{"annotation cache owner label", func(*servingCase) {}, "serving-cache-0", func(cudadiag.ServingSummary) []string { return []string{} }},
		{"native cache owner label", func(c *servingCase) {
			c.serving.CacheUses[0].Basis, c.serving.CacheUses[0].Evidence = "native_id", native
		}, "serving-cache-0", func(cudadiag.ServingSummary) []string { return []string{native} }},
		{"annotation cache owner digest", func(c *servingCase) { c.serving.Caches[0].Evidence = owner }, "serving-cache-0", func(cudadiag.ServingSummary) []string { return []string{owner} }},
		{"native cache owner digest", func(c *servingCase) {
			c.serving.Caches[0].Evidence = owner
			c.serving.CacheUses[0].Basis, c.serving.CacheUses[0].Evidence = "native_id", native
		}, "serving-cache-0", func(cudadiag.ServingSummary) []string { return []string{native, owner} }},
		{"withheld cache owner digest", func(c *servingCase) {
			c.serving.Caches[0].Evidence, c.serving.Caches[0].WorkerEpoch = owner, "e2"
			c.serving.CacheUses[0].Basis, c.serving.CacheUses[0].Evidence = "native_id", native
		}, "serving-cache-0", func(cudadiag.ServingSummary) []string { return []string{} }},
		{"native transfer", func(c *servingCase) {
			c.serving.Transfers[1].Basis, c.serving.Transfers[1].Evidence = "native_id", nixl
		}, "serving-transfer-1", func(s cudadiag.ServingSummary) []string {
			return []string{nixl, s.Movement.Context.Transfers[1].RawSHA256}
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := newServingCase()
			tt.edit(c)
			s := summarizeServing(t, c)
			e := servingEdge(s, tt.id)
			if e.Finding.ID == "" {
				t.Fatalf("edge %s missing", tt.id)
			}
			if got, want := e.Finding.Artifacts, tt.want(s); !reflect.DeepEqual(got, want) {
				t.Fatalf("%s artifacts %v, want %v (admitted %v)", tt.id, got, want, e.Finding.Admission.Admitted)
			}
			// The extracted finding keeps its lineage without the edge.
			raw, err := e.Finding.JSON()
			if err != nil {
				t.Fatal(err)
			}
			for _, a := range e.Finding.Artifacts {
				if !bytes.Contains(raw, []byte(a)) {
					t.Fatalf("finding JSON lost artifact %s", a)
				}
			}
		})
	}
}

func TestServingSummaryRefusesDuplicateCacheUse(t *testing.T) {
	c := newServingCase()
	c.serving.CacheUses = append(c.serving.CacheUses, c.serving.CacheUses[0])
	raw, err := json.Marshal(c.serving)
	if err != nil {
		t.Fatal(err)
	}
	in := newServingCase().input(t)
	in.Serving = bytes.NewReader(raw)
	if _, err := cudadiag.SummarizeServing(context.Background(), in); err == nil {
		t.Fatal("duplicate cache use admitted twice")
	}
}
