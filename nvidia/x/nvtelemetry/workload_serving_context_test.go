package nvtelemetry_test

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaworkload"
	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func servingSource() nvtelemetry.WorkloadSource {
	return nvtelemetry.WorkloadSource{Provider: "application", Version: "serving-app-1", RunID: "run"}
}

func servingContextV2Fixture() nvtelemetry.ServingContextV2 {
	tokens := uint64(1<<53 + 1)
	a1 := nvtelemetry.ServingAttemptRef{RequestID: "r1", Attempt: "a1", Worker: "decode-0", WorkerEpoch: "e1"}
	a2 := nvtelemetry.ServingAttemptRef{RequestID: "r2", Attempt: "a1", Worker: "decode-0", WorkerEpoch: "e1"}
	return nvtelemetry.ServingContextV2{
		Version: 2, RunID: "run",
		Attempts: []nvtelemetry.ServingAttempt{
			{RequestID: "r1", Attempt: "a1", Worker: "decode-0", WorkerEpoch: "e1", Role: "decode", Outcome: "accepted", OutputTokens: &tokens, OutputPolicy: "slo-v1", Evidence: "log-1"},
			{RequestID: "r2", Attempt: "a1", Worker: "decode-0", WorkerEpoch: "e1", Role: "decode", Outcome: "failed", Evidence: "log-2"},
		},
		Routes:    []nvtelemetry.ServingRoute{{Attempt: a1, Frontend: "fe-0", Decision: "routed", Basis: "native_id", Evidence: movementDigest("router log")}},
		Batches:   []nvtelemetry.ServingBatch{{ID: "b1", Worker: "decode-0", WorkerEpoch: "e1", Role: "decode", Members: []nvtelemetry.ServingAttemptRef{a1, a2}, OpIDs: []cudaworkload.OpID{9}, Basis: "annotation", Evidence: "batch-log"}},
		Caches:    []nvtelemetry.ServingCache{{Namespace: "kv", Generation: "g1", Worker: "decode-0", WorkerEpoch: "e1", Evidence: "cache-log"}},
		CacheUses: []nvtelemetry.ServingCacheUse{{Attempt: a1, Namespace: "kv", Generation: "g1", Event: "hit", Basis: "annotation", Evidence: "cache-log"}},
		Transfers: []nvtelemetry.ServingTransferRef{{Attempt: a1, TransferID: "t2", Basis: "annotation", Evidence: "nixl-log"}},
	}
}

func TestServingContextV2RoundTrip(t *testing.T) {
	var b bytes.Buffer
	c := servingContextV2Fixture()
	if err := nvtelemetry.WriteServingContextV2(&b, c, nvtelemetry.WorkloadLimits{}); err != nil {
		t.Fatal(err)
	}
	e, got, err := nvtelemetry.ReadServingContextV2(bytes.NewReader(b.Bytes()), servingSource(), nvtelemetry.WorkloadLimits{})
	if err != nil {
		t.Fatal(err)
	}
	h := sha256.Sum256(b.Bytes())
	if e.RawSHA256 != hex.EncodeToString(h[:]) || *got.Attempts[0].OutputTokens != 1<<53+1 || len(got.Batches[0].Members) != 2 || got.Batches[0].OpIDs[0] != 9 {
		t.Fatalf("v2 identity lost: %+v", got)
	}
	var again bytes.Buffer
	if err := nvtelemetry.WriteServingContextV2(&again, got, nvtelemetry.WorkloadLimits{}); err != nil || !bytes.Equal(again.Bytes(), b.Bytes()) {
		t.Fatalf("v2 round trip changed bytes: %v", err)
	}
	if _, _, err := nvtelemetry.ReadServingContext(bytes.NewReader(b.Bytes()), servingSource(), nvtelemetry.WorkloadLimits{}); err == nil {
		t.Fatal("version 1 reader accepted version 2")
	}
}

func TestServingContextV1Unchanged(t *testing.T) {
	c := servingContextV2Fixture().V1()
	var b bytes.Buffer
	if err := nvtelemetry.WriteServingContext(&b, c, nvtelemetry.WorkloadLimits{}); err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(b.String(), `{"version":1,"run_id":"run","attempts":[`) || strings.Contains(b.String(), "routes") {
		t.Fatalf("v1 wire changed: %s", b.String())
	}
	_, got, err := nvtelemetry.ReadServingContext(bytes.NewReader(b.Bytes()), servingSource(), nvtelemetry.WorkloadLimits{})
	if err != nil {
		t.Fatal(err)
	}
	var again bytes.Buffer
	if err := nvtelemetry.WriteServingContext(&again, got, nvtelemetry.WorkloadLimits{}); err != nil || !bytes.Equal(again.Bytes(), b.Bytes()) {
		t.Fatal("v1 producer round trip changed")
	}
	if _, _, err := nvtelemetry.ReadServingContextV2(bytes.NewReader(b.Bytes()), servingSource(), nvtelemetry.WorkloadLimits{}); err == nil {
		t.Fatal("version 2 reader accepted version 1")
	}
	v2 := servingContextV2Fixture()
	v1 := v2.V1()
	v1.Attempts[0].Outcome = "unknown"
	if v2.Attempts[0].Outcome != "accepted" {
		t.Fatal("projection shares attempts")
	}
}

func TestServingContextV2Refusals(t *testing.T) {
	var b bytes.Buffer
	if err := nvtelemetry.WriteServingContextV2(&b, servingContextV2Fixture(), nvtelemetry.WorkloadLimits{}); err != nil {
		t.Fatal(err)
	}
	good := b.String()
	for _, tt := range []struct{ name, old, new string }{
		{"future version", `"version":2`, `"version":3`},
		{"duplicate key", `"run_id":"run"`, `"run_id":"run","run_id":"run"`},
		{"unknown field", `"run_id":"run"`, `"run_id":"run","extra":[]`},
		{"null extension", `"routes":[`, `"routes":null,"x":[`},
		{"absent extension", `"caches":[{"namespace":"kv","generation":"g1","worker":"decode-0","worker_epoch":"e1","evidence":"cache-log"}],`, ``},
		{"fractional op", `"op_ids":[9]`, `"op_ids":[9.5]`},
		{"negative op", `"op_ids":[9]`, `"op_ids":[-9]`},
	} {
		data := strings.Replace(good, tt.old, tt.new, 1)
		if data == good {
			t.Fatalf("%s mutation did not apply", tt.name)
		}
		if _, _, err := nvtelemetry.ReadServingContextV2(strings.NewReader(data), servingSource(), nvtelemetry.WorkloadLimits{}); err == nil {
			t.Errorf("%s accepted", tt.name)
		}
	}
	for _, tt := range []struct {
		name string
		edit func(*nvtelemetry.ServingContextV2)
	}{
		{"duplicate batch", func(c *nvtelemetry.ServingContextV2) { c.Batches = append(c.Batches, c.Batches[0]) }},
		{"duplicate member", func(c *nvtelemetry.ServingContextV2) {
			c.Batches[0].Members = append(c.Batches[0].Members, c.Batches[0].Members[0])
		}},
		{"zero op", func(c *nvtelemetry.ServingContextV2) { c.Batches[0].OpIDs = []cudaworkload.OpID{0} }},
		{"duplicate cache generation", func(c *nvtelemetry.ServingContextV2) { c.Caches = append(c.Caches, c.Caches[0]) }},
		{"unknown basis with evidence", func(c *nvtelemetry.ServingContextV2) { c.Routes[0].Basis = "unknown" }},
		{"annotation without evidence", func(c *nvtelemetry.ServingContextV2) { c.Transfers[0].Evidence = "" }},
		{"invented basis", func(c *nvtelemetry.ServingContextV2) { c.CacheUses[0].Basis = "timestamp" }},
		{"route decision", func(c *nvtelemetry.ServingContextV2) { c.Routes[0].Decision = "load_balanced" }},
		{"native id without digest", func(c *nvtelemetry.ServingContextV2) { c.Routes[0].Evidence = "router-log" }},
		{"invalid utf-8 identity", func(c *nvtelemetry.ServingContextV2) { c.Routes[0].Frontend = "fe-\xff" }},
		{"invalid batch process", func(c *nvtelemetry.ServingContextV2) { c.Batches[0].Process = &cudaworkload.ProcessGenID{PID: 1} }},
		{"cache event", func(c *nvtelemetry.ServingContextV2) { c.CacheUses[0].Event = "warm" }},
		{"batch role", func(c *nvtelemetry.ServingContextV2) { c.Batches[0].Role = "mixed" }},
		{"duplicate transfer ref", func(c *nvtelemetry.ServingContextV2) { c.Transfers = append(c.Transfers, c.Transfers[0]) }},
		{"invalid v1 attempt", func(c *nvtelemetry.ServingContextV2) { c.Attempts[0].OutputPolicy = "" }},
		{"version 1", func(c *nvtelemetry.ServingContextV2) { c.Version = 1 }},
		{"nil routes", func(c *nvtelemetry.ServingContextV2) { c.Routes = nil }},
	} {
		c := servingContextV2Fixture()
		tt.edit(&c)
		if err := nvtelemetry.WriteServingContextV2(&bytes.Buffer{}, c, nvtelemetry.WorkloadLimits{}); err == nil {
			t.Errorf("%s written", tt.name)
		}
	}
	if err := nvtelemetry.WriteServingContextV2(&bytes.Buffer{}, servingContextV2Fixture(), nvtelemetry.WorkloadLimits{MaxBytes: int64(len(good) - 1)}); err == nil {
		t.Error("byte budget ignored")
	}
	if err := nvtelemetry.WriteServingContextV2(&bytes.Buffer{}, servingContextV2Fixture(), nvtelemetry.WorkloadLimits{MaxMetrics: 8}); err == nil {
		t.Error("record budget ignored")
	}
}

func TestServingContextV2RetainsMissingEdgeIdentity(t *testing.T) {
	c := servingContextV2Fixture()
	c.Batches[0].Members[1].RequestID = ""
	c.Transfers[0].Attempt.WorkerEpoch = ""
	var b bytes.Buffer
	if err := nvtelemetry.WriteServingContextV2(&b, c, nvtelemetry.WorkloadLimits{}); err != nil {
		t.Fatal(err)
	}
	_, got, err := nvtelemetry.ReadServingContextV2(&b, servingSource(), nvtelemetry.WorkloadLimits{})
	if err != nil || got.Batches[0].Members[1].RequestID != "" || got.Transfers[0].Attempt.WorkerEpoch != "" {
		t.Fatalf("missing edge identity synthesized or refused: %v", err)
	}
}

func TestServingContextV2CacheUseIdentity(t *testing.T) {
	for _, tt := range []struct {
		name string
		edit func(*nvtelemetry.ServingContextV2)
		ok   bool
	}{
		{"identical", func(c *nvtelemetry.ServingContextV2) { c.CacheUses = append(c.CacheUses, c.CacheUses[0]) }, false},
		{"other evidence", func(c *nvtelemetry.ServingContextV2) {
			u := c.CacheUses[0]
			u.Basis, u.Evidence = "native_id", movementDigest("cache log")
			c.CacheUses = append(c.CacheUses, u)
		}, false},
		{"other event", func(c *nvtelemetry.ServingContextV2) {
			u := c.CacheUses[0]
			u.Event = "store"
			c.CacheUses = append(c.CacheUses, u)
		}, true},
		{"other attempt", func(c *nvtelemetry.ServingContextV2) {
			u := c.CacheUses[0]
			u.Attempt.RequestID = "r2"
			c.CacheUses = append(c.CacheUses, u)
		}, true},
		{"other generation", func(c *nvtelemetry.ServingContextV2) {
			u := c.CacheUses[0]
			u.Generation = "g2"
			c.CacheUses = append(c.CacheUses, u)
		}, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			c := servingContextV2Fixture()
			tt.edit(&c)
			if err := nvtelemetry.WriteServingContextV2(&bytes.Buffer{}, c, nvtelemetry.WorkloadLimits{}); (err == nil) != tt.ok {
				t.Errorf("writer: %v, want ok %v", err, tt.ok)
			}
			raw, err := json.Marshal(c)
			if err != nil {
				t.Fatal(err)
			}
			if _, _, err := nvtelemetry.ReadServingContextV2(bytes.NewReader(raw), servingSource(), nvtelemetry.WorkloadLimits{}); (err == nil) != tt.ok {
				t.Errorf("reader: %v, want ok %v", err, tt.ok)
			}
		})
	}
}
