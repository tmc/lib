package nvtelemetry_test

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func movementSource() nvtelemetry.WorkloadSource {
	return nvtelemetry.WorkloadSource{Provider: "application", Version: "movement-context-v1", RunID: "run"}
}

func movementDigest(s string) string {
	h := sha256.Sum256([]byte(s))
	return hex.EncodeToString(h[:])
}

func movementFixture() nvtelemetry.MovementContext {
	n := uint64(1<<53 + 1)
	enq, start, done, unc := int64(100), int64(110), int64(200), int64(5)
	return nvtelemetry.MovementContext{
		Version: 1, RunID: "run", Producer: "app", ProducerVersion: "1", ProducerSchema: "app.movement.v1", SourceClass: "imported",
		LossByFamily: map[string]tracecontract.Loss{"cuda_memcpy": {Total: 1}},
		Transfers: []nvtelemetry.MovementTransfer{{
			ID: "t1", Family: "cuda_memcpy", RawSHA256: movementDigest("raw t1"), RequestID: "r1", Attempt: "a1", OpID: 9,
			Source:      nvtelemetry.MovementEndpoint{Entity: "GPU-0", Kind: "device"},
			Destination: nvtelemetry.MovementEndpoint{Entity: "GPU-1", Kind: "device"},
			Direction:   "peer_to_peer", Bytes: &n, Unit: "bytes", State: "completed",
			EnqueueNanos: &enq, StartNanos: &start, CompleteNanos: &done,
			ClockDomain: "monotonic", ClockEpoch: "boot", UncertaintyNS: &unc, TopologyEpoch: "topo",
			Path: nvtelemetry.MovementPath{Mechanism: "nvlink", Basis: "independent_artifact", Evidence: movementDigest("path")},
		}},
	}
}

func TestMovementContextRoundTrip(t *testing.T) {
	var b bytes.Buffer
	c := movementFixture()
	if err := nvtelemetry.WriteMovementContext(&b, c, nvtelemetry.WorkloadLimits{}); err != nil {
		t.Fatal(err)
	}
	e, got, err := nvtelemetry.ReadMovementContext(bytes.NewReader(b.Bytes()), movementSource(), nvtelemetry.WorkloadLimits{})
	if err != nil {
		t.Fatal(err)
	}
	if *got.Transfers[0].Bytes != 1<<53+1 || got.Transfers[0].OpID != 9 || got.Transfers[0].Attempt != "a1" {
		t.Fatalf("exact transfer identity lost: %+v", got.Transfers[0])
	}
	h := sha256.Sum256(b.Bytes())
	if e.RawSHA256 != hex.EncodeToString(h[:]) || !bytes.Equal(e.Raw, b.Bytes()) {
		t.Fatal("raw lineage lost")
	}
	var again bytes.Buffer
	if err := nvtelemetry.WriteMovementContext(&again, got, nvtelemetry.WorkloadLimits{}); err != nil || !bytes.Equal(again.Bytes(), b.Bytes()) {
		t.Fatalf("round trip changed bytes: %v", err)
	}
}

func TestMovementContextRefusals(t *testing.T) {
	var b bytes.Buffer
	if err := nvtelemetry.WriteMovementContext(&b, movementFixture(), nvtelemetry.WorkloadLimits{}); err != nil {
		t.Fatal(err)
	}
	good := b.String()
	if _, _, err := nvtelemetry.ReadMovementContext(strings.NewReader(good), movementSource(), nvtelemetry.WorkloadLimits{}); err != nil {
		t.Fatalf("control fixture refused: %v", err)
	}
	raw := []struct {
		name, data string
	}{
		{"future version", strings.Replace(good, `"version":1`, `"version":2`, 1)},
		{"duplicate key", strings.Replace(good, `"run_id":"run"`, `"run_id":"run","run_id":"run"`, 1)},
		{"unknown field", strings.Replace(good, `"run_id":"run"`, `"run_id":"run","extra":1`, 1)},
		{"fractional bytes", strings.Replace(good, `"bytes":9007199254740993`, `"bytes":1.5`, 1)},
		{"negative bytes", strings.Replace(good, `"bytes":9007199254740993`, `"bytes":-1`, 1)},
		{"scaled unit", strings.Replace(good, `"unit":"bytes"`, `"unit":"KiB"`, 1)},
		{"trailing", good + "{}"},
	}
	for _, tt := range raw {
		if tt.data == good {
			t.Fatalf("%s mutation did not apply", tt.name)
		}
		if _, _, err := nvtelemetry.ReadMovementContext(strings.NewReader(tt.data), movementSource(), nvtelemetry.WorkloadLimits{}); err == nil {
			t.Errorf("%s accepted", tt.name)
		}
	}
	mutations := []struct {
		name string
		edit func(*nvtelemetry.MovementContext)
	}{
		{"duplicate transfer", func(c *nvtelemetry.MovementContext) { c.Transfers = append(c.Transfers, c.Transfers[0]) }},
		{"missing raw lineage", func(c *nvtelemetry.MovementContext) { c.Transfers[0].RawSHA256 = "" }},
		{"missing family", func(c *nvtelemetry.MovementContext) { c.Transfers[0].Family = "" }},
		{"attempt without request", func(c *nvtelemetry.MovementContext) { c.Transfers[0].RequestID = "" }},
		{"completion without completed state", func(c *nvtelemetry.MovementContext) { c.Transfers[0].State = "enqueued" }},
		{"completed without completion time", func(c *nvtelemetry.MovementContext) { c.Transfers[0].CompleteNanos = nil }},
		{"unordered time", func(c *nvtelemetry.MovementContext) { v := int64(300); c.Transfers[0].StartNanos = &v }},
		{"times without clock", func(c *nvtelemetry.MovementContext) { c.Transfers[0].ClockEpoch = "" }},
		{"bytes without unit", func(c *nvtelemetry.MovementContext) { c.Transfers[0].Unit = "" }},
		{"progress beyond bytes", func(c *nvtelemetry.MovementContext) {
			p, n := uint64(1<<60), int64(150)
			c.Transfers[0].ProgressBytes, c.Transfers[0].ProgressNanos = &p, &n
		}},
		{"path without evidence", func(c *nvtelemetry.MovementContext) { c.Transfers[0].Path.Evidence = "" }},
		{"unknown path with evidence", func(c *nvtelemetry.MovementContext) { c.Transfers[0].Path.Mechanism = "unknown" }},
		{"invented mechanism", func(c *nvtelemetry.MovementContext) { c.Transfers[0].Path.Mechanism = "vclock" }},
		{"endpoint kind", func(c *nvtelemetry.MovementContext) { c.Transfers[0].Source.Kind = "gpu" }},
		{"direction", func(c *nvtelemetry.MovementContext) { c.Transfers[0].Direction = "sideways" }},
		{"source class", func(c *nvtelemetry.MovementContext) { c.SourceClass = "vendor" }},
		{"loss", func(c *nvtelemetry.MovementContext) {
			c.LossByFamily["cuda_memcpy"] = tracecontract.Loss{Total: 1, Dropped: 2}
		}},
		{"null transfers", func(c *nvtelemetry.MovementContext) { c.Transfers = nil }},
		{"loss total below transfers", func(c *nvtelemetry.MovementContext) { c.LossByFamily["cuda_memcpy"] = tracecontract.Loss{} }},
		{"invalid utf-8 id", func(c *nvtelemetry.MovementContext) { c.Transfers[0].ID = "a\xff" }},
	}
	for _, tt := range mutations {
		c := movementFixture()
		tt.edit(&c)
		if err := nvtelemetry.WriteMovementContext(&bytes.Buffer{}, c, nvtelemetry.WorkloadLimits{}); err == nil {
			t.Errorf("%s written", tt.name)
		}
	}
	if _, _, err := nvtelemetry.ReadMovementContext(strings.NewReader(good), nvtelemetry.WorkloadSource{Provider: "application", Version: "operator-context-v1", RunID: "run"}, nvtelemetry.WorkloadLimits{}); err == nil {
		t.Error("wrong source schema accepted")
	}
	if _, _, err := nvtelemetry.ReadMovementContext(strings.NewReader(good), nvtelemetry.WorkloadSource{Provider: "application", Version: "movement-context-v1", RunID: "other"}, nvtelemetry.WorkloadLimits{}); err == nil {
		t.Error("wrong run accepted")
	}
	if err := nvtelemetry.WriteMovementContext(&bytes.Buffer{}, movementFixture(), nvtelemetry.WorkloadLimits{MaxBytes: int64(len(good) - 1)}); err == nil {
		t.Error("byte budget ignored")
	}
}

func TestMovementContextOptionalFieldsStayAbsent(t *testing.T) {
	c := movementFixture()
	tr := &c.Transfers[0]
	tr.Bytes, tr.Unit, tr.RequestID, tr.Attempt, tr.OpID = nil, "", "", "", 0
	tr.State, tr.CompleteNanos = "enqueued", nil
	var b bytes.Buffer
	if err := nvtelemetry.WriteMovementContext(&b, c, nvtelemetry.WorkloadLimits{}); err != nil {
		t.Fatal(err)
	}
	for _, field := range []string{`"bytes"`, `"request_id"`, `"op_id"`, `"complete_nanos"`} {
		if strings.Contains(b.String(), field) {
			t.Errorf("absent %s synthesized", field)
		}
	}
	_, got, err := nvtelemetry.ReadMovementContext(&b, movementSource(), nvtelemetry.WorkloadLimits{})
	if err != nil || got.Transfers[0].Bytes != nil || got.Transfers[0].CompleteNanos != nil {
		t.Fatalf("unreported value became zero: %+v %v", got.Transfers[0], err)
	}
}
