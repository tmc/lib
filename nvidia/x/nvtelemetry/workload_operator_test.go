package nvtelemetry_test

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func operatorContextFixture() nvtelemetry.OperatorContext {
	p := cudaworkload.ProcessGenID{PID: 1, Generation: 2}
	r := tracecontract.Receipt{Version: 1, Command: "constructed", InputPath: "activities.jsonl", InputSHA256: strings.Repeat("a", 64), Kind: "capture", Backends: []tracecontract.Backend{{Name: "synthetic"}}, Gates: map[string]tracecontract.Gate{}}
	for _, name := range []string{"collector", "import", "workload", "stationarity", "shape", "correctness"} {
		r.Gates[name] = tracecontract.Gate{Disposition: "unavailable", Reason: "constructed input", Provenance: "host control"}
	}
	return nvtelemetry.OperatorContext{Version: 1, RunID: "run", Producer: "constructed", ProducerVersion: "1", ProducerSchema: "host-control-v1", SourceClass: "synthetic", Batch: cudaworkload.ActivityBatch{Session: "s", Run: "run", ClockEpoch: "epoch", EvidenceRef: r.InputSHA256, Process: p}, Receipt: r, Groups: []nvtelemetry.OperatorGroup{}}
}

func operatorSource() nvtelemetry.WorkloadSource {
	return nvtelemetry.WorkloadSource{Provider: "application", Version: "operator-context-v1", RunID: "run"}
}

func operatorDigest(raw []byte) string {
	sum := sha256.Sum256(raw)
	return hex.EncodeToString(sum[:])
}

func TestOperatorContextPublicRoundTrip(t *testing.T) {
	c := operatorContextFixture()
	raw := []byte(`{"constructed":"member metadata"}`)
	ref := nvtelemetry.OperatorReference{SHA256: operatorDigest(raw), Locator: "record:1"}
	m := nvtelemetry.OperatorMember{ID: "native:1", Name: "same", Schema: "constructed-v1", Shapes: [][]int64{{2, -1}}, Source: ref}
	binding := nvtelemetry.OperatorBinding{Process: c.Batch.Process, OpID: 9, CompilerBuild: "declared"}
	inventory := nvtelemetry.OperatorInventory{Version: 1, RunID: "run", Binding: binding, Members: []nvtelemetry.OperatorMember{m}}
	inv, err := json.Marshal(inventory)
	if err != nil {
		t.Fatal(err)
	}
	c.Artifacts = map[string][]byte{ref.SHA256: raw, operatorDigest(inv): inv}
	c.Groups = []nvtelemetry.OperatorGroup{{ID: "fusion", Binding: binding, Members: []nvtelemetry.OperatorMember{m}, InventorySource: &nvtelemetry.OperatorReference{SHA256: operatorDigest(inv), Locator: "inventory:1"}}}
	var encoded bytes.Buffer
	if err := nvtelemetry.WriteOperatorContext(&encoded, c, nvtelemetry.WorkloadLimits{}); err != nil {
		t.Fatal(err)
	}
	export, got, err := nvtelemetry.ReadOperatorContext(bytes.NewReader(encoded.Bytes()), operatorSource(), nvtelemetry.WorkloadLimits{})
	if err != nil || got.Groups[0].Inventory == nil || got.Groups[0].Members[0].Shapes[0][1] != -1 || export.RawSHA256 != operatorDigest(encoded.Bytes()) || !bytes.Equal(export.Raw, encoded.Bytes()) {
		t.Fatal(got, err)
	}
	// Input bytes and mutable metadata do not alias the decoded result.
	c.Groups[0].Members[0].Shapes[0][0] = 99
	raw[0] = 'X'
	encoded.Bytes()[0] = 'X'
	if got.Groups[0].Members[0].Shapes[0][0] != 2 || got.Artifacts[ref.SHA256][0] != '{' || export.Raw[0] != '{' {
		t.Fatal("caller mutation rewrote retained metadata")
	}
}

func TestOperatorContextRefusals(t *testing.T) {
	c := operatorContextFixture()
	var valid bytes.Buffer
	if err := nvtelemetry.WriteOperatorContext(&valid, c, nvtelemetry.WorkloadLimits{}); err != nil {
		t.Fatal(err)
	}
	// Separate version1 application formats must not accept one another.
	for _, legacy := range []string{`{"version":1,"run_id":"run","updates":[]}`, `{"version":1,"run_id":"run","attempts":[]}`} {
		if _, _, err := nvtelemetry.ReadOperatorContext(strings.NewReader(legacy), operatorSource(), nvtelemetry.WorkloadLimits{}); err == nil {
			t.Fatal("legacy application format accepted as operator context")
		}
	}
	if _, _, err := nvtelemetry.ReadTrainingContext(bytes.NewReader(valid.Bytes()), operatorSource(), nvtelemetry.WorkloadLimits{}); err == nil {
		t.Fatal("training reader accepted operator context")
	}
	if _, _, err := nvtelemetry.ReadServingContext(bytes.NewReader(valid.Bytes()), operatorSource(), nvtelemetry.WorkloadLimits{}); err == nil {
		t.Fatal("serving reader accepted operator context")
	}
	for _, tc := range []struct {
		name string
		raw  []byte
	}{
		{"duplicate", bytes.Replace(valid.Bytes(), []byte(`"version":1`), []byte(`"version":1,"version":1`), 1)},
		{"conflicting alias", bytes.Replace(valid.Bytes(), []byte(`"version":1`), []byte(`"version":1,"Version":2`), 1)},
		{"future", bytes.Replace(valid.Bytes(), []byte(`"version":1`), []byte(`"version":2`), 1)},
		{"unknown", append([]byte(`{"extra":true,`), valid.Bytes()[1:]...)},
		{"trailing", append(append([]byte(nil), valid.Bytes()...), []byte(`{}`)...)},
		{"truncated", valid.Bytes()[:len(valid.Bytes())-1]},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, _, err := nvtelemetry.ReadOperatorContext(bytes.NewReader(tc.raw), operatorSource(), nvtelemetry.WorkloadLimits{}); err == nil {
				t.Fatal("invalid context admitted")
			}
		})
	}
	for _, mutate := range []func(*nvtelemetry.OperatorContext){
		func(c *nvtelemetry.OperatorContext) { c.RunID = "other" },
		func(c *nvtelemetry.OperatorContext) { c.SourceClass = "native" },
		func(c *nvtelemetry.OperatorContext) {
			c.Artifacts = map[string][]byte{strings.Repeat("b", 64): []byte("wrong digest")}
		},
	} {
		copy := operatorContextFixture()
		mutate(&copy)
		raw, err := json.Marshal(copy)
		if err != nil {
			t.Fatal(err)
		}
		if _, _, err := nvtelemetry.ReadOperatorContext(bytes.NewReader(raw), operatorSource(), nvtelemetry.WorkloadLimits{}); err == nil {
			t.Fatal("contradictory context admitted")
		}
	}
	for _, limits := range []nvtelemetry.WorkloadLimits{{MaxBytes: 1}, {MaxMetrics: 1}, {MaxStringBytes: 1}, {MaxBytes: -1}, {MaxMetrics: 65537}} {
		if err := nvtelemetry.WriteOperatorContext(io.Discard, c, limits); err == nil {
			t.Fatal("invalid/bounded writer admitted", limits)
		}
	}
}

type operatorShortWriter struct{}

func (operatorShortWriter) Write(p []byte) (int, error) { return len(p) - 1, nil }

type operatorErrorWriter struct{ err error }

func (w operatorErrorWriter) Write([]byte) (int, error) { return 0, w.err }

func TestOperatorContextWriterErrors(t *testing.T) {
	c := operatorContextFixture()
	if err := nvtelemetry.WriteOperatorContext(operatorShortWriter{}, c, nvtelemetry.WorkloadLimits{}); !errors.Is(err, io.ErrShortWrite) {
		t.Fatal(err)
	}
	failure := errors.New("caller write failure")
	if err := nvtelemetry.WriteOperatorContext(operatorErrorWriter{failure}, c, nvtelemetry.WorkloadLimits{}); !errors.Is(err, failure) {
		t.Fatal(err)
	}
}

func TestOperatorInventoryUnknownAndScopedIDs(t *testing.T) {
	c := operatorContextFixture()
	a, b := []byte("constructed A"), []byte("constructed B")
	members := []nvtelemetry.OperatorMember{{ID: "same", Source: nvtelemetry.OperatorReference{SHA256: operatorDigest(a), Locator: "1"}}, {ID: "same", Source: nvtelemetry.OperatorReference{SHA256: operatorDigest(b), Locator: "1"}}}
	future := []byte(`{"version":2,"run_id":"run","binding":{},"members":[]}`)
	c.Artifacts = map[string][]byte{operatorDigest(a): a, operatorDigest(b): b, operatorDigest(future): future}
	c.Groups = []nvtelemetry.OperatorGroup{{ID: "group", Binding: nvtelemetry.OperatorBinding{Process: c.Batch.Process, OpID: 1}, Members: members, InventorySource: &nvtelemetry.OperatorReference{SHA256: operatorDigest(future), Locator: "inventory"}}}
	var buf bytes.Buffer
	if err := nvtelemetry.WriteOperatorContext(&buf, c, nvtelemetry.WorkloadLimits{}); err != nil {
		t.Fatal(err)
	}
	_, got, err := nvtelemetry.ReadOperatorContext(&buf, operatorSource(), nvtelemetry.WorkloadLimits{})
	if err != nil || len(got.Groups[0].Members) != 2 || got.Groups[0].Inventory != nil || got.Groups[0].InventoryReason == "" {
		t.Fatal(got, err)
	}
}

func TestOperatorContextUnknownDoesNotBecomeZero(t *testing.T) {
	c := operatorContextFixture()
	c.Receipt.LostRecords = map[string]tracecontract.Loss{"activity": {}}
	raw := []byte("constructed member")
	c.Artifacts = map[string][]byte{operatorDigest(raw): raw}
	c.Groups = []nvtelemetry.OperatorGroup{{ID: "g", Binding: nvtelemetry.OperatorBinding{Process: c.Batch.Process, OpID: 1}, Members: []nvtelemetry.OperatorMember{{ID: "1", Source: nvtelemetry.OperatorReference{SHA256: operatorDigest(raw), Locator: "1"}, Shapes: [][]int64{{0}}}}}}
	var valid bytes.Buffer
	if err := nvtelemetry.WriteOperatorContext(&valid, c, nvtelemetry.WorkloadLimits{}); err != nil {
		t.Fatal(err)
	}
	if _, _, err := nvtelemetry.ReadOperatorContext(bytes.NewReader(valid.Bytes()), operatorSource(), nvtelemetry.WorkloadLimits{}); err != nil {
		t.Fatal("explicit measured zero refused", err)
	}
	for _, tc := range []struct{ name, from, to string }{
		{"null dimension", `"shapes":[[0]]`, `"shapes":[[null]]`},
		{"null shape", `"shapes":[[0]]`, `"shapes":[null]`},
		{"null loss", `"activity":{"dropped":0,"total":0}`, `"activity":null`},
		{"missing loss counts", `"activity":{"dropped":0,"total":0}`, `"activity":{}`},
		{"null loss count", `"activity":{"dropped":0,"total":0}`, `"activity":{"dropped":null,"total":0}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if !bytes.Contains(valid.Bytes(), []byte(tc.from)) {
				t.Fatal("fixture preimage absent")
			}
			changed := bytes.Replace(valid.Bytes(), []byte(tc.from), []byte(tc.to), 1)
			if _, _, err := nvtelemetry.ReadOperatorContext(bytes.NewReader(changed), operatorSource(), nvtelemetry.WorkloadLimits{}); err == nil {
				t.Fatal("unknown became zero")
			}
		})
	}
}
