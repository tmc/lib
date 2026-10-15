package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cudadiag"
	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func commandOperatorDigest(raw []byte) string {
	sum := sha256.Sum256(raw)
	return hex.EncodeToString(sum[:])
}

func operatorCommandInputs(t *testing.T) ([]string, string, string) {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", "cudadiag", "testdata", "activity-v1.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	p := cudaworkload.ProcessGenID{PID: 1, Generation: 2}
	cs := cudaworkload.ContextStreamID{Context: 3, Stream: 4}
	r := tracecontract.Receipt{Version: 1, Command: "constructed", InputPath: "activities.jsonl", InputSHA256: commandOperatorDigest(raw), Kind: "capture", Backends: []tracecontract.Backend{{Name: "synthetic"}}, Gates: map[string]tracecontract.Gate{}, Partial: true}
	for _, name := range []string{"collector", "import", "workload", "stationarity", "shape", "correctness"} {
		r.Gates[name] = tracecontract.Gate{Disposition: "unavailable", Reason: "constructed", Provenance: "host control"}
	}
	memberRaw := []byte("constructed framework member records")
	ref := nvtelemetry.OperatorReference{SHA256: commandOperatorDigest(memberRaw), Locator: "record:1"}
	members := []nvtelemetry.OperatorMember{{ID: "fx:1", Name: "same", Schema: "constructed", Source: ref}, {ID: "fx:2", Name: "same", Schema: "constructed", Source: ref}}
	binding := nvtelemetry.OperatorBinding{Process: p, OpID: 9, Context: cs, CompilerBuild: "constructed"}
	inv, err := json.Marshal(nvtelemetry.OperatorInventory{Version: 1, RunID: "run", Binding: binding, Members: members})
	if err != nil {
		t.Fatal(err)
	}
	c := nvtelemetry.OperatorContext{Version: 1, RunID: "run", Producer: "constructed", ProducerVersion: "1", ProducerSchema: "host-control-v1", SourceClass: "synthetic", Receipt: r, Batch: cudaworkload.ActivityBatch{Session: "session", Run: "run", ClockEpoch: "epoch", EvidenceRef: r.InputSHA256, Process: p, ExternalKind: cupti.ExternalCorrelationKindCustom0}, Annotations: []cudaworkload.HostSubmission{{Process: p, OpID: 9, Symbol: "cuLaunchKernel", Workload: cudaworkload.WorkloadIdentity{Process: p, WorkloadName: "constructed", RunID: "run"}, Context: cs, Provenance: cudaworkload.ProvenanceCUPTI, EvidenceRef: "constructed annotation"}}, Artifacts: map[string][]byte{ref.SHA256: memberRaw, commandOperatorDigest(inv): inv}, Groups: []nvtelemetry.OperatorGroup{{ID: "fusion", Binding: binding, Members: members, InventorySource: &nvtelemetry.OperatorReference{SHA256: commandOperatorDigest(inv), Locator: "inventory"}}}}
	var metadata bytes.Buffer
	if err := nvtelemetry.WriteOperatorContext(&metadata, c, nvtelemetry.WorkloadLimits{}); err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	mp, ap := filepath.Join(dir, "context.json"), filepath.Join(dir, "activities.jsonl")
	if err := os.WriteFile(mp, metadata.Bytes(), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(ap, raw, 0600); err != nil {
		t.Fatal(err)
	}
	return []string{"-run-id", "run", mp, ap}, mp, ap
}

func TestOperatorSummaryCommandPublicDispatch(t *testing.T) {
	args, _, _ := operatorCommandInputs(t)
	command, ok := lookupCommand("operator-summary")
	if !ok {
		t.Fatal("command not registered")
	}
	output := captureStdout(t, func() error { return command.invoke(append([]string{"operator-summary"}, args...)) })
	var result cudadiag.OperatorSummary
	if err := json.Unmarshal([]byte(output), &result); err != nil {
		t.Fatal(err)
	}
	if result.KernelPopulation.Exact != 1 || result.MemberEdgePopulation.Associated != 2 || result.MemberEdgePopulation.Exact != 0 || result.Membership[0].Status != "complete_relative_to_inventory" || result.Context.SourceClass != "synthetic" {
		t.Fatal(result)
	}
	if !strings.Contains(usage, "cudaprof operator-summary") {
		t.Fatal("missing top-level usage")
	}
}

type operatorCommandShortWriter struct{}

func (operatorCommandShortWriter) Write(p []byte) (int, error) { return len(p) - 1, nil }

func TestOperatorSummaryCommandRefusals(t *testing.T) {
	args, mp, ap := operatorCommandInputs(t)
	for _, tc := range []struct {
		name string
		args []string
	}{
		{"missing run", []string{mp, ap}},
		{"wrong run", []string{"-run-id", "other", mp, ap}},
		{"input limit", append([]string{"-max-bytes", "1"}, args...)},
		{"output limit", append([]string{"-output-bytes", "1"}, args...)},
		{"record limit", append([]string{"-max-records", "1"}, args...)},
		{"directory", []string{"-run-id", "run", filepath.Dir(mp), ap}},
		{"missing input", []string{"-run-id", "run", mp, ap + "missing"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var out bytes.Buffer
			if err := runOperatorSummary(context.Background(), tc.args, &out); err == nil || out.Len() != 0 {
				t.Fatal(err, out.String())
			}
		})
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := runOperatorSummary(ctx, args, io.Discard); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if err := runOperatorSummary(context.Background(), args, operatorCommandShortWriter{}); !errors.Is(err, io.ErrShortWrite) {
		t.Fatal(err)
	}
	if err := os.WriteFile(ap, []byte("changed\n"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := runOperatorSummary(context.Background(), args, io.Discard); err == nil {
		t.Fatal("changed activity artifact admitted")
	}
}
