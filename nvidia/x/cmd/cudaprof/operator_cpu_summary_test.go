package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudadiag"
	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func cpuOperatorCommandInputs(t *testing.T) ([]string, string) {
	t.Helper()
	raw := []byte(`{"schemaVersion":1,"deviceProperties":[],"traceEvents":[{"name":"annotation","cat":"user_annotation","ph":"X","ts":1,"dur":2,"pid":1,"tid":2},{"name":"aten::add","cat":"cpu_op","ph":"X","ts":1,"dur":2},{"name":"PyTorch Profiler","cat":"Trace","ph":"X","ts":0,"dur":5},{"name":"process_name","ph":"M","pid":1}]}`)
	hash := commandOperatorDigest(raw)
	c := nvtelemetry.OperatorCPUContext{Version: 1, RunID: "run", TraceSource: nvtelemetry.WorkloadSource{Provider: "pytorch", Version: "2.10.0", RunID: "run"}, Trace: nvtelemetry.OperatorReference{SHA256: hash, Locator: "traceEvents"}, Artifacts: map[string][]byte{hash: raw}}
	var wire bytes.Buffer
	if err := nvtelemetry.WriteOperatorCPUContext(&wire, c, nvtelemetry.WorkloadLimits{}); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "cpu-context.json")
	if err := os.WriteFile(path, wire.Bytes(), 0600); err != nil {
		t.Fatal(err)
	}
	return []string{"-profile", "operator-cpu-context-v1", "-run-id", "run", path}, path
}

func TestOperatorCPUCommandPublicDispatch(t *testing.T) {
	args, _ := cpuOperatorCommandInputs(t)
	command, ok := lookupCommand("operator-summary")
	if !ok {
		t.Fatal("existing operator-summary command missing")
	}
	output := captureStdout(t, func() error { return command.invoke(append([]string{"operator-summary"}, args...)) })
	var got cudadiag.OperatorCPUSummary
	if err := json.Unmarshal([]byte(output), &got); err != nil {
		t.Fatal(err)
	}
	if got.Profile != "operator-cpu-summary-v1" || got.ObservedCPUAnnotations != 1 || got.NativePopulation != nil || got.NativeAnalysisStatus != "not_collected" || got.MembershipStatus != "unavailable" {
		t.Fatal("CPU command invented native population or membership", got)
	}
}

func TestOperatorCPUCommandRefusals(t *testing.T) {
	args, path := cpuOperatorCommandInputs(t)
	for _, tc := range []struct {
		name string
		args []string
	}{
		{"native input", append(append([]string(nil), args...), path)},
		{"native default", []string{"-run-id", "run", path}},
		{"unknown profile", []string{"-profile", "other", "-run-id", "run", path}},
		{"wrong run", []string{"-profile", "operator-cpu-context-v1", "-run-id", "other", path}},
		{"input limit", append([]string{"-max-bytes", "1"}, args...)},
		{"output limit", append([]string{"-output-bytes", "1"}, args...)},
		{"directory", []string{"-profile", "operator-cpu-context-v1", "-run-id", "run", filepath.Dir(path)}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var out bytes.Buffer
			if err := runOperatorSummary(context.Background(), tc.args, &out); err == nil || out.Len() != 0 {
				t.Fatal("invalid CPU command emitted output", err, out.String())
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
}

func TestOperatorCPUCommandKeepsNativeDefault(t *testing.T) {
	args, _, _ := operatorCommandInputs(t)
	var oldDefault, explicitNative bytes.Buffer
	if err := runOperatorSummary(context.Background(), args, &oldDefault); err != nil {
		t.Fatal(err)
	}
	if err := runOperatorSummary(context.Background(), append([]string{"-profile", "operator-context-v1"}, args...), &explicitNative); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(oldDefault.Bytes(), explicitNative.Bytes()) {
		t.Fatal("default native summary bytes changed with explicit profile selection")
	}
}
