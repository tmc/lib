package cudadiag_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudadiag"
	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func cpuSummaryInputs() (nvtelemetry.OperatorCPUContext, nvtelemetry.WorkloadSource) {
	trace := []byte(`{"schemaVersion":1,"deviceProperties":[],"traceEvents":[{"name":"same","cat":"user_annotation","ph":"X","ts":1.25,"dur":2.5,"pid":1,"tid":2,"args":{"iteration":0}},{"name":"aten::add","cat":"cpu_op","ph":"X","ts":1,"dur":2},{"name":"PyTorch Profiler","cat":"Trace","ph":"X","ts":0,"dur":5},{"name":"process_name","ph":"M","pid":1,"args":{"name":"python"}}]}`)
	code := []byte("constructed C++ fact; no member relation")
	ref := nvtelemetry.OperatorReference{SHA256: summaryDigest(trace), Locator: "traceEvents"}
	c := nvtelemetry.OperatorCPUContext{Version: 1, RunID: "run", TraceSource: nvtelemetry.WorkloadSource{Provider: "pytorch", Version: "2.10.0", RunID: "run"}, Trace: ref,
		Artifacts: map[string][]byte{ref.SHA256: trace, summaryDigest(code): code}, CodeFacts: []nvtelemetry.OperatorReference{{SHA256: summaryDigest(code), Locator: "generated.cpp"}}}
	return c, nvtelemetry.WorkloadSource{Provider: "application", Version: "operator-cpu-context-v1", RunID: "run"}
}

func analyzeCPUContext(t *testing.T, c nvtelemetry.OperatorCPUContext, s nvtelemetry.WorkloadSource) cudadiag.OperatorCPUSummary {
	t.Helper()
	var wire bytes.Buffer
	if err := nvtelemetry.WriteOperatorCPUContext(&wire, c, nvtelemetry.WorkloadLimits{}); err != nil {
		t.Fatal(err)
	}
	out, err := cudadiag.AnalyzeOperatorCPUJSON(context.Background(), &wire, s, nvtelemetry.WorkloadLimits{})
	if err != nil {
		t.Fatal(err)
	}
	return out
}

func TestOperatorCPUSummaryProjection(t *testing.T) {
	c, s := cpuSummaryInputs()
	out := analyzeCPUContext(t, c, s)
	if out.Profile != "operator-cpu-summary-v1" || out.Version != 1 || out.ObservedCPUAnnotations != 1 || len(out.Annotations) != 1 {
		t.Fatal(out)
	}
	if out.NativeAnalysisStatus != "not_collected" || out.NativePopulation != nil || out.MembershipStatus != "unavailable" {
		t.Fatal("CPU annotations invented native population or membership", out)
	}
	a := out.Annotations[0]
	if a.Source.SHA256 != c.Trace.SHA256 || a.Source.Locator != "user_annotation/0" || a.Annotation.Name != "same" || a.Annotation.RawStart != "1.25" || a.Annotation.RawDuration != "2.5" || string(a.Annotation.PID) != "1" || string(a.Annotation.TID) != "2" || string(a.Annotation.Args) != `{"iteration":0}` || !reflect.DeepEqual(out.CodeFacts, c.CodeFacts) {
		t.Fatal("CPU annotation or separate raw references changed", out)
	}
	wire, err := json.Marshal(out)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(wire, []byte(`"native_population":null`)) {
		t.Fatal("unknown native population became an empty qualified population")
	}
	for _, field := range []string{`"op_id"`, `"member_id"`, `"edges"`, `"observed_kernel_population"`, `"process_generation"`} {
		if bytes.Contains(wire, []byte(field)) {
			t.Fatalf("invented native relation %s: %s", field, wire)
		}
	}
}

func TestOperatorCPUSummarySchemaAndCancel(t *testing.T) {
	c, source := cpuSummaryInputs()
	bad := bytes.Replace(c.Artifacts[c.Trace.SHA256], []byte(`"schemaVersion":1`), []byte(`"schemaVersion":2`), 1)
	delete(c.Artifacts, c.Trace.SHA256)
	c.Trace.SHA256 = summaryDigest(bad)
	c.Artifacts[c.Trace.SHA256] = bad
	var wire bytes.Buffer
	if err := nvtelemetry.WriteOperatorCPUContext(&wire, c, nvtelemetry.WorkloadLimits{}); err != nil {
		t.Fatal(err)
	}
	if _, err := cudadiag.AnalyzeOperatorCPUJSON(context.Background(), bytes.NewReader(wire.Bytes()), source, nvtelemetry.WorkloadLimits{}); err == nil {
		t.Fatal("CPU summary bypassed the existing PyTorch schema reader")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := cudadiag.AnalyzeOperatorCPUJSON(ctx, bytes.NewReader(wire.Bytes()), source, nvtelemetry.WorkloadLimits{}); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if _, err := cudadiag.AnalyzeOperatorCPUJSON(nil, &wire, source, nvtelemetry.WorkloadLimits{}); err == nil {
		t.Fatal("nil context accepted")
	}
	if _, err := cudadiag.AnalyzeOperatorCPUJSON(context.Background(), nil, source, nvtelemetry.WorkloadLimits{}); err == nil {
		t.Fatal("nil metadata accepted")
	}
}

func checkRetainedCPUProjection(t *testing.T, c nvtelemetry.OperatorCPUContext, count int) {
	t.Helper()
	source := nvtelemetry.WorkloadSource{Provider: "application", Version: "operator-cpu-context-v1", RunID: c.RunID}
	out := analyzeCPUContext(t, c, source)
	_, expected, err := nvtelemetry.ReadPyTorchAnnotations(bytes.NewReader(c.Artifacts[c.Trace.SHA256]), c.TraceSource, nvtelemetry.WorkloadLimits{})
	if err != nil || len(expected) != count || len(out.Annotations) != count || out.ObservedCPUAnnotations != uint64(count) || out.NativePopulation != nil || out.NativeAnalysisStatus != "not_collected" || out.MembershipStatus != "unavailable" {
		t.Fatal("retained CPU projection changed population or invented native membership", out, err)
	}
	if !reflect.DeepEqual(out.CodeFacts, c.CodeFacts) || !reflect.DeepEqual(out.Evidence, c.Evidence) {
		t.Fatal("separate code facts or evidence references changed")
	}
	compiled := 0
	for i, a := range out.Annotations {
		if !reflect.DeepEqual(a.Annotation, expected[i]) || a.Source.SHA256 != c.Trace.SHA256 || a.Source.Locator != "user_annotation/"+strconv.Itoa(i) {
			t.Fatal("retained annotation fields or raw ordinal changed", a)
		}
		if strings.Contains(a.Annotation.Name, "CompiledFxGraph") {
			compiled++
		}
	}
	if count == 4 && compiled != 2 {
		t.Fatal("genuine CompiledFxGraph annotations were dropped")
	}
}

func retainedCPUContext(t *testing.T, path, hash string) nvtelemetry.OperatorCPUContext {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil || summaryDigest(raw) != hash {
		t.Fatal("retained trace hash differs", err)
	}
	return nvtelemetry.OperatorCPUContext{Version: 1, RunID: "r122-cpu-eager-inductor-20261027", TraceSource: nvtelemetry.WorkloadSource{Provider: "pytorch", Version: "2.10.0", RunID: "r122-cpu-eager-inductor-20261027"}, Trace: nvtelemetry.OperatorReference{SHA256: hash, Locator: filepath.Base(path)}, Artifacts: map[string][]byte{hash: raw}}
}

// The actual A exports are committed fixtures: this required positive coverage
// runs in an ordinary checkout without private paths or environment settings.
func TestOperatorCPUSummaryRetainedExports(t *testing.T) {
	for _, tc := range []struct {
		file, hash string
		count      int
	}{
		{"pytorch-2.10.0-cpu-eager.chrome.json", "da43e72ecac5387566568d1f139af9b4aaefa3cb5861ce11ff5028e6e96cad47", 2},
		{"pytorch-2.10.0-cpu-inductor.chrome.json", "39a23a35df382fc29e27e3781849810c74d66d294ed3eac4a9bed401486f01ad", 4},
	} {
		t.Run(tc.file, func(t *testing.T) {
			c := retainedCPUContext(t, filepath.Join("..", "nvtelemetry", "testdata", tc.file), tc.hash)
			checkRetainedCPUProjection(t, c, tc.count)
		})
	}
}

// The compiler/application provenance remains an explicitly supplied external
// corpus. It is not invented as part of the portable annotation fixtures.
func TestOperatorCPUSummaryExternalCodeFacts(t *testing.T) {
	dir := os.Getenv("OPERATOR_CPU_FIXTURE_DIR")
	if dir == "" {
		t.Skip("external CPU compiler/application corpus not supplied")
	}
	code, err := os.ReadFile(os.Getenv("OPERATOR_CPU_CODE_FACT"))
	if err != nil {
		t.Fatal(err)
	}
	application, err := os.ReadFile(filepath.Join(dir, "application.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		file, hash string
		count      int
	}{
		{"eager.chrome.json", "da43e72ecac5387566568d1f139af9b4aaefa3cb5861ce11ff5028e6e96cad47", 2},
		{"torch_compile_inductor.chrome.json", "39a23a35df382fc29e27e3781849810c74d66d294ed3eac4a9bed401486f01ad", 4},
	} {
		t.Run(tc.file, func(t *testing.T) {
			c := retainedCPUContext(t, filepath.Join(dir, tc.file), tc.hash)
			c.Artifacts[summaryDigest(code)] = code
			c.Artifacts[summaryDigest(application)] = application
			c.CodeFacts = []nvtelemetry.OperatorReference{{SHA256: summaryDigest(code), Locator: "retained/generated.cpp"}}
			c.Evidence = []nvtelemetry.OperatorReference{{SHA256: summaryDigest(application), Locator: "application.jsonl"}}
			checkRetainedCPUProjection(t, c, tc.count)
		})
	}
}
