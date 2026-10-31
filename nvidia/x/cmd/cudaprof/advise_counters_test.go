package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprof/advise"
)

func adviseCountersFixture(t *testing.T) ([]advise.CounterAction, []byte) {
	t.Helper()
	// These are CPU placeholders, not native measurements or authority.
	digest := strings.Repeat("a", 64)
	ref := advise.CounterReference{Locator: "CPU_SYNTHETIC", SHA256: digest}
	actions := []advise.CounterAction{{
		ID: "action", Kernel: "kernel", AcquisitionID: "acquisition",
		Identity: advise.CounterIdentity{
			SourceSelectionSHA256: digest, CandidateSourceSHA256: digest,
			CaseConfigSHA256: digest, ImageSHA256: digest, DeviceUUID: "CPU_SYNTHETIC",
			RawReport: ref, TypedCensus: ref, Admission: ref,
		},
		Descriptors: []cudacapture.MetricDescriptor{{
			ID: "duration", Name: "gpu__time_duration.sum", NumericType: "uint64",
			Unit: "ns", Scope: "ACTION", Denominator: "action", Aggregation: "sum",
		}},
		Observations: []cudacapture.Observation{{
			DescriptorID: "duration", EntityID: "action", RunID: "acquisition",
			Availability: "MEASURED", Value: json.RawMessage(`18446744073709551615`),
		}},
	}}
	data, err := json.Marshal(actions)
	if err != nil {
		t.Fatal(err)
	}
	return actions, data
}

func TestAdviseCountersJSONParity(t *testing.T) {
	actions, data := adviseCountersFixture(t)
	want, err := advise.AnalyzeCounters(actions)
	if err != nil {
		t.Fatal(err)
	}
	var out, repeated bytes.Buffer
	for _, dst := range []*bytes.Buffer{&out, &repeated} {
		if err := runAdviseCountersCommand([]string{"-json", "-"}, bytes.NewReader(data), dst); err != nil {
			t.Fatal(err)
		}
	}
	if out.String() != repeated.String() {
		t.Fatal("counter report is not deterministic")
	}
	var got advise.CounterReport
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(&got, want) {
		t.Fatal("CLI changed library counter report")
	}
	if !bytes.Contains(out.Bytes(), []byte(`18446744073709551615`)) {
		t.Fatal("lost exact uint64")
	}
	for _, finding := range got.Findings {
		if !finding.Item.NoRuntimeAttribution || finding.Item.Severity != advise.SeverityContext {
			t.Fatal("counter CLI invented runtime attribution")
		}
	}
}

func TestAdviseCountersFileTextAndDispatch(t *testing.T) {
	_, data := adviseCountersFixture(t)
	path := filepath.Join(t.TempDir(), "actions.json")
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	command, ok := lookupCommand("advise-counters")
	if !ok || command.name != "advise-counters" {
		t.Fatal("counter command not registered")
	}
	out := captureStdout(t, func() error { return command.invoke([]string{"advise-counters", path}) })
	for _, text := range []string{"Counter evidence: 1 actions, 1 findings", "Structural validation only", "producer guarantees", "no runtime attribution", "18446744073709551615", "domain ACTION"} {
		if !strings.Contains(out, text) {
			t.Errorf("missing %q in %s", text, out)
		}
	}
}

func TestAdviseCountersMalformedInputs(t *testing.T) {
	_, data := adviseCountersFixture(t)
	valid := string(data)
	tests := []struct {
		name, input, want string
	}{
		{"empty", "", "decode actions"},
		{"empty array", "[]", "at least one"},
		{"null", "null", "at least one"},
		{"object", "{}", "decode actions"},
		{"malformed", "[", "decode actions"},
		{"trailing value", valid + " []", "trailing JSON"},
		{"trailing garbage", valid + " x", "trailing JSON"},
		{"unknown field", strings.Replace(valid, `"id":"action"`, `"id":"action","typo":1`, 1), "unknown or case-alias"},
		{"duplicate field", strings.Replace(valid, `"id":"action"`, `"id":"action","id":"other"`, 1), "duplicate or case-alias"},
		{"case alias", strings.Replace(valid, `"id":"action"`, `"ID":"action"`, 1), "unknown or case-alias"},
		{"missing identity", strings.Replace(valid, `"device_uuid":"CPU_SYNTHETIC"`, `"device_uuid":""`, 1), "missing evidence identity"},
		{"wrong unit", strings.Replace(valid, `"unit":"ns"`, `"unit":"us"`, 1), "incompatible unit"},
		{"wrong domain", strings.Replace(valid, `"scope":"ACTION"`, `"scope":"THREAD"`, 1), "incompatible unit"},
		{"cross action", strings.Replace(valid, `"entity_id":"action"`, `"entity_id":"other"`, 1), "mismatched observation"},
		{"nonfinite JSON", strings.Replace(valid, "18446744073709551615", "NaN", 1), "decode actions"},
		{"overflow integer", strings.Replace(valid, "18446744073709551615", "18446744073709551616", 1), "invalid uint64"},
		{"extra grid dimension", strings.Replace(valid, `"kernel":"kernel"`, `"kernel":"kernel","launch":{"grid":[1,1,1,1],"source":{"locator":"CPU","sha256":"`+strings.Repeat("a", 64)+`"}}`, 1), "three dimensions"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var out bytes.Buffer
			err := runAdviseCountersCommand([]string{"-json", "-"}, strings.NewReader(tt.input), &out)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("got %v, want %q", err, tt.want)
			}
			if out.Len() != 0 {
				t.Fatalf("invalid input produced a report: %s", &out)
			}
		})
	}
}

func TestAdviseCountersOptions(t *testing.T) {
	_, data := adviseCountersFixture(t)
	tests := []struct {
		args []string
		want string
	}{
		{nil, "usage:"},
		{[]string{"-", "extra"}, "usage:"},
		{[]string{"-unknown", "-"}, "flag provided"},
		{[]string{"-max-bytes=0", "-"}, "byte budget"},
		{[]string{"-max-bytes=16777217", "-"}, "byte budget"},
		{[]string{"-max-bytes=1", "-"}, "budget exceeded"},
	}
	for _, tt := range tests {
		t.Run(strings.Join(tt.args, " "), func(t *testing.T) {
			err := runAdviseCountersCommand(tt.args, bytes.NewReader(data), io.Discard)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("got %v, want %q", err, tt.want)
			}
		})
	}
	var out bytes.Buffer
	if err := runAdviseCountersCommand([]string{"-h"}, nil, &out); err != nil || !strings.Contains(out.String(), "structur") {
		t.Fatalf("help: %v: %s", err, &out)
	}
}

type adviseCountersBrokenIO struct{}

func (adviseCountersBrokenIO) Read([]byte) (int, error)  { return 0, errors.New("read failure") }
func (adviseCountersBrokenIO) Write([]byte) (int, error) { return 0, errors.New("write failure") }

func TestAdviseCountersIOErrors(t *testing.T) {
	_, data := adviseCountersFixture(t)
	for _, tt := range []struct {
		name string
		args []string
		in   io.Reader
		out  io.Writer
		want string
	}{
		{"missing file", []string{"missing-actions.json"}, nil, io.Discard, "open input"},
		{"missing reader", []string{"-"}, nil, io.Discard, "reader unavailable"},
		{"read failure", []string{"-"}, adviseCountersBrokenIO{}, io.Discard, "read failure"},
		{"JSON write failure", []string{"-json", "-"}, bytes.NewReader(data), adviseCountersBrokenIO{}, "write failure"},
		{"text write failure", []string{"-"}, bytes.NewReader(data), adviseCountersBrokenIO{}, "write failure"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			err := runAdviseCountersCommand(tt.args, tt.in, tt.out)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("got %v, want %q", err, tt.want)
			}
		})
	}
}

func TestAdviseCountersUnknownRawValue(t *testing.T) {
	actions, _ := adviseCountersFixture(t)
	actions[0].Descriptors[0].Name = "future__unknown"
	actions[0].Descriptors[0].NumericType = "unknown"
	actions[0].Observations[0].Value = json.RawMessage(`{"native_payload":[1,"UNAVAILABLE",null]}`)
	data, err := json.Marshal(actions)
	if err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	if err := runAdviseCountersCommand([]string{"-json", "-"}, bytes.NewReader(data), &out); err != nil {
		t.Fatal(err)
	}
	var report advise.CounterReport
	if err := json.Unmarshal(out.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	var retained, original bytes.Buffer
	if err := json.Compact(&retained, report.Actions[0].Observations[0].Value); err != nil {
		t.Fatal(err)
	}
	if err := json.Compact(&original, actions[0].Observations[0].Value); err != nil {
		t.Fatal(err)
	}
	if len(report.Findings) != 0 || retained.String() != original.String() || report.Actions[0].Descriptors[0] != actions[0].Descriptors[0] {
		t.Fatal("unknown counter was interpreted or discarded")
	}
}

func TestAdviseCountersOffline(t *testing.T) {
	if os.Getenv("CUDAPROF_TEST_COUNTERS_OFFLINE") != "1" {
		library := ""
		switch runtime.GOOS {
		case "darwin":
			library = "/usr/lib/libSystem.B.dylib"
		case "linux":
			library = "libc.so.6"
		default:
			t.Skip("no inert system library fixture")
		}
		cmd := exec.Command(os.Args[0], "-test.run=^TestAdviseCountersOffline$", "-test.v")
		for _, entry := range os.Environ() {
			if !strings.HasPrefix(entry, "CUDA_LIB_PATH=") && !strings.HasPrefix(entry, "CUDAPROF_TEST_COUNTERS_OFFLINE=") {
				cmd.Env = append(cmd.Env, entry)
			}
		}
		cmd.Env = append(cmd.Env, "CUDAPROF_TEST_COUNTERS_OFFLINE=1", "CUDA_LIB_PATH="+library)
		out, err := cmd.CombinedOutput()
		if err != nil || !strings.Contains(string(out), "--- PASS: TestAdviseCountersOffline") {
			t.Fatalf("offline child: %v\n%s", err, out)
		}
		return
	}
	if driver.Loaded() {
		t.Fatal("CUDA loaded before counter analysis")
	}
	_, data := adviseCountersFixture(t)
	if err := runAdviseCountersCommand([]string{"-json", "-"}, bytes.NewReader(data), io.Discard); err != nil {
		t.Fatal(err)
	}
	if driver.Loaded() {
		t.Fatal("counter advice loaded CUDA")
	}
}
