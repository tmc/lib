package nvtelemetry_test

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"net/http"
	"os"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func pytorch210Fixture(t *testing.T, mode string) []byte {
	t.Helper()
	data, err := os.ReadFile("testdata/pytorch-2.10.0-cpu-" + mode + ".chrome.json")
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func pytorch210Source() nvtelemetry.WorkloadSource {
	return nvtelemetry.WorkloadSource{Provider: "pytorch", Version: "2.10.0", RunID: "retained-cpu-run", Units: map[string]string{}}
}

func TestPyTorch210CapturedAnnotations(t *testing.T) {
	for _, tc := range []struct {
		mode, digest     string
		names            []string
		times, durations []string
	}{
		{"eager", "da43e72ecac5387566568d1f139af9b4aaefa3cb5861ce11ff5028e6e96cad47",
			[]string{"r122:r122-cpu-eager-inductor-20261027:eager:0", "r122:r122-cpu-eager-inductor-20261027:eager:1"},
			[]string{"2230438240646.515", "2230438241176.140"}, []string{"118.958", "25.625"}},
		{"inductor", "39a23a35df382fc29e27e3781849810c74d66d294ed3eac4a9bed401486f01ad",
			[]string{"r122:r122-cpu-eager-inductor-20261027:torch_compile_inductor:0", "## Call CompiledFxGraph None ##", "r122:r122-cpu-eager-inductor-20261027:torch_compile_inductor:1", "## Call CompiledFxGraph None ##"},
			[]string{"2230438249740.917", "2230438249868.095", "2230438250241.712", "2230438250267.122"}, []string{"162.710", "19.078", "45.696", "10.748"}},
	} {
		t.Run(tc.mode, func(t *testing.T) {
			data := pytorch210Fixture(t, tc.mode)
			sum := sha256.Sum256(data)
			if hex.EncodeToString(sum[:]) != tc.digest {
				t.Fatal("authentic producer fixture changed")
			}
			source := pytorch210Source()
			export, annotations, err := nvtelemetry.ReadPyTorchAnnotations(bytes.NewReader(data), source, nvtelemetry.WorkloadLimits{})
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(export.Raw, data) || export.RawSHA256 != tc.digest || !reflect.DeepEqual(export.Source, source) || len(export.Metrics) != 0 || len(annotations) != len(tc.names) {
				t.Fatalf("producer preservation: hash=%s source=%+v metrics=%d annotations=%d", export.RawSHA256, export.Source, len(export.Metrics), len(annotations))
			}
			// Independent raw-field readback, without the production helper or
			// interpreting native IDs, compiler membership, or producer clocks.
			var raw struct {
				Events []struct {
					Cat            string `json:"cat"`
					PID, TID, Args json.RawMessage
				} `json:"traceEvents"`
			}
			if err := json.Unmarshal(data, &raw); err != nil {
				t.Fatal(err)
			}
			i := 0
			for _, event := range raw.Events {
				if event.Cat != "user_annotation" {
					continue
				}
				got := annotations[i]
				if got.Name != tc.names[i] || got.RawStart != tc.times[i] || got.RawDuration != tc.durations[i] || !bytes.Equal(got.PID, event.PID) || !bytes.Equal(got.TID, event.TID) || !bytes.Equal(got.Args, event.Args) || string(got.PID) != "7768" || string(got.TID) != "675422" {
					t.Fatalf("raw annotation %d changed: %+v", i, got)
				}
				if got.StartNanos <= 0 || got.DurationNanos <= 0 || export.Source.ClockDomain != "" || export.Source.ClockEpochID != "" {
					t.Fatalf("unknown clock context or interval changed: %+v", got)
				}
				i++
			}
			if i != len(tc.names) {
				t.Fatal("annotation population changed")
			}
		})
	}
}

// These mutations are refusal controls, not substitute authentic captures.
func pytorch210Mutation(t *testing.T, change func(map[string]json.RawMessage)) []byte {
	t.Helper()
	var root map[string]json.RawMessage
	if err := json.Unmarshal(pytorch210Fixture(t, "eager"), &root); err != nil {
		t.Fatal(err)
	}
	change(root)
	data, err := json.Marshal(root)
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func TestPyTorch210SchemaControls(t *testing.T) {
	for _, tc := range []struct{ name, value string }{
		{"missing", ""}, {"null", "null"}, {"string", `"1"`}, {"newer", "2"}, {"fractional", "1.0"}, {"negative", "-1"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			data := pytorch210Mutation(t, func(root map[string]json.RawMessage) {
				if tc.value == "" {
					delete(root, "schemaVersion")
				} else {
					root["schemaVersion"] = json.RawMessage(tc.value)
				}
			})
			if _, _, err := nvtelemetry.ReadPyTorchAnnotations(bytes.NewReader(data), pytorch210Source(), nvtelemetry.WorkloadLimits{}); err == nil {
				t.Fatal("unsupported 2.10 trace schema admitted")
			}
		})
	}
	t.Run("duplicate", func(t *testing.T) {
		data := pytorch210Fixture(t, "eager")
		start := bytes.IndexByte(data, '{')
		data = append([]byte(`{"schemaVersion":1,`), data[start+1:]...)
		if _, _, err := nvtelemetry.ReadPyTorchAnnotations(bytes.NewReader(data), pytorch210Source(), nvtelemetry.WorkloadLimits{}); err == nil {
			t.Fatal("duplicate schema key admitted")
		}
	})
}

func TestPyTorch210CPUProfileControls(t *testing.T) {
	for _, tc := range []struct{ name, device, category string }{
		{"device_missing", "", ""}, {"device_null", "null", ""},
		{"device_nonempty", `[{"id":0}]`, ""}, {"device_string", `"cpu"`, ""}, {"device_object", `{}`, ""},
		{"category_kernel", "", "kernel"}, {"category_cuda_runtime", "", "cuda_runtime"}, {"category_unknown", "", "future_category"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			data := pytorch210Mutation(t, func(root map[string]json.RawMessage) {
				if tc.category == "" {
					if tc.device == "" {
						delete(root, "deviceProperties")
					} else {
						root["deviceProperties"] = json.RawMessage(tc.device)
					}
				} else {
					var events []map[string]json.RawMessage
					if err := json.Unmarshal(root["traceEvents"], &events); err != nil {
						t.Fatal(err)
					}
					events[0]["cat"], _ = json.Marshal(tc.category)
					var err error
					root["traceEvents"], err = json.Marshal(events)
					if err != nil {
						t.Fatal(err)
					}
				}
			})
			if _, _, err := nvtelemetry.ReadPyTorchAnnotations(bytes.NewReader(data), pytorch210Source(), nvtelemetry.WorkloadLimits{}); err == nil {
				t.Fatal("non-CPU or unknown 2.10 profile admitted")
			}
		})
	}
}

func TestPyTorch210PhaseLimitsAndVersions(t *testing.T) {
	data := pytorch210Mutation(t, func(root map[string]json.RawMessage) {
		var events []map[string]json.RawMessage
		if err := json.Unmarshal(root["traceEvents"], &events); err != nil {
			t.Fatal(err)
		}
		for _, event := range events {
			if string(event["cat"]) == `"user_annotation"` {
				event["ph"] = json.RawMessage(`"B"`)
				break
			}
		}
		var err error
		root["traceEvents"], err = json.Marshal(events)
		if err != nil {
			t.Fatal(err)
		}
	})
	if _, _, err := nvtelemetry.ReadPyTorchAnnotations(bytes.NewReader(data), pytorch210Source(), nvtelemetry.WorkloadLimits{}); err == nil || !strings.Contains(err.Error(), "annotation phase") {
		t.Fatalf("incomplete annotation error=%v", err)
	}
	data = pytorch210Fixture(t, "eager")
	for _, tc := range []struct {
		name   string
		limits nvtelemetry.WorkloadLimits
	}{
		{"bytes", nvtelemetry.WorkloadLimits{MaxBytes: int64(len(data) - 1)}},
		{"events", nvtelemetry.WorkloadLimits{MaxMetrics: 1833}},
		{"name", nvtelemetry.WorkloadLimits{MaxStringBytes: 32}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, _, err := nvtelemetry.ReadPyTorchAnnotations(bytes.NewReader(data), pytorch210Source(), tc.limits); err == nil {
				t.Fatal("reader limit ignored")
			}
		})
	}
	for _, version := range []string{"2.10.1", "2.9.0", "latest", ""} {
		t.Run("version_"+version, func(t *testing.T) {
			s := pytorch210Source()
			s.Version = version
			if _, _, err := nvtelemetry.ReadPyTorchAnnotations(bytes.NewReader(data), s, nvtelemetry.WorkloadLimits{}); err == nil {
				t.Fatal("unselected version admitted")
			}
		})
	}
}

func TestPyTorch210LegacyBoundary(t *testing.T) {
	data, err := os.ReadFile("testdata/workload/pytorch.json")
	if err != nil {
		t.Fatal(err)
	}
	// Legacy 2.7 previously ignored schemaVersion. Do not silently strengthen it.
	data = bytes.Replace(data, []byte(`"schemaVersion":1`), []byte(`"schemaVersion":"legacy-uninterpreted"`), 1)
	var root map[string]json.RawMessage
	if err := json.Unmarshal(data, &root); err != nil {
		t.Fatal(err)
	}
	root["deviceProperties"] = json.RawMessage(`[{"id":0}]`)
	var events []json.RawMessage
	if err := json.Unmarshal(root["traceEvents"], &events); err != nil {
		t.Fatal(err)
	}
	events = append(events, json.RawMessage(`{"name":"raw kernel","cat":"kernel","ph":"X"}`), json.RawMessage(`{"name":"raw flow","cat":"cuda_runtime","ph":"s","id":1}`))
	root["traceEvents"], err = json.Marshal(events)
	if err != nil {
		t.Fatal(err)
	}
	data, err = json.Marshal(root)
	if err != nil {
		t.Fatal(err)
	}
	s := pytorch210Source()
	s.Version = "2.7.0"
	if e, a, err := nvtelemetry.ReadPyTorchAnnotations(bytes.NewReader(data), s, nvtelemetry.WorkloadLimits{}); err != nil || len(a) != 1 || !bytes.Equal(e.Raw, data) {
		t.Fatalf("legacy behavior changed: annotations=%d error=%v", len(a), err)
	}
}

type pytorch210Transport func(*http.Request) (*http.Response, error)

func (f pytorch210Transport) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func TestPyTorch210NoHTTPAdmission(t *testing.T) {
	calls := 0
	client := &http.Client{Transport: pytorch210Transport(func(*http.Request) (*http.Response, error) { calls++; return nil, errors.New("unexpected transport") })}
	request, err := http.NewRequest("GET", "https://example.invalid/profile", nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := nvtelemetry.FetchWorkload(context.Background(), client, request, pytorch210Source(), nvtelemetry.WorkloadLimits{}); err == nil || calls != 0 {
		t.Fatalf("HTTP contract widened: calls=%d error=%v", calls, err)
	}
}
