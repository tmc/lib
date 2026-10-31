package nvtelemetry_test

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func mlflowGaugeInput(t *testing.T) []byte {
	t.Helper()
	b, err := os.ReadFile("testdata/workload/mlflow-native.json")
	if err != nil {
		t.Fatal(err)
	}
	if fmt.Sprintf("%x", sha256.Sum256(b)) != "241bf82ed4d325c84e7b2c9591b1a583b24bed75d15a55d1fc9c0627ce30c5b5" {
		t.Fatal("native SDK response changed")
	}
	return b
}
func mlflowGaugeSource() nvtelemetry.WorkloadSource {
	return nvtelemetry.WorkloadSource{Provider: "mlflow", Version: "2.22.0", RunID: "4790069ba3984e409f20782b56bb9e3b", ClockDomain: "unix_utc"}
}
func mlflowGaugeOptions() nvtelemetry.TrainingWorkloadOTLPOptions {
	return nvtelemetry.TrainingWorkloadOTLPOptions{UnixClockProvenance: "caller-pinned MLflow2.22.0 file-store recipe passed declared historical Unix milliseconds, not current sensor time"}
}
func mlflowGaugeRead(t *testing.T, b []byte) nvtelemetry.WorkloadExport {
	t.Helper()
	e, err := nvtelemetry.ReadMLflowHistory(bytes.NewReader(b), mlflowGaugeSource(), nvtelemetry.WorkloadLimits{})
	if err != nil {
		t.Fatal(err)
	}
	return e
}
func mlflowGaugeWire(t *testing.T, out []byte) trainingGaugeWire {
	t.Helper()
	var w trainingGaugeWire
	if err := json.Unmarshal(out, &w); err != nil {
		t.Fatal(err)
	}
	if len(w.ResourceMetrics) != 1 || len(w.ResourceMetrics[0].ScopeMetrics) != 1 {
		t.Fatal("scope")
	}
	return w
}
func mlflowResourceAttrs(w trainingGaugeWire) map[string]string {
	a := map[string]string{}
	for _, p := range w.ResourceMetrics[0].Resource.Attributes {
		a[p.Key] = p.Value.StringValue
	}
	return a
}
func TestWriteTrainingWorkloadOTLPMLflowActualSDK(t *testing.T) {
	e := mlflowGaugeRead(t, mlflowGaugeInput(t))
	var out bytes.Buffer
	if err := nvtelemetry.WriteTrainingWorkloadOTLP(&out, e, mlflowGaugeOptions()); err != nil {
		t.Fatal(err)
	}
	w := mlflowGaugeWire(t, out.Bytes())
	a := mlflowResourceAttrs(w)
	if a["nvtelemetry.workload.source_report_sha256"] != "241bf82ed4d325c84e7b2c9591b1a583b24bed75d15a55d1fc9c0627ce30c5b5" || a["nvtelemetry.workload.provider"] != "mlflow" || a["nvtelemetry.workload.run_cohort"] != "4790069ba3984e409f20782b56bb9e3b" || a["nvtelemetry.workload.snapshot_scope"] != "supplied_page_only" || a["nvtelemetry.workload.paging_status"] != "no_next_page_reported" {
		t.Fatal(a)
	}
	m := w.ResourceMetrics[0].ScopeMetrics[0].Metrics
	if len(m) != 1 || m[0].Name != "nvtelemetry.workload.mlflow.loss" || m[0].Unit != "" || m[0].Gauge == nil || len(m[0].Sum) != 0 || len(m[0].Histogram) != 0 || len(m[0].Gauge.DataPoints) != 3 {
		t.Fatal(m)
	}
	// Literal independent oracle from the retained SDK recipe/response. Its
	// timestamps were supplied historical Unix milliseconds, not capture time.
	times := []string{"1700000000000000000", "1700000000010000000", "1700000000030000000"}
	values := []float64{0.5, 0.3, 0.2}
	steps := []string{"0", "1", "3"}
	for i, p := range m[0].Gauge.DataPoints {
		if p.TimeUnixNano != times[i] || p.AsDouble != values[i] {
			t.Fatal("millisecond conversion or value", i, p)
		}
		a := map[string]string{}
		for _, v := range p.Attributes {
			a[v.Key] = v.Value.StringValue
		}
		if a["nvtelemetry.workload.source_time_unit"] != "milliseconds" || a["nvtelemetry.workload.step_kind"] != "logged_metric_step" || a["nvtelemetry.workload.logged_metric_step"] != steps[i] {
			t.Fatal(a)
		}
		if _, ok := a["nvtelemetry.workload.history_step"]; ok {
			t.Fatal("MLflow logging step relabeled as W&B history")
		}
	}
}

// These are constructed paging/identity controls around the unchanged real
// SDK metrics, not additional authentic captures or an endpoint traversal.
func mlflowGaugeVariant(t *testing.T, change func(map[string]json.RawMessage, []map[string]json.RawMessage)) []byte {
	t.Helper()
	var root map[string]json.RawMessage
	json.Unmarshal(mlflowGaugeInput(t), &root)
	var metrics []map[string]json.RawMessage
	json.Unmarshal(root["metrics"], &metrics)
	change(root, metrics)
	root["metrics"], _ = json.Marshal(metrics)
	b, err := json.Marshal(root)
	if err != nil {
		t.Fatal(err)
	}
	return b
}
func TestWriteTrainingWorkloadOTLPMLflowPagingAndIdentity(t *testing.T) {
	t.Run("partial", func(t *testing.T) {
		const token = "private-opaque-next-page-token"
		raw := mlflowGaugeVariant(t, func(root map[string]json.RawMessage, _ []map[string]json.RawMessage) {
			root["next_page_token"] = json.RawMessage(`"` + token + `"`)
		})
		e := mlflowGaugeRead(t, raw)
		if e.NextPageToken != token {
			t.Fatal("reader token")
		}
		var out bytes.Buffer
		if err := nvtelemetry.WriteTrainingWorkloadOTLP(&out, e, mlflowGaugeOptions()); err != nil {
			t.Fatal(err)
		}
		a := mlflowResourceAttrs(mlflowGaugeWire(t, out.Bytes()))
		if a["nvtelemetry.workload.paging_status"] != "has_next_page" || a["nvtelemetry.workload.snapshot_scope"] != "supplied_page_only" || strings.Contains(out.String(), token) {
			t.Fatal("partial status or token privacy", a)
		}
		e.NextPageToken = "changed"
		out.Reset()
		if err := nvtelemetry.WriteTrainingWorkloadOTLP(&out, e, mlflowGaugeOptions()); err == nil || out.Len() != 0 {
			t.Fatal("changed paging projection accepted", err)
		}
	})
	t.Run("repeated logging step", func(t *testing.T) {
		raw := mlflowGaugeVariant(t, func(_ map[string]json.RawMessage, m []map[string]json.RawMessage) {
			m[1]["step"] = json.RawMessage(`0`)
		})
		var out bytes.Buffer
		if err := nvtelemetry.WriteTrainingWorkloadOTLP(&out, mlflowGaugeRead(t, raw), mlflowGaugeOptions()); err != nil {
			t.Fatal("a repeated MLflow logging step at a distinct time is not duplicate optimizer work", err)
		}
		if strings.Count(out.String(), `"key":"nvtelemetry.workload.logged_metric_step","value":{"stringValue":"0"}`) != 2 {
			t.Fatal("repeated logging step not preserved")
		}
	})
	t.Run("absent logging step", func(t *testing.T) {
		raw := mlflowGaugeVariant(t, func(_ map[string]json.RawMessage, m []map[string]json.RawMessage) { delete(m[0], "step") })
		var out bytes.Buffer
		if err := nvtelemetry.WriteTrainingWorkloadOTLP(&out, mlflowGaugeRead(t, raw), mlflowGaugeOptions()); err != nil {
			t.Fatal(err)
		}
		if strings.Count(out.String(), `"key":"nvtelemetry.workload.logged_metric_step"`) != 2 {
			t.Fatal("missing logging step fabricated")
		}
	})
}
func TestWriteTrainingWorkloadOTLPMLflowRefusals(t *testing.T) {
	for _, name := range []string{"missing time", "zero time", "duplicate time", "integer precision", "clock evidence", "unit evidence", "provider version", "source identity", "step kind", "UTF8"} {
		t.Run(name, func(t *testing.T) {
			raw := mlflowGaugeInput(t)
			switch name {
			case "missing time":
				raw = mlflowGaugeVariant(t, func(_ map[string]json.RawMessage, m []map[string]json.RawMessage) { delete(m[0], "timestamp") })
			case "zero time":
				raw = mlflowGaugeVariant(t, func(_ map[string]json.RawMessage, m []map[string]json.RawMessage) {
					m[0]["timestamp"] = json.RawMessage(`0`)
				})
			case "duplicate time":
				raw = mlflowGaugeVariant(t, func(_ map[string]json.RawMessage, m []map[string]json.RawMessage) {
					m[1]["timestamp"] = m[0]["timestamp"]
				})
			case "integer precision":
				raw = mlflowGaugeVariant(t, func(_ map[string]json.RawMessage, m []map[string]json.RawMessage) {
					m[0]["value"] = json.RawMessage(`9007199254740993`)
				})
			}
			e := mlflowGaugeRead(t, raw)
			o := mlflowGaugeOptions()
			switch name {
			case "clock evidence":
				o.UnixClockProvenance = ""
			case "unit evidence":
				s := mlflowGaugeSource()
				s.Units = map[string]string{"loss": "1"}
				var err error
				e, err = nvtelemetry.ReadMLflowHistory(bytes.NewReader(raw), s, nvtelemetry.WorkloadLimits{})
				if err != nil {
					t.Fatal(err)
				}
			case "provider version":
				e.Source.Version = "2.22.1"
			case "source identity":
				e.Source.RunID = "other"
			case "step kind":
				e.Metrics[0].StepKind = "history_step"
			case "UTF8":
				s := mlflowGaugeSource()
				s.Phase = string([]byte{0xff})
				var err error
				e, err = nvtelemetry.ReadMLflowHistory(bytes.NewReader(raw), s, nvtelemetry.WorkloadLimits{})
				if err != nil {
					t.Fatal(err)
				}
			}
			var out bytes.Buffer
			if err := nvtelemetry.WriteTrainingWorkloadOTLP(&out, e, o); err == nil || out.Len() != 0 {
				t.Fatal("MLflow ambiguity accepted or emitted bytes", name, err)
			}
		})
	}
}
