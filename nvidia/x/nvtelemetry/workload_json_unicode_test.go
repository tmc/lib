package nvtelemetry_test

import (
	"bytes"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func TestWorkloadReadersRejectInvalidUnicode(t *testing.T) {
	type reader func([]byte) (nvtelemetry.WorkloadExport, error)
	wandb := func(raw []byte) (nvtelemetry.WorkloadExport, error) {
		return nvtelemetry.ReadWandBHistory(bytes.NewReader(raw), nvtelemetry.WorkloadSource{Provider: "wandb", Version: "0.19.11", RunID: "run"}, nvtelemetry.WorkloadLimits{})
	}
	mlflow := func(raw []byte) (nvtelemetry.WorkloadExport, error) {
		return nvtelemetry.ReadMLflowHistory(bytes.NewReader(raw), nvtelemetry.WorkloadSource{Provider: "mlflow", Version: "2.22.0", RunID: "run"}, nvtelemetry.WorkloadLimits{})
	}
	clearml := func(raw []byte) (nvtelemetry.WorkloadExport, error) {
		return nvtelemetry.ReadClearMLScalars(bytes.NewReader(raw), nvtelemetry.WorkloadSource{Provider: "clearml", Version: "1.18.0/api-2.9", RunID: "run"}, nvtelemetry.WorkloadLimits{})
	}
	tests := []struct {
		name string
		read reader
		raw  []byte
	}{
		{"wandb_distinct_keys", wandb, []byte("{\"_timestamp\":1,\"_step\":0,\"\\ud800\":1}\n{\"_timestamp\":2,\"_step\":1,\"\\ud801\":2}\n")},
		{"wandb_ignored_value", wandb, []byte(`{"_timestamp":1,"loss":1,"note":"\ud800"}`)},
		{"wandb_invalid_utf8", wandb, append(append([]byte(`{"_timestamp":1,"`), 0xff), []byte(`":1}`)...)},
		{"mlflow_key", mlflow, []byte(`{"metrics":[{"key":"\ud800","value":1,"timestamp":1000,"step":0}]}`)},
		{"mlflow_ignored_value", mlflow, []byte(`{"metrics":[],"unknown":"\udc00"}`)},
		{"clearml_variant", clearml, []byte(`{"events":[{"type":"training_stats_scalar","task":"run","metric":"loss","variant":"\ud800","value":1,"timestamp":1000,"iter":0}]}`)},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			e, err := tc.read(tc.raw)
			if err == nil {
				t.Errorf("invalid Unicode accepted: metrics=%+v", e.Metrics)
			}
		})
	}
}

func TestWorkloadReaderValidUnicode(t *testing.T) {
	const raw = `{"_timestamp":1,"_step":0,"\ud83d\ude80":1,"�":2,"note":"\\ud800"}`
	source := nvtelemetry.WorkloadSource{Provider: "wandb", Version: "0.19.11", RunID: "run"}
	e, err := nvtelemetry.ReadWandBHistory(strings.NewReader(raw), source, nvtelemetry.WorkloadLimits{})
	if err != nil {
		t.Fatal(err)
	}
	found := map[string]bool{}
	for _, m := range e.Metrics {
		found[m.Name] = true
	}
	if len(e.Metrics) != 2 || !found["🚀"] || !found["�"] {
		t.Fatalf("decoded metrics = %+v", e.Metrics)
	}
}
