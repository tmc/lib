package nvtelemetry_test

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func TestWorkloadProviderKeyAliases(t *testing.T) {
	source := nvtelemetry.WorkloadSource{Provider: "mlflow", Version: "2.22.0", RunID: "run"}
	for _, tt := range []struct {
		name string
		body string
	}{
		{"timestamp", `{"metrics":[{"key":"loss","value":1,"timestamp":1000,"Timestamp":2000}]}`},
		{"timestamp_reverse", `{"metrics":[{"key":"loss","value":1,"Timestamp":2000,"timestamp":1000}]}`},
		{"timestamp_escaped", `{"metrics":[{"key":"loss","value":1,"timestamp":1000,"\u0054imestamp":2000}]}`},
		{"value", `{"metrics":[{"key":"loss","value":1,"Value":2}]}`},
		{"key", `{"metrics":[{"key":"loss","Key":"accuracy","value":1}]}`},
		{"step", `{"metrics":[{"key":"loss","value":1,"step":0,"Step":1}]}`},
		{"metrics", `{"metrics":[{"key":"loss","value":1}],"Metrics":[]}`},
		{"page_token", `{"metrics":[],"next_page_token":"first","Next_Page_Token":"second"}`},
		{"exact_duplicate", `{"metrics":[{"key":"loss","value":1,"value":2}]}`},
		{"unknown_exact_duplicate", `{"metrics":[],"extra":{"x":1,"x":2}}`},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := nvtelemetry.ReadMLflowHistory(strings.NewReader(tt.body), source, nvtelemetry.WorkloadLimits{}); err == nil {
				t.Fatal("accepted contradictory JSON keys")
			}
		})
	}

	for _, key := range []string{"timestamp", "Timestamp", "TIMESTAMP"} {
		t.Run("single_"+key, func(t *testing.T) {
			body := `{"metrics":[{"key":"loss","value":1,"` + key + `":1000}],"extra":{"x":1,"X":2}}`
			export, err := nvtelemetry.ReadMLflowHistory(strings.NewReader(body), source, nvtelemetry.WorkloadLimits{})
			if err != nil {
				t.Fatal(err)
			}
			if len(export.Metrics) != 1 || !export.Metrics[0].HasTime || export.Metrics[0].TimeNanos != 1000000000 {
				t.Fatalf("timestamp lost: %+v", export.Metrics)
			}
		})
	}

	t.Run("case_sensitive_metric_map", func(t *testing.T) {
		source := nvtelemetry.WorkloadSource{Provider: "wandb", Version: "0.19.11", RunID: "run"}
		export, err := nvtelemetry.ReadWandBHistory(strings.NewReader(`{"Loss":2,"loss":1,"_step":0}`), source, nvtelemetry.WorkloadLimits{})
		if err != nil {
			t.Fatal(err)
		}
		if len(export.Metrics) != 2 || export.Metrics[0].Name != "Loss" || export.Metrics[0].RawValue != "2" || export.Metrics[1].Name != "loss" || export.Metrics[1].RawValue != "1" {
			t.Fatalf("case-sensitive metric keys lost: %+v", export.Metrics)
		}
	})
}
