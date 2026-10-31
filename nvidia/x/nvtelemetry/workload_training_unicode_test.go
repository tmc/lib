package nvtelemetry_test

import (
	"bytes"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func TestTrainingOTLPRejectsUnpairedSurrogates(t *testing.T) {
	cases := []struct{ name, raw string }{
		{"distinct-high-keys", "{\"_timestamp\":1,\"_step\":0,\"\\ud800\":1}\n{\"_timestamp\":2,\"_step\":1,\"\\ud801\":2}\n"},
		{"low-key", `{"_timestamp":1,"_step":0,"\udc00":1}`},
		{"high-non-low", `{"_timestamp":1,"_step":0,"\ud800\u0041":1}`},
		{"ignored-value", `{"_timestamp":1,"_step":0,"loss":1,"note":"\ud800"}`},
	}
	source := nvtelemetry.WorkloadSource{Provider: "wandb", Version: "0.19.11", RunID: "controlled", ClockDomain: "unix_utc"}
	options := nvtelemetry.TrainingWorkloadOTLPOptions{UnixClockProvenance: "controlled Unix seconds"}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := nvtelemetry.ReadWandBHistory(strings.NewReader(tc.raw), source, nvtelemetry.WorkloadLimits{}); err == nil {
				t.Error("reader accepted invalid Unicode")
			}
			e := nvtelemetry.WorkloadExport{Source: source, Raw: []byte(tc.raw)}
			var out bytes.Buffer
			if err := nvtelemetry.WriteTrainingWorkloadOTLP(&out, e, options); err == nil || !strings.Contains(err.Error(), "workload JSON is not valid Unicode") {
				t.Errorf("WriteTrainingWorkloadOTLP error = %v, want Unicode refusal", err)
			}
			if out.Len() != 0 {
				t.Errorf("wrote %d bytes on refusal", out.Len())
			}
		})
	}
}

func TestTrainingOTLPValidUnicodeEscapes(t *testing.T) {
	const raw = `{"_timestamp":1,"_step":0,"\ud83d\ude80":1,"�":2,"note":"\\ud800"}`
	source := nvtelemetry.WorkloadSource{Provider: "wandb", Version: "0.19.11", RunID: "controlled", ClockDomain: "unix_utc"}
	e, err := nvtelemetry.ReadWandBHistory(strings.NewReader(raw), source, nvtelemetry.WorkloadLimits{})
	if err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	if err := nvtelemetry.WriteTrainingWorkloadOTLP(&out, e, nvtelemetry.TrainingWorkloadOTLPOptions{UnixClockProvenance: "controlled Unix seconds"}); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), "nvtelemetry.workload.wandb.🚀") || !strings.Contains(out.String(), "nvtelemetry.workload.wandb.�") {
		t.Fatalf("names lost: %s", out.String())
	}
}

func TestTrainingOTLPRejectsMLflowUnpairedSurrogates(t *testing.T) {
	const raw = `{"metrics":[{"key":"\ud800","value":1,"timestamp":1000,"step":0},{"key":"\ud801","value":2,"timestamp":2000,"step":1}]}`
	source := nvtelemetry.WorkloadSource{Provider: "mlflow", Version: "2.22.0", RunID: "controlled", ClockDomain: "unix_utc"}
	if _, err := nvtelemetry.ReadMLflowHistory(strings.NewReader(raw), source, nvtelemetry.WorkloadLimits{}); err == nil {
		t.Error("reader accepted invalid Unicode")
	}
	e := nvtelemetry.WorkloadExport{Source: source, Raw: []byte(raw)}
	var out bytes.Buffer
	if err := nvtelemetry.WriteTrainingWorkloadOTLP(&out, e, nvtelemetry.TrainingWorkloadOTLPOptions{UnixClockProvenance: "controlled Unix milliseconds"}); err == nil || !strings.Contains(err.Error(), "workload JSON is not valid Unicode") {
		t.Errorf("WriteTrainingWorkloadOTLP error = %v, want Unicode refusal", err)
	}
	if out.Len() != 0 {
		t.Errorf("wrote %d bytes on refusal", out.Len())
	}
}
