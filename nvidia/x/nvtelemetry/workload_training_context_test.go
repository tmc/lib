package nvtelemetry_test

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func phaseContextFixture() nvtelemetry.TrainingPhaseContext {
	return nvtelemetry.TrainingPhaseContext{Version: 2, RunID: "run", Policies: []nvtelemetry.TrainingQualityPolicy{{ID: "q", Numerical: "unknown"}}, Phases: []nvtelemetry.TrainingPhase{{ID: "forward:0", Attempt: "attempt", Restart: "boot1", Worker: "worker", WorldSize: 1, AccumulationCount: 1, Phase: "forward", QualityPolicy: "q", Evidence: "phase-log", Operators: []nvtelemetry.TrainingOperatorReference{}}}}
}

func TestTrainingPhaseContextRoundTrip(t *testing.T) {
	c := phaseContextFixture()
	c.Phases[0].OptimizerStep = 9007199254740993
	c.Phases[0].LoggingStep = 18446744073709551615
	var b bytes.Buffer
	if err := nvtelemetry.WriteTrainingPhaseContext(&b, c, nvtelemetry.WorkloadLimits{}); err != nil {
		t.Fatal(err)
	}
	e, got, err := nvtelemetry.ReadTrainingPhaseContext(&b, nvtelemetry.WorkloadSource{Provider: "application", Version: "training-context-v2", RunID: "run"}, nvtelemetry.WorkloadLimits{})
	if err != nil || got.Phases[0].OptimizerStep != 9007199254740993 || got.Phases[0].LoggingStep != 18446744073709551615 || e.RawSHA256 == "" {
		t.Fatal(got, err)
	}
	if _, _, err = nvtelemetry.ReadTrainingContext(bytes.NewReader(e.Raw), e.Source, nvtelemetry.WorkloadLimits{}); err == nil {
		t.Fatal("v1 reader admitted v2")
	}
	var old bytes.Buffer
	if err := nvtelemetry.WriteTrainingContext(&old, nvtelemetry.TrainingContext{Version: 1, RunID: "run", Updates: []nvtelemetry.TrainingUpdate{}}, nvtelemetry.WorkloadLimits{}); err != nil {
		t.Fatal(err)
	}
	if _, _, err = nvtelemetry.ReadTrainingContext(bytes.NewReader(old.Bytes()), nvtelemetry.WorkloadSource{Provider: "application", Version: "training-context-v1", RunID: "run"}, nvtelemetry.WorkloadLimits{}); err != nil {
		t.Fatal(err)
	}
	if _, _, err = nvtelemetry.ReadTrainingPhaseContext(&old, e.Source, nvtelemetry.WorkloadLimits{}); err == nil {
		t.Fatal("v2 admitted v1")
	}
}

func TestTrainingPhaseContextRefusals(t *testing.T) {
	raw, err := json.Marshal(phaseContextFixture())
	if err != nil {
		t.Fatal(err)
	}
	for _, tt := range []struct{ name, from, to string }{
		{"future", "\"version\":2", "\"version\":3"},
		{"null", "\"rank\":0", "\"rank\":null"},
		{"absent", "\"rank\":0,", ""},
		{"fraction", "\"logging_step\":0", "\"logging_step\":0.0"},
		{"exponent", "\"epoch\":0", "\"epoch\":0e0"},
		{"negative", "\"batch\":0", "\"batch\":-0"},
		{"overflow", "\"optimizer_step\":0", "\"optimizer_step\":18446744073709551616"},
		{"duplicate", "\"rank\":0", "\"rank\":0,\"rank\":0"},
		{"unknown", "\"rank\":0", "\"extra\":0,\"rank\":0"},
		{"phase", "\"phase\":\"forward\"", "\"phase\":\"invented\""},
	} {
		t.Run(tt.name, func(t *testing.T) {
			changed := strings.Replace(string(raw), tt.from, tt.to, 1)
			if changed == string(raw) {
				t.Fatal("ineffective fixture")
			}
			if _, _, err := nvtelemetry.ReadTrainingPhaseContext(strings.NewReader(changed), nvtelemetry.WorkloadSource{Provider: "application", Version: "training-context-v2", RunID: "run"}, nvtelemetry.WorkloadLimits{}); err == nil {
				t.Fatal("invalid context accepted")
			}
		})
	}
	for _, mode := range []string{"bytes", "string", "count", "accumulation", "policy", "duplicate"} {
		t.Run(mode, func(t *testing.T) {
			c := phaseContextFixture()
			l := nvtelemetry.WorkloadLimits{}
			switch mode {
			case "bytes":
				l.MaxBytes = 16
			case "string":
				l.MaxStringBytes = 3
			case "count":
				l.MaxMetrics = 1
				c.Phases = append(c.Phases, c.Phases[0])
			case "accumulation":
				c.Phases[0].AccumulationStep = 1
			case "policy":
				c.Phases[0].QualityPolicy = "missing"
			case "duplicate":
				c.Policies = append(c.Policies, c.Policies[0])
			}
			var b bytes.Buffer
			if err := nvtelemetry.WriteTrainingPhaseContext(&b, c, l); err == nil || b.Len() != 0 {
				t.Fatal("invalid write admitted or wrote before refusal", err, b.Len())
			}
		})
	}
}
