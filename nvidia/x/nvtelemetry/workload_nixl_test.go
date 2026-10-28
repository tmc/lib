package nvtelemetry_test

import (
	"crypto/sha256"
	"fmt"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func nixlControl() ([]nvtelemetry.NIXLAgent, []nvtelemetry.NIXLAgentObservation, nvtelemetry.NIXLInventoryOptions) {
	opts := nvtelemetry.NIXLInventoryOptions{RunID: "run", StartNanos: 10, EndNanos: 20, MaxAgeNanos: 2, ClockDomain: "mono", ClockEpochID: "boot"}
	expected := []nvtelemetry.NIXLAgent{{Agent: "a", Rank: 0, RestartEpoch: "worker-1", Build: nvtelemetry.NIXLTelemetryBuild, Mode: "prometheus"}, {Agent: "b", Rank: 1, RestartEpoch: "worker-1", Build: nvtelemetry.NIXLTelemetryBuild, Mode: "prometheus"}}
	observed := make([]nvtelemetry.NIXLAgentObservation, len(expected))
	for i, a := range expected {
		observed[i] = nvtelemetry.NIXLAgentObservation{Producer: a, Sink: "ready", Scrape: "ready", HasLastObservation: true, LastObservationNanos: 20, EvidenceSHA256: strings.Repeat("a", 64), Before: nixlControlExport(a, 10, "9007199254740993"), After: nixlControlExport(a, 20, "9007199254740993")}
	}
	return expected, observed, opts
}

// These are supplied canonical host controls, not authentic NIXL captures.
// The existing public window consumer performs their exact interval arithmetic.
func nixlControlExport(a nvtelemetry.NIXLAgent, when int64, value string) nvtelemetry.WorkloadExport {
	rank := a.Rank
	source := nvtelemetry.WorkloadSource{Provider: "nixl", Version: a.Build, RunID: "run", Rank: &rank, ClockDomain: "mono", ClockEpochID: "boot"}
	raw := []byte(fmt.Sprintf("agent_telemetry_events_dropped_total{agent_name=%q} %s at %d", a.Agent, value, when))
	return nvtelemetry.WorkloadExport{Source: source, Raw: raw, RawSHA256: fmt.Sprintf("%x", sha256.Sum256(raw)), Metrics: []nvtelemetry.WorkloadMetric{{Source: source, Name: "agent_telemetry_events_dropped_total", Semantic: "telemetry_staging_drops", Unit: "events", Kind: "counter", Temporality: "cumulative", Labels: map[string]string{"agent_name": a.Agent}, RawValue: value, HasTime: true, TimeNanos: when}}}
}

func TestNIXLInventoryPopulation(t *testing.T) {
	for _, tt := range []struct {
		name, status string
		mutate       func(*[]nvtelemetry.NIXLAgentObservation)
	}{
		{"missing", "missing", func(o *[]nvtelemetry.NIXLAgentObservation) { *o = (*o)[:1] }},
		{"occupied port", "sink_failed", func(o *[]nvtelemetry.NIXLAgentObservation) {
			(*o)[1].Sink = "failed"
			(*o)[1].Detail = "bind: address already in use"
		}},
		{"unknown sink", "sink_unknown", func(o *[]nvtelemetry.NIXLAgentObservation) { (*o)[1].Sink = "unknown" }},
		{"scrape failure", "scrape_failed", func(o *[]nvtelemetry.NIXLAgentObservation) { (*o)[1].Scrape = "failed" }},
		{"unknown scrape", "scrape_unknown", func(o *[]nvtelemetry.NIXLAgentObservation) { (*o)[1].Scrape = "unknown" }},
		{"stale", "stale", func(o *[]nvtelemetry.NIXLAgentObservation) { (*o)[1].LastObservationNanos = 17 }},
		{"unknown last", "stale", func(o *[]nvtelemetry.NIXLAgentObservation) { (*o)[1].HasLastObservation = false }},
		{"future last", "stale", func(o *[]nvtelemetry.NIXLAgentObservation) { (*o)[1].LastObservationNanos = 21 }},
		{"restart", "restarted", func(o *[]nvtelemetry.NIXLAgentObservation) { (*o)[1].Producer.RestartEpoch = "worker-2" }},
		{"build drift", "producer_changed", func(o *[]nvtelemetry.NIXLAgentObservation) { (*o)[1].Producer.Build = "other-build" }},
		{"mode drift", "producer_changed", func(o *[]nvtelemetry.NIXLAgentObservation) { (*o)[1].Producer.Mode = "prometheus_mp" }},
		{"missing export", "window_unavailable", func(o *[]nvtelemetry.NIXLAgentObservation) { (*o)[1].After = nvtelemetry.WorkloadExport{} }},
		{"wrong agent", "window_unavailable", func(o *[]nvtelemetry.NIXLAgentObservation) { (*o)[1].After.Metrics[0].Labels["agent_name"] = "a" }},
		{"raw corruption", "window_unavailable", func(o *[]nvtelemetry.NIXLAgentObservation) { (*o)[1].After.Raw[0] = 'X' }},
		{"counter reset", "window_unavailable", func(o *[]nvtelemetry.NIXLAgentObservation) {
			(*o)[1].After = nixlControlExport((*o)[1].Producer, 20, "0")
		}},
		{"epoch drift", "window_unavailable", func(o *[]nvtelemetry.NIXLAgentObservation) { (*o)[1].After.Metrics[0].Source.ClockEpochID = "other" }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			expected, observed, opts := nixlControl()
			tt.mutate(&observed)
			got, err := nvtelemetry.SummarizeNIXLInventory(expected, observed, opts)
			if err != nil {
				t.Fatal(err)
			}
			if got.InventoryComplete || got.Expected != 2 || got.Ready != 1 || len(got.Agents) != 2 || got.Agents[1].Status != tt.status {
				t.Fatalf("missing expected population: %+v", got)
			}
			if tt.name == "occupied port" && (got.Agents[1].Sink != "failed" || got.Agents[1].Scrape != "ready" || got.Agents[1].Detail != "bind: address already in use") {
				t.Fatalf("lost bind failure: %+v", got.Agents[1])
			}
			if got.Agents[0].StagingDrops != "0" || got.Agents[0].RetentionLoss != "unknown" {
				t.Fatalf("loss domains confused: %+v", got.Agents[0])
			}
		})
	}
}

func TestNIXLInventoryReady(t *testing.T) {
	expected, observed, opts := nixlControl()
	observed[1].After = nixlControlExport(expected[1], 20, "9007199254740994")
	got, err := nvtelemetry.SummarizeNIXLInventory(expected, observed, opts)
	if err != nil {
		t.Fatal(err)
	}
	if !got.InventoryComplete || got.Expected != 2 || got.Observed != 2 || got.Ready != 2 || got.Agents[1].StagingDrops != "1" {
		t.Fatalf("inventory %+v", got)
	}
	if got.Options.RunID != opts.RunID || got.Options.ClockEpochID != opts.ClockEpochID || got.Options.Limits.MaxBytes == 0 {
		t.Fatal("summary lost interval/limit context")
	}
	for i, a := range got.Agents {
		if a.Sink != "ready" || a.Scrape != "ready" || !a.HasLastObservation || a.LastObservationNanos != 20 {
			t.Fatalf("lost supplied status metadata: %+v", a)
		}
		if a.Status != "ready" || a.RetentionLoss != "unknown" || a.BufferBehavior != "unknown" || a.Observed == nil || a.BeforeSHA256 != observed[i].Before.RawSHA256 || a.AfterSHA256 != observed[i].After.RawSHA256 {
			t.Fatalf("scoped evidence %+v", a)
		}
	}
	observed[0].Before.Metrics[0].Labels["agent_name"] = "changed"
	observed[0].Producer.Agent = "changed"
	expected[0].Agent = "changed"
	if got.Agents[0].Windows[0].Labels["agent_name"] != "a" || got.Agents[0].Producer.Agent != "a" || got.Agents[0].Observed.Agent != "a" {
		t.Fatal("result aliases mutable input")
	}
}

func TestNIXLInventorySemantics(t *testing.T) {
	for _, tt := range []struct{ name, build, mode, behavior, drops string }{
		{"pinned buffer", nvtelemetry.NIXLTelemetryBuild, "buffer", "reject_when_full", ""},
		{"other buffer", "other-build", "buffer", "unknown", ""},
		{"unqualified multiprocess", nvtelemetry.NIXLTelemetryBuild, "prometheus_mp", "unknown", ""},
	} {
		t.Run(tt.name, func(t *testing.T) {
			e, o, opts := nixlControl()
			e = e[:1]
			o = o[:1]
			e[0].Build = tt.build
			e[0].Mode = tt.mode
			o[0].Producer = e[0]
			o[0].Before = nixlControlExport(e[0], 10, "0")
			o[0].After = nixlControlExport(e[0], 20, "0")
			got, err := nvtelemetry.SummarizeNIXLInventory(e, o, opts)
			if err != nil {
				t.Fatal(err)
			}
			if !got.InventoryComplete || got.Agents[0].BufferBehavior != tt.behavior || got.Agents[0].StagingDrops != tt.drops || got.Agents[0].RetentionLoss != "unknown" {
				t.Fatalf("invented loss semantics %+v", got)
			}
		})
	}
}

func TestNIXLInventoryRefusals(t *testing.T) {
	for _, tt := range []struct {
		name   string
		mutate func(*[]nvtelemetry.NIXLAgent, *[]nvtelemetry.NIXLAgentObservation, *nvtelemetry.NIXLInventoryOptions)
	}{
		{"empty", func(e *[]nvtelemetry.NIXLAgent, o *[]nvtelemetry.NIXLAgentObservation, p *nvtelemetry.NIXLInventoryOptions) {
			*e = nil
		}},
		{"duplicate expected", func(e *[]nvtelemetry.NIXLAgent, o *[]nvtelemetry.NIXLAgentObservation, p *nvtelemetry.NIXLInventoryOptions) {
			(*e)[1] = (*e)[0]
		}},
		{"duplicate observed", func(e *[]nvtelemetry.NIXLAgent, o *[]nvtelemetry.NIXLAgentObservation, p *nvtelemetry.NIXLInventoryOptions) {
			(*o)[1] = (*o)[0]
		}},
		{"unexpected", func(e *[]nvtelemetry.NIXLAgent, o *[]nvtelemetry.NIXLAgentObservation, p *nvtelemetry.NIXLInventoryOptions) {
			(*o)[1].Producer.Agent = "other"
		}},
		{"rank", func(e *[]nvtelemetry.NIXLAgent, o *[]nvtelemetry.NIXLAgentObservation, p *nvtelemetry.NIXLInventoryOptions) {
			(*e)[0].Rank = -1
		}},
		{"status", func(e *[]nvtelemetry.NIXLAgent, o *[]nvtelemetry.NIXLAgentObservation, p *nvtelemetry.NIXLInventoryOptions) {
			(*o)[0].Sink = "yes"
		}},
		{"evidence", func(e *[]nvtelemetry.NIXLAgent, o *[]nvtelemetry.NIXLAgentObservation, p *nvtelemetry.NIXLInventoryOptions) {
			(*o)[0].EvidenceSHA256 = "missing"
		}},
		{"cardinality", func(e *[]nvtelemetry.NIXLAgent, o *[]nvtelemetry.NIXLAgentObservation, p *nvtelemetry.NIXLInventoryOptions) {
			p.Limits.MaxMetrics = 1
		}},
		{"raw bytes", func(e *[]nvtelemetry.NIXLAgent, o *[]nvtelemetry.NIXLAgentObservation, p *nvtelemetry.NIXLInventoryOptions) {
			p.Limits.MaxBytes = 1
		}},
		{"metric string", func(e *[]nvtelemetry.NIXLAgent, o *[]nvtelemetry.NIXLAgentObservation, p *nvtelemetry.NIXLInventoryOptions) {
			(*o)[0].Before.Metrics[0].Name = strings.Repeat("x", 4097)
		}},
		{"age", func(e *[]nvtelemetry.NIXLAgent, o *[]nvtelemetry.NIXLAgentObservation, p *nvtelemetry.NIXLInventoryOptions) {
			p.MaxAgeNanos = 0
		}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			e, o, p := nixlControl()
			tt.mutate(&e, &o, &p)
			if _, err := nvtelemetry.SummarizeNIXLInventory(e, o, p); err == nil {
				t.Fatal("invalid inventory accepted")
			}
		})
	}
}
