package nvtelemetry_test

import (
	"crypto/sha256"
	"fmt"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func ExampleSummarizeNIXLInventory() {
	expected := []nvtelemetry.NIXLAgent{
		{Agent: "prefill", Rank: 0, RestartEpoch: "start-1", Build: nvtelemetry.NIXLTelemetryBuild, Mode: "prometheus"},
		{Agent: "decode", Rank: 1, RestartEpoch: "start-1", Build: nvtelemetry.NIXLTelemetryBuild, Mode: "prometheus"},
	}
	// Constructed canonical snapshots illustrate supplied evidence. An actual
	// importer must retain the original bytes and independently map their fields.
	snapshot := func(a nvtelemetry.NIXLAgent, nanos int64) nvtelemetry.WorkloadExport {
		rank := a.Rank
		source := nvtelemetry.WorkloadSource{Provider: "nixl", Version: a.Build, RunID: "example", Rank: &rank, ClockDomain: "mono", ClockEpochID: "boot-1"}
		raw := []byte("agent_telemetry_events_dropped_total 0")
		return nvtelemetry.WorkloadExport{Source: source, Raw: raw, RawSHA256: fmt.Sprintf("%x", sha256.Sum256(raw)), Metrics: []nvtelemetry.WorkloadMetric{{Source: source, Name: "agent_telemetry_events_dropped_total", Semantic: "telemetry_staging_drops", Unit: "events", Kind: "counter", Temporality: "cumulative", Labels: map[string]string{"agent_name": a.Agent}, RawValue: "0", HasTime: true, TimeNanos: nanos}}}
	}
	statusRaw := []byte("supplied sink/scrape status artifact")
	evidence := fmt.Sprintf("%x", sha256.Sum256(statusRaw))
	observed := []nvtelemetry.NIXLAgentObservation{
		{Producer: expected[0], Sink: "ready", Scrape: "ready", EvidenceSHA256: evidence, HasLastObservation: true, LastObservationNanos: 20, Before: snapshot(expected[0], 10), After: snapshot(expected[0], 20)},
		{Producer: expected[1], Sink: "failed", Scrape: "failed", EvidenceSHA256: evidence, Detail: "bind: address already in use"},
	}
	inventory, err := nvtelemetry.SummarizeNIXLInventory(expected, observed, nvtelemetry.NIXLInventoryOptions{RunID: "example", StartNanos: 10, EndNanos: 20, MaxAgeNanos: 2, ClockDomain: "mono", ClockEpochID: "boot-1"})
	if err != nil {
		panic(err)
	}
	fmt.Println(inventory.Ready, inventory.Expected, inventory.InventoryComplete)
	fmt.Println(inventory.Agents[0].StagingDrops, inventory.Agents[0].RetentionLoss)
	fmt.Println(inventory.Agents[1].Status)
	// Output:
	// 1 2 false
	// 0 unknown
	// sink_failed
}
