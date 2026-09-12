package metalprof

import (
	"encoding/json"
	"fmt"

	"github.com/tmc/lib/apple/x/metal/metalcapture"
)

// Timeline totals describe a replay run, not the live workload or one kernel.
func replayTimelineObservations(data []byte, device string) ([]metalcapture.Observation, error) {
	var export struct {
		Timeline map[string]json.RawMessage `json:"timeline"`
	}
	if err := json.Unmarshal(data, &export); err != nil {
		return nil, err
	}
	var observations []metalcapture.Observation
	for _, metric := range []struct{ key, name string }{
		{"command_buffer_active_time_ns", "Replay command-buffer active time"},
		{"command_buffer_wall_time_ns", "Replay command-buffer wall span"},
		{"restore_active_time_ns", "Replay restore active time"},
		{"restore_wall_time_ns", "Replay restore wall span"},
	} {
		raw, ok := export.Timeline[metric.key]
		if !ok || string(raw) == "null" {
			continue
		}
		var value int64
		if err := json.Unmarshal(raw, &value); err != nil {
			return nil, fmt.Errorf("metalprof: timeline %s: %w", metric.key, err)
		}
		if value < 0 {
			return nil, fmt.Errorf("metalprof: negative timeline %s", metric.key)
		}
		observations = append(observations, metalcapture.Observation{
			ID: "replay_" + metric.key, Name: metric.name, Unit: "nanoseconds",
			Kind: metalcapture.MetricKindDuration, Aggregation: metalcapture.AggregationLast,
			Scope: metalcapture.ScopeRun, Origin: metalcapture.OriginReplay,
			Status: metalcapture.StatusCollected, Device: device,
			Source: "replay_export.json:timeline." + metric.key,
			Reason: "APSTimelineData replay timestamps; includes captured warmup; active time and wall span are distinct",
			Value:  metalcapture.NewIntObservationValue(value), NonAdditive: true,
		})
	}
	return observations, nil
}
