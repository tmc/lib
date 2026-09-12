package metalprof

import (
	"encoding/json"
	"fmt"

	"github.com/tmc/lib/apple/x/metal/metalcapture"
)

// Export archive coverage without assigning machine-wide records to kernels.
func replayCounterObservations(data []byte, device string) ([]metalcapture.Observation, error) {
	var export struct {
		Raw map[string]json.RawMessage `json:"counter_archive"`
	}
	if err := json.Unmarshal(data, &export); err != nil {
		return nil, err
	}
	var observations []metalcapture.Observation
	counts := make(map[string]int64)
	for _, field := range []struct{ key, name string }{
		{"total_samples", "Decoded counter records"},
		{"attributed_samples", "Counter records naming capture encoders"},
		{"machine_wide_samples", "Machine-wide counter records"},
		{"blobs", "Counter archive blobs"},
		{"stride_mismatches", "Counter blobs with stride mismatches"},
	} {
		raw, ok := export.Raw[field.key]
		if !ok || string(raw) == "null" {
			continue
		}
		var value int64
		if err := json.Unmarshal(raw, &value); err != nil {
			return nil, fmt.Errorf("metalprof: counter %s: %w", field.key, err)
		}
		if value < 0 {
			return nil, fmt.Errorf("metalprof: negative counter %s", field.key)
		}
		counts[field.key] = value
		observations = append(observations, metalcapture.Observation{
			ID: "replay_counter_" + field.key, Name: field.name, Unit: "count",
			Kind: metalcapture.MetricKindCount, Aggregation: metalcapture.AggregationLast,
			Scope: metalcapture.ScopeRun, Origin: metalcapture.OriginReplay, Status: metalcapture.StatusCollected,
			Source: "replay_export.json:counter_archive." + field.key,
			Reason: "Archive coverage, not kernel execution counts; categories overlap with the total",
			Device: device, Value: metalcapture.NewIntObservationValue(value), NonAdditive: true,
		})
	}
	if total, ok := counts["total_samples"]; ok {
		attributed := counts["attributed_samples"]
		machine := counts["machine_wide_samples"]
		if attributed > total || machine > total-attributed {
			return nil, fmt.Errorf("metalprof: counter record categories exceed total")
		}
	}
	if blobs, ok := counts["blobs"]; ok && counts["stride_mismatches"] > blobs {
		return nil, fmt.Errorf("metalprof: counter stride mismatches exceed blob count")
	}
	return observations, nil
}
