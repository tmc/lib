package metalprof

import (
	"encoding/json"
	"fmt"
	"slices"

	"github.com/tmc/lib/apple/x/metal/metalcapture"
)

// The archive names establish presence independently of the exporter's numeric
// defaults. These are compiler statistics, not dynamic instruction executions.
var pipelineMetrics = []struct {
	key, archive, unit string
	kind               metalcapture.MetricKind
}{
	{"temporary_register_count", "Temporary register count", "registers", metalcapture.MetricKindCount},
	{"uniform_register_count", "Uniform register count", "registers", metalcapture.MetricKindCount},
	{"constant_calculation_temporary_register_count", "Constant calculation temporary register count", "registers", metalcapture.MetricKindCount},
	{"thread_invariant_spilled", "Thread invariant spilled bytes", "bytes", metalcapture.MetricKindBytes},
	{"threadgroup_memory", "Threadgroup memory", "bytes", metalcapture.MetricKindBytes},
	{"alu_instruction_count", "ALU instruction count", "instructions", metalcapture.MetricKindCount},
	{"fp32_instruction_count", "FP32 instruction count", "instructions", metalcapture.MetricKindCount},
	{"fp16_instruction_count", "FP16 instruction count", "instructions", metalcapture.MetricKindCount},
	{"int32_instruction_count", "INT32 instruction count", "instructions", metalcapture.MetricKindCount},
	{"int16_instruction_count", "INT16 instruction count", "instructions", metalcapture.MetricKindCount},
	{"branch_instruction_count", "Branch instruction count", "instructions", metalcapture.MetricKindCount},
	{"device_load_instruction_count", "Device load instruction count", "instructions", metalcapture.MetricKindCount},
	{"device_store_instruction_count", "Device store instruction count", "instructions", metalcapture.MetricKindCount},
	{"device_atomic_instruction_count", "Device atomic instruction count", "instructions", metalcapture.MetricKindCount},
	{"texture_reads_instruction_count", "Texture reads instruction count", "instructions", metalcapture.MetricKindCount},
	{"texture_writes_instruction_count", "Texture writes instruction count", "instructions", metalcapture.MetricKindCount},
	{"threadgroup_load_instruction_count", "Threadgroup load instruction count", "instructions", metalcapture.MetricKindCount},
	{"threadgroup_store_instruction_count", "Threadgroup store instruction count", "instructions", metalcapture.MetricKindCount},
	{"threadgroup_atomic_instruction_count", "Threadgroup atomic instruction count", "instructions", metalcapture.MetricKindCount},
	{"wait_instruction_count", "Wait instruction count", "instructions", metalcapture.MetricKindCount},
	{"compilation_time_ms", "Compilation time in milliseconds", "milliseconds", metalcapture.MetricKindDuration},
}

func replayPipelineObservations(data []byte, device string) ([]metalcapture.Observation, error) {
	var export struct {
		Pipelines []map[string]json.RawMessage `json:"pipelines"`
	}
	if err := json.Unmarshal(data, &export); err != nil {
		return nil, err
	}
	var observations []metalcapture.Observation
	for i, p := range export.Pipelines {
		var identity struct {
			ID       *int     `json:"pipeline_id"`
			Name     string   `json:"function_name"`
			Recorded []string `json:"recorded_statistics"`
		}
		b, err := json.Marshal(p)
		if err != nil {
			return nil, err
		}
		if err := json.Unmarshal(b, &identity); err != nil {
			return nil, err
		}
		for _, metric := range pipelineMetrics {
			if !slices.Contains(identity.Recorded, metric.archive) {
				continue
			}
			raw, ok := p[metric.key]
			if !ok || string(raw) == "null" {
				return nil, fmt.Errorf("metalprof: pipeline %d recorded %s without a value", i, metric.key)
			}
			var value metalcapture.ObservationValue
			if err := json.Unmarshal(raw, &value); err != nil {
				return nil, fmt.Errorf("metalprof: pipeline %d %s: %w", i, metric.key, err)
			}
			if value.Float64() < 0 || metric.kind != metalcapture.MetricKindDuration && !value.IsInt {
				return nil, fmt.Errorf("metalprof: pipeline %d invalid %s", i, metric.key)
			}
			id := fmt.Sprintf("row_%d", i)
			if identity.ID != nil {
				id = fmt.Sprint(*identity.ID)
			}
			observations = append(observations, metalcapture.Observation{
				ID:   "compiled_pipeline_" + id + "_" + metric.key,
				Name: "Compiler " + metric.archive, Unit: metric.unit, Kind: metric.kind,
				Aggregation: metalcapture.AggregationLast, Scope: metalcapture.ScopePipeline,
				Origin: metalcapture.OriginReplay, Status: metalcapture.StatusCollected,
				Device: device, Pipeline: identity.Name, PipelineID: identity.ID,
				Value: value, NonAdditive: true,
			})
		}
	}
	return observations, nil
}
