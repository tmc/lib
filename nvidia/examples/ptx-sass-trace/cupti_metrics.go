package main

import (
	"encoding/json"
	"fmt"
	"runtime"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudasass"
	"github.com/tmc/lib/nvidia/x/cuptipmsampling"
	"github.com/tmc/lib/nvidia/x/cuptisass"
)

// sassMetricsResult holds CUPTI SASS metrics data and instruction census.
type sassMetricsResult struct {
	Chip             string         `json:"chip"`
	SupportedMetrics []string       `json:"supported_metrics"`
	Census           map[string]int `json:"instruction_census"`
	RecordsCollected int            `json:"records_collected"`
	Status           string         `json:"status"`
	Details          string         `json:"details,omitempty"`
}

// pmSamplingResult holds CUPTI PM sampling counter availability and status.
type pmSamplingResult struct {
	DeviceIndex      uint32 `json:"device_index"`
	AvailabilitySize int    `json:"availability_size_bytes"`
	Status           string `json:"status"`
	Details          string `json:"details,omitempty"`
}

// collectSassMetrics queries SASS metrics and computes the instruction census
// from the kernel disassemblies.
func collectSassMetrics(
	dev *cuda.Device,
	dis *cudasass.Disassembly,
) (*sassMetricsResult, []byte, error) {
	res := &sassMetricsResult{
		Status: "unsupported",
		Census: make(map[string]int),
	}

	if dis != nil {
		res.Census = cuptisass.InstructionCensus(dis)
	}

	if runtime.GOOS != "linux" {
		res.Status = "skipped_non_linux"
		res.Details = fmt.Sprintf("CUPTI SASS metrics require Linux host; running on %s", runtime.GOOS)
		b, _ := json.MarshalIndent(res, "", "  ")
		return res, b, nil
	}

	chip, err := cuptisass.DeviceChipName(0)
	if err != nil {
		res.Status = "chip_query_failed"
		res.Details = err.Error()
		b, _ := json.MarshalIndent(res, "", "  ")
		return res, b, nil
	}
	res.Chip = chip

	metrics, err := cuptisass.SupportedMetrics(chip)
	if err != nil {
		res.Status = "metrics_query_failed"
		res.Details = err.Error()
	} else {
		for _, m := range metrics {
			res.SupportedMetrics = append(res.SupportedMetrics, m.Name)
		}
		res.Status = "available"
		res.Details = fmt.Sprintf("found %d supported SASS metrics on %s", len(metrics), chip)
	}

	b, _ := json.MarshalIndent(res, "", "  ")
	return res, b, nil
}

// collectPMSampling queries CUPTI PM counter availability for the device.
func collectPMSampling(dev *cuda.Device) (*pmSamplingResult, []byte, error) {
	res := &pmSamplingResult{
		DeviceIndex: 0,
		Status:      "unsupported",
	}

	if runtime.GOOS != "linux" {
		res.Status = "skipped_non_linux"
		res.Details = fmt.Sprintf("CUPTI PM sampling requires Linux host; running on %s", runtime.GOOS)
		b, _ := json.MarshalIndent(res, "", "  ")
		return res, b, nil
	}

	avail, err := cuptipmsampling.CounterAvailability(0)
	if err != nil {
		res.Status = "availability_query_failed"
		res.Details = err.Error()
		b, _ := json.MarshalIndent(res, "", "  ")
		return res, b, nil
	}

	res.AvailabilitySize = len(avail)
	res.Status = "active"
	res.Details = fmt.Sprintf("retrieved %d byte counter availability image", len(avail))

	b, _ := json.MarshalIndent(res, "", "  ")
	return res, b, nil
}
