package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"runtime"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudasass"
	"github.com/tmc/lib/nvidia/x/cuptipmsampling"
	"github.com/tmc/lib/nvidia/x/cuptisass"
)

// Status values name what a run established, never what a collector would
// have established had one run. This file only discovers what a device
// reports as supported and counts instructions in a disassembly; it starts no
// CUPTI collector, so no status here may read as "hardware data was
// collected".
const (
	statusNotQueried        = "not_queried"
	statusSkippedNonLinux   = "skipped_non_linux"
	statusNoDevice          = "no_device"
	statusChipQueryFailed   = "chip_query_failed"
	statusDiscoveryFailed   = "metric_discovery_failed"
	statusMetricsListed     = "supported_metrics_listed"
	statusAvailQueryFailed  = "availability_query_failed"
	statusAvailabilityShown = "counter_availability_retrieved"
)

// collectionNotRun marks a collector this example never starts. It is a
// separate field from the status so a reader cannot take a successful
// discovery as evidence that samples or records exist.
const collectionNotRun = "not_run"

// These are variables so the tests can exercise device plumbing and status
// selection on a host with no CUPTI and no GPU.
var (
	deviceIndex         = cuptiDeviceIndex
	chipName            = cuptisass.DeviceChipName
	supportedMetrics    = cuptisass.SupportedMetrics
	counterAvailability = cuptipmsampling.CounterAvailability
	hostOS              = runtime.GOOS
)

// sassMetricsResult reports three separate things that must not be conflated:
// a static instruction census computed from a disassembly, the metrics the
// device reports as supported, and whether a native SASS collection ran.
// SassCollection is always collectionNotRun here, so every count in this
// struct is static or declarative and none of it is measured hardware data.
type sassMetricsResult struct {
	DeviceIndex      uint32         `json:"device_index"`
	Chip             string         `json:"chip"`
	SupportedMetrics []string       `json:"supported_metrics_discovered"`
	Census           map[string]int `json:"static_instruction_census"`
	SassCollection   string         `json:"sass_collection"`
	Status           string         `json:"discovery_status"`
	Details          string         `json:"details,omitempty"`
}

// pmSamplingResult reports whether a PM counter availability image could be
// retrieved for the device. Availability is not collection: Sampling is always
// collectionNotRun here because this example starts no PM sampler, and
// AvailabilitySize is the size of a capability image, not a sample count.
type pmSamplingResult struct {
	DeviceIndex      uint32 `json:"device_index"`
	AvailabilitySize int    `json:"availability_size_bytes"`
	Sampling         string `json:"pm_sampling"`
	Status           string `json:"availability_status"`
	Details          string `json:"details,omitempty"`
}

// cuptiDeviceIndex returns the CUPTI device index for dev. CUPTI indexes
// devices by the same ordinal the CUDA driver uses, so a caller that supplies
// a device must have its queries answered for that device and not for
// whichever device happens to be ordinal 0.
func cuptiDeviceIndex(dev *cuda.Device) (uint32, error) {
	if dev == nil {
		return 0, errors.New("no device supplied")
	}
	ord := dev.Ordinal()
	if ord < 0 {
		return 0, fmt.Errorf("device ordinal %d is negative", ord)
	}
	return uint32(ord), nil
}

// collectSassMetrics counts instructions in dis and asks the supplied device
// which SASS metrics it supports. It does not run the CUPTI SASS collector.
func collectSassMetrics(
	dev *cuda.Device,
	dis *cudasass.Disassembly,
) (*sassMetricsResult, []byte, error) {
	res := &sassMetricsResult{
		Status:         statusNotQueried,
		SassCollection: collectionNotRun,
		Census:         make(map[string]int),
	}

	// The census is static: it reads a disassembly and needs neither a device
	// nor CUPTI, so it is computed even when discovery below cannot run.
	if dis != nil {
		res.Census = cuptisass.InstructionCensus(dis)
	}

	index, err := deviceIndex(dev)
	if err != nil {
		res.Status = statusNoDevice
		res.Details = fmt.Sprintf("cannot discover supported metrics: %v", err)
		return res, marshal(res), nil
	}
	res.DeviceIndex = index

	if hostOS != "linux" {
		res.Status = statusSkippedNonLinux
		res.Details = fmt.Sprintf("CUPTI SASS metrics require Linux host; running on %s", hostOS)
		return res, marshal(res), nil
	}

	chip, err := chipName(index)
	if err != nil {
		res.Status = statusChipQueryFailed
		res.Details = err.Error()
		return res, marshal(res), nil
	}
	res.Chip = chip

	metrics, err := supportedMetrics(chip)
	if err != nil {
		res.Status = statusDiscoveryFailed
		res.Details = err.Error()
		return res, marshal(res), nil
	}
	for _, m := range metrics {
		res.SupportedMetrics = append(res.SupportedMetrics, m.Name)
	}
	res.Status = statusMetricsListed
	res.Details = fmt.Sprintf("device %d (%s) reports %d supported SASS metrics; no SASS collection was run",
		index, chip, len(metrics))
	return res, marshal(res), nil
}

// collectPMSampling retrieves the CUPTI PM counter availability image for the
// supplied device. It does not start a PM sampler and collects no samples.
func collectPMSampling(dev *cuda.Device) (*pmSamplingResult, []byte, error) {
	res := &pmSamplingResult{
		Status:   statusNotQueried,
		Sampling: collectionNotRun,
	}

	index, err := deviceIndex(dev)
	if err != nil {
		res.Status = statusNoDevice
		res.Details = fmt.Sprintf("cannot query counter availability: %v", err)
		return res, marshal(res), nil
	}
	res.DeviceIndex = index

	if hostOS != "linux" {
		res.Status = statusSkippedNonLinux
		res.Details = fmt.Sprintf("CUPTI PM sampling requires Linux host; running on %s", hostOS)
		return res, marshal(res), nil
	}

	avail, err := counterAvailability(index)
	if err != nil {
		res.Status = statusAvailQueryFailed
		res.Details = err.Error()
		return res, marshal(res), nil
	}

	res.AvailabilitySize = len(avail)
	res.Status = statusAvailabilityShown
	res.Details = fmt.Sprintf("device %d returned a %d byte counter availability image; no PM samples were collected",
		index, len(avail))
	return res, marshal(res), nil
}

// marshal renders a diagnostic result for the capture bundle. A diagnostic
// that cannot be rendered is reported in place rather than dropped, so the
// bundle never contains a silently empty file.
func marshal(v any) []byte {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return []byte(fmt.Sprintf("{%q: %q}\n", "marshal_error", err.Error()))
	}
	return b
}
