package cuptibenchcapture

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"math"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cuptibench"
)

// Descriptor identities for the separately timed execution. Timing is not a
// counter: it comes from a different execution of the same workload, so it
// carries its own descriptors and cuptibench's own timing capture identity.
const (
	timingTotal      = "cuptibench/timing.total"
	timingOperations = "cuptibench/timing.operations"
)

// observe records what the pprof profile does not carry: the separately timed
// execution, and the range metrics cuptibench dropped from the profile because
// their additivity or unit was never verified. Metrics already in the profile
// are not repeated here, so a reader cannot add the same counter twice.
func observe(results []*cuptibench.Result) ([]cudacapture.MetricDescriptor, []cudacapture.Observation, error) {
	var descriptors []cudacapture.MetricDescriptor
	var observations []cudacapture.Observation
	byID := make(map[string]cudacapture.MetricDescriptor)
	declare := func(d cudacapture.MetricDescriptor) error {
		if prior, ok := byID[d.ID]; ok {
			if prior != d {
				return fmt.Errorf("cuptibenchcapture: conflicting descriptions of metric %q", d.ID)
			}
			return nil
		}
		byID[d.ID] = d
		descriptors = append(descriptors, d)
		return nil
	}
	var walk func(r *cuptibench.Result) error
	walk = func(r *cuptibench.Result) error {
		if r.Timing != nil {
			method := r.Timing.Method
			if method == "" {
				method = "unspecified"
			}
			total := cudacapture.MetricDescriptor{
				ID:          timingTotal,
				Namespace:   "cuptibench",
				Name:        "timing_total",
				Definition:  "elapsed time of the timed execution, which is a different execution of the workload from the counter replay",
				Unit:        "nanoseconds",
				NumericType: "int64",
				Aggregation: "sum",
				Scope:       "benchmark",
				Denominator: timingOperations,
				Window:      "uninstrumented timed execution",
				Collector:   method,
			}
			operations := cudacapture.MetricDescriptor{
				ID:          timingOperations,
				Namespace:   "cuptibench",
				Name:        "timing_operations",
				Definition:  "logical operations the timed execution completed; the denominator for timing_total",
				Unit:        "operations",
				NumericType: "int64",
				Aggregation: "sum",
				Scope:       "benchmark",
				Window:      "uninstrumented timed execution",
				Collector:   method,
			}
			if err := declare(total); err != nil {
				return err
			}
			if err := declare(operations); err != nil {
				return err
			}
			// cuptibench names the timed execution separately from the counter
			// capture; reuse that identity rather than inventing another.
			entity := r.Metadata["timing_capture_id"]
			if entity == "" {
				return fmt.Errorf("cuptibenchcapture: capture %q reports timing without a timing capture identity", r.Metadata["capture_id"])
			}
			for _, o := range []struct {
				id    string
				value int64
			}{{timingTotal, int64(r.Timing.Total)}, {timingOperations, r.Timing.Operations}} {
				value, err := json.Marshal(o.value)
				if err != nil {
					return err
				}
				observations = append(observations, cudacapture.Observation{
					DescriptorID: o.id,
					EntityID:     entity,
					RunID:        r.Metadata["run_id"],
					Value:        value,
					Availability: "collected",
				})
			}
		}
		for _, rng := range r.Ranges {
			for _, m := range rng.Metrics {
				// The condition cuptibench applies before writing a counter
				// into pprof; see x/cuptibench/export.go. What it drops there
				// would otherwise survive only as prose.
				if m.Additive && m.Unit != "" && m.Unit != "unknown" {
					continue
				}
				if math.IsNaN(m.Value) || math.IsInf(m.Value, 0) {
					return fmt.Errorf("cuptibenchcapture: metric %q of range %q is not finite", m.Name, rng.ID)
				}
				unit := m.Unit
				if unit == "" {
					unit = "unknown"
				}
				d := cudacapture.MetricDescriptor{
					ID:          "cupti/" + m.Name,
					Namespace:   "cupti",
					Name:        m.Name,
					Definition:  "native CUPTI range metric kept out of the pprof profile because its additivity or unit was not verified; it is scoped to the range that collected it and is not an instruction count",
					Unit:        unit,
					NumericType: "float64",
					Aggregation: "unverified",
					Scope:       "range",
					Window:      "counter replay",
					Collector:   "cupti-range-profiler",
				}
				if err := declare(d); err != nil {
					return err
				}
				value, err := json.Marshal(m.Value)
				if err != nil {
					return err
				}
				observations = append(observations, cudacapture.Observation{
					DescriptorID: d.ID,
					EntityID:     r.Metadata["capture_id"] + "/" + rng.ID,
					RunID:        r.Metadata["run_id"],
					Value:        value,
					Availability: "collected",
				})
			}
		}
		if r.Aggregate != nil {
			return walk(r.Aggregate)
		}
		return nil
	}
	for _, r := range results {
		if err := walk(r); err != nil {
			return nil, nil, err
		}
	}
	return descriptors, observations, nil
}

// file records one retained artifact and the digest a reader can recompute.
type file struct {
	Name   string `json:"name"`
	SHA256 string `json:"sha256"`
}

type moduleProvenance struct {
	ID          string          `json:"id"`
	Name        string          `json:"name,omitempty"`
	Generation  uint64          `json:"generation,omitempty"`
	ImageSHA256 string          `json:"loaded_image_sha256"`
	Files       map[string]file `json:"files"`
	Kernels     []string        `json:"kernels,omitempty"`
}

type captureProvenance struct {
	CaptureID          string `json:"capture_id"`
	Benchmark          string `json:"benchmark"`
	RunID              string `json:"run_id"`
	Mode               string `json:"mode"`
	View               string `json:"view,omitempty"`
	ReplayMode         string `json:"replay_mode,omitempty"`
	Metrics            string `json:"metrics,omitempty"`
	Operations         int    `json:"operations"`
	Chip               string `json:"chip,omitempty"`
	DeviceUUID         string `json:"device_uuid,omitempty"`
	DriverVersion      string `json:"cuda_driver_version,omitempty"`
	CUPTIVersion       string `json:"cupti_version,omitempty"`
	Preparation        string `json:"preparation,omitempty"`
	TimingCaptureID    string `json:"timing_capture_id,omitempty"`
	AggregateCaptureID string `json:"aggregate_capture_id,omitempty"`
}

type composition struct {
	Composition string              `json:"composition"`
	Profile     string              `json:"cuptibench_profile"`
	Results     string              `json:"cuptibench_results_file"`
	Module      moduleProvenance    `json:"module"`
	Captures    []captureProvenance `json:"captures"`
	Notes       []string            `json:"notes"`
}

// describe records where the capture came from and what its numbers do not mean.
func describe(e Export, a Artifacts, results benchResults, kernels []string) (json.RawMessage, error) {
	m := composition{
		Composition: "cuptibench counter profile packaged with the code the benchmark executed",
		Profile:     e.Name,
		Results:     a.Prefix + ".cuptibench.json",
		Module: moduleProvenance{
			ID:          "sha256:" + a.ImageSHA256,
			Name:        a.Name,
			Generation:  a.Generation,
			ImageSHA256: a.ImageSHA256,
			Files: map[string]file{
				"source": {a.Prefix + ".cu", digest(a.Source)},
				"ptx":    {a.Prefix + ".ptx", digest(a.PTX)},
				"cubin":  {a.Prefix + ".cubin", digest(a.CUBIN)},
				"sass":   {a.Prefix + ".sass", digest(a.SASS)},
			},
			Kernels: kernels,
		},
		Notes: []string{
			"the counters were collected by counter replay; the benchmark timing is a separate uninstrumented execution and carries its own capture identity",
			"module identity is the digest of the image the producer loaded, recomputed here over the retained cubin; it is a producer record, not a driver attestation",
			"counters are scoped to the ranges cuptibench selected for each capture's mode; a region or kernel range counter is not an instruction count",
			"captures and views are separate experiments and are not additive; ranges cuptibench excluded from the profile remain in the attached results file",
		},
	}
	for _, r := range results.Results {
		m.Captures = append(m.Captures, captureProvenance{
			CaptureID:          r.Metadata["capture_id"],
			Benchmark:          r.Metadata["benchmark"],
			RunID:              r.Metadata["run_id"],
			Mode:               r.Metadata["mode"],
			View:               r.Metadata["view"],
			ReplayMode:         r.Metadata["replay_mode"],
			Metrics:            r.Metadata["metrics"],
			Operations:         r.Operations,
			Chip:               r.Metadata["chip"],
			DeviceUUID:         r.Metadata["device_uuid"],
			DriverVersion:      r.Metadata["cuda_driver_version"],
			CUPTIVersion:       r.Metadata["cupti_version"],
			Preparation:        r.Metadata["preparation"],
			TimingCaptureID:    r.Metadata["timing_capture_id"],
			AggregateCaptureID: r.Metadata["aggregate_capture_id"],
		})
	}
	data, err := json.Marshal(m)
	if err != nil {
		return nil, fmt.Errorf("cuptibenchcapture: describe composition: %w", err)
	}
	return data, nil
}

func digest(data []byte) string { return fmt.Sprintf("%x", sha256.Sum256(data)) }
