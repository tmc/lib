package cudaflightrecorder

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math/big"
	"reflect"
	"strings"
	"sync"
	"time"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

// WorkloadTriggerConfig freezes one serving series' observation profile.
// Mode is "counter_delta" or "histogram_mean"; Metric is the counter name or
// histogram base name. Labels selects exactly one series, excluding histogram
// le labels. Unit is the counter or histogram sum unit. WindowDuration and
// TimeSource ("producer" or "collector_observation") must remain fixed.
// MinPopulation must be positive for a histogram mean. StaticThreshold is a
// separate high-side comparison in Unit, not an SLO or capture admission policy.
//
// Provider/Version must be supported by nvtelemetry.ReadServingExposition:
// Triton 25.06, vLLM 0.8.5, Ray 2.44.0, SGLang 0.5.21 or Dynamo 1.5.1.
// Training exports and arbitrary WorkloadExport producers are not supported.
type WorkloadTriggerConfig struct {
	Provider, Version, Metric, Unit string
	Mode, TimeSource                string
	Labels                          map[string]string
	WindowDuration                  time.Duration
	MinPopulation                   uint64
	StaticThreshold                 float64
	Detector                        DetectorConfig
	Limits                          nvtelemetry.WorkloadLimits
	TenantID                        string
	DeviceOrdinal                   uint32
}

// WorkloadTriggerWindow names the exact aggregate observation interval and
// clock. Collector observation times are caller assertions, never sensor or
// request event times. Phase, run and rank remain in the supplied exports.
type WorkloadTriggerWindow struct {
	StartNanos, EndNanos      int64
	ClockDomain, ClockEpochID string
}

// WorkloadTriggerResult retains raw/normalized reader evidence, exact window
// values and the Detector's float64 representation separately. Rounded reports
// conversion rounding. Population is empty for counters. StaticExceeded is not
// an individual-request SLO result. Decision never implies capture admission;
// Incident is present only when an anomaly was sent to the existing coordinator.
// Error describes validation failure or capture refusal/failure.
type WorkloadTriggerResult struct {
	Config         WorkloadTriggerConfig
	Window         WorkloadTriggerWindow
	Before, After  nvtelemetry.WorkloadExport
	Series         []nvtelemetry.WorkloadDelta
	ExactValue     string
	Population     string
	Rounded        bool
	StaticExceeded bool
	Observation    DetectorObservation
	Decision       DetectorDecision
	Incident       *IncidentReceipt
	Error          string
}

// WorkloadTrigger owns one series' Detector history and uses an existing
// coordinator for capture admission. Its zero value is unusable. Do not copy it.
// Observe is synchronous and serialized; the coordinator's existing capture
// budgets and cooperative provider/writer cancellation remain authoritative.
// Use separate instances for concurrent series. Invalid observations reset only
// this instance; run/phase/epoch/bucket changes restart its warmup.
type WorkloadTrigger struct {
	mu          sync.Mutex
	config      WorkloadTriggerConfig
	detector    *Detector
	coordinator *EscalationCoordinator
}

// NewWorkloadTrigger freezes a series profile and creates its existing Detector.
// Provider/version admission remains with the serving reader during Observe.
func NewWorkloadTrigger(c WorkloadTriggerConfig, coordinator *EscalationCoordinator) (*WorkloadTrigger, error) {
	if coordinator == nil || c.WindowDuration <= 0 || c.Detector.MaxGap < c.WindowDuration || !finiteDetector(c.StaticThreshold) || c.StaticThreshold < 0 {
		return nil, errors.New("cudaflightrecorder: invalid workload trigger dependency, window or threshold")
	}
	for _, s := range []string{c.Provider, c.Version, c.Metric, c.Unit, c.TenantID} {
		if strings.TrimSpace(s) == "" || len(s) > 256 {
			return nil, errors.New("cudaflightrecorder: invalid workload trigger identity")
		}
	}
	if c.TimeSource != "producer" && c.TimeSource != "collector_observation" {
		return nil, errors.New("cudaflightrecorder: invalid workload trigger time source")
	}
	if c.Mode != "counter_delta" && c.Mode != "histogram_mean" || c.Mode == "histogram_mean" && c.MinPopulation == 0 {
		return nil, errors.New("cudaflightrecorder: invalid workload trigger mode or population")
	}
	if len(c.Labels) > 64 {
		return nil, errors.New("cudaflightrecorder: workload trigger label limit exceeded")
	}
	for k, v := range c.Labels {
		if k == "" || k == "le" || len(k) > 256 || len(v) > 256 {
			return nil, errors.New("cudaflightrecorder: invalid workload trigger labels")
		}
	}
	d, err := NewDetector(c.Detector)
	if err != nil {
		return nil, err
	}
	c.Labels = copyWorkloadTriggerLabels(c.Labels)
	return &WorkloadTrigger{config: c, detector: d, coordinator: coordinator}, nil
}

func copyWorkloadTriggerLabels(labels map[string]string) map[string]string {
	out := make(map[string]string, len(labels))
	for k, v := range labels {
		out[k] = v
	}
	return out
}

// Observe authenticates parsed exports against their bounded serving-reader raw
// bytes, validates a counter/histogram window, and feeds its value to Detector.
// Collector timestamps are reproduced only through StampWorkloadSnapshot; native
// timestamps are never overwritten. Missing/reset/invalid data clears warmup
// without adding a healthy zero. All returned raw bytes are owned by the result.
// The raw SHA references are also retained in incident trigger details.
// The full exact rational is returned in ExactValue; incident details retain its
// SHA-256 so high precision cannot exceed the coordinator's detail-string limit.
func (t *WorkloadTrigger) Observe(ctx context.Context, before, after nvtelemetry.WorkloadExport, window WorkloadTriggerWindow) (result WorkloadTriggerResult, err error) {
	if t == nil || t.detector == nil || t.coordinator == nil {
		return result, errors.New("cudaflightrecorder: unconfigured workload trigger")
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	c := t.config
	result.Config, result.Window = c, window
	result.Config.Labels = copyWorkloadTriggerLabels(c.Labels)
	invalid := func(err error) (WorkloadTriggerResult, error) {
		result.Error = err.Error()
		result.Decision = t.detector.Observe(DetectorObservation{Available: false})
		return result, err
	}
	if ctx == nil {
		return invalid(errors.New("cudaflightrecorder: nil workload trigger context"))
	}
	if err := ctx.Err(); err != nil {
		return invalid(err)
	}
	if window.StartNanos < 0 || window.EndNanos <= window.StartNanos || window.EndNanos-window.StartNanos != int64(c.WindowDuration) || window.ClockDomain == "" || len(window.ClockDomain) > 256 || window.ClockEpochID == "" {
		return invalid(errors.New("cudaflightrecorder: incompatible workload trigger window"))
	}
	result.Before, err = t.read(ctx, before, window.StartNanos, window)
	if err != nil {
		return invalid(err)
	}
	result.After, err = t.read(ctx, after, window.EndNanos, window)
	if err != nil {
		return invalid(err)
	}
	var deltas []nvtelemetry.WorkloadDelta
	if c.Mode == "counter_delta" {
		deltas, err = nvtelemetry.CounterWorkloadWindow(result.Before, result.After, window.StartNanos, window.EndNanos, window.ClockDomain, window.ClockEpochID)
	} else {
		deltas, err = nvtelemetry.HistogramWorkloadWindow(result.Before, result.After, window.StartNanos, window.EndNanos, window.ClockDomain, window.ClockEpochID)
	}
	if err != nil {
		return invalid(fmt.Errorf("cudaflightrecorder: workload trigger window: %w", err))
	}
	value, err := t.value(deltas, &result)
	if err != nil {
		return invalid(err)
	}
	result.ExactValue = value.RatString()
	v, exact := value.Float64()
	if !finiteDetector(v) || v == 0 && value.Sign() != 0 {
		return invalid(errors.New("cudaflightrecorder: workload trigger value outside float64 range"))
	}
	result.Rounded = !exact
	result.StaticExceeded = (nvtelemetry.TriggerPolicy{MetricName: c.Metric, Threshold: c.StaticThreshold, Comparison: ">"}).Matches(nvtelemetry.Reading{Name: c.Metric, Unit: c.Unit, Evidence: nvtelemetry.EvidenceDerived, Value: v})
	// Values/timestamps are excluded: this fingerprint identifies comparable
	// provider/run/phase/rank/clock/series/bucket schema, not a particular sample.
	series := append([]nvtelemetry.WorkloadDelta(nil), result.Series...)
	for i := range series {
		series[i].Delta, series[i].StartNanos, series[i].EndNanos = "", 0, 0
	}
	identity, err := json.Marshal(struct {
		Source nvtelemetry.WorkloadSource
		Series []nvtelemetry.WorkloadDelta
	}{result.After.Source, series})
	if err != nil {
		return invalid(err)
	}
	h := sha256.Sum256(identity)
	result.Observation = DetectorObservation{Metric: c.Metric, Unit: c.Unit, Epoch: hex.EncodeToString(h[:]), Phase: result.After.Source.Phase, Clock: window.ClockDomain, Timestamp: window.EndNanos, Value: v, Available: true}
	result.Decision = t.detector.Observe(result.Observation)
	if !result.Decision.Anomaly {
		return result, nil
	}
	valueDigest := sha256.Sum256([]byte(result.ExactValue))
	result.Incident, err = t.coordinator.Trigger(ctx, TriggerRequest{
		TenantID: c.TenantID, DeviceOrdinal: c.DeviceOrdinal, Reason: result.Decision.Reason,
		MetricName: c.Metric, MetricValue: v, Threshold: result.Decision.Threshold,
		Details: map[string]string{
			"before_sha256": result.Before.RawSHA256, "after_sha256": result.After.RawSHA256,
			"window_start_ns": fmt.Sprint(window.StartNanos), "window_end_ns": fmt.Sprint(window.EndNanos),
			"clock_domain": window.ClockDomain, "clock_epoch": window.ClockEpochID,
			"provider": c.Provider, "provider_version": c.Version, "run_id": result.After.Source.RunID,
			"phase": result.After.Source.Phase, "unit": c.Unit, "mode": c.Mode,
			"time_source": c.TimeSource, "exact_value_sha256": hex.EncodeToString(valueDigest[:]), "population": result.Population,
			"series_fingerprint": result.Observation.Epoch,
		},
	})
	if err != nil {
		result.Error = err.Error()
	}
	return result, err
}

func (t *WorkloadTrigger) read(ctx context.Context, supplied nvtelemetry.WorkloadExport, timestamp int64, window WorkloadTriggerWindow) (nvtelemetry.WorkloadExport, error) {
	c := t.config
	if supplied.Source.Provider != c.Provider || supplied.Source.Version != c.Version || len(supplied.Source.Phase) > 256 {
		return nvtelemetry.WorkloadExport{}, errors.New("cudaflightrecorder: workload trigger source mismatch")
	}
	e, err := nvtelemetry.ReadServingExposition(ctx, bytes.NewReader(supplied.Raw), supplied.Source, c.Limits)
	if err != nil {
		return e, err
	}
	if c.TimeSource == "collector_observation" {
		e, err = nvtelemetry.StampWorkloadSnapshot(e, timestamp, window.ClockDomain, window.ClockEpochID)
		if err != nil {
			return e, err
		}
	}
	if !reflect.DeepEqual(e, supplied) {
		return e, errors.New("cudaflightrecorder: workload trigger export contradicts raw reader evidence")
	}
	return e, nil
}

func (t *WorkloadTrigger) value(deltas []nvtelemetry.WorkloadDelta, result *WorkloadTriggerResult) (*big.Rat, error) {
	c := t.config
	var sum, count *big.Rat
	for _, p := range deltas {
		labels := copyWorkloadTriggerLabels(p.Labels)
		if c.Mode == "histogram_mean" && p.Kind == "histogram_bucket" {
			delete(labels, "le")
		}
		if !reflect.DeepEqual(labels, c.Labels) {
			continue
		}
		if c.Mode == "counter_delta" && p.Name == c.Metric && p.Kind == "counter" && p.Unit == c.Unit {
			result.Series = append(result.Series, p)
			sum, _ = new(big.Rat).SetString(p.Delta)
		}
		if c.Mode == "histogram_mean" && (p.Name == c.Metric+"_bucket" || p.Name == c.Metric+"_count" || p.Name == c.Metric+"_sum") {
			result.Series = append(result.Series, p)
			if p.Kind == "histogram_sum" && p.Unit == c.Unit {
				sum, _ = new(big.Rat).SetString(p.Delta)
			}
			if p.Kind == "histogram_count" && p.Unit == "observations" {
				count, _ = new(big.Rat).SetString(p.Delta)
			}
		}
	}
	if sum == nil {
		return nil, errors.New("cudaflightrecorder: workload trigger series or unit unavailable")
	}
	if c.Mode == "histogram_mean" {
		minimum := new(big.Rat).SetInt(new(big.Int).SetUint64(c.MinPopulation))
		if count == nil || count.Cmp(minimum) < 0 {
			return nil, errors.New("cudaflightrecorder: workload trigger population unavailable")
		}
		result.Population = count.RatString()
		sum.Quo(sum, count)
	}
	return sum, nil
}
