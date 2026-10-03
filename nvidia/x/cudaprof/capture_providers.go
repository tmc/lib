package cudaprof

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"reflect"
	"strings"

	"github.com/tmc/lib/nvidia/x/cuptipmsampling"
	"github.com/tmc/lib/nvidia/x/cuptirange"
	"github.com/tmc/lib/nvidia/x/cuptisass"
)

// RangeCaptureProviderResult converts an owner's capture into transport evidence.
// receipt supplies target identity, environment, pass and native time boundaries.
// configImage is the actual Plan.Configuration image; metrics are the names
// retained by the caller from its NewPlan request, in scheduling order.
// CleanupComplete reports the caller's successful Close, not an adapter action.
// A capture returned with an error remains diagnostic; the caller must retain
// ownership and retry Close until cleanup succeeds. No workload is executed here.
func RangeCaptureProviderResult(receipt CaptureProviderResult, capture *cuptirange.Capture, configImage []byte, metrics []string, collectionErr error) (CaptureProviderResult, error) {
	r, err := providerReceipt(receipt, "range", "target-context")
	if err != nil {
		return CaptureProviderResult{}, err
	}
	r.Status = "partial"
	r.Reason = "range collection incomplete"
	if capture == nil {
		r.Status = "unavailable"
		r.Reason = "range capture unavailable"
		if collectionErr != nil {
			r.Reason = collectionErr.Error()
		}
		return r, nil
	}
	r.Config, err = observedProviderConfig(configImage, metrics)
	if err != nil {
		return CaptureProviderResult{}, err
	}
	raw := struct {
		Image           []byte                       `json:"image"`
		DiagnosticImage []byte                       `json:"diagnostic_image,omitempty"`
		Quality         cuptirange.CollectionQuality `json:"quality"`
		Ranges          uint                         `json:"ranges"`
		RangeInfo       []cuptirange.RangeInfo       `json:"range_info,omitempty"`
		DiagnosticTrace *cuptirange.LaunchTrace      `json:"diagnostic_trace,omitempty"`
	}{Image: capture.Image, DiagnosticImage: capture.DiagnosticImage, Quality: capture.Quality, Ranges: capture.Ranges, RangeInfo: capture.RangeInfo, DiagnosticTrace: capture.DiagnosticTrace}
	r.Raw, err = marshalOwnerRaw(raw)
	if err != nil {
		return CaptureProviderResult{}, fmt.Errorf("encode range provider: %w", err)
	}
	q := capture.Quality
	r.NativePasses = &CaptureNativePasses{Started: uint64(q.PassesStarted), Stopped: uint64(q.PassesStopped), AllSubmitted: q.AllPassesSubmitted}
	r.Decoded = q.Decoded
	if q.Decoded {
		r.Counts.Dropped = count(q.DroppedRanges)
	}
	if collectionErr == nil && q.Decoded {
		r.Counts.Retained = count(uint64(capture.Ranges))
		if uint64(capture.Ranges) > ^uint64(0)-q.DroppedRanges {
			return CaptureProviderResult{}, fmt.Errorf("range provider: record count overflow")
		}
		r.Counts.Offered = count(uint64(capture.Ranges) + q.DroppedRanges)
		// Range records have neither PC non-user samples nor anonymous PC names.
		r.Counts.NonUser = count(0)
		r.Counts.Unnamed = count(0)
	}
	if collectionErr != nil {
		r.Reason = "range collection: " + collectionErr.Error()
		return r, nil
	}
	if !rangeInventoryComplete(capture) {
		r.Reason = "range attribution inventory incomplete"
		return r, nil
	}
	if q.Decoded && q.AllPassesSubmitted && q.PassesStarted > 0 && q.PassesStarted == q.PassesStopped && capture.Ranges > 0 && len(capture.Image) > 0 && len(capture.DiagnosticImage) == 0 && q.DroppedRanges == 0 && r.CleanupComplete {
		r = completeOwnerResult(r)
	}
	return r, nil
}

// PMCaptureProviderResult converts actual Decode, SampleInfo and CounterData
// results collected by the target caller before closing its PM session. PM is
// device-wide: this result does not attribute device counters to a child kernel.
// config is the actual Session Config retained by
// the caller. Both adapters admit at most 64 MiB of configuration and 128
// metric names of at most 256 bytes each.
// receipt supplies collection boundaries and successful cleanup. collectionErr
// must include failures from Start, Stop, Decode, sample queries and Close.
// No session is started, stopped, decoded or closed by this conversion.
func PMCaptureProviderResult(receipt CaptureProviderResult, decoded cuptipmsampling.DecodeResult, samples []cuptipmsampling.SampleInfo, image []byte, config cuptipmsampling.Config, collectionErr error) (CaptureProviderResult, error) {
	r, err := providerReceipt(receipt, "pm", "device-wide")
	if err != nil {
		return CaptureProviderResult{}, err
	}
	r.Config, err = observedProviderConfig(config.ConfigImage, config.MetricNames)
	if err != nil {
		return CaptureProviderResult{}, err
	}
	if config.SamplingInterval == 0 || config.MaxSamples == 0 || len(samples) > int(config.MaxSamples) {
		return CaptureProviderResult{}, fmt.Errorf("pm provider: invalid sampling limits")
	}
	r.Config.SamplingInterval = config.SamplingInterval
	r.Config.TriggerMode = int32(config.TriggerMode)
	r.Config.AppendMode = int32(config.AppendMode)
	r.Status = "partial"
	r.Reason = "PM collection incomplete"
	raw := struct {
		Image   []byte                       `json:"image"`
		Decode  cuptipmsampling.DecodeResult `json:"decode"`
		Samples []cuptipmsampling.SampleInfo `json:"samples"`
		Config  cuptipmsampling.Config       `json:"config"`
	}{image, decoded, samples, config}
	r.Raw, err = marshalOwnerRaw(raw)
	if err != nil {
		return CaptureProviderResult{}, fmt.Errorf("encode PM provider: %w", err)
	}
	if collectionErr != nil {
		r.Reason = "PM collection: " + collectionErr.Error()
		return r, nil
	}
	info := decoded.Info
	if info.NumCompletedSamples > info.NumPopulatedSamples || info.NumPopulatedSamples > info.NumTotalSamples {
		return CaptureProviderResult{}, fmt.Errorf("pm provider: inconsistent sample counts")
	}
	r.Decoded = true
	r.Counts.Retained = count(uint64(info.NumCompletedSamples))
	// NumTotalSamples is counter-image capacity, not attempted samples.
	// CUPTI does not expose an offered-sample denominator in this result.
	r.Counts.Offered = nil
	// The PM API reports overflow, not a dropped-record count. Even with no
	// overflow, the unpopulated capacity is not a loss denominator.
	if !decoded.Overflow {
		r.Counts.Dropped = count(0)
	}
	r.Counts.NonUser = count(0)
	r.Counts.Unnamed = count(0)
	if uint(len(samples)) != info.NumCompletedSamples {
		r.Reason = "PM sample boundaries incomplete"
		return r, nil
	}
	for i, s := range samples {
		if s.Index != uint(i) || s.StartTimestamp == 0 || s.EndTimestamp < s.StartTimestamp {
			return CaptureProviderResult{}, fmt.Errorf("pm provider: invalid sample boundary")
		}
	}
	if decoded.Overflow {
		r.Reason = "PM hardware overflow; dropped count unavailable"
		return r, nil
	}
	if decoded.StopReason != cuptipmsampling.DecodeStopReasonEndOfRecords {
		r.Reason = "PM decode did not reach end of records"
		return r, nil
	}
	if info.NumCompletedSamples > 0 && info.NumCompletedSamples == info.NumPopulatedSamples && len(image) > 0 && r.CleanupComplete {
		r.Reason = "PM offered-sample denominator unavailable"
	}
	return r, nil
}

func providerReceipt(r CaptureProviderResult, provider CaptureProvider, scope string) (CaptureProviderResult, error) {
	if strings.TrimSpace(r.PassID) == "" || strings.TrimSpace(r.Replay) == "" {
		return CaptureProviderResult{}, fmt.Errorf("provider receipt: missing pass or replay")
	}
	for _, digest := range []string{r.Identity.ArtifactDigest, r.Identity.CompilerKey, r.Identity.WorkloadKey} {
		if digest != "" {
			if err := captureDigest(digest); err != nil {
				return CaptureProviderResult{}, err
			}
		}
	}
	if r.Stop < r.Start {
		return CaptureProviderResult{}, fmt.Errorf("provider receipt: invalid boundaries")
	}
	r.NativePasses = nil
	r.Provider = provider
	if r.Scope == "" {
		r.Scope = scope
	}
	r.Counts = CaptureCounts{}
	r.Values = nil
	r.Raw = nil
	r.Decoded = false
	r.EvidenceClass = "measured"
	return r, nil
}

func completeOwnerResult(r CaptureProviderResult) CaptureProviderResult {
	complete := r
	complete.Status = "complete"
	complete.Reason = ""
	if validateCaptureResult(complete) == nil {
		return complete
	}
	r.Reason = "provider provenance or collection qualification incomplete"
	return r
}

func count(n uint64) *uint64 { return &n }

// Configuration evidence is bounded separately from the manifest budget.
// Metric order is the actual order scheduled by the caller, not a sorted alias.
func observedProviderConfig(image []byte, metrics []string) (CaptureProviderConfig, error) {
	if len(image) == 0 || len(image) > 64<<20 || len(metrics) == 0 || len(metrics) > 128 {
		return CaptureProviderConfig{}, fmt.Errorf("provider config: invalid image or metric count")
	}
	seen := make(map[string]bool, len(metrics))
	for _, metric := range metrics {
		if strings.TrimSpace(metric) == "" || len(metric) > 256 || seen[metric] {
			return CaptureProviderConfig{}, fmt.Errorf("provider config: invalid metric name")
		}
		seen[metric] = true
	}
	digest := sha256.Sum256(image)
	return CaptureProviderConfig{ConfigDigest: hex.EncodeToString(digest[:]), Metrics: append([]string(nil), metrics...)}, nil
}

// SASSCaptureProviderResult transports actual SASS Config, drained records and
// the dropped-record count returned by Session.Disable. Nil dropped means the
// loss count was not obtained. The caller owns Disable and Close, including
// retries after cleanup failure. Instruction values stay exact uint64 in Raw;
// this helper does not reinterpret them as instruction costs or local timing.
func SASSCaptureProviderResult(receipt CaptureProviderResult, config cuptisass.Config, records []cuptisass.Record, dropped *uint64, collectionErr error) (CaptureProviderResult, error) {
	r, err := providerReceipt(receipt, "sass", "target-context")
	if err != nil {
		return CaptureProviderResult{}, err
	}
	if len(config.Metrics) == 0 || len(config.Metrics) > 128 {
		return CaptureProviderResult{}, fmt.Errorf("sass provider: invalid metric count")
	}
	r.Config = CaptureProviderConfig{LazyPatching: config.LazyPatching}
	allowed := make(map[uint64]bool, len(config.Metrics))
	for _, metric := range config.Metrics {
		if allowed[metric.MetricID] || metric.Granularity.String() == "Unknown" {
			return CaptureProviderResult{}, fmt.Errorf("sass provider: invalid or duplicate metric")
		}
		allowed[metric.MetricID] = true
		r.Config.SASSMetrics = append(r.Config.SASSMetrics, CaptureSASSMetric{MetricID: metric.MetricID, Granularity: uint8(metric.Granularity)})
	}
	raw := struct {
		Config  cuptisass.Config   `json:"config"`
		Records []cuptisass.Record `json:"records"`
		Dropped *uint64            `json:"dropped"`
	}{config, records, dropped}
	r.Raw, err = marshalOwnerRaw(raw)
	if err != nil {
		return CaptureProviderResult{}, fmt.Errorf("encode SASS provider: %w", err)
	}
	r.Status = "partial"
	r.Reason = "SASS collection incomplete"
	if collectionErr != nil {
		r.Reason = "SASS collection: " + collectionErr.Error()
		return r, nil
	}
	r.Decoded = true
	r.Counts.Retained = count(uint64(len(records)))
	r.Counts.NonUser = count(0)
	var unnamed uint64
	matched := true
	instancesKnown := true
	for _, metric := range config.Metrics {
		if metric.Granularity != cuptisass.GranularityGPU {
			instancesKnown = false
		}
	}
	for _, record := range records {
		if record.FunctionName == "" {
			unnamed++
		}
		if record.FunctionName != r.Identity.Entry || len(record.Values) == 0 {
			matched = false
		}
		present := make(map[uint64]bool, len(record.Values))
		for _, value := range record.Values {
			if present[value.MetricID] && instancesKnown {
				matched = false
			}
			present[value.MetricID] = true
			if !allowed[value.MetricID] {
				matched = false
			}
		}
		for metric := range allowed {
			if !present[metric] {
				matched = false
			}
		}
	}
	r.Counts.Unnamed = count(unnamed)
	if dropped == nil {
		r.Reason = "SASS dropped-record count unavailable"
		return r, nil
	}
	r.Counts.Dropped = count(*dropped)
	if *dropped > ^uint64(0)-uint64(len(records)) {
		return CaptureProviderResult{}, fmt.Errorf("sass provider: record count overflow")
	}
	r.Counts.Offered = count(uint64(len(records)) + *dropped)
	if !matched {
		r.Reason = "SASS function or metric identity unresolved"
		return r, nil
	}
	if !instancesKnown {
		r.Reason = "SASS instance cardinality unavailable"
		return r, nil
	}
	if len(records) > 0 && *dropped == 0 && r.CleanupComplete {
		r = completeOwnerResult(r)
	}
	return r, nil
}

func rangeInventoryComplete(c *cuptirange.Capture) bool {
	if uint(len(c.RangeInfo)) != c.Ranges {
		return false
	}
	indexes := make(map[uint]bool, len(c.RangeInfo))
	ids := make(map[string]bool, len(c.RangeInfo))
	for _, r := range c.RangeInfo {
		if r.Index >= c.Ranges || indexes[r.Index] || strings.TrimSpace(r.ID) == "" || ids[r.ID] {
			return false
		}
		indexes[r.Index] = true
		ids[r.ID] = true
	}
	return true
}

// marshalOwnerRaw conservatively bounds all input vectors and strings before
// allocating their JSON representation. This is a transport cap, not a native
// collection budget. The caller must also enforce its requested manifest budget.
func marshalOwnerRaw(raw any) ([]byte, error) {
	const limit = 64 << 20
	remaining := limit
	take := func(n int) bool {
		if n > remaining {
			return false
		}
		remaining -= n
		return true
	}
	var visit func(reflect.Value) bool
	visit = func(v reflect.Value) bool {
		switch v.Kind() {
		case reflect.String:
			if v.Len() > remaining/6 {
				return false
			}
			return take(6*v.Len() + 2)
		case reflect.Pointer:
			if v.IsNil() {
				return take(4)
			}
			return visit(v.Elem())
		case reflect.Struct:
			if v.NumField() > remaining/64 || !take(v.NumField()*64) {
				return false
			}
			for i := 0; i < v.NumField(); i++ {
				if !visit(v.Field(i)) {
					return false
				}
			}
		case reflect.Slice:
			if v.Type().Elem().Kind() == reflect.Uint8 {
				if v.Len() > remaining/4*3 {
					return false
				}
				return take((v.Len()+2)/3*4 + 2)
			}
			if v.Len() > remaining/24 || !take(v.Len()*24) {
				return false
			}
			for i := 0; i < v.Len(); i++ {
				if !visit(v.Index(i)) {
					return false
				}
			}
		default:
			return take(24)
		}
		return true
	}
	if !visit(reflect.ValueOf(raw)) {
		return nil, fmt.Errorf("owner raw: preflight byte budget exceeded")
	}
	data, err := json.Marshal(raw)
	if err != nil {
		return nil, err
	}
	if len(data) > limit {
		return nil, fmt.Errorf("owner raw: byte budget exceeded")
	}
	return data, nil
}
