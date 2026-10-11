package nvtelemetry

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"math"
	"math/big"
	"sort"
	"strconv"
	"strings"
)

// WorkloadOTLPOptions binds an aggregate interval to a caller-proven Unix clock
// mapping. Equal native and Unix durations are required. MappingProvenance names
// an external calibration receipt; uncertainty and exact joins are not inferred.
type WorkloadOTLPOptions struct {
	StartNanos, EndNanos         int64
	ClockDomain, ClockEpochID    string
	HasUnixMapping               bool
	StartUnixNanos, EndUnixNanos int64
	MappingProvenance            string
	Limits                       WorkloadLimits
}

// WriteWorkloadOTLP writes bounded OTLP JSON delta sums and histograms from
// pinned serving exports. It recomputes typed windows, refusing visible resets.
// Counts/times are exact decimal strings; finite bounds/sums use doubles.
// Raw labels are replaced by a series hash. No NVML histogram, training progress,
// exact join or successful-token policy is manufactured.
func WriteWorkloadOTLP(w io.Writer, before, after WorkloadExport, o WorkloadOTLPOptions) error {
	l, err := workloadLimits(o.Limits)
	if err != nil {
		return err
	}
	if w == nil || !o.HasUnixMapping || o.StartUnixNanos <= 0 || o.EndUnixNanos <= o.StartUnixNanos || o.StartNanos < 0 || o.EndNanos <= o.StartNanos || o.EndUnixNanos-o.StartUnixNanos != o.EndNanos-o.StartNanos || o.ClockDomain == "" || o.ClockEpochID == "" || o.MappingProvenance == "" {
		return errors.New("invalid workload OTLP interval or Unix mapping")
	}
	for _, s := range []string{o.ClockDomain, o.ClockEpochID, o.MappingProvenance} {
		if len(s) > l.MaxStringBytes {
			return errors.New("workload OTLP string limit exceeded")
		}
	}
	if len(before.Metrics) > l.MaxMetrics || len(after.Metrics) > l.MaxMetrics {
		return errors.New("workload OTLP metric limit exceeded")
	}
	if _, _, err := readWorkload(strings.NewReader(""), before.Source, l); err != nil {
		return err
	}
	mappings := map[string]workloadMapping{}
	switch {
	case before.Source.Provider == "triton" && before.Source.Version == "25.06":
		mappings = tritonWorkloadMappings
	case before.Source.Provider == "vllm" && before.Source.Version == "0.8.5":
		mappings = vllmWorkloadMappings
	default:
		return errors.New("unsupported workload OTLP provider or version")
	}
	counters, hists := false, false
	for _, list := range [][]WorkloadMetric{before.Metrics, after.Metrics} {
		for _, p := range list {
			if p.Kind != "counter" && !strings.HasPrefix(p.Kind, "histogram_") {
				continue
			}
			if strings.HasPrefix(p.Kind, "histogram_") && p.Kind != "histogram_bucket" && p.Kind != "histogram_count" && p.Kind != "histogram_sum" {
				return errors.New("unsupported workload OTLP histogram component")
			}
			base, kind := p.Name, p.Kind
			if strings.HasPrefix(kind, "histogram_") {
				base = strings.TrimSuffix(base, "_"+strings.TrimPrefix(kind, "histogram_"))
				kind = "histogram"
				hists = true
			} else {
				counters = true
			}
			m, ok := mappings[base]
			expectedUnit := m.unit
			if p.Kind == "histogram_bucket" || p.Kind == "histogram_count" {
				expectedUnit = "observations"
			}
			if !ok || m.kind != kind || expectedUnit != p.Unit || m.semantic != p.Semantic {
				return errors.New("workload OTLP pinned schema mismatch")
			}
			if len(p.Labels) > 32 {
				return errors.New("workload OTLP label limit exceeded")
			}
			for k, v := range p.Labels {
				allowed := k == "le" && p.Kind == "histogram_bucket" || before.Source.Provider == "triton" && (k == "model" || k == "version" || k == "reason") || before.Source.Provider == "vllm" && (k == "model_name" || k == "finished_reason")
				if !allowed {
					return errors.New("unsupported workload OTLP label")
				}
				if len(k) > l.MaxStringBytes || len(v) > l.MaxStringBytes {
					return errors.New("workload OTLP label limit exceeded")
				}
			}
		}
	}
	var metrics []any
	encodedSize := int64(256)
	appendMetric := func(metric any) error {
		data, err := json.Marshal(metric)
		if err != nil {
			return err
		}
		encodedSize += int64(len(data)) + 1
		if encodedSize > l.MaxBytes {
			return errors.New("workload OTLP byte limit exceeded")
		}
		metrics = append(metrics, metric)
		return nil
	}
	if counters {
		deltas, e := CounterWorkloadWindow(before, after, o.StartNanos, o.EndNanos, o.ClockDomain, o.ClockEpochID)
		if e != nil {
			return e
		}
		for _, p := range deltas {
			n, e := workloadNumber(p.Delta)
			if e != nil || n.Sign() < 0 {
				return errors.New("invalid workload OTLP counter")
			}
			if p.Unit != "s" && p.Unit != "ms" && p.Unit != "us" && !n.IsInt() {
				return errors.New("fractional workload OTLP count")
			}
			dp := workloadOTLPPoint(p, before.Source, o)
			if n.IsInt() {
				if !n.Num().IsInt64() {
					return errors.New("workload OTLP signed counter overflow")
				}
				dp["asInt"] = n.Num().String()
			} else {
				v, e := workloadDouble(n)
				if e != nil {
					return e
				}
				dp["asDouble"] = v
			}
			unit, e := workloadOTLPUnit(p.Unit)
			if e != nil {
				return e
			}
			if err := appendMetric(map[string]any{"name": "nvtelemetry.workload." + before.Source.Provider + "." + p.Name, "unit": unit, "sum": map[string]any{"dataPoints": []any{dp}, "aggregationTemporality": 1, "isMonotonic": true}}); err != nil {
				return err
			}
		}
	}
	if hists {
		deltas, e := HistogramWorkloadWindow(before, after, o.StartNanos, o.EndNanos, o.ClockDomain, o.ClockEpochID)
		if e != nil {
			return e
		}
		groups := map[string][]WorkloadDelta{}
		var keys []string
		for _, p := range deltas {
			q := workloadHistogramBase(p)
			key := workloadSeriesKey(WorkloadMetric{Name: q.Name, Labels: q.Labels})
			if _, ok := groups[key]; !ok {
				keys = append(keys, key)
			}
			groups[key] = append(groups[key], p)
		}
		sort.Strings(keys)
		for _, key := range keys {
			parts := groups[key]
			p := workloadHistogramBase(parts[0])
			p.Unit = mappings[p.Name].unit
			dp := workloadOTLPPoint(p, before.Source, o)
			var buckets []WorkloadDelta
			for _, c := range parts {
				n, e := workloadNumber(c.Delta)
				if e != nil || n.Sign() < 0 {
					return errors.New("invalid workload OTLP histogram")
				}
				switch c.Kind {
				case "histogram_bucket":
					buckets = append(buckets, c)
				case "histogram_count":
					v, e := workloadCount(n)
					if e != nil {
						return e
					}
					dp["count"] = v
				case "histogram_sum":
					v, e := workloadDouble(n)
					if e != nil {
						return e
					}
					dp["sum"] = v
				}
			}
			sort.Slice(buckets, func(i, j int) bool {
				a, b := buckets[i].Labels["le"], buckets[j].Labels["le"]
				if a == "+Inf" {
					return false
				}
				if b == "+Inf" {
					return true
				}
				x, _ := workloadNumber(a)
				y, _ := workloadNumber(b)
				return x.Cmp(y) < 0
			})
			bounds := []float64{}
			counts := []string{}
			previous := new(big.Rat)
			lastBound := math.Inf(-1)
			for _, b := range buckets {
				n, _ := workloadNumber(b.Delta)
				v, e := workloadCount(new(big.Rat).Sub(n, previous))
				if e != nil {
					return e
				}
				counts = append(counts, v)
				previous = n
				if b.Labels["le"] != "+Inf" {
					r, _ := workloadNumber(b.Labels["le"])
					v, e := workloadDouble(r)
					if e != nil || v <= lastBound {
						return errors.New("workload OTLP histogram bounds lose precision")
					}
					bounds = append(bounds, v)
					lastBound = v
				}
			}
			dp["explicitBounds"] = bounds
			dp["bucketCounts"] = counts
			unit, e := workloadOTLPUnit(p.Unit)
			if e != nil {
				return e
			}
			if err := appendMetric(map[string]any{"name": "nvtelemetry.workload." + before.Source.Provider + "." + p.Name, "unit": unit, "histogram": map[string]any{"dataPoints": []any{dp}, "aggregationTemporality": 1}}); err != nil {
				return err
			}
		}
	}
	if len(metrics) == 0 {
		return errors.New("no supported workload OTLP windows")
	}
	raw, err := json.Marshal(map[string]any{"resourceMetrics": []any{map[string]any{"scopeMetrics": []any{map[string]any{"scope": map[string]string{"name": "github.com/tmc/lib/nvidia/x/nvtelemetry"}, "metrics": metrics}}}}})
	if err != nil {
		return err
	}
	if int64(len(raw)) > l.MaxBytes {
		return errors.New("workload OTLP byte limit exceeded")
	}
	n, err := w.Write(raw)
	if err == nil && n != len(raw) {
		err = io.ErrShortWrite
	}
	return err
}
func workloadHistogramBase(p WorkloadDelta) WorkloadDelta {
	p.Name = strings.TrimSuffix(p.Name, "_"+strings.TrimPrefix(p.Kind, "histogram_"))
	labels := map[string]string{}
	for k, v := range p.Labels {
		if k != "le" {
			labels[k] = v
		}
	}
	p.Labels = labels
	return p
}
func workloadOTLPPoint(p WorkloadDelta, s WorkloadSource, o WorkloadOTLPOptions) map[string]any {
	key := workloadSeriesKey(WorkloadMetric{Name: p.Name, Labels: p.Labels})
	hash := sha256.Sum256([]byte(key))
	pairs := [][2]string{{"provider", s.Provider}, {"version", s.Version}, {"run_cohort", s.RunID}, {"series_sha256", hex.EncodeToString(hash[:])}, {"evidence", "derived_aggregate_delta"}, {"clock_domain", o.ClockDomain}, {"clock_epoch", o.ClockEpochID}, {"clock_mapping", o.MappingProvenance}}
	if s.Phase != "" {
		pairs = append(pairs, [2]string{"phase", s.Phase})
	}
	if s.Rank != nil {
		pairs = append(pairs, [2]string{"rank", strconv.Itoa(*s.Rank)})
	}
	attrs := make([]any, 0, len(pairs))
	for _, p := range pairs {
		attrs = append(attrs, map[string]any{"key": "nvtelemetry.workload." + p[0], "value": map[string]string{"stringValue": p[1]}})
	}
	return map[string]any{"startTimeUnixNano": strconv.FormatInt(o.StartUnixNanos, 10), "timeUnixNano": strconv.FormatInt(o.EndUnixNanos, 10), "attributes": attrs}
}
func workloadCount(n *big.Rat) (string, error) {
	if !n.IsInt() || n.Sign() < 0 || !n.Num().IsUint64() {
		return "", errors.New("workload OTLP unsigned count overflow")
	}
	return n.Num().String(), nil
}
func workloadDouble(n *big.Rat) (float64, error) {
	v, _ := n.Float64()
	if math.IsInf(v, 0) || math.IsNaN(v) || v == 0 && n.Sign() != 0 {
		return 0, errors.New("workload OTLP non-finite or underflow value")
	}
	return v, nil
}
func workloadOTLPUnit(unit string) (string, error) {
	switch unit {
	case "s", "ms", "us":
		return unit, nil
	case "tokens":
		return "{token}", nil
	case "requests":
		return "{request}", nil
	case "inferences":
		return "{inference}", nil
	case "executions":
		return "{execution}", nil
	case "lookups":
		return "{lookup}", nil
	}
	return "", errors.New("unsupported workload OTLP unit")
}
