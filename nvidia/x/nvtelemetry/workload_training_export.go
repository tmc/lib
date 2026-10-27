package nvtelemetry

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"math/big"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"
)

// TrainingWorkloadOTLPOptions declares the original producer timestamps as Unix
// seconds. UnixClockProvenance is required caller evidence, not authenticated
// synchronization or a clock transform. The source ClockDomain must be unix_utc.
// UnitEvidence names caller evidence for each explicit Source.Units entry used;
// absent units remain unknown. Limits bound input, points, names and output.
// A finite export may have one series per retained history step and raw digest;
// these bounds are not a continuous collector's cardinality policy.
type TrainingWorkloadOTLPOptions struct {
	UnixClockProvenance string
	UnitEvidence        map[string]string
	Limits              WorkloadLimits
}

type trainingOTLPGauge struct {
	Name  string `json:"name"`
	Unit  string `json:"unit"`
	Gauge struct {
		DataPoints []trainingOTLPPoint `json:"dataPoints"`
	} `json:"gauge"`
}
type trainingOTLPPoint struct {
	TimeUnixNano string  `json:"timeUnixNano"`
	AsDouble     float64 `json:"asDouble"`
	Attributes   []any   `json:"attributes"`
}

// WriteTrainingWorkloadOTLP writes bounded instantaneous scalar gauges from a
// W&B 0.19.11 history-compatible export, including an explicitly decoded SDK
// datastore derivative. It replays Raw through ReadWandBHistory and refuses a
// changed digest or metric projection, missing time, ambiguous samples, and
// integers outside [-2^53, 2^53] before converting to float64. Other finite
// decimals use float64 rounding; original bytes remain identified by their hash.
// Only unknown units or explicitly evidenced 1, s, ms, us, bytes/By are supported.
// Source provenance remains caller supplied; no native origin, training success,
// convergence, counters, epochs, clock bounds or backend acceptance is inferred.
// Every emitted caller/source string must be valid UTF-8. Validation completes
// before the first Write; an output error may leave bytes.
func WriteTrainingWorkloadOTLP(w io.Writer, e WorkloadExport, o TrainingWorkloadOTLPOptions) error {
	l, err := workloadLimits(o.Limits)
	if err != nil {
		return err
	}
	if w == nil || e.Source.ClockDomain != "unix_utc" || strings.TrimSpace(o.UnixClockProvenance) == "" || len(o.UnixClockProvenance) > l.MaxStringBytes || !utf8.ValidString(o.UnixClockProvenance) {
		return errors.New("training OTLP requires declared Unix clock provenance")
	}
	for _, value := range []string{e.Source.Provider, e.Source.Version, e.Source.RunID, e.Source.Phase, e.Source.ClockDomain, e.Source.ClockEpochID} {
		if !utf8.ValidString(value) {
			return errors.New("training OTLP source identity is not UTF-8")
		}
	}
	if len(o.UnitEvidence) > l.MaxMetrics {
		return errors.New("training OTLP unit evidence limit exceeded")
	}
	for name, evidence := range o.UnitEvidence {
		if name == "" || len(name) > l.MaxStringBytes || strings.TrimSpace(evidence) == "" || len(evidence) > l.MaxStringBytes || !utf8.ValidString(name) || !utf8.ValidString(evidence) {
			return errors.New("invalid training OTLP unit evidence")
		}
	}
	if !utf8.Valid(e.Raw) {
		return errors.New("training OTLP source is not UTF-8")
	}
	replay, err := ReadWandBHistory(bytes.NewReader(e.Raw), e.Source, l)
	if err != nil {
		return err
	}
	if e.RawSHA256 != replay.RawSHA256 || !reflect.DeepEqual(e.Metrics, replay.Metrics) {
		return errors.New("training OTLP source digest or projection mismatch")
	}
	if len(replay.Metrics) == 0 {
		return errors.New("training OTLP has no scalar points")
	}
	pair := func(k, v string) any {
		return map[string]any{"key": "nvtelemetry.workload." + k, "value": map[string]string{"stringValue": v}}
	}
	attrs := []any{pair("provider", e.Source.Provider), pair("version", e.Source.Version), pair("run_cohort", e.Source.RunID), pair("source_report_sha256", replay.RawSHA256), pair("clock_domain", e.Source.ClockDomain), pair("clock_provenance", o.UnixClockProvenance), pair("evidence", "imported_user_defined_metric"), pair("clock_basis", "caller_declared_unix")}
	if e.Source.ClockEpochID != "" {
		attrs = append(attrs, pair("clock_epoch", e.Source.ClockEpochID))
	}
	if e.Source.Phase != "" {
		attrs = append(attrs, pair("phase", e.Source.Phase))
	}
	if e.Source.Rank != nil {
		attrs = append(attrs, pair("rank", strconv.Itoa(*e.Source.Rank)))
	}
	request := func(metrics []*trainingOTLPGauge) map[string]any {
		return map[string]any{"resourceMetrics": []any{map[string]any{"resource": map[string]any{"attributes": attrs}, "scopeMetrics": []any{map[string]any{"scope": map[string]string{"name": "github.com/tmc/lib/nvidia/x/nvtelemetry"}, "metrics": metrics}}}}}
	}
	metrics := []*trainingOTLPGauge{}
	head, err := json.Marshal(request(metrics))
	if err != nil {
		return err
	}
	size := int64(len(head))
	groups := map[string]*trainingOTLPGauge{}
	type timeKey struct {
		name string
		time int64
	}
	type stepKey struct {
		name string
		step int64
	}
	times := map[timeKey]bool{}
	steps := map[stepKey]bool{}
	maxInt := big.NewInt(1 << 53)
	for _, p := range replay.Metrics {
		if !p.HasTime || p.TimeNanos <= 0 || p.RawTimeUnit != "seconds" || p.TimeSource != "producer" {
			return errors.New("training OTLP requires present producer Unix timestamp")
		}
		tk := timeKey{p.Name, p.TimeNanos}
		if times[tk] {
			return errors.New("training OTLP duplicate sample time")
		}
		times[tk] = true
		if p.HasStep {
			sk := stepKey{p.Name, p.Step}
			if steps[sk] {
				return errors.New("training OTLP duplicate history step")
			}
			steps[sk] = true
		}
		n, err := workloadNumber(p.RawValue)
		if err != nil {
			return err
		}
		if n.IsInt() && new(big.Int).Abs(n.Num()).Cmp(maxInt) > 0 {
			return errors.New("training OTLP integer would exceed exact float64 profile")
		}
		value, err := workloadDouble(n)
		if err != nil {
			return err
		}
		unit := ""
		evidence := o.UnitEvidence[p.Name]
		if p.Unit != "unknown" && p.Unit != "" {
			if evidence == "" {
				return errors.New("training OTLP explicit unit lacks evidence")
			}
			switch p.Unit {
			case "1", "s", "ms", "us", "By":
				unit = p.Unit
			case "bytes":
				unit = "By"
			default:
				return errors.New("training OTLP unsupported unit")
			}
		} else if evidence != "" {
			return errors.New("training OTLP unit evidence lacks declared unit")
		}
		g := groups[p.Name]
		if g == nil {
			g = &trainingOTLPGauge{Name: "nvtelemetry.workload.wandb." + p.Name, Unit: unit}
			g.Gauge.DataPoints = []trainingOTLPPoint{}
			raw, err := json.Marshal(g)
			if err != nil {
				return err
			}
			size += int64(len(raw))
			if len(metrics) > 0 {
				size++
			}
			groups[p.Name] = g
			metrics = append(metrics, g)
		}
		point := trainingOTLPPoint{TimeUnixNano: strconv.FormatInt(p.TimeNanos, 10), AsDouble: value, Attributes: []any{pair("source_time_unit", p.RawTimeUnit), pair("step_kind", p.StepKind)}}
		if p.HasStep {
			point.Attributes = append(point.Attributes, pair("history_step", strconv.FormatInt(p.Step, 10)))
		}
		if evidence != "" {
			point.Attributes = append(point.Attributes, pair("unit_evidence", evidence))
		}
		raw, err := json.Marshal(point)
		if err != nil {
			return err
		}
		size += int64(len(raw))
		if len(g.Gauge.DataPoints) > 0 {
			size++
		}
		if size > l.MaxBytes {
			return errors.New("training OTLP byte limit exceeded")
		}
		g.Gauge.DataPoints = append(g.Gauge.DataPoints, point)
	}
	sort.Slice(metrics, func(i, j int) bool { return metrics[i].Name < metrics[j].Name })
	raw, err := json.Marshal(request(metrics))
	if err != nil {
		return err
	}
	if int64(len(raw)) > l.MaxBytes {
		return errors.New("training OTLP byte limit exceeded")
	}
	n, err := w.Write(raw)
	if err == nil && n != len(raw) {
		return io.ErrShortWrite
	}
	return err
}
