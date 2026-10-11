package nvtelemetry

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"math"
	"math/big"
	"sort"
	"strconv"
	"strings"
)

// WorkloadSource identifies an offline producer export. RunID, Phase and Rank
// are caller-supplied export context, never inferred from hardware activity.
type WorkloadSource struct {
	Provider     string
	Version      string
	RunID        string
	Phase        string
	Rank         *int
	ClockDomain  string
	ClockEpochID string
	// Units declares units for user-defined training metrics. Missing units
	// remain unknown; a metric named "loss" does not prove convergence.
	Units map[string]string
}

// WorkloadMetric preserves a producer metric's identity and numeric spelling.
// TimeNanos is meaningful only when HasTime is true. Clock metadata identifies
// an aggregate observation, not a request-to-kernel clock calibration.
type WorkloadMetric struct {
	Source      WorkloadSource
	Name        string
	Semantic    string
	Unit        string
	Kind        string // gauge, counter, histogram_bucket, histogram_count, histogram_sum
	Temporality string // instantaneous or cumulative
	Labels      map[string]string
	RawValue    string
	RawTime     string
	RawTimeUnit string
	TimeNanos   int64
	HasTime     bool
	TimeSource  string // producer or collector_observation; never sensor
	Step        int64
	HasStep     bool
	StepKind    string // logged_metric_step, history_step or reported_iteration
}

// WorkloadExport contains bounded original bytes and normalized observations.
// RawSHA256 is computed by the reader and can be checked independently.
type WorkloadExport struct {
	Source        WorkloadSource
	Raw           []byte
	RawSHA256     string
	Metrics       []WorkloadMetric
	NextPageToken string
	UnmappedNames []string
}

// WorkloadLimits bounds offline export readers. Zero values select defaults:
// 4 MiB of input, 10000 observations and 4096 bytes per string.
type WorkloadLimits struct {
	MaxBytes       int64
	MaxMetrics     int
	MaxStringBytes int
}

func workloadLimits(l WorkloadLimits) (WorkloadLimits, error) {
	if l.MaxBytes < 0 || l.MaxMetrics < 0 || l.MaxStringBytes < 0 {
		return l, errors.New("negative workload limit")
	}
	if l.MaxBytes == 0 {
		l.MaxBytes = 4 << 20
	}
	if l.MaxMetrics == 0 {
		l.MaxMetrics = 10000
	}
	if l.MaxStringBytes == 0 {
		l.MaxStringBytes = 4096
	}
	if l.MaxBytes == math.MaxInt64 {
		return l, errors.New("workload byte limit too large")
	}
	if l.MaxBytes > 64<<20 || l.MaxMetrics > 65536 || l.MaxStringBytes > 4096 {
		return l, errors.New("workload limit exceeds hard cap")
	}
	return l, nil
}

func readWorkload(r io.Reader, s WorkloadSource, l WorkloadLimits) (WorkloadExport, WorkloadLimits, error) {
	l, err := workloadLimits(l)
	if err != nil {
		return WorkloadExport{}, l, err
	}
	if r == nil || strings.TrimSpace(s.Provider) == "" || strings.TrimSpace(s.Version) == "" || strings.TrimSpace(s.RunID) == "" {
		return WorkloadExport{}, l, errors.New("missing workload reader or source identity")
	}
	if s.Rank != nil && *s.Rank < 0 {
		return WorkloadExport{}, l, errors.New("negative workload rank")
	}
	for _, v := range []string{s.Provider, s.Version, s.RunID, s.Phase, s.ClockDomain, s.ClockEpochID} {
		if len(v) > l.MaxStringBytes {
			return WorkloadExport{}, l, errors.New("workload source string limit exceeded")
		}
	}
	// Copy mutable caller metadata so later configuration changes cannot rewrite
	// the provenance of observations already read.
	if s.Rank != nil {
		rank := *s.Rank
		s.Rank = &rank
	}
	units := make(map[string]string, len(s.Units))
	if len(s.Units) > l.MaxMetrics {
		return WorkloadExport{}, l, errors.New("workload unit count limit exceeded")
	}
	for k, v := range s.Units {
		if len(k) > l.MaxStringBytes || len(v) > l.MaxStringBytes {
			return WorkloadExport{}, l, errors.New("workload unit string limit exceeded")
		}
		units[k] = v
	}
	s.Units = units
	raw, err := io.ReadAll(io.LimitReader(r, l.MaxBytes+1))
	if err != nil {
		return WorkloadExport{}, l, fmt.Errorf("read workload export: %w", err)
	}
	if int64(len(raw)) > l.MaxBytes {
		return WorkloadExport{}, l, errors.New("workload byte limit exceeded")
	}
	h := sha256.Sum256(raw)
	return WorkloadExport{Source: s, Raw: raw, RawSHA256: hex.EncodeToString(h[:])}, l, nil
}

func workloadNumber(raw string) (*big.Rat, error) {
	// A short decimal exponent can expand into arbitrarily large big.Int
	// storage. Bound expansion before handing untrusted producer text to Rat.
	if len(raw) > 4096 {
		return nil, errors.New("workload numeric precision limit exceeded")
	}
	if i := strings.IndexAny(raw, "eE"); i >= 0 {
		exp, err := strconv.ParseInt(raw[i+1:], 10, 64)
		if err != nil || exp < -1000 || exp > 1000 {
			return nil, errors.New("workload numeric exponent limit exceeded")
		}
	}
	v, ok := new(big.Rat).SetString(raw)
	if !ok {
		return nil, fmt.Errorf("invalid workload number %q", raw)
	}
	return v, nil
}

func workloadTime(raw string, scale int64) (int64, error) {
	v, err := workloadNumber(raw)
	if err != nil {
		return 0, err
	}
	v.Mul(v, new(big.Rat).SetInt64(scale))
	if !v.IsInt() || !v.Num().IsInt64() {
		return 0, errors.New("workload timestamp not representable in nanoseconds")
	}
	return v.Num().Int64(), nil
}

// WorkloadDelta is an exact decimal counter difference between two aggregate
// observations. It never supplies request identity or successful-token policy.
type WorkloadDelta struct {
	Name       string
	Unit       string
	Semantic   string
	Kind       string
	Labels     map[string]string
	Delta      string
	StartNanos int64
	EndNanos   int64
}

// CounterWorkloadWindow computes differences for matching cumulative counters.
// Both exports must bound the requested window exactly in the declared clock
// domain and epoch. Visible decreases, missing series and rolling versions are
// errors. A counter reset that recovers between observations is not observable.
// Gauges and histogram components remain in the exports, without interpolation.
func CounterWorkloadWindow(before, after WorkloadExport, start, end int64, domain, epoch string) ([]WorkloadDelta, error) {
	return aggregateWorkloadWindow(before, after, start, end, domain, epoch, false)
}

// HistogramWorkloadWindow differences compatible cumulative histogram buckets,
// counts and sums. Bounds and all label identities must match between snapshots.
// The result describes aggregate observations, never individual latencies.
func HistogramWorkloadWindow(before, after WorkloadExport, start, end int64, domain, epoch string) ([]WorkloadDelta, error) {
	if err := validateWorkloadHistograms(before.Metrics); err != nil {
		return nil, err
	}
	if err := validateWorkloadHistograms(after.Metrics); err != nil {
		return nil, err
	}
	return aggregateWorkloadWindow(before, after, start, end, domain, epoch, true)
}

func aggregateWorkloadWindow(before, after WorkloadExport, start, end int64, domain, epoch string, histograms bool) ([]WorkloadDelta, error) {
	if end <= start || strings.TrimSpace(domain) == "" || strings.TrimSpace(epoch) == "" {
		return nil, errors.New("invalid workload window or unknown clock")
	}
	if before.Source.Provider != after.Source.Provider || before.Source.Version != after.Source.Version || before.Source.RunID != after.Source.RunID || before.Source.Phase != after.Source.Phase || !sameWorkloadRank(before.Source.Rank, after.Source.Rank) {
		return nil, errors.New("workload source changed across window")
	}
	if before.NextPageToken != "" || after.NextPageToken != "" {
		return nil, errors.New("paginated workload export is incomplete")
	}
	index := func(e WorkloadExport, t int64) (map[string]WorkloadMetric, error) {
		m := make(map[string]WorkloadMetric)
		for _, p := range e.Metrics {
			if histograms && !strings.HasPrefix(p.Kind, "histogram_") || !histograms && p.Kind != "counter" {
				continue
			}
			if !sameWorkloadContext(e.Source, p.Source) {
				return nil, errors.New("workload point source contradicts export")
			}
			if p.Temporality != "cumulative" || !p.HasTime || p.TimeNanos != t || p.Source.ClockDomain != domain || p.Source.ClockEpochID != epoch {
				return nil, errors.New("workload counter clock or temporality mismatch")
			}
			key := workloadSeriesKey(p)
			if _, ok := m[key]; ok {
				return nil, errors.New("duplicate workload counter series")
			}
			m[key] = p
		}
		return m, nil
	}
	a, err := index(before, start)
	if err != nil {
		return nil, err
	}
	b, err := index(after, end)
	if err != nil {
		return nil, err
	}
	if len(a) == 0 || len(a) != len(b) {
		return nil, errors.New("missing workload counter series")
	}
	var keys []string
	for k := range a {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var out []WorkloadDelta
	for _, k := range keys {
		p := a[k]
		q, ok := b[k]
		if !ok || p.Unit != q.Unit || p.Semantic != q.Semantic || p.Kind != q.Kind {
			return nil, errors.New("workload counter schema changed")
		}
		x, err := workloadNumber(p.RawValue)
		if err != nil {
			return nil, err
		}
		y, err := workloadNumber(q.RawValue)
		if err != nil {
			return nil, err
		}
		if x.Sign() < 0 || y.Cmp(x) < 0 {
			return nil, errors.New("workload counter reset")
		}
		d := new(big.Rat).Sub(y, x)
		out = append(out, WorkloadDelta{Name: p.Name, Unit: p.Unit, Semantic: p.Semantic, Kind: p.Kind, Labels: p.Labels, Delta: d.RatString(), StartNanos: start, EndNanos: end})
	}
	if histograms {
		// Independently valid endpoints do not imply a valid interval CDF.
		// Partial resets can make a finite bucket's delta exceed +Inf/count.
		var deltas []WorkloadMetric
		for _, p := range out {
			deltas = append(deltas, WorkloadMetric{Name: p.Name, Kind: p.Kind, Labels: p.Labels, RawValue: p.Delta})
		}
		if err := validateWorkloadHistograms(deltas); err != nil {
			return nil, err
		}
	}
	return out, nil
}

func sameWorkloadRank(a, b *int) bool {
	return a == nil && b == nil || a != nil && b != nil && *a == *b
}

func sameWorkloadContext(a, b WorkloadSource) bool {
	return a.Provider == b.Provider && a.Version == b.Version && a.RunID == b.RunID && a.Phase == b.Phase && sameWorkloadRank(a.Rank, b.Rank) && a.ClockDomain == b.ClockDomain && a.ClockEpochID == b.ClockEpochID
}
func workloadSeriesKey(p WorkloadMetric) string {
	var keys []string
	for k := range p.Labels {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var b strings.Builder
	fmt.Fprintf(&b, "%q", p.Name)
	for _, k := range keys {
		fmt.Fprintf(&b, "%q=%q;", k, p.Labels[k])
	}
	return b.String()
}

// ExactWorkloadJoin refuses to turn aggregated exports into individual request
// or kernel records. Explicit benchmark records use CorrelateBenchmark instead.
func ExactWorkloadJoin(WorkloadExport) error {
	return errors.New("aggregate workload exports cannot establish exact request or kernel joins")
}

// TrainingStepReport exposes observed steps and missing expected ranks. It does
// not infer convergence, phase transitions, or checkpoint correctness.
type TrainingStepReport struct {
	StepKind           string
	Steps              []int64
	GapAfter           []int64
	MissingRanks       []int
	UnknownRank        bool
	GapAfterByRank     map[int][]int64
	MissingRanksByStep map[int64][]int
}

// InspectTrainingSteps checks one metric and run across exported rank histories.
// ExpectedRanks is an explicit caller contract, not a count inferred from GPUs.
func InspectTrainingSteps(exports []WorkloadExport, metric string, expectedRanks []int) (TrainingStepReport, error) {
	var out TrainingStepReport
	if len(exports) == 0 || metric == "" {
		return out, errors.New("missing training export or metric")
	}
	steps := map[int64]bool{}
	ranks := map[int]bool{}
	rankSteps := map[int]map[int64]bool{}
	run := exports[0].Source.RunID
	series := ""
	for _, e := range exports {
		if e.NextPageToken != "" {
			return out, errors.New("paginated training history is incomplete")
		}
		if e.Source.RunID != run || e.Source.Provider != exports[0].Source.Provider || e.Source.Version != exports[0].Source.Version || e.Source.Phase != exports[0].Source.Phase {
			return out, errors.New("concurrent training runs require separate reports")
		}
		found := false
		for _, p := range e.Metrics {
			if p.Name == metric && p.HasStep {
				if !sameWorkloadContext(e.Source, p.Source) {
					return out, errors.New("training point source contradicts export")
				}
				key := workloadSeriesKey(p)
				if series != "" && series != key {
					return out, errors.New("ambiguous training metric label series")
				}
				series = key
				if p.Step < 0 {
					return out, errors.New("negative training step")
				}
				kind := p.StepKind
				if kind == "" {
					kind = "unknown"
				}
				if out.StepKind != "" && out.StepKind != kind {
					return out, errors.New("incompatible training step semantics")
				}
				out.StepKind = kind
				steps[p.Step] = true
				if e.Source.Rank != nil {
					rank := *e.Source.Rank
					if rankSteps[rank] == nil {
						rankSteps[rank] = map[int64]bool{}
					}
					rankSteps[rank][p.Step] = true
				}
				found = true
			}
		}
		if found {
			if e.Source.Rank == nil {
				out.UnknownRank = true
			} else {
				ranks[*e.Source.Rank] = true
			}
		}
	}
	if len(steps) == 0 {
		return out, errors.New("no explicit training steps observed")
	}
	out.GapAfterByRank = map[int][]int64{}
	for rank, seen := range rankSteps {
		var observed []int64
		for step := range seen {
			observed = append(observed, step)
		}
		sort.Slice(observed, func(i, j int) bool { return observed[i] < observed[j] })
		for i := 1; i < len(observed); i++ {
			if observed[i]-observed[i-1] > 1 {
				out.GapAfterByRank[rank] = append(out.GapAfterByRank[rank], observed[i-1])
			}
		}
	}
	for step := range steps {
		out.Steps = append(out.Steps, step)
	}
	sort.Slice(out.Steps, func(i, j int) bool { return out.Steps[i] < out.Steps[j] })
	for i := 1; i < len(out.Steps); i++ {
		if out.Steps[i]-out.Steps[i-1] > 1 {
			out.GapAfter = append(out.GapAfter, out.Steps[i-1])
		}
	}
	seen := map[int]bool{}
	out.MissingRanksByStep = map[int64][]int{}
	for _, rank := range expectedRanks {
		if rank < 0 || seen[rank] {
			return out, errors.New("invalid expected training ranks")
		}
		seen[rank] = true
		if !ranks[rank] {
			out.MissingRanks = append(out.MissingRanks, rank)
		}
		for _, step := range out.Steps {
			if !rankSteps[rank][step] {
				out.MissingRanksByStep[step] = append(out.MissingRanksByStep[step], rank)
			}
		}
	}
	for _, missing := range out.MissingRanksByStep {
		sort.Ints(missing)
	}
	sort.Ints(out.MissingRanks)
	return out, nil
}
