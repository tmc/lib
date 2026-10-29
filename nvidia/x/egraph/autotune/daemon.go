package autotune

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"math"
	"slices"
	"sync"
	"time"

	"github.com/tmc/lib/nvidia/x/cudahotswap"
)

// ErrSpillDisqualified reports a candidate whose compiled kernel spills
// registers to local memory. Such candidates are never selected.
var ErrSpillDisqualified = errors.New("autotune: candidate spills registers")

// ErrNumericalDrift reports a candidate whose outputs disagree with the
// serving kernel's. Such candidates are never selected.
var ErrNumericalDrift = errors.New("autotune: candidate output drifts from serving kernel")

// A Variant is a compiled schedule candidate.
type Variant struct {
	Candidate ScheduleCandidate
	Source    cudahotswap.Source // the kernel source the variant was compiled from
	PTX       string
	Cubin     []byte

	// SpillStores and SpillLoads count local-memory STL and LDL
	// instructions in the entry function's SASS, as found by the Builder.
	SpillStores, SpillLoads int
}

// ProfileObservation is one sampled run of a variant.
type ProfileObservation struct {
	MedianUS float64      // median kernel duration over the sampled launches
	Metrics  RangeMetrics // register count and local-memory spill bytes
}

// A Builder emits and compiles a variant for a schedule candidate.
type Builder interface {
	Build(ctx context.Context, c ScheduleCandidate) (Variant, error)
}

// A LiveCollector launches a variant iters times under CUPTI PC sampling
// and range profiling.
type LiveCollector interface {
	SampleKernel(ctx context.Context, v Variant, iters int) (ProfileObservation, StallHistogram, error)
}

// An Agreement compares a candidate variant's outputs with the serving
// variant's and returns an error if they differ beyond tolerance.
type Agreement interface {
	Agree(ctx context.Context, serving, candidate Variant) error
}

// A Swapper publishes a variant as the serving kernel. Publication must be
// atomic with respect to in-flight launches.
type Swapper interface {
	Swap(ctx context.Context, v Variant) error
}

// ServiceSwapper publishes variants through a [cudahotswap.Service] (or
// anything with its Push method) under the kernel name Name. The service
// runs its own compile, replay and correctness checks before publishing.
type ServiceSwapper struct {
	Service interface {
		Push(ctx context.Context, name string, source cudahotswap.Source, label string) (*cudahotswap.PushResult, error)
	}
	Name string
}

// Swap pushes v's source and reports an error unless the service
// published it.
func (s ServiceSwapper) Swap(ctx context.Context, v Variant) error {
	r, err := s.Service.Push(ctx, s.Name, v.Source, v.Candidate.Name)
	if err != nil {
		return fmt.Errorf("swap %s: %w", v.Candidate.Name, err)
	}
	if r == nil || !r.Published {
		return fmt.Errorf("swap %s: service did not publish the candidate", v.Candidate.Name)
	}
	return nil
}

// DaemonConfig configures a [Daemon].
type DaemonConfig struct {
	Twin           HardwareTwin
	ShapeKey       string        // the workload the candidates implement, e.g. a graph hash
	MaxTrials      int           // candidates sampled per Step; default 2
	Iters          int           // launches per sample; default 20
	ImproveFloor   float64       // fractional speedup required to swap; default 0.015
	SampleInterval time.Duration // time between steps in Run; default 1s
	Builder        Builder
	Agreement      Agreement // nil skips the numerical agreement check
	Swapper        Swapper   // nil records the winner without publishing it
}

// A Trial is one candidate sampled by [Daemon.Step].
type Trial struct {
	Candidate   ScheduleCandidate
	Observation ProfileObservation
	Stalls      StallHistogram
	Err         error // nonnil if the candidate was disqualified
}

// A StepResult reports one [Daemon.Step].
type StepResult struct {
	Trials     []Trial
	Winner     ScheduleCandidate // the fastest qualified measured candidate
	WinnerUS   float64
	BaselineUS float64 // measured latency of the analytical model's choice
	Speedup    float64 // BaselineUS / WinnerUS
	Serving    ScheduleCandidate
	Swapped    bool
	Weights    ScheduleWeights
}

type measurement struct {
	variant Variant
	obs     ProfileObservation
	stalls  StallHistogram
	err     error
}

// A Daemon runs the closed loop: rank candidates by stall-directed
// weights, compile and sample the best untried ones, reject spilling or
// drifting variants, refine the weights from their stall histograms, and
// swap the serving kernel when a candidate beats it by ImproveFloor.
//
// The loop logic runs on the host; compiling, sampling, agreement and
// swapping are delegated to the configured interfaces, which on a GPU are
// backed by NVRTC, CUPTI and cudahotswap.
type Daemon struct {
	cfg       DaemonConfig
	collector LiveCollector

	mu       sync.Mutex
	weights  ScheduleWeights
	model    *OnlineCostModel
	measured map[string]*measurement
	serving  string
	warm     string // cached winner to sample first
}

// NewDaemon returns a daemon sampling through collector.
func NewDaemon(cfg DaemonConfig, collector LiveCollector) (*Daemon, error) {
	if collector == nil {
		return nil, errors.New("new daemon: nil collector")
	}
	if cfg.Builder == nil {
		return nil, errors.New("new daemon: nil builder")
	}
	if cfg.MaxTrials <= 0 {
		cfg.MaxTrials = 2
	}
	if cfg.Iters <= 0 {
		cfg.Iters = 20
	}
	if cfg.ImproveFloor <= 0 {
		cfg.ImproveFloor = 0.015
	}
	if cfg.SampleInterval <= 0 {
		cfg.SampleInterval = time.Second
	}
	return &Daemon{
		cfg:       cfg,
		collector: collector,
		model:     NewOnlineCostModel(),
		measured:  make(map[string]*measurement),
	}, nil
}

// Weights returns the daemon's current schedule weights.
func (d *Daemon) Weights() ScheduleWeights {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.weights
}

// Model returns the online cost model the daemon records observations in.
func (d *Daemon) Model() *OnlineCostModel { return d.model }

// Step runs one closed-loop iteration over candidates. The first step
// samples the candidate with the lowest analytical cost, which is taken to
// be serving. Candidates that fail [ScheduleCandidate.Validate] are
// skipped. Step returns an error only if no candidate could be measured.
func (d *Daemon) Step(ctx context.Context, candidates []ScheduleCandidate) (StepResult, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	var valid []ScheduleCandidate
	for _, c := range candidates {
		if c.Validate(d.cfg.Twin) == nil {
			valid = append(valid, c)
		}
	}
	if len(valid) == 0 {
		return StepResult{}, errors.New("daemon step: no valid candidates")
	}
	baseline := slices.MinFunc(valid, func(a, b ScheduleCandidate) int {
		return cmp.Compare(a.AnalyticalCostUS, b.AnalyticalCostUS)
	})

	var res StepResult
	for range d.cfg.MaxTrials {
		c, ok := d.next(valid, baseline)
		if !ok {
			break
		}
		res.Trials = append(res.Trials, d.trial(ctx, c))
	}
	if len(res.Trials) == 0 && d.serving != "" {
		// Everything is measured: re-sample the serving kernel.
		res.Trials = append(res.Trials, d.trial(ctx, d.measured[d.serving].variant.Candidate))
	}
	if d.serving == "" {
		if m := d.measured[baseline.Name]; m != nil && m.err == nil {
			d.serving = baseline.Name
		}
	}

	winner := ""
	for _, c := range valid {
		m := d.measured[c.Name]
		if m == nil || m.err != nil {
			continue
		}
		if winner == "" || m.obs.MedianUS < d.measured[winner].obs.MedianUS {
			winner = c.Name
		}
	}
	if winner == "" {
		err := errors.Join(trialErrs(res.Trials)...)
		if err == nil {
			err = errors.New("no trials")
		}
		return res, fmt.Errorf("daemon step: no qualified candidate measured: %w", err)
	}
	if d.serving == "" {
		d.serving = winner
	}
	if winner != d.serving {
		cur, w := d.measured[d.serving], d.measured[winner]
		if w.obs.MedianUS < cur.obs.MedianUS*(1-d.cfg.ImproveFloor) {
			swapped, err := d.swap(ctx, cur, w)
			if err != nil {
				return res, err
			}
			res.Swapped = swapped
		}
	}
	// Agreement may have disqualified the winner.
	if d.measured[winner].err != nil {
		winner = d.serving
	}
	w := d.measured[winner]
	res.Winner, res.WinnerUS = w.variant.Candidate, w.obs.MedianUS
	if b := d.measured[baseline.Name]; b != nil && b.obs.MedianUS > 0 {
		res.BaselineUS = b.obs.MedianUS
		res.Speedup = b.obs.MedianUS / w.obs.MedianUS
	}
	res.Serving = d.measured[d.serving].variant.Candidate
	res.Weights = d.weights
	return res, nil
}

// next returns the next candidate to sample: the analytical baseline if
// unmeasured, else the unmeasured candidate with the lowest predicted cost
// under the current weights.
func (d *Daemon) next(valid []ScheduleCandidate, baseline ScheduleCandidate) (ScheduleCandidate, bool) {
	if d.measured[baseline.Name] == nil {
		return baseline, true
	}
	if d.warm != "" && d.measured[d.warm] == nil {
		if i := slices.IndexFunc(valid, func(c ScheduleCandidate) bool { return c.Name == d.warm }); i >= 0 {
			return valid[i], true
		}
	}
	var rest []ScheduleCandidate
	for _, c := range valid {
		if d.measured[c.Name] == nil {
			rest = append(rest, c)
		}
	}
	if len(rest) == 0 {
		return ScheduleCandidate{}, false
	}
	return slices.MinFunc(rest, func(a, b ScheduleCandidate) int {
		if c := cmp.Compare(d.weights.Predict(a), d.weights.Predict(b)); c != 0 {
			return c
		}
		return cmp.Compare(a.Name, b.Name)
	}), true
}

// trial builds and samples c, records the observation and refines the
// weights. A candidate that fails to build, spills in SASS, or fails to
// sample is recorded as disqualified so it is not tried again.
func (d *Daemon) trial(ctx context.Context, c ScheduleCandidate) Trial {
	t := Trial{Candidate: c}
	fail := func(v Variant, err error) Trial {
		t.Err = err
		d.measured[c.Name] = &measurement{variant: v, err: err}
		return t
	}
	v, err := d.cfg.Builder.Build(ctx, c)
	if err != nil {
		return fail(Variant{Candidate: c}, fmt.Errorf("build %s: %w", c.Name, err))
	}
	if v.SpillStores > 0 || v.SpillLoads > 0 {
		return fail(v, fmt.Errorf("%s: %d STL and %d LDL in SASS: %w", c.Name, v.SpillStores, v.SpillLoads, ErrSpillDisqualified))
	}
	obs, h, err := d.collector.SampleKernel(ctx, v, d.cfg.Iters)
	if err != nil {
		return fail(v, fmt.Errorf("sample %s: %w", c.Name, err))
	}
	t.Observation, t.Stalls = obs, h
	m := &measurement{variant: v, obs: obs, stalls: h}
	if obs.Metrics.LocalSpillLoadBytes > 0 || obs.Metrics.LocalSpillStoreBytes > 0 {
		m.err = fmt.Errorf("%s: %d bytes spilled, %d reloaded: %w", c.Name,
			obs.Metrics.LocalSpillStoreBytes, obs.Metrics.LocalSpillLoadBytes, ErrSpillDisqualified)
	} else if obs.MedianUS <= 0 || math.IsNaN(obs.MedianUS) || math.IsInf(obs.MedianUS, 0) {
		m.err = fmt.Errorf("%s: bad median %v us", c.Name, obs.MedianUS)
	}
	t.Err = m.err
	d.measured[c.Name] = m
	metrics := obs.Metrics
	metrics.DurationNS = uint64(obs.MedianUS * 1000)
	d.model.RecordObservation(d.cfg.ShapeKey, c, h.StallProfile(), metrics)
	d.weights = d.weights.Refine(d.cfg.Twin, c, h)
	return t
}

// swap checks agreement and publishes w in place of cur.
func (d *Daemon) swap(ctx context.Context, cur, w *measurement) (bool, error) {
	if d.cfg.Agreement != nil {
		if err := d.cfg.Agreement.Agree(ctx, cur.variant, w.variant); err != nil {
			w.err = fmt.Errorf("%s: %w: %w", w.variant.Candidate.Name, ErrNumericalDrift, err)
			return false, nil
		}
	}
	if d.cfg.Swapper != nil {
		if err := d.cfg.Swapper.Swap(ctx, w.variant); err != nil {
			return false, fmt.Errorf("daemon step: %w", err)
		}
	}
	d.serving = w.variant.Candidate.Name
	return true, nil
}

// Run calls Step every SampleInterval until ctx is done, sending each
// result to results if it is not nil. It returns ctx's error, or the first
// Step error.
func (d *Daemon) Run(ctx context.Context, candidates []ScheduleCandidate, results chan<- StepResult) error {
	tick := time.NewTicker(d.cfg.SampleInterval)
	defer tick.Stop()
	for {
		res, err := d.Step(ctx, candidates)
		if err != nil {
			return err
		}
		if results != nil {
			select {
			case results <- res:
			case <-ctx.Done():
				return ctx.Err()
			}
		}
		select {
		case <-tick.C:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
}

func trialErrs(ts []Trial) []error {
	var errs []error
	for _, t := range ts {
		if t.Err != nil {
			errs = append(errs, t.Err)
		}
	}
	return errs
}
