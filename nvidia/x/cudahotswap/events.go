package cudahotswap

import (
	"context"
	"errors"
	"math"
	"strings"
	"sync"
)

// Gap identifies an inclusive range of report sequences that is no longer
// retained. It does not stand for an observed report or a candidate verdict.
type Gap struct {
	First uint64 `json:"first"`
	Last  uint64 `json:"last"`
}

// Event is a detached report snapshot or an explicit retention gap. Report and
// Gap are mutually exclusive. For a gap, Sequence is Gap.Last; resuming after
// that cursor requests the first report following the lost range. Report events
// carry the kernel mutation and configuration revisions supplied by the producer.
type Event struct {
	Sequence       uint64  `json:"sequence"`
	Name           string  `json:"name,omitempty"`
	Generation     uint64  `json:"generation,omitempty"`
	Version        uint64  `json:"version,omitempty"`
	ConfigRevision uint64  `json:"config_revision,omitempty"`
	Label          string  `json:"label,omitempty"`
	Report         *Report `json:"report,omitempty"`
	Gap            *Gap    `json:"gap,omitempty"`
}

var errSlowSubscriber = errors.New("report subscriber queue is full")

type reportEventLimits struct {
	ring        int
	queue       int
	subscribers int
	bytes       int
}

// reportEvents does not collect or qualify reports. The caller must serialize
// report production with its generation/configuration snapshot and must not
// mutate a report while append copies it. No Service API is installed here.
type reportEvents struct {
	mu           sync.Mutex
	limits       reportEventLimits
	ring         []Event
	start, count int
	sequence     uint64
	subs         map[*reportSubscription]struct{}
}

type reportSubscription struct {
	owner  *reportEvents
	events chan Event
	done   chan struct{}
	cause  error // protected by owner.mu
}

func newReportEvents(limits reportEventLimits) (*reportEvents, error) {
	// One replay consists of the retained ring plus at most one gap. Bounding
	// both that queue and subscriber count also bounds total retained snapshots.
	if limits.ring < 1 || limits.ring > 1024 || limits.queue < limits.ring+1 || limits.queue > 2048 || limits.subscribers < 1 || limits.subscribers > 256 || limits.bytes < 1024 || limits.bytes > 1<<20 {
		return nil, errors.New("invalid report event limits")
	}
	return &reportEvents{limits: limits, ring: make([]Event, limits.ring), subs: make(map[*reportSubscription]struct{})}, nil
}

func (r *reportEvents) append(state State, report *Report) (uint64, error) {
	if r == nil || report == nil || state.Name == "" || state.Kind != "kernel" || state.Generation == 0 || state.Version == 0 {
		return 0, errors.New("report event requires a kernel state and report")
	}
	if !reportEventFits(state, report, r.limits.bytes) {
		return 0, errors.New("report event exceeds snapshot budget")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.sequence == math.MaxUint64 {
		return 0, errors.New("report event sequence exhausted")
	}
	r.sequence++
	e := Event{Sequence: r.sequence, Name: strings.Clone(state.Name), Generation: state.Generation, Version: state.Version, ConfigRevision: state.ConfigRevision, Label: strings.Clone(state.Label), Report: cloneEventReport(report)}
	index := (r.start + r.count) % len(r.ring)
	if r.count == len(r.ring) {
		index = r.start
		r.start = (r.start + 1) % len(r.ring)
	} else {
		r.count++
	}
	r.ring[index] = e
	for sub := range r.subs {
		select {
		case sub.events <- cloneReportEvent(e):
		default:
			r.dropLocked(sub, errSlowSubscriber)
		}
	}
	return e.Sequence, nil
}

func (r *reportEvents) subscribe(ctx context.Context, since uint64) (*reportSubscription, error) {
	if r == nil || ctx == nil {
		return nil, errors.New("report subscription requires an owner and context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if since > r.sequence {
		return nil, errors.New("report subscription cursor is in the future")
	}
	if len(r.subs) >= r.limits.subscribers {
		return nil, errors.New("report subscriber limit reached")
	}
	s := &reportSubscription{owner: r, events: make(chan Event, r.limits.queue), done: make(chan struct{})}
	if r.count != 0 {
		first := r.ring[r.start].Sequence
		if since < first-1 {
			s.events <- Event{Sequence: first - 1, Gap: &Gap{First: since + 1, Last: first - 1}}
		}
		for i := 0; i < r.count; i++ {
			e := r.ring[(r.start+i)%len(r.ring)]
			if e.Sequence > since {
				s.events <- cloneReportEvent(e)
			}
		}
	}
	r.subs[s] = struct{}{}
	go func() {
		select {
		case <-ctx.Done():
			s.cancel(ctx.Err())
		case <-s.done:
		}
	}()
	return s, nil
}

func (r *reportEvents) dropLocked(s *reportSubscription, cause error) {
	if _, ok := r.subs[s]; !ok {
		return
	}
	delete(r.subs, s)
	s.cause = cause
	close(s.events)
	close(s.done)
}

func (s *reportSubscription) cancel(cause error) {
	if s == nil {
		return
	}
	s.owner.mu.Lock()
	defer s.owner.mu.Unlock()
	s.owner.dropLocked(s, cause)
}

func (s *reportSubscription) err() error {
	if s == nil {
		return nil
	}
	s.owner.mu.Lock()
	defer s.owner.mu.Unlock()
	return s.cause
}

func cloneReportEvent(e Event) Event {
	e.Report = cloneEventReport(e.Report)
	if e.Gap != nil {
		g := *e.Gap
		e.Gap = &g
	}
	return e
}

func cloneEventReport(r *Report) *Report {
	if r == nil {
		return nil
	}
	c := *r
	c.Outcome = Outcome(strings.Clone(string(c.Outcome)))
	c.Task = strings.Clone(c.Task)
	c.TaskDef = strings.Clone(c.TaskDef)
	c.Candidate = strings.Clone(c.Candidate)
	c.Entry = strings.Clone(c.Entry)
	c.Reason = strings.Clone(c.Reason)
	if r.Device != nil {
		x := *r.Device
		x.Name = strings.Clone(x.Name)
		x.Compute = strings.Clone(x.Compute)
		c.Device = &x
	}
	if r.Compile != nil {
		x := *r.Compile
		x.SourceSHA256 = strings.Clone(x.SourceSHA256)
		x.PTXSHA256 = strings.Clone(x.PTXSHA256)
		x.Log = strings.Clone(x.Log)
		x.Arch = strings.Clone(x.Arch)
		c.Compile = &x
	}
	if r.Correctness != nil {
		x := *r.Correctness
		x.Reference = strings.Clone(x.Reference)
		x.OutputPrefill = strings.Clone(x.OutputPrefill)
		if r.Correctness.Trials != nil {
			x.Trials = make([]TrialReport, len(r.Correctness.Trials))
			copy(x.Trials, r.Correctness.Trials)
		}
		for i := range x.Trials {
			if w := x.Trials[i].Comparison.Worst; w != nil {
				v := *w
				x.Trials[i].Comparison.Worst = &v
			}
		}
		c.Correctness = &x
	}
	cloneTiming := func(t *TimingReport) *TimingReport {
		if t == nil {
			return nil
		}
		x := *t
		x.Source = strings.Clone(x.Source)
		x.InputPolicy = strings.Clone(x.InputPolicy)
		if t.SampleNanos != nil {
			x.SampleNanos = make([]int64, len(t.SampleNanos))
			copy(x.SampleNanos, t.SampleNanos)
		}
		return &x
	}
	c.Timing = cloneTiming(r.Timing)
	c.Baseline = cloneTiming(r.Baseline)
	if r.Replay != nil {
		x := *r.Replay
		x.Clock = strings.Clone(x.Clock)
		if x.SignP != nil {
			v := *x.SignP
			x.SignP = &v
		}
		if r.Replay.Outputs != nil {
			x.Outputs = make([]ReplayOutputReport, len(r.Replay.Outputs))
			copy(x.Outputs, r.Replay.Outputs)
		}
		cloneMetrics := func(m ReplayMetrics) ReplayMetrics {
			if m.MaxAbs != nil {
				v := *m.MaxAbs
				m.MaxAbs = &v
			}
			if m.RMS != nil {
				v := *m.RMS
				m.RMS = &v
			}
			if m.MaxULP != nil {
				v := *m.MaxULP
				m.MaxULP = &v
			}
			return m
		}
		for i := range x.Outputs {
			x.Outputs[i].Noise = cloneMetrics(x.Outputs[i].Noise)
			x.Outputs[i].Candidate = cloneMetrics(x.Outputs[i].Candidate)
		}
		if r.Replay.Trials != nil {
			x.Trials = make([]ReplayTrialReport, len(r.Replay.Trials))
			copy(x.Trials, r.Replay.Trials)
		}
		for i := range x.Trials {
			t := &x.Trials[i]
			t.Arm, t.Digest, t.Error = strings.Clone(t.Arm), strings.Clone(t.Digest), strings.Clone(t.Error)
			if t.DeviceNanos != nil {
				v := *t.DeviceNanos
				t.DeviceNanos = &v
			}
		}
		c.Replay = &x
	}
	if r.Speedup != nil {
		x := *r.Speedup
		x.Method = strings.Clone(x.Method)
		x.Note = strings.Clone(x.Note)
		c.Speedup = &x
	}
	if r.Unmeasured != nil {
		c.Unmeasured = make([]string, len(r.Unmeasured))
		copy(c.Unmeasured, r.Unmeasured)
	}
	for i := range c.Unmeasured {
		c.Unmeasured[i] = strings.Clone(c.Unmeasured[i])
	}
	return &c
}

// The budget bounds retained Go values, not encoded JSON bytes. Check before
// allocating a clone, with conservative fixed/per-element overhead and bounded
// strings. String clones avoid retaining oversized substring backing arrays.
func reportEventFits(s State, r *Report, limit int) bool {
	left := limit
	add := func(n int) bool {
		if n < 0 || n > left {
			return false
		}
		left -= n
		return true
	}
	text := func(xs ...string) bool {
		for _, x := range xs {
			// Include conservative string allocation slack, not just its
			// logical length. Check before multiplying to avoid overflow.
			if len(x) > left/2 || !add(2*len(x)) || !add(16) {
				return false
			}
		}
		return true
	}
	items := func(n, size int) bool {
		if n > left/size {
			return false
		}
		return add(n * size)
	}
	if !add(1024) || !text(s.Name, s.Label, r.Task, r.TaskDef, r.Candidate, r.Entry, r.Reason, string(r.Outcome)) || !items(len(r.Unmeasured), 16) {
		return false
	}
	for _, x := range r.Unmeasured {
		if !text(x) {
			return false
		}
	}
	if d := r.Device; d != nil && !text(d.Name, d.Compute) {
		return false
	}
	if c := r.Compile; c != nil && !text(c.SourceSHA256, c.PTXSHA256, c.Log, c.Arch) {
		return false
	}
	if c := r.Correctness; c != nil && (!text(c.Reference, c.OutputPrefill) || !items(len(c.Trials), 128)) {
		return false
	}
	for _, t := range []*TimingReport{r.Timing, r.Baseline} {
		if t != nil && (!text(t.Source, t.InputPolicy) || !items(len(t.SampleNanos), 8)) {
			return false
		}
	}
	if s := r.Speedup; s != nil && !text(s.Method, s.Note) {
		return false
	}
	if p := r.Replay; p != nil {
		if !text(p.Clock) || !items(len(p.Outputs), 256) || !items(len(p.Trials), 128) {
			return false
		}
		for _, trial := range p.Trials {
			if !text(trial.Arm, trial.Digest, trial.Error) {
				return false
			}
		}
	}
	return true
}
