package nvtelemetry

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"slices"
)

// WorkloadRefresher periodically fetches one workload source and serves the
// last successful result while it is within MaxAge. A snapshot whose metrics
// carry no producer timestamps is stamped with the collector's observation
// time; producer timestamps are never replaced. It is exercised only against
// loopback fixtures; it does not qualify a live framework endpoint. The zero
// value is not usable; use NewWorkloadRefresher.
type WorkloadRefresher struct {
	s      *refreshState
	fetch  func(context.Context) (WorkloadHTTPResult, error)
	domain string
	epoch  string
	latest WorkloadHTTPResult // guarded by s.mu
}

// NewWorkloadRefresher returns a refresher that calls fetch, typically a
// closure over FetchWorkload, once per cycle. Untimed snapshots are stamped
// with clock.NowUnix in the named wall clock domain and epoch. Freshness uses
// the clock's monotonic time.
func NewWorkloadRefresher(fetch func(context.Context) (WorkloadHTTPResult, error), clock Clock, domain, epoch string, cfg RefreshConfig) (*WorkloadRefresher, error) {
	if fetch == nil {
		return nil, errors.New("workload refresh: nil fetch")
	}
	if domain == "" || epoch == "" {
		return nil, errors.New("workload refresh: unknown observation clock")
	}
	s, err := newRefreshState(clock, cfg)
	if err != nil {
		return nil, err
	}
	return &WorkloadRefresher{s: s, fetch: fetch, domain: domain, epoch: epoch}, nil
}

// Step runs one bounded fetch. On failure the previous result is kept and
// ages toward MaxAge.
func (w *WorkloadRefresher) Step(ctx context.Context) error {
	s := w.s
	s.stepMu.Lock()
	defer s.stepMu.Unlock()
	res, err := refreshDo(ctx, s, w.fetch)
	if err == nil && len(res.Export.Metrics) > 0 && !anyWorkloadTime(res.Export.Metrics) {
		res.Export, err = StampWorkloadSnapshot(res.Export, s.clock.NowUnix(), w.domain, w.epoch)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.status.Closed {
		return errors.New("workload refresh: closed")
	}
	if err != nil {
		s.fail(err)
		return fmt.Errorf("workload refresh: %w", err)
	}
	w.latest = cloneWorkloadResult(res)
	s.succeed(len(res.Export.Metrics)+len(res.Annotations), 0)
	return nil
}

func anyWorkloadTime(metrics []WorkloadMetric) bool {
	for _, m := range metrics {
		if m.HasTime {
			return true
		}
	}
	return false
}

// Run calls Step immediately and then every Interval until ctx is done or the
// refresher is closed. Step errors are reported by Status. Run returns
// ctx.Err() on cancellation and nil after Close.
func (w *WorkloadRefresher) Run(ctx context.Context) error {
	return w.s.run(ctx, w.Step)
}

// Latest returns a copy of the last successful result and whether it was
// obtained within MaxAge of now, a time on the refresher clock's monotonic
// scale. Before any success it returns the zero result and false.
func (w *WorkloadRefresher) Latest(now int64) (WorkloadHTTPResult, bool) {
	w.s.mu.Lock()
	defer w.s.mu.Unlock()
	return cloneWorkloadResult(w.latest), w.s.fresh(now)
}

// Ready reports whether a fetch has succeeded within MaxAge.
func (w *WorkloadRefresher) Ready() bool {
	w.s.mu.Lock()
	defer w.s.mu.Unlock()
	return w.s.fresh(w.s.clock.NowMonotonic())
}

// Status returns a copy of the refresher's cycle accounting.
func (w *WorkloadRefresher) Status() RefreshStatus {
	w.s.mu.Lock()
	defer w.s.mu.Unlock()
	return w.s.status
}

// Close stops Run, cancels an in-flight fetch and refuses later cycles. Close
// is safe to call more than once.
func (w *WorkloadRefresher) Close() error {
	w.s.close()
	return nil
}

func cloneWorkloadSource(s WorkloadSource) WorkloadSource {
	if s.Rank != nil {
		r := *s.Rank
		s.Rank = &r
	}
	s.Units = maps.Clone(s.Units)
	return s
}

func cloneWorkloadResult(r WorkloadHTTPResult) WorkloadHTTPResult {
	e := r.Export
	e.Source = cloneWorkloadSource(e.Source)
	e.Raw = slices.Clone(e.Raw)
	e.UnmappedNames = slices.Clone(e.UnmappedNames)
	e.Metrics = slices.Clone(e.Metrics)
	for i := range e.Metrics {
		e.Metrics[i].Source = cloneWorkloadSource(e.Metrics[i].Source)
		e.Metrics[i].Labels = maps.Clone(e.Metrics[i].Labels)
	}
	r.Export = e
	r.Annotations = slices.Clone(r.Annotations)
	for i := range r.Annotations {
		a := &r.Annotations[i]
		a.PID = json.RawMessage(slices.Clone([]byte(a.PID)))
		a.TID = json.RawMessage(slices.Clone([]byte(a.TID)))
		a.Args = json.RawMessage(slices.Clone([]byte(a.Args)))
	}
	return r
}
