package nvtelemetry_test

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

// workloadServer serves testdata/workload fixtures on a loopback listener.
type workloadServer struct {
	*httptest.Server
	mu     sync.Mutex
	status int
	body   []byte
	hits   atomic.Int64
	other  atomic.Int64
}

func newWorkloadServer(t *testing.T, fixture string) *workloadServer {
	t.Helper()
	b, err := os.ReadFile("testdata/workload/" + fixture)
	if err != nil {
		t.Fatal(err)
	}
	s := &workloadServer{status: 200, body: b}
	s.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/export" {
			s.other.Add(1)
			w.Write([]byte("redirect target"))
			return
		}
		s.hits.Add(1)
		s.mu.Lock()
		status, body := s.status, s.body
		s.mu.Unlock()
		switch status {
		case http.StatusFound:
			http.Redirect(w, r, "/elsewhere", status)
		case -1:
			<-r.Context().Done()
		default:
			w.WriteHeader(status)
			w.Write(body)
		}
	}))
	t.Cleanup(s.Close)
	return s
}

func (s *workloadServer) set(status int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.status = status
}

func (s *workloadServer) fetch(src nvtelemetry.WorkloadSource, limits nvtelemetry.WorkloadLimits) func(context.Context) (nvtelemetry.WorkloadHTTPResult, error) {
	return func(ctx context.Context) (nvtelemetry.WorkloadHTTPResult, error) {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, s.URL+"/export", nil)
		if err != nil {
			return nvtelemetry.WorkloadHTTPResult{}, err
		}
		return nvtelemetry.FetchWorkload(ctx, s.Client(), req, src, limits)
	}
}

var (
	vllmSource   = nvtelemetry.WorkloadSource{Provider: "vllm", Version: "0.8.5", RunID: "run"}
	mlflowSource = nvtelemetry.WorkloadSource{Provider: "mlflow", Version: "2.22.0", RunID: "run"}
)

const (
	refreshWallDomain = "synthetic_unix_utc"
	refreshWallEpoch  = "fixture-wall-1"
)

func newTestWorkloadRefresher(t *testing.T, fetch func(context.Context) (nvtelemetry.WorkloadHTTPResult, error), clock *nvtelemetry.FakeClock, cfg nvtelemetry.RefreshConfig) *nvtelemetry.WorkloadRefresher {
	t.Helper()
	r, err := nvtelemetry.NewWorkloadRefresher(fetch, clock, refreshWallDomain, refreshWallEpoch, cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { r.Close() })
	return r
}

func TestWorkloadRefresherStampsUntimedSnapshot(t *testing.T) {
	srv := newWorkloadServer(t, "vllm.prom")
	clock := nvtelemetry.NewFakeClock(0, 1_700_000_000_000_000_000)
	r := newTestWorkloadRefresher(t, srv.fetch(vllmSource, nvtelemetry.WorkloadLimits{}), clock, refreshTestConfig)
	if _, fresh := r.Latest(clock.NowMonotonic()); fresh || r.Ready() {
		t.Fatal("fresh before first fetch")
	}
	if err := r.Step(context.Background()); err != nil {
		t.Fatal(err)
	}
	got, fresh := r.Latest(clock.NowMonotonic())
	if !fresh || len(got.Export.Metrics) == 0 {
		t.Fatalf("fresh %v metrics %d", fresh, len(got.Export.Metrics))
	}
	for _, m := range got.Export.Metrics {
		if !m.HasTime || m.TimeSource != "collector_observation" || m.TimeNanos != 1_700_000_000_000_000_000 || m.Source.ClockDomain != refreshWallDomain || m.Source.ClockEpochID != refreshWallEpoch {
			t.Fatalf("metric %s not stamped with the collector observation: %+v", m.Name, m)
		}
	}
	if !strings.Contains(string(got.Export.Raw), "vllm:prompt_tokens_total") {
		t.Fatal("raw bytes lost")
	}
}

func TestWorkloadRefresherPreservesNativeTimestamps(t *testing.T) {
	srv := newWorkloadServer(t, "mlflow.json")
	clock := nvtelemetry.NewFakeClock(0, 42)
	r := newTestWorkloadRefresher(t, srv.fetch(mlflowSource, nvtelemetry.WorkloadLimits{}), clock, refreshTestConfig)
	if err := r.Step(context.Background()); err != nil {
		t.Fatal(err)
	}
	got, fresh := r.Latest(clock.NowMonotonic())
	if !fresh || len(got.Export.Metrics) != 2 {
		t.Fatalf("fresh %v metrics %d", fresh, len(got.Export.Metrics))
	}
	want := []int64{1700000000000000000, 1700000000020000000}
	for i, m := range got.Export.Metrics {
		if !m.HasTime || m.TimeSource != "producer" || m.TimeNanos != want[i] {
			t.Fatalf("metric %d native time replaced: %+v", i, m)
		}
		if m.Source.ClockDomain == refreshWallDomain {
			t.Fatalf("metric %d took the collector clock domain", i)
		}
	}
}

func TestWorkloadRefresherFailuresGoStale(t *testing.T) {
	tests := []struct {
		name        string
		breakServer func(*workloadServer)
		limits      nvtelemetry.WorkloadLimits
	}{
		{"server error", func(s *workloadServer) { s.set(500) }, nvtelemetry.WorkloadLimits{}},
		{"oversize", func(s *workloadServer) {
			s.mu.Lock()
			s.body = append(s.body, strings.Repeat("# padding\n", 64)...)
			s.mu.Unlock()
		}, nvtelemetry.WorkloadLimits{MaxBytes: 1024}},
		{"redirect", func(s *workloadServer) { s.set(http.StatusFound) }, nvtelemetry.WorkloadLimits{}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := newWorkloadServer(t, "vllm.prom")
			clock := nvtelemetry.NewFakeClock(0, 1000)
			r := newTestWorkloadRefresher(t, srv.fetch(vllmSource, tt.limits), clock, refreshTestConfig)
			if err := r.Step(context.Background()); err != nil {
				t.Fatal(err)
			}
			first, _ := r.Latest(clock.NowMonotonic())
			tt.breakServer(srv)
			clock.Advance(time.Second)
			if err := r.Step(context.Background()); err == nil {
				t.Fatal("step succeeded")
			}
			got, fresh := r.Latest(clock.NowMonotonic())
			if !fresh || got.Export.RawSHA256 != first.Export.RawSHA256 {
				t.Fatal("last good result not served within MaxAge")
			}
			clock.Advance(5 * time.Second)
			if _, fresh := r.Latest(clock.NowMonotonic()); fresh || r.Ready() {
				t.Fatal("fresh past MaxAge after failures")
			}
			if st := r.Status(); st.Failures != 1 || st.Cycles != 2 || st.LastError == "" {
				t.Fatalf("status = %+v", st)
			}
			if srv.other.Load() != 0 {
				t.Fatal("redirect target was fetched")
			}
		})
	}
}

func TestWorkloadRefresherLatestIsDeepCopy(t *testing.T) {
	srv := newWorkloadServer(t, "vllm.prom")
	clock := nvtelemetry.NewFakeClock(0, 1000)
	rank := 3
	src := vllmSource
	src.Rank = &rank
	src.Units = map[string]string{"x": "1"}
	r := newTestWorkloadRefresher(t, srv.fetch(src, nvtelemetry.WorkloadLimits{}), clock, refreshTestConfig)
	if err := r.Step(context.Background()); err != nil {
		t.Fatal(err)
	}
	a, _ := r.Latest(clock.NowMonotonic())
	a.Export.Raw[0] = 'X'
	a.Export.Metrics[0].Labels["model_name"] = "mutated"
	a.Export.Metrics[0].Name = "mutated"
	*a.Export.Source.Rank = 99
	*a.Export.Metrics[0].Source.Rank = 99
	a.Export.Source.Units["x"] = "mutated"
	a.Export.Metrics[0].Source.Units["x"] = "mutated"
	b, _ := r.Latest(clock.NowMonotonic())
	if b.Export.Raw[0] == 'X' || b.Export.Metrics[0].Labels["model_name"] == "mutated" || b.Export.Metrics[0].Name == "mutated" ||
		*b.Export.Source.Rank != 3 || *b.Export.Metrics[0].Source.Rank != 3 || b.Export.Source.Units["x"] != "1" || b.Export.Metrics[0].Source.Units["x"] != "1" {
		t.Fatal("Latest shares memory with the refresher")
	}
}

func TestWorkloadRefresherAnnotationsAreCopied(t *testing.T) {
	srv := newWorkloadServer(t, "pytorch.json")
	clock := nvtelemetry.NewFakeClock(0, 1000)
	src := nvtelemetry.WorkloadSource{Provider: "pytorch", Version: "2.7.0", RunID: "run"}
	r := newTestWorkloadRefresher(t, srv.fetch(src, nvtelemetry.WorkloadLimits{}), clock, refreshTestConfig)
	if err := r.Step(context.Background()); err != nil {
		t.Fatal(err)
	}
	a, _ := r.Latest(clock.NowMonotonic())
	if len(a.Annotations) == 0 || len(a.Annotations[0].Args) == 0 {
		t.Fatalf("annotations = %+v", a.Annotations)
	}
	a.Annotations[0].Name = "mutated"
	a.Annotations[0].Args[0] = 'X'
	b, _ := r.Latest(clock.NowMonotonic())
	if b.Annotations[0].Name == "mutated" || b.Annotations[0].Args[0] == 'X' {
		t.Fatal("annotations share memory with the refresher")
	}
}

func TestWorkloadRefresherBlockingFetchTimesOut(t *testing.T) {
	srv := newWorkloadServer(t, "vllm.prom")
	srv.set(-1)
	clock := nvtelemetry.NewFakeClock(0, 1000)
	r := newTestWorkloadRefresher(t, srv.fetch(vllmSource, nvtelemetry.WorkloadLimits{}), clock, nvtelemetry.RefreshConfig{Interval: time.Second, Timeout: 20 * time.Millisecond, MaxAge: time.Second})
	start := time.Now()
	if err := r.Step(context.Background()); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("step = %v, want deadline exceeded", err)
	}
	if d := time.Since(start); d > 5*time.Second {
		t.Fatalf("step took %v", d)
	}
	if r.Ready() {
		t.Fatal("ready after timeout")
	}
}

func TestWorkloadRefresherRunClose(t *testing.T) {
	srv := newWorkloadServer(t, "vllm.prom")
	clock := nvtelemetry.NewFakeClock(0, 1000)
	r := newTestWorkloadRefresher(t, srv.fetch(vllmSource, nvtelemetry.WorkloadLimits{}), clock, nvtelemetry.RefreshConfig{Interval: time.Millisecond, Timeout: time.Second, MaxAge: time.Hour})
	done := make(chan error, 1)
	go func() { done <- r.Run(context.Background()) }()
	var wg sync.WaitGroup
	for range 4 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 50 {
				if res, ok := r.Latest(clock.NowMonotonic()); ok && len(res.Export.Metrics) > 0 {
					res.Export.Metrics[0].Labels["model_name"] = "mutated"
				}
				r.Status()
				r.Ready()
			}
		}()
	}
	for r.Status().Cycles < 3 {
		time.Sleep(time.Millisecond)
	}
	if err := r.Close(); err != nil {
		t.Fatal(err)
	}
	wg.Wait()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("run = %v, want nil after close", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("run did not return after close")
	}
	hits := srv.hits.Load()
	if err := r.Step(context.Background()); err == nil {
		t.Fatal("step after close succeeded")
	}
	if srv.hits.Load() != hits {
		t.Fatal("fetched after close")
	}
	if _, fresh := r.Latest(clock.NowMonotonic()); fresh || r.Ready() {
		t.Fatal("closed refresher reports fresh")
	}
}

func TestNewWorkloadRefresherRefusals(t *testing.T) {
	fetch := func(context.Context) (nvtelemetry.WorkloadHTTPResult, error) {
		return nvtelemetry.WorkloadHTTPResult{}, nil
	}
	clock := nvtelemetry.NewFakeClock(0, 0)
	tests := []struct {
		name          string
		fetch         func(context.Context) (nvtelemetry.WorkloadHTTPResult, error)
		clock         nvtelemetry.Clock
		domain, epoch string
		cfg           nvtelemetry.RefreshConfig
	}{
		{"nil fetch", nil, clock, "d", "e", refreshTestConfig},
		{"nil clock", fetch, nil, "d", "e", refreshTestConfig},
		{"missing domain", fetch, clock, "", "e", refreshTestConfig},
		{"missing epoch", fetch, clock, "d", "", refreshTestConfig},
		{"zero config", fetch, clock, "d", "e", nvtelemetry.RefreshConfig{}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := nvtelemetry.NewWorkloadRefresher(tt.fetch, tt.clock, tt.domain, tt.epoch, tt.cfg); err == nil {
				t.Fatal("constructor succeeded")
			}
		})
	}
}

func TestWorkloadRefresherCloseDuringFetch(t *testing.T) {
	started := make(chan struct{})
	fetch := func(ctx context.Context) (nvtelemetry.WorkloadHTTPResult, error) {
		close(started)
		<-ctx.Done()
		return nvtelemetry.WorkloadHTTPResult{}, ctx.Err()
	}
	clock := nvtelemetry.NewFakeClock(0, 1000)
	r := newTestWorkloadRefresher(t, fetch, clock, refreshTestConfig)
	errc := make(chan error, 1)
	go func() { errc <- r.Step(context.Background()) }()
	<-started
	r.Close()
	if err := <-errc; err == nil || !strings.Contains(err.Error(), "closed") {
		t.Fatalf("step = %v, want closed", err)
	}
	if st := r.Status(); st.Failures != 0 || st.HasSuccess || !st.Closed {
		t.Fatalf("status after close during fetch = %+v", st)
	}
}
