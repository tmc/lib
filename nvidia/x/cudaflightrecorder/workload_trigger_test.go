package cudaflightrecorder_test

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/driver"
	fr "github.com/tmc/lib/nvidia/x/cudaflightrecorder"
	nt "github.com/tmc/lib/nvidia/x/nvtelemetry"
)

type workloadHostCapture struct {
	calls atomic.Int64
	err   error
}

func (p *workloadHostCapture) Capture(context.Context, *fr.Recorder, fr.TriggerRequest, func()) (fr.CaptureReport, error) {
	p.calls.Add(1)
	// This fake provider starts no native session and produces no GPU records.
	return fr.CaptureReport{Provider: "host_fake", Closed: true}, p.err
}

func workloadCoordinator(t *testing.T, armed bool, provider *workloadHostCapture) (*fr.EscalationCoordinator, *fr.BufferWriterFactory) {
	t.Helper()
	if driver.Loaded() || cupti.Loaded() {
		t.Fatal("host control loaded a vendor library")
	}
	t.Cleanup(func() {
		if driver.Loaded() || cupti.Loaded() {
			t.Error("host bridge loaded a vendor library")
		}
	})
	rec, err := fr.New(fr.Config{Capacity: 8, MaxSnapshotBytes: 1 << 20, MaxProviderOutputBytes: 1 << 16})
	if err != nil {
		t.Fatal(err)
	}
	if err := rec.Start(); err != nil {
		t.Fatal(err)
	}
	factory := fr.NewBufferWriterFactory()
	budgets := fr.QualificationBudgets()
	budgets.HysteresisCount, budgets.MaxRingRecords = 1, 8
	budgets.MaxRetainedBytes, budgets.MaxCaptureDuration, budgets.CooldownDuration = 1<<20, time.Second, time.Nanosecond
	coord, err := fr.NewEscalationCoordinator(rec, budgets,
		fr.WithClock(fr.NewFakeClock(time.Unix(1, 0))), fr.WithResourceSampler(&fr.FakeResourceSampler{}),
		fr.WithCUPTINegotiator(&fr.FakeCUPTINegotiator{}), fr.WithCaptureWriterFactory(factory), fr.WithIncidentCapture(provider))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := coord.Stop(); err != nil {
			t.Error(err)
		}
		rec.Stop()
	})
	if armed {
		if err := coord.Arm(); err != nil {
			t.Fatal(err)
		}
	}
	return coord, factory
}

func workloadProfile() fr.WorkloadTriggerConfig {
	return fr.WorkloadTriggerConfig{
		Provider: "triton", Version: "25.06", Metric: "nv_inference_queue_duration_us", Unit: "us",
		Mode: "counter_delta", TimeSource: "collector_observation",
		Labels: map[string]string{"model": "fixture", "version": "1"}, TenantID: "host-fixture",
		WindowDuration: time.Second, StaticThreshold: 20,
		Detector: fr.DetectorConfig{Mode: "rolling", Window: 3, Warmup: 2, Delta: 5, MaxGap: 2 * time.Second},
		Limits:   nt.WorkloadLimits{MaxBytes: 4096, MaxMetrics: 64, MaxStringBytes: 256},
	}
}

func workloadSource() nt.WorkloadSource {
	return nt.WorkloadSource{Provider: "triton", Version: "25.06", RunID: "run1", Phase: "steady", ClockDomain: "sample-clock", ClockEpochID: "epoch1"}
}

func workloadRead(t *testing.T, raw string, source nt.WorkloadSource, when int64, stamp bool) nt.WorkloadExport {
	t.Helper()
	e, err := nt.ReadServingExposition(context.Background(), strings.NewReader(raw), source, nt.WorkloadLimits{})
	if err != nil {
		t.Fatal(err)
	}
	if stamp {
		e, err = nt.StampWorkloadSnapshot(e, when, source.ClockDomain, source.ClockEpochID)
		if err != nil {
			t.Fatal(err)
		}
	}
	return e
}

func workloadCounter(t *testing.T, value string, when int64, source nt.WorkloadSource) nt.WorkloadExport {
	t.Helper()
	return workloadRead(t, "# TYPE nv_inference_queue_duration_us counter\nnv_inference_queue_duration_us{model=\"fixture\",version=\"1\"} "+value+"\n", source, when, true)
}

func workloadWindow(i int64) fr.WorkloadTriggerWindow {
	return fr.WorkloadTriggerWindow{StartNanos: i * int64(time.Second), EndNanos: (i + 1) * int64(time.Second), ClockDomain: "sample-clock", ClockEpochID: "epoch1"}
}

func workloadObserve(t *testing.T, bridge *fr.WorkloadTrigger, i int64, before, after string) (fr.WorkloadTriggerResult, error) {
	t.Helper()
	w := workloadWindow(i)
	return bridge.Observe(context.Background(), workloadCounter(t, before, w.StartNanos, workloadSource()), workloadCounter(t, after, w.EndNanos, workloadSource()), w)
}

func workloadWarm(t *testing.T, bridge *fr.WorkloadTrigger) {
	t.Helper()
	for i := int64(0); i < 2; i++ {
		r, err := workloadObserve(t, bridge, i, fmt.Sprint(10*i), fmt.Sprint(10*(i+1)))
		if err != nil || r.Decision.Ready || r.Observation.Value != 10 || r.Incident != nil {
			t.Fatalf("warmup %+v %v", r, err)
		}
	}
}

func TestWorkloadTriggerPublicChain(t *testing.T) {
	for _, mode := range []string{"rolling", "ewma"} {
		t.Run(mode, func(t *testing.T) {
			provider := &workloadHostCapture{}
			coord, factory := workloadCoordinator(t, true, provider)
			profile := workloadProfile()
			profile.Detector.Mode, profile.Detector.Alpha = mode, .5
			bridge, err := fr.NewWorkloadTrigger(profile, coord)
			if err != nil {
				t.Fatal(err)
			}
			profile.Labels["model"] = "caller mutation"
			workloadWarm(t, bridge)
			r, err := workloadObserve(t, bridge, 2, "20", "50")
			if err != nil || !r.Decision.Ready || !r.Decision.Anomaly || !r.StaticExceeded || r.ExactValue != "30" || r.Rounded || r.Incident == nil || r.Incident.Outcome != "captured" || provider.calls.Load() != 1 {
				t.Fatalf("public chain %+v %v calls=%d", r, err, provider.calls.Load())
			}
			if r.Incident.Capture.Started || r.Incident.Capture.Provider != "host_fake" || !r.Incident.Capture.Closed || r.Incident.Trigger.PreTrigger.Available {
				t.Fatalf("fake capture upgraded to native/prehistory proof: %+v", r.Incident)
			}
			if r.Before.Source.RunID != "run1" || r.Before.Source.Phase != "steady" || r.Series[0].Semantic != "cumulative_queue_time_including_cache" || r.Series[0].Unit != "us" || r.Incident.Trigger.Details["clock_epoch"] != "epoch1" {
				t.Fatal("window context or cumulative-time semantics lost")
			}
			for _, e := range []nt.WorkloadExport{r.Before, r.After} {
				h := sha256.Sum256(e.Raw)
				if e.RawSHA256 != hex.EncodeToString(h[:]) {
					t.Fatal("raw digest lost")
				}
			}
			bundle, err := fr.ReadBundle(bytes.NewReader(factory.Buffers[r.Incident.IncidentID].Bytes()))
			if err != nil {
				t.Fatal(err)
			}
			var trigger fr.TriggerEvidence
			if err := json.Unmarshal(bundle.Files["incident.json"], &trigger); err != nil {
				t.Fatal(err)
			}
			if trigger.Details["before_sha256"] != r.Before.RawSHA256 || trigger.Details["after_sha256"] != r.After.RawSHA256 || bundle.Manifest.GPURecords != 0 {
				t.Fatal("incident raw closure changed or fake GPU records appeared")
			}
			data, err := json.Marshal(r)
			if err != nil {
				t.Fatal(err)
			}
			var replay fr.WorkloadTriggerResult
			if err := json.Unmarshal(data, &replay); err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(replay.Before.Raw, r.Before.Raw) || replay.ExactValue != "30" || replay.Incident.Trigger.Details["after_sha256"] != r.After.RawSHA256 {
				t.Fatal("serialized window/decision/receipt closure lost")
			}
		})
	}
}

func TestWorkloadTriggerRefusalAndProviderFailure(t *testing.T) {
	for _, name := range []string{"disarmed", "provider failure", "static differs"} {
		t.Run(name, func(t *testing.T) {
			fault := errors.New("host provider failure")
			provider := &workloadHostCapture{}
			if name == "provider failure" {
				provider.err = fault
			}
			coord, _ := workloadCoordinator(t, name == "provider failure", provider)
			bridge, err := fr.NewWorkloadTrigger(workloadProfile(), coord)
			if err != nil {
				t.Fatal(err)
			}
			workloadWarm(t, bridge)
			after := "50"
			if name == "static differs" {
				after = "37"
			}
			r, err := workloadObserve(t, bridge, 2, "20", after)
			if !r.Decision.Anomaly || r.Incident == nil || r.Incident.Trigger.Details["before_sha256"] != r.Before.RawSHA256 {
				t.Fatalf("decision/refusal evidence lost %+v %v", r, err)
			}
			if name == "provider failure" {
				if !errors.Is(err, fault) || r.Incident.Outcome != "failed" || provider.calls.Load() != 1 {
					t.Fatalf("provider failure lost %+v %v", r, err)
				}
			} else if !errors.Is(err, fr.ErrEscalationRefused) || r.Incident.RefusalReason != fr.RefusalDisarmed || provider.calls.Load() != 0 {
				t.Fatalf("refusal replaced %+v %v", r, err)
			}
			if name == "static differs" && r.StaticExceeded {
				t.Fatal("detector anomaly relabeled a static breach")
			}
		})
	}
}

func TestWorkloadTriggerExactValueDoesNotOverflowIncidentDetails(t *testing.T) {
	provider := &workloadHostCapture{}
	coord, _ := workloadCoordinator(t, true, provider)
	bridge, err := fr.NewWorkloadTrigger(workloadProfile(), coord)
	if err != nil {
		t.Fatal(err)
	}
	workloadWarm(t, bridge)
	result, err := workloadObserve(t, bridge, 2, "20", "40."+strings.Repeat("3", 2500))
	if err != nil || len(result.ExactValue) <= 4096 || !result.Decision.Anomaly || result.Incident == nil || result.Incident.Outcome != "captured" || provider.calls.Load() != 1 {
		t.Fatalf("legitimate high-precision value lost incident admission: %v %+v", err, result.Decision)
	}
	digest := sha256.Sum256([]byte(result.ExactValue))
	if result.Incident.Trigger.Details["exact_value_sha256"] != hex.EncodeToString(digest[:]) {
		t.Fatal("exact value reference does not identify the retained rational")
	}
	if _, ok := result.Incident.Trigger.Details["exact_value"]; ok {
		t.Fatal("unbounded exact rational copied into bounded trigger details")
	}
}

func TestWorkloadTriggerInvalidClearsOnlyOwnHistory(t *testing.T) {
	for _, name := range []string{"digest", "raw", "numeric mutation", "unit", "kind", "source context", "counter reset", "missing series", "duration", "epoch", "stale time", "non-serving", "extra counter label", "cancel"} {
		t.Run(name, func(t *testing.T) {
			coord, _ := workloadCoordinator(t, false, &workloadHostCapture{})
			bridge, err := fr.NewWorkloadTrigger(workloadProfile(), coord)
			if err != nil {
				t.Fatal(err)
			}
			other, err := fr.NewWorkloadTrigger(workloadProfile(), coord)
			if err != nil {
				t.Fatal(err)
			}
			workloadWarm(t, bridge)
			workloadWarm(t, other)
			w := workloadWindow(2)
			a := workloadCounter(t, "20", w.StartNanos, workloadSource())
			b := workloadCounter(t, "50", w.EndNanos, workloadSource())
			ctx := context.Background()
			switch name {
			case "digest":
				b.RawSHA256 = strings.Repeat("0", 64)
			case "raw":
				b.Raw = append(b.Raw, []byte("# changed raw\n")...)
			case "numeric mutation":
				b.Metrics[0].RawValue = "999"
			case "unit":
				b.Metrics[0].Unit = "requests"
			case "kind":
				b.Metrics[0].Kind = "gauge"
			case "source context":
				s := workloadSource()
				s.RunID = "other-run"
				b = workloadCounter(t, "50", w.EndNanos, s)
			case "counter reset":
				b = workloadCounter(t, "19", w.EndNanos, workloadSource())
			case "missing series":
				b = workloadRead(t, "# missing selected metric\n", workloadSource(), w.EndNanos, true)
			case "duration":
				w.EndNanos++
			case "epoch":
				w.ClockEpochID = "different"
			case "stale time":
				b.Metrics[0].TimeNanos--
			case "non-serving":
				b.Source.Provider, b.Source.Version = "wandb", "0.19.11"
			case "extra counter label":
				rawA := strings.Replace(string(a.Raw), `version="1"}`, `version="1",le="1"}`, 1)
				a = workloadRead(t, rawA, workloadSource(), w.StartNanos, true)
				raw := strings.Replace(string(b.Raw), `version="1"}`, `version="1",le="1"}`, 1)
				b = workloadRead(t, raw, workloadSource(), w.EndNanos, true)
			case "cancel":
				var cancel context.CancelFunc
				ctx, cancel = context.WithCancel(ctx)
				cancel()
			}
			r, err := bridge.Observe(ctx, a, b, w)
			if err == nil || r.Decision.Ready || r.Decision.Anomaly || r.Observation.Available || r.Incident != nil || r.Decision.Reason != "missing" {
				t.Fatalf("invalid input became a healthy sample %+v %v", r, err)
			}
			r, err = workloadObserve(t, bridge, 3, "50", "80")
			if err != nil || r.Decision.Ready || r.Decision.Samples != 0 {
				t.Fatalf("invalid input retained old baseline %+v %v", r.Decision, err)
			}
			r, err = workloadObserve(t, other, 3, "50", "80")
			if !errors.Is(err, fr.ErrEscalationRefused) || !r.Decision.Anomaly || !r.Decision.Ready {
				t.Fatalf("another instance's history erased %+v %v", r.Decision, err)
			}
		})
	}
}

func workloadHistogram(t *testing.T, count, sum, bound string, when int64, source nt.WorkloadSource) nt.WorkloadExport {
	t.Helper()
	raw := fmt.Sprintf(`# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{model_name="fixture",le="%s"} 0
vllm:time_to_first_token_seconds_bucket{model_name="fixture",le="+Inf"} %s
vllm:time_to_first_token_seconds_count{model_name="fixture"} %s
vllm:time_to_first_token_seconds_sum{model_name="fixture"} %s
`, bound, count, count, sum)
	return workloadRead(t, raw, source, when, true)
}

func histogramProfile() fr.WorkloadTriggerConfig {
	c := workloadProfile()
	c.Provider, c.Version, c.Metric, c.Unit = "vllm", "0.8.5", "vllm:time_to_first_token_seconds", "s"
	c.Mode, c.MinPopulation = "histogram_mean", 2
	c.Labels = map[string]string{"model_name": "fixture"}
	c.Detector.Delta, c.StaticThreshold = .5, 2
	return c
}

func TestWorkloadTriggerHistogramAndContext(t *testing.T) {
	coord, _ := workloadCoordinator(t, false, &workloadHostCapture{})
	bridge, err := fr.NewWorkloadTrigger(histogramProfile(), coord)
	if err != nil {
		t.Fatal(err)
	}
	s := workloadSource()
	s.Provider, s.Version = "vllm", "0.8.5"
	observe := func(i int64, beforeCount, afterCount, beforeSum, afterSum, bound string, source nt.WorkloadSource) (fr.WorkloadTriggerResult, error) {
		w := workloadWindow(i)
		return bridge.Observe(context.Background(), workloadHistogram(t, beforeCount, beforeSum, bound, w.StartNanos, source), workloadHistogram(t, afterCount, afterSum, bound, w.EndNanos, source), w)
	}
	for i := int64(0); i < 2; i++ {
		r, e := observe(i, fmt.Sprint(10*i), fmt.Sprint(10*(i+1)), fmt.Sprint(10*i), fmt.Sprint(10*(i+1)), "0.1", s)
		if e != nil || r.ExactValue != "1" || r.Population != "10" || r.Decision.Ready {
			t.Fatalf("hist warmup %+v %v", r, e)
		}
	}
	r, err := observe(2, "20", "30", "20", "50", "0.1", s)
	if !errors.Is(err, fr.ErrEscalationRefused) || !r.Decision.Anomaly || !r.StaticExceeded || r.ExactValue != "3" || len(r.Series) != 4 {
		t.Fatalf("aggregate mean chain %+v %v", r, err)
	}
	// Two valid windows with changed bucket bounds cannot share a baseline.
	r, err = observe(3, "30", "40", "50", "80", "0.2", s)
	if err != nil || r.Decision.Ready || r.Decision.Reason != "identity_changed" {
		t.Fatalf("bucket schema reused %+v %v", r.Decision, err)
	}
	for _, change := range []func(*nt.WorkloadSource){func(s *nt.WorkloadSource) { s.RunID = "run2" }, func(s *nt.WorkloadSource) { s.Phase = "new-phase" }, func(s *nt.WorkloadSource) { s.ClockEpochID = "epoch2" }} {
		s2 := s
		change(&s2)
		w := workloadWindow(4)
		w.ClockEpochID = s2.ClockEpochID
		r, err = bridge.Observe(context.Background(), workloadHistogram(t, "40", "80", "0.2", w.StartNanos, s2), workloadHistogram(t, "50", "110", "0.2", w.EndNanos, s2), w)
		if err != nil || r.Decision.Ready || r.Decision.Reason != "identity_changed" {
			t.Fatalf("context reused %+v %v", r.Decision, err)
		}
	}
}

func TestWorkloadTriggerHistogramRefusalsAndRounding(t *testing.T) {
	for _, name := range []string{"bounds change within window", "reset", "empty", "small population", "rounded"} {
		t.Run(name, func(t *testing.T) {
			coord, _ := workloadCoordinator(t, false, &workloadHostCapture{})
			bridge, err := fr.NewWorkloadTrigger(histogramProfile(), coord)
			if err != nil {
				t.Fatal(err)
			}
			s := workloadSource()
			s.Provider, s.Version = "vllm", "0.8.5"
			w := workloadWindow(0)
			a := workloadHistogram(t, "10", "10", "0.1", w.StartNanos, s)
			b := workloadHistogram(t, "20", "20", "0.1", w.EndNanos, s)
			switch name {
			case "bounds change within window":
				b = workloadHistogram(t, "20", "20", "0.2", w.EndNanos, s)
			case "reset":
				b = workloadHistogram(t, "9", "20", "0.1", w.EndNanos, s)
			case "empty":
				b = workloadHistogram(t, "10", "10", "0.1", w.EndNanos, s)
			case "small population":
				b = workloadHistogram(t, "11", "11", "0.1", w.EndNanos, s)
			case "rounded":
				b = workloadHistogram(t, "13", "11", "0.1", w.EndNanos, s)
			}
			r, err := bridge.Observe(context.Background(), a, b, w)
			if name == "rounded" {
				if err != nil || r.ExactValue != "1/3" || !r.Rounded || r.Population != "3" {
					t.Fatalf("exact mean lost %+v %v", r, err)
				}
			} else if err == nil || r.Observation.Available || r.Decision.Ready || r.Incident != nil {
				t.Fatalf("invalid histogram admitted %+v %v", r, err)
			}
		})
	}
}

func TestWorkloadTriggerProducerTimeAndBounds(t *testing.T) {
	coord, _ := workloadCoordinator(t, false, &workloadHostCapture{})
	profile := workloadProfile()
	profile.TimeSource = "producer"
	bridge, err := fr.NewWorkloadTrigger(profile, coord)
	if err != nil {
		t.Fatal(err)
	}
	w := workloadWindow(0)
	read := func(v string, when int64) nt.WorkloadExport {
		return workloadRead(t, fmt.Sprintf("# TYPE nv_inference_queue_duration_us counter\nnv_inference_queue_duration_us{model=\"fixture\",version=\"1\"} %s %d\n", v, when/int64(time.Millisecond)), workloadSource(), when, false)
	}
	r, err := bridge.Observe(context.Background(), read("0", w.StartNanos), read("9007199254740993", w.EndNanos), w)
	if err != nil || r.ExactValue != "9007199254740993" || !r.Rounded || r.Before.Metrics[0].TimeSource != "producer" {
		t.Fatalf("exact/native-time spelling lost %+v %v", r, err)
	}
	profile.TimeSource = "collector_observation"
	collector, err := fr.NewWorkloadTrigger(profile, coord)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := collector.Observe(context.Background(), read("0", w.StartNanos), read("10", w.EndNanos), w); err == nil {
		t.Fatal("native timestamp overwritten")
	}
	profile.Limits.MaxBytes = 8
	small, err := fr.NewWorkloadTrigger(profile, coord)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := small.Observe(context.Background(), workloadCounter(t, "0", 0, workloadSource()), workloadCounter(t, "10", int64(time.Second), workloadSource()), w); err == nil {
		t.Fatal("raw reader byte bound bypassed")
	}
	profile = workloadProfile()
	bridge, err = fr.NewWorkloadTrigger(profile, coord)
	if err != nil {
		t.Fatal(err)
	}
	// The serving parser refuses out-of-range raw numbers before a comparable
	// window exists. Deliver malformed Raw to the bridge, rather than asking the
	// fixture reader to produce an export it cannot admit.
	oversized := workloadCounter(t, "10", w.EndNanos, workloadSource())
	oversized.Raw = []byte(strings.Replace(string(oversized.Raw), " 10\n", " 1e1000\n", 1))
	digest := sha256.Sum256(oversized.Raw)
	oversized.RawSHA256 = hex.EncodeToString(digest[:])
	if _, err := bridge.Observe(context.Background(), workloadCounter(t, "0", w.StartNanos, workloadSource()), oversized, w); err == nil {
		t.Fatal("raw reader overflow accepted")
	}
	if _, err := workloadObserve(t, bridge, 0, "0", "1e-1000"); err == nil {
		t.Fatal("underflow fabricated healthy zero")
	}
}

func TestWorkloadTriggerConfigAndConcurrency(t *testing.T) {
	coord, _ := workloadCoordinator(t, false, &workloadHostCapture{})
	for _, change := range []func(*fr.WorkloadTriggerConfig){
		func(c *fr.WorkloadTriggerConfig) { c.Mode = "gauge" },
		func(c *fr.WorkloadTriggerConfig) { c.TimeSource = "sensor" },
		func(c *fr.WorkloadTriggerConfig) { c.WindowDuration = 0 },
		func(c *fr.WorkloadTriggerConfig) { c.WindowDuration = 3 * time.Second },
		func(c *fr.WorkloadTriggerConfig) { c.StaticThreshold = math.NaN() },
		func(c *fr.WorkloadTriggerConfig) { c.Metric = "" },
		func(c *fr.WorkloadTriggerConfig) { c.Labels["le"] = "1" },
		func(c *fr.WorkloadTriggerConfig) { c.Mode = "histogram_mean"; c.MinPopulation = 0 },
		func(c *fr.WorkloadTriggerConfig) { c.Detector.Mode = "other" },
	} {
		c := workloadProfile()
		change(&c)
		if _, err := fr.NewWorkloadTrigger(c, coord); err == nil {
			t.Fatalf("invalid config accepted %+v", c)
		}
	}
	if _, err := fr.NewWorkloadTrigger(workloadProfile(), nil); err == nil {
		t.Fatal("nil coordinator accepted")
	}
	var zero fr.WorkloadTrigger
	if _, err := zero.Observe(context.Background(), nt.WorkloadExport{}, nt.WorkloadExport{}, workloadWindow(0)); err == nil {
		t.Fatal("zero bridge accepted")
	}
	bridge, err := fr.NewWorkloadTrigger(workloadProfile(), coord)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := bridge.Observe(nil, nt.WorkloadExport{}, nt.WorkloadExport{}, workloadWindow(0)); err == nil {
		t.Fatal("nil context accepted")
	}
	var group sync.WaitGroup
	for i := int64(0); i < 8; i++ {
		a := workloadCounter(t, "0", i*int64(time.Second), workloadSource())
		b := workloadCounter(t, "10", (i+1)*int64(time.Second), workloadSource())
		group.Go(func() {
			r, err := bridge.Observe(context.Background(), a, b, workloadWindow(i))
			if err != nil || r.Decision.Anomaly || r.Incident != nil {
				t.Errorf("concurrent equal windows %+v %v", r.Decision, err)
			}
		})
	}
	group.Wait()
}
