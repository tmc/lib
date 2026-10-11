package nvtelemetry

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"reflect"
	"strings"
	"testing"
)

func workloadFixture(t *testing.T, name string) []byte {
	t.Helper()
	raw, err := os.ReadFile("testdata/workload/" + name)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

func TestServingWorkloadReplay(t *testing.T) {
	s := WorkloadSource{Provider: "triton", Version: "25.06", RunID: "serving1"}
	read := func(name string, time int64) WorkloadExport {
		raw := workloadFixture(t, name)
		e, err := ReadServingExposition(context.Background(), bytes.NewReader(raw), s, WorkloadLimits{})
		if err != nil {
			t.Fatal(err)
		}
		h := sha256.Sum256(raw)
		if e.RawSHA256 != hex.EncodeToString(h[:]) || !bytes.Equal(e.Raw, raw) {
			t.Fatal("raw changed")
		}
		for _, p := range e.Metrics {
			if p.HasTime {
				t.Fatal("scrape timestamp invented")
			}
		}
		e, err = StampWorkloadSnapshot(e, time, "test_monotonic", "test_epoch")
		if err != nil {
			t.Fatal(err)
		}
		return e
	}
	a, b := read("triton-before.prom", 0), read("triton-after.prom", 100)
	delta, err := CounterWorkloadWindow(a, b, 0, 100, "test_monotonic", "test_epoch")
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]string{"nv_inference_request_success": "3", "nv_inference_request_failure": "1", "nv_inference_count": "8", "nv_inference_exec_count": "2", "nv_cache_num_hits_per_model": "1", "nv_inference_queue_duration_us": "70"}
	if len(delta) != len(want) {
		t.Fatal(delta)
	}
	for _, d := range delta {
		if d.Delta != want[d.Name] {
			t.Fatalf("%s delta %s", d.Name, d.Delta)
		}
		if d.Labels["model"] != "batching" {
			t.Fatal("model lost")
		}
	}
	if ExactWorkloadJoin(a) == nil {
		t.Fatal("aggregate became exact request records")
	}
	for _, tc := range []struct {
		name   string
		mutate func(*WorkloadExport)
	}{
		{"counter_reset", func(e *WorkloadExport) { e.Metrics[0].RawValue = "1" }},
		{"rolling_version", func(e *WorkloadExport) { e.Source.Version = "25.07" }},
		{"missing_series", func(e *WorkloadExport) { e.Metrics = e.Metrics[1:] }},
		{"wrong_clock", func(e *WorkloadExport) { e.Metrics[0].Source.ClockEpochID = "other" }},
		{"token_semantics", func(e *WorkloadExport) { e.Metrics[0].Semantic = "processed_generation_tokens" }},
		{"batch_denominator", func(e *WorkloadExport) { e.Metrics[3].Unit = "requests" }},
		{"point_identity", func(e *WorkloadExport) { e.Metrics[0].Source.RunID = "other" }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			changed := b
			changed.Metrics = append([]WorkloadMetric(nil), b.Metrics...)
			tc.mutate(&changed)
			if _, err := CounterWorkloadWindow(a, changed, 0, 100, "test_monotonic", "test_epoch"); err == nil {
				t.Fatal("mutation accepted")
			}
		})
	}
}

func TestServingMappingsAndControls(t *testing.T) {
	for _, tc := range []struct {
		provider, version, file string
		n                       int
	}{
		{"vllm", "0.8.5", "vllm.prom", 10},
		{"ray", "2.44.0", "ray.prom", 2},
	} {
		t.Run(tc.provider, func(t *testing.T) {
			e, err := ReadServingExposition(context.Background(), bytes.NewReader(workloadFixture(t, tc.file)), WorkloadSource{Provider: tc.provider, Version: tc.version, RunID: "serve"}, WorkloadLimits{})
			if err != nil {
				t.Fatal(err)
			}
			if len(e.Metrics) != tc.n {
				t.Fatalf("got %d metrics", len(e.Metrics))
			}
			if tc.provider == "vllm" {
				if e.Metrics[0].RawValue != "9007199254740993" || e.Metrics[0].Semantic != "processed_prefill_tokens" {
					t.Fatal("token semantics or integer rounded")
				}
				if e.Metrics[2].Labels["finished_reason"] != "stop" || e.Metrics[3].Labels["finished_reason"] != "abort" {
					t.Fatal("cancellation identity lost")
				}
			} else if e.Metrics[0].Semantic != "logical_resource_assignment" {
				t.Fatal("assigned GPUs became physical activity")
			}
		})
	}
	raw := string(workloadFixture(t, "vllm.prom"))
	for _, tc := range []struct{ name, old, new string }{
		{"tokens_as_gauge", "vllm:prompt_tokens_total counter", "vllm:prompt_tokens_total gauge"},
		{"batch_histogram_denominator", "vllm:time_to_first_token_seconds_count{model_name=\"example\"} 3", "vllm:time_to_first_token_seconds_count{model_name=\"example\"} 2"},
		{"histogram_order", "le=\"1\"} 2", "le=\"1\"} 0"},
		{"cache_as_memory", "} 0.25", "} 25"},
		{"negative_tokens", "} 40", "} -40"},
		{"fractional_tokens", "} 40", "} 1.5"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			mutated := strings.Replace(raw, tc.old, tc.new, 1)
			if mutated == raw {
				t.Fatal("mutation did not apply")
			}
			if _, err := ReadServingExposition(context.Background(), strings.NewReader(mutated), WorkloadSource{Provider: "vllm", Version: "0.8.5", RunID: "serve"}, WorkloadLimits{}); err == nil {
				t.Fatal("mutated producer admitted")
			}
		})
	}
}

func TestWorkloadHistogramWindow(t *testing.T) {
	raw := string(workloadFixture(t, "vllm.prom"))
	s := WorkloadSource{Provider: "vllm", Version: "0.8.5", RunID: "serve"}
	a, err := ReadServingExposition(context.Background(), strings.NewReader(raw), s, WorkloadLimits{})
	if err != nil {
		t.Fatal(err)
	}
	b, err := ReadServingExposition(context.Background(), strings.NewReader(raw), s, WorkloadLimits{})
	if err != nil {
		t.Fatal(err)
	}
	a, err = StampWorkloadSnapshot(a, 0, "test", "epoch")
	if err != nil {
		t.Fatal(err)
	}
	b, err = StampWorkloadSnapshot(b, 10, "test", "epoch")
	if err != nil {
		t.Fatal(err)
	}
	delta, err := HistogramWorkloadWindow(a, b, 0, 10, "test", "epoch")
	if err != nil {
		t.Fatal(err)
	}
	if len(delta) != 5 {
		t.Fatal(delta)
	}
	for _, d := range delta {
		if d.Delta != "0" {
			t.Fatal("unchanged histogram became observations", delta)
		}
	}
	// A rolling histogram with a different bucket boundary is not compatible
	// even when each snapshot is independently valid.
	mutated := strings.Replace(raw, `le="0.1"`, `le="0.2"`, 1)
	c, err := ReadServingExposition(context.Background(), strings.NewReader(mutated), s, WorkloadLimits{})
	if err != nil {
		t.Fatal(err)
	}
	c, err = StampWorkloadSnapshot(c, 10, "test", "epoch")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := HistogramWorkloadWindow(a, c, 0, 10, "test", "epoch"); err == nil {
		t.Fatal("rolling bucket boundary accepted")
	}
	// Individually valid endpoints can still imply an impossible interval
	// distribution after a partial reset of some cumulative components.
	before := `# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{model_name="example",le="0.1"} 10
vllm:time_to_first_token_seconds_bucket{model_name="example",le="+Inf"} 20
vllm:time_to_first_token_seconds_count{model_name="example"} 20
vllm:time_to_first_token_seconds_sum{model_name="example"} 2
`
	after := strings.ReplaceAll(strings.ReplaceAll(before, `} 10`, `} 20`), `} 20`, `} 25`)
	after = strings.Replace(after, `le="0.1"} 25`, `le="0.1"} 20`, 1)
	after = strings.Replace(after, "} 2\n", "} 3\n", 1)
	d, err := ReadServingExposition(context.Background(), strings.NewReader(before), s, WorkloadLimits{})
	if err != nil {
		t.Fatal(err)
	}
	e, err := ReadServingExposition(context.Background(), strings.NewReader(after), s, WorkloadLimits{})
	if err != nil {
		t.Fatal(err)
	}
	d, err = StampWorkloadSnapshot(d, 0, "test", "epoch")
	if err != nil {
		t.Fatal(err)
	}
	e, err = StampWorkloadSnapshot(e, 10, "test", "epoch")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := HistogramWorkloadWindow(d, e, 0, 10, "test", "epoch"); err == nil {
		t.Fatal("impossible histogram delta accepted")
	}
}

func TestTrainingWorkloadReplay(t *testing.T) {
	rank := 0
	s := WorkloadSource{Provider: "mlflow", Version: "2.22.0", RunID: "train1", Rank: &rank, ClockDomain: "unix_utc", Units: map[string]string{"loss": "1"}}
	e, err := ReadMLflowHistory(bytes.NewReader(workloadFixture(t, "mlflow.json")), s, WorkloadLimits{})
	if err != nil {
		t.Fatal(err)
	}
	if e.Metrics[0].TimeNanos != 1700000000000000000 || e.Metrics[0].RawTime != "1700000000000" || !e.Metrics[0].HasStep || e.Metrics[0].Step != 0 {
		t.Fatal("training timestamp or explicit step zero lost")
	}
	report, err := InspectTrainingSteps([]WorkloadExport{e}, "loss", []int{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(report.GapAfter, []int64{0}) || !reflect.DeepEqual(report.MissingRanks, []int{1}) || !reflect.DeepEqual(report.GapAfterByRank[0], []int64{0}) || !reflect.DeepEqual(report.MissingRanksByStep[0], []int{1}) {
		t.Fatal(report)
	}
	if report.StepKind != "logged_metric_step" {
		t.Fatal("metric step semantics lost")
	}
	mutated := e
	mutated.Metrics = append([]WorkloadMetric(nil), e.Metrics...)
	mutated.Metrics[1].StepKind = "history_step"
	if _, err := InspectTrainingSteps([]WorkloadExport{mutated}, "loss", []int{0}); err == nil {
		t.Fatal("mixed step semantics admitted")
	}
	contradictory := e
	contradictory.Metrics = append([]WorkloadMetric(nil), e.Metrics...)
	contradictory.Metrics[0].Source.RunID = "other"
	if _, err := InspectTrainingSteps([]WorkloadExport{contradictory}, "loss", nil); err == nil {
		t.Fatal("contradictory point identity admitted")
	}
	page := e
	page.NextPageToken = "page2"
	if _, err := InspectTrainingSteps([]WorkloadExport{page}, "loss", []int{0}); err == nil {
		t.Fatal("partial history became complete step gaps")
	}
	ambiguous := e
	ambiguous.Metrics = append([]WorkloadMetric(nil), e.Metrics...)
	ambiguous.Metrics[1].Labels = map[string]string{"variant": "validation"}
	if _, err := InspectTrainingSteps([]WorkloadExport{ambiguous}, "loss", nil); err == nil {
		t.Fatal("different training metric series conflated")
	}
	other := e
	other.Source.RunID = "train2"
	if _, err := InspectTrainingSteps([]WorkloadExport{e, other}, "loss", nil); err == nil {
		t.Fatal("concurrent runs conflated")
	}
	if ExactWorkloadJoin(e) == nil {
		t.Fatal("training aggregate became kernel join")
	}
	// Caller mutation cannot change already-recorded source metadata.
	rank = 4
	s.Units["loss"] = "tokens"
	if *e.Source.Rank != 0 || e.Source.Units["loss"] != "1" {
		t.Fatal("source aliases caller configuration")
	}
}

func TestMLflowNativeReplay(t *testing.T) {
	raw := workloadFixture(t, "mlflow-native.json")
	rank := 0
	e, err := ReadMLflowHistory(bytes.NewReader(raw), WorkloadSource{Provider: "mlflow", Version: "2.22.0", RunID: "4790069ba3984e409f20782b56bb9e3b", Rank: &rank, Units: map[string]string{"loss": "1"}}, WorkloadLimits{})
	if err != nil {
		t.Fatal(err)
	}
	const hash = "241bf82ed4d325c84e7b2c9591b1a583b24bed75d15a55d1fc9c0627ce30c5b5"
	if e.RawSHA256 != hash {
		t.Fatalf("native fixture hash %s", e.RawSHA256)
	}
	report, err := InspectTrainingSteps([]WorkloadExport{e}, "loss", []int{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(report.Steps, []int64{0, 1, 3}) || !reflect.DeepEqual(report.GapAfter, []int64{1}) || !reflect.DeepEqual(report.MissingRanks, []int{1}) {
		t.Fatal(report)
	}
	if !bytes.Equal(e.Raw, raw) || ExactWorkloadJoin(e) == nil {
		t.Fatal("native export changed or exact join admitted")
	}
	t.Logf("MLflow2.22 native raw hash=%s metrics=%d steps=%v gaps=%v missing_ranks=%v exact_join=refused", e.RawSHA256, len(e.Metrics), report.Steps, report.GapAfter, report.MissingRanks)
}

func TestTrainingReadersAndControls(t *testing.T) {
	w, err := ReadWandBHistory(bytes.NewReader(workloadFixture(t, "wandb.jsonl")), WorkloadSource{Provider: "wandb", Version: "0.19.11", RunID: "train1"}, WorkloadLimits{})
	if err != nil {
		t.Fatal(err)
	}
	if len(w.Metrics) != 2 || w.Metrics[0].TimeNanos != 1700000000125000000 || w.Metrics[0].Unit != "unknown" {
		t.Fatal(w.Metrics)
	}
	if w.Metrics[0].StepKind != "history_step" {
		t.Fatal("SDK history index became optimizer step")
	}
	c, err := ReadClearMLScalars(bytes.NewReader(workloadFixture(t, "clearml.json")), WorkloadSource{Provider: "clearml", Version: "1.18.0/api-2.9", RunID: "train1"}, WorkloadLimits{})
	if err != nil {
		t.Fatal(err)
	}
	if len(c.Metrics) != 2 || c.Metrics[0].Labels["variant"] != "train" {
		t.Fatal(c.Metrics)
	}
	if c.Metrics[0].StepKind != "reported_iteration" {
		t.Fatal("reported iteration semantics lost")
	}
	for _, raw := range []string{
		`{"metrics":[{"key":"loss","value":1,"value":2}]}`,
		`{"metrics":[{"key":"loss","value":"0.5"}]}`,
		`{"metrics":[{"key":"loss","value":1,"timestamp":9223372036854775807}]}`,
		`{"metrics":[{"key":"loss","value":1,"step":-1}]}`,
		`{"metrics":[{"key":"loss","value":1e-1000000000}]}`,
		`{"metrics":[{"key":"loss","value":1}]} {}`,
	} {
		if _, err := ReadMLflowHistory(strings.NewReader(raw), WorkloadSource{Provider: "mlflow", Version: "2.22.0", RunID: "train1"}, WorkloadLimits{}); err == nil {
			t.Fatalf("accepted %s", raw)
		}
	}
	m, err := ReadMLflowHistory(strings.NewReader(`{"metrics":[{"key":"loss","value":1}]}`), WorkloadSource{Provider: "mlflow", Version: "2.22.0", RunID: "train1"}, WorkloadLimits{})
	if err != nil {
		t.Fatal(err)
	}
	if m.Metrics[0].HasTime || m.Metrics[0].HasStep {
		t.Fatal("missing clock or step became zero")
	}
	if _, err := InspectTrainingSteps([]WorkloadExport{m}, "loss", nil); err == nil {
		t.Fatal("omitted step became observed step report")
	}
	if _, err := ReadClearMLScalars(bytes.NewReader(workloadFixture(t, "clearml.json")), WorkloadSource{Provider: "clearml", Version: "1.18.0/api-2.9", RunID: "other"}, WorkloadLimits{}); err == nil {
		t.Fatal("wrong native task identity admitted")
	}
	envelope := `{"meta":{"result_code":200},"data":` + string(workloadFixture(t, "clearml.json")) + `}`
	wrapped, err := ReadClearMLScalars(strings.NewReader(envelope), c.Source, WorkloadLimits{})
	if err != nil || len(wrapped.Metrics) != 2 {
		t.Fatal("native HTTP envelope", err)
	}
	failed := strings.Replace(envelope, `"result_code":200`, `"result_code":500`, 1)
	if _, err := ReadClearMLScalars(strings.NewReader(failed), c.Source, WorkloadLimits{}); err == nil {
		t.Fatal("failed native HTTP response admitted")
	}
}

func TestPyTorchAnnotations(t *testing.T) {
	e, a, err := ReadPyTorchAnnotations(bytes.NewReader(workloadFixture(t, "pytorch.json")), WorkloadSource{Provider: "pytorch", Version: "2.7.0", RunID: "train1", ClockDomain: "trace_offset"}, WorkloadLimits{})
	if err != nil {
		t.Fatal(err)
	}
	if len(a) != 1 || a[0].StartNanos != 123125 || a[0].DurationNanos != 25500 || len(e.Metrics) != 0 {
		t.Fatal(a)
	}
	bad := strings.Replace(string(e.Raw), `"dur":25.5`, `"dur":-1`, 1)
	if _, _, err := ReadPyTorchAnnotations(strings.NewReader(bad), e.Source, WorkloadLimits{}); err == nil {
		t.Fatal("negative native interval admitted")
	}
}

func TestWorkloadLimitsAndCancellation(t *testing.T) {
	s := WorkloadSource{Provider: "triton", Version: "25.06", RunID: "serve"}
	if _, err := ReadServingExposition(context.Background(), bytes.NewReader(workloadFixture(t, "triton-before.prom")), s, WorkloadLimits{MaxBytes: 5}); err == nil {
		t.Fatal("byte limit bypassed")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := ReadServingExposition(ctx, strings.NewReader(""), s, WorkloadLimits{}); err != context.Canceled {
		t.Fatal(err)
	}
	if _, err := ReadServingExposition(context.Background(), strings.NewReader(""), WorkloadSource{Provider: "triton", Version: "unknown", RunID: "serve"}, WorkloadLimits{}); err == nil {
		t.Fatal("unknown rolling version admitted")
	}
}

func FuzzReadMLflowHistory(f *testing.F) {
	f.Add(`{"metrics":[{"key":"loss","value":0,"timestamp":0,"step":0}]}`)
	f.Add(`{"metrics":[]}`)
	f.Fuzz(func(t *testing.T, raw string) {
		if len(raw) > 4096 {
			return
		}
		_, _ = ReadMLflowHistory(strings.NewReader(raw), WorkloadSource{Provider: "mlflow", Version: "2.22.0", RunID: "fuzz"}, WorkloadLimits{MaxBytes: 4096, MaxMetrics: 64})
	})
}
