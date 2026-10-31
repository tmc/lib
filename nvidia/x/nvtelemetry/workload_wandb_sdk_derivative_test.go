package nvtelemetry_test

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

// The fixture is a byte-identical copy of a derivative decoded from one native
// W&B 0.19.11 offline .wandb transaction log through the SDK datastore reader
// (see wandb01911-sdk-decoded.meta.json). The SDK did not emit it as JSONL.
// These tests replay it through the public ReadWandBHistory and
// InspectTrainingSteps path. They are not evidence of native
// wandb-history.jsonl, a vendor download, authentic training or full
// original120 coverage. Negative controls edit in-test copies of the bytes and
// are not authentic producer output.

const (
	wandbSDKFixture = "testdata/workload/wandb01911-sdk-decoded.jsonl"
	wandbSDKMeta    = "testdata/workload/wandb01911-sdk-decoded.meta.json"
	wandbSDKSHA256  = "0166da6ec1fd06277146913760c73cc38d69fbd1881b8ca5983600a1b6776bb6"
	wandbSDKNative  = "dad963039ddd336f160c34182e2ba86ecbc436d3ea3c47692bbd5550ab7cd721"
	wandbSDKRun     = "5f5i3ig2"
)

// wandbSDKSource is the caller-declared context. No engine, rank, unit or
// clock mapping was observed, so none is declared.
func wandbSDKSource(run string) nvtelemetry.WorkloadSource {
	return nvtelemetry.WorkloadSource{Provider: "wandb", Version: "0.19.11", RunID: run}
}

func wandbSDKBytes(t *testing.T) []byte {
	t.Helper()
	raw, err := os.ReadFile(wandbSDKFixture)
	if err != nil {
		t.Fatal(err)
	}
	if sum := sha256.Sum256(raw); len(raw) != 274 || hex.EncodeToString(sum[:]) != wandbSDKSHA256 {
		t.Fatalf("fixture: %d bytes, sha256 %x; want 274 bytes, %s", len(raw), sum, wandbSDKSHA256)
	}
	return raw
}

// wandbSDKEdit returns a copy of raw with from, which must occur exactly
// once, replaced by to.
func wandbSDKEdit(t *testing.T, raw []byte, from, to string) []byte {
	t.Helper()
	if n := bytes.Count(raw, []byte(from)); n != 1 {
		t.Fatalf("%q occurs %d times, want 1", from, n)
	}
	return bytes.ReplaceAll(raw, []byte(from), []byte(to))
}

func wandbSDKRead(t *testing.T, raw []byte, run string) nvtelemetry.WorkloadExport {
	t.Helper()
	e, err := nvtelemetry.ReadWandBHistory(bytes.NewReader(raw), wandbSDKSource(run), nvtelemetry.WorkloadLimits{})
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(raw)
	if !bytes.Equal(e.Raw, raw) || e.RawSHA256 != hex.EncodeToString(sum[:]) {
		t.Fatalf("raw: %d bytes, sha256 %s; want %d bytes, %x", len(e.Raw), e.RawSHA256, len(raw), sum)
	}
	return e
}

// wandbSDKWant is the audit's oracle: loss 4, 2 and 1 at explicit history
// steps 0, 1 and 3, with producer timestamps in seconds and no unit.
func wandbSDKWant() []nvtelemetry.WorkloadMetric {
	point := func(value, time string, nanos, step int64) nvtelemetry.WorkloadMetric {
		return nvtelemetry.WorkloadMetric{
			Name:        "loss",
			Semantic:    "user_defined_training_metric",
			Unit:        "unknown",
			Kind:        "gauge",
			Temporality: "instantaneous",
			RawValue:    value,
			RawTime:     time,
			RawTimeUnit: "seconds",
			TimeNanos:   nanos,
			HasTime:     true,
			TimeSource:  "producer",
			Step:        step,
			HasStep:     true,
			StepKind:    "history_step",
		}
	}
	return []nvtelemetry.WorkloadMetric{
		point("4", "1.793108838593494e+09", 1793108838593494000, 0),
		point("2", "1.793108838593622e+09", 1793108838593622000, 1),
		point("1", "1.793108838593757e+09", 1793108838593757000, 3),
	}
}

func wandbSDKReport() nvtelemetry.TrainingStepReport {
	return nvtelemetry.TrainingStepReport{
		StepKind:           "history_step",
		Steps:              []int64{0, 1, 3},
		GapAfter:           []int64{1},
		UnknownRank:        true,
		GapAfterByRank:     map[int][]int64{},
		MissingRanksByStep: map[int64][]int{},
	}
}

func checkWandBSDKSource(t *testing.T, what string, s nvtelemetry.WorkloadSource, run string) {
	t.Helper()
	if s.Provider != "wandb" || s.Version != "0.19.11" || s.RunID != run || s.Engine != "" || s.Phase != "" ||
		s.Rank != nil || s.ClockDomain != "" || s.ClockEpochID != "" || len(s.Units) != 0 {
		t.Errorf("%s source = %+v, want declared wandb 0.19.11 run %s only", what, s, run)
	}
}

// wandbSDKMetrics returns e's metrics, each with its source checked against
// the declared context and then cleared for comparison.
func wandbSDKMetrics(t *testing.T, e nvtelemetry.WorkloadExport) []nvtelemetry.WorkloadMetric {
	t.Helper()
	checkWandBSDKSource(t, "export", e.Source, e.Source.RunID)
	var out []nvtelemetry.WorkloadMetric
	for i, p := range e.Metrics {
		checkWandBSDKSource(t, fmt.Sprintf("metric %d", i), p.Source, e.Source.RunID)
		p.Source = nvtelemetry.WorkloadSource{}
		out = append(out, p)
	}
	return out
}

// wandbSDKLoss decodes the loss spellings in raw with encoding/json,
// independently of the reader.
func wandbSDKLoss(t *testing.T, raw []byte) []string {
	t.Helper()
	var out []string
	d := json.NewDecoder(bytes.NewReader(raw))
	d.UseNumber()
	for {
		var row struct {
			Loss json.Number `json:"loss"`
		}
		err := d.Decode(&row)
		if err == io.EOF {
			return out
		}
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, row.Loss.String())
	}
}

func rawWandBSDKValues(e nvtelemetry.WorkloadExport) []string {
	var out []string
	for _, p := range e.Metrics {
		out = append(out, p.RawValue)
	}
	return out
}

// checkWandBSDKMeta checks that the sidecar describes the fixture, its native
// source and run, and does not claim that the SDK emitted JSONL.
func checkWandBSDKMeta(t *testing.T, e nvtelemetry.WorkloadExport) {
	t.Helper()
	b, err := os.ReadFile(wandbSDKMeta)
	if err != nil {
		t.Fatal(err)
	}
	var m struct {
		SHA256       string `json:"sha256"`
		Bytes        int    `json:"bytes"`
		Rows         int    `json:"rows"`
		Kind         string `json:"kind"`
		SDKJSONL     *bool  `json:"sdk_emitted_jsonl"`
		RunID        string `json:"run_id"`
		NativeSource struct {
			SHA256 string `json:"sha256"`
		} `json:"native_source"`
		SDK struct {
			Name    string `json:"name"`
			Version string `json:"version"`
		} `json:"sdk"`
	}
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatal(err)
	}
	if m.SHA256 != e.RawSHA256 || m.Bytes != len(e.Raw) || m.Rows != 3 || m.Kind != "sdk_datastore_decoded_derivative" ||
		m.SDKJSONL == nil || *m.SDKJSONL || m.RunID != e.Source.RunID || m.NativeSource.SHA256 != wandbSDKNative ||
		m.SDK.Name != e.Source.Provider || m.SDK.Version != e.Source.Version {
		t.Errorf("sidecar does not describe the fixture:\n%s", b)
	}
}

func TestWandBSDKDerivative(t *testing.T) {
	e := wandbSDKRead(t, wandbSDKBytes(t), wandbSDKRun)
	if e.RawSHA256 != wandbSDKSHA256 || e.NextPageToken != "" || e.UnmappedNames != nil {
		t.Fatalf("export sha256 %s, page token %q, unmapped %q", e.RawSHA256, e.NextPageToken, e.UnmappedNames)
	}
	// Nested values, strings and _runtime stay only in Raw, spelled as decoded.
	for _, s := range []string{`"nested":{"b":{"c":"x"},"a":1}`, `"note":"short"`, `"_runtime":0.274079`} {
		if !bytes.Contains(e.Raw, []byte(s)) {
			t.Errorf("raw lacks %s", s)
		}
	}
	if got, want := wandbSDKMetrics(t, e), wandbSDKWant(); !reflect.DeepEqual(got, want) {
		t.Fatalf("metrics\n%+v\nwant\n%+v", got, want)
	}
	want := []string{"4", "2", "1"}
	if got, raw := rawWandBSDKValues(e), wandbSDKLoss(t, e.Raw); !slices.Equal(got, want) || !slices.Equal(raw, want) {
		t.Fatalf("loss: reader %q, independent decode %q; want %q", got, raw, want)
	}

	r, err := nvtelemetry.InspectTrainingSteps([]nvtelemetry.WorkloadExport{e}, "loss", nil)
	if err != nil {
		t.Fatal(err)
	}
	if want := wandbSDKReport(); !reflect.DeepEqual(r, want) {
		t.Fatalf("report\n%+v\nwant\n%+v", r, want)
	}
	// Rank 0 is not inferred: a caller who expects rank 0 finds it missing.
	r, err = nvtelemetry.InspectTrainingSteps([]nvtelemetry.WorkloadExport{e}, "loss", []int{0})
	if err != nil {
		t.Fatal(err)
	}
	wantRank := wandbSDKReport()
	wantRank.MissingRanks = []int{0}
	wantRank.MissingRanksByStep = map[int64][]int{0: {0}, 1: {0}, 3: {0}}
	if !reflect.DeepEqual(r, wantRank) {
		t.Fatalf("expected rank 0 report\n%+v\nwant\n%+v", r, wantRank)
	}

	if err := nvtelemetry.ExactWorkloadJoin(e); err == nil {
		t.Error("ExactWorkloadJoin accepted the derivative")
	}
	checkWandBSDKMeta(t, e)
}

func TestWandBSDKDerivativeReaderRefuses(t *testing.T) {
	raw := wandbSDKBytes(t)
	tests := []struct {
		name     string
		from, to string
		limits   nvtelemetry.WorkloadLimits
		want     string
	}{
		{"duplicate_loss", `"loss":2,`, `"loss":2,"loss":2,`, nvtelemetry.WorkloadLimits{}, `duplicate stream json key "loss"`},
		{"duplicate_timestamp", `"_timestamp":1.793108838593757e+09`, `"_timestamp":1.793108838593757e+09,"_timestamp":1.793108838593757e+09`, nvtelemetry.WorkloadLimits{}, `duplicate stream json key "_timestamp"`},
		{"duplicate_nested_key", `{"c":"x"}`, `{"c":"x","c":"x"}`, nvtelemetry.WorkloadLimits{}, `duplicate stream json key "c"`},
		{"timestamp_exponent", `1.793108838593622e+09`, `1.793108838593622e+999`, nvtelemetry.WorkloadLimits{}, "nonfinite workload JSON number"},
		{"negative_step", `"_step":3,`, `"_step":-3,`, nvtelemetry.WorkloadLimits{}, "negative training step"},
		{"max_bytes", "", "", nvtelemetry.WorkloadLimits{MaxBytes: 273}, "workload byte limit exceeded"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			b := raw
			if tt.from != "" {
				b = wandbSDKEdit(t, raw, tt.from, tt.to)
			}
			_, err := nvtelemetry.ReadWandBHistory(bytes.NewReader(b), wandbSDKSource(wandbSDKRun), tt.limits)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("ReadWandBHistory error %v, want %q", err, tt.want)
			}
		})
	}
}

// TestWandBSDKDerivativeMissingTimestamp checks that a row without
// _timestamp withholds its time rather than being refused or given one.
func TestWandBSDKDerivativeMissingTimestamp(t *testing.T) {
	b := wandbSDKEdit(t, wandbSDKBytes(t), `,"_timestamp":1.793108838593622e+09`, "")
	e := wandbSDKRead(t, b, wandbSDKRun)
	want := wandbSDKWant()
	want[1].RawTime, want[1].RawTimeUnit, want[1].TimeNanos, want[1].HasTime, want[1].TimeSource = "", "", 0, false, ""
	if got := wandbSDKMetrics(t, e); !reflect.DeepEqual(got, want) {
		t.Fatalf("metrics\n%+v\nwant\n%+v", got, want)
	}
}

func TestWandBSDKDerivativeConsumerRefuses(t *testing.T) {
	raw := wandbSDKBytes(t)
	tests := []struct {
		name    string
		exports func(t *testing.T) []nvtelemetry.WorkloadExport
		want    string
	}{
		{"missing_steps", func(t *testing.T) []nvtelemetry.WorkloadExport {
			b := raw
			for _, s := range []string{`"_step":0,`, `"_step":1,`, `"_step":3,`} {
				b = wandbSDKEdit(t, b, s, "")
			}
			e := wandbSDKRead(t, b, wandbSDKRun)
			want := wandbSDKWant()
			for i := range want {
				want[i].Step, want[i].HasStep, want[i].StepKind = 0, false, ""
			}
			if got := wandbSDKMetrics(t, e); !reflect.DeepEqual(got, want) {
				t.Fatalf("metrics\n%+v\nwant\n%+v", got, want)
			}
			return []nvtelemetry.WorkloadExport{e}
		}, "no explicit training steps observed"},
		{"point_context", func(t *testing.T) []nvtelemetry.WorkloadExport {
			e := wandbSDKRead(t, raw, wandbSDKRun)
			e.Metrics = slices.Clone(e.Metrics)
			rank := 0
			e.Metrics[1].Source.Rank = &rank
			return []nvtelemetry.WorkloadExport{e}
		}, "training point source contradicts export"},
		{"concurrent_runs", func(t *testing.T) []nvtelemetry.WorkloadExport {
			return []nvtelemetry.WorkloadExport{wandbSDKRead(t, raw, wandbSDKRun), wandbSDKRead(t, raw, wandbSDKRun+"-other")}
		}, "concurrent training runs require separate reports"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := nvtelemetry.InspectTrainingSteps(tt.exports(t), "loss", nil)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("InspectTrainingSteps error %v, want %q", err, tt.want)
			}
		})
	}
}

// TestWandBSDKDerivativeChangedLoss checks that a changed loss value is
// visible to the literal loss oracle, not to the step-only consumer.
func TestWandBSDKDerivativeChangedLoss(t *testing.T) {
	b := wandbSDKEdit(t, wandbSDKBytes(t), `"loss":2,`, `"loss":3,`)
	e := wandbSDKRead(t, b, wandbSDKRun)
	want := []string{"4", "3", "1"}
	if got, raw := rawWandBSDKValues(e), wandbSDKLoss(t, e.Raw); !slices.Equal(got, want) || !slices.Equal(raw, want) {
		t.Fatalf("loss: reader %q, independent decode %q; want %q", got, raw, want)
	}
	if reflect.DeepEqual(wandbSDKMetrics(t, e), wandbSDKWant()) {
		t.Fatal("changed loss matches the fixture oracle")
	}
	r, err := nvtelemetry.InspectTrainingSteps([]nvtelemetry.WorkloadExport{e}, "loss", nil)
	if err != nil {
		t.Fatal(err)
	}
	if want := wandbSDKReport(); !reflect.DeepEqual(r, want) {
		t.Fatalf("report\n%+v\nwant\n%+v", r, want)
	}
}
