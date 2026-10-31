package nvtelemetry_test

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"io"
	"net/http"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

// rayUnitsInput is a constructed Prometheus text 0.0.4 control, not a Ray
// capture. Names, HELP text and labels follow Ray 2.44.0 reporter_agent.py;
// the values and timestamps are chosen. Ray's own exporter formats integer
// MiB as Python floats ("1234.0"); the exponent spellings test the protocol.
const rayUnitsInput = `# HELP ray_node_gram_used Total GPU RAM usage on a ray node
# TYPE ray_node_gram_used gauge
ray_node_gram_used{GpuDeviceName="NVIDIA A100-SXM4-80GB",GpuIndex="0",IsHeadNode="true",SessionName="session_control",Version="2.44.0",ip="10.0.0.1"} 1234.0 1700000000000
ray_node_gram_used{GpuDeviceName="NVIDIA A100-SXM4-80GB",GpuIndex="1",IsHeadNode="true",SessionName="session_control",Version="2.44.0",ip="10.0.0.1"} 8.1559e+04 1700000000000
# HELP ray_node_gram_available Total GPU RAM available on a ray node
# TYPE ray_node_gram_available gauge
ray_node_gram_available{GpuDeviceName="NVIDIA A100-SXM4-80GB",GpuIndex="0",IsHeadNode="true",SessionName="session_control",Version="2.44.0",ip="10.0.0.1"} 80686.0 1700000000000
ray_node_gram_available{GpuDeviceName="NVIDIA A100-SXM4-80GB",GpuIndex="1",IsHeadNode="true",SessionName="session_control",Version="2.44.0",ip="10.0.0.1"} 3.61E2 1700000000000
# HELP ray_node_gpus_utilization Total GPUs usage on a ray node
# TYPE ray_node_gpus_utilization gauge
ray_node_gpus_utilization{GpuDeviceName="NVIDIA A100-SXM4-80GB",GpuIndex="0",IsHeadNode="true",SessionName="session_control",Version="2.44.0",ip="10.0.0.1"} 37.5 1700000000000
# TYPE ray_resources gauge
ray_resources{Name="GPU",State="USED",instance="node1"} 0.5 1700000000000
# HELP ray_node_mem_used Memory usage on a ray node
# TYPE ray_node_mem_used gauge
ray_node_mem_used{IsHeadNode="true",SessionName="session_control",Version="2.44.0",ip="10.0.0.1"} 1073741824 1700000000000
`

var raySource = nvtelemetry.WorkloadSource{Provider: "ray", Version: "2.44.0", RunID: "ray-units"}

func rayLabels(gpu string) map[string]string {
	return map[string]string{"GpuDeviceName": "NVIDIA A100-SXM4-80GB", "GpuIndex": gpu, "IsHeadNode": "true", "SessionName": "session_control", "Version": "2.44.0", "ip": "10.0.0.1"}
}

var rayUnitsWant = []struct {
	name                        string
	metric, semantic, unit, raw string
	labels                      map[string]string
}{
	{"gram_used_decimal", "ray_node_gram_used", "node_gpu_memory_used", "MiB", "1234.0", rayLabels("0")},
	{"gram_used_exponent", "ray_node_gram_used", "node_gpu_memory_used", "MiB", "8.1559e+04", rayLabels("1")},
	{"gram_available_decimal", "ray_node_gram_available", "node_gpu_memory_available", "MiB", "80686.0", rayLabels("0")},
	{"gram_available_exponent", "ray_node_gram_available", "node_gpu_memory_available", "MiB", "3.61E2", rayLabels("1")},
	{"gpu_utilization_control", "ray_node_gpus_utilization", "node_gpu_utilization", "percent", "37.5", rayLabels("0")},
	{"logical_assignment_control", "ray_resources", "logical_resource_assignment", "resources", "0.5", map[string]string{"Name": "GPU", "State": "USED", "instance": "node1"}},
}

func readRay(t *testing.T, raw string) nvtelemetry.WorkloadExport {
	t.Helper()
	e, err := nvtelemetry.ReadServingExposition(context.Background(), strings.NewReader(raw), raySource, nvtelemetry.WorkloadLimits{})
	if err != nil {
		t.Fatal(err)
	}
	return e
}

// checkRayGauge checks that p is the expected Ray gauge. The unit is
// checked first: the pre-repair reader labelled integer MiB as "bytes".
func checkRayGauge(t *testing.T, p nvtelemetry.WorkloadMetric, metric, semantic, unit, raw string, labels map[string]string) {
	t.Helper()
	if p.Unit != unit {
		t.Errorf("%s Unit = %q, want %q", p.Name, p.Unit, unit)
	}
	if p.Name != metric || p.Semantic != semantic || p.Kind != "gauge" || p.Temporality != "instantaneous" {
		t.Errorf("%s identity = %s %s %s, want %s %s gauge instantaneous", p.Name, p.Semantic, p.Kind, p.Temporality, metric, semantic)
	}
	if p.RawValue != raw {
		t.Errorf("%s RawValue = %q, want unchanged %q", p.Name, p.RawValue, raw)
	}
	if !reflect.DeepEqual(p.Labels, labels) {
		t.Errorf("%s Labels = %v, want %v", p.Name, p.Labels, labels)
	}
}

func TestReadServingExpositionRayMemoryUnits(t *testing.T) {
	e := readRay(t, rayUnitsInput)
	h := sha256.Sum256([]byte(rayUnitsInput))
	if string(e.Raw) != rayUnitsInput || e.RawSHA256 != hex.EncodeToString(h[:]) {
		t.Fatal("raw bytes or hash changed")
	}
	if e.Source.Provider != "ray" || e.Source.Version != "2.44.0" || e.Source.RunID != "ray-units" {
		t.Fatalf("Source = %+v", e.Source)
	}
	if !reflect.DeepEqual(e.UnmappedNames, []string{"ray_node_mem_used"}) {
		t.Fatalf("UnmappedNames = %v, want [ray_node_mem_used]", e.UnmappedNames)
	}
	if len(e.Metrics) != len(rayUnitsWant) {
		t.Fatalf("got %d metrics, want %d", len(e.Metrics), len(rayUnitsWant))
	}
	for i, tt := range rayUnitsWant {
		t.Run(tt.name, func(t *testing.T) {
			p := e.Metrics[i]
			checkRayGauge(t, p, tt.metric, tt.semantic, tt.unit, tt.raw, tt.labels)
			if !reflect.DeepEqual(p.Source, e.Source) {
				t.Errorf("Source = %+v, want %+v", p.Source, e.Source)
			}
			if !p.HasTime || p.TimeNanos != 1700000000000000000 || p.RawTime != "1700000000000" || p.RawTimeUnit != "milliseconds" || p.TimeSource != "producer" {
				t.Errorf("time = %v %d %q %q %q, want producer milliseconds", p.HasTime, p.TimeNanos, p.RawTime, p.RawTimeUnit, p.TimeSource)
			}
		})
	}
}

// cancelReader cancels its context after the first Read.
type cancelReader struct {
	r      io.Reader
	cancel context.CancelFunc
}

func (c cancelReader) Read(p []byte) (int, error) {
	n, err := c.r.Read(p[:min(len(p), 16)])
	c.cancel()
	return n, err
}

func TestReadServingExpositionRayMemoryRefusals(t *testing.T) {
	gram := "ray_node_gram_used{GpuIndex=\"0\"} 1234.0\n"
	tests := []struct {
		name   string
		source nvtelemetry.WorkloadSource
		raw    string
		limits nvtelemetry.WorkloadLimits
	}{
		{"rolling_version", nvtelemetry.WorkloadSource{Provider: "ray", Version: "2.45.0", RunID: "r"}, rayUnitsInput, nvtelemetry.WorkloadLimits{}},
		{"unknown_provider", nvtelemetry.WorkloadSource{Provider: "rayx", Version: "2.44.0", RunID: "r"}, rayUnitsInput, nvtelemetry.WorkloadLimits{}},
		{"counter_type", raySource, "# TYPE ray_node_gram_used counter\n" + gram, nvtelemetry.WorkloadLimits{}},
		{"untyped", raySource, gram, nvtelemetry.WorkloadLimits{}},
		{"negative", raySource, "# TYPE ray_node_gram_available gauge\nray_node_gram_available{GpuIndex=\"0\"} -1\n", nvtelemetry.WorkloadLimits{}},
		{"byte_limit", raySource, rayUnitsInput, nvtelemetry.WorkloadLimits{MaxBytes: int64(len(rayUnitsInput) - 1)}},
		{"metric_limit", raySource, rayUnitsInput, nvtelemetry.WorkloadLimits{MaxMetrics: 6}},
		{"string_limit", raySource, rayUnitsInput, nvtelemetry.WorkloadLimits{MaxStringBytes: 16}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			e, err := nvtelemetry.ReadServingExposition(context.Background(), strings.NewReader(tt.raw), tt.source, tt.limits)
			if err == nil {
				t.Fatalf("admitted %d metrics", len(e.Metrics))
			}
		})
	}
	t.Run("canceled", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		if _, err := nvtelemetry.ReadServingExposition(ctx, strings.NewReader(rayUnitsInput), raySource, nvtelemetry.WorkloadLimits{}); !errors.Is(err, context.Canceled) {
			t.Fatalf("err = %v, want context.Canceled", err)
		}
	})
	t.Run("canceled_between_reads", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		r := cancelReader{strings.NewReader(rayUnitsInput), cancel}
		if _, err := nvtelemetry.ReadServingExposition(ctx, r, raySource, nvtelemetry.WorkloadLimits{}); !errors.Is(err, context.Canceled) {
			t.Fatalf("err = %v, want context.Canceled", err)
		}
	})
}

type rayRoundTrip func(*http.Request) (*http.Response, error)

func (f rayRoundTrip) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

// TestRayMemoryUnitsConsumers checks that the existing public consumers of a
// Ray export keep the MiB unit and raw values and do not turn gauges into
// windows or OTLP sums.
func TestRayMemoryUnitsConsumers(t *testing.T) {
	gram := func(t *testing.T, e nvtelemetry.WorkloadExport) {
		t.Helper()
		for i, tt := range rayUnitsWant[:4] {
			checkRayGauge(t, e.Metrics[i], tt.metric, tt.semantic, tt.unit, tt.raw, tt.labels)
		}
	}
	untimed := strings.ReplaceAll(rayUnitsInput, " 1700000000000\n", "\n")
	stamp := func(t *testing.T, nanos int64) nvtelemetry.WorkloadExport {
		t.Helper()
		e, err := nvtelemetry.StampWorkloadSnapshot(readRay(t, untimed), nanos, "replay", "epoch1")
		if err != nil {
			t.Fatal(err)
		}
		return e
	}

	t.Run("fetch", func(t *testing.T) {
		client := &http.Client{Transport: rayRoundTrip(func(*http.Request) (*http.Response, error) {
			return &http.Response{StatusCode: 200, Body: io.NopCloser(strings.NewReader(rayUnitsInput))}, nil
		})}
		req, err := http.NewRequest("GET", "https://example.invalid/metrics", nil)
		if err != nil {
			t.Fatal(err)
		}
		result, err := nvtelemetry.FetchWorkload(context.Background(), client, req, raySource, nvtelemetry.WorkloadLimits{})
		if err != nil {
			t.Fatal(err)
		}
		if string(result.Export.Raw) != rayUnitsInput {
			t.Fatal("raw bytes changed")
		}
		gram(t, result.Export)
	})
	t.Run("stamp", func(t *testing.T) {
		e := stamp(t, 100)
		gram(t, e)
		if p := e.Metrics[0]; p.TimeSource != "collector_observation" || p.TimeNanos != 100 {
			t.Errorf("stamped time = %q %d", p.TimeSource, p.TimeNanos)
		}
		if _, err := nvtelemetry.StampWorkloadSnapshot(readRay(t, rayUnitsInput), 100, "replay", "epoch1"); err == nil {
			t.Error("replaced native Ray timestamp")
		}
	})
	t.Run("window", func(t *testing.T) {
		a, b := stamp(t, 0), stamp(t, 100)
		if d, err := nvtelemetry.CounterWorkloadWindow(a, b, 0, 100, "replay", "epoch1"); err == nil {
			t.Errorf("counter window over Ray gauges = %v", d)
		}
		if d, err := nvtelemetry.HistogramWorkloadWindow(a, b, 0, 100, "replay", "epoch1"); err == nil {
			t.Errorf("histogram window over Ray gauges = %v", d)
		}
	})
	t.Run("otlp", func(t *testing.T) {
		a, b := stamp(t, 0), stamp(t, 100)
		var out bytes.Buffer
		o := nvtelemetry.WorkloadOTLPOptions{StartNanos: 0, EndNanos: 100, ClockDomain: "replay", ClockEpochID: "epoch1", HasUnixMapping: true, StartUnixNanos: 1, EndUnixNanos: 101, MappingProvenance: "control"}
		if err := nvtelemetry.WriteWorkloadOTLP(&out, a, b, o); err == nil || out.Len() != 0 {
			t.Errorf("WriteWorkloadOTLP over Ray gauges = %v, %d bytes", err, out.Len())
		}
	})
	t.Run("join", func(t *testing.T) {
		if nvtelemetry.ExactWorkloadJoin(readRay(t, rayUnitsInput)) == nil {
			t.Error("Ray aggregate became exact records")
		}
	})
	t.Run("old_bytes_result", func(t *testing.T) {
		// A result typed by the old reader carried Unit "bytes" over the
		// same integer MiB. No validator migrates it; rereading its
		// retained Raw yields MiB with the same hash and spellings.
		old := readRay(t, rayUnitsInput)
		old.Metrics = append([]nvtelemetry.WorkloadMetric(nil), old.Metrics...)
		for i := range old.Metrics[:4] {
			old.Metrics[i].Unit = "bytes"
		}
		e, err := nvtelemetry.ReadServingExposition(context.Background(), bytes.NewReader(old.Raw), old.Source, nvtelemetry.WorkloadLimits{})
		if err != nil {
			t.Fatal(err)
		}
		if e.RawSHA256 != old.RawSHA256 {
			t.Fatal("reread changed raw hash")
		}
		gram(t, e)
	})
}
