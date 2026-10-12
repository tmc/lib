package nvtelemetry_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func v3Header() nvtelemetry.Manifest {
	return nvtelemetry.Manifest{RunID: "v3-protocol", SchemaVersion: nvtelemetry.SchemaVersionV3, Devices: []nvtelemetry.DeviceMetadata{{UUID: "GPU-v3"}}}
}
func v3NumberReading(n *nvtelemetry.Number) nvtelemetry.Reading {
	return nvtelemetry.Reading{Name: "scalar", Unit: "count", DeviceUUID: "GPU-v3", Evidence: nvtelemetry.EvidenceMeasured, Number: n}
}

func TestV3ExactNumberRoundtrip(t *testing.T) {
	lo, hi, u, zero := int64(math.MinInt64), int64(math.MaxInt64), uint64(math.MaxUint64), float64(0)
	cases := []struct {
		name string
		n    *nvtelemetry.Number
	}{{"signed-min", &nvtelemetry.Number{Int: &lo}}, {"signed-max", &nvtelemetry.Number{Int: &hi}}, {"unsigned-max", &nvtelemetry.Number{Uint: &u}}, {"measured-zero", &nvtelemetry.Number{Float: &zero}}}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var b bytes.Buffer
			w := nvtelemetry.NewWriter(&b, nvtelemetry.CollectorConfig{})
			m := v3Header()
			if err := w.WriteManifest(&m); err != nil {
				t.Fatal(err)
			}
			s := nvtelemetry.Sample{Seq: 1, Readings: []nvtelemetry.Reading{v3NumberReading(tc.n)}}
			if err := w.WriteSample(&s); err != nil {
				t.Fatal(err)
			}
			m.TotalSamples = 1
			m.IsComplete = true
			if err := w.WriteManifest(&m); err != nil {
				t.Fatal(err)
			}
			raw := append([]byte(nil), b.Bytes()...)
			got, _, samples, err := nvtelemetry.ReadStreamStrict(&b, 0)
			if err != nil {
				t.Fatal(err)
			}
			if got.SchemaVersion != 3 || !reflect.DeepEqual(samples[0].Readings[0].Number, tc.n) {
				t.Fatalf("precision/presence lost: %s", raw)
			}
			if tc.name == "measured-zero" && !bytes.Contains(raw, []byte(`"float":0`)) {
				t.Fatalf("zero omitted: %s", raw)
			}
		})
	}
}

func TestV3UnavailableAndMalformedValues(t *testing.T) {
	z := float64(0)
	nan := math.NaN()
	u := uint64(0)
	cases := []struct {
		name string
		r    nvtelemetry.Reading
		bad  bool
	}{
		{"absent-unavailable", nvtelemetry.Reading{DeviceUUID: "GPU-v3", Evidence: nvtelemetry.EvidenceUnavailable, UnavailableReason: "denied"}, false},
		{"absent-measured", v3NumberReading(nil), true},
		{"two-arms", v3NumberReading(&nvtelemetry.Number{Float: &z, Uint: &u}), true},
		{"no-arm", v3NumberReading(&nvtelemetry.Number{}), true},
		{"nonfinite", v3NumberReading(&nvtelemetry.Number{Float: &nan}), true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var b bytes.Buffer
			w := nvtelemetry.NewWriter(&b, nvtelemetry.CollectorConfig{})
			m := v3Header()
			if err := w.WriteManifest(&m); err != nil {
				t.Fatal(err)
			}
			err := w.WriteSample(&nvtelemetry.Sample{Seq: 1, Readings: []nvtelemetry.Reading{tc.r}})
			if tc.bad {
				if !errors.Is(err, nvtelemetry.ErrMalformedStream) {
					t.Fatalf("got %v", err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			_, _, samples, err := nvtelemetry.ReadStream(&b, 0)
			if err != nil || samples[0].Readings[0].Number != nil {
				t.Fatalf("absent became zero: %v %+v", err, samples)
			}
		})
	}
}

func TestV3StrictMigration(t *testing.T) {
	cases := []struct {
		name, raw string
		want      error
	}{
		{"unknown-critical", `{"manifest":{"schema_version":3,"devices":[{"uuid":"GPU-v3"}],"new_loss_contract":{}}}`, nvtelemetry.ErrMalformedStream},
		{"legacy-loss-extension", `{"manifest":{"schema_version":2,"devices":[{"uuid":"GPU-v3"}],"lost_records_by_family":{"activity":{"dropped":0,"total":0}}}}`, nvtelemetry.ErrMalformedStream},
		{"legacy-null-loss", `{"manifest":{"schema_version":2,"devices":[{"uuid":"GPU-v3"}],"lost_records_by_family":null}}`, nvtelemetry.ErrMalformedStream},
		{"V3-legacy-zero", `{"manifest":{"schema_version":3,"devices":[{"uuid":"GPU-v3"}]}}` + "\n" + `{"sample":{"seq":1,"readings":[{"device_uuid":"GPU-v3","evidence":"measured","value":0,"number":{"float":0}}]}}`, nvtelemetry.ErrMalformedStream},
		{"future", `{"manifest":{"schema_version":5}}`, nvtelemetry.ErrUnsupportedSchema},
		{"duplicate", `{"manifest":{"schema_version":3,"schema_version":3}}`, nvtelemetry.ErrMalformedStream},
		{"overflow", `{"manifest":{"schema_version":3,"devices":[{"uuid":"GPU-v3"}]}}` + "\n" + `{"sample":{"seq":1,"readings":[{"device_uuid":"GPU-v3","evidence":"measured","number":{"int":9223372036854775808}}]}}`, nvtelemetry.ErrMalformedStream},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, _, _, err := nvtelemetry.ReadStream(strings.NewReader(tc.raw), 0)
			if !errors.Is(err, tc.want) {
				t.Fatalf("got %v want %v", err, tc.want)
			}
		})
	}
	// Frozen old-reader behavior: the old boundary stops at the version header.
	oldReader := func(raw []byte) error {
		var rec nvtelemetry.StreamRecord
		if err := json.Unmarshal(bytes.Split(raw, []byte("\n"))[0], &rec); err != nil {
			return err
		}
		if rec.Manifest.SchemaVersion > nvtelemetry.SchemaVersionV2 {
			return nvtelemetry.ErrUnsupportedSchema
		}
		return nil
	}
	var b bytes.Buffer
	m := v3Header()
	for _, version := range []int{nvtelemetry.SchemaVersionV3, nvtelemetry.SchemaVersionV4} {
		m.SchemaVersion = version
		b.Reset()
		if err := nvtelemetry.NewWriter(&b, nvtelemetry.CollectorConfig{}).WriteManifest(&m); err != nil {
			t.Fatal(err)
		}
		if !errors.Is(oldReader(b.Bytes()), nvtelemetry.ErrUnsupportedSchema) {
			t.Fatalf("old reader accepted V%d", version)
		}
	}
	for _, version := range []int{1, 2} {
		m.SchemaVersion = version
		b.Reset()
		if err := nvtelemetry.NewWriter(&b, nvtelemetry.CollectorConfig{}).WriteManifest(&m); err != nil {
			t.Fatal(err)
		}
		if _, _, _, err := nvtelemetry.ReadStream(&b, 0); err != nil {
			t.Fatalf("legacy %d: %v", version, err)
		}
	}
}

type v3ShortWriter struct{}

func (v3ShortWriter) Write(p []byte) (int, error) { return len(p) - 1, nil }

func TestV3BoundsShortWriteAndFamilyLoss(t *testing.T) {
	m := v3Header()
	w := nvtelemetry.NewWriter(v3ShortWriter{}, nvtelemetry.CollectorConfig{})
	if !errors.Is(w.WriteManifest(&m), io.ErrShortWrite) {
		t.Fatal("short write accepted")
	}
	var b bytes.Buffer
	w = nvtelemetry.NewWriter(&b, nvtelemetry.CollectorConfig{MaxRecordBudget: 1, BudgetPolicy: nvtelemetry.BudgetExceededDrop})
	if err := w.WriteManifest(&m); err != nil {
		t.Fatal(err)
	}
	v := uint64(0)
	r := v3NumberReading(&nvtelemetry.Number{Uint: &v})
	r.Observation = &nvtelemetry.Observation{Family: "activity"}
	s := nvtelemetry.Sample{Seq: 1, Readings: []nvtelemetry.Reading{r}}
	if err := w.WriteSample(&s); err != nil {
		t.Fatal(err)
	}
	s.Seq++
	if err := w.WriteSample(&s); err != nil {
		t.Fatal(err)
	}
	loss := w.FamilyLoss()
	if loss["activity"] != (tracecontract.Loss{Dropped: 1, Total: 2}) {
		t.Fatalf("loss erased: %+v", loss)
	}
	loss["activity"] = tracecontract.Loss{}
	if w.FamilyLoss()["activity"].Dropped != 1 {
		t.Fatal("loss alias")
	}
	if _, ok := w.FamilyLoss()["unreported"]; ok {
		t.Fatal("unknown family became zero loss")
	}
	if _, _, _, err := nvtelemetry.ReadStream(bytes.NewReader(b.Bytes()), int64(b.Len()-1)); !errors.Is(err, nvtelemetry.ErrBudgetExceeded) {
		t.Fatalf("byte bound: %v", err)
	}
	m.RunID = strings.Repeat("x", 4097)
	if !errors.Is(w.WriteManifest(&m), nvtelemetry.ErrMalformedStream) {
		t.Fatal("string bound ignored")
	}
}

func TestCollectorV3ExactConversion(t *testing.T) {
	u := uint64(math.MaxUint64)
	v := float64(0)
	sampler := &mockSampler{devices: []nvtelemetry.DeviceMetadata{{UUID: "GPU-v3"}}, batches: [][]nvtelemetry.Reading{{{Name: "counter", DeviceUUID: "GPU-v3", IsInt: true, IntValue: u, Evidence: nvtelemetry.EvidenceMeasured, Observation: &nvtelemetry.Observation{Family: "nvml"}}, {Name: "zero", DeviceUUID: "GPU-v3", Value: v, Evidence: nvtelemetry.EvidenceMeasured, Observation: &nvtelemetry.Observation{Family: "nvml"}}}}}
	var b bytes.Buffer
	cfg := nvtelemetry.CollectorConfig{SchemaVersion: 3}
	c := nvtelemetry.NewCollector("v3", sampler, nvtelemetry.NewFakeClock(1, 1), cfg, nvtelemetry.NewWriter(&b, cfg))
	if err := c.InitStream(); err != nil {
		t.Fatal(err)
	}
	if err := c.Step(context.Background()); err != nil {
		t.Fatal(err)
	}
	if _, err := c.Stop(); err != nil {
		t.Fatal(err)
	}
	m, _, samples, err := nvtelemetry.ReadStreamStrict(&b, 0)
	if err != nil {
		t.Fatal(err)
	}
	if *samples[0].Readings[0].Number.Uint != u || *samples[0].Readings[1].Number.Float != 0 || m.LostRecordsByFamily["nvml"].Total != 2 {
		t.Fatalf("collector V3 %+v %+v", m, samples)
	}
	if sampler.batches[0][0].Number != nil {
		t.Fatal("sampler mutated")
	}
}

func ExampleNumber() {
	n := int64(-9223372036854775808)
	b, _ := json.Marshal(nvtelemetry.Number{Int: &n})
	fmt.Println(string(b)) // Output: {"int":-9223372036854775808}
}
func ExampleHistogram() {
	b, _ := json.Marshal(nvtelemetry.Histogram{Bounds: []string{"1", "+Inf"}, Counts: []uint64{0, 2}})
	fmt.Println(string(b)) // Output: {"bounds":["1","+Inf"],"counts":[0,2]}
}
func ExampleObserver() {
	o := nvtelemetry.Observer{ID: "capture-1", Mode: "activity", Pass: "single"}
	fmt.Println(o.ID, o.Mode, o.Pass) // Output: capture-1 activity single
}
func ExampleObservation() {
	o := nvtelemetry.Observation{Family: "nvml", SourceClass: "synthetic"}
	fmt.Println(o.Family, o.SourceClass) // Output: nvml synthetic
}
func ExampleWriter_FamilyLoss() {
	w := nvtelemetry.NewWriter(io.Discard, nvtelemetry.CollectorConfig{})
	_, ok := w.FamilyLoss()["activity"]
	fmt.Println(ok) // Output: false
}
