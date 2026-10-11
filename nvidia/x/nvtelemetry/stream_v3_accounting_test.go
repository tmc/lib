package nvtelemetry

import (
	"bytes"
	"encoding/json"
	"errors"
	"math"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
)

func accountingHeader() Manifest {
	return Manifest{RunID: "accounting", SchemaVersion: SchemaVersionV3, Devices: []DeviceMetadata{{UUID: "GPU-accounting"}}}
}

func accountingSample(n int) Sample {
	v := uint64(0)
	s := Sample{Seq: 1}
	for i := 0; i < n; i++ {
		s.Readings = append(s.Readings, Reading{Name: "kernel", DeviceUUID: "GPU-accounting", Evidence: EvidenceMeasured, Number: &Number{Uint: &v}, Observation: &Observation{Family: "activity"}})
	}
	return s
}

func TestV3PhaseValidationIsAtomic(t *testing.T) {
	var b bytes.Buffer
	w := NewWriter(&b, CollectorConfig{})
	m := accountingHeader()
	if err := w.WriteManifest(&m); err != nil {
		t.Fatal(err)
	}
	before := b.String()
	beforeBytes := w.BytesWritten()
	for _, tc := range []struct {
		name  string
		write func() error
	}{
		{"long phase", func() error { return w.WritePhase(&PhaseMarker{Name: strings.Repeat("x", 4097)}) }},
		{"nil phase", func() error { return w.WritePhase(nil) }},
		{"nil sample", func() error { return w.WriteSample(nil) }},
		{"invalid sample", func() error {
			s := accountingSample(1)
			s.Readings[0].Name = strings.Repeat("x", 4097)
			return w.WriteSample(&s)
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if !errors.Is(tc.write(), ErrMalformedStream) {
				t.Fatal("invalid envelope accepted")
			}
			if b.String() != before || w.BytesWritten() != beforeBytes || w.LostRecords() != 0 || len(w.FamilyLoss()) != 0 || w.recordsCount != 0 {
				t.Fatal("validation changed bytes or counters")
			}
		})
	}
	if err := w.WritePhase(&PhaseMarker{Name: strings.Repeat("x", 4096)}); err != nil {
		t.Fatal(err)
	}
	if _, phases, _, err := ReadStream(&b, 0); err != nil || len(phases) != 1 {
		t.Fatalf("valid phase: %d, %v", len(phases), err)
	}
}

func TestWriterWithoutOutput(t *testing.T) {
	w := NewWriter(nil, CollectorConfig{})
	m := accountingHeader()
	s := accountingSample(1)
	for _, write := range []func() error{func() error { return w.WriteManifest(&m) }, func() error { return w.WriteSample(&s) }, func() error { return w.WritePhase(&PhaseMarker{Name: "ok"}) }} {
		if !errors.Is(write(), ErrMalformedStream) {
			t.Fatal("missing output accepted")
		}
	}
	if w.BytesWritten() != 0 || w.LostRecords() != 0 || len(w.FamilyLoss()) != 0 || w.schema != 0 {
		t.Fatal("missing output changed counters")
	}
}

func TestV3CompletionReadingCounts(t *testing.T) {
	for _, tc := range []struct {
		name              string
		loss              *tracecontract.Loss
		readings          int
		complete, wantErr bool
	}{
		{"two readings", &tracecontract.Loss{Total: 2}, 2, true, false},
		{"retained plus dropped", &tracecontract.Loss{Total: 5, Dropped: 3}, 2, true, false},
		{"entirely dropped", &tracecontract.Loss{Total: 3, Dropped: 3}, 0, true, false},
		{"known empty", &tracecontract.Loss{}, 0, true, false},
		{"zero with retained", &tracecontract.Loss{}, 2, true, true},
		{"under count", &tracecontract.Loss{Total: 1}, 2, true, true},
		{"over count", &tracecontract.Loss{Total: 3}, 2, true, true},
		{"sum overflow", &tracecontract.Loss{Total: 1, Dropped: math.MaxUint64}, 2, true, true},
		{"max subtraction", &tracecontract.Loss{Total: math.MaxUint64, Dropped: math.MaxUint64 - 2}, 2, true, false},
		{"unknown", &tracecontract.Loss{Unknown: true}, 2, true, false},
		{"missing family", nil, 2, true, false},
		{"initial is not final", &tracecontract.Loss{}, 2, false, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := accountingHeader()
			m.LostRecordsByFamily = map[string]tracecontract.Loss{"activity": {}}
			var b bytes.Buffer
			enc := json.NewEncoder(&b)
			if err := enc.Encode(StreamRecord{Manifest: &m}); err != nil {
				t.Fatal(err)
			}
			samples := uint64(0)
			if tc.readings > 0 {
				s := accountingSample(tc.readings)
				if err := enc.Encode(StreamRecord{Sample: &s}); err != nil {
					t.Fatal(err)
				}
				samples = 1
			}
			if tc.complete {
				m.TotalSamples = samples
				m.IsComplete = true
				m.LostRecordsByFamily = nil
				if tc.loss != nil {
					m.LostRecordsByFamily = map[string]tracecontract.Loss{"activity": *tc.loss}
				}
				if err := enc.Encode(StreamRecord{Manifest: &m}); err != nil {
					t.Fatal(err)
				}
			}
			got, _, _, err := ReadStream(&b, 0)
			if tc.wantErr {
				if !errors.Is(err, ErrMalformedStream) {
					t.Fatalf("got %v", err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if got.IsComplete != tc.complete {
				t.Fatal("completion status changed")
			}
		})
	}
}

func TestV3WriterReadingUnitsAndOverflow(t *testing.T) {
	var b bytes.Buffer
	w := NewWriter(&b, CollectorConfig{MaxRecordBudget: 1, BudgetPolicy: BudgetExceededDrop})
	m := accountingHeader()
	if err := w.WriteManifest(&m); err != nil {
		t.Fatal(err)
	}
	s := accountingSample(2)
	if err := w.WriteSample(&s); err != nil {
		t.Fatal(err)
	}
	s.Seq++
	if err := w.WriteSample(&s); err != nil {
		t.Fatal(err)
	}
	if w.FamilyLoss()["activity"] != (tracecontract.Loss{Total: 4, Dropped: 2}) || w.LostRecords() != 1 {
		t.Fatal("sample records and readings conflated")
	}
	before := w.FamilyLoss()
	s.Readings[0].Name = strings.Repeat("x", 4097)
	if !errors.Is(w.WriteSample(&s), ErrMalformedStream) || !reflect.DeepEqual(w.FamilyLoss(), before) {
		t.Fatal("invalid sample counted")
	}
	s = accountingSample(1)
	w.familyLoss["activity"] = tracecontract.Loss{Total: math.MaxUint64, Dropped: math.MaxUint64}
	if err := w.WriteSample(&s); err != nil {
		t.Fatal(err)
	}
	if w.FamilyLoss()["activity"] != (tracecontract.Loss{Total: math.MaxUint64, Dropped: math.MaxUint64, Unknown: true}) {
		t.Fatal("overflow manufactured known-zero loss")
	}
}
