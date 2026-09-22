package cudacensus

import (
	"errors"
	"math"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/dcgm"
)

func dcgmFixture(t *testing.T, want []Counter) *DCGMReader {
	t.Helper()
	s, err := NewDCGMReader(1, 0, want, time.Second)
	if err != nil {
		t.Fatal(err)
	}
	s.now = func() time.Time { return time.Unix(100, 0) }
	return s
}

func dcgmValue(field uint16, value float64) dcgm.FieldValueV1 {
	v := dcgm.FieldValueV1{Version: dcgm.FieldValueVersion1, FieldId: field, FieldType: dcgm.FieldTypeDouble, Ts: 99_500_000}
	v.Value[0] = math.Float64bits(value)
	return v
}

func TestDCGMReaderFieldsAndFractions(t *testing.T) {
	want := []Counter{FP32Active, DMMATensorActive}
	s := dcgmFixture(t, want)
	want[0] = FP16Active
	fields := s.Fields()
	if len(fields) != 1 || fields[0] != dcgm.FieldProfPipeFP32Active {
		t.Fatalf("fields=%v", fields)
	}
	fields[0] = 0
	s.query = func(fields *uint16, count uint32, values *dcgm.FieldValueV1) error {
		if count != 1 || *fields != uint16(dcgm.FieldProfPipeFP32Active) || values.Version != dcgm.FieldValueVersion1 {
			t.Fatal("invalid query arguments")
		}
		*values = dcgmValue(*fields, 0.25)
		return nil
	}
	dst := make([]Reading, 2)
	sample, err := s.Read(dst)
	if err != nil || sample.IntervalNs != -1 || !dst[0].Available || dst[0].Value != 0.25 || dst[1].Available || dst[1].Reason == "" {
		t.Fatalf("sample=%+v error=%v", sample, err)
	}
	if !sample.SampleTimes[0].Equal(time.UnixMicro(99_500_000)) || !sample.SampleTimes[1].IsZero() {
		t.Fatalf("times=%v", sample.SampleTimes)
	}
	if allocs := testing.AllocsPerRun(100, func() { _, _ = s.Read(dst) }); allocs != 0 {
		t.Fatalf("steady read allocations=%g", allocs)
	}
}

func TestDCGMReaderRefusesInvalidFields(t *testing.T) {
	for _, name := range []string{"unwritten", "no_payload", "wrong_id", "unsupported", "blank", "integer", "nan", "infinity", "negative", "percent", "no_time", "stale", "future"} {
		t.Run(name, func(t *testing.T) {
			s := dcgmFixture(t, []Counter{SMActive})
			s.query = func(fields *uint16, count uint32, values *dcgm.FieldValueV1) error {
				if name == "unwritten" {
					return nil
				}
				if name == "no_payload" {
					values.FieldId, values.FieldType, values.Status, values.Ts = *fields, dcgm.FieldTypeDouble, 0, 99_500_000
					return nil
				}
				*values = dcgmValue(*fields, 0.5)
				switch name {
				case "wrong_id":
					values.FieldId++
				case "unsupported":
					values.Status = int32(dcgm.ErrNotSupported)
				case "blank":
					values.Value[0] = math.Float64bits(dcgm.Fp64Blank)
				case "integer":
					values.FieldType = dcgm.FieldTypeInt64
				case "nan":
					values.Value[0] = math.Float64bits(math.NaN())
				case "infinity":
					values.Value[0] = math.Float64bits(math.Inf(1))
				case "negative":
					values.Value[0] = math.Float64bits(-0.1)
				case "percent":
					values.Value[0] = math.Float64bits(50)
				case "no_time":
					values.Ts = 0
				case "stale":
					values.Ts = 98_999_999
				case "future":
					values.Ts = 100_000_001
				}
				return nil
			}
			s.values[0] = dcgmValue(s.fields[0], 0.75)
			dst := []Reading{{Counter: SMActive, Value: 0.75, Available: true}}
			if _, err := s.Read(dst); err != nil || dst[0].Available || dst[0].Value != 0 || dst[0].Reason == "" {
				t.Fatalf("reading=%+v err=%v", dst[0], err)
			}
		})
	}
}

func TestDCGMReaderFailureAndBoundaries(t *testing.T) {
	s := dcgmFixture(t, []Counter{SMActive})
	dst := []Reading{{Counter: SMActive, Value: 1, Available: true}}
	failure := errors.New("query failed")
	s.query = func(*uint16, uint32, *dcgm.FieldValueV1) error { return failure }
	if _, err := s.Read(dst); !errors.Is(err, failure) || dst[0].Available || dst[0].Value != 0 {
		t.Fatalf("query failure: %v %v", dst, err)
	}
	for _, value := range []float64{0, 1} {
		s.query = func(fields *uint16, count uint32, values *dcgm.FieldValueV1) error {
			*values = dcgmValue(*fields, value)
			values.Ts = 99_000_000 // Exactly maxAge is still usable.
			return nil
		}
		if _, err := s.Read(dst); err != nil || !dst[0].Available || dst[0].Value != value {
			t.Fatalf("boundary: %v %v", dst, err)
		}
	}
	if _, err := s.Read(nil); err == nil {
		t.Fatal("accepted short destination")
	}
	var zero DCGMReader
	if _, err := zero.Read(dst); err == nil {
		t.Fatal("accepted zero reader")
	}
	for _, want := range [][]Counter{nil, {SMActive, SMActive}, {Counter("unknown")}} {
		if _, err := NewDCGMReader(1, 0, want, time.Second); err == nil {
			t.Fatalf("accepted request %v", want)
		}
	}
}
