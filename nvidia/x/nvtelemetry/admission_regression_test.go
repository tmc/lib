package nvtelemetry

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"testing"
)

func TestClockCalibrationAdmission(t *testing.T) {
	raw, err := os.ReadFile("testdata/canonical_v2.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	var records []StreamRecord
	for _, line := range bytes.Split(bytes.TrimSpace(raw), []byte("\n")) {
		var r StreamRecord
		if err := json.Unmarshal(line, &r); err != nil {
			t.Fatal(err)
		}
		records = append(records, r)
	}
	for _, name := range []string{"reversed", "presence", "bounds", "from", "until"} {
		t.Run(name, func(t *testing.T) {
			var rs []StreamRecord
			data, _ := json.Marshal(records)
			json.Unmarshal(data, &rs)
			first, last := rs[0].Manifest, rs[len(rs)-1].Manifest
			for _, m := range []*Manifest{first, last} {
				m.HasClockCalibration = true
				m.ClockMappingValidFrom = 10
				m.ClockMappingValidUntil = 20
				m.ClockCalibrationBoundsNanos = 1
			}
			switch name {
			case "reversed":
				first.ClockMappingValidUntil = 9
			case "presence":
				last.HasClockCalibration = false
			case "bounds":
				last.ClockCalibrationBoundsNanos = 2
			case "from":
				last.ClockMappingValidFrom = 11
			case "until":
				last.ClockMappingValidUntil = 21
			}
			var out bytes.Buffer
			for _, r := range rs {
				json.NewEncoder(&out).Encode(r)
			}
			if _, _, _, err := ReadStream(&out, 0); err == nil {
				t.Fatal("contradictory calibration admitted")
			}
		})
	}
}

type admissionClock struct{ values []int64 }

func (c *admissionClock) NowMonotonic() int64 { v := c.values[0]; c.values = c.values[1:]; return v }
func (c *admissionClock) NowUnix() int64      { return 0 }

type admissionSampler struct{}

func (admissionSampler) Devices() []DeviceMetadata                 { return nil }
func (admissionSampler) Sample(context.Context) ([]Reading, error) { return nil, nil }
func TestCollectorClockDiscontinuity(t *testing.T) {
	for _, values := range [][]int64{{0, 0, -1, -1}, {0, -1, 0, 0}} {
		c := &Collector{clock: &admissionClock{values: values}, sampler: admissionSampler{}}
		if err := c.collectOnce(context.Background()); err != nil {
			t.Fatal(err)
		}
		if err := c.collectOnce(context.Background()); err != nil {
			t.Fatal(err)
		}
		if !c.samples[0].ClockDiscontinuity && !c.samples[1].ClockDiscontinuity {
			t.Fatal("clock reversal unmarked")
		}
	}
}
