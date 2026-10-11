package nvtelemetry

import (
	"context"
	"testing"
)

type reusedReadingSampler struct{ readings []Reading }

func (s *reusedReadingSampler) Devices() []DeviceMetadata                 { return []DeviceMetadata{{UUID: "GPU-a"}} }
func (s *reusedReadingSampler) Sample(context.Context) ([]Reading, error) { return s.readings, nil }

func TestCollectorReadingOwnership(t *testing.T) {
	s := &reusedReadingSampler{[]Reading{{Name: "power_milliwatts", Unit: "milliwatts", DeviceUUID: "GPU-a", Evidence: EvidenceMeasured, Value: 1, Source: &ReadingSource{RawValue: "1", Labels: map[string]string{"UUID": "GPU-a"}}}}}
	c := NewCollector("copy", s, NewFakeClock(1, 1), CollectorConfig{}, nil)
	c.SetSampleHook(func(sample Sample) {
		sample.Readings[0].Value = 99
		sample.Readings[0].Source.RawValue = "hook"
		sample.Readings[0].Source.Labels["UUID"] = "hook"
	})
	if e := c.Step(context.Background()); e != nil {
		t.Fatal(e)
	}
	if s.readings[0].HasQueryBounds || s.readings[0].Source.RawValue != "1" {
		t.Fatal("collector changed producer")
	}
	s.readings[0].Value = 2
	s.readings[0].Source.RawValue = "2"
	s.readings[0].Source.Labels["UUID"] = "second"
	if e := c.Step(context.Background()); e != nil {
		t.Fatal(e)
	}
	copy := c.Samples()
	if copy[0].Readings[0].Value != 1 || copy[0].Readings[0].Source.RawValue != "1" || copy[0].Readings[0].Source.Labels["UUID"] != "GPU-a" {
		t.Fatal("retained sample aliased", copy)
	}
	copy[0].Readings[0].Source.RawValue = "caller"
	copy[0].Readings[0].Source.Labels["UUID"] = "caller"
	copy[0].Readings[0].Value = 100
	if c.Samples()[0].Readings[0].Source.RawValue != "1" {
		t.Fatal("samples aliases history")
	}
	done := make(chan struct{})
	go func() {
		defer close(done)
		for i := 0; i < 100; i++ {
			r := c.Samples()
			r[0].Readings[0].Source.Labels["UUID"] = "race"
		}
	}()
	for i := 0; i < 100; i++ {
		c.Samples()
	}
	<-done
}
