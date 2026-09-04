package inferbench

import (
	"errors"
	"reflect"
	"strings"
	"testing"
)

func TestCollectUsesPreregisteredIntervalOrder(t *testing.T) {
	plan, samples := collectionPlan(t)
	var called []ScheduledSample
	var raw []IntervalReceipt
	got, err := Collect(plan, func(k ScheduledSample) (Sample, error) {
		called = append(called, k)
		return samples[k], nil
	}, func(r IntervalReceipt) error {
		raw = append(raw, r)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(called, got.Order) || !reflect.DeepEqual(got.Order, Schedule(plan.Runs, plan.RandomSeed)) {
		t.Fatal("execution order differs from preregistered schedule")
	}
	if reflect.DeepEqual(got.Order, fixedOrder(plan.Runs)) {
		t.Fatal("seed produced fixed arm order")
	}
	if len(raw) != len(got.Order) {
		t.Fatalf("raw intervals = %d, want %d", len(raw), len(got.Order))
	}
}

func TestCollectRejectsInvalidPlansAndIntervals(t *testing.T) {
	tests := []struct {
		name   string
		change func(*Campaign, map[ScheduledSample]Sample)
		fail   func(ScheduledSample) error
	}{
		{"stale identity", func(c *Campaign, _ map[ScheduledSample]Sample) {
			c.Runs[1].Identity.SourceSHA256 = strings.Repeat("b", 64)
		}, nil},
		{"unequal work", func(c *Campaign, _ map[ScheduledSample]Sample) { c.Runs[1].Workload.GeneratedTokens++ }, nil},
		{"async undermeasurement", func(c *Campaign, _ map[ScheduledSample]Sample) { c.Runs[0].Evidence.DeviceSynchronized = false }, nil},
		{"fixed order", func(c *Campaign, _ map[ScheduledSample]Sample) { c.Order = fixedOrder(c.Runs) }, nil},
		{"skipped arm", func(_ *Campaign, _ map[ScheduledSample]Sample) {}, func(k ScheduledSample) error {
			if k.Arm == "b" {
				return errors.New("skipped")
			}
			return nil
		}},
		{"unequal output", func(_ *Campaign, samples map[ScheduledSample]Sample) {
			for key, sample := range samples {
				if key.Arm != "b" {
					continue
				}
				tokens := append([]int32(nil), sample.GeneratedTokenIDs[0]...)
				sample.GeneratedTokenIDs = [][]int32{tokens}
				sample.GeneratedTokenIDs[0][0]++
				sample.OutputSHA256, _ = HashJSON(sample.GeneratedTokenIDs)
				samples[key] = sample
			}
		}, nil},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			plan, samples := collectionPlan(t)
			test.change(&plan, samples)
			_, err := Collect(plan, func(k ScheduledSample) (Sample, error) {
				if test.fail != nil {
					if err := test.fail(k); err != nil {
						return Sample{}, err
					}
				}
				return samples[k], nil
			}, func(IntervalReceipt) error { return nil })
			if err == nil {
				t.Fatal("accepted invalid campaign")
			}
		})
	}
}

func collectionPlan(t *testing.T) (Campaign, map[ScheduledSample]Sample) {
	t.Helper()
	c := validCampaign()
	samples := make(map[ScheduledSample]Sample)
	for i := range c.Runs {
		for j := range c.Runs[i].Samples {
			s := c.Runs[i].Samples[j]
			samples[ScheduledSample{Arm: c.Runs[i].Arm, ID: s.ID}] = s
			c.Runs[i].Samples[j] = Sample{ID: s.ID}
		}
	}
	c.Order = Schedule(c.Runs, c.RandomSeed)
	return c, samples
}

func fixedOrder(runs []Run) []ScheduledSample {
	var out []ScheduledSample
	for _, r := range runs {
		for _, s := range r.Samples {
			out = append(out, ScheduledSample{Arm: r.Arm, ID: s.ID})
		}
	}
	return out
}
