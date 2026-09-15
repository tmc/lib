package cuptibench

import (
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cuptirange"
)

// fakeCatalog reports a fixed inventory. It answers the two submetric spellings
// CUPTI uses, so the walk is exercised against both.
type fakeCatalog struct {
	base  map[cupti.MetricType][]string
	subs  map[string][]string
	props map[string]cuptirange.MetricProperties
	err   error
}

func (f *fakeCatalog) BaseMetrics(kind cupti.MetricType, _, _ int) ([]string, error) {
	return f.base[kind], nil
}

func (f *fakeCatalog) SubMetrics(name string, _ cupti.MetricType, _, _ int) ([]string, error) {
	if f.err != nil {
		return nil, f.err
	}
	return f.subs[name], nil
}

func (f *fakeCatalog) Properties(name string, _ int) (cuptirange.MetricProperties, error) {
	prop, ok := f.props[name]
	if !ok {
		return cuptirange.MetricProperties{}, errors.New("no such metric")
	}
	return prop, nil
}

func TestInventoryQualifiesAndClassifies(t *testing.T) {
	f := &fakeCatalog{
		base: map[cupti.MetricType][]string{
			cupti.MetricTypeCounter:    {"sm__ctas_launched", "widget__events"},
			cupti.MetricTypeThroughput: {"dram__throughput"},
		},
		subs: map[string][]string{
			// Fully qualified, as CUPTI reports for some bases.
			"sm__ctas_launched": {"sm__ctas_launched.sum", "sm__ctas_launched.avg"},
			// Bare rollups, the other spelling.
			"widget__events":   {"sum", ".max"},
			"dram__throughput": {"dram__throughput.sum"},
		},
		props: map[string]cuptirange.MetricProperties{
			"sm__ctas_launched.sum": {Dimension: "cta", Type: cupti.MetricTypeCounter},
			"sm__ctas_launched.avg": {Dimension: "cta", Type: cupti.MetricTypeCounter},
			"widget__events.sum":    {Dimension: "event", Type: cupti.MetricTypeCounter},
			"widget__events.max":    {Dimension: "event", Type: cupti.MetricTypeCounter},
			"dram__throughput.sum":  {Dimension: "byte/s", Type: cupti.MetricTypeThroughput},
		},
	}
	got, err := inventory(f)
	if err != nil {
		t.Fatalf("inventory: %v", err)
	}
	want := []struct {
		name     string
		base     string
		additive bool
	}{
		{"dram__throughput.sum", "dram__throughput", false},   // throughput never sums
		{"sm__ctas_launched.avg", "sm__ctas_launched", false}, // wrong rollup
		{"sm__ctas_launched.sum", "sm__ctas_launched", true},  // curated, expected unit
		{"widget__events.max", "widget__events", false},       // bare rollup, wrong kind
		{"widget__events.sum", "widget__events", true},        // discovered counter rollup
	}
	if len(got) != len(want) {
		t.Fatalf("got %d metrics, want %d: %+v", len(got), len(want), got)
	}
	for i, w := range want {
		// Sorted by name, so position is part of the contract.
		if got[i].Name != w.name || got[i].Base != w.base || got[i].Additive != w.additive {
			t.Errorf("metric %d = %q base %q additive %v, want %q base %q additive %v",
				i, got[i].Name, got[i].Base, got[i].Additive, w.name, w.base, w.additive)
		}
	}
}

// A partial inventory must not read as a complete one.
func TestInventoryPropagatesListingFailure(t *testing.T) {
	f := &fakeCatalog{
		base: map[cupti.MetricType][]string{cupti.MetricTypeCounter: {"sm__ctas_launched"}},
		err:  errors.New("bounds exceeded"),
	}
	if _, err := inventory(f); err == nil {
		t.Fatal("inventory hid a failed submetric listing")
	}
}

func TestQualifySpellings(t *testing.T) {
	for _, tc := range []struct{ base, sub, want string }{
		{"a__b", "a__b.sum", "a__b.sum"},
		{"a__b", "sum", "a__b.sum"},
		{"a__b", ".sum", "a__b.sum"},
		{"a__b", "", "a__b"},
	} {
		if got := qualify(tc.base, tc.sub); got != tc.want {
			t.Errorf("qualify(%q, %q) = %q, want %q", tc.base, tc.sub, got, tc.want)
		}
	}
}
