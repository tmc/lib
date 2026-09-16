//go:build linux

package cuptibench_test

import (
	"os"
	"sort"
	"testing"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cuptibench"
)

// TestReachableCoversDefaultSet checks the inventory against the chip rather
// than against a recorded list, which would only restate what was recorded.
// The load-bearing claim is that every metric Default names is one the device
// actually offers: the default set is collected by every benchmark that does
// not override it, so a name that drifted out of the chip's vocabulary must
// fail here and not at collection time.
func TestReachableCoversDefaultSet(t *testing.T) {
	if os.Getenv("CUPTIBENCH_PHYSICAL_TEST") != "1" {
		t.Skip("unqualified: set CUPTIBENCH_PHYSICAL_TEST=1 on an exclusively owned CUDA device")
	}
	dev, err := cuda.NewDevice(0)
	if err != nil {
		t.Fatalf("open device: %v", err)
	}
	defer func() {
		if err := dev.Close(); err != nil {
			t.Error(err)
		}
	}()
	all, err := cuptibench.Reachable(dev)
	if err != nil {
		t.Fatalf("reachable: %v", err)
	}
	if len(all) == 0 {
		t.Fatal("reachable returned no metrics")
	}
	index := make(map[string]cuptibench.MetricInfo, len(all))
	kinds := make(map[cupti.MetricType]int)
	additive := 0
	for i, m := range all {
		if m.Name == "" || m.Base == "" {
			t.Fatalf("metric %d: empty name %q or base %q", i, m.Name, m.Base)
		}
		if _, dup := index[m.Name]; dup {
			t.Fatalf("metric %q listed twice", m.Name)
		}
		if i > 0 && all[i-1].Name >= m.Name {
			t.Fatalf("not sorted: %q before %q", all[i-1].Name, m.Name)
		}
		index[m.Name] = m
		kinds[m.Type]++
		if m.Additive {
			additive++
		}
	}
	for _, name := range cuptibench.Default() {
		m, ok := index[name]
		if !ok {
			t.Errorf("default metric %q is not reachable on this chip", name)
			continue
		}
		if !m.Additive {
			t.Errorf("default metric %q is not additive (unit %q, type %v)", name, m.Unit, m.Type)
		}
	}
	types := make([]int, 0, len(kinds))
	for k := range kinds {
		types = append(types, int(k))
	}
	sort.Ints(types)
	t.Logf("reachable: %d metrics, %d additive", len(all), additive)
	for _, k := range types {
		t.Logf("  type %d: %d", k, kinds[cupti.MetricType(k)])
	}
	for _, name := range cuptibench.Default() {
		if m, ok := index[name]; ok {
			t.Logf("  default %s: base=%s unit=%q type=%v", m.Name, m.Base, m.Unit, m.Type)
		}
	}
}
