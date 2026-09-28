//go:build linux

package cuptibench_test

import (
	"encoding/json"
	"os"
	"strconv"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cuptibench"
	"github.com/tmc/lib/nvidia/x/cuptirange"
	"github.com/tmc/lib/nvidia/x/cuptisass"
)

// TestPhysicalKernelReplayHostBudget requires a multi-pass configuration to
// complete in one host Start/Stop pair. Internal kernel passes are unobserved.
// CUPTIBENCH_KERNEL_BUDGET_METRICS optionally selects a bounded alternative to
// the GB20B candidate; the existing physical-test opt-in is still required.
func TestPhysicalKernelReplayHostBudget(t *testing.T) {
	p := newPhysicalPipeline(t, 4)
	selectors := []string{cuptibench.DefaultSet, "l1tex__t_requests.sum"}
	if selected := os.Getenv("CUPTIBENCH_KERNEL_BUDGET_METRICS"); selected != "" {
		if len(selected) > 4096 {
			t.Fatal("metric selection exceeds 4096 bytes")
		}
		selectors = strings.Split(selected, ",")
	}
	names, err := cuptibench.Resolve(append([]string{physicalMetric}, selectors...)...)
	physicalCheck(t, err)
	if len(names) > 16 {
		t.Fatal("metric selection exceeds 16 metrics")
	}
	var chip string
	physicalCheck(t, p.dev.WithContext(func() error {
		var err error
		chip, err = cuptisass.DeviceChipName(uint32(p.dev.Ordinal()))
		return err
	}))
	availability, err := cuptirange.CounterAvailability(p.dev, 16<<20)
	physicalCheck(t, err)
	query, err := cuptirange.NewPlan(chip, availability, names, 16<<20)
	if query != nil {
		physicalClose(t, query.Close)
	}
	if err != nil {
		t.Fatalf("no discriminating configuration: metrics=%q: %v", names, err)
	}
	_, passes := query.Configuration()
	physicalCheck(t, query.Close())
	t.Logf("chip=%s metrics=%q configuration_passes=%d host_budget=1; internal kernel passes unobserved", chip, names, passes)
	if passes <= 1 {
		t.Fatalf("nondiscriminating configuration: got %d passes, need more than one", passes)
	}
	plan, err := cuptibench.New(p.dev, cuptibench.Config{
		Mode: cuptibench.Kernels, ReplayMode: cuptirange.KernelReplay,
		Metrics: names, Operations: 1, MaxConfigBytes: 16 << 20,
		CachePolicy: cuptirange.CachePreserve,
		// The default set's traffic counters only widen the configuration past
		// one pass; this test asserts pass accounting and CTA counts, never
		// traffic, so the warm cache is recorded rather than evicted.
		AllowWarmCacheTraffic: true,
		Limits:                cuptirange.Limits{MaxRanges: 8, MaxRangeTreeNodes: 16, MaxCounterBytes: 32 << 20, MaxPasses: 1},
	})
	if plan != nil {
		physicalClose(t, plan.Close)
	}
	physicalCheck(t, err)
	r, err := plan.Collect(p.workload(""))
	if r != nil {
		data, encodeErr := json.Marshal(r)
		physicalCheck(t, encodeErr)
		t.Logf("host budget result=%s", data)
	}
	physicalCheck(t, err)
	q := r.Quality
	if !q.Decoded || !q.AllPassesSubmitted || q.DroppedRanges != 0 || q.PassesStarted != 1 || q.PassesStopped != 1 || p.prepares != 1 || p.runs != 1 {
		t.Fatalf("host replay accounting quality=%+v prepares=%d runs=%d", q, p.prepares, p.runs)
	}
	if r.Metadata["configuration_passes"] != strconv.FormatUint(uint64(passes), 10) || r.Metadata["replay_mode"] != "kernel" {
		t.Fatalf("configuration/replay provenance=%v", r.Metadata)
	}
	if len(r.Ranges) != 2 {
		t.Fatalf("ranges=%+v", r.Ranges)
	}
	for _, rg := range r.Ranges {
		if len(rg.Metrics) == 0 || rg.Metrics[0].Name != physicalMetric || rg.Metrics[0].Value != 4 {
			t.Fatalf("missing four-CTA kernel control: %+v", rg)
		}
	}
	checkPhysicalOutput(t, p, "", 1)
}
