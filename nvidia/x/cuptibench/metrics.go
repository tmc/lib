package cuptibench

import (
	"errors"
	"fmt"
	"github.com/tmc/lib/nvidia/cupti"
	"sort"
	"strings"
)

// Curated sets name event counts, not rates or per-instance averages.
// NVIDIA's metric transition table identifies inst_executed and l2_read_transactions:
// https://docs.nvidia.com/cuda/archive/10.2/nsight-compute/docs/NsightComputeCli/index.html
// The L2 count retains its TEX read filter and sector unit; no DRAM or byte
// conversion is implied. Native Type and Dimension must also match.
//
// DefaultSet spans the three things a launch is usually judged by: how much
// work was scheduled, how many instructions ran, and how much the memory
// system moved. It is deliberately small, since every metric costs replay
// passes.
const DefaultSet = "default"

var metricSets = map[string][]string{
	DefaultSet:     {"sm__ctas_launched.sum", "smsp__inst_executed.sum", "lts__t_sectors_srcunit_tex_op_read.sum"},
	"instructions": {"smsp__inst_executed.sum"},
	"launches":     {"sm__ctas_launched.sum"},
	"l2-read":      {"lts__t_sectors_srcunit_tex_op_read.sum"},
}

// additiveUnits records the dimension each curated metric must carry. A metric
// whose native dimension leaves this set is not the quantity the set was
// written for, so it loses additivity rather than being summed on trust.
var additiveUnits = map[string]map[string]bool{
	"sm__ctas_launched.sum":                              {"cta": true, "ctas": true},
	"smsp__inst_executed.sum":                            {"inst": true, "instruction": true, "instructions": true},
	"lts__t_sectors_srcunit_tex_op_read.sum":             {"sector": true, "sectors": true, "l2_sector": true},
	"lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum": {"sector": true, "sectors": true, "l2_sector": true},
}

// additiveMetric reports whether values may be summed across ranges. Only a
// counter rollup qualifies: CUPTI's ".sum" submetric totals events over the
// collection domain, so totalling disjoint ranges stays an event count.
// Ratios, throughputs and the other rollups (".avg", ".max", ".peak_sustained")
// do not survive addition. A curated name must additionally still carry its
// recorded dimension, which catches a redefinition upstream; a discovered
// counter has no such expectation to check.
func additiveMetric(name, unit string, kind cupti.MetricType) bool {
	if kind != cupti.MetricTypeCounter || !strings.HasSuffix(name, ".sum") {
		return false
	}
	if want, curated := additiveUnits[name]; curated {
		return want[unit]
	}
	return unit != "" && unit != "unknown" && !timeDimension(unit)
}

// timeDimension reports whether a CUPTI dimension measures time. CUPTI writes a
// dimension as its unit tokens concatenated, so "gpc_cyclewarp" is warps times
// GPC cycles and a substring test is the available test.
//
// A time-dimensioned ".sum" is not additive across ranges. gpu__time_start.sum
// is a timestamp, and adding two timestamps means nothing; gpu__time_active.sum
// and the *_cycles_elapsed.sum family are durations summed over hardware units,
// so adding them across ranges counts overlapping wall-clock twice. On GB10 this
// excludes 222 of the 3271 counter rollups that would otherwise qualify,
// leaving 3049.
func timeDimension(unit string) bool {
	return strings.Contains(unit, "cycle") || strings.Contains(unit, "second")
}

// Sets returns the curated set names in sorted order.
func Sets() []string {
	names := make([]string, 0, len(metricSets))
	for name := range metricSets {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// Default returns the metrics Resolve gives for DefaultSet.
func Default() []string { return append([]string(nil), metricSets[DefaultSet]...) }

// metricName reports whether s is shaped like a native metric name rather than
// a set name. CUPTI names carry a submetric suffix, so the dot distinguishes
// them; validate rejects the remaining malformed cases, and New validates the
// name against the device.
func metricName(s string) bool {
	return strings.Contains(s, ".") && !strings.ContainsAny(s, " \t\x00")
}

func resolveSets(sets []string) ([]string, error) {
	if len(sets) == 0 {
		return nil, errors.New("cuptibench: no metric sets")
	}
	var names []string
	seen := make(map[string]bool)
	for _, set := range sets {
		set = strings.TrimSpace(set)
		metrics, ok := metricSets[set]
		if !ok {
			if !metricName(set) {
				return nil, fmt.Errorf("cuptibench: unknown metric set %q, have %s", set, strings.Join(Sets(), ", "))
			}
			metrics = []string{set}
		}
		for _, name := range metrics {
			if !seen[name] {
				names = append(names, name)
				seen[name] = true
			}
		}
	}
	return names, nil
}

// Resolve expands metric selectors into exact metric names, in the order given
// and without duplicates. A selector is either a curated set name from Sets()
// or a native metric name, which lets a caller name a metric the curated sets
// do not cover. "default" spans scheduled work, executed instructions and L2
// read traffic; "instructions" is executed warp instructions, "launches" is
// thread blocks, and "l2-read" is TEX-source L2 read sectors.
//
// Resolve performs no native calls, so it cannot tell whether a metric exists
// on any particular chip: New validates the resolved names against the device
// and its observed counter availability, and Reachable lists what that device
// actually offers.
func Resolve(sets ...string) ([]string, error) { return resolveSets(sets) }
