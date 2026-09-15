package cuptibench

import (
	"errors"
	"fmt"
	"github.com/tmc/lib/nvidia/cupti"
	"strings"
)

// The exact metrics below are event counts, not rates or per-instance averages.
// NVIDIA's metric transition table identifies inst_executed and l2_read_transactions:
// https://docs.nvidia.com/cuda/archive/10.2/nsight-compute/docs/NsightComputeCli/index.html
// The L2 count retains its TEX read filter and sector unit; no DRAM or byte
// conversion is implied. Native Type and Dimension must also match.
var metricSets = map[string][]string{
	"instructions": {"smsp__inst_executed.sum"},
	"launches":     {"sm__ctas_launched.sum"},
	"l2-read":      {"lts__t_sectors_srcunit_tex_op_read.sum"},
}

func additiveMetric(name, unit string, kind cupti.MetricType) bool {
	if kind != cupti.MetricTypeCounter {
		return false
	}
	switch name {
	case "sm__ctas_launched.sum":
		return unit == "cta" || unit == "ctas"
	case "smsp__inst_executed.sum":
		return unit == "inst" || unit == "instruction" || unit == "instructions"
	case "lts__t_sectors_srcunit_tex_op_read.sum", "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum":
		return unit == "sector" || unit == "sectors" || unit == "l2_sector"
	}
	return false
}
func resolveSets(sets []string) ([]string, error) {
	if len(sets) == 0 {
		return nil, errors.New("cuptibench: no metric sets")
	}
	var names []string
	seen := make(map[string]bool)
	for _, set := range sets {
		metrics, ok := metricSets[strings.TrimSpace(set)]
		if !ok {
			return nil, fmt.Errorf("cuptibench: unknown metric set %q", set)
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

// Resolve expands named metric sets into exact metric names. It performs no
// native calls; New validates the resolved names against the device's chip and
// observed counter availability. Supported sets are "instructions" (executed
// warp instructions), "launches" (thread blocks), and "l2-read" (TEX-source L2 read sectors).
func Resolve(sets ...string) ([]string, error) { return resolveSets(sets) }
