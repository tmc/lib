package traceanalysis

import (
	"encoding/json"
	"sort"
)

// LaunchDistribution groups raw native symbol and geometry within a journal
// and device. It does not qualify application equivalence or engine-run samples.
type LaunchDistribution struct {
	Journal string   `json:"journal"`
	Device  uint32   `json:"device"`
	Symbol  string   `json:"raw_symbol"`
	Grid    [3]int   `json:"grid"`
	Block   [3]int   `json:"block"`
	Sources []Source `json:"sources"`
	Count   int      `json:"count"`
	Min     int64    `json:"min_ns"`
	Median  int64    `json:"median_ns"`
	Max     int64    `json:"max_ns"`
	Sum     int64    `json:"sum_ns"`
}

// KernelShapes separates raw launch distributions from shape qualification.
type KernelShapes struct {
	Launches      []Launch             `json:"launches"`
	Distributions []LaunchDistribution `json:"raw_launch_distributions"`
	Qualification string               `json:"shape_qualification"`
}

func kernelShapes(launches []Launch) KernelShapes {
	report := KernelShapes{Launches: launches, Distributions: []LaunchDistribution{}, Qualification: "unqualified: raw symbol/geometry do not establish matched application work"}
	groups := map[string][]Launch{}
	keys := []string{}
	for _, launch := range launches {
		key, _ := json.Marshal(struct {
			Journal     string
			Device      uint32
			Symbol      string
			Grid, Block [3]int
		}{launch.Source.Journal, launch.Device, launch.Symbol, launch.Grid, launch.Block})
		text := string(key)
		if _, ok := groups[text]; !ok {
			keys = append(keys, text)
		}
		groups[text] = append(groups[text], launch)
	}
	sort.Strings(keys)
	for _, key := range keys {
		group := groups[key]
		first := group[0]
		row := LaunchDistribution{Journal: first.Source.Journal, Device: first.Device, Symbol: first.Symbol, Grid: first.Grid, Block: first.Block, Sources: []Source{}, Count: len(group)}
		times := make([]int64, len(group))
		for i, launch := range group {
			times[i] = launch.Duration
			row.Sum += launch.Duration
			row.Sources = append(row.Sources, launch.Source)
		}
		sort.Slice(times, func(i, j int) bool { return times[i] < times[j] })
		row.Min = times[0]
		row.Median = times[len(times)/2]
		row.Max = times[len(times)-1]
		report.Distributions = append(report.Distributions, row)
	}
	return report
}
