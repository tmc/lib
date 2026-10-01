package main

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudaprof/overview"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
)

// projectionTrackBound is a conservative descriptor bound. Timestamp/identity
// refusals may omit an auxiliary track, but never require more than this count.
// A comparison checks the sum for both arms before writing either member.
func projectionTrackBound(raw []traceRecord, summary *overview.Report, records *timeline.Records, telemetry int, links, lifecycle, memory, comparison bool) int64 {
	_, names := assignTracks(raw)
	count := int64(len(names) + 2) // export qualification and isolation
	if comparison {
		count += 2
	} // selected-window and boundary tracks
	if summary != nil {
		var nodeCount func(*overview.Node) int64
		journals := func(intervals []overview.Interval) int64 {
			seen := map[string]bool{}
			for _, interval := range intervals {
				seen[interval.Journal] = true
			}
			return int64(len(seen))
		}
		nodeCount = func(node *overview.Node) int64 {
			if node == nil {
				return 0
			}
			n := int64(1) + journals(node.Metrics.HostRanges)
			for _, intervals := range node.Metrics.GPUUnion {
				n += journals(intervals)
			}
			for _, child := range node.Children {
				n += nodeCount(child)
			}
			return n
		}
		count += nodeCount(summary.Hierarchy)
		for _, token := range summary.Tokens {
			count += int64(len(token.MaterializationRanges))
			if token.Delivery != nil {
				count++
			}
		}
		count += int64(len(summary.Inferred))
	}
	count += int64(telemetry)
	if records != nil {
		if links {
			type host struct {
				journal         string
				process, thread uint32
			}
			type device struct {
				journal        string
				device, stream uint32
			}
			hosts, devices := map[host]bool{}, map[device]bool{}
			for _, call := range records.Calls {
				hosts[host{call.Journal, call.Process, call.Thread}] = true
			}
			for _, activity := range records.Activities {
				devices[device{activity.Journal, activity.Device, activity.Stream}] = true
			}
			count += int64(len(hosts) + len(devices) + 2)
		}
		if lifecycle {
			count += int64(len(records.GraphLifecycle) + 1)
		}
		if memory {
			count += int64(len(records.UnifiedMemory) + len(records.MemoryPolicies) + 1)
		}
	}
	return count
}

func requireProjectionTracks(count, limit int64) error {
	if limit <= 0 {
		return fmt.Errorf("track budget must be positive")
	}
	if count > limit {
		return fmt.Errorf("combined trace track budget exceeded: conservative bound %d > %d", count, limit)
	}
	return nil
}

func summaryReferenceBound(report *overview.Report) int64 {
	if report == nil {
		return 0
	}
	var countNode func(*overview.Node) int64
	countNode = func(node *overview.Node) int64 {
		if node == nil {
			return 0
		}
		count := int64(len(node.References))
		for _, child := range node.Children {
			count += countNode(child)
		}
		return count
	}
	count := countNode(report.Hierarchy)
	for _, token := range report.Tokens {
		count += int64(len(token.References) + len(token.Materialization))
	}
	for _, token := range report.Inferred {
		count += int64(len(token.References))
	}
	return count
}
func requireSummaryReferences(count, limit int64) error {
	if limit <= 0 {
		return fmt.Errorf("summary reference budget must be positive")
	}
	if count > limit {
		return fmt.Errorf("combined summary reference budget exceeded: %d > %d", count, limit)
	}
	return nil
}
