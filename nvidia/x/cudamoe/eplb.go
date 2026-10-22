// SPDX-License-Identifier: Apache-2.0

package cudamoe

import (
	"fmt"
	"math"
	"sort"
)

// EPLBConfig configures the Expert Parallelism Load Balancer (EPLB).
type EPLBConfig struct {
	NumRoutedExperts   int
	NumPhysicalExperts int
	EPSize             int
	Decay              float64
}

// EPLBPolicy wraps [EPLBConfig] and provides stateful or stateless expert load
// rebalancing across EP ranks.
type EPLBPolicy struct {
	Config EPLBConfig
}

// Rebalance computes a new [EPLBPlacement] for loads using p.Config.
func (p EPLBPolicy) Rebalance(loads []float64) (EPLBPlacement, error) {
	return RebalanceExpertsGlobal(loads, p.Config)
}

// EPLBPlacement holds the bidirectional mapping between physical expert slots
// and logical routed experts across EP ranks.
type EPLBPlacement struct {
	PhysicalToLogical  []int32
	LogicalToPhysical  [][]int32
	ExpertReplicaCount []int32
	RankLoads          []float64
}

// ResolveNumPhysicalExperts returns the total number of physical experts by
// adding numRedundantExperts to numRoutedExperts and rounding up to the next
// multiple of epSize.
func ResolveNumPhysicalExperts(numRoutedExperts, numRedundantExperts, epSize int) (int, error) {
	if numRoutedExperts <= 0 {
		return 0, fmt.Errorf("cudamoe: numRoutedExperts %d must be positive", numRoutedExperts)
	}
	if numRedundantExperts < 0 {
		return 0, fmt.Errorf("cudamoe: numRedundantExperts %d must be non-negative", numRedundantExperts)
	}
	if epSize <= 0 {
		return 0, fmt.Errorf("cudamoe: epSize %d must be positive", epSize)
	}
	total := numRoutedExperts + numRedundantExperts
	rem := total % epSize
	if rem != 0 {
		total += epSize - rem
	}
	return total, nil
}

type physicalInstance struct {
	logicalExpert   int32
	effectiveWeight float64
}

// RebalanceExpertsGlobal implements the two-phase global EPLB algorithm:
//  1. Greedy replica assignment allocating redundant physical slots to logical
//     experts with maximal load-per-replica (ties broken by lowest logical ID).
//  2. Min-peak greedy rank packing sorting physical instances by descending
//     effective weight (ties broken by lowest logical ID) and assigning each
//     to the non-full rank with minimum cumulative load (ties broken by lowest
//     rank index) while spreading replicas of the same logical expert across
//     distinct ranks whenever possible.
func RebalanceExpertsGlobal(loads []float64, cfg EPLBConfig) (EPLBPlacement, error) {
	if cfg.NumRoutedExperts <= 0 {
		return EPLBPlacement{}, fmt.Errorf("cudamoe: NumRoutedExperts %d must be positive", cfg.NumRoutedExperts)
	}
	if cfg.EPSize <= 0 {
		return EPLBPlacement{}, fmt.Errorf("cudamoe: EPSize %d must be positive", cfg.EPSize)
	}
	if cfg.NumPhysicalExperts < cfg.NumRoutedExperts {
		return EPLBPlacement{}, fmt.Errorf("cudamoe: NumPhysicalExperts %d must be >= NumRoutedExperts %d", cfg.NumPhysicalExperts, cfg.NumRoutedExperts)
	}
	if cfg.NumPhysicalExperts%cfg.EPSize != 0 {
		return EPLBPlacement{}, fmt.Errorf("cudamoe: NumPhysicalExperts %d must be divisible by EPSize %d", cfg.NumPhysicalExperts, cfg.EPSize)
	}
	if len(loads) != cfg.NumRoutedExperts {
		return EPLBPlacement{}, fmt.Errorf("cudamoe: len(loads) %d must equal NumRoutedExperts %d", len(loads), cfg.NumRoutedExperts)
	}
	for i, v := range loads {
		if math.IsNaN(v) || math.IsInf(v, 0) || v < 0 {
			return EPLBPlacement{}, fmt.Errorf("cudamoe: load[%d] = %v must be finite and non-negative", i, v)
		}
	}

	replicas := make([]int32, cfg.NumRoutedExperts)
	for e := range replicas {
		replicas[e] = 1
	}

	numRedundant := cfg.NumPhysicalExperts - cfg.NumRoutedExperts
	for i := 0; i < numRedundant; i++ {
		bestExpert := 0
		bestScore := loads[0] / float64(replicas[0])
		for e := 1; e < cfg.NumRoutedExperts; e++ {
			score := loads[e] / float64(replicas[e])
			if score > bestScore {
				bestScore = score
				bestExpert = e
			}
		}
		replicas[bestExpert]++
	}

	instances := make([]physicalInstance, 0, cfg.NumPhysicalExperts)
	for e := 0; e < cfg.NumRoutedExperts; e++ {
		eff := loads[e] / float64(replicas[e])
		for r := int32(0); r < replicas[e]; r++ {
			instances = append(instances, physicalInstance{
				logicalExpert:   int32(e),
				effectiveWeight: eff,
			})
		}
	}

	sort.SliceStable(instances, func(i, j int) bool {
		if instances[i].effectiveWeight != instances[j].effectiveWeight {
			return instances[i].effectiveWeight > instances[j].effectiveWeight
		}
		return instances[i].logicalExpert < instances[j].logicalExpert
	})

	expertsPerRank := cfg.NumPhysicalExperts / cfg.EPSize
	physicalToLogical := make([]int32, cfg.NumPhysicalExperts)
	logicalToPhysical := make([][]int32, cfg.NumRoutedExperts)
	for e := 0; e < cfg.NumRoutedExperts; e++ {
		logicalToPhysical[e] = make([]int32, 0, replicas[e])
	}

	rankOccupancy := make([]int, cfg.EPSize)
	rankLoads := make([]float64, cfg.EPSize)
	rankHasExpert := make([][]bool, cfg.EPSize)
	for r := 0; r < cfg.EPSize; r++ {
		rankHasExpert[r] = make([]bool, cfg.NumRoutedExperts)
	}

	for _, inst := range instances {
		e := int(inst.logicalExpert)
		bestRank := -1
		bestLoad := math.Inf(1)

		// First prefer non-full ranks that do not yet hold a replica of logical expert e.
		for r := 0; r < cfg.EPSize; r++ {
			if rankOccupancy[r] >= expertsPerRank || rankHasExpert[r][e] {
				continue
			}
			if rankLoads[r] < bestLoad {
				bestLoad = rankLoads[r]
				bestRank = r
			}
		}
		// Fall back to any non-full rank if every available rank already holds a replica of e.
		if bestRank == -1 {
			for r := 0; r < cfg.EPSize; r++ {
				if rankOccupancy[r] >= expertsPerRank {
					continue
				}
				if rankLoads[r] < bestLoad {
					bestLoad = rankLoads[r]
					bestRank = r
				}
			}
		}

		slot := int32(bestRank*expertsPerRank + rankOccupancy[bestRank])
		physicalToLogical[slot] = inst.logicalExpert
		logicalToPhysical[e] = append(logicalToPhysical[e], slot)
		rankOccupancy[bestRank]++
		rankLoads[bestRank] += inst.effectiveWeight
		rankHasExpert[bestRank][e] = true
	}

	for e := range logicalToPhysical {
		slots := logicalToPhysical[e]
		sort.Slice(slots, func(i, j int) bool { return slots[i] < slots[j] })
	}

	return EPLBPlacement{
		PhysicalToLogical:  physicalToLogical,
		LogicalToPhysical:  logicalToPhysical,
		ExpertReplicaCount: replicas,
		RankLoads:          rankLoads,
	}, nil
}
