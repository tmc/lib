package advise

import (
	"fmt"
	"sort"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
)

func auditLaunchGeometry(cap *cudacapture.Capture, report *Report) {
	if cap == nil {
		return
	}
	launches := extractLaunches(cap)

	if cap.Profile != nil && len(cap.Profile.Sample) > 0 && len(launches) == 0 {
		report.Items = append(report.Items, Item{
			Severity:                 SeverityContext,
			Category:                 CategoryArchitecture,
			EvidenceLevel:            EvidenceUnmeasured,
			Finding:                  "Launch geometry labels ('grid', 'block') are absent in this profile (requires profiling with updated cupti_agent); launch geometry and SM occupancy audits skipped.",
			ActionableRecommendation: "Re-capture the profile using an updated libcupti_agent that serializes kernel launch dimensions.",
			NextMeasurement:          "launch geometry attributes ('grid', 'block') in profile samples",
		})
		return
	}

	smCount, hasSMCount := determineSMCount(cap)
	maxWarpsPerSM, maxBlocksPerSM, hasLimits := determineDeviceLimits(cap)

	if !hasLimits && !hasSMCount && cap.Profile != nil && len(launches) > 0 {
		report.Items = append(report.Items, Item{
			Severity:                 SeverityContext,
			Category:                 CategoryArchitecture,
			EvidenceLevel:            EvidenceUnmeasured,
			Finding:                  "Device architecture and SM count metadata are not recorded in this capture; advise does not query the local host's device, so theoretical SM occupancy and SM grid starvation audits were skipped.",
			ActionableRecommendation: "Record device provenance in the capture (e.g. capture-provenance.json with compute capability and SM count) and re-run advise.",
			NextMeasurement:          "capture-provenance.json with device compute capability and SM count",
		})
	}

	// 1. Underpopulated Thread Block / Single-Warp Warning (<= 32 threads)
	if hasLimits {
		seenSingleWarp := make(map[string]bool)
		for _, launch := range launches {
			if launch.BlockThreads > 0 && launch.BlockThreads <= 32 {
				if seenSingleWarp[launch.Kernel] {
					continue
				}
				// Gating: suppress single-block launches (Grid(1)), which are whole-GPU
				// starvation issues rather than CTA-geometry issues.
				if launch.GridBlocks == 1 {
					continue
				}

				warpsPerBlock := (launch.BlockThreads + 31) / 32
				if warpsPerBlock <= 0 {
					warpsPerBlock = 1
				}
				blocksPerSM := maxBlocksPerSM
				if maxWarpsPerSM/warpsPerBlock < blocksPerSM {
					blocksPerSM = maxWarpsPerSM / warpsPerBlock
				}
				residentWarpsPerSM := blocksPerSM * warpsPerBlock
				occupancyPct := (residentWarpsPerSM * 100) / maxWarpsPerSM

				// Gating: fire only when the launch's geometry leaves the SM underfilled
				// (i.e. capped by maxBlocksPerSM below the hardware's warp limit).
				if residentWarpsPerSM >= maxWarpsPerSM {
					continue
				}
				seenSingleWarp[launch.Kernel] = true

				finding := fmt.Sprintf("Single-warp thread blocks limit theoretical SM occupancy: kernel %q launched with %d block threads (<= 32) reaches %d%% theoretical occupancy with %d/%d resident warps/SM across %d blocks/SM",
					launch.Kernel, launch.BlockThreads, occupancyPct, residentWarpsPerSM, maxWarpsPerSM, blocksPerSM)

				report.Items = append(report.Items, Item{
					rule:                     "single-warp",
					Severity:                 SeverityCost,
					Category:                 CategoryArchitectureCompute,
					EvidenceLevel:            EvidenceSuspected,
					Location:                 launch.Location,
					Finding:                  finding,
					ActionableRecommendation: "Restructure thread block geometry to >=128 threads (4 warps) to enable warp-level latency hiding." + occupancyCaveat,
					NextMeasurement:          "sm__warps_active.avg.pct_of_peak_sustained_active or A/B wall-clock timing",
				})
			}
		}
	}

	// Multi-Warp Sequential Reduction Starvation (e.g. ds4_f16_pair_ordered_chunks)
	seenChunkReduction := make(map[string]bool)
	for _, launch := range launches {
		if launch.BlockThreads >= 128 {
			kLower := strings.ToLower(launch.Kernel)
			if strings.Contains(kLower, "ordered_chunks") || strings.Contains(kLower, "pair_ordered") || (strings.Contains(kLower, "chunk") && strings.Contains(kLower, "reduce")) {
				if seenChunkReduction[launch.Kernel] {
					continue
				}
				seenChunkReduction[launch.Kernel] = true
				report.Items = append(report.Items, Item{
					Severity:                 SeverityCost,
					Category:                 CategoryArchitectureCompute,
					EvidenceLevel:            EvidenceSuspected,
					Location:                 launch.Location,
					Finding:                  fmt.Sprintf("Multi-warp thread block underutilization in row reduction: kernel %q launched with %d threads/block (>= 4 warps) executes sequential chunk reductions with idle warps during accumulation", launch.Kernel, launch.BlockThreads),
					ActionableRecommendation: "Group multiple rows per thread block (e.g. 4 rows per 128-thread block so warp i computes row 4k+i) to maintain 100% warp utilization during chunk accumulation." + occupancyCaveat,
					NextMeasurement:          "sm__warps_active.avg.pct_of_peak_sustained_active or A/B wall-clock timing",
				})
			}
		}
	}

	// 2. SM Grid Starvation (< SMCount)
	//
	// One finding for all of them, and it recommends nothing.
	//
	// This was one finding per kernel, each recommending 2D tiling or split-K.
	// On a ds4 capture that was 23 of 66 findings, all saying the same thing,
	// and the recommendation named a cause that did not exist: no grid in that
	// attention path consults any device property, so there was no tuning
	// decision to revisit. The grids were literals mirroring a .cu define, and
	// they were correct -- the work really is that shape.
	//
	// A capture cannot tell a grid that was tuned from one that was fixed: it
	// sees a grid and an SM count. Absence of a device-property query is not
	// evidence either, only absence of visibility. So the observation stays,
	// because it is true and sometimes the reason, and the prescription goes.
	if hasSMCount {
		type starved struct {
			kernel  string
			blocks  int
			idlePct int
		}
		var list []starved
		seenGridStarvation := make(map[string]bool)
		for _, launch := range launches {
			if launch.GridBlocks > 0 && launch.GridBlocks < smCount {
				if seenGridStarvation[launch.Kernel] {
					continue
				}
				seenGridStarvation[launch.Kernel] = true
				list = append(list, starved{launch.Kernel, launch.GridBlocks, (smCount - launch.GridBlocks) * 100 / smCount})
			}
		}
		if len(list) > 0 {
			sort.Slice(list, func(i, j int) bool {
				if list[i].idlePct != list[j].idlePct {
					return list[i].idlePct > list[j].idlePct
				}
				return list[i].kernel < list[j].kernel
			})
			// Named in the finding, not a table: the finding renders as a
			// markdown heading, so it has to stay one line. The cap keeps a
			// capture with hundreds of kernels from filling the report with
			// the same sentence it just collapsed.
			const named = 8
			var parts []string
			for i, st := range list {
				if i == named {
					parts = append(parts, fmt.Sprintf("and %d more", len(list)-named))
					break
				}
				parts = append(parts, fmt.Sprintf("%s (%d blocks, %d%% idle)", st.kernel, st.blocks, st.idlePct))
			}
			loc := Location{}
			if len(list) == 1 {
				loc = Location{Kernel: list[0].kernel}
			}
			report.Items = append(report.Items, Item{
				Severity:      SeverityContext,
				Category:      CategoryArchitecture,
				EvidenceLevel: EvidenceObserved,
				Location:      loc,
				Finding: fmt.Sprintf("%d kernel(s) launched with fewer grid blocks than the GPU has SMs (%d SMs): %s. Whether that is a shortfall depends on how much work there is; this capture does not record how the grids were chosen.",
					len(list), smCount, strings.Join(parts, ", ")),
				ActionableRecommendation: "Compare achieved occupancy against these grids before changing any of them. A grid smaller than the SM count is a fact about the launch, not a defect: it is a problem only where there is more work available than the grid exposes.",
				NextMeasurement:          "sm__warps_active.avg.pct_of_peak_sustained_active per kernel, against the grid sizes above",
			})
		}
	}
}

// dim3Threads returns the product of a launch dimension, or 0 when any
// component is unset, which means the capture did not record the geometry.
func dim3Threads(d [3]int) int {
	n := 1
	for _, v := range d {
		if v <= 0 {
			return 0
		}
		n *= v
	}
	return n
}

// underSubscribedStat accumulates the launches of one kernel that cannot
// occupy the device.
type underSubscribedStat struct {
	kernel  string
	threads int
	count   int
	totalNs uint64
}

// formatUnderSubscribed names the worst offenders by accumulated device time.
func formatUnderSubscribed(stats map[string]*underSubscribedStat) string {
	ordered := make([]*underSubscribedStat, 0, len(stats))
	for _, st := range stats {
		ordered = append(ordered, st)
	}
	sort.Slice(ordered, func(i, j int) bool {
		if ordered[i].totalNs != ordered[j].totalNs {
			return ordered[i].totalNs > ordered[j].totalNs
		}
		return ordered[i].kernel < ordered[j].kernel
	})
	var parts []string
	for i := 0; i < len(ordered) && i < 3; i++ {
		st := ordered[i]
		meanUs := float64(st.totalNs) / float64(st.count) / 1000.0
		threadWord := "threads"
		if st.threads == 1 {
			threadWord = "thread"
		}
		parts = append(parts, fmt.Sprintf("%s (%d %s, %d launches, %.1f µs mean, %.2f ms total)", cleanKernelName(st.kernel), st.threads, threadWord, st.count, meanUs, float64(st.totalNs)/1e6))
	}
	return strings.Join(parts, ", ")
}

// worstKernel returns the kernel holding the most device time in stats.
func worstKernel(stats map[string]*underSubscribedStat) (string, uint64) {
	var name string
	var best uint64
	for _, st := range stats {
		if st.totalNs > best || (st.totalNs == best && (name == "" || st.kernel < name)) {
			name, best = st.kernel, st.totalNs
		}
	}
	return name, best
}

// auditUnderSubscribedLaunches reports launches whose total thread count cannot
// occupy the device: single-threaded launches, and launches that cannot supply
// one warp per SM. The per-SM rule needs the device SM count; when the capture
// does not record it, only the single-thread rule runs and the finding says so.
func auditUnderSubscribedLaunches(cap *cudacapture.Capture, report *Report) {
	if cap == nil {
		return
	}
	streamEvents, _ := timeline.Events(cap)
	if len(streamEvents) == 0 {
		return
	}

	smCount, hasSMCount := determineSMCount(cap)
	warpPerSMThreads := 0
	if hasSMCount {
		warpPerSMThreads = smCount * warpSize
	}

	single := make(map[string]*underSubscribedStat)
	subWarp := make(map[string]*underSubscribedStat)
	var singleCount, subWarpCount, geometryLaunches int
	var singleNs, subWarpNs uint64

	accumulate := func(stats map[string]*underSubscribedStat, ev timeline.Event, threads int) {
		st := stats[ev.Name]
		if st == nil {
			st = &underSubscribedStat{kernel: ev.Name, threads: threads}
			stats[ev.Name] = st
		}
		if threads < st.threads {
			st.threads = threads
		}
		st.count++
		st.totalNs += ev.DurationNs
	}

	for _, events := range streamEvents {
		for _, ev := range events {
			sub := strings.ToLower(ev.Subtype)
			if sub != "kernel" && sub != "" {
				continue
			}
			threads := dim3Threads(ev.Grid) * dim3Threads(ev.Block)
			if threads <= 0 || ev.Name == "" {
				continue
			}
			geometryLaunches++
			switch {
			case threads == 1:
				singleCount++
				singleNs += ev.DurationNs
				accumulate(single, ev, threads)
			case warpPerSMThreads > 0 && threads < warpPerSMThreads:
				subWarpCount++
				subWarpNs += ev.DurationNs
				accumulate(subWarp, ev, threads)
			}
		}
	}
	if geometryLaunches == 0 {
		return
	}

	if singleCount > 0 {
		finding := fmt.Sprintf("Single-threaded kernel launches leave the device idle: %d of %d captured launches (%.1f%%) ran one thread (grid 1 x block 1), consuming %.2f ms of device time; worst offenders: %s",
			singleCount, geometryLaunches, float64(singleCount)*100.0/float64(geometryLaunches), float64(singleNs)/1e6, formatUnderSubscribed(single))
		if !hasSMCount {
			finding += " (device SM count is not recorded in this capture and advise does not query the local device, so the one-warp-per-SM rule was skipped)"
		}
		kernel, _ := worstKernel(single)
		report.Items = append(report.Items, Item{
			Severity:        SeverityReadFirst,
			Category:        CategoryArchitectureCompute,
			EvidenceLevel:   EvidenceObserved,
			DeviceTimeNanos: int64(singleNs),
			Location: Location{
				Kernel: kernel,
			},
			Finding:                  finding,
			ActionableRecommendation: "Parallelize these kernels across threads and blocks, or fold their work into an adjacent kernel: a single-thread launch occupies one lane of one warp on one SM and leaves the rest of the device idle for its whole duration.",
			NextMeasurement:          "A/B wall-clock timing with parallelized kernel",
		})
	}

	if subWarpCount > 0 {
		kernel, _ := worstKernel(subWarp)
		report.Items = append(report.Items, Item{
			Severity:        SeverityCost,
			Category:        CategoryArchitectureCompute,
			EvidenceLevel:   EvidenceSuspected,
			DeviceTimeNanos: int64(subWarpNs),
			Location: Location{
				Kernel: kernel,
			},
			Finding: fmt.Sprintf("Launch geometry cannot fill one warp per SM: %d of %d captured launches (%.1f%%) ran fewer than %d total threads (%d SMs x %d threads per warp), consuming %.2f ms of device time; worst offenders: %s",
				subWarpCount, geometryLaunches, float64(subWarpCount)*100.0/float64(geometryLaunches), warpPerSMThreads, smCount, warpSize, float64(subWarpNs)/1e6, formatUnderSubscribed(subWarp)),
			ActionableRecommendation: fmt.Sprintf("Grow the grid so at least one warp is resident per SM (>= %d total threads), or batch the independent launches of this kernel into a single dispatch.", warpPerSMThreads),
			NextMeasurement:          "A/B wall-clock timing with increased grid size",
		})
	}
}
