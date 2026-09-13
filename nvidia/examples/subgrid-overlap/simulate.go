package main

import (
	"fmt"
	"math"
	"strings"
	"time"
)

// ExecutionMode defines how the producer-consumer pair is dispatched.
type ExecutionMode int

const (
	// ModeUnpipelined runs producer to completion, followed by host-driven
	// launch of consumer with stream barrier.
	ModeUnpipelined ExecutionMode = iota

	// ModeSequentialGraph runs producer to completion with zero host latency,
	// enforcing standard CUDA Graph node edge serialization.
	ModeSequentialGraph

	// ModePDLOverlap uses griddepcontrol.launch_dependents to trigger consumer
	// scheduling when all producer CTAs reach trigger_tile, allowing concurrent
	// CTA execution across SMs.
	ModePDLOverlap
)

func (m ExecutionMode) String() string {
	switch m {
	case ModeUnpipelined:
		return "Unpipelined Sequential"
	case ModeSequentialGraph:
		return "Sequential CUDA Graph"
	case ModePDLOverlap:
		return "PDL Sub-Grid Overlap"
	default:
		return fmt.Sprintf("Mode(%d)", m)
	}
}

// HardwareConfig models GPU execution resources.
type HardwareConfig struct {
	Name           string
	SMCount        int
	MaxWarpsPerSM  int
	MaxBlocksPerSM int
	ClockMHz       float64
}

// DefaultHardware returns a model of an NVIDIA Hopper H100 GPU.
func DefaultHardware() HardwareConfig {
	return HardwareConfig{
		Name:           "NVIDIA H100 (sm_90)",
		SMCount:        132,
		MaxWarpsPerSM:  64,
		MaxBlocksPerSM: 16,
		ClockMHz:       1755.0,
	}
}

// WorkloadConfig describes the producer and consumer execution parameters.
type WorkloadConfig struct {
	GridDimX           int     // Number of CTAs per kernel
	BlockDimX          int     // Threads per CTA (32 threads per warp)
	NumTiles           int     // Total tiles of data per CTA
	TileCycles         int     // Compute cycles per tile
	TriggerPercent     float64 // Midway trigger point (e.g. 0.25, 0.50, 0.75)
	HostLaunchLatency  int     // Cycles for host stream launch overhead (unpipelined)
	GraphLaunchLatency int     // Cycles for graph node transition overhead
}

// DefaultWorkload returns a standard balanced workload.
func DefaultWorkload() WorkloadConfig {
	return WorkloadConfig{
		GridDimX:           132, // 1 CTA per SM on H100
		BlockDimX:          256, // 8 warps per CTA
		NumTiles:           4,   // 4 tiles of work
		TileCycles:         25000,
		TriggerPercent:     0.50,
		HostLaunchLatency:  15000, // ~8.5 microseconds host launch
		GraphLaunchLatency: 1500,  // ~0.8 microseconds graph launch
	}
}

// SimConfig encapsulates the full simulation parameters.
type SimConfig struct {
	Hardware HardwareConfig
	Workload WorkloadConfig
	Mode     ExecutionMode
}

// TriggerTile returns the integer index of the tile that triggers early release.
func (w WorkloadConfig) TriggerTile() int {
	t := int(math.Round(float64(w.NumTiles) * w.TriggerPercent))
	if t < 1 {
		t = 1
	}
	if t >= w.NumTiles {
		t = w.NumTiles - 1
	}
	return t
}

// TimeSample captures instantaneous SM utilization.
type TimeSample struct {
	Cycle         int64
	ActiveWarps   int
	ActiveCTAs    int
	ProducerCTAs  int
	ConsumerCTAs  int
	OccupancyPcnt float64 // Active warps / Max theoretical warps
}

// SimResult summarizes simulated timing, concurrency, and occupancy metrics.
type SimResult struct {
	Mode               ExecutionMode
	Hardware           HardwareConfig
	Workload           WorkloadConfig
	TriggerTile        int
	TotalCycles        int64
	TotalDuration      time.Duration
	ProducerStartCycle int64
	ProducerEndCycle   int64
	ConsumerStartCycle int64
	ConsumerEndCycle   int64
	PDLTriggerCycle    int64
	OverlapCycles      int64
	OverlapDuration    time.Duration
	OverlapPercent     float64
	AverageOccupancy   float64 // Mean warp occupancy (0.0 - 1.0)
	PeakOccupancy      float64
	Samples            []TimeSample
}

// smResource tracks current CTA allocation on a single SM.
type smResource struct {
	id          int
	activeCTAs  int
	activeWarps int
}

func (sm *smResource) canFit(warps int, maxBlocks, maxWarps int) bool {
	return sm.activeCTAs+1 <= maxBlocks && sm.activeWarps+warps <= maxWarps
}

type ctaKind int

const (
	kindProducer ctaKind = iota
	kindConsumer
)

type ctaTask struct {
	id         int
	kind       ctaKind
	warps      int
	assignedSM int
	startCycle int64
	trigCycle  int64
	waitCycle  int64
	endCycle   int64
}

// Simulate runs a discrete-event execution simulation of the producer-consumer pair.
func Simulate(cfg SimConfig) (*SimResult, error) {
	hw := cfg.Hardware
	wl := cfg.Workload

	if hw.SMCount <= 0 {
		return nil, fmt.Errorf("simulate: invalid SMCount %d", hw.SMCount)
	}
	if wl.GridDimX <= 0 || wl.BlockDimX <= 0 || wl.NumTiles <= 0 {
		return nil, fmt.Errorf("simulate: invalid workload parameters")
	}

	warpsPerCTA := (wl.BlockDimX + 31) / 32
	if warpsPerCTA > hw.MaxWarpsPerSM {
		return nil, fmt.Errorf("simulate: CTA requires %d warps, max is %d", warpsPerCTA, hw.MaxWarpsPerSM)
	}

	triggerTile := wl.TriggerTile()
	earlyTileCount := triggerTile
	lateTileCount := wl.NumTiles - triggerTile

	earlyCycles := int64(earlyTileCount * wl.TileCycles)
	lateCycles := int64(lateTileCount * wl.TileCycles)

	// Step 1: Simulate Producer CTAs
	sms := make([]smResource, hw.SMCount)
	for i := range sms {
		sms[i].id = i
	}

	producerCTAs := make([]ctaTask, wl.GridDimX)
	for i := range producerCTAs {
		producerCTAs[i] = ctaTask{
			id:    i,
			kind:  kindProducer,
			warps: warpsPerCTA,
		}
	}

	// CTA schedule timeline tracking
	type ctaReleaseEvent struct {
		cycle int64
		sm    int
	}

	// Schedule Producer
	var curCycle int64
	var prodEndCycle int64
	var latestTriggerCycle int64

	ctaPerSMWait := make([][]int64, hw.SMCount)
	for i := range ctaPerSMWait {
		ctaPerSMWait[i] = make([]int64, 0)
	}

	// Schedule producer waves onto SMs
	assigned := 0
	smIdx := 0
	for assigned < wl.GridDimX {
		assignedThisRound := false
		for i := 0; i < hw.SMCount && assigned < wl.GridDimX; i++ {
			targetSM := (smIdx + i) % hw.SMCount
			if sms[targetSM].canFit(warpsPerCTA, hw.MaxBlocksPerSM, hw.MaxWarpsPerSM) {
				cta := &producerCTAs[assigned]
				cta.assignedSM = targetSM
				cta.startCycle = 0
				if len(ctaPerSMWait[targetSM]) > 0 {
					cta.startCycle = ctaPerSMWait[targetSM][len(ctaPerSMWait[targetSM])-1]
				}
				cta.trigCycle = cta.startCycle + earlyCycles
				cta.endCycle = cta.trigCycle + lateCycles
				ctaPerSMWait[targetSM] = append(ctaPerSMWait[targetSM], cta.endCycle)

				if cta.trigCycle > latestTriggerCycle {
					latestTriggerCycle = cta.trigCycle
				}
				if cta.endCycle > prodEndCycle {
					prodEndCycle = cta.endCycle
				}
				assigned++
				assignedThisRound = true
			}
		}
		if !assignedThisRound {
			// Find earliest completing CTA across all SMs
			minEnd := int64(math.MaxInt64)
			for _, ends := range ctaPerSMWait {
				for _, e := range ends {
					if e > curCycle && e < minEnd {
						minEnd = e
					}
				}
			}
			curCycle = minEnd
		}
	}

	// Step 2: Determine Consumer release condition based on Mode
	var consumerEligibleCycle int64
	switch cfg.Mode {
	case ModeUnpipelined:
		consumerEligibleCycle = prodEndCycle + int64(wl.HostLaunchLatency)
	case ModeSequentialGraph:
		consumerEligibleCycle = prodEndCycle + int64(wl.GraphLaunchLatency)
	case ModePDLOverlap:
		// Programmatic Dependent Launch releases consumer grid when all producer
		// CTAs execute griddepcontrol.launch_dependents!
		consumerEligibleCycle = latestTriggerCycle
	}

	// Schedule Consumer CTAs
	consumerCTAs := make([]ctaTask, wl.GridDimX)
	for i := range consumerCTAs {
		consumerCTAs[i] = ctaTask{
			id:    i,
			kind:  kindConsumer,
			warps: warpsPerCTA,
		}
	}

	// In PDL mode, consumer CTAs can be co-resident on SMs if slots are free,
	// or occupy SMs as producer CTAs retire.
	// For simulation accuracy, build an event-driven scheduler from consumerEligibleCycle.
	var consEndCycle int64
	var earliestConsStart int64 = math.MaxInt64

	// Clear SM state for integrated execution timeline
	for i := range sms {
		sms[i].activeCTAs = 0
		sms[i].activeWarps = 0
	}

	// Schedule Consumer CTAs onto SMs based on availability
	consumerAssigned := 0
	smCursor := 0

	// Track per-SM busy intervals [start, end]
	type busySpan struct {
		start, end int64
		warps      int
	}
	smSpans := make([][]busySpan, hw.SMCount)
	for _, p := range producerCTAs {
		smSpans[p.assignedSM] = append(smSpans[p.assignedSM], busySpan{
			start: p.startCycle,
			end:   p.endCycle,
			warps: p.warps,
		})
	}

	for consumerAssigned < wl.GridDimX {
		cta := &consumerCTAs[consumerAssigned]
		bestSM := -1
		var bestStart int64 = math.MaxInt64

		// Find SM that can host this consumer CTA at the earliest time >= consumerEligibleCycle
		for i := 0; i < hw.SMCount; i++ {
			candidateSM := (smCursor + i) % hw.SMCount
			// Find earliest cycle >= consumerEligibleCycle where SM capacity allows cta
			// Check candidate timeline:
			canStartAt := consumerEligibleCycle
			// In PDL mode, if SM has capacity alongside producer, it can start immediately!
			// Check existing spans on candidateSM
			for _, span := range smSpans[candidateSM] {
				// If adding this CTA would exceed max blocks or max warps during span:
				if span.warps+warpsPerCTA > hw.MaxWarpsPerSM {
					if span.end > canStartAt {
						canStartAt = span.end
					}
				}
			}
			if canStartAt < bestStart {
				bestStart = canStartAt
				bestSM = candidateSM
				if bestStart == consumerEligibleCycle {
					break // optimal
				}
			}
		}

		cta.assignedSM = bestSM
		cta.startCycle = bestStart
		if cta.startCycle < earliestConsStart {
			earliestConsStart = cta.startCycle
		}
		// Consumer processes early tiles
		cta.trigCycle = cta.startCycle + earlyCycles

		// At trigger point, consumer executes griddepcontrol.wait;
		// If producer is still in-flight, consumer must wait for producer grid completion!
		cta.waitCycle = cta.trigCycle
		if cfg.Mode == ModePDLOverlap && cta.waitCycle < prodEndCycle {
			// Stall waiting for producer grid completion
			cta.waitCycle = prodEndCycle
		}
		cta.endCycle = cta.waitCycle + lateCycles
		if cta.endCycle > consEndCycle {
			consEndCycle = cta.endCycle
		}

		smSpans[bestSM] = append(smSpans[bestSM], busySpan{
			start: cta.startCycle,
			end:   cta.endCycle,
			warps: cta.warps,
		})
		smCursor = (bestSM + 1) % hw.SMCount
		consumerAssigned++
	}

	totalCycles := consEndCycle
	if prodEndCycle > totalCycles {
		totalCycles = prodEndCycle
	}

	// Calculate Overlap Window
	// Overlap occurs when both producer and consumer have active CTAs
	var overlapCycles int64
	if earliestConsStart < prodEndCycle {
		overlapCycles = prodEndCycle - earliestConsStart
	}

	// Generate Time Samples for SM Occupancy
	numSamples := 100
	step := totalCycles / int64(numSamples)
	if step < 1 {
		step = 1
	}

	samples := make([]TimeSample, 0, numSamples+1)
	maxTheoreticalWarps := hw.SMCount * hw.MaxWarpsPerSM
	var sumOccupancy float64
	var peakOccupancy float64

	for c := int64(0); c <= totalCycles; c += step {
		var activeWarps, activeCTAs int
		var prodCount, consCount int

		for _, p := range producerCTAs {
			if c >= p.startCycle && c < p.endCycle {
				activeWarps += p.warps
				activeCTAs++
				prodCount++
			}
		}
		for _, k := range consumerCTAs {
			if c >= k.startCycle && c < k.endCycle {
				// If consumer is stalled at griddepcontrol.wait:
				activeWarps += k.warps
				activeCTAs++
				consCount++
			}
		}

		occ := float64(activeWarps) / float64(maxTheoreticalWarps)
		if occ > 1.0 {
			occ = 1.0
		}
		if occ > peakOccupancy {
			peakOccupancy = occ
		}
		sumOccupancy += occ

		samples = append(samples, TimeSample{
			Cycle:         c,
			ActiveWarps:   activeWarps,
			ActiveCTAs:    activeCTAs,
			ProducerCTAs:  prodCount,
			ConsumerCTAs:  consCount,
			OccupancyPcnt: occ * 100.0,
		})
	}

	avgOccupancy := 0.0
	if len(samples) > 0 {
		avgOccupancy = sumOccupancy / float64(len(samples))
	}

	// Convert cycles to time.Duration using clock MHz
	cycleDurationNs := 1e9 / (hw.ClockMHz * 1e6)
	totalDuration := time.Duration(float64(totalCycles) * cycleDurationNs)
	overlapDuration := time.Duration(float64(overlapCycles) * cycleDurationNs)

	overlapPcnt := 0.0
	if totalCycles > 0 {
		overlapPcnt = float64(overlapCycles) / float64(totalCycles) * 100.0
	}

	return &SimResult{
		Mode:               cfg.Mode,
		Hardware:           hw,
		Workload:           wl,
		TriggerTile:        triggerTile,
		TotalCycles:        totalCycles,
		TotalDuration:      totalDuration,
		ProducerStartCycle: 0,
		ProducerEndCycle:   prodEndCycle,
		ConsumerStartCycle: earliestConsStart,
		ConsumerEndCycle:   consEndCycle,
		PDLTriggerCycle:    latestTriggerCycle,
		OverlapCycles:      overlapCycles,
		OverlapDuration:    overlapDuration,
		OverlapPercent:     overlapPcnt,
		AverageOccupancy:   avgOccupancy,
		PeakOccupancy:      peakOccupancy,
		Samples:            samples,
	}, nil
}

// ASCIIChart renders an ASCII representation of the SM occupancy over time.
func (r *SimResult) ASCIIChart(width, height int) string {
	if len(r.Samples) == 0 {
		return "No data samples"
	}
	if width < 20 {
		width = 50
	}
	if height < 5 {
		height = 10
	}

	grid := make([][]rune, height)
	for i := range grid {
		grid[i] = make([]rune, width)
		for j := range grid[i] {
			grid[i][j] = ' '
		}
	}

	// Map samples to grid
	for col := 0; col < width; col++ {
		sampleIdx := int(math.Floor(float64(col) / float64(width) * float64(len(r.Samples))))
		if sampleIdx >= len(r.Samples) {
			sampleIdx = len(r.Samples) - 1
		}
		s := r.Samples[sampleIdx]
		occ := s.OccupancyPcnt / 100.0 // 0 to 1
		barHeight := int(math.Round(occ * float64(height-1)))

		for row := height - 1; row >= height-1-barHeight && row >= 0; row-- {
			if s.ProducerCTAs > 0 && s.ConsumerCTAs > 0 {
				grid[row][col] = '#' // Overlap zone
			} else if s.ProducerCTAs > 0 {
				grid[row][col] = '=' // Producer only
			} else if s.ConsumerCTAs > 0 {
				grid[row][col] = '*' // Consumer only
			} else {
				grid[row][col] = '.'
			}
		}
	}

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("SM Occupancy Timeline [%s]\n", r.Mode))
	sb.WriteString("Legend: '=' Producer only, '#' Overlap (both active), '*' Consumer only\n")
	sb.WriteString("100% |" + string(grid[0]) + "|\n")
	for row := 1; row < height-1; row++ {
		pct := int(100.0 * float64(height-1-row) / float64(height-1))
		sb.WriteString(fmt.Sprintf(" %2d%% |", pct) + string(grid[row]) + "|\n")
	}
	sb.WriteString("  0% |" + string(grid[height-1]) + "|\n")
	sb.WriteString("     +" + strings.Repeat("-", width) + "+\n")
	sb.WriteString(fmt.Sprintf("     0.0us%s%.2fus\n",
		strings.Repeat(" ", width-12),
		float64(r.TotalDuration.Microseconds())))

	return sb.String()
}
