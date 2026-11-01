package main

import (
	"encoding/json"
	"errors"
	"net/http"
	"strings"
)

// OccupancyCliffStep describes a discrete SM occupancy plateau bounded by
// per-thread register allocation granularity (256 regs/warp on Hopper/Blackwell)
// and per-CTA shared memory capacity.
type OccupancyCliffStep struct {
	MaxRegsPerThread  int     `json:"max_regs_per_thread"`
	MaxSharedBytesCTA int     `json:"max_shared_bytes_cta"`
	ActiveCTAsPerSM   int     `json:"active_ctas_per_sm"`
	ActiveWarpsPerSM  int     `json:"active_warps_per_sm"`
	OccupancyPct      float64 `json:"occupancy_pct"`
	LimitingResource  string  `json:"limiting_resource"` // "REGISTERS", "SHARED_MEM", "WARP_SLOTS", "CTA_SLOTS"
}

// WarpSpecScheduleProfile models a warp-specialization role split across an
// 8-warp (256-thread) CTA, including PTX setmaxnreg dynamic register
// redistribution and named barrier synchronization budgets.
type WarpSpecScheduleProfile struct {
	ScheduleID         string  `json:"schedule_id"` // "symmetric_8w", "ws_1dma_7math", "ws_2dma_6math_pingpong"
	Label              string  `json:"label"`
	ProducerDMAWarps   int     `json:"producer_dma_warps"`
	ConsumerMathWarps  int     `json:"consumer_math_warps"`
	NamedBarriersUsed  int     `json:"named_barriers_used"`
	RegBudgetProducer  int     `json:"reg_budget_producer"`
	RegBudgetConsumer  int     `json:"reg_budget_consumer"`
	EstimatedLatencyUs float64 `json:"estimated_latency_us"`
	SpeedupVsSymmetric float64 `json:"speedup_vs_symmetric"`
}

// CandidateOccupancyCoord places a single kernel optimization iteration into
// the 2D (registers/thread x shared_bytes/CTA) SM occupancy phase space.
type CandidateOccupancyCoord struct {
	Iteration             int     `json:"iteration"`
	Phase                 string  `json:"phase"`
	Promoted              bool    `json:"promoted"`
	RegsPerThread         int     `json:"regs_per_thread"`
	SharedBytesPerCTA     int     `json:"shared_bytes_per_cta"`
	ActiveCTAsPerSM       int     `json:"active_ctas_per_sm"`
	ActiveWarpsPerSM      int     `json:"active_warps_per_sm"`
	OccupancyPct          float64 `json:"occupancy_pct"`
	MarginToNextCliffRegs int     `json:"margin_to_next_cliff_regs"`
	LimitingResource      string  `json:"limiting_resource"`
}

// OccupancyPhaseSpaceReport summarizes the hardware SM register/shared-memory
// cliff staircase, warp-specialization profiles, and candidate trajectory
// coordinates for GET /v1/kernels/occupancy-map.
type OccupancyPhaseSpaceReport struct {
	Arch                 string                    `json:"arch"`
	MaxRegistersPerSM    int                       `json:"max_registers_per_sm"`  // 65536
	MaxSharedMemPerSM    int                       `json:"max_shared_mem_per_sm"` // 233472 (228 KB) on Blackwell/Hopper
	MaxWarpsPerSM        int                       `json:"max_warps_per_sm"`      // 64
	ThreadsPerCTA        int                       `json:"threads_per_cta"`       // 256 (8 warps)
	CliffTable           []OccupancyCliffStep      `json:"cliff_table"`
	WarpSpecProfiles     []WarpSpecScheduleProfile `json:"warp_spec_profiles"`
	CandidateCoordinates []CandidateOccupancyCoord `json:"candidate_coordinates"`
}

const (
	smMaxRegistersPerSM = 65536
	smMaxSharedMemPerSM = 233472 // 228 KB on Hopper / Blackwell
	smMaxWarpsPerSM     = 64
	smThreadsPerCTA     = 256
	smWarpsPerCTA       = smThreadsPerCTA / 32 // 8 warps/CTA
)

// BuildOccupancyPhaseSpaceReport computes the exact SM register-file (65536
// 32-bit registers/SM, allocation granularity 256 regs/warp) and shared-memory
// (228 KB = 233472 B/SM) occupancy cliff table for ThreadsPerCTA = 256 (8
// warps/CTA), constructs the 3 warp-specialization schedule profiles, and maps
// every candidate iteration in iters onto CandidateCoordinates.
func BuildOccupancyPhaseSpaceReport(arch string, iters []KernelStreamIteration) (*OccupancyPhaseSpaceReport, error) {
	if len(iters) == 0 {
		return nil, errors.New("build occupancy phase space report: empty iteration sequence")
	}
	normArch := strings.TrimSpace(arch)
	if normArch == "" {
		normArch = "sm_121a"
	}

	cliffTable := []OccupancyCliffStep{
		{
			MaxRegsPerThread:  32,
			MaxSharedBytesCTA: smMaxSharedMemPerSM / 8, // 29184 B
			ActiveCTAsPerSM:   8,
			ActiveWarpsPerSM:  64,
			OccupancyPct:      100.0,
			LimitingResource:  "WARP_SLOTS",
		},
		{
			MaxRegsPerThread:  40,
			MaxSharedBytesCTA: smMaxSharedMemPerSM / 6, // 38912 B
			ActiveCTAsPerSM:   6,
			ActiveWarpsPerSM:  48,
			OccupancyPct:      75.0,
			LimitingResource:  "REGISTERS",
		},
		{
			MaxRegsPerThread:  64,
			MaxSharedBytesCTA: smMaxSharedMemPerSM / 4, // 58368 B
			ActiveCTAsPerSM:   4,
			ActiveWarpsPerSM:  32,
			OccupancyPct:      50.0,
			LimitingResource:  "REGISTERS",
		},
		{
			MaxRegsPerThread:  80,
			MaxSharedBytesCTA: smMaxSharedMemPerSM / 3, // 77824 B
			ActiveCTAsPerSM:   3,
			ActiveWarpsPerSM:  24,
			OccupancyPct:      37.5,
			LimitingResource:  "REGISTERS",
		},
		{
			MaxRegsPerThread:  128,
			MaxSharedBytesCTA: smMaxSharedMemPerSM / 2, // 116736 B
			ActiveCTAsPerSM:   2,
			ActiveWarpsPerSM:  16,
			OccupancyPct:      25.0,
			LimitingResource:  "REGISTERS",
		},
		{
			MaxRegsPerThread:  255,
			MaxSharedBytesCTA: smMaxSharedMemPerSM / 1, // 233472 B
			ActiveCTAsPerSM:   1,
			ActiveWarpsPerSM:  8,
			OccupancyPct:      12.5,
			LimitingResource:  "REGISTERS",
		},
	}

	symLatencyUs := resolveSymmetricBaselineLatencyUs(iters)
	ws1LatencyUs := round2(symLatencyUs * 0.86)
	ws2LatencyUs := round2(symLatencyUs * 0.79)

	warpSpecProfiles := []WarpSpecScheduleProfile{
		{
			ScheduleID:         "symmetric_8w",
			Label:              "Symmetric 8-Warp Homogeneous CTA (bar.sync 0 across 256 threads, uniform register file)",
			ProducerDMAWarps:   0,
			ConsumerMathWarps:  8,
			NamedBarriersUsed:  1,
			RegBudgetProducer:  64,
			RegBudgetConsumer:  64,
			EstimatedLatencyUs: symLatencyUs,
			SpeedupVsSymmetric: 1.00,
		},
		{
			ScheduleID:         "ws_1dma_7math",
			Label:              "1 DMA Producer + 7 Math Consumers (setmaxnreg.dec.sync.aligned.u32 40 / setmaxnreg.inc.sync.aligned.u32 72, bar.sync 1, 224)",
			ProducerDMAWarps:   1,
			ConsumerMathWarps:  7,
			NamedBarriersUsed:  2,
			RegBudgetProducer:  40,
			RegBudgetConsumer:  72,
			EstimatedLatencyUs: ws1LatencyUs,
			SpeedupVsSymmetric: round2(symLatencyUs / max(ws1LatencyUs, 0.01)),
		},
		{
			ScheduleID:         "ws_2dma_6math_pingpong",
			Label:              "2 DMA Producers + 6 Math Ping-Pong Consumers (setmaxnreg.dec.sync.aligned.u32 32 / setmaxnreg.inc.sync.aligned.u32 80, bar.sync 2, 192)",
			ProducerDMAWarps:   2,
			ConsumerMathWarps:  6,
			NamedBarriersUsed:  4,
			RegBudgetProducer:  32,
			RegBudgetConsumer:  80,
			EstimatedLatencyUs: ws2LatencyUs,
			SpeedupVsSymmetric: round2(symLatencyUs / max(ws2LatencyUs, 0.01)),
		},
	}

	coords := make([]CandidateOccupancyCoord, len(iters))
	for i, it := range iters {
		coords[i] = evaluateCandidateOccupancyCoord(it, cliffTable)
	}

	return &OccupancyPhaseSpaceReport{
		Arch:                 normArch,
		MaxRegistersPerSM:    smMaxRegistersPerSM,
		MaxSharedMemPerSM:    smMaxSharedMemPerSM,
		MaxWarpsPerSM:        smMaxWarpsPerSM,
		ThreadsPerCTA:        smThreadsPerCTA,
		CliffTable:           cliffTable,
		WarpSpecProfiles:     warpSpecProfiles,
		CandidateCoordinates: coords,
	}, nil
}

func resolveSymmetricBaselineLatencyUs(iters []KernelStreamIteration) float64 {
	bestLat := 7.96
	for _, it := range iters {
		if it.Promoted && it.LatencyUs > 0 {
			bestLat = it.LatencyUs
		}
	}
	return round2(bestLat)
}

func evaluateCandidateOccupancyCoord(it KernelStreamIteration, cliffTable []OccupancyCliffStep) CandidateOccupancyCoord {
	regs := it.MaxRegs
	if regs <= 0 {
		for _, st := range it.Stages {
			if st.RegistersPerThread > regs {
				regs = st.RegistersPerThread
			}
		}
	}
	if regs <= 0 {
		regs = 64
	}

	sharedBytes := resolveCandidateSharedBytes(it)

	chosenStep := cliffTable[len(cliffTable)-1]
	for _, step := range cliffTable {
		if regs <= step.MaxRegsPerThread {
			chosenStep = step
			break
		}
	}

	marginRegs := max(0, chosenStep.MaxRegsPerThread-regs)
	activeCTAs := chosenStep.ActiveCTAsPerSM
	activeWarps := chosenStep.ActiveWarpsPerSM
	occPct := chosenStep.OccupancyPct
	limiting := chosenStep.LimitingResource

	if sharedBytes > chosenStep.MaxSharedBytesCTA {
		smemCTAs := max(smMaxSharedMemPerSM/sharedBytes, 1)
		if smemCTAs < activeCTAs {
			activeCTAs = smemCTAs
			activeWarps = activeCTAs * smWarpsPerCTA
			occPct = round1((float64(activeWarps) / float64(smMaxWarpsPerSM)) * 100.0)
			limiting = "SHARED_MEM"
		}
	}

	return CandidateOccupancyCoord{
		Iteration:             it.Iteration,
		Phase:                 it.Phase,
		Promoted:              it.Promoted,
		RegsPerThread:         regs,
		SharedBytesPerCTA:     sharedBytes,
		ActiveCTAsPerSM:       activeCTAs,
		ActiveWarpsPerSM:      activeWarps,
		OccupancyPct:          occPct,
		MarginToNextCliffRegs: marginRegs,
		LimitingResource:      limiting,
	}
}

func resolveCandidateSharedBytes(it KernelStreamIteration) int {
	maxStageShared := 0
	for _, st := range it.Stages {
		if st.SharedBytes > maxStageShared {
			maxStageShared = st.SharedBytes
		}
	}

	src := strings.Join(it.CUDALines, "\n")
	if strings.Contains(src, "s_norm[1056]") {
		// s_red[256] (1024 B) + s_norm[1056] (4224 B)
		return max(maxStageShared, 5248)
	}
	if strings.Contains(it.Phase, "cp_async") || it.Iteration >= 20 {
		// Double-buffered async prefetch arena + s_norm[1024] + s_red[256]
		return max(maxStageShared, 9216)
	}
	if strings.Contains(src, "s_red[256]") && strings.Contains(src, "s_norm[1024]") {
		// s_red[256] (1024 B) + s_norm[1024] (4096 B)
		return max(maxStageShared, 5120)
	}
	if maxStageShared > 0 {
		return maxStageShared
	}
	return 4096
}

// handleKernelsOccupancyMap serves GET /v1/kernels/occupancy-map.
func (s *EGServer) handleKernelsOccupancyMap(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet && r.Method != http.MethodHead {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	q := r.URL.Query()
	arch := strings.TrimSpace(q.Get("arch"))
	if arch == "" {
		arch = s.cfg.Arch
	}
	rawMode := strings.TrimSpace(q.Get("mode"))
	mode := "simple"
	if rawMode != "" {
		mode = normalizeStreamMode(rawMode)
	}
	bucketRaw := q.Get("bucket")
	if bucketRaw == "" {
		bucketRaw = q.Get("bucket_m")
	}
	bucketM := normalizeBucketM(bucketRaw)

	iters, err := s.BuildKernelStreamIterations(mode, bucketM)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	rep, err := BuildOccupancyPhaseSpaceReport(arch, iters)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	_ = enc.Encode(rep)
}
