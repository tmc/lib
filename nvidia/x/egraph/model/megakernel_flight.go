package model

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"slices"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudahotswap"
	"github.com/tmc/lib/nvidia/x/cudashape"
	"github.com/tmc/lib/nvidia/x/cuptipcsampling"
	"github.com/tmc/lib/nvidia/x/egraph/autotune"
	"github.com/tmc/lib/nvidia/x/egraph/hopper"
	"github.com/tmc/lib/nvidia/x/egraph/qualify"
)

// StageAttributionMap binds a lowered persistent [hopper.Megakernel] and its
// #line stage spans to non-overlapping SASS PC intervals for hardware stall
// attribution.
type StageAttributionMap struct {
	Kernel string
	Source string
	Spans  []hopper.StageLineSpan
	Ranges []autotune.PCRange
}

// StageProfile summarizes hardware PC-sampling stalls and stall-derived
// schedule hypotheses for a single stage inside a persistent megakernel.
type StageProfile struct {
	StageIndex int
	StageName  string
	Kernel     hopper.PlanKernel
	MegaStage  hopper.MegaStage
	Span       hopper.StageLineSpan
	PCRanges   []autotune.PCRange
	Histogram  autotune.StallHistogram
	Stalls     qualify.StallSample
	Hypotheses []qualify.StallHypothesis
}

// StageRewriteCandidate is a lowered, megakernel-verified schedule rewrite for
// one attributed stage of a [hopper.Plan], packaged with 3D launch geometry and
// a [cudahotswap.Source] ready for holdout-gated hot-swap.
type StageRewriteCandidate struct {
	Name        string
	StageIndex  int
	StageName   string
	Proposal    qualify.CandidateStepProposal
	Plan        *hopper.Plan
	Megakernel  hopper.Megakernel
	Source      string
	Spans       []hopper.StageLineSpan
	Grid        [3]int
	Block       [3]int
	SharedBytes int
	Geometry    cudahotswap.LaunchGeometry
	Hotswap     cudahotswap.Source
}

// MegakernelEpochSnapshot records one promoted megakernel version, its launch
// geometry, and the cumulative persistent-megakernel ready-barrier counter
// expectation at an epoch boundary.
type MegakernelEpochSnapshot struct {
	Epoch         uint64
	Blocks        int
	ExpectedReady uint64
	State         cudahotswap.State
	Geometry      *cudahotswap.LaunchGeometry
}

// EpochConfig configures per-epoch profiling and promotion parameters for a
// persistent megakernel flight controller.
type EpochConfig struct {
	Daemon       autotune.DaemonConfig
	MaxTrials    int
	Iters        int
	ImproveFloor float64
}

// Norm returns c with default parameters populated.
func (c EpochConfig) Norm() (EpochConfig, error) {
	if c.MaxTrials <= 0 {
		if c.Daemon.MaxTrials > 0 {
			c.MaxTrials = c.Daemon.MaxTrials
		} else {
			c.MaxTrials = 2
		}
	}
	if c.Iters <= 0 {
		if c.Daemon.Iters > 0 {
			c.Iters = c.Daemon.Iters
		} else {
			c.Iters = 20
		}
	}
	if c.ImproveFloor <= 0 {
		if c.Daemon.ImproveFloor > 0 {
			c.ImproveFloor = c.Daemon.ImproveFloor
		} else {
			c.ImproveFloor = 0.015
		}
	}
	return c, nil
}

// MegakernelEpochTracker tracks epoch transitions, persistent megakernel
// ready-barrier counts, and retained version history for multi-step hot-swap
// and immediate-prior rollback.
type MegakernelEpochTracker struct {
	Config  EpochConfig
	Service *cudahotswap.Service
	Kernel  string
	Blocks  int
	Epoch   uint64
	History []MegakernelEpochSnapshot
}

// NewMegakernelEpochTracker creates an epoch tracker for a persistent
// megakernel registered under kernel in service.
func NewMegakernelEpochTracker(cfg EpochConfig, service *cudahotswap.Service, kernel string, blocks int) (*MegakernelEpochTracker, error) {
	normCfg, err := cfg.Norm()
	if err != nil {
		return nil, fmt.Errorf("megakernel epoch tracker: %w", err)
	}
	if blocks <= 0 {
		blocks = 1
	}
	t := &MegakernelEpochTracker{
		Config:  normCfg,
		Service: service,
		Kernel:  kernel,
		Blocks:  blocks,
	}
	if service != nil && kernel != "" {
		if st, err := service.Status(kernel); err == nil {
			t.History = append(t.History, MegakernelEpochSnapshot{
				Epoch:         0,
				Blocks:        blocks,
				ExpectedReady: 0,
				State:         st,
			})
		}
	}
	return t, nil
}

// ExpectedReadyCounter returns the device ready-counter value expected after
// Epoch completed persistent-megakernel launches (`Epoch * Blocks`).
func (t *MegakernelEpochTracker) ExpectedReadyCounter() uint64 {
	if t == nil || t.Blocks <= 0 {
		return 0
	}
	return t.Epoch * uint64(t.Blocks)
}

// AdvanceEpoch increments the megakernel epoch counter and records the active
// state and launch geometry in the rollback history.
func (t *MegakernelEpochTracker) AdvanceEpoch(state cudahotswap.State, geom *cudahotswap.LaunchGeometry) MegakernelEpochSnapshot {
	t.Epoch++
	var gcopy *cudahotswap.LaunchGeometry
	if geom != nil {
		g := *geom
		gcopy = &g
	}
	snap := MegakernelEpochSnapshot{
		Epoch:         t.Epoch,
		Blocks:        t.Blocks,
		ExpectedReady: t.ExpectedReadyCounter(),
		State:         state,
		Geometry:      gcopy,
	}
	t.History = append(t.History, snap)
	return snap
}

// RestorePrevious rolls the tracked kernel in Service back to the immediate
// prior version in History (not merely the initial version 1) and restores its
// launch geometry.
func (t *MegakernelEpochTracker) RestorePrevious(ctx context.Context) (cudahotswap.State, error) {
	if t == nil || t.Service == nil || t.Kernel == "" {
		return cudahotswap.State{}, errors.New("restore previous: tracker service and kernel name are required")
	}
	if len(t.History) < 2 {
		return cudahotswap.State{}, errors.New("restore previous: no prior version in history")
	}
	prior := t.History[len(t.History)-2]
	state, err := t.Service.Select(ctx, t.Kernel, prior.State.Version, "restore_previous")
	if err != nil {
		return cudahotswap.State{}, fmt.Errorf("restore previous version %d: %w", prior.State.Version, err)
	}
	if prior.Geometry != nil {
		state, err = t.Service.SetGeometry(t.Kernel, prior.Geometry, "restore_previous_geometry")
		if err != nil {
			return state, fmt.Errorf("restore previous geometry: %w", err)
		}
	}
	t.History = t.History[:len(t.History)-1]
	t.History[len(t.History)-1].State = state
	return state, nil
}

// StageHotSwapRequest configures a holdout-gated megakernel stage rewrite
// evaluation and live hot-swap.
type StageHotSwapRequest struct {
	Service       *cudahotswap.Service
	KernelName    string
	Candidate     StageRewriteCandidate
	PriorGeometry *cudahotswap.LaunchGeometry
	Identity      qualify.MeasuredIdentity
	Evidence      qualify.HoldoutCandidateEvidence
	Verify        func(qualify.MeasuredPair) error
	Epoch         *MegakernelEpochTracker
}

// StageHotSwapResult reports the holdout gate outcome, any published
// [cudahotswap.PushResult], the immediate prior [cudahotswap.State], and the
// updated megakernel epoch snapshot.
type StageHotSwapResult struct {
	Promoted      bool
	KernelName    string
	Candidate     StageRewriteCandidate
	Promotion     qualify.HoldoutPromotionResult
	PriorState    cudahotswap.State
	PriorGeometry *cudahotswap.LaunchGeometry
	SelectedState cudahotswap.State
	PushResult    *cudahotswap.PushResult
	EpochSnapshot MegakernelEpochSnapshot
	Service       *cudahotswap.Service
	Epoch         *MegakernelEpochTracker
}

// RestorePrevious rolls the hot-swapped kernel back to the immediate prior
// version recorded before this promotion (via [cudahotswap.Service.Select]),
// restoring any prior launch geometry and updating the epoch tracker.
func (r *StageHotSwapResult) RestorePrevious(ctx context.Context) (cudahotswap.State, error) {
	if r == nil {
		return cudahotswap.State{}, errors.New("restore previous: nil hot-swap result")
	}
	if !r.Promoted {
		return cudahotswap.State{}, errors.New("restore previous: candidate was not promoted")
	}
	if r.Epoch != nil && r.Epoch.Service != nil && len(r.Epoch.History) >= 2 {
		state, err := r.Epoch.RestorePrevious(ctx)
		if err == nil {
			r.SelectedState = state
		}
		return state, err
	}
	if r.Service == nil || r.KernelName == "" || r.PriorState.Version == 0 {
		return cudahotswap.State{}, errors.New("restore previous: service, kernel name, and prior version are required")
	}
	state, err := r.Service.Select(ctx, r.KernelName, r.PriorState.Version, r.Candidate.Name+": restore_previous")
	if err != nil {
		return cudahotswap.State{}, fmt.Errorf("restore previous version %d: %w", r.PriorState.Version, err)
	}
	if r.PriorGeometry != nil {
		state, err = r.Service.SetGeometry(r.KernelName, r.PriorGeometry, r.Candidate.Name+": restore_previous_geometry")
		if err != nil {
			return state, fmt.Errorf("restore previous geometry: %w", err)
		}
	}
	r.SelectedState = state
	return state, nil
}

// VerifyDigest returns a [qualify.SelectHoldoutPromoted] admission verifier
// that enforces exact equality with expected's non-zero Context and Admission
// digests, distinct reference/candidate handles, and non-empty timing blocks.
func VerifyDigest(expected qualify.MeasuredIdentity) func(qualify.MeasuredPair) error {
	return func(pair qualify.MeasuredPair) error {
		if expected.Context == ([32]byte{}) || expected.Admission == ([32]byte{}) {
			return errors.New("expected measured identity has zero digest")
		}
		if pair.Identity != expected {
			return fmt.Errorf("measured identity mismatch: got context=%x admission=%x, want context=%x admission=%x",
				pair.Identity.Context[:4], pair.Identity.Admission[:4],
				expected.Context[:4], expected.Admission[:4])
		}
		if pair.Reference == "" || pair.Candidate == "" || pair.Reference == pair.Candidate {
			return errors.New("distinct non-empty reference and candidate handles required")
		}
		if len(pair.Blocks) == 0 {
			return errors.New("measured pair has no timing blocks")
		}
		return nil
	}
}

// VerifyTokenSequence verifies that reference and candidate frozen-prompt
// greedy token-ID trajectories are non-empty, equal in length, and identical at
// every token position.
func VerifyTokenSequence(referenceTokens, candidateTokens []int) error {
	if len(referenceTokens) == 0 || len(candidateTokens) == 0 {
		return errors.New("non-empty frozen-prompt greedy token-ID trajectories required")
	}
	n := min(len(referenceTokens), len(candidateTokens))
	for i := range n {
		if referenceTokens[i] != candidateTokens[i] {
			return fmt.Errorf("greedy token ID mismatch at index %d: reference=%d candidate=%d", i, referenceTokens[i], candidateTokens[i])
		}
	}
	if len(referenceTokens) != len(candidateTokens) {
		return fmt.Errorf("greedy token trajectory length mismatch at index %d: reference=%d candidate=%d", n, len(referenceTokens), len(candidateTokens))
	}
	return nil
}

// DefaultReasonInventory returns a standard CUPTI PC-sampling stall reason
// inventory mapping indices 1..5 to the five canonical Hopper/Blackwell stall
// buckets used by [autotune.HistogramsFromBatch].
func DefaultReasonInventory() cuptipcsampling.ReasonInventory {
	return cuptipcsampling.ReasonInventory{
		Reasons: []cuptipcsampling.StallReason{
			{Index: 1, Name: "smsp__pcsamp_warps_issue_stalled_tmem_wait"},
			{Index: 2, Name: "smsp__pcsamp_warps_issue_stalled_barrier"},
			{Index: 3, Name: "smsp__pcsamp_warps_issue_stalled_mio_throttle"},
			{Index: 4, Name: "smsp__pcsamp_warps_issue_stalled_math_pipe_throttle"},
			{Index: 5, Name: "smsp__pcsamp_warps_issue_stalled_long_scoreboard"},
		},
	}
}

// StallSampleFromHistogram converts an [autotune.StallHistogram] into a
// [qualify.StallSample] for stage-level bottleneck hypothesis generation.
func StallSampleFromHistogram(h autotune.StallHistogram) qualify.StallSample {
	return qualify.StallSample{
		TotalSamples:   h.TotalSamples,
		WaitTMEM:       h.WaitTMEM,
		WaitMBarrier:   h.WaitMBarrier,
		MIOThrottle:    h.MIOThrottle,
		MathThrottle:   h.MathThrottle,
		LongScoreboard: h.LongScoreboard,
	}
}

// StageLinePCMap maps a persistent [hopper.Megakernel]'s `#line` stage spans
// `[StartLine, EndLine]` through lineToPC into validated, non-overlapping
// hardware `[StartPC, EndPC)` ranges with Label set to each stage's name and
// Kernel set to mega.Name.
func StageLinePCMap(mega hopper.Megakernel, lineToPC map[int][2]uint64) (*StageAttributionMap, error) {
	if len(lineToPC) == 0 {
		return nil, errors.New("stage line pc map: empty lineToPC map")
	}
	src, spans, err := mega.LineInfoSource()
	if err != nil {
		return nil, fmt.Errorf("stage line pc map: %w", err)
	}
	if len(spans) == 0 {
		return nil, errors.New("stage line pc map: megakernel has no stage spans")
	}

	var ranges []autotune.PCRange
	for _, span := range spans {
		var (
			minStart uint64
			maxEnd   uint64
			matched  bool
		)
		for line := span.StartLine; line <= span.EndLine; line++ {
			pc, ok := lineToPC[line]
			if !ok {
				continue
			}
			if pc[0] >= pc[1] {
				return nil, fmt.Errorf("stage line pc map: stage %s line %d has invalid PC range [0x%x, 0x%x)", span.StageName, line, pc[0], pc[1])
			}
			if !matched || pc[0] < minStart {
				minStart = pc[0]
			}
			if !matched || pc[1] > maxEnd {
				maxEnd = pc[1]
			}
			matched = true
		}
		if !matched {
			continue
		}
		ranges = append(ranges, autotune.PCRange{
			Start: minStart,
			End:   maxEnd,
			Label: span.StageName,
		})
	}
	if len(ranges) == 0 {
		return nil, errors.New("stage line pc map: no stage line spans matched lineToPC")
	}
	if _, err := autotune.AttributionMap(ranges, nil); err != nil {
		return nil, fmt.Errorf("stage line pc map: %w", err)
	}
	return &StageAttributionMap{
		Kernel: mega.Name,
		Source: src,
		Spans:  spans,
		Ranges: ranges,
	}, nil
}

// AttributeMegakernelBatch attributes a CUPTI PC-sampling batch across the
// stages of mega using amap, converts each stage's [autotune.StallHistogram] to
// a [qualify.StallSample], attaches the stage's [hopper.PlanKernel] and
// [hopper.StageLineSpan], and sorts stages by descending issued stall sample
// count (WHERE to optimize).
func AttributeMegakernelBatch(
	plan *hopper.Plan,
	mega hopper.Megakernel,
	amap *StageAttributionMap,
	batch cuptipcsampling.Batch,
	inv ...cuptipcsampling.ReasonInventory,
) ([]StageProfile, error) {
	if plan == nil {
		return nil, errors.New("attribute megakernel batch: nil plan")
	}
	if amap == nil || len(amap.Ranges) == 0 {
		return nil, errors.New("attribute megakernel batch: empty stage attribution map")
	}
	if len(mega.Stages) == 0 {
		return nil, errors.New("attribute megakernel batch: megakernel has no stages")
	}
	if len(plan.Kernels) != len(mega.Stages) {
		return nil, fmt.Errorf("attribute megakernel batch: plan has %d kernels but megakernel has %d stages", len(plan.Kernels), len(mega.Stages))
	}

	reasonInv := DefaultReasonInventory()
	if len(inv) > 0 && len(inv[0].Reasons) > 0 {
		reasonInv = inv[0]
	}
	kernelName := amap.Kernel
	if kernelName == "" {
		kernelName = mega.Name
	}
	_, byStage, err := autotune.AttributeBatch([]cuptipcsampling.Batch{batch}, reasonInv, kernelName, amap.Ranges)
	if err != nil {
		return nil, fmt.Errorf("attribute megakernel batch: %w", err)
	}

	spanByStage := make(map[string]hopper.StageLineSpan, len(amap.Spans))
	for _, s := range amap.Spans {
		spanByStage[s.StageName] = s
	}
	rangesByStage := make(map[string][]autotune.PCRange, len(amap.Ranges))
	for _, r := range amap.Ranges {
		rangesByStage[r.Label] = append(rangesByStage[r.Label], r)
	}

	profiles := make([]StageProfile, len(mega.Stages))
	for i, st := range mega.Stages {
		hist := byStage[st.Name]
		stalls := StallSampleFromHistogram(hist)
		profiles[i] = StageProfile{
			StageIndex: i,
			StageName:  st.Name,
			Kernel:     clonePlanKernel(plan.Kernels[i]),
			MegaStage:  st,
			Span:       spanByStage[st.Name],
			PCRanges:   slices.Clone(rangesByStage[st.Name]),
			Histogram:  hist,
			Stalls:     stalls,
			Hypotheses: qualify.HypothesesFromStalls(stalls),
		}
	}

	slices.SortStableFunc(profiles, func(a, b StageProfile) int {
		if c := cmp.Compare(b.Stalls.TotalSamples, a.Stalls.TotalSamples); c != 0 {
			return c
		}
		return cmp.Compare(a.StageIndex, b.StageIndex)
	})
	return profiles, nil
}

// ProposeMegakernelStageRewrites iterates over attributed stages in descending
// stall hotness order, invokes [qualify.ProposeStepCandidates] to generate
// legal schedule rewrites (WHAT to try) without mutating NodeCost or
// conflating stall ratios with causal hardware counters, substitutes each
// candidate step into a cloned [hopper.Plan], and verifies that the resulting
// plan lowers cleanly to a persistent [hopper.Megakernel].
func ProposeMegakernelStageRewrites(
	plan *hopper.Plan,
	profiles []StageProfile,
	causal map[string]qualify.KernelObservation,
) ([]StageRewriteCandidate, error) {
	if plan == nil {
		return nil, errors.New("propose megakernel stage rewrites: nil plan")
	}
	if len(profiles) == 0 {
		return nil, errors.New("propose megakernel stage rewrites: empty stage profiles")
	}

	megaBaseName := plan.Name + "_mega"
	if plan.Name == "" {
		megaBaseName = "plan_mega"
	}

	var candidates []StageRewriteCandidate
	for _, prof := range profiles {
		if prof.StageIndex < 0 || prof.StageIndex >= len(plan.Kernels) {
			return nil, fmt.Errorf("propose megakernel stage rewrites: stage index %d out of range [0, %d)", prof.StageIndex, len(plan.Kernels))
		}
		var obs qualify.KernelObservation
		if causal != nil {
			obs = causal[prof.StageName]
		}
		if prof.Stalls.TotalSamples == 0 && len(qualify.ClassifyBottlenecks(obs)) == 0 {
			continue
		}

		baseStep := plan.Kernels[prof.StageIndex]
		proposals := qualify.ProposeStepCandidates(baseStep, obs, prof.Stalls)
		for pIdx, prop := range proposals {
			candPlan, candStep, ok := substituteMegakernelStage(plan, prof.StageIndex, baseStep, prop.Step)
			if !ok {
				continue
			}
			candProp := prop
			candProp.Step = candStep

			candMega, err := candPlan.Megakernel(megaBaseName)
			if err != nil {
				continue
			}
			src, spans, err := candMega.LineInfoSource()
			if err != nil {
				continue
			}

			grid, block := candMega.Grid()
			sharedBytes := candMega.SharedBytes()
			candName := fmt.Sprintf("%s_stage%d_%s_cand%d", megaBaseName, prof.StageIndex, prof.StageName, pIdx+1)
			geom := cudahotswap.LaunchGeometry{
				Grid: cudashape.Grid3D{
					X: uint32(max(grid[0], 1)),
					Y: uint32(max(grid[1], 1)),
					Z: uint32(max(grid[2], 1)),
				},
				Block: driver.Dim3{
					X: uint32(max(block[0], 1)),
					Y: uint32(max(block[1], 1)),
					Z: uint32(max(block[2], 1)),
				},
				SharedMem: uint32(max(sharedBytes, 0)),
			}
			candidates = append(candidates, StageRewriteCandidate{
				Name:        candName,
				StageIndex:  prof.StageIndex,
				StageName:   prof.StageName,
				Proposal:    candProp,
				Plan:        candPlan,
				Megakernel:  candMega,
				Source:      src,
				Spans:       spans,
				Grid:        grid,
				Block:       block,
				SharedBytes: sharedBytes,
				Geometry:    geom,
				Hotswap: cudahotswap.Source{
					Code:  src,
					Entry: candMega.Name,
				},
			})
		}
	}
	return candidates, nil
}

// EvaluateAndHotSwapStageRewrite runs the two-stage holdout promotion gate
// ([qualify.SelectHoldoutPromoted] with [VerifyDigest] and
// [VerifyTokenSequence] on frozen-prompt greedy token IDs, 3-block ABBA screen,
// >= 6-block held-out ABBA, and A/A null calibration) and, upon promotion,
// configures 3D launch geometry and pushes the winning candidate into
// [cudahotswap.Service] with [StageHotSwapResult.RestorePrevious] rollback
// protection and [MegakernelEpochTracker] epoch tracking.
func EvaluateAndHotSwapStageRewrite(ctx context.Context, req StageHotSwapRequest) (*StageHotSwapResult, error) {
	if ctx == nil {
		return nil, errors.New("evaluate and hot-swap stage rewrite: nil context")
	}
	identity := req.Identity
	if identity == (qualify.MeasuredIdentity{}) {
		identity = req.Evidence.Screen.Identity
	}
	verify := req.Verify
	if verify == nil && identity != (qualify.MeasuredIdentity{}) {
		verify = VerifyDigest(identity)
	}

	ev := req.Evidence
	if ev.Screen.Reference == "" && req.KernelName != "" {
		ev.Screen.Reference = req.KernelName
	}
	if ev.Screen.Candidate == "" && req.Candidate.Name != "" {
		ev.Screen.Candidate = req.Candidate.Name
	}

	kernelName := req.KernelName
	if kernelName == "" {
		kernelName = req.Candidate.Megakernel.Name
	}

	res := &StageHotSwapResult{
		KernelName:    kernelName,
		Candidate:     req.Candidate,
		PriorGeometry: req.PriorGeometry,
		Service:       req.Service,
		Epoch:         req.Epoch,
	}
	if req.Service != nil && kernelName != "" {
		if st, err := req.Service.Status(kernelName); err == nil {
			res.PriorState = st
			res.SelectedState = st
		}
	}

	promotion, err := qualify.SelectHoldoutPromoted(identity, ev, verify)
	res.Promotion = promotion
	if err != nil {
		return res, fmt.Errorf("evaluate stage rewrite %s: %w", req.Candidate.Name, err)
	}
	if err := VerifyTokenSequence(ev.ReferenceTokens, ev.CandidateTokens); err != nil {
		res.Promoted = false
		return res, nil
	}
	if promotion.Status != "EXACT_HOLDOUT_PROMOTED" || promotion.CandidateState != "PROMOTED_HOLDOUT_VERIFIED" {
		res.Promoted = false
		return res, nil
	}

	if req.Service != nil {
		if kernelName == "" {
			return res, errors.New("evaluate and hot-swap stage rewrite: kernel name required for service push")
		}
		priorState, err := req.Service.Status(kernelName)
		if err != nil {
			return res, fmt.Errorf("query prior kernel status %s: %w", kernelName, err)
		}
		res.PriorState = priorState

		if req.Candidate.Geometry.Block.X > 0 {
			geom := req.Candidate.Geometry
			if _, err := req.Service.SetGeometry(kernelName, &geom, req.Candidate.Name+": geometry"); err != nil {
				return res, fmt.Errorf("set candidate geometry %s: %w", kernelName, err)
			}
		}
		src := req.Candidate.Hotswap
		if src.Code == "" {
			src.Code = req.Candidate.Source
		}
		if src.Entry == "" {
			src.Entry = kernelName
		}
		label := req.Candidate.Name
		if label == "" {
			label = kernelName + "_rewrite"
		}
		pushRes, err := req.Service.Push(ctx, kernelName, src, label)
		res.PushResult = pushRes
		if err != nil {
			if req.PriorGeometry != nil {
				_, _ = req.Service.SetGeometry(kernelName, req.PriorGeometry, label+": rollback_geometry")
			}
			return res, fmt.Errorf("hot-swap push %s: %w", label, err)
		}
		res.SelectedState = pushRes.Selected
	}

	res.Promoted = true
	if req.Epoch != nil {
		if req.Service != nil && req.Epoch.Service == nil {
			req.Epoch.Service = req.Service
		}
		if req.Epoch.Kernel == "" {
			req.Epoch.Kernel = kernelName
		}
		if req.Epoch.Blocks <= 0 && req.Candidate.Grid[0] > 0 {
			req.Epoch.Blocks = req.Candidate.Grid[0]
		}
		if len(req.Epoch.History) == 0 && res.PriorState.Version != 0 {
			req.Epoch.History = append(req.Epoch.History, MegakernelEpochSnapshot{
				Epoch:         0,
				Blocks:        req.Epoch.Blocks,
				ExpectedReady: 0,
				State:         res.PriorState,
				Geometry:      req.PriorGeometry,
			})
		}
		geom := req.Candidate.Geometry
		res.EpochSnapshot = req.Epoch.AdvanceEpoch(res.SelectedState, &geom)
	}
	return res, nil
}

func substituteMegakernelStage(
	plan *hopper.Plan,
	stageIdx int,
	baseStep hopper.PlanKernel,
	candStep hopper.PlanKernel,
) (*hopper.Plan, hopper.PlanKernel, bool) {
	cloned := cloneHopperPlan(plan)
	step := clonePlanKernel(candStep)
	if len(step.Args) == 0 {
		step.Args = slices.Clone(baseStep.Args)
	}
	if step.M == 0 {
		step.M = baseStep.M
	}
	if step.Batch == 0 {
		step.Batch = baseStep.Batch
	}
	if step.MaxKeys == 0 {
		step.MaxKeys = baseStep.MaxKeys
	}

	// Normalize schedule knobs that persistent megakernel lowering forbids.
	if step.Row != nil && step.Row.Cluster {
		step.Row.Cluster = false
	}
	if step.Tile != nil && step.Tile.WarpSpecialized {
		step.Tile.WarpSpecialized = false
		if baseStep.Tile != nil && step.Tile.Stages == baseStep.Tile.Stages && step.Tile.Stages < 3 {
			step.Tile.Stages = 3
		}
		if _, err := step.Tile.Norm(); err != nil {
			step.Tile.Stages = 2
		}
	}
	if sameKernelSchedule(baseStep, step) {
		return nil, hopper.PlanKernel{}, false
	}

	prevStreamK := baseStep.Tile != nil && baseStep.Tile.StreamK > 0
	if step.Tile != nil && step.Tile.StreamK > 0 && !prevStreamK {
		if wsBytes := step.Tile.WorkspaceBytes(); wsBytes > 0 {
			wsName := fmt.Sprintf("%s_sk_ws%d", cloned.Name, stageIdx+1)
			if cloned.Name == "" {
				wsName = fmt.Sprintf("plan_sk_ws%d", stageIdx+1)
			}
			cloned.Temps = append(cloned.Temps, hopper.Tensor{
				Name:  wsName,
				DType: hopper.F32,
				Shape: []int{(wsBytes + 3) / 4},
			})
			step.Args = append(slices.Clone(baseStep.Args), wsName)
		}
	}
	if step.Attention != nil && len(step.Args) > 0 {
		maxKeys := step.MaxKeys
		if maxKeys <= 0 {
			if step.Attention.Window > 0 {
				maxKeys = step.Attention.Window + max(step.Attention.QLen, 1) - 1
			} else {
				maxKeys = 1024
			}
		}
		wsBytes := max(step.Attention.WorkspaceBytes(max(step.Batch, 1), maxKeys), 64)
		wsName := step.Args[len(step.Args)-1]
		for j := range cloned.Temps {
			if cloned.Temps[j].Name == wsName {
				cloned.Temps[j].Shape = []int{wsBytes}
			}
		}
	}

	cloned.Kernels[stageIdx] = step
	return cloned, step, true
}

func sameKernelSchedule(a, b hopper.PlanKernel) bool {
	switch {
	case a.Tile != nil && b.Tile != nil:
		ta, tb := *a.Tile, *b.Tile
		return ta.BM == tb.BM &&
			ta.BN == tb.BN &&
			ta.Stages == tb.Stages &&
			ta.WGMMA == tb.WGMMA &&
			ta.Swizzle128B == tb.Swizzle128B &&
			ta.WarpSpecialized == tb.WarpSpecialized &&
			ta.Repacked == tb.Repacked &&
			ta.DequantWords == tb.DequantWords &&
			ta.StreamK == tb.StreamK &&
			ta.GroupM == tb.GroupM
	case a.GEMM != nil && b.GEMM != nil:
		ga, gb := *a.GEMM, *b.GEMM
		return ga.RowsPerBlock == gb.RowsPerBlock &&
			ga.Warps == gb.Warps &&
			ga.SplitK == gb.SplitK &&
			ga.Unroll == gb.Unroll &&
			ga.Prefetch == gb.Prefetch &&
			ga.BlackwellFP4 == gb.BlackwellFP4
	case a.Attention != nil && b.Attention != nil:
		aa, ab := *a.Attention, *b.Attention
		return aa.Chunk == ab.Chunk &&
			aa.Depth == ab.Depth &&
			aa.Stages == ab.Stages &&
			aa.TableAhead == ab.TableAhead &&
			aa.WGMMA == ab.WGMMA &&
			aa.QKFP8 == ab.QKFP8 &&
			aa.Swizzle == ab.Swizzle &&
			aa.ClusterCombine == ab.ClusterCombine &&
			aa.KVHeadsPerBlock == ab.KVHeadsPerBlock &&
			aa.DeviceSplits == ab.DeviceSplits
	case a.Row != nil && b.Row != nil:
		return a.Row.Cluster == b.Row.Cluster
	default:
		return false
	}
}

func clonePlanKernel(k hopper.PlanKernel) hopper.PlanKernel {
	out := k
	out.Args = slices.Clone(k.Args)
	if k.GEMM != nil {
		g := *k.GEMM
		g.Extras = slices.Clone(k.GEMM.Extras)
		out.GEMM = &g
	}
	if k.Tile != nil {
		t := *k.Tile
		t.Extras = slices.Clone(k.Tile.Extras)
		out.Tile = &t
	}
	if k.Attention != nil {
		a := *k.Attention
		out.Attention = &a
	}
	if k.Row != nil {
		r := *k.Row
		r.Inputs = slices.Clone(k.Row.Inputs)
		r.Values = make([]hopper.RowValue, len(k.Row.Values))
		for i, v := range k.Row.Values {
			v.Args = slices.Clone(v.Args)
			r.Values[i] = v
		}
		r.Outputs = slices.Clone(k.Row.Outputs)
		out.Row = &r
	}
	if k.TopK != nil {
		tk := *k.TopK
		out.TopK = &tk
	}
	if k.Align != nil {
		al := *k.Align
		out.Align = &al
	}
	if k.ExpertSum != nil {
		es := *k.ExpertSum
		out.ExpertSum = &es
	}
	return out
}

func cloneHopperPlan(p *hopper.Plan) *hopper.Plan {
	if p == nil {
		return nil
	}
	out := &hopper.Plan{
		Name:    p.Name,
		Inputs:  make([]hopper.Tensor, len(p.Inputs)),
		Temps:   make([]hopper.Tensor, len(p.Temps)),
		Output:  hopper.Tensor{Name: p.Output.Name, DType: p.Output.DType, Shape: slices.Clone(p.Output.Shape)},
		Kernels: make([]hopper.PlanKernel, len(p.Kernels)),
	}
	for i, t := range p.Inputs {
		out.Inputs[i] = hopper.Tensor{Name: t.Name, DType: t.DType, Shape: slices.Clone(t.Shape)}
	}
	for i, t := range p.Temps {
		out.Temps[i] = hopper.Tensor{Name: t.Name, DType: t.DType, Shape: slices.Clone(t.Shape)}
	}
	for i, k := range p.Kernels {
		out.Kernels[i] = clonePlanKernel(k)
	}
	return out
}
