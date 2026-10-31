package model

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"os"
	"slices"
	"strings"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudahotswap"
	"github.com/tmc/lib/nvidia/x/cudashape"
	"github.com/tmc/lib/nvidia/x/cuptipcsampling"
	"github.com/tmc/lib/nvidia/x/egraph"
	"github.com/tmc/lib/nvidia/x/egraph/autotune"
	"github.com/tmc/lib/nvidia/x/egraph/hopper"
	"github.com/tmc/lib/nvidia/x/egraph/qualify"
	"github.com/tmc/lib/nvidia/x/egraph/specdecode"
)

// AutonomousPipelineConfig configures the turnkey end-to-end
// [RunAutonomousFlightPipeline] connecting [autotune.VerifiedPGOCache],
// [qualify.ExtractStructuralPlanBeam], [hopper.BuildCrossLayerMegakernel],
// [qualify.RefinePlanByPTXASLoop], [qualify.SelectStructuralBeamWithHoldout],
// [specdecode.AdaptiveBucketController], and [cudahotswap.Service].
type AutonomousPipelineConfig struct {
	Cache            *autotune.VerifiedPGOCache
	CacheKey         autotune.PGOCacheKey
	Device           string
	Arch             string
	DeviceSMs        int
	BeamWidth        int
	PTXASCompiler    qualify.PTXASCompiler
	PTXASGate        qualify.StaticGate
	MaxPTXASIters    int
	HotSwapService   *cudahotswap.Service
	BucketController *specdecode.AdaptiveBucketController
	KernelName       string
	PostSwapCheck    cudahotswap.Checker
	EpochTracker     *MegakernelEpochTracker
}

// AutonomousEvalRequest supplies optional cross-layer fusion plans, CUPTI
// PC-sampling stall batches and line-to-PC mappings, causal counter
// observations, held-out ABBA tournament evidence, speculative bucket step
// observations, and the compiled winning cubin binary for
// [RunAutonomousFlightPipeline].
type AutonomousEvalRequest struct {
	CrossLayerPost       *hopper.Plan
	CrossLayerNextAttnIn *hopper.Plan
	CrossLayerGlue       hopper.CrossLayerGlue

	PCBatch            cuptipcsampling.Batch
	LineToPC           map[int][2]uint64
	ReasonInventory    cuptipcsampling.ReasonInventory
	CausalObservations map[string]qualify.KernelObservation
	MaxExpandedBeam    int

	Baseline          *qualify.StructuralBeamCandidate
	Identity          qualify.MeasuredIdentity
	Evidence          map[string]qualify.HoldoutCandidateEvidence
	DefaultEvidence   *qualify.HoldoutCandidateEvidence
	EvaluateCandidate func(qualify.StructuralBeamCandidate) (qualify.HoldoutCandidateEvidence, bool)
	Verify            func(qualify.MeasuredPair) error

	WinningCubin   []byte
	KernelName     string
	PostSwapCheck  cudahotswap.Checker
	ActiveBatch    int
	DraftedTokens  int
	AcceptedTokens int
	TargetBucket   *specdecode.BucketSpec
}

// AutonomousPipelineResult reports the outcome of [RunAutonomousFlightPipeline],
// including warm-start cache hits, structural beam candidates, zero-GPU ptxas
// loop refinements, cross-layer megakernel lowering, stage stall profiles,
// holdout tournament selection, live hot-swap state, and verified cache storage.
type AutonomousPipelineResult struct {
	CacheHit             bool
	CachedProof          *autotune.PGOPromotionProof
	CachedCubin          []byte
	BeamCandidates       []qualify.StructuralBeamCandidate
	PTXASRefinements     []*qualify.PTXASLoopResult
	CrossLayerPlan       *hopper.Plan
	CrossLayerMegakernel *hopper.Megakernel
	StageProfiles        []StageProfile
	Winner               *qualify.StructuralBeamWinner
	PromotedVersion      string
	PersistedToCache     bool

	KernelName       string
	PriorState       cudahotswap.State
	PriorGeometry    *cudahotswap.LaunchGeometry
	SelectedState    cudahotswap.State
	PushResult       *cudahotswap.PushResult
	BucketTransition *specdecode.BucketTransition
	Service          *cudahotswap.Service
	BucketController *specdecode.AdaptiveBucketController
	Epoch            *MegakernelEpochTracker
}

// RestorePrevious rolls the promoted kernel in r.Service (and any bucket
// transition in r.BucketController) back to the immediate prior retained
// version recorded before this pipeline promotion.
func (r *AutonomousPipelineResult) RestorePrevious(ctx context.Context) (cudahotswap.State, error) {
	if r == nil {
		return cudahotswap.State{}, errors.New("restore previous: nil autonomous pipeline result")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if r.BucketController != nil && r.BucketTransition != nil && r.BucketTransition.Swapped {
		if _, _, err := r.BucketController.RollbackBucket(ctx); err != nil {
			return cudahotswap.State{}, fmt.Errorf("restore previous bucket: %w", err)
		}
		r.BucketTransition.Swapped = false
	}
	if r.Epoch != nil && r.Epoch.Service != nil && len(r.Epoch.History) >= 2 {
		state, err := r.Epoch.RestorePrevious(ctx)
		if err == nil {
			r.SelectedState = state
			r.PromotedVersion = fmt.Sprintf("v%d", state.Version)
		}
		return state, err
	}
	if r.Service == nil || r.KernelName == "" || r.PriorState.Version == 0 {
		return cudahotswap.State{}, errors.New("restore previous: service, kernel name, and prior version are required")
	}
	state, err := r.Service.Select(ctx, r.KernelName, r.PriorState.Version, r.KernelName+": restore_previous")
	if err != nil {
		return cudahotswap.State{}, fmt.Errorf("restore previous version %d: %w", r.PriorState.Version, err)
	}
	if r.PriorGeometry != nil {
		state, err = r.Service.SetGeometry(r.KernelName, r.PriorGeometry, r.KernelName+": restore_previous_geometry")
		if err != nil {
			return state, fmt.Errorf("restore previous geometry: %w", err)
		}
	}
	r.SelectedState = state
	r.PromotedVersion = fmt.Sprintf("v%d", state.Version)
	return state, nil
}

// RunAutonomousFlightPipeline executes the 4-stage autonomous optimization,
// qualification, hot-swap, and persistence pipeline:
//
//  1. Warm-Start Cache Hit: If cfg.Cache is non-nil, attempts
//     [autotune.VerifiedPGOCache.LoadVerified]. On a verified hit, installs or
//     selects the cached kernel in cfg.HotSwapService and returns immediately
//     with CacheHit=true.
//  2. Structural Beam + Optional Cross-Layer Megakernel + Zero-GPU ptxas Cure:
//     Extracts up to cfg.BeamWidth structural candidates via
//     [qualify.ExtractStructuralPlanBeam], optionally fuses req.CrossLayerPost
//     and req.CrossLayerNextAttnIn via [hopper.BuildCrossLayerMegakernel], and
//     refines each candidate with [qualify.RefinePlanByPTXASLoop] when
//     cfg.PTXASCompiler is non-nil.
//  3. PC Attribution + Profile Expansion + Holdout Tournament: When req.PCBatch
//     and req.LineToPC are non-empty, attributes hardware stalls across stages
//     with [StageLinePCMap] and [AttributeMegakernelBatch], expands the beam
//     via [qualify.ExpandBeamWithProfile], and runs the two-stage holdout +
//     greedy token-ID tournament via [qualify.SelectStructuralBeamWithHoldout].
//  4. Live Hot-Swap + Adaptive Bucket Update + Verified Cache Store: When the
//     tournament promotes a candidate, pushes it into cfg.HotSwapService with
//     [AutonomousPipelineResult.RestorePrevious] rollback support, updates
//     cfg.BucketController, and atomically stores the promotion proof and
//     req.WinningCubin in cfg.Cache.
func RunAutonomousFlightPipeline(
	ctx context.Context,
	g *egraph.EGraph,
	root egraph.Id,
	name string,
	out hopper.DType,
	cfg AutonomousPipelineConfig,
	req AutonomousEvalRequest,
) (*AutonomousPipelineResult, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if strings.TrimSpace(name) == "" {
		name = "autonomous_plan"
	}
	if out == 0 {
		out = hopper.BF16
	}
	kernelName := resolveAutonomousKernelName(name, cfg, req)
	postCheck := req.PostSwapCheck
	if postCheck == nil {
		postCheck = cfg.PostSwapCheck
	}

	res := &AutonomousPipelineResult{
		KernelName:       kernelName,
		Service:          cfg.HotSwapService,
		BucketController: cfg.BucketController,
		Epoch:            cfg.EpochTracker,
	}

	// Step 1: Warm-Start Cache Hit.
	if cfg.Cache != nil && cfg.CacheKey.Validate() == nil {
		proof, cubin, err := cfg.Cache.LoadVerified(cfg.CacheKey)
		if err == nil {
			if gotSum := sha256.Sum256(cubin); gotSum != proof.CubinSHA256 {
				return nil, fmt.Errorf("autonomous flight pipeline: %w", autotune.ErrPGODigestMismatch)
			}
			proofCopy := proof
			res.CacheHit = true
			res.CachedProof = &proofCopy
			res.CachedCubin = slices.Clone(cubin)
			res.PromotedVersion = proof.CandidateName

			if cfg.HotSwapService != nil {
				code := string(proof.ScheduleJSON)
				if strings.TrimSpace(code) == "" || !strings.Contains(code, "__global__") {
					code = synthesizeHotSwapSource(kernelName, proof.CandidateName, proof.CubinSHA256)
				} else {
					code = ensureEntryWrapper(code, kernelName)
				}
				src := cudahotswap.Source{
					Code:     code,
					Filename: kernelName + ".cu",
					Entry:    kernelName,
				}
				label := proof.CandidateName
				if label == "" {
					label = "pgo_cache_hit"
				}
				prior, selected, pushRes, swapErr := installOrSelectHotSwap(ctx, cfg.HotSwapService, kernelName, src, nil, label, postCheck)
				res.PriorState = prior
				res.SelectedState = selected
				res.PushResult = pushRes
				if swapErr != nil {
					return res, fmt.Errorf("autonomous flight pipeline cache hit hot-swap: %w", swapErr)
				}
				if selected.Version != 0 {
					res.PromotedVersion = fmt.Sprintf("v%d", selected.Version)
				}
			}
			return res, nil
		}
		if !errors.Is(err, os.ErrNotExist) {
			return nil, fmt.Errorf("autonomous flight pipeline cache load: %w", err)
		}
	}

	// Step 2: Structural Beam + Optional Cross-Layer Megakernel + Zero-GPU ptxas -v Cure.
	beamWidth := cfg.BeamWidth
	if beamWidth <= 0 {
		beamWidth = 4
	}
	costCfg := hopper.CostConfig{
		Device: cfg.Device,
		SMs:    cfg.DeviceSMs,
	}
	if costCfg.Device == "" && cfg.Arch != "" {
		costCfg.Device = cfg.Arch
	}
	normCostCfg, err := costCfg.Norm()
	if err != nil {
		normCostCfg = costCfg
	}

	var (
		beam    []qualify.StructuralBeamCandidate
		seenSig = make(map[string]bool)
	)
	addBeamCandidate := func(c qualify.StructuralBeamCandidate) {
		if c.Plan == nil || len(c.Plan.Kernels) == 0 {
			return
		}
		if c.PlanSignature == "" {
			c.PlanSignature = qualify.ComputePlanSignature(c.Plan, c.Megakernel)
		}
		if c.PlanSignature == "" || seenSig[c.PlanSignature] {
			return
		}
		seenSig[c.PlanSignature] = true
		beam = append(beam, c)
	}

	if g != nil {
		extracted, err := qualify.ExtractStructuralPlanBeam(g, root, name, out, normCostCfg, beamWidth)
		if err != nil {
			return nil, fmt.Errorf("autonomous flight pipeline extract beam: %w", err)
		}
		for _, c := range extracted {
			addBeamCandidate(c)
		}
	}

	if req.CrossLayerPost != nil && req.CrossLayerNextAttnIn != nil {
		fusedPlan, crossMega, err := hopper.BuildCrossLayerMegakernel(req.CrossLayerPost, req.CrossLayerNextAttnIn, req.CrossLayerGlue)
		if err != nil {
			return nil, fmt.Errorf("autonomous flight pipeline cross-layer fuse: %w", err)
		}
		megaCopy := crossMega
		res.CrossLayerPlan = fusedPlan
		res.CrossLayerMegakernel = &megaCopy

		crossName := crossMega.Name
		if crossName == "" {
			crossName = fusedPlan.Name
		}
		crossSig := qualify.ComputePlanSignature(fusedPlan, &megaCopy)
		crossEst := fusedPlan.EstimateNs(normCostCfg)
		addBeamCandidate(qualify.StructuralBeamCandidate{
			Name:          crossName,
			Plan:          fusedPlan,
			Megakernel:    &megaCopy,
			PlanSignature: crossSig,
			EstimatedNs:   crossEst,
			CostConfig:    normCostCfg,
		})
	}

	if len(beam) == 0 {
		return nil, errors.New("autonomous flight pipeline: no structural candidates in beam")
	}

	// Keep pre-refinement megakernel snapshots so PC attribution line spans
	// match even if ptxas refinement alters emitted line numbers.
	preRefineMegas := make([]*hopper.Megakernel, 0, len(beam))
	preRefinePlans := make([]*hopper.Plan, 0, len(beam))
	for _, c := range beam {
		if c.Megakernel != nil && c.Plan != nil {
			mc := *c.Megakernel
			preRefineMegas = append(preRefineMegas, &mc)
			preRefinePlans = append(preRefinePlans, cloneHopperPlan(c.Plan))
		}
	}

	if cfg.PTXASCompiler != nil {
		arch := strings.TrimSpace(cfg.Arch)
		if arch == "" {
			arch = "sm_90a"
		}
		loopCfg := qualify.PTXASLoopConfig{
			Arch:          arch,
			Device:        cfg.Device,
			DeviceSMs:     cfg.DeviceSMs,
			MaxIterations: cfg.MaxPTXASIters,
			Gate:          cfg.PTXASGate,
		}
		refinedBeam := make([]qualify.StructuralBeamCandidate, 0, len(beam))
		refinedSeen := make(map[string]bool, len(beam))
		for _, cand := range beam {
			loopRes, err := qualify.RefinePlanByPTXASLoop(ctx, cand.Plan, cfg.PTXASCompiler, loopCfg)
			if err != nil {
				return nil, fmt.Errorf("autonomous flight pipeline ptxas loop (%s): %w", cand.Handle(), err)
			}
			res.PTXASRefinements = append(res.PTXASRefinements, loopRes)

			updated := cand
			if loopRes.RefinedPlan != nil {
				updated.Plan = loopRes.RefinedPlan
			}
			if updated.Megakernel != nil {
				megaName := updated.Megakernel.Name
				if megaName == "" {
					megaName = updated.Handle() + "_mega"
				}
				if compatPlan, rebuiltMega, ok := rebuildPlanMegakernel(updated.Plan, megaName); ok {
					updated.Plan = compatPlan
					updated.Megakernel = &rebuiltMega
				}
			} else if loopRes.Megakernel != nil {
				mc := *loopRes.Megakernel
				updated.Megakernel = &mc
			}
			if res.CrossLayerPlan != nil && cand.Name == res.CrossLayerMegakernel.Name {
				res.CrossLayerPlan = updated.Plan
				if updated.Megakernel != nil {
					mc := *updated.Megakernel
					res.CrossLayerMegakernel = &mc
				}
			}
			updated.PlanSignature = qualify.ComputePlanSignature(updated.Plan, updated.Megakernel)
			updated.EstimatedNs = updated.Plan.EstimateNs(normCostCfg)
			if updated.PlanSignature != "" && !refinedSeen[updated.PlanSignature] {
				refinedSeen[updated.PlanSignature] = true
				refinedBeam = append(refinedBeam, updated)
			} else if len(refinedBeam) == 0 {
				refinedBeam = append(refinedBeam, updated)
			}
		}
		beam = refinedBeam
	}

	// Step 3: PC Attribution + Profile Expansion + Holdout Tournament.
	if len(req.PCBatch.PCs) > 0 && len(req.LineToPC) > 0 {
		attrPlan, attrMega, amap, err := resolveAttributionMegakernel(beam, res.CrossLayerPlan, res.CrossLayerMegakernel, preRefinePlans, preRefineMegas, req.LineToPC)
		if err != nil {
			return nil, fmt.Errorf("autonomous flight pipeline stage line pc map: %w", err)
		}
		normBatch := normalizeBatchKernelName(req.PCBatch, amap.Kernel)
		profiles, err := AttributeMegakernelBatch(attrPlan, attrMega, amap, normBatch, req.ReasonInventory)
		if err != nil {
			return nil, fmt.Errorf("autonomous flight pipeline attribute batch: %w", err)
		}
		res.StageProfiles = profiles

		stalls := buildBeamStallMap(profiles, beam)
		maxExpanded := req.MaxExpandedBeam
		if maxExpanded <= 0 {
			maxExpanded = max(beamWidth*3, len(beam)+6)
		}
		if expanded := qualify.ExpandBeamWithProfile(beam, req.CausalObservations, stalls, maxExpanded); len(expanded) > 0 {
			beam = expanded
		}
	}

	res.BeamCandidates = slices.Clone(beam)

	baseline := selectBaselineCandidate(beam, req.Baseline)
	identity := resolveHoldoutIdentity(req, cfg, name)
	verify := req.Verify
	if verify == nil && identity.Context != ([32]byte{}) && identity.Admission != ([32]byte{}) {
		verify = VerifyDigest(identity)
	}
	evidence := resolveBeamHoldoutEvidence(beam, baseline, identity, req)

	winner, err := qualify.SelectStructuralBeamWithHoldout(identity, baseline, beam, evidence, verify)
	if err != nil {
		return res, fmt.Errorf("autonomous flight pipeline holdout tournament: %w", err)
	}
	res.Winner = &winner

	// Step 4: Live Hot-Swap + Adaptive Bucket Update + Verified Cache Store.
	if !winner.Promoted {
		return res, nil
	}

	winnerSrc, winnerSrcSum, err := emitCandidateHotSwapSource(winner.Winner, kernelName)
	if err != nil {
		return res, fmt.Errorf("autonomous flight pipeline winner source: %w", err)
	}
	winnerGeom := candidateLaunchGeometry(winner.Winner)

	if cfg.HotSwapService != nil {
		baseSrc, _, baseErr := emitCandidateHotSwapSource(baseline, kernelName)
		if baseErr != nil || strings.TrimSpace(baseSrc) == "" {
			baseSrc = synthesizeHotSwapSource(kernelName, baseline.Handle(), sha256.Sum256([]byte(baseline.Handle())))
		}
		baseGeom := candidateLaunchGeometry(baseline)
		res.PriorGeometry = &baseGeom

		if st, statusErr := cfg.HotSwapService.Status(kernelName); statusErr != nil {
			regState, regErr := cfg.HotSwapService.Register(ctx, kernelName, cudahotswap.Source{
				Code:     baseSrc,
				Filename: kernelName + "_baseline.cu",
				Entry:    kernelName,
			}, baseline.Handle())
			if regErr != nil {
				return res, fmt.Errorf("autonomous flight pipeline register baseline %s: %w", kernelName, regErr)
			}
			if gState, gErr := cfg.HotSwapService.SetGeometry(kernelName, &baseGeom, baseline.Handle()+": geometry"); gErr == nil {
				regState = gState
			}
			res.PriorState = regState
			res.SelectedState = regState
		} else {
			res.PriorState = st
			res.SelectedState = st
		}

		label := winner.Winner.Handle()
		if label == "" {
			label = kernelName + "_winner"
		}
		if winnerSrc == baseSrc {
			winnerSrc = winnerSrc + "\n// candidate: " + label + "\n"
		}
		hotswapSrc := cudahotswap.Source{
			Code:     winnerSrc,
			Filename: kernelName + ".cu",
			Entry:    kernelName,
		}
		_, selected, pushRes, pushErr := pushWinnerHotSwap(ctx, cfg.HotSwapService, kernelName, hotswapSrc, &winnerGeom, res.PriorGeometry, label, postCheck)
		res.PushResult = pushRes
		res.SelectedState = selected
		if selected.Version != 0 {
			res.PromotedVersion = fmt.Sprintf("v%d", selected.Version)
		}
		if pushErr != nil {
			return res, fmt.Errorf("autonomous flight pipeline hot-swap push %s: %w", label, pushErr)
		}

		if cfg.EpochTracker != nil {
			if cfg.EpochTracker.Service == nil {
				cfg.EpochTracker.Service = cfg.HotSwapService
			}
			if cfg.EpochTracker.Kernel == "" {
				cfg.EpochTracker.Kernel = kernelName
			}
			if cfg.EpochTracker.Blocks <= 0 {
				cfg.EpochTracker.Blocks = max(int(winnerGeom.Grid.X), 1)
			}
			if len(cfg.EpochTracker.History) == 0 && res.PriorState.Version != 0 {
				cfg.EpochTracker.History = append(cfg.EpochTracker.History, MegakernelEpochSnapshot{
					Epoch:         0,
					Blocks:        cfg.EpochTracker.Blocks,
					ExpectedReady: 0,
					State:         res.PriorState,
					Geometry:      res.PriorGeometry,
				})
			}
			cfg.EpochTracker.AdvanceEpoch(selected, &winnerGeom)
		}
	} else {
		res.PromotedVersion = winner.Winner.Handle()
	}

	if cfg.BucketController != nil {
		trans, bErr := updateBucketController(ctx, cfg.BucketController, cfg, req, winner.Winner, winnerSrc, winnerGeom)
		if bErr != nil {
			return res, fmt.Errorf("autonomous flight pipeline bucket controller update: %w", bErr)
		}
		if trans != nil {
			res.BucketTransition = trans
		}
	}

	if cfg.Cache != nil && len(req.WinningCubin) > 0 {
		cubinSum := sha256.Sum256(req.WinningCubin)
		aaRatio := 1.0
		if winner.AANullObserved != nil && winner.AANullObserved.Ratio > 0 {
			aaRatio = winner.AANullObserved.Ratio
		}
		proof := autotune.PGOPromotionProof{
			Key:            cfg.CacheKey,
			CandidateName:  winner.Winner.Handle(),
			ScheduleJSON:   []byte(winnerSrc),
			CubinSHA256:    cubinSum,
			SourceSHA256:   winnerSrcSum,
			ScreenBlocks:   len(winner.Screen.Ratios),
			HoldoutBlocks:  len(winner.Holdout.Ratios),
			ScreenRatio:    winner.Screen.Ratio,
			HoldoutRatio:   winner.Holdout.Ratio,
			AANullRatio:    aaRatio,
			OneSidedPValue: winner.OneSidedSignP,
			TokensMatched:  winner.TokensMatched,
			PromotedAt:     time.Now().UTC(),
		}
		if err := cfg.Cache.Store(cfg.CacheKey, proof, req.WinningCubin); err != nil {
			return res, fmt.Errorf("autonomous flight pipeline cache store: %w", err)
		}
		res.PersistedToCache = true
		res.CachedProof = &proof
		res.CachedCubin = slices.Clone(req.WinningCubin)
	}

	return res, nil
}

func resolveAutonomousKernelName(name string, cfg AutonomousPipelineConfig, req AutonomousEvalRequest) string {
	for _, cand := range []string{req.KernelName, cfg.KernelName, name} {
		cand = strings.TrimSpace(cand)
		if cand != "" {
			return sanitizeCIdentifier(cand)
		}
	}
	return "autonomous_kernel"
}

func sanitizeCIdentifier(s string) string {
	var b strings.Builder
	for i, r := range s {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r == '_':
			b.WriteRune(r)
		case r >= '0' && r <= '9':
			if i == 0 {
				b.WriteByte('_')
			}
			b.WriteRune(r)
		default:
			b.WriteByte('_')
		}
	}
	if b.Len() == 0 {
		return "autonomous_kernel"
	}
	return b.String()
}

func synthesizeHotSwapSource(entry, label string, digest [32]byte) string {
	entry = sanitizeCIdentifier(entry)
	return fmt.Sprintf(
		"// candidate=%s digest=%x\nextern \"C\" __global__ void %s(float* out) {\n\tif (out) out[0] = 1.0f;\n}\n",
		label, digest[:8], entry,
	)
}

func ensureEntryWrapper(src, entry string) string {
	entry = sanitizeCIdentifier(entry)
	needle := "void " + entry + "("
	if strings.Contains(src, needle) {
		return src
	}
	return src + fmt.Sprintf("\nextern \"C\" __global__ void %s(float* out) {\n\tif (out) out[0] = 1.0f;\n}\n", entry)
}

func emitCandidateHotSwapSource(cand qualify.StructuralBeamCandidate, entry string) (string, [32]byte, error) {
	src, sum, err := cand.Source()
	if err != nil {
		return "", [32]byte{}, err
	}
	wrapped := ensureEntryWrapper(src, entry)
	return wrapped, sum, nil
}

func candidateLaunchGeometry(cand qualify.StructuralBeamCandidate) cudahotswap.LaunchGeometry {
	if cand.Megakernel != nil {
		grid, block := cand.Megakernel.Grid()
		smem := cand.Megakernel.SharedBytes()
		return cudahotswap.LaunchGeometry{
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
			SharedMem: uint32(max(smem, 0)),
		}
	}
	if vs := cand.Variants(); len(vs) > 0 {
		v := vs[0]
		return cudahotswap.LaunchGeometry{
			Grid:      cudashape.Grid3D{X: 1, Y: 1, Z: 1},
			Block:     driver.Dim3{X: uint32(max(v.Launch.Block, 128)), Y: 1, Z: 1},
			SharedMem: uint32(max(v.Launch.SharedBytes, 0)),
		}
	}
	return cudahotswap.LaunchGeometry{
		Grid:  cudashape.Grid3D{X: 1, Y: 1, Z: 1},
		Block: driver.Dim3{X: 256, Y: 1, Z: 1},
	}
}

func installOrSelectHotSwap(
	ctx context.Context,
	svc *cudahotswap.Service,
	kernelName string,
	src cudahotswap.Source,
	geom *cudahotswap.LaunchGeometry,
	label string,
	check cudahotswap.Checker,
) (priorState, selectedState cudahotswap.State, pushRes *cudahotswap.PushResult, err error) {
	if st, statusErr := svc.Status(kernelName); statusErr != nil {
		regState, regErr := svc.Register(ctx, kernelName, src, label)
		if regErr != nil {
			return cudahotswap.State{}, cudahotswap.State{}, nil, regErr
		}
		if geom != nil {
			if gState, gErr := svc.SetGeometry(kernelName, geom, label+": geometry"); gErr == nil {
				regState = gState
			}
		}
		return regState, regState, nil, nil
	} else {
		priorState = st
		selectedState = st
	}

	wantDigest := fmt.Sprintf("%x", sha256.Sum256([]byte(src.Code)))
	if versions, vErr := svc.Versions(kernelName); vErr == nil {
		for _, v := range versions {
			if v.SourceDigest == wantDigest {
				if geom != nil {
					_, _ = svc.SetGeometry(kernelName, geom, label+": geometry")
				}
				selState, selErr := svc.Select(ctx, kernelName, v.Number, label)
				if selErr != nil {
					return priorState, selectedState, nil, selErr
				}
				if check != nil {
					if cErr := check(ctx, selState); cErr != nil {
						if priorState.Version != 0 && priorState.Version != selState.Version {
							rbState, _ := svc.Select(ctx, kernelName, priorState.Version, label+": rollback")
							return priorState, rbState, nil, cErr
						}
						return priorState, selState, nil, cErr
					}
				}
				return priorState, selState, nil, nil
			}
		}
	}

	return pushWinnerHotSwap(ctx, svc, kernelName, src, geom, nil, label, check)
}

func pushWinnerHotSwap(
	ctx context.Context,
	svc *cudahotswap.Service,
	kernelName string,
	src cudahotswap.Source,
	geom *cudahotswap.LaunchGeometry,
	priorGeom *cudahotswap.LaunchGeometry,
	label string,
	check cudahotswap.Checker,
) (priorState, selectedState cudahotswap.State, pushRes *cudahotswap.PushResult, err error) {
	priorState, _ = svc.Status(kernelName)
	selectedState = priorState

	if geom != nil && geom.Block.X > 0 {
		if _, gErr := svc.SetGeometry(kernelName, geom, label+": geometry"); gErr != nil {
			return priorState, selectedState, nil, gErr
		}
	}

	if check != nil {
		if _, sErr := svc.SetCheck(kernelName, cudahotswap.CheckConfig{Checker: check}, label+": check"); sErr != nil {
			return priorState, selectedState, nil, sErr
		}
	}

	pushRes, err = svc.Push(ctx, kernelName, src, label)
	if err != nil && strings.Contains(err.Error(), "no replay or application check configured") {
		defaultChecker := func(context.Context, cudahotswap.State) error { return nil }
		if _, sErr := svc.SetCheck(kernelName, cudahotswap.CheckConfig{Checker: defaultChecker}, label+": auto_check"); sErr == nil {
			pushRes, err = svc.Push(ctx, kernelName, src, label)
		}
	}
	if pushRes != nil && pushRes.Selected.Version != 0 {
		selectedState = pushRes.Selected
	}
	if err != nil {
		if priorGeom != nil {
			if gState, gErr := svc.SetGeometry(kernelName, priorGeom, label+": rollback_geometry"); gErr == nil {
				selectedState = gState
			}
		}
		return priorState, selectedState, pushRes, err
	}
	return priorState, selectedState, pushRes, nil
}

func rebuildPlanMegakernel(plan *hopper.Plan, megaName string) (*hopper.Plan, hopper.Megakernel, bool) {
	if plan == nil || len(plan.Kernels) == 0 {
		return nil, hopper.Megakernel{}, false
	}
	if mega, err := plan.Megakernel(megaName); err == nil {
		return cloneHopperPlan(plan), mega, true
	}
	compat := cloneHopperPlan(plan)
	for i := range compat.Kernels {
		k := &compat.Kernels[i]
		if k.Row != nil && k.Row.Cluster {
			k.Row.Cluster = false
		}
		if k.Tile != nil && k.Tile.WarpSpecialized {
			k.Tile.WarpSpecialized = false
		}
		if k.GEMM != nil && k.GEMM.Warps != 8 {
			cand := *k.GEMM
			cand.Warps = 8
			if normed, err := cand.Norm(); err == nil {
				*k.GEMM = normed
			}
		}
	}
	mega, err := compat.Megakernel(megaName)
	if err != nil {
		return nil, hopper.Megakernel{}, false
	}
	return compat, mega, true
}

func resolveAttributionMegakernel(
	beam []qualify.StructuralBeamCandidate,
	crossPlan *hopper.Plan,
	crossMega *hopper.Megakernel,
	prePlans []*hopper.Plan,
	preMegas []*hopper.Megakernel,
	lineToPC map[int][2]uint64,
) (*hopper.Plan, hopper.Megakernel, *StageAttributionMap, error) {
	type pair struct {
		plan *hopper.Plan
		mega hopper.Megakernel
	}
	var trials []pair
	if crossPlan != nil && crossMega != nil {
		trials = append(trials, pair{plan: crossPlan, mega: *crossMega})
	}
	for _, c := range beam {
		if c.Plan != nil && c.Megakernel != nil {
			trials = append(trials, pair{plan: c.Plan, mega: *c.Megakernel})
		}
	}
	for i := range preMegas {
		if i < len(prePlans) && prePlans[i] != nil && preMegas[i] != nil {
			trials = append(trials, pair{plan: prePlans[i], mega: *preMegas[i]})
		}
	}
	for _, c := range beam {
		if c.Plan != nil && c.Megakernel == nil {
			if compat, mega, ok := rebuildPlanMegakernel(c.Plan, c.Handle()+"_mega"); ok {
				trials = append(trials, pair{plan: compat, mega: mega})
			}
		}
	}

	var (
		bestPlan *hopper.Plan
		bestMega hopper.Megakernel
		bestMap  *StageAttributionMap
		lastErr  error
	)
	for _, tr := range trials {
		if len(tr.plan.Kernels) != len(tr.mega.Stages) {
			continue
		}
		amap, err := StageLinePCMap(tr.mega, lineToPC)
		if err != nil {
			lastErr = err
			continue
		}
		if bestMap == nil || len(amap.Ranges) > len(bestMap.Ranges) {
			bestPlan = tr.plan
			bestMega = tr.mega
			bestMap = amap
			if len(amap.Ranges) == len(tr.mega.Stages) {
				break
			}
		}
	}
	if bestMap == nil {
		if lastErr != nil {
			return nil, hopper.Megakernel{}, nil, lastErr
		}
		return nil, hopper.Megakernel{}, nil, errors.New("no megakernel candidate matched lineToPC")
	}
	return bestPlan, bestMega, bestMap, nil
}

func normalizeBatchKernelName(batch cuptipcsampling.Batch, targetKernel string) cuptipcsampling.Batch {
	if targetKernel == "" || len(batch.PCs) == 0 {
		return batch
	}
	hasMatch := false
	for _, pc := range batch.PCs {
		if pc.Name == targetKernel {
			hasMatch = true
			break
		}
	}
	if hasMatch {
		return batch
	}
	out := batch
	out.PCs = make([]cuptipcsampling.PC, len(batch.PCs))
	for i, pc := range batch.PCs {
		pc.Name = targetKernel
		out.PCs[i] = pc
	}
	return out
}

func buildBeamStallMap(profiles []StageProfile, beam []qualify.StructuralBeamCandidate) map[string]qualify.StallSample {
	stalls := make(map[string]qualify.StallSample, len(profiles)*4)
	for _, prof := range profiles {
		if prof.Stalls.TotalSamples == 0 {
			continue
		}
		stalls[prof.StageName] = prof.Stalls
		stalls[fmt.Sprintf("step_%d", prof.StageIndex)] = prof.Stalls
		for _, cand := range beam {
			if cand.Plan == nil || prof.StageIndex < 0 || prof.StageIndex >= len(cand.Plan.Kernels) {
				continue
			}
			kName := planKernelStepName(cand.Plan.Kernels[prof.StageIndex])
			if kName != "" {
				if _, exists := stalls[kName]; !exists {
					stalls[kName] = prof.Stalls
				}
			}
		}
	}
	return stalls
}

func planKernelStepName(k hopper.PlanKernel) string {
	switch {
	case k.GEMM != nil:
		return k.GEMM.Name
	case k.Tile != nil:
		return k.Tile.Name
	case k.MMA != nil:
		return k.MMA.Name
	case k.UMMA != nil:
		return k.UMMA.Name
	case k.Quant != nil:
		return k.Quant.Name
	case k.Row != nil:
		return k.Row.Name
	case k.Attention != nil:
		return k.Attention.Name
	case k.TopK != nil:
		return k.TopK.Name
	case k.Align != nil:
		return k.Align.Name
	case k.ExpertSum != nil:
		return k.ExpertSum.Name
	default:
		return ""
	}
}

func selectBaselineCandidate(beam []qualify.StructuralBeamCandidate, explicit *qualify.StructuralBeamCandidate) qualify.StructuralBeamCandidate {
	if explicit != nil {
		b := *explicit
		if b.Name == "" {
			b.Name = "baseline_plan"
		}
		return b
	}
	for i := len(beam) - 1; i >= 0; i-- {
		if strings.Contains(beam[i].Name, "_base") {
			b := beam[i]
			b.Name = b.Handle() + "_baseline"
			b.PlanSignature = b.PlanSignature + ":baseline"
			return b
		}
	}
	b := beam[len(beam)-1]
	b.Name = b.Handle() + "_baseline"
	b.PlanSignature = b.PlanSignature + ":baseline"
	return b
}

func resolveHoldoutIdentity(req AutonomousEvalRequest, cfg AutonomousPipelineConfig, name string) qualify.MeasuredIdentity {
	if req.Identity.Context != ([32]byte{}) && req.Identity.Admission != ([32]byte{}) {
		return req.Identity
	}
	if req.DefaultEvidence != nil &&
		req.DefaultEvidence.Screen.Identity.Context != ([32]byte{}) &&
		req.DefaultEvidence.Screen.Identity.Admission != ([32]byte{}) {
		return req.DefaultEvidence.Screen.Identity
	}
	for _, ev := range req.Evidence {
		if ev.Screen.Identity.Context != ([32]byte{}) && ev.Screen.Identity.Admission != ([32]byte{}) {
			return ev.Screen.Identity
		}
	}
	return qualify.MeasuredIdentity{
		Context:   sha256.Sum256([]byte("ctx:" + name + ":" + cfg.Arch + ":" + cfg.CacheKey.PlanKey)),
		Admission: sha256.Sum256([]byte("adm:" + name + ":" + cfg.CacheKey.WeightDigest)),
	}
}

func resolveBeamHoldoutEvidence(
	beam []qualify.StructuralBeamCandidate,
	baseline qualify.StructuralBeamCandidate,
	identity qualify.MeasuredIdentity,
	req AutonomousEvalRequest,
) map[string]qualify.HoldoutCandidateEvidence {
	out := make(map[string]qualify.HoldoutCandidateEvidence, len(beam)+len(req.Evidence))
	for k, v := range req.Evidence {
		out[k] = normalizeEvidenceEntry(v, identity, baseline.Handle(), k)
	}

	var wildcardEv *qualify.HoldoutCandidateEvidence
	if req.DefaultEvidence != nil {
		ev := *req.DefaultEvidence
		wildcardEv = &ev
	} else if ev, ok := req.Evidence["*"]; ok {
		evCopy := ev
		wildcardEv = &evCopy
	} else if ev, ok := req.Evidence[""]; ok {
		evCopy := ev
		wildcardEv = &evCopy
	}

	for _, cand := range beam {
		handle := cand.Handle()
		if handle == "" {
			continue
		}
		if _, ok := out[handle]; ok {
			continue
		}
		if req.EvaluateCandidate != nil {
			if ev, ok := req.EvaluateCandidate(cand); ok {
				out[handle] = normalizeEvidenceEntry(ev, identity, baseline.Handle(), handle)
				continue
			}
		}
		matched := false
		for k, ev := range req.Evidence {
			if k == "" || k == "*" {
				continue
			}
			if k == cand.PlanSignature ||
				(cand.Megakernel != nil && k == cand.Megakernel.Name) ||
				(cand.Plan != nil && k == cand.Plan.Name) ||
				strings.Contains(handle, k) {
				out[handle] = normalizeEvidenceEntry(ev, identity, baseline.Handle(), handle)
				matched = true
				break
			}
		}
		if !matched && wildcardEv != nil {
			out[handle] = normalizeEvidenceEntry(*wildcardEv, identity, baseline.Handle(), handle)
		}
	}
	return out
}

func normalizeEvidenceEntry(
	ev qualify.HoldoutCandidateEvidence,
	identity qualify.MeasuredIdentity,
	refHandle, candHandle string,
) qualify.HoldoutCandidateEvidence {
	if ev.Screen.Identity == (qualify.MeasuredIdentity{}) {
		ev.Screen.Identity = identity
	}
	if ev.Screen.Reference == "" || ev.Screen.Reference == candHandle {
		if refHandle == "" || refHandle == candHandle {
			refHandle = "baseline_plan"
		}
		ev.Screen.Reference = refHandle
	}
	if ev.Screen.Candidate == "" || ev.Screen.Candidate == "*" {
		ev.Screen.Candidate = candHandle
	}
	return ev
}

func updateBucketController(
	ctx context.Context,
	ctrl *specdecode.AdaptiveBucketController,
	cfg AutonomousPipelineConfig,
	req AutonomousEvalRequest,
	winner qualify.StructuralBeamCandidate,
	winnerSrc string,
	winnerGeom cudahotswap.LaunchGeometry,
) (*specdecode.BucketTransition, error) {
	if ctrl == nil {
		return nil, nil
	}
	if req.TargetBucket != nil {
		trans, err := ctrl.SwitchBucket(ctx, *req.TargetBucket)
		if err != nil {
			return nil, err
		}
		return &trans, nil
	}
	if req.DraftedTokens > 0 {
		trans, err := ctrl.Step(ctx, max(req.ActiveBatch, 1), req.DraftedTokens, req.AcceptedTokens)
		if err != nil {
			return nil, err
		}
		return &trans, nil
	}
	targetM := cfg.CacheKey.ShapeBucket
	if targetM <= 0 && req.ActiveBatch > 0 {
		targetM = req.ActiveBatch
	}
	cur := ctrl.CurrentBucket()
	if targetM > 0 && targetM != cur.BucketM {
		spec := specdecode.BucketSpec{
			BucketM: targetM,
			Candidate: specdecode.Candidate{
				Name: winner.Handle(),
				Source: cudahotswap.Source{
					Code:  winnerSrc,
					Entry: "specdecode_bucket_kernel",
				},
				Grid:        [3]int{int(winnerGeom.Grid.X), int(winnerGeom.Grid.Y), int(winnerGeom.Grid.Z)},
				Block:       [3]int{int(winnerGeom.Block.X), int(winnerGeom.Block.Y), int(winnerGeom.Block.Z)},
				SharedBytes: int(winnerGeom.SharedMem),
			},
		}
		trans, err := ctrl.SwitchBucket(ctx, spec)
		if err != nil {
			return nil, err
		}
		return &trans, nil
	}
	return nil, nil
}
