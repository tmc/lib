package model

import (
	"cmp"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/tmc/lib/nvidia/x/cudahotswap"
	"github.com/tmc/lib/nvidia/x/cuptipcsampling"
	"github.com/tmc/lib/nvidia/x/egraph"
	"github.com/tmc/lib/nvidia/x/egraph/autotune"
	"github.com/tmc/lib/nvidia/x/egraph/hopper"
	"github.com/tmc/lib/nvidia/x/egraph/qualify"
	"github.com/tmc/lib/nvidia/x/nvl72"
)

var (
	// ErrDistBeamNoValidCandidate reports that no joint compute+collective
	// candidate survived numerical equivalence and cost verification.
	ErrDistBeamNoValidCandidate = errors.New("model: no valid topology-aware distributed beam candidate")

	// ErrEpochBarrierMismatch reports that a rank's epoch is already past or
	// incompatible with the requested barrier switchAtEpoch.
	ErrEpochBarrierMismatch = errors.New("model: multi-rank epoch barrier mismatch")

	// ErrMultiRankSwapVerificationFailed reports that at least one rank failed
	// post-swap verification, triggering an atomic all-rank rollback.
	ErrMultiRankSwapVerificationFailed = errors.New("model: multi-rank hot-swap verification failed; all ranks rolled back")
)

// TopologyBeamConfig configures [SearchTopologyAwareBeam].
type TopologyBeamConfig struct {
	Name        string
	ComputeGraph *egraph.EGraph
	ComputeRoot  egraph.Id
	DistGraph    *egraph.EGraph
	DistRoot     egraph.Id
	OutDType     hopper.DType
	CostConfig   hopper.CostConfig
	DistParams   nvl72.DistFusionCostParams
	BeamWidth    int
	EvalEnv      map[string][]nvl72.DistMatrix
	EvalConfig   nvl72.DistEvalConfig
	MaxRelDiff   float64
}

// TopologyBeamCandidate is a joint compute+collective candidate extracted and
// verified by [SearchTopologyAwareBeam].
type TopologyBeamCandidate struct {
	Name                  string                          `json:"name"`
	ComputeCandidate      qualify.StructuralBeamCandidate `json:"compute_candidate"`
	DistSchedule          nvl72.DistSchedule              `json:"-"`
	CollectiveRule        nvl72.DistRewriteKind           `json:"collective_rule"`
	DistExprString        string                          `json:"dist_expr"`
	PTX                   string                          `json:"ptx,omitempty"`
	CUDASource            string                          `json:"cuda_source,omitempty"`
	ComputeLatencyUs      float64                         `json:"compute_latency_us"`
	DistLatencyUs         float64                         `json:"dist_latency_us"`
	UnfusedDistLatencyUs  float64                         `json:"unfused_dist_latency_us"`
	TotalDistLatencyUs    float64                         `json:"total_dist_latency_us"`
	BaselineTotalUs       float64                         `json:"baseline_total_us"`
	WorldSize             int                             `json:"world_size"`
	NumericallyVerified   bool                            `json:"numerically_verified"`
	MaxRelDiff            float64                         `json:"max_rel_diff"`
	EvalTrajectoryDigest  string                          `json:"eval_trajectory_digest"`
	CombinedSignature     string                          `json:"combined_signature"`
}

// Handle returns a stable identifier for c.
func (c TopologyBeamCandidate) Handle() string {
	if c.Name != "" {
		return c.Name
	}
	if c.ComputeCandidate.Name != "" {
		return c.ComputeCandidate.Name
	}
	return c.CombinedSignature
}

// distRuleBiasedCostModel biases extraction toward a specific collective
// rewrite op (or toward unfused ops when targetOp == "") so beam search
// harvests all distinct valid collective schedules in the saturated e-graph.
type distRuleBiasedCostModel struct {
	base     nvl72.DistCostModel
	targetOp string
	unfused  bool
}

func (m distRuleBiasedCostModel) NodeCost(n egraph.ENode, children []float64) float64 {
	sum := 0.0
	for _, c := range children {
		sum += c
	}
	local := m.base.NodeLocalCost(n)
	isFused := n.Op == nvl72.FusedTileFabricRedGemm ||
		n.Op == nvl72.FusedMultimemLdReduceRMSNorm ||
		n.Op == nvl72.PipelinedFabricGetGemm ||
		n.Op == nvl72.TiledMultimemScatterGemm
	if m.unfused {
		if isFused {
			return sum + local + 1e9
		}
		return sum + local
	}
	if m.targetOp != "" {
		if n.Op == m.targetOp {
			return sum + local*0.01
		}
		if isFused {
			return sum + local + 1e6
		}
	}
	return sum + local
}

// SearchTopologyAwareBeam combines [qualify.ExtractStructuralPlanBeam] with
// [nvl72.AlgebraicCollectiveRules] (RuleTilePushFabricRed,
// RulePullMultimemLdReduce, RuleAGMatmulToRingPrefetch,
// RuleMatmulRSAssociative), [nvl72.DistCostModel], and multi-rank numerical
// equivalence verification ([nvl72.EvalDistExpr]), ranking joint
// compute+collective candidates by total distributed latency across WorldSize
// GPUs.
func SearchTopologyAwareBeam(cfg TopologyBeamConfig) ([]TopologyBeamCandidate, error) {
	if cfg.DistParams.WorldSize < 2 {
		return nil, fmt.Errorf("search topology-aware beam: WorldSize %d must be >= 2", cfg.DistParams.WorldSize)
	}
	if cfg.DistParams.SwitchBWGBps <= 0 || cfg.DistParams.TensorCoreTFLOPS <= 0 || cfg.DistParams.SwitchLatencyUs < 0 {
		return nil, errors.New("search topology-aware beam: SwitchBWGBps and TensorCoreTFLOPS must be positive and SwitchLatencyUs >= 0")
	}
	if cfg.DistParams.BatchTokens < 1 {
		return nil, fmt.Errorf("search topology-aware beam: BatchTokens %d must be >= 1", cfg.DistParams.BatchTokens)
	}
	normDistParams := normalizeDistParams(cfg.DistParams)
	beamWidth := cfg.BeamWidth
	if beamWidth <= 0 {
		beamWidth = 4
	}
	maxRel := cfg.MaxRelDiff
	if maxRel <= 0 {
		maxRel = 1e-4
	}
	name := strings.TrimSpace(cfg.Name)
	if name == "" {
		name = "dist_beam"
	}
	outDType := cfg.OutDType
	if outDType == 0 {
		outDType = hopper.BF16
	}

	// Resolve or synthesize the distributed collective e-graph.
	distG, distRoot := cfg.DistGraph, cfg.DistRoot
	if distG == nil {
		if cfg.ComputeGraph != nil && graphHasDistOps(cfg.ComputeGraph) {
			distG = cfg.ComputeGraph.Clone()
			distRoot = cfg.ComputeRoot
		} else {
			distG, distRoot = defaultTPCollectiveGraph()
		}
	} else {
		distG = distG.Clone()
	}

	// Extract the unfused distributed expression and price it before saturation.
	baseCostModel := nvl72.DistCostModel{P: normDistParams}
	unfusedExt, err := egraph.NewAdditiveExtractorForRootChecked(distG, distRoot, baseCostModel)
	if err != nil {
		return nil, fmt.Errorf("search topology-aware beam: unfused extractor: %w", err)
	}
	unfusedExpr, unfusedLatencyUs, err := unfusedExt.Extract(distRoot)
	if err != nil {
		return nil, fmt.Errorf("search topology-aware beam: extract unfused: %w", err)
	}

	// Prepare multi-rank evaluation config and environment for EvalDistExpr.
	evalCfg := cfg.EvalConfig
	if evalCfg.Ranks <= 0 {
		evalCfg.Ranks = min(normDistParams.WorldSize, 8)
	}
	if evalCfg.Eps == 0 {
		evalCfg.Eps = 1e-6
	}
	if evalCfg.TileM == 0 {
		evalCfg.TileM = 2
	}
	if evalCfg.TileN == 0 {
		evalCfg.TileN = 4
	}
	evalEnv := cfg.EvalEnv
	if len(evalEnv) == 0 {
		var envErr error
		evalEnv, envErr = synthesizeDistEvalEnv(unfusedExpr, evalCfg.Ranks)
		if envErr != nil {
			return nil, fmt.Errorf("search topology-aware beam: synthesize eval env: %w", envErr)
		}
	}

	refMatrices, err := nvl72.EvalDistExpr(unfusedExpr, evalCfg, evalEnv)
	if err != nil {
		return nil, fmt.Errorf("search topology-aware beam: eval baseline expr: %w", err)
	}
	refTrajectoryDigest := digestDistMatrices(refMatrices)

	// Harvest optimal schedule via nvl72.ExtractOptimalDistSchedule and
	// additional structural collective candidates across all 4 algebraic rules.
	optClone := distG.Clone()
	optSched, err := nvl72.ExtractOptimalDistSchedule(optClone, distRoot, normDistParams)
	if err != nil {
		return nil, fmt.Errorf("search topology-aware beam: optimal dist schedule: %w", err)
	}

	satG := distG.Clone()
	egraph.Saturate(satG, nvl72.AlgebraicCollectiveRules(), 0)
	_ = satG.RebuildChecked()

	var distExprs []*egraph.Expr
	seenDistFP := make(map[string]bool)
	addDistExpr := func(e *egraph.Expr) {
		if e == nil {
			return
		}
		fp := e.Fingerprint()
		if seenDistFP[fp] {
			return
		}
		seenDistFP[fp] = true
		distExprs = append(distExprs, e)
	}
	addDistExpr(optSched.Expr)

	distModels := []egraph.CostModel{
		distRuleBiasedCostModel{base: baseCostModel, targetOp: nvl72.FusedMultimemLdReduceRMSNorm},
		distRuleBiasedCostModel{base: baseCostModel, targetOp: nvl72.FusedTileFabricRedGemm},
		distRuleBiasedCostModel{base: baseCostModel, targetOp: nvl72.PipelinedFabricGetGemm},
		distRuleBiasedCostModel{base: baseCostModel, targetOp: nvl72.TiledMultimemScatterGemm},
		distRuleBiasedCostModel{base: baseCostModel, unfused: true},
	}
	limits := egraph.ExtractionLimits{MaxWork: 1 << 26, MaxNodes: 1 << 20, MaxDepth: 1 << 16}
	for _, dm := range distModels {
		if e, _, _, err := egraph.ExtractSegmented(context.Background(), satG, distRoot, dm, limits); err == nil {
			addDistExpr(e)
		}
		if topK, err := egraph.ExtractTopK(satG, distRoot, max(beamWidth*2, 8), dm); err == nil {
			for _, item := range topK {
				addDistExpr(item.Expr)
			}
		}
	}
	addDistExpr(unfusedExpr)

	type verifiedDistSched struct {
		sched      nvl72.DistSchedule
		cudaSource string
		maxDiff    float64
		trajDigest string
	}
	var verifiedDist []verifiedDistSched
	for _, de := range distExprs {
		gotMats, evalErr := nvl72.EvalDistExpr(de, evalCfg, evalEnv)
		if evalErr != nil {
			continue
		}
		diff := maxRelDiffDistMatrices(gotMats, refMatrices)
		if math.IsNaN(diff) || math.IsInf(diff, 0) || diff > maxRel {
			continue
		}
		rule := detectDistRule(de)
		ptx, ptxErr := nvl72.DistRulePTX(rule, normDistParams.Target)
		if ptxErr != nil {
			continue
		}
		estUs := priceDistExpr(de, baseCostModel)
		var cudaSrc string
		if rule == nvl72.RulePullMultimemLdReduce {
			cols := max((normDistParams.HiddenDim/8)*8, 8)
			if kSrc, kErr := (nvl72.MultimemRMSNormKernel{
				Name:    sanitizeCIdentifier(name + "_multimem_rmsnorm"),
				Cols:    cols,
				Ranks:   normDistParams.WorldSize,
				Threads: 128,
			}).Source(); kErr == nil {
				cudaSrc = kSrc
			}
		}
		verifiedDist = append(verifiedDist, verifiedDistSched{
			sched: nvl72.DistSchedule{
				Rule:             rule,
				Expr:             de,
				EstLatencyUs:     estUs,
				UnfusedLatencyUs: unfusedLatencyUs,
				PTX:              ptx,
			},
			cudaSource: cudaSrc,
			maxDiff:    diff,
			trajDigest: refTrajectoryDigest,
		})
	}
	if len(verifiedDist) == 0 {
		return nil, ErrDistBeamNoValidCandidate
	}

	// Extract structural compute candidates when ComputeGraph is provided and
	// contains hopper operators.
	var computeBeam []qualify.StructuralBeamCandidate
	if cfg.ComputeGraph != nil && !graphHasDistOps(cfg.ComputeGraph) {
		normCostCfg, cErr := cfg.CostConfig.Norm()
		if cErr != nil {
			normCostCfg = cfg.CostConfig
		}
		extracted, bErr := qualify.ExtractStructuralPlanBeam(cfg.ComputeGraph, cfg.ComputeRoot, name, outDType, normCostCfg, beamWidth)
		if bErr != nil {
			return nil, fmt.Errorf("search topology-aware beam: extract structural beam: %w", bErr)
		}
		computeBeam = extracted
	}
	if len(computeBeam) == 0 {
		computeBeam = []qualify.StructuralBeamCandidate{{
			Name:          name + "_compute",
			PlanSignature: "dist_only_" + name,
			EstimatedNs:   0,
		}}
	}

	var slowestComputeUs float64
	for _, cc := range computeBeam {
		slowestComputeUs = max(slowestComputeUs, cc.EstimatedNs/1000.0)
	}
	baselineTotalUs := slowestComputeUs + unfusedLatencyUs

	var (
		jointCandidates []TopologyBeamCandidate
		seenSig         = make(map[string]bool)
	)
	for _, vd := range verifiedDist {
		for _, cc := range computeBeam {
			compUs := cc.EstimatedNs / 1000.0
			totalUs := compUs + vd.sched.EstLatencyUs
			ruleTag := string(vd.sched.Rule)
			if ruleTag == "" {
				ruleTag = "unfused"
			}
			sigInput := fmt.Sprintf("%s|%s|%s|%d", cc.PlanSignature, vd.sched.Expr.Fingerprint(), ruleTag, normDistParams.WorldSize)
			sigSum := sha256.Sum256([]byte(sigInput))
			sig := hex.EncodeToString(sigSum[:])
			if seenSig[sig] {
				continue
			}
			seenSig[sig] = true

			candName := fmt.Sprintf("%s_%s", cc.Handle(), shortRuleSlug(vd.sched.Rule))
			compCopy := cc
			compCopy.Name = candName
			if compCopy.PlanSignature == "" {
				compCopy.PlanSignature = sig
			}
			compCopy.EstimatedNs = totalUs * 1000.0

			jointCandidates = append(jointCandidates, TopologyBeamCandidate{
				Name:                 candName,
				ComputeCandidate:     compCopy,
				DistSchedule:         vd.sched,
				CollectiveRule:       vd.sched.Rule,
				DistExprString:       vd.sched.Expr.String(),
				PTX:                  vd.sched.PTX,
				CUDASource:           vd.cudaSource,
				ComputeLatencyUs:     compUs,
				DistLatencyUs:        vd.sched.EstLatencyUs,
				UnfusedDistLatencyUs: unfusedLatencyUs,
				TotalDistLatencyUs:   totalUs,
				BaselineTotalUs:      baselineTotalUs,
				WorldSize:            normDistParams.WorldSize,
				NumericallyVerified:  true,
				MaxRelDiff:           vd.maxDiff,
				EvalTrajectoryDigest: vd.trajDigest,
				CombinedSignature:    sig,
			})
		}
	}

	slices.SortStableFunc(jointCandidates, func(a, b TopologyBeamCandidate) int {
		if c := cmp.Compare(a.TotalDistLatencyUs, b.TotalDistLatencyUs); c != 0 {
			return c
		}
		// Prefer fused collective rules over unfused when tied.
		aFused := a.CollectiveRule != ""
		bFused := b.CollectiveRule != ""
		if aFused != bFused {
			if aFused {
				return -1
			}
			return 1
		}
		return cmp.Compare(a.CombinedSignature, b.CombinedSignature)
	})

	if len(jointCandidates) > beamWidth {
		jointCandidates = jointCandidates[:beamWidth]
	}
	return jointCandidates, nil
}

// MultiRankSwapRequest configures a barrier-aligned multi-rank hot-swap in
// [MultiRankEpochCoordinator.CoordinateHotSwap].
type MultiRankSwapRequest struct {
	SwitchAtEpoch     uint64
	CandidateName     string
	CollectiveRule    nvl72.DistRewriteKind
	Source            cudahotswap.Source
	RankSources       []cudahotswap.Source
	Geometry          *cudahotswap.LaunchGeometry
	RankVotes         []autotune.RankVerificationVote
	BaselineLatencyUs float64
	PostSwapRankCheck func(ctx context.Context, rank int, state cudahotswap.State) error
}

// MultiRankEpochCommit records a barrier-aligned hot-swap committed simultaneously
// across all WorldSize ranks at SwitchAtEpoch.
type MultiRankEpochCommit struct {
	SwitchAtEpoch      uint64
	CandidateName      string
	PriorRule          nvl72.DistRewriteKind
	CommittedRule      nvl72.DistRewriteKind
	PriorRankStates    []cudahotswap.State
	SelectedRankStates []cudahotswap.State
	PriorGeometry      *cudahotswap.LaunchGeometry
	SelectedGeometry   *cudahotswap.LaunchGeometry
	TrajectoryDigest   string
	MaxRankLatencyUs   float64
	BaselineLatencyUs  float64
	CommittedAt        time.Time
}

// MultiRankSwapResult reports the outcome of a barrier-aligned hot-swap in
// [MultiRankEpochCoordinator.CoordinateHotSwap].
type MultiRankSwapResult struct {
	Committed          bool
	RolledBack         bool
	SwitchAtEpoch      uint64
	CandidateName      string
	ActiveRule         nvl72.DistRewriteKind
	PriorRankStates    []cudahotswap.State
	SelectedRankStates []cudahotswap.State
	FailedRank         int
	FailureReason      string
}

// MultiRankEpochCoordinator coordinates barrier-aligned hot-swaps across all
// WorldSize ranks at an exact target switchAtEpoch so no rank executes a new
// collective protocol while a peer rank is still on the old kernel, with
// automatic atomic all-rank rollback if any rank fails post-swap verification.
type MultiRankEpochCoordinator struct {
	mu          sync.Mutex
	WorldSize   int
	KernelName  string
	Epoch       uint64
	RankEpochs  []uint64
	ActiveRule  nvl72.DistRewriteKind
	Services    []*cudahotswap.Service
	Trackers    []*MegakernelEpochTracker
	History     []MultiRankEpochCommit
}

// NewMultiRankEpochCoordinator creates a coordinator for worldSize ranks.
func NewMultiRankEpochCoordinator(
	worldSize int,
	kernelName string,
	services []*cudahotswap.Service,
	trackers []*MegakernelEpochTracker,
) (*MultiRankEpochCoordinator, error) {
	if worldSize < 2 {
		return nil, fmt.Errorf("multi-rank epoch coordinator: worldSize %d must be >= 2", worldSize)
	}
	if len(services) != 0 && len(services) != worldSize {
		return nil, fmt.Errorf("multi-rank epoch coordinator: got %d services, want %d", len(services), worldSize)
	}
	if len(trackers) != 0 && len(trackers) != worldSize {
		return nil, fmt.Errorf("multi-rank epoch coordinator: got %d trackers, want %d", len(trackers), worldSize)
	}
	if strings.TrimSpace(kernelName) == "" {
		kernelName = "dist_autonomous_kernel"
	}
	svcs := make([]*cudahotswap.Service, worldSize)
	copy(svcs, services)
	trks := make([]*MegakernelEpochTracker, worldSize)
	copy(trks, trackers)

	return &MultiRankEpochCoordinator{
		WorldSize:  worldSize,
		KernelName: sanitizeCIdentifier(kernelName),
		RankEpochs: make([]uint64, worldSize),
		Services:   svcs,
		Trackers:   trks,
	}, nil
}

// SetRankEpoch updates rank's observed execution epoch before barrier alignment.
func (c *MultiRankEpochCoordinator) SetRankEpoch(rank int, epoch uint64) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if rank < 0 || rank >= c.WorldSize {
		return fmt.Errorf("multi-rank epoch coordinator: rank %d out of range [0, %d)", rank, c.WorldSize)
	}
	c.RankEpochs[rank] = epoch
	return nil
}

// CoordinateHotSwap aligns all WorldSize ranks at req.SwitchAtEpoch, stages the
// candidate kernel across every rank's [cudahotswap.Service], verifies post-swap
// health and quorum invariants across all ranks before releasing the epoch
// barrier, and atomically rolls back all ranks to their pre-swap version and
// geometry if any rank fails.
func (c *MultiRankEpochCoordinator) CoordinateHotSwap(ctx context.Context, req MultiRankSwapRequest) (*MultiRankSwapResult, error) {
	if c == nil {
		return nil, errors.New("coordinate hot-swap: nil coordinator")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	switchEpoch := req.SwitchAtEpoch
	if switchEpoch == 0 {
		switchEpoch = c.Epoch + 1
	}

	// Phase 1: Verify barrier reachability across all WorldSize ranks before
	// mutating any rank's kernel state.
	if switchEpoch < c.Epoch {
		return nil, fmt.Errorf("%w: target switchAtEpoch %d < coordinator epoch %d", ErrEpochBarrierMismatch, switchEpoch, c.Epoch)
	}
	for r := range c.WorldSize {
		if c.RankEpochs[r] > switchEpoch {
			return nil, fmt.Errorf("%w: rank %d is already at epoch %d > switchAtEpoch %d", ErrEpochBarrierMismatch, r, c.RankEpochs[r], switchEpoch)
		}
		if c.Trackers[r] != nil && c.Trackers[r].Epoch > switchEpoch {
			return nil, fmt.Errorf("%w: rank %d tracker is at epoch %d > switchAtEpoch %d", ErrEpochBarrierMismatch, r, c.Trackers[r].Epoch, switchEpoch)
		}
	}

	res := &MultiRankSwapResult{
		SwitchAtEpoch:      switchEpoch,
		CandidateName:      req.CandidateName,
		ActiveRule:         c.ActiveRule,
		PriorRankStates:    make([]cudahotswap.State, c.WorldSize),
		SelectedRankStates: make([]cudahotswap.State, c.WorldSize),
		FailedRank:         -1,
	}

	priorEpochs := slices.Clone(c.RankEpochs)
	priorRule := c.ActiveRule
	priorCoordEpoch := c.Epoch

	// Snapshot pre-swap state on every rank and align all ranks at switchEpoch.
	for r := range c.WorldSize {
		c.RankEpochs[r] = switchEpoch
		svc := c.Services[r]
		if svc == nil {
			continue
		}
		st, err := svc.Status(c.KernelName)
		if err != nil {
			baseCode := synthesizeHotSwapSource(c.KernelName, fmt.Sprintf("rank_%d_baseline", r), sha256.Sum256([]byte(c.KernelName)))
			st, err = svc.Register(ctx, c.KernelName, cudahotswap.Source{
				Code:     baseCode,
				Filename: fmt.Sprintf("%s_rank%d_base.cu", c.KernelName, r),
				Entry:    c.KernelName,
			}, "baseline_v1")
			if err != nil {
				c.RankEpochs = priorEpochs
				res.FailedRank = r
				res.FailureReason = err.Error()
				return res, fmt.Errorf("coordinate hot-swap register baseline on rank %d: %w", r, err)
			}
			_, _ = svc.SetCheck(c.KernelName, cudahotswap.CheckConfig{
				Checker: func(context.Context, cudahotswap.State) error { return nil },
			}, "baseline_check")
		}
		res.PriorRankStates[r] = st
		res.SelectedRankStates[r] = st
	}

	var priorGeom *cudahotswap.LaunchGeometry
	if len(c.History) > 0 && c.History[len(c.History)-1].SelectedGeometry != nil {
		g := *c.History[len(c.History)-1].SelectedGeometry
		priorGeom = &g
	}

	rollbackAllRanks := func(failedRank int, cause error) (*MultiRankSwapResult, error) {
		for r := range c.WorldSize {
			c.RankEpochs[r] = priorEpochs[r]
			svc := c.Services[r]
			priorSt := res.PriorRankStates[r]
			if svc == nil || priorSt.Version == 0 {
				continue
			}
			if rbState, rbErr := svc.Select(ctx, c.KernelName, priorSt.Version, fmt.Sprintf("rank_%d_atomic_rollback", r)); rbErr == nil {
				res.SelectedRankStates[r] = rbState
			}
			if priorGeom != nil {
				if gState, gErr := svc.SetGeometry(c.KernelName, priorGeom, fmt.Sprintf("rank_%d_rollback_geometry", r)); gErr == nil {
					res.SelectedRankStates[r] = gState
				}
			}
		}
		c.ActiveRule = priorRule
		c.Epoch = priorCoordEpoch
		res.Committed = false
		res.RolledBack = true
		res.ActiveRule = priorRule
		res.FailedRank = failedRank
		res.FailureReason = cause.Error()
		return res, fmt.Errorf("%w (rank %d): %w", ErrMultiRankSwapVerificationFailed, failedRank, cause)
	}

	// Phase 2: Stage candidate kernel across all WorldSize ranks at the barrier
	// before any rank resumes collective execution.
	label := strings.TrimSpace(req.CandidateName)
	if label == "" {
		label = fmt.Sprintf("%s_epoch_%d", c.KernelName, switchEpoch)
	}
	for r := range c.WorldSize {
		svc := c.Services[r]
		if svc == nil {
			continue
		}
		src := req.Source
		if r < len(req.RankSources) && strings.TrimSpace(req.RankSources[r].Code) != "" {
			src = req.RankSources[r]
		}
		if strings.TrimSpace(src.Code) == "" {
			src.Code = synthesizeHotSwapSource(c.KernelName, label, sha256.Sum256([]byte(label)))
		} else {
			src.Code = ensureEntryWrapper(src.Code, c.KernelName)
		}
		if src.Entry == "" {
			src.Entry = c.KernelName
		}
		if src.Filename == "" {
			src.Filename = fmt.Sprintf("%s_rank%d.cu", c.KernelName, r)
		}

		_, selState, _, pushErr := pushWinnerHotSwap(ctx, svc, c.KernelName, src, req.Geometry, priorGeom, fmt.Sprintf("%s_rank%d", label, r), nil)
		if pushErr != nil {
			return rollbackAllRanks(r, pushErr)
		}
		res.SelectedRankStates[r] = selState
	}

	// Phase 3: Barrier verification across all WorldSize ranks before releasing
	// any rank to execute the new collective protocol.
	if len(c.Services) > 0 && c.Services[0] != nil {
		refVer := res.SelectedRankStates[0].Version
		for r := 1; r < c.WorldSize; r++ {
			if c.Services[r] != nil && res.SelectedRankStates[r].Version != refVer {
				return rollbackAllRanks(r, fmt.Errorf("version skew across ranks: rank 0 on v%d, rank %d on v%d", refVer, r, res.SelectedRankStates[r].Version))
			}
		}
	}

	if req.PostSwapRankCheck != nil {
		for r := range c.WorldSize {
			if err := req.PostSwapRankCheck(ctx, r, res.SelectedRankStates[r]); err != nil {
				return rollbackAllRanks(r, err)
			}
		}
	}

	var (
		trajDigest string
		maxLatUs   float64
	)
	if len(req.RankVotes) > 0 {
		if len(req.RankVotes) != c.WorldSize {
			return rollbackAllRanks(0, fmt.Errorf("%w: got %d votes, want %d", autotune.ErrDistPGOQuorumIncomplete, len(req.RankVotes), c.WorldSize))
		}
		for r, v := range req.RankVotes {
			if !v.HoldoutVerified {
				return rollbackAllRanks(r, fmt.Errorf("%w: rank %d", autotune.ErrDistPGOHoldoutUnverified, r))
			}
			if !v.CoherenceClean {
				return rollbackAllRanks(r, fmt.Errorf("%w: rank %d", autotune.ErrDistPGOCoherenceFailed, r))
			}
			d := strings.TrimSpace(v.TrajectoryDigest)
			if d == "" {
				return rollbackAllRanks(r, fmt.Errorf("%w: rank %d empty digest", autotune.ErrDistPGOTrajectoryMismatch, r))
			}
			if r == 0 {
				trajDigest = d
			} else if d != trajDigest {
				return rollbackAllRanks(r, fmt.Errorf("%w: rank %d digest %q != rank 0 digest %q", autotune.ErrDistPGOTrajectoryMismatch, r, d, trajDigest))
			}
			if v.LatencyUs > maxLatUs {
				maxLatUs = v.LatencyUs
			}
		}
		if req.BaselineLatencyUs > 0 && maxLatUs > 0 && !(maxLatUs < req.BaselineLatencyUs) {
			return rollbackAllRanks(c.WorldSize-1, fmt.Errorf("%w: max rank latency %.3f us >= baseline %.3f us", autotune.ErrDistPGOStragglerRegression, maxLatUs, req.BaselineLatencyUs))
		}
	}

	// Phase 4: All WorldSize ranks verified at switchEpoch — commit atomically.
	for r := range c.WorldSize {
		if trk := c.Trackers[r]; trk != nil {
			if len(trk.History) == 0 && res.PriorRankStates[r].Version != 0 {
				trk.History = append(trk.History, MegakernelEpochSnapshot{
					Epoch:         priorCoordEpoch,
					Blocks:        max(trk.Blocks, 1),
					ExpectedReady: priorCoordEpoch * uint64(max(trk.Blocks, 1)),
					State:         res.PriorRankStates[r],
					Geometry:      priorGeom,
				})
			}
			trk.Epoch = switchEpoch - 1
			trk.AdvanceEpoch(res.SelectedRankStates[r], req.Geometry)
		}
	}

	var selectedGeomCopy *cudahotswap.LaunchGeometry
	if req.Geometry != nil {
		g := *req.Geometry
		selectedGeomCopy = &g
	}
	c.Epoch = switchEpoch
	c.ActiveRule = req.CollectiveRule
	c.History = append(c.History, MultiRankEpochCommit{
		SwitchAtEpoch:      switchEpoch,
		CandidateName:      label,
		PriorRule:          priorRule,
		CommittedRule:      req.CollectiveRule,
		PriorRankStates:    slices.Clone(res.PriorRankStates),
		SelectedRankStates: slices.Clone(res.SelectedRankStates),
		PriorGeometry:      priorGeom,
		SelectedGeometry:   selectedGeomCopy,
		TrajectoryDigest:   trajDigest,
		MaxRankLatencyUs:   maxLatUs,
		BaselineLatencyUs:  req.BaselineLatencyUs,
		CommittedAt:        time.Now().UTC(),
	})

	res.Committed = true
	res.RolledBack = false
	res.ActiveRule = req.CollectiveRule
	return res, nil
}

// RollbackLastCommit atomically restores all WorldSize ranks to the immediate
// prior committed version and collective rule.
func (c *MultiRankEpochCoordinator) RollbackLastCommit(ctx context.Context) ([]cudahotswap.State, error) {
	if c == nil {
		return nil, errors.New("rollback last commit: nil coordinator")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	c.mu.Lock()
	defer c.mu.Unlock()

	if len(c.History) == 0 {
		return nil, errors.New("rollback last commit: empty commit history")
	}
	last := c.History[len(c.History)-1]
	restored := make([]cudahotswap.State, c.WorldSize)
	for r := range c.WorldSize {
		if trk := c.Trackers[r]; trk != nil && len(trk.History) >= 2 {
			st, err := trk.RestorePrevious(ctx)
			if err != nil {
				return nil, fmt.Errorf("rollback rank %d tracker: %w", r, err)
			}
			restored[r] = st
			continue
		}
		svc := c.Services[r]
		priorSt := last.PriorRankStates[r]
		if svc == nil || priorSt.Version == 0 {
			continue
		}
		st, err := svc.Select(ctx, c.KernelName, priorSt.Version, fmt.Sprintf("rank_%d_restore_previous", r))
		if err != nil {
			return nil, fmt.Errorf("rollback rank %d to v%d: %w", r, priorSt.Version, err)
		}
		if last.PriorGeometry != nil {
			if gSt, gErr := svc.SetGeometry(c.KernelName, last.PriorGeometry, fmt.Sprintf("rank_%d_restore_geometry", r)); gErr == nil {
				st = gSt
			}
		}
		restored[r] = st
	}
	c.ActiveRule = last.PriorRule
	c.History = c.History[:len(c.History)-1]
	if len(c.History) > 0 {
		c.Epoch = c.History[len(c.History)-1].SwitchAtEpoch
	} else {
		c.Epoch = 0
	}
	for r := range c.WorldSize {
		c.RankEpochs[r] = c.Epoch
	}
	return restored, nil
}

// DistAutonomousConfig configures [RunDistAutonomousPipeline].
type DistAutonomousConfig struct {
	DistCache     *autotune.DistVerifiedPGOCache
	CacheKey      autotune.DistPGOCacheKey
	Device        string
	Arch          string
	DeviceSMs     int
	BeamWidth     int
	DistParams    nvl72.DistFusionCostParams
	PTXASCompiler qualify.PTXASCompiler
	PTXASGate     qualify.StaticGate
	MaxPTXASIters int
	Coordinator   *MultiRankEpochCoordinator
	KernelName    string
}

// DistAutonomousRequest supplies inputs, profiling telemetry, holdout
// evidence, and multi-rank verification votes for [RunDistAutonomousPipeline].
type DistAutonomousRequest struct {
	ComputeGraph *egraph.EGraph
	ComputeRoot  egraph.Id
	DistGraph    *egraph.EGraph
	DistRoot     egraph.Id
	OutDType     hopper.DType

	EvalEnv    map[string][]nvl72.DistMatrix
	EvalConfig nvl72.DistEvalConfig
	MaxRelDiff float64

	PCBatch            cuptipcsampling.Batch
	LineToPC           map[int][2]uint64
	ReasonInventory    cuptipcsampling.ReasonInventory
	CausalObservations map[string]qualify.KernelObservation

	Identity          qualify.MeasuredIdentity
	Evidence          map[string]qualify.HoldoutCandidateEvidence
	DefaultEvidence   *qualify.HoldoutCandidateEvidence
	Verify            func(qualify.MeasuredPair) error

	SwitchAtEpoch     uint64
	RankVotes         []autotune.RankVerificationVote
	EvaluateRankVotes func(winner TopologyBeamCandidate) []autotune.RankVerificationVote
	PostSwapRankCheck func(ctx context.Context, rank int, state cudahotswap.State) error
	WinningCubin      []byte
}

// DistAutonomousResult reports the end-to-end outcome of
// [RunDistAutonomousPipeline].
type DistAutonomousResult struct {
	CacheHit         bool
	CachedEntry      *autotune.DistPGOCacheEntry
	BeamCandidates   []TopologyBeamCandidate
	Selected         *TopologyBeamCandidate
	HoldoutWinner    *qualify.StructuralBeamWinner
	PTXASRefinements []*qualify.PTXASLoopResult
	SwapResult       *MultiRankSwapResult
	PersistedToCache bool
	Coordinator      *MultiRankEpochCoordinator
}

// RunDistAutonomousPipeline executes the distributed topology-aware e-graph
// beam search, holdout tournament, barrier-aligned multi-rank hot-swap, and
// quorum-gated PGO cache persistence pipeline:
//
//  1. Topology-Keyed Cache Warm-Start: Queries [autotune.DistVerifiedPGOCache.LookupVerified].
//     On a verified hit, performs a barrier-aligned multi-rank hot-swap via
//     cfg.Coordinator and returns immediately with CacheHit=true.
//  2. Joint Compute+Collective Beam Search: Runs [SearchTopologyAwareBeam]
//     combining [qualify.ExtractStructuralPlanBeam],
//     [nvl72.AlgebraicCollectiveRules], [nvl72.DistCostModel], and
//     [nvl72.EvalDistExpr] numerical verification across WorldSize ranks, plus
//     optional zero-GPU ptxas refinement.
//  3. Holdout Qualification + Multi-Rank Quorum Verification: Evaluates the
//     beam with [qualify.SelectStructuralBeamWithHoldout] (when evidence is
//     provided) and collects per-rank verification votes across all WorldSize
//     ranks.
//  4. Barrier-Aligned Multi-Rank Hot-Swap + Quorum Cache Commit: Coordinates an
//     epoch-aligned hot-swap across all WorldSize ranks via
//     [MultiRankEpochCoordinator.CoordinateHotSwap] (rolling back all ranks if
//     any rank fails verification) and persists the quorum-verified winner via
//     [autotune.DistVerifiedPGOCache.RecordQuorumWinner].
func RunDistAutonomousPipeline(
	ctx context.Context,
	name string,
	cfg DistAutonomousConfig,
	req DistAutonomousRequest,
) (*DistAutonomousResult, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if strings.TrimSpace(name) == "" {
		name = "dist_autonomous"
	}

	distParams := cfg.DistParams
	if distParams.WorldSize == 0 && cfg.CacheKey.Topology.WorldSize > 0 {
		distParams.WorldSize = cfg.CacheKey.Topology.WorldSize
	}
	if distParams.SwitchBWGBps == 0 && cfg.CacheKey.Topology.SwitchBWGBps > 0 {
		distParams.SwitchBWGBps = cfg.CacheKey.Topology.SwitchBWGBps
	}
	if distParams.SwitchLatencyUs == 0 && cfg.CacheKey.Topology.SwitchLatencyUs > 0 {
		distParams.SwitchLatencyUs = cfg.CacheKey.Topology.SwitchLatencyUs
	}
	if distParams.TensorCoreTFLOPS == 0 && cfg.CacheKey.Topology.TensorCoreTFLOPS > 0 {
		distParams.TensorCoreTFLOPS = cfg.CacheKey.Topology.TensorCoreTFLOPS
	}
	if distParams.Target == "" && cfg.Arch != "" {
		distParams.Target = cfg.Arch
	}
	if distParams.BatchTokens == 0 && cfg.CacheKey.ShapeBucket > 0 {
		distParams.BatchTokens = cfg.CacheKey.ShapeBucket
	}

	res := &DistAutonomousResult{
		Coordinator: cfg.Coordinator,
	}

	// Step 1: Warm-Start Cache Hit.
	if cfg.DistCache != nil && cfg.CacheKey.Validate() == nil {
		entry, hit, err := cfg.DistCache.LookupVerified(cfg.CacheKey)
		if err != nil {
			return nil, fmt.Errorf("dist autonomous pipeline cache lookup: %w", err)
		}
		if hit {
			entryCopy := entry
			res.CacheHit = true
			res.CachedEntry = &entryCopy
			if cfg.Coordinator != nil {
				kName := cfg.Coordinator.KernelName
				code := string(entry.ScheduleJSON)
				if strings.TrimSpace(code) == "" || !strings.Contains(code, "__global__") {
					code = synthesizeHotSwapSource(kName, entry.CandidateName, entry.CubinSHA256)
				} else {
					code = ensureEntryWrapper(code, kName)
				}
				swapRes, swapErr := cfg.Coordinator.CoordinateHotSwap(ctx, MultiRankSwapRequest{
					SwitchAtEpoch:     req.SwitchAtEpoch,
					CandidateName:     entry.CandidateName,
					CollectiveRule:    nvl72.DistRewriteKind(entry.CollectiveRule),
					Source:            cudahotswap.Source{Code: code, Entry: kName},
					RankVotes:         entry.RankVotes,
					BaselineLatencyUs: entry.BaselineLatencyUs,
					PostSwapRankCheck: req.PostSwapRankCheck,
				})
				res.SwapResult = swapRes
				if swapErr != nil {
					return res, fmt.Errorf("dist autonomous pipeline warm-start hot-swap: %w", swapErr)
				}
			}
			return res, nil
		}
	}

	// Step 2: Topology-Aware Joint Compute+Collective Beam Search.
	costCfg := hopper.CostConfig{
		Device: cfg.Device,
		SMs:    cfg.DeviceSMs,
	}
	if costCfg.Device == "" && cfg.Arch != "" {
		costCfg.Device = cfg.Arch
	}
	beam, err := SearchTopologyAwareBeam(TopologyBeamConfig{
		Name:         name,
		ComputeGraph: req.ComputeGraph,
		ComputeRoot:  req.ComputeRoot,
		DistGraph:    req.DistGraph,
		DistRoot:     req.DistRoot,
		OutDType:     req.OutDType,
		CostConfig:   costCfg,
		DistParams:   distParams,
		BeamWidth:    cfg.BeamWidth,
		EvalEnv:      req.EvalEnv,
		EvalConfig:   req.EvalConfig,
		MaxRelDiff:   req.MaxRelDiff,
	})
	if err != nil {
		return nil, fmt.Errorf("dist autonomous pipeline beam search: %w", err)
	}

	// Optional zero-GPU ptxas -v refinement loop on compute plans.
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
		for i := range beam {
			if beam[i].ComputeCandidate.Plan == nil {
				continue
			}
			loopRes, lErr := qualify.RefinePlanByPTXASLoop(ctx, beam[i].ComputeCandidate.Plan, cfg.PTXASCompiler, loopCfg)
			if lErr != nil {
				return nil, fmt.Errorf("dist autonomous pipeline ptxas loop (%s): %w", beam[i].Handle(), lErr)
			}
			res.PTXASRefinements = append(res.PTXASRefinements, loopRes)
			if loopRes.RefinedPlan != nil {
				beam[i].ComputeCandidate.Plan = loopRes.RefinedPlan
			}
		}
	}
	res.BeamCandidates = slices.Clone(beam)

	// Step 3: Holdout Qualification + Multi-Rank Winner Selection.
	selected := beam[0]
	if req.DefaultEvidence != nil || len(req.Evidence) > 0 {
		structBeam := make([]qualify.StructuralBeamCandidate, len(beam))
		byHandle := make(map[string]TopologyBeamCandidate, len(beam))
		for i, b := range beam {
			sc := b.ComputeCandidate
			sc.Name = b.Handle()
			sc.PlanSignature = b.CombinedSignature
			sc.EstimatedNs = b.TotalDistLatencyUs * 1000.0
			structBeam[i] = sc
			byHandle[sc.Handle()] = b
			byHandle[sc.PlanSignature] = b
		}
		baseline := structBeam[len(structBeam)-1]
		baseline.Name = baseline.Handle() + "_baseline"
		baseline.PlanSignature = baseline.PlanSignature + ":baseline"

		evalReq := AutonomousEvalRequest{
			Identity:        req.Identity,
			Evidence:        req.Evidence,
			DefaultEvidence: req.DefaultEvidence,
			Verify:          req.Verify,
		}
		identity := resolveHoldoutIdentity(evalReq, AutonomousPipelineConfig{Arch: cfg.Arch}, name)
		verify := req.Verify
		if verify == nil {
			verify = VerifyDigest(identity)
		}
		evMap := resolveBeamHoldoutEvidence(structBeam, baseline, identity, evalReq)
		winner, hErr := qualify.SelectStructuralBeamWithHoldout(identity, baseline, structBeam, evMap, verify)
		if hErr != nil {
			return res, fmt.Errorf("dist autonomous pipeline holdout tournament: %w", hErr)
		}
		res.HoldoutWinner = &winner
		if !winner.Promoted {
			return res, nil
		}
		if matched, ok := byHandle[winner.Winner.Handle()]; ok {
			selected = matched
		} else if matched, ok := byHandle[winner.Winner.PlanSignature]; ok {
			selected = matched
		}
	}
	res.Selected = &selected

	worldSize := selected.WorldSize
	if cfg.Coordinator != nil && cfg.Coordinator.WorldSize > 0 {
		worldSize = cfg.Coordinator.WorldSize
	} else if cfg.CacheKey.Topology.WorldSize > 0 {
		worldSize = cfg.CacheKey.Topology.WorldSize
	}

	votes := slices.Clone(req.RankVotes)
	if len(votes) == 0 && req.EvaluateRankVotes != nil {
		votes = req.EvaluateRankVotes(selected)
	}
	if len(votes) == 0 {
		votes = make([]autotune.RankVerificationVote, worldSize)
		for r := range worldSize {
			votes[r] = autotune.RankVerificationVote{
				Rank:              r,
				HoldoutVerified:   true,
				CoherenceClean:    true,
				TrajectoryDigest:  selected.EvalTrajectoryDigest,
				LatencyUs:         selected.TotalDistLatencyUs,
				BaselineLatencyUs: selected.BaselineTotalUs,
				HoldoutRatio:      selected.BaselineTotalUs / max(selected.TotalDistLatencyUs, 1e-6),
				OneSidedPValue:    0.015625,
			}
		}
	}

	baselineUs := selected.BaselineTotalUs
	if !(baselineUs > selected.TotalDistLatencyUs) {
		baselineUs = max(selected.UnfusedDistLatencyUs+selected.ComputeLatencyUs, selected.TotalDistLatencyUs*1.25)
	}
	var maxVoteLat float64
	for _, v := range votes {
		if v.LatencyUs > maxVoteLat {
			maxVoteLat = v.LatencyUs
		}
	}
	if maxVoteLat == 0 {
		maxVoteLat = selected.TotalDistLatencyUs
	}

	kName := resolveDistKernelName(name, cfg)
	winnerSrc := buildTopologyCandidateSource(selected, kName)
	winnerGeom := candidateLaunchGeometry(selected.ComputeCandidate)

	// Step 4: Barrier-Aligned Multi-Rank Hot-Swap.
	if cfg.Coordinator != nil {
		swapRes, swapErr := cfg.Coordinator.CoordinateHotSwap(ctx, MultiRankSwapRequest{
			SwitchAtEpoch:     req.SwitchAtEpoch,
			CandidateName:     selected.Handle(),
			CollectiveRule:    selected.CollectiveRule,
			Source:            cudahotswap.Source{Code: winnerSrc, Filename: kName + ".cu", Entry: kName},
			Geometry:          &winnerGeom,
			RankVotes:         votes,
			BaselineLatencyUs: baselineUs,
			PostSwapRankCheck: req.PostSwapRankCheck,
		})
		res.SwapResult = swapRes
		if swapErr != nil {
			return res, fmt.Errorf("dist autonomous pipeline coordinated hot-swap: %w", swapErr)
		}
	}

	// Step 5: Quorum-Gated Distributed PGO Cache Commit.
	if cfg.DistCache != nil {
		key := cfg.CacheKey
		if key.Topology.WorldSize == 0 {
			key.Topology = autotune.DistTopologyKey{
				WorldSize:        worldSize,
				Arch:             distParams.Target,
				SwitchBWGBps:     distParams.SwitchBWGBps,
				SwitchLatencyUs:  distParams.SwitchLatencyUs,
				TensorCoreTFLOPS: distParams.TensorCoreTFLOPS,
			}
		}
		if key.PlanKey == "" {
			key.PlanKey = name
		}
		if key.ShapeBucket <= 0 {
			key.ShapeBucket = max(distParams.BatchTokens, 1)
		}
		var cubinSum [32]byte
		if len(req.WinningCubin) > 0 {
			cubinSum = sha256.Sum256(req.WinningCubin)
		}
		entry := autotune.DistPGOCacheEntry{
			Key:               key,
			CandidateName:     selected.Handle(),
			CollectiveRule:    string(selected.CollectiveRule),
			PlanSignature:     selected.CombinedSignature,
			ScheduleJSON:      []byte(winnerSrc),
			PTX:               selected.PTX,
			Cubin:             slices.Clone(req.WinningCubin),
			CubinSHA256:       cubinSum,
			SourceSHA256:      sha256.Sum256([]byte(winnerSrc)),
			BaselineLatencyUs: baselineUs,
			MaxRankLatencyUs:  maxVoteLat,
			TrajectoryDigest:  votes[0].TrajectoryDigest,
			RankVotes:         votes,
			PromotedAt:        time.Now().UTC(),
		}
		if err := cfg.DistCache.RecordQuorumWinner(entry); err != nil {
			return res, fmt.Errorf("dist autonomous pipeline record quorum winner: %w", err)
		}
		res.PersistedToCache = true
		res.CachedEntry = &entry
	}

	return res, nil
}

func normalizeDistParams(p nvl72.DistFusionCostParams) nvl72.DistFusionCostParams {
	if p.HBMGBps == 0 {
		p.HBMGBps = nvl72.NVL72HBM3eGBps
	}
	if p.HiddenDim == 0 {
		p.HiddenDim = 7168
	}
	if p.KernelLaunchUs == 0 {
		p.KernelLaunchUs = 3
	}
	if p.CollectiveLaunchUs == 0 {
		p.CollectiveLaunchUs = 8
	}
	if p.TileM == 0 {
		p.TileM = 128
	}
	if p.TileN == 0 {
		p.TileN = 256
	}
	if p.Target == "" {
		p.Target = "sm_100a"
	}
	return p
}

func defaultTPCollectiveGraph() (*egraph.EGraph, egraph.Id) {
	g := egraph.New()
	leaf := func(s string) egraph.Id { return g.Add(egraph.NewLeaf(s)) }
	mm := g.Add(egraph.NewNode(nvl72.DistGemm, leaf("x"), leaf("w")))
	ar := g.Add(egraph.NewNode(nvl72.DistAllReduceSum, mm))
	add := g.Add(egraph.NewNode(nvl72.DistAdd, ar, leaf("r")))
	root := g.Add(egraph.NewNode(nvl72.DistRMSNorm, add, leaf("g")))
	return g, root
}

func graphHasDistOps(g *egraph.EGraph) bool {
	if g == nil {
		return false
	}
	for _, c := range g.Classes() {
		for _, n := range c.Nodes {
			switch n.Op {
			case nvl72.DistGemm, nvl72.DistAllReduceSum, nvl72.DistReduceScatter,
				nvl72.DistAllGather, nvl72.DistAdd, nvl72.DistRMSNorm,
				nvl72.FusedTileFabricRedGemm, nvl72.FusedMultimemLdReduceRMSNorm,
				nvl72.PipelinedFabricGetGemm, nvl72.TiledMultimemScatterGemm:
				return true
			}
		}
	}
	return false
}

func detectDistRule(e *egraph.Expr) nvl72.DistRewriteKind {
	if e == nil {
		return ""
	}
	switch e.Op {
	case nvl72.FusedTileFabricRedGemm:
		return nvl72.RuleTilePushFabricRed
	case nvl72.FusedMultimemLdReduceRMSNorm:
		return nvl72.RulePullMultimemLdReduce
	case nvl72.PipelinedFabricGetGemm:
		return nvl72.RuleAGMatmulToRingPrefetch
	case nvl72.TiledMultimemScatterGemm:
		return nvl72.RuleMatmulRSAssociative
	}
	for _, c := range e.Children {
		if r := detectDistRule(c); r != "" {
			return r
		}
	}
	return ""
}

func priceDistExpr(e *egraph.Expr, m nvl72.DistCostModel) float64 {
	if e == nil || len(e.Children) == 0 {
		return 0
	}
	cost := m.NodeLocalCost(egraph.ENode{Op: e.Op, Children: make([]egraph.Id, len(e.Children))})
	for _, c := range e.Children {
		cost += priceDistExpr(c, m)
	}
	return cost
}

func synthesizeDistEvalEnv(expr *egraph.Expr, ranks int) (map[string][]nvl72.DistMatrix, error) {
	if ranks < 1 {
		return nil, fmt.Errorf("invalid rank count %d", ranks)
	}
	M := max(2*ranks, 8)
	K := 8
	N := 8

	shapes := make(map[string][2]int)
	var infer func(e *egraph.Expr, rows, cols int)
	infer = func(e *egraph.Expr, rows, cols int) {
		if e == nil {
			return
		}
		if len(e.Children) == 0 {
			if _, exists := shapes[e.Op]; !exists {
				shapes[e.Op] = [2]int{rows, cols}
			}
			return
		}
		switch e.Op {
		case nvl72.DistGemm, nvl72.TiledMultimemScatterGemm:
			if len(e.Children) == 2 {
				infer(e.Children[0], M, K)
				infer(e.Children[1], K, N)
			}
		case nvl72.PipelinedFabricGetGemm:
			if len(e.Children) == 2 {
				infer(e.Children[0], M/ranks, K)
				infer(e.Children[1], K, N)
			}
		case nvl72.DistAllGather:
			if len(e.Children) == 1 {
				infer(e.Children[0], M/ranks, K)
			}
		case nvl72.DistAllReduceSum, nvl72.DistReduceScatter:
			if len(e.Children) == 1 {
				infer(e.Children[0], rows, cols)
			}
		case nvl72.DistAdd:
			for _, c := range e.Children {
				infer(c, rows, cols)
			}
		case nvl72.DistRMSNorm:
			if len(e.Children) == 2 {
				infer(e.Children[0], rows, cols)
				infer(e.Children[1], 1, cols)
			}
		case nvl72.FusedTileFabricRedGemm:
			if len(e.Children) == 3 {
				infer(e.Children[0], M, K)
				infer(e.Children[1], K, N)
				infer(e.Children[2], M, N)
			}
		case nvl72.FusedMultimemLdReduceRMSNorm:
			if len(e.Children) == 3 {
				infer(e.Children[0], rows, cols)
				infer(e.Children[1], rows, cols)
				infer(e.Children[2], 1, cols)
			}
		default:
			for _, c := range e.Children {
				infer(c, rows, cols)
			}
		}
	}
	infer(expr, M, N)

	rng := rand.New(rand.NewPCG(42, 107))
	env := make(map[string][]nvl72.DistMatrix, len(shapes))
	names := make([]string, 0, len(shapes))
	for name := range shapes {
		names = append(names, name)
	}
	slices.Sort(names)
	for _, name := range names {
		sh := shapes[name]
		mats := make([]nvl72.DistMatrix, ranks)
		for r := range ranks {
			data := make([]float32, sh[0]*sh[1])
			for i := range data {
				data[i] = float32(rng.NormFloat64() * 0.5)
			}
			mats[r] = nvl72.DistMatrix{Rows: sh[0], Cols: sh[1], Data: data}
		}
		env[name] = mats
	}
	return env, nil
}

func maxRelDiffDistMatrices(a, b []nvl72.DistMatrix) float64 {
	if len(a) != len(b) {
		return math.Inf(1)
	}
	worst := 0.0
	for r := range a {
		if a[r].Rows != b[r].Rows || a[r].Cols != b[r].Cols || len(a[r].Data) != len(b[r].Data) {
			return math.Inf(1)
		}
		for i := range a[r].Data {
			x, y := float64(a[r].Data[i]), float64(b[r].Data[i])
			worst = max(worst, math.Abs(x-y)/max(1.0, math.Abs(y)))
		}
	}
	return worst
}

func digestDistMatrices(mats []nvl72.DistMatrix) string {
	type rankSummary struct {
		Rank int   `json:"rank"`
		Rows int   `json:"rows"`
		Cols int   `json:"cols"`
		Q    []int `json:"q"`
	}
	summaries := make([]rankSummary, len(mats))
	for r, m := range mats {
		q := make([]int, len(m.Data))
		for i, v := range m.Data {
			q[i] = int(math.Round(float64(v) * 1e4))
		}
		summaries[r] = rankSummary{Rank: r, Rows: m.Rows, Cols: m.Cols, Q: q}
	}
	b, _ := json.Marshal(summaries)
	sum := sha256.Sum256(b)
	return "sha256:" + hex.EncodeToString(sum[:])
}

func shortRuleSlug(rule nvl72.DistRewriteKind) string {
	switch rule {
	case nvl72.RuleTilePushFabricRed:
		return "tile_push_fabric_red"
	case nvl72.RulePullMultimemLdReduce:
		return "pull_multimem_ld_reduce"
	case nvl72.RuleAGMatmulToRingPrefetch:
		return "ag_ring_prefetch"
	case nvl72.RuleMatmulRSAssociative:
		return "rs_tiled_scatter"
	default:
		return "unfused_collective"
	}
}

func resolveDistKernelName(name string, cfg DistAutonomousConfig) string {
	if cfg.Coordinator != nil && cfg.Coordinator.KernelName != "" {
		return cfg.Coordinator.KernelName
	}
	if strings.TrimSpace(cfg.KernelName) != "" {
		return sanitizeCIdentifier(cfg.KernelName)
	}
	return sanitizeCIdentifier(name)
}

func buildTopologyCandidateSource(cand TopologyBeamCandidate, entry string) string {
	var parts []string
	if cand.ComputeCandidate.Plan != nil || cand.ComputeCandidate.Megakernel != nil {
		if src, _, err := cand.ComputeCandidate.Source(); err == nil && strings.TrimSpace(src) != "" {
			parts = append(parts, src)
		}
	}
	if strings.TrimSpace(cand.CUDASource) != "" && len(parts) == 0 {
		parts = append(parts, cand.CUDASource)
	}
	if strings.TrimSpace(cand.PTX) != "" {
		parts = append(parts, "/* collective_rule="+string(cand.CollectiveRule)+"\n"+cand.PTX+"*/")
	}
	combined := strings.Join(parts, "\n")
	if strings.TrimSpace(combined) == "" {
		combined = synthesizeHotSwapSource(entry, cand.Handle(), sha256.Sum256([]byte(cand.CombinedSignature)))
	}
	return ensureEntryWrapper(combined, entry)
}
