package main

import (
	"cmp"
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"math"
	"slices"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudagraph"
	"github.com/tmc/lib/nvidia/x/cudahotswap"
	"github.com/tmc/lib/nvidia/x/cudashape"
	"github.com/tmc/lib/nvidia/x/cuptipcsampling"
	"github.com/tmc/lib/nvidia/x/egraph/autotune"
	"github.com/tmc/lib/nvidia/x/egraph/hopper"
	"github.com/tmc/lib/nvidia/x/egraph/qualify"
	"github.com/tmc/lib/nvidia/x/pairedtiming"
)

// StageAttribution binds a persistent megakernel stage to its synthetic #line
// file, 1-based source line span, and disjoint SASS [autotune.PCRange].
type StageAttribution struct {
	StageIndex int
	StageName  string
	Region     string
	File       string
	StartLine  int
	EndLine    int
	PCRange    autotune.PCRange
}

// StageStallRank pairs a megakernel stage with its attributed CUPTI PC-sampling
// stall histogram and [qualify.StallSample] for stage hotspot ranking.
type StageStallRank struct {
	StageIndex  int
	StageName   string
	Region      string
	Histogram   autotune.StallHistogram
	StallSample qualify.StallSample
}

// EpochConfig configures isolated CUPTI PC-sampling epochs so profiling state
// is torn down before any live module load or graph hot-swap.
type EpochConfig struct {
	NewEpochCollector func() (autotune.LiveCollector, io.Closer, error)
}

// HoldoutGateConfig configures the two-stage ABBA promotion gate combining a
// 3-block screen, held-out ABBA confirmation, A/A null calibration, source
// digest verification, and frozen-prompt greedy token-ID agreement.
type HoldoutGateConfig struct {
	Identity                qualify.MeasuredIdentity
	ReferenceSource         string
	CandidateSource         string
	ExpectedReferenceDigest [32]byte
	ExpectedCandidateDigest [32]byte
	RequireAANull           bool
	PairTimeUnit            time.Duration
}

// HoldoutReceipt records the outcome of [SelectMeasuredWithHoldout].
type HoldoutReceipt struct {
	Promotion       qualify.HoldoutPromotionResult
	ABBASummary     *ABBASummary
	ReferenceDigest [32]byte
	CandidateDigest [32]byte
}

// egraphFlightLoop coordinates per-stage megakernel PC attribution,
// profile-guided step candidate generation, two-stage held-out ABBA + token-ID
// qualification, and flight-control hot-swap with multi-version rollback.
type egraphFlightLoop struct {
	name              string
	plan              *hopper.Plan
	mega              hopper.Megakernel
	source            string
	stages            []StageAttribution
	ranges            []autotune.PCRange
	service           *cudahotswap.Service
	launchCfg         cudahotswap.GraphLaunchConfig
	currentState      cudahotswap.State
	versionStack      []uint64
	newEpochCollector func() (autotune.LiveCollector, io.Closer, error)
	activeEpochs      int
}

// newEGraphFlightLoop lowers plan into a persistent [hopper.Megakernel],
// extracts its per-stage #line attribution table, and prepares launch geometry
// for [cudahotswap.Service].
func newEGraphFlightLoop(plan *hopper.Plan, megaName string, svc *cudahotswap.Service, points []autotune.PCFilePoint) (*egraphFlightLoop, error) {
	if plan == nil {
		return nil, errors.New("flight loop: nil hopper plan")
	}
	mega, err := plan.Megakernel(megaName)
	if err != nil {
		return nil, fmt.Errorf("flight loop megakernel: %w", err)
	}
	src, _, err := mega.LineInfoSource()
	if err != nil {
		return nil, fmt.Errorf("flight loop lineinfo source: %w", err)
	}
	stages, ranges, err := StageAttributionMap(mega, points)
	if err != nil {
		return nil, fmt.Errorf("flight loop stage attribution: %w", err)
	}
	grid, block := mega.Grid()
	smem := mega.SharedBytes()
	launchCfg := cudahotswap.GraphLaunchConfig{
		Grid:           driver.Dims(uint32(grid[0]), uint32(grid[1]), uint32(grid[2])),
		Block:          driver.Dims(uint32(block[0]), uint32(block[1]), uint32(block[2])),
		SharedMemBytes: uint32(smem),
	}
	return &egraphFlightLoop{
		name:      mega.Name,
		plan:      plan,
		mega:      mega,
		source:    src,
		stages:    stages,
		ranges:    ranges,
		service:   svc,
		launchCfg: launchCfg,
	}, nil
}

// SetEpochConfig configures isolated profiling epochs on f.
func (f *egraphFlightLoop) SetEpochConfig(cfg EpochConfig) {
	if f == nil {
		return
	}
	f.newEpochCollector = cfg.NewEpochCollector
}

// StageAttributionMap extracts the per-stage #line spans from m and binds each
// stage to a disjoint [autotune.PCRange] keyed by stage.StageName. When points
// is non-empty, ranges are coalesced from points via
// [autotune.PCRangesFromFilePoints] and [hopper.Megakernel.StageFileLabels];
// when points is empty, deterministic non-overlapping PC ranges are derived
// from the 1-based line spans emitted by [hopper.Megakernel.LineInfoSource].
func StageAttributionMap(m hopper.Megakernel, points []autotune.PCFilePoint) ([]StageAttribution, []autotune.PCRange, error) {
	_, spans, err := m.LineInfoSource()
	if err != nil {
		return nil, nil, err
	}
	if len(spans) == 0 {
		return nil, nil, errors.New("stage attribution: megakernel has no stages")
	}
	labels := m.StageFileLabels()
	var ranges []autotune.PCRange
	if len(points) > 0 {
		ranges, err = autotune.PCRangesFromFilePoints(points, labels)
		if err != nil {
			return nil, nil, err
		}
	} else {
		ranges = make([]autotune.PCRange, len(spans))
		for i, sp := range spans {
			if sp.StartLine <= 0 || sp.EndLine < sp.StartLine {
				return nil, nil, fmt.Errorf("stage attribution: invalid line span [%d, %d] for %q", sp.StartLine, sp.EndLine, sp.StageName)
			}
			ranges[i] = autotune.PCRange{
				Start: uint64(sp.StartLine) * 16,
				End:   uint64(sp.EndLine+1) * 16,
				Label: sp.StageName,
			}
		}
	}
	byLabel := make(map[string]autotune.PCRange, len(ranges))
	for _, r := range ranges {
		if existing, ok := byLabel[r.Label]; ok {
			if r.Start < existing.Start {
				existing.Start = r.Start
			}
			if r.End > existing.End {
				existing.End = r.End
			}
			byLabel[r.Label] = existing
		} else {
			byLabel[r.Label] = r
		}
	}
	out := make([]StageAttribution, len(spans))
	for i, sp := range spans {
		out[i] = StageAttribution{
			StageIndex: sp.StageIndex,
			StageName:  sp.StageName,
			Region:     sp.StageName,
			File:       sp.File,
			StartLine:  sp.StartLine,
			EndLine:    sp.EndLine,
			PCRange:    byLabel[sp.StageName],
		}
	}
	return out, ranges, nil
}

// HistogramsByRegionFromBatch attributes CUPTI PC-sampling batches for function
// fn across ranges, returning the kernel-wide total histogram and a per-region
// map keyed by stage name (plus [autotune.Unattributed] for out-of-range PCs).
func HistogramsByRegionFromBatch(batches []cuptipcsampling.Batch, inv cuptipcsampling.ReasonInventory, fn string, ranges []autotune.PCRange) (autotune.StallHistogram, map[string]autotune.StallHistogram, error) {
	return autotune.AttributeBatch(batches, inv, fn, ranges)
}

// StallSampleFromHistogram converts an [autotune.StallHistogram] into a
// [qualify.StallSample] for profile-guided hypothesis generation.
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

func stallSampleTotal(s qualify.StallSample) uint64 {
	sum := s.WaitTMEM + s.WaitMBarrier + s.MIOThrottle + s.MathThrottle + s.LongScoreboard
	return max(s.TotalSamples, sum)
}

// AttributeAndRankStages attributes batches across the megakernel's stage PC
// ranges and ranks stages in descending order of total issued stall samples
// (breaking ties by StageIndex).
func (f *egraphFlightLoop) AttributeAndRankStages(batches []cuptipcsampling.Batch, inv cuptipcsampling.ReasonInventory) ([]StageStallRank, error) {
	if f == nil {
		return nil, errors.New("attribute stages: nil flight loop")
	}
	_, byRegion, err := HistogramsByRegionFromBatch(batches, inv, f.name, f.ranges)
	if err != nil {
		return nil, err
	}
	ranks := make([]StageStallRank, len(f.stages))
	for i, st := range f.stages {
		h := byRegion[st.Region]
		ranks[i] = StageStallRank{
			StageIndex:  st.StageIndex,
			StageName:   st.StageName,
			Region:      st.Region,
			Histogram:   h,
			StallSample: StallSampleFromHistogram(h),
		}
	}
	slices.SortStableFunc(ranks, func(a, b StageStallRank) int {
		ta, tb := stallSampleTotal(a.StallSample), stallSampleTotal(b.StallSample)
		if c := cmp.Compare(tb, ta); c != 0 {
			return c
		}
		return cmp.Compare(a.StageIndex, b.StageIndex)
	})
	return ranks, nil
}

// ProposeHottestStageCandidates selects the highest-stall stage from ranks and
// generates legal candidate schedule rewrites via [qualify.ProposeStepCandidates].
// Direct causal observations in causal (keyed by StageName) are kept strictly
// separate from PC-sampling stall hypotheses.
func (f *egraphFlightLoop) ProposeHottestStageCandidates(ranks []StageStallRank, causal map[string]qualify.KernelObservation) (StageStallRank, []qualify.CandidateStepProposal, error) {
	if f == nil || f.plan == nil {
		return StageStallRank{}, nil, errors.New("propose candidates: nil flight loop or plan")
	}
	if len(ranks) == 0 {
		return StageStallRank{}, nil, errors.New("propose candidates: empty stage ranks")
	}
	for _, r := range ranks {
		if r.StageIndex < 0 || r.StageIndex >= len(f.plan.Kernels) {
			return StageStallRank{}, nil, fmt.Errorf("propose candidates: stage index %d out of range [0, %d)", r.StageIndex, len(f.plan.Kernels))
		}
		step := f.plan.Kernels[r.StageIndex]
		var obs qualify.KernelObservation
		if causal != nil {
			obs = causal[r.StageName]
		}
		if obs.Name == "" {
			obs.Name = r.StageName
		}
		proposals := qualify.ProposeStepCandidates(step, obs, r.StallSample)
		if len(proposals) > 0 {
			return r, proposals, nil
		}
	}
	return ranks[0], nil, nil
}

// CandidateMegakernel clones f's underlying plan, substitutes cand at
// stageIdx, lowers the modified plan to a persistent [hopper.Megakernel], and
// returns its #line-annotated CUDA translation unit.
func (f *egraphFlightLoop) CandidateMegakernel(stageIdx int, cand qualify.Step) (*hopper.Plan, hopper.Megakernel, string, error) {
	if f == nil || f.plan == nil {
		return nil, hopper.Megakernel{}, "", errors.New("candidate megakernel: nil flight loop or plan")
	}
	if stageIdx < 0 || stageIdx >= len(f.plan.Kernels) {
		return nil, hopper.Megakernel{}, "", fmt.Errorf("candidate megakernel: stage index %d out of range [0, %d)", stageIdx, len(f.plan.Kernels))
	}
	cloned := cloneHopperPlan(f.plan)
	prevStreamK := cloned.Kernels[stageIdx].Tile != nil && cloned.Kernels[stageIdx].Tile.StreamK > 0
	cloned.Kernels[stageIdx] = cand
	if cand.Tile != nil && cand.Tile.StreamK > 0 && !prevStreamK {
		if wsBytes := cand.Tile.WorkspaceBytes(); wsBytes > 0 {
			wsName := fmt.Sprintf("%s_sk_ws%d", cloned.Name, stageIdx+1)
			cloned.Temps = append(cloned.Temps, hopper.Tensor{
				Name:  wsName,
				DType: hopper.F32,
				Shape: []int{(wsBytes + 3) / 4},
			})
			cloned.Kernels[stageIdx].Args = append(slices.Clone(cloned.Kernels[stageIdx].Args), wsName)
		}
	}
	mega, err := cloned.Megakernel(f.name)
	if err != nil {
		return nil, hopper.Megakernel{}, "", err
	}
	src, _, err := mega.LineInfoSource()
	if err != nil {
		return nil, hopper.Megakernel{}, "", err
	}
	return cloned, mega, src, nil
}

func cloneHopperPlan(p *hopper.Plan) *hopper.Plan {
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
		ck := k
		ck.Args = slices.Clone(k.Args)
		if k.GEMM != nil {
			g := *k.GEMM
			g.Extras = slices.Clone(k.GEMM.Extras)
			ck.GEMM = &g
		}
		if k.Tile != nil {
			t := *k.Tile
			t.Extras = slices.Clone(k.Tile.Extras)
			ck.Tile = &t
		}
		if k.Attention != nil {
			a := *k.Attention
			ck.Attention = &a
		}
		if k.Row != nil {
			r := *k.Row
			r.Inputs = slices.Clone(k.Row.Inputs)
			r.Values = make([]hopper.RowValue, len(k.Row.Values))
			for j, v := range k.Row.Values {
				v.Args = slices.Clone(v.Args)
				r.Values[j] = v
			}
			r.Outputs = slices.Clone(k.Row.Outputs)
			ck.Row = &r
		}
		out.Kernels[i] = ck
	}
	return out
}

// VerifyDigest checks that source is non-empty and its SHA-256 digest matches want.
func VerifyDigest(source string, want [32]byte) error {
	if source == "" {
		return errors.New("verify digest: empty source")
	}
	if want == ([32]byte{}) {
		return errors.New("verify digest: zero expected digest")
	}
	got := sha256.Sum256([]byte(source))
	if got != want {
		return fmt.Errorf("verify digest: sha256 mismatch: got %x, want %x", got[:8], want[:8])
	}
	return nil
}

// VerifyTokenSequence checks that referenceTokens and candidateTokens are
// non-empty and identical at every position. Floating-point logit tolerance
// (logitMaxAbs) is reported for diagnostics only and never excuses a greedy
// token-ID divergence.
func VerifyTokenSequence(referenceTokens, candidateTokens []int, logitMaxAbs float64) error {
	if len(referenceTokens) == 0 || len(candidateTokens) == 0 {
		return errors.New("verify token sequence: non-empty frozen-prompt greedy token-ID trajectories required")
	}
	n := min(len(referenceTokens), len(candidateTokens))
	for i := range n {
		if referenceTokens[i] != candidateTokens[i] {
			return fmt.Errorf("verify token sequence: greedy token ID mismatch at index %d: reference=%d candidate=%d (logit_max_abs=%g)", i, referenceTokens[i], candidateTokens[i], logitMaxAbs)
		}
	}
	if len(referenceTokens) != len(candidateTokens) {
		return fmt.Errorf("verify token sequence: trajectory length mismatch at index %d: reference=%d candidate=%d", n, len(referenceTokens), len(candidateTokens))
	}
	return nil
}

// HoldoutBlocksToABBAPairs unpacks each held-out [pairedtiming.Block]
// (baseline A1, candidate B1, candidate B2, baseline A2) into its two paired
// AB and BA [ABBAPair] trials suitable for [EvaluateABBAPairs] in bench_abba.go.
func HoldoutBlocksToABBAPairs(blocks []pairedtiming.Block, unit time.Duration, tokensMatch bool) ([]ABBAPair, error) {
	if len(blocks) == 0 {
		return nil, errors.New("holdout blocks to ABBA pairs: empty blocks")
	}
	if unit <= 0 {
		unit = time.Microsecond
	}
	pairs := make([]ABBAPair, 0, 2*len(blocks))
	toDuration := func(v float64) (time.Duration, error) {
		if math.IsNaN(v) || math.IsInf(v, 0) || v <= 0 {
			return 0, fmt.Errorf("nonfinite or nonpositive timing sample %v", v)
		}
		d := time.Duration(math.Round(v * float64(unit)))
		if d <= 0 {
			d = 1
		}
		return d, nil
	}
	for i, b := range blocks {
		a1, err := toDuration(b.A1)
		if err != nil {
			return nil, fmt.Errorf("block %d A1: %w", i, err)
		}
		b1, err := toDuration(b.B1)
		if err != nil {
			return nil, fmt.Errorf("block %d B1: %w", i, err)
		}
		b2, err := toDuration(b.B2)
		if err != nil {
			return nil, fmt.Errorf("block %d B2: %w", i, err)
		}
		a2, err := toDuration(b.A2)
		if err != nil {
			return nil, fmt.Errorf("block %d A2: %w", i, err)
		}
		pairs = append(pairs,
			ABBAPair{
				Order:         "AB",
				BaselineTPOT:  a1,
				CandidateTPOT: b1,
				TokensMatch:   tokensMatch,
			},
			ABBAPair{
				Order:         "BA",
				BaselineTPOT:  a2,
				CandidateTPOT: b2,
				TokensMatch:   tokensMatch,
			},
		)
	}
	return pairs, nil
}

// SelectMeasuredWithHoldout runs the two-stage ABBA promotion gate:
//  1. Optional source SHA-256 verification via [VerifyDigest].
//  2. Pre-gate token check via [VerifyTokenSequence] and 3-block fast screen +
//     A/A null calibration + >= 6-block held-out ABBA gate via
//     [qualify.SelectHoldoutPromoted].
//  3. Bridge evaluation via [HoldoutBlocksToABBAPairs] and [EvaluateABBAPairs]
//     when at least 5 held-out blocks (>= 10 AB/BA pairs) are present.
func SelectMeasuredWithHoldout(cfg HoldoutGateConfig, ev qualify.HoldoutCandidateEvidence, verify func(qualify.MeasuredPair) error) (HoldoutReceipt, error) {
	var receipt HoldoutReceipt
	if cfg.ReferenceSource != "" {
		receipt.ReferenceDigest = sha256.Sum256([]byte(cfg.ReferenceSource))
	}
	if cfg.CandidateSource != "" {
		receipt.CandidateDigest = sha256.Sum256([]byte(cfg.CandidateSource))
	}
	if cfg.ExpectedReferenceDigest != ([32]byte{}) {
		if err := VerifyDigest(cfg.ReferenceSource, cfg.ExpectedReferenceDigest); err != nil {
			receipt.Promotion = qualify.HoldoutPromotionResult{
				Status:         "NOT_EVALUATED",
				Reason:         "reference " + err.Error(),
				CandidateState: "MEASURED_NO_PROMOTION",
			}
			return receipt, err
		}
	}
	if cfg.ExpectedCandidateDigest != ([32]byte{}) {
		if err := VerifyDigest(cfg.CandidateSource, cfg.ExpectedCandidateDigest); err != nil {
			receipt.Promotion = qualify.HoldoutPromotionResult{
				Status:         "NOT_EVALUATED",
				Reason:         "candidate " + err.Error(),
				CandidateState: "MEASURED_NO_PROMOTION",
			}
			return receipt, err
		}
	}
	if cfg.RequireAANull && len(ev.AANullBlocks) == 0 {
		refVariant := &qualify.Variant{Name: ev.Screen.Reference}
		receipt.Promotion = qualify.HoldoutPromotionResult{
			Status:         "REJECTED_AA_NULL_DRIFT",
			Reason:         "A/A null-control blocks required by gate policy",
			CandidateState: "MEASURED_NO_PROMOTION",
			Selected:       refVariant,
		}
		return receipt, nil
	}

	res, err := qualify.SelectHoldoutPromoted(cfg.Identity, ev, verify)
	receipt.Promotion = res
	if err != nil {
		return receipt, err
	}

	if len(ev.HoldoutBlocks) >= 5 {
		if pairs, perr := HoldoutBlocksToABBAPairs(ev.HoldoutBlocks, cfg.PairTimeUnit, res.TokensMatched); perr == nil {
			if sum, serr := EvaluateABBAPairs(pairs); serr == nil {
				receipt.ABBASummary = &sum
			}
		}
	}
	return receipt, nil
}

// RegisterBaseline registers the initial megakernel code with f's
// [cudahotswap.Service], configures its 3D block and dynamic shared memory
// geometry, and seeds the multi-step rollback stack with version 1.
func (f *egraphFlightLoop) RegisterBaseline(ctx context.Context, code string) (cudahotswap.State, error) {
	if f == nil || f.service == nil {
		return cudahotswap.State{}, errors.New("register baseline: nil flight loop or service")
	}
	if code == "" {
		code = f.source
	}
	state, err := f.service.Register(ctx, f.name, cudahotswap.Source{
		Code:     code,
		Filename: f.name + ".cu",
		Entry:    f.name,
	}, "baseline")
	if err != nil {
		return cudahotswap.State{}, err
	}
	geom := &cudahotswap.LaunchGeometry{
		Grid:      cudashape.Grid3D{X: f.launchCfg.Grid.X, Y: f.launchCfg.Grid.Y, Z: f.launchCfg.Grid.Z},
		Block:     f.launchCfg.Block,
		SharedMem: f.launchCfg.SharedMemBytes,
	}
	if gState, gErr := f.service.SetGeometry(f.name, geom, "baseline-geometry"); gErr == nil {
		state = gState
	} else {
		return state, gErr
	}
	f.currentState = state
	f.versionStack = []uint64{state.Version}
	return state, nil
}

// BindGraphNode binds a CUDA graph kernel node to f's registered megakernel
// using 3D Grid, 3D Block, and SharedMemBytes via [cudahotswap.RegisterGraphWithConfig].
func (f *egraphFlightLoop) BindGraphNode(ctx context.Context, exec *cudagraph.Exec, node *cudagraph.Node, stream *cuda.Stream, args ...any) (cudahotswap.State, error) {
	if f == nil || f.service == nil {
		return cudahotswap.State{}, errors.New("bind graph node: nil flight loop or service")
	}
	return cudahotswap.RegisterGraphWithConfig(ctx, f.service, f.name, exec, node, stream, f.launchCfg, args...)
}

// RunIsolatedEpoch opens a fresh CUPTI profiling epoch via f's configured
// [EpochConfig], executes fn, and closes the epoch before returning so no
// profiling state remains active during subsequent hot-swaps.
func (f *egraphFlightLoop) RunIsolatedEpoch(ctx context.Context, fn func(context.Context, autotune.LiveCollector) error) (err error) {
	if f == nil {
		return errors.New("run isolated epoch: nil flight loop")
	}
	if f.newEpochCollector == nil {
		return errors.New("run isolated epoch: no epoch collector configured")
	}
	collector, closer, err := f.newEpochCollector()
	if err != nil {
		return fmt.Errorf("open profiling epoch: %w", err)
	}
	f.activeEpochs++
	defer func() {
		f.activeEpochs--
		if closer != nil {
			err = errors.Join(err, closer.Close())
		}
	}()
	return fn(ctx, collector)
}

// PromoteCandidate gates a candidate megakernel source through
// [SelectMeasuredWithHoldout] (3-block screen, >= 6-block held-out ABBA, A/A
// null calibration, and exact greedy token-ID equality) before pushing it into
// f's [cudahotswap.Service]. If postSwapCheck fails during [cudahotswap.Service.Push],
// the service automatically rolls back to the exact prior version.
func (f *egraphFlightLoop) PromoteCandidate(
	ctx context.Context,
	label string,
	candidateSource string,
	gateCfg HoldoutGateConfig,
	ev qualify.HoldoutCandidateEvidence,
	verify func(qualify.MeasuredPair) error,
	postSwapCheck cudahotswap.Checker,
) (*cudahotswap.PushResult, HoldoutReceipt, error) {
	if f == nil || f.service == nil {
		return nil, HoldoutReceipt{}, errors.New("promote candidate: nil flight loop or service")
	}
	if f.activeEpochs != 0 {
		return nil, HoldoutReceipt{}, fmt.Errorf("promote candidate: %d profiling epochs still active; close before swap", f.activeEpochs)
	}
	if err := VerifyTokenSequence(ev.ReferenceTokens, ev.CandidateTokens, ev.LogitMaxAbs); err != nil {
		receipt, _ := SelectMeasuredWithHoldout(gateCfg, ev, verify)
		return nil, receipt, err
	}
	receipt, err := SelectMeasuredWithHoldout(gateCfg, ev, verify)
	if err != nil {
		return nil, receipt, err
	}
	if receipt.Promotion.Status != "EXACT_HOLDOUT_PROMOTED" || receipt.Promotion.CandidateState != "PROMOTED_HOLDOUT_VERIFIED" {
		return nil, receipt, fmt.Errorf("promote candidate: gate rejected (%s: %s)", receipt.Promotion.Status, receipt.Promotion.Reason)
	}

	checker := postSwapCheck
	if checker == nil {
		checker = func(_ context.Context, state cudahotswap.State) error {
			if state.Version == 0 {
				return errors.New("invalid zero version after push")
			}
			return nil
		}
	}
	if _, err := f.service.SetCheck(f.name, cudahotswap.CheckConfig{
		Checker: checker,
	}, label+"-check"); err != nil {
		return nil, receipt, err
	}

	pushRes, pushErr := f.service.Push(ctx, f.name, cudahotswap.Source{
		Code:     candidateSource,
		Filename: f.name + ".cu",
		Entry:    f.name,
	}, label)
	if pushRes != nil && pushRes.Selected.Version != 0 {
		f.currentState = pushRes.Selected
	}
	if pushErr != nil {
		return pushRes, receipt, pushErr
	}
	if pushRes == nil || !pushRes.Published || pushRes.RolledBack {
		return pushRes, receipt, errors.New("promote candidate: candidate was not published")
	}
	f.versionStack = append(f.versionStack, pushRes.Selected.Version)
	return pushRes, receipt, nil
}

// RestorePrevious rolls back the active megakernel in f's [cudahotswap.Service]
// to the immediately preceding retained version on the promotion stack
// (for example, v1 -> v2 -> v3 -> RestorePrevious selects v2, not v1).
func (f *egraphFlightLoop) RestorePrevious(ctx context.Context, label string) (cudahotswap.State, error) {
	if f == nil || f.service == nil {
		return cudahotswap.State{}, errors.New("restore previous: nil flight loop or service")
	}
	if len(f.versionStack) < 2 {
		return cudahotswap.State{}, errors.New("restore previous: no prior version on stack")
	}
	prevVersion := f.versionStack[len(f.versionStack)-2]
	state, err := f.service.Select(ctx, f.name, prevVersion, label)
	if err != nil {
		return cudahotswap.State{}, err
	}
	f.versionStack = f.versionStack[:len(f.versionStack)-1]
	f.currentState = state
	return state, nil
}

// CurrentState returns the currently active [cudahotswap.State] tracked by f.
func (f *egraphFlightLoop) CurrentState() cudahotswap.State {
	if f == nil {
		return cudahotswap.State{}
	}
	return f.currentState
}

// VersionHistory returns a copy of the active version stack from oldest to current.
func (f *egraphFlightLoop) VersionHistory() []uint64 {
	if f == nil {
		return nil
	}
	return slices.Clone(f.versionStack)
}
