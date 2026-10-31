package main

import (
	"context"
	"errors"
	"fmt"
	"math"
	"slices"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudahotswap"
	"github.com/tmc/lib/nvidia/x/egraph"
	"github.com/tmc/lib/nvidia/x/egraph/autotune"
	"github.com/tmc/lib/nvidia/x/egraph/hopper"
	"github.com/tmc/lib/nvidia/x/egraph/model"
	"github.com/tmc/lib/nvidia/x/egraph/qualify"
	"github.com/tmc/lib/nvidia/x/egraph/specdecode"
)

// AutonomousDecodeConfig configures the end-to-end autonomous flight pipeline
// for a pair of adjacent LLaMA decoder layers in llama-infer.
type AutonomousDecodeConfig struct {
	Model            *ModelConfig
	QKVBias          bool
	LayerL           int
	LayerNext        int
	BatchM           int
	Buckets          []int
	Target           string
	Device           string
	DeviceSMs        int
	DriverVersion    string
	CUDAVersion      string
	WeightDigest     string
	CacheDir         string
	Cache            *autotune.VerifiedPGOCache
	BeamWidth        int
	PTXASCompiler    qualify.PTXASCompiler
	PTXASGate        qualify.StaticGate
	MaxPTXASIters    int
	HotSwapService   *cudahotswap.Service
	BucketController *specdecode.AdaptiveBucketController
	BucketConfig     *specdecode.AdaptiveBucketConfig
	KernelName       string
}

// AutonomousDecodeEngine wires [model.RunAutonomousFlightPipeline],
// [buildCrossLayerDecodeMegakernel], [specdecode.AdaptiveBucketController],
// [autotune.VerifiedPGOCache], and [cudahotswap.Service] for a LLaMA decoder
// layer boundary.
type AutonomousDecodeEngine struct {
	Config               AutonomousDecodeConfig
	Cache                *autotune.VerifiedPGOCache
	BucketController     *specdecode.AdaptiveBucketController
	HotSwapService       *cudahotswap.Service
	PostGraph            *egraph.EGraph
	PostRoot             egraph.Id
	LayerLPostPlan       *hopper.Plan
	LayerNextAttnInPlan  *hopper.Plan
	CrossLayerPlan       *hopper.Plan
	CrossLayerMegakernel hopper.Megakernel
	BucketPlans          map[int]*hopper.Plan
	BucketMegakernels    map[int]hopper.Megakernel
	Glue                 hopper.CrossLayerGlue
	KernelName           string
}

// NewAutonomousDecodeEngine constructs an [AutonomousDecodeEngine] for the
// inter-layer boundary (cfg.LayerL -> cfg.LayerNext).
func NewAutonomousDecodeEngine(cfg AutonomousDecodeConfig) (*AutonomousDecodeEngine, error) {
	if cfg.Model == nil {
		return nil, errors.New("autonomous decode engine: nil model config")
	}
	if cfg.LayerL == 0 && cfg.LayerNext == 0 {
		cfg.LayerNext = 1
	}
	if cfg.BatchM <= 0 {
		cfg.BatchM = 1
	}
	target := strings.TrimSpace(cfg.Target)
	if target == "" {
		target = "sm_121a"
	}
	cfg.Target = target
	if strings.TrimSpace(cfg.Device) == "" {
		cfg.Device = target
	}
	if strings.TrimSpace(cfg.DriverVersion) == "" {
		cfg.DriverVersion = "570.86.10"
	}
	if strings.TrimSpace(cfg.CUDAVersion) == "" {
		cfg.CUDAVersion = "12.8"
	}
	if strings.TrimSpace(cfg.WeightDigest) == "" {
		cfg.WeightDigest = fmt.Sprintf("llama_h%d_i%d_l%d",
			cfg.Model.HiddenSize, cfg.Model.IntermediateSize, cfg.Model.NumHiddenLayers)
	}
	if cfg.BeamWidth <= 0 {
		cfg.BeamWidth = 4
	}

	crossPlan, crossMega, err := buildCrossLayerDecodeMegakernel(
		cfg.Model, cfg.QKVBias, cfg.LayerL, cfg.LayerNext, cfg.BatchM, target,
	)
	if err != nil {
		return nil, fmt.Errorf("autonomous decode engine: %w", err)
	}

	bucketPlans := map[int]*hopper.Plan{
		cfg.BatchM: crossPlan,
	}
	bucketMegas := map[int]hopper.Megakernel{
		cfg.BatchM: crossMega,
	}
	for _, m := range cfg.Buckets {
		if m <= 0 {
			return nil, fmt.Errorf("autonomous decode engine: invalid bucket M=%d", m)
		}
		if _, ok := bucketPlans[m]; ok {
			continue
		}
		bPlan, bMega, err := buildCrossLayerDecodeMegakernel(
			cfg.Model, cfg.QKVBias, cfg.LayerL, cfg.LayerNext, m, target,
		)
		if err != nil {
			return nil, fmt.Errorf("autonomous decode engine bucket M=%d: %w", m, err)
		}
		bucketPlans[m] = bPlan
		bucketMegas[m] = bMega
	}

	prefixL := fmt.Sprintf("l%d", cfg.LayerL)
	prefixNext := fmt.Sprintf("l%d", cfg.LayerNext)
	_, postL, err := buildLayerPlans(cfg.Model, cfg.QKVBias, false, prefixL, cfg.BatchM, hopper.F32, target, true)
	if err != nil {
		return nil, fmt.Errorf("autonomous decode engine post plan: %w", err)
	}
	attnInNext, _, err := buildLayerPlans(cfg.Model, cfg.QKVBias, false, prefixNext, cfg.BatchM, hopper.F32, target, true)
	if err != nil {
		return nil, fmt.Errorf("autonomous decode engine next attnIn plan: %w", err)
	}

	postGraph, postRoot := buildLayerPostEGraph(cfg.Model, cfg.BatchM, hopper.F32, true)

	cache := cfg.Cache
	if cache == nil && strings.TrimSpace(cfg.CacheDir) != "" {
		cache = autotune.NewVerifiedPGOCache(cfg.CacheDir)
	}

	bucketCtrl := cfg.BucketController
	if bucketCtrl == nil {
		var bcfg specdecode.AdaptiveBucketConfig
		if cfg.BucketConfig != nil {
			bcfg = *cfg.BucketConfig
		} else {
			bcfg = specdecode.AdaptiveBucketConfig{
				HysteresisSteps: 2,
				InitialBucketM:  cfg.BatchM,
			}
		}
		bucketCtrl, err = specdecode.NewAdaptiveBucketController(bcfg)
		if err != nil {
			return nil, fmt.Errorf("autonomous decode engine bucket controller: %w", err)
		}
	}

	glue := hopper.CrossLayerGlue{
		Name:               crossMega.Name,
		LayerLPrefix:       prefixL + "_",
		LayerNextPrefix:    prefixNext + "_",
		BridgeInput:        "xn",
		RetainBridgeOutput: true,
	}
	kernelName := strings.TrimSpace(cfg.KernelName)
	if kernelName == "" {
		kernelName = crossMega.Name
	}

	return &AutonomousDecodeEngine{
		Config:               cfg,
		Cache:                cache,
		BucketController:     bucketCtrl,
		HotSwapService:       cfg.HotSwapService,
		PostGraph:            postGraph,
		PostRoot:             postRoot,
		LayerLPostPlan:       postL,
		LayerNextAttnInPlan:  attnInNext,
		CrossLayerPlan:       crossPlan,
		CrossLayerMegakernel: crossMega,
		BucketPlans:          bucketPlans,
		BucketMegakernels:    bucketMegas,
		Glue:                 glue,
		KernelName:           kernelName,
	}, nil
}

func buildLayerPostEGraph(cfg *ModelConfig, m int, attnDType hopper.DType, chain bool) (*egraph.EGraph, egraph.Id) {
	h := cfg.HiddenSize
	qDim := cfg.NumAttentionHeads * cfg.HeadDim()
	inter := cfg.IntermediateSize
	eps := cfg.RmsNormEps

	b := hopper.NewBuilder()
	ao := b.Input("attn", attnDType, m, qDim)
	if attnDType != hopper.BF16 {
		ao = b.Cast(hopper.BF16, ao)
	}
	x := b.Input("x", hopper.F32, m, h)
	hid := b.Map(hopper.MustLambda("(add $0 $1)"), b.MatMul(ao, b.Input("w_o", hopper.BF16, h, qDim)), x)
	norm := b.Cast(hopper.BF16, b.RMSNorm(hid, b.Input("ffn_norm", hopper.F32, h), h, eps))
	act := b.SwiGLU(b.MatMul(norm, b.Input("w_gu", hopper.BF16, 2*inter, h)))
	out := b.Map(hopper.MustLambda("(add $0 $1)"), b.MatMul(act, b.Input("w_down", hopper.BF16, h, inter)), hid)
	if chain {
		out = b.Cast(hopper.BF16, b.RMSNorm(out, b.Input("next_norm", hopper.F32, h), h, eps))
	}
	return b.G, out
}

// CacheKey returns the hardware-and-model-bound [autotune.PGOCacheKey] for the
// engine's cross-layer decode plan at shape bucket bucketM.
func (e *AutonomousDecodeEngine) CacheKey(bucketM int) autotune.PGOCacheKey {
	if bucketM <= 0 {
		if e.BucketController != nil && e.BucketController.CurrentBucket().BucketM > 0 {
			bucketM = e.BucketController.CurrentBucket().BucketM
		} else if e.Config.BatchM > 0 {
			bucketM = e.Config.BatchM
		} else {
			bucketM = 1
		}
	}
	return autotune.PGOCacheKey{
		Arch:          e.Config.Target,
		DriverVersion: e.Config.DriverVersion,
		CUDAVersion:   e.Config.CUDAVersion,
		WeightDigest:  e.Config.WeightDigest,
		PlanKey:       e.KernelName,
		ShapeBucket:   bucketM,
	}
}

// Run executes [model.RunAutonomousFlightPipeline] for the engine's decoder
// layer boundary, wiring the verified PGO cache, structural beam extraction,
// cross-layer megakernel, ptxas loop refinement, PC stall attribution, holdout
// promotion gate, hot-swap service, and adaptive bucket controller.
func (e *AutonomousDecodeEngine) Run(ctx context.Context, req model.AutonomousEvalRequest) (*model.AutonomousPipelineResult, error) {
	if e == nil {
		return nil, errors.New("autonomous decode engine: nil engine")
	}
	if req.CrossLayerPost == nil {
		req.CrossLayerPost = e.LayerLPostPlan
	}
	if req.CrossLayerNextAttnIn == nil {
		req.CrossLayerNextAttnIn = e.LayerNextAttnInPlan
	}
	if req.CrossLayerGlue == (hopper.CrossLayerGlue{}) {
		req.CrossLayerGlue = e.Glue
	}
	if req.KernelName == "" {
		req.KernelName = e.KernelName
	}

	bucketM := e.Config.BatchM
	if e.BucketController != nil && e.BucketController.CurrentBucket().BucketM > 0 {
		bucketM = e.BucketController.CurrentBucket().BucketM
	}
	pCfg := model.AutonomousPipelineConfig{
		Cache:            e.Cache,
		CacheKey:         e.CacheKey(bucketM),
		Device:           e.Config.Device,
		Arch:             e.Config.Target,
		DeviceSMs:        e.Config.DeviceSMs,
		BeamWidth:        e.Config.BeamWidth,
		PTXASCompiler:    e.Config.PTXASCompiler,
		PTXASGate:        e.Config.PTXASGate,
		MaxPTXASIters:    e.Config.MaxPTXASIters,
		HotSwapService:   e.HotSwapService,
		BucketController: e.BucketController,
		KernelName:       e.KernelName,
	}
	return model.RunAutonomousFlightPipeline(
		ctx, e.PostGraph, e.PostRoot, e.KernelName, hopper.BF16, pCfg, req,
	)
}

// StepBuckets records a speculative verification step on the engine's
// [specdecode.AdaptiveBucketController] and returns the resulting transition.
func (e *AutonomousDecodeEngine) StepBuckets(ctx context.Context, batchSize, draftedTokens, acceptedTokens int) (specdecode.BucketTransition, error) {
	if e == nil || e.BucketController == nil {
		return specdecode.BucketTransition{}, errors.New("autonomous decode engine: nil bucket controller")
	}
	return e.BucketController.Step(ctx, batchSize, draftedTokens, acceptedTokens)
}

// ObserveStep records a decode or speculative verification step with
// activeTokens total tokens and observed acceptanceRate in [0, 1], updating
// the engine's [specdecode.AdaptiveBucketController] and returning the
// resulting [specdecode.BucketTransition].
func (e *AutonomousDecodeEngine) ObserveStep(activeTokens int, acceptanceRate float64) (specdecode.BucketTransition, error) {
	if e == nil || e.BucketController == nil {
		return specdecode.BucketTransition{}, errors.New("autonomous decode engine: nil bucket controller")
	}
	if activeTokens <= 0 {
		activeTokens = 1
	}
	priorBucket := e.BucketController.CurrentBucket()
	rate := min(max(acceptanceRate, 0.0), 1.0)
	drafted := max(0, activeTokens-1)
	accepted := int(math.Round(rate * float64(drafted)))
	var trans specdecode.BucketTransition
	if drafted > 0 {
		var err error
		trans, err = e.BucketController.Step(context.Background(), 1, drafted, accepted)
		if err != nil {
			return trans, err
		}
	} else {
		trans = specdecode.BucketTransition{
			PriorBucket: priorBucket,
			NewBucket:   priorBucket,
			State:       e.BucketController.CurrentState(),
			Geometry:    e.BucketController.CurrentGeometry(),
		}
	}

	buckets := e.availableBuckets()
	targetM := buckets[len(buckets)-1]
	if activeTokens <= 1 {
		targetM = buckets[0]
	} else if acceptanceRate > 0 && rate < 0.30 {
		want := 1
		if rate >= 0.15 {
			want = 2
		}
		for _, m := range buckets {
			if m >= want {
				targetM = m
				break
			}
		}
	} else {
		for _, m := range buckets {
			if m >= activeTokens {
				targetM = m
				break
			}
		}
	}

	if e.BucketController.CurrentBucket().BucketM != targetM {
		spec := specdecode.BucketSpec{BucketM: targetM}
		if defSpecs, dErr := specdecode.DefaultBucketSpecs(); dErr == nil {
			for _, ds := range defSpecs {
				if ds.BucketM == targetM {
					spec = ds
					break
				}
			}
		}
		var err error
		trans, err = e.BucketController.SwitchBucket(context.Background(), spec)
		if err != nil {
			return trans, err
		}
	}
	trans.PriorBucket = priorBucket
	trans.Swapped = priorBucket.BucketM != trans.NewBucket.BucketM
	return trans, nil
}

func (e *AutonomousDecodeEngine) availableBuckets() []int {
	var out []int
	for _, m := range e.Config.Buckets {
		if m > 0 && !slices.Contains(out, m) {
			out = append(out, m)
		}
	}
	for m := range e.BucketPlans {
		if m > 0 && !slices.Contains(out, m) {
			out = append(out, m)
		}
	}
	if len(out) == 0 {
		out = []int{1, 2, 4, 8}
	}
	slices.Sort(out)
	return out
}

// SelectBucketMegakernel observes a decode or speculative verification step
// with draftTokens proposed draft tokens and acceptanceRate, selects the
// active bucket M, and returns the bucket's cross-layer [*hopper.Plan] and
// [hopper.Megakernel].
func (e *AutonomousDecodeEngine) SelectBucketMegakernel(draftTokens int, acceptanceRate float64) (int, *hopper.Plan, hopper.Megakernel, specdecode.BucketTransition, error) {
	if e == nil {
		return 0, nil, hopper.Megakernel{}, specdecode.BucketTransition{}, errors.New("autonomous decode engine: nil engine")
	}
	activeTokens := max(1, 1+draftTokens)
	trans, err := e.ObserveStep(activeTokens, acceptanceRate)
	if err != nil {
		return 0, nil, hopper.Megakernel{}, trans, err
	}
	bucketM := trans.NewBucket.BucketM
	if bucketM <= 0 {
		bucketM = max(1, e.Config.BatchM)
	}
	if e.BucketPlans == nil {
		e.BucketPlans = make(map[int]*hopper.Plan)
	}
	if e.BucketMegakernels == nil {
		e.BucketMegakernels = make(map[int]hopper.Megakernel)
	}
	plan, okP := e.BucketPlans[bucketM]
	mega, okM := e.BucketMegakernels[bucketM]
	if !okP || !okM || plan == nil {
		var bErr error
		plan, mega, bErr = buildCrossLayerDecodeMegakernel(
			e.Config.Model, e.Config.QKVBias, e.Config.LayerL, e.Config.LayerNext, bucketM, e.Config.Target,
		)
		if bErr != nil {
			return 0, nil, hopper.Megakernel{}, trans, bErr
		}
		e.BucketPlans[bucketM] = plan
		e.BucketMegakernels[bucketM] = mega
	}
	e.CrossLayerPlan = plan
	e.CrossLayerMegakernel = mega
	return bucketM, plan, mega, trans, nil
}

// ObserveAndSelectBucket is an alias for [AutonomousDecodeEngine.SelectBucketMegakernel].
func (e *AutonomousDecodeEngine) ObserveAndSelectBucket(draftTokens int, acceptanceRate float64) (int, *hopper.Plan, hopper.Megakernel, specdecode.BucketTransition, error) {
	return e.SelectBucketMegakernel(draftTokens, acceptanceRate)
}

// RestorePrevious rolls the engine's hot-swap service and bucket controller
// back to the immediate prior retained version using lastResult.
func (e *AutonomousDecodeEngine) RestorePrevious(ctx context.Context, lastResult *model.AutonomousPipelineResult) (cudahotswap.State, error) {
	if lastResult != nil {
		return lastResult.RestorePrevious(ctx)
	}
	if e == nil || e.BucketController == nil {
		return cudahotswap.State{}, errors.New("autonomous decode engine: nil result and bucket controller")
	}
	return e.BucketController.RestorePrevious(ctx)
}

// runAutonomousDecodePipeline constructs an [AutonomousDecodeEngine] from cfg
// and runs [AutonomousDecodeEngine.Run].
func runAutonomousDecodePipeline(
	ctx context.Context,
	cfg AutonomousDecodeConfig,
	req model.AutonomousEvalRequest,
) (*model.AutonomousPipelineResult, error) {
	eng, err := NewAutonomousDecodeEngine(cfg)
	if err != nil {
		return nil, err
	}
	return eng.Run(ctx, req)
}
