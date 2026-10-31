package specdecode

import (
	"cmp"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/tmc/lib/nvidia/x/cudahotswap"
	"github.com/tmc/lib/nvidia/x/egraph/autotune"
	"github.com/tmc/lib/nvidia/x/egraph/hopper"
)

const (
	// RegimeSkinnyGEMV aliases [RegimeSkinnyDecode] for M in {1, 2} skinny GEMV
	// decode schedules.
	RegimeSkinnyGEMV BucketRegime = RegimeSkinnyDecode

	// RegimeTensorCoreTile aliases [RegimeMTPTreeVerify] for M in {4, 8}
	// speculative tree verification schedules.
	RegimeTensorCoreTile BucketRegime = RegimeMTPTreeVerify
)

// BucketPlanBuilder builds (or lowers) the cross-layer decode plan and
// persistent [hopper.Megakernel] for a specific token batch bucket bucketM and
// [BucketRegime].
type BucketPlanBuilder func(bucketM int, regime BucketRegime) (*hopper.Plan, hopper.Megakernel, error)

// BucketHotSwapFunc is an optional callback invoked when
// [MultiBucketMegakernelTable.ObserveAndSelect] transitions to a new bucket so
// the caller ([cudahotswap.Service] or a decode graph) can hot-swap the active
// graph kernel node in place.
type BucketHotSwapFunc func(ctx context.Context, entry *BucketMegakernelEntry, transition BucketTransition) error

// BucketMegakernelEntry holds the pre-warmed [BucketSpec], cross-layer
// [hopper.Plan], persistent [hopper.Megakernel], emitted CUDA translation unit,
// and [autotune.VerifiedPGOCache] metadata for one shape bucket M.
type BucketMegakernelEntry struct {
	Spec       BucketSpec
	Plan       *hopper.Plan
	Megakernel hopper.Megakernel
	CUDASource string
	CacheKey   autotune.PGOCacheKey
	CacheHit   bool
	Cubin      []byte
}

// MultiBucketTableConfig configures a [MultiBucketMegakernelTable].
type MultiBucketTableConfig struct {
	Arch          string
	DriverVersion string
	CUDAVersion   string
	WeightDigest  string
	PlanKeyPrefix string
	Cache         *autotune.VerifiedPGOCache
	Controller    *AdaptiveBucketController
	Builder       BucketPlanBuilder
	OnTransition  BucketHotSwapFunc
}

type cachedBucketSchedulePayload struct {
	Plan       *hopper.Plan      `json:"plan,omitempty"`
	Megakernel hopper.Megakernel `json:"megakernel,omitempty"`
	CUDASource string            `json:"cuda_source,omitempty"`
}

// MultiBucketMegakernelTable manages a pre-warmed table of cross-layer plans
// and persistent [hopper.Megakernel] instances across speculative verification
// shape buckets (e.g. M in {1, 2, 4, 8}), integrated with
// [AdaptiveBucketController] and [autotune.VerifiedPGOCache].
type MultiBucketMegakernelTable struct {
	mu           sync.RWMutex
	cfg          MultiBucketTableConfig
	cache        *autotune.VerifiedPGOCache
	controller   *AdaptiveBucketController
	builder      BucketPlanBuilder
	onTransition BucketHotSwapFunc
	entries      map[int]*BucketMegakernelEntry
	orderedM     []int
	hotSwaps     atomic.Int64
}

// defaultMultiBucketSpecs returns the canonical 4-bucket speculative
// verification table M in {1, 2, 4, 8}.
func defaultMultiBucketSpecs() ([]BucketSpec, error) {
	all, err := DefaultBucketSpecs()
	if err != nil {
		return nil, err
	}
	var out []BucketSpec
	for _, b := range all {
		switch b.BucketM {
		case 1, 2, 4, 8:
			out = append(out, b)
		}
	}
	return out, nil
}

// NewMultiBucketMegakernelTable pre-warms all buckets configured on
// cfg.Controller (defaulting to M in {1, 2, 4, 8} via
// [NewAdaptiveBucketController] when cfg.Controller is nil), restoring entries
// from cfg.Cache on verified PGO hits or building them via cfg.Builder.
func NewMultiBucketMegakernelTable(cfg MultiBucketTableConfig) (*MultiBucketMegakernelTable, error) {
	ctrl := cfg.Controller
	if ctrl == nil {
		specs, err := defaultMultiBucketSpecs()
		if err != nil {
			return nil, err
		}
		ctrl, err = NewAdaptiveBucketController(AdaptiveBucketConfig{
			Buckets:               specs,
			EMAAlpha:              0.50,
			HysteresisSteps:       1,
			LowAcceptanceThreshold:  0.30,
			HighAcceptanceThreshold: 0.65,
			InitialBucketM:        1,
		})
		if err != nil {
			return nil, fmt.Errorf("multi-bucket megakernel table: default controller: %w", err)
		}
	}

	builder := cfg.Builder
	if builder == nil {
		arch := strings.TrimSpace(cfg.Arch)
		if arch == "" {
			arch = "sm_121a"
		}
		builder = DefaultCrossLayerBucketBuilder(256, 512, 256, arch)
	}

	prefix := strings.TrimSpace(cfg.PlanKeyPrefix)
	if prefix == "" {
		prefix = "cross_layer_decode"
	}

	buckets := ctrl.cfg.Buckets
	if len(buckets) == 0 {
		var err error
		buckets, err = defaultMultiBucketSpecs()
		if err != nil {
			return nil, err
		}
	}

	entries := make(map[int]*BucketMegakernelEntry, len(buckets))
	orderedM := make([]int, 0, len(buckets))

	for i := range buckets {
		spec := buckets[i]
		m := spec.BucketM
		key := autotune.PGOCacheKey{
			Arch:          cfg.Arch,
			DriverVersion: cfg.DriverVersion,
			CUDAVersion:   cfg.CUDAVersion,
			WeightDigest:  cfg.WeightDigest,
			PlanKey:       fmt.Sprintf("%s_m%d", prefix, m),
			ShapeBucket:   m,
		}

		var (
			plan     *hopper.Plan
			mega     hopper.Megakernel
			cudaSrc  string
			cubin    []byte
			cacheHit bool
		)

		if cfg.Cache != nil && key.Validate() == nil {
			proof, cachedCubin, err := cfg.Cache.LoadVerified(key)
			if err == nil {
				cacheHit = true
				cubin = slices.Clone(cachedCubin)
				if len(proof.ScheduleJSON) > 0 {
					var payload cachedBucketSchedulePayload
					if uErr := json.Unmarshal(proof.ScheduleJSON, &payload); uErr == nil && payload.Plan != nil && len(payload.Megakernel.Stages) > 0 {
						plan = payload.Plan
						mega = payload.Megakernel
						cudaSrc = payload.CUDASource
					} else if s := string(proof.ScheduleJSON); strings.Contains(s, "__global__") {
						cudaSrc = s
					}
				}
			} else if !errors.Is(err, os.ErrNotExist) {
				return nil, fmt.Errorf("multi-bucket megakernel table M=%d cache lookup: %w", m, err)
			}
		}

		if plan == nil || len(mega.Stages) == 0 {
			builtPlan, builtMega, err := builder(m, spec.Regime)
			if err != nil {
				return nil, fmt.Errorf("multi-bucket megakernel table M=%d build: %w", m, err)
			}
			if builtPlan == nil {
				return nil, fmt.Errorf("multi-bucket megakernel table M=%d: builder returned nil plan", m)
			}
			plan = builtPlan
			mega = builtMega
		}

		if cudaSrc == "" {
			src, err := mega.Source()
			if err != nil {
				return nil, fmt.Errorf("multi-bucket megakernel table M=%d source: %w", m, err)
			}
			cudaSrc = src
		}

		grid, block := mega.Grid()
		if grid[0] > 0 && block[0] > 0 {
			spec.Candidate.Grid = grid
			spec.Candidate.Block = block
			spec.Candidate.SharedBytes = mega.SharedBytes()
		}
		if mega.Name != "" {
			spec.Candidate.Name = mega.Name
			spec.Candidate.Source = cudahotswap.Source{
				Code:     cudaSrc,
				Filename: mega.Name + ".cu",
				Entry:    mega.Name,
			}
		}
		ctrl.cfg.Buckets[i] = spec
		if ctrl.currentBucket.BucketM == m {
			ctrl.currentBucket = spec
			ctrl.currentGeometry = spec.LaunchGeometry()
		}

		entries[m] = &BucketMegakernelEntry{
			Spec:       spec,
			Plan:       plan,
			Megakernel: mega,
			CUDASource: cudaSrc,
			CacheKey:   key,
			CacheHit:   cacheHit,
			Cubin:      cubin,
		}
		orderedM = append(orderedM, m)
	}

	slices.Sort(orderedM)

	return &MultiBucketMegakernelTable{
		cfg:          cfg,
		cache:        cfg.Cache,
		controller:   ctrl,
		builder:      builder,
		onTransition: cfg.OnTransition,
		entries:      entries,
		orderedM:     orderedM,
	}, nil
}

// Controller returns the underlying [AdaptiveBucketController].
func (t *MultiBucketMegakernelTable) Controller() *AdaptiveBucketController {
	if t == nil {
		return nil
	}
	return t.controller
}

// HotSwaps returns the total number of bucket hot-swap transitions triggered by
// [MultiBucketMegakernelTable.ObserveAndSelect].
func (t *MultiBucketMegakernelTable) HotSwaps() int {
	if t == nil {
		return 0
	}
	return int(t.hotSwaps.Load())
}

// Entry returns the pre-warmed [BucketMegakernelEntry] for bucketM.
func (t *MultiBucketMegakernelTable) Entry(bucketM int) (*BucketMegakernelEntry, bool) {
	if t == nil {
		return nil, false
	}
	t.mu.RLock()
	defer t.mu.RUnlock()
	e, ok := t.entries[bucketM]
	return e, ok
}

// Entries returns all pre-warmed [BucketMegakernelEntry] items in ascending
// BucketM order.
func (t *MultiBucketMegakernelTable) Entries() []*BucketMegakernelEntry {
	if t == nil {
		return nil
	}
	t.mu.RLock()
	defer t.mu.RUnlock()
	out := make([]*BucketMegakernelEntry, 0, len(t.orderedM))
	for _, m := range t.orderedM {
		if e, ok := t.entries[m]; ok {
			out = append(out, e)
		}
	}
	return out
}

// ObserveAndSelect observes a decode verification step with draftTokens
// speculative draft tokens and acceptanceRate in [0, 1], selects the target
// bucket via the underlying [AdaptiveBucketController], and invokes
// OnTransition when a bucket transition occurs.
func (t *MultiBucketMegakernelTable) ObserveAndSelect(ctx context.Context, draftTokens int, acceptanceRate float64) (*BucketMegakernelEntry, BucketTransition, error) {
	if t == nil {
		return nil, BucketTransition{}, errors.New("observe and select: nil table")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	activeTokens := max(1, 1+draftTokens)

	t.mu.Lock()
	trans, err := t.controller.ObserveStep(ctx, activeTokens, acceptanceRate)
	if err != nil {
		t.mu.Unlock()
		return nil, trans, err
	}
	targetM := trans.NewBucket.BucketM
	entry, ok := t.entries[targetM]
	if !ok && len(t.orderedM) > 0 {
		// Pick smallest pre-warmed bucket >= targetM, or largest available.
		chosenM := t.orderedM[len(t.orderedM)-1]
		for _, m := range t.orderedM {
			if m >= targetM {
				chosenM = m
				break
			}
		}
		entry = t.entries[chosenM]
	}
	t.mu.Unlock()

	if entry == nil {
		return nil, trans, fmt.Errorf("observe and select: bucket M=%d not found in table", targetM)
	}

	if trans.Swapped {
		t.hotSwaps.Add(1)
		if t.onTransition != nil {
			if err := t.onTransition(ctx, entry, trans); err != nil {
				return entry, trans, err
			}
		}
	}
	return entry, trans, nil
}

// PersistEntry stores a hardware-qualified bucket entry and its compiled cubin
// into the table's [autotune.VerifiedPGOCache].
func (t *MultiBucketMegakernelTable) PersistEntry(bucketM int, cubin []byte, proof autotune.PGOPromotionProof) error {
	if t == nil {
		return errors.New("persist entry: nil table")
	}
	if t.cache == nil {
		return errors.New("persist entry: nil VerifiedPGOCache")
	}

	t.mu.Lock()
	defer t.mu.Unlock()

	entry, ok := t.entries[bucketM]
	if !ok || entry == nil {
		return fmt.Errorf("persist entry: bucket M=%d not found", bucketM)
	}
	if len(cubin) == 0 {
		return fmt.Errorf("persist entry M=%d: %w", bucketM, autotune.ErrPGODigestMismatch)
	}

	key := entry.CacheKey
	proof.Key = key
	if proof.CandidateName == "" {
		proof.CandidateName = entry.Megakernel.Name
		if proof.CandidateName == "" {
			proof.CandidateName = fmt.Sprintf("bucket_m%d", bucketM)
		}
	}
	if len(proof.ScheduleJSON) == 0 {
		payload, err := json.Marshal(cachedBucketSchedulePayload{
			Plan:       entry.Plan,
			Megakernel: entry.Megakernel,
			CUDASource: entry.CUDASource,
		})
		if err == nil {
			proof.ScheduleJSON = payload
		}
	}
	if proof.CubinSHA256 == ([32]byte{}) {
		proof.CubinSHA256 = sha256.Sum256(cubin)
	}
	if proof.SourceSHA256 == ([32]byte{}) {
		proof.SourceSHA256 = sha256.Sum256([]byte(entry.CUDASource))
	}
	if proof.PromotedAt.IsZero() {
		proof.PromotedAt = time.Now().UTC()
	}

	if err := t.cache.Store(key, proof, cubin); err != nil {
		return err
	}
	entry.Cubin = slices.Clone(cubin)
	return nil
}

// ObserveStep updates the controller's acceptance-rate EMA from activeTokens
// (1 bonus token + draftTokens) and acceptanceRate in [0, 1], selects the
// target bucket for the verification step, and switches buckets when the
// hysteresis gate is satisfied.
func (c *AdaptiveBucketController) ObserveStep(ctx context.Context, activeTokens int, acceptanceRate float64) (BucketTransition, error) {
	if c == nil {
		return BucketTransition{}, errors.New("adaptive bucket observe step: nil controller")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	activeTokens = max(activeTokens, 1)
	rate := min(max(acceptanceRate, 0.0), 1.0)
	if math.IsNaN(rate) {
		rate = 0.0
	}

	if !c.emaInitialized {
		c.acceptanceEMA = rate
		c.emaInitialized = true
	} else {
		alpha := c.cfg.EMAAlpha
		c.acceptanceEMA = alpha*rate + (1.0-alpha)*c.acceptanceEMA
	}
	c.totalSteps++

	rec := c.recommendBucketForTokens(activeTokens)
	if rec.BucketM == c.currentBucket.BucketM && rec.Regime == c.currentBucket.Regime {
		c.pendingBucketM = 0
		c.pendingSteps = 0
		return BucketTransition{
			Swapped:     false,
			PriorBucket: c.currentBucket,
			NewBucket:   c.currentBucket,
			State:       c.currentState,
			Geometry:    c.currentGeometry,
		}, nil
	}

	if c.pendingBucketM == rec.BucketM {
		c.pendingSteps++
	} else {
		c.pendingBucketM = rec.BucketM
		c.pendingSteps = 1
	}

	if c.pendingSteps < c.cfg.HysteresisSteps {
		return BucketTransition{
			Swapped:     false,
			PriorBucket: c.currentBucket,
			NewBucket:   c.currentBucket,
			State:       c.currentState,
			Geometry:    c.currentGeometry,
		}, nil
	}

	c.pendingBucketM = 0
	c.pendingSteps = 0
	return c.SwitchBucket(ctx, rec)
}

func (c *AdaptiveBucketController) recommendBucketForTokens(activeTokens int) BucketSpec {
	if c == nil || len(c.cfg.Buckets) == 0 {
		return BucketSpec{}
	}
	ema := c.acceptanceEMA
	lowThresh := c.cfg.LowAcceptanceThreshold
	highThresh := c.cfg.HighAcceptanceThreshold

	var targetM int
	var targetRegime BucketRegime
	switch {
	case ema < lowThresh || activeTokens <= 1:
		targetRegime = RegimeSkinnyDecode
		if activeTokens >= 2 && ema >= lowThresh*0.5 {
			targetM = 2
		} else {
			targetM = 1
		}
	case activeTokens == 2:
		targetRegime = RegimeSkinnyDecode
		targetM = 2
	case ema < highThresh:
		targetRegime = RegimeMTPTreeVerify
		targetM = min(max(activeTokens, 4), 4)
	default:
		targetRegime = RegimeMTPTreeVerify
		targetM = max(activeTokens, 4)
	}

	if targetM >= 16 {
		targetRegime = RegimeTilePrefill
	}

	var regimeBuckets []BucketSpec
	for _, b := range c.cfg.Buckets {
		if b.Regime == targetRegime {
			regimeBuckets = append(regimeBuckets, b)
		}
	}
	if len(regimeBuckets) > 0 {
		for _, b := range regimeBuckets {
			if b.BucketM >= targetM {
				return b
			}
		}
		return regimeBuckets[len(regimeBuckets)-1]
	}

	for _, b := range c.cfg.Buckets {
		if b.BucketM >= targetM {
			return b
		}
	}
	return c.cfg.Buckets[len(c.cfg.Buckets)-1]
}

// Changed reports whether the step resulted in a bucket transition.
func (t BucketTransition) Changed() bool {
	return t.Swapped
}

// From returns the bucket active prior to the step.
func (t BucketTransition) From() BucketSpec {
	return t.PriorBucket
}

// To returns the bucket active after the step.
func (t BucketTransition) To() BucketSpec {
	return t.NewBucket
}

// DefaultCrossLayerBucketBuilder returns a [BucketPlanBuilder] that constructs
// the cross-layer (post + nextAttnIn) [hopper.Plan] and persistent
// [hopper.Megakernel] for any bucketM in {1, 2, 4, 8} (or larger) using
// [hopper.NewBuilder], [hopper.Optimize], [hopper.CompileFor], and
// [hopper.BuildCrossLayerMegakernel].
func DefaultCrossLayerBucketBuilder(hiddenDim, interDim, qkvDim int, target string) BucketPlanBuilder {
	return func(bucketM int, regime BucketRegime) (*hopper.Plan, hopper.Megakernel, error) {
		if bucketM < 1 {
			return nil, hopper.Megakernel{}, fmt.Errorf("default cross-layer bucket builder: bucketM %d < 1", bucketM)
		}
		if hiddenDim <= 0 || interDim <= 0 || qkvDim <= 0 {
			return nil, hopper.Megakernel{}, fmt.Errorf("default cross-layer bucket builder: invalid dimensions hidden=%d inter=%d qkv=%d", hiddenDim, interDim, qkvDim)
		}
		const eps = 1e-5

		// Layer L post: o_proj -> residual_rmsnorm -> gate_up -> down -> post_residual_rmsnorm -> xn
		bPost := hopper.NewBuilder()
		attn := bPost.Input("attn", hopper.BF16, bucketM, hiddenDim)
		x := bPost.Input("x", hopper.F32, bucketM, hiddenDim)
		wo := bPost.Input("w_o", hopper.BF16, hiddenDim, hiddenDim)
		hid := bPost.Map(hopper.MustLambda("(add $0 $1)"), bPost.MatMul(attn, wo), x)
		ffnNorm := bPost.Cast(hopper.BF16, bPost.RMSNorm(hid, bPost.Input("ffn_norm", hopper.F32, hiddenDim), hiddenDim, eps))
		act := bPost.SwiGLU(bPost.MatMul(ffnNorm, bPost.Input("w_gu", hopper.BF16, 2*interDim, hiddenDim)))
		down := bPost.Map(hopper.MustLambda("(add $0 $1)"), bPost.MatMul(act, bPost.Input("w_down", hopper.BF16, hiddenDim, interDim)), hid)
		xnOut := bPost.Cast(hopper.BF16, bPost.RMSNorm(down, bPost.Input("next_norm", hopper.F32, hiddenDim), hiddenDim, eps))

		postExpr, err := hopper.Optimize(bPost.G, xnOut)
		if err != nil {
			return nil, hopper.Megakernel{}, fmt.Errorf("optimize post M=%d: %w", bucketM, err)
		}
		postName := fmt.Sprintf("l0_post_m%d", bucketM)
		postPlan, err := hopper.CompileFor(postExpr, postName, hopper.BF16, target)
		if err != nil {
			return nil, hopper.Megakernel{}, fmt.Errorf("compile post M=%d: %w", bucketM, err)
		}

		// Layer L+1 nextAttnIn: qkv_proj over xn
		bAttn := hopper.NewBuilder()
		xnIn := bAttn.Input("xn", hopper.BF16, bucketM, hiddenDim)
		wqkv := bAttn.Input("w_qkv", hopper.BF16, qkvDim, hiddenDim)
		qkv := bAttn.MatMul(xnIn, wqkv)

		attnExpr, err := hopper.Optimize(bAttn.G, qkv)
		if err != nil {
			return nil, hopper.Megakernel{}, fmt.Errorf("optimize nextAttnIn M=%d: %w", bucketM, err)
		}
		attnName := fmt.Sprintf("l1_attn_in_m%d", bucketM)
		nextAttnInPlan, err := hopper.CompileFor(attnExpr, attnName, hopper.F32, target)
		if err != nil {
			return nil, hopper.Megakernel{}, fmt.Errorf("compile nextAttnIn M=%d: %w", bucketM, err)
		}

		// Ensure RowsPerBlock * bucketM <= 256 for all SkinnyGEMMs in the megakernel.
		for _, pl := range []*hopper.Plan{postPlan, nextAttnInPlan} {
			for i := range pl.Kernels {
				if g := pl.Kernels[i].GEMM; g != nil {
					for _, r := range []int{32, 16, 8, 4, 2, 1} {
						if g.GLU && r%2 != 0 {
							continue
						}
						if g.N%r == 0 && r*max(bucketM, g.MaxM, 1) <= 256 {
							g.RowsPerBlock = r
							break
						}
					}
				}
			}
		}

		regimeSuffix := "skinny"
		if regime == RegimeTensorCoreTile || regime == RegimeTilePrefill || bucketM >= 4 {
			regimeSuffix = "tc_verify"
		}
		glue := hopper.CrossLayerGlue{
			Name:               fmt.Sprintf("cross_l0_l1_%s_m%d", regimeSuffix, bucketM),
			LayerLPrefix:       "l0_",
			LayerNextPrefix:    "l1_",
			BridgeInput:        "xn",
			RetainBridgeOutput: true,
		}
		return hopper.BuildCrossLayerMegakernel(postPlan, nextAttnInPlan, glue)
	}
}

func sortEntriesByM(entries []*BucketMegakernelEntry) {
	slices.SortFunc(entries, func(a, b *BucketMegakernelEntry) int {
		return cmp.Compare(a.Spec.BucketM, b.Spec.BucketM)
	})
}
