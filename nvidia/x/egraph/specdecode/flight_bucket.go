package specdecode

import (
	"cmp"
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"math"
	"slices"
	"strings"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudahotswap"
	"github.com/tmc/lib/nvidia/x/cudashape"
	"github.com/tmc/lib/nvidia/x/egraph/hopper"
)

// BucketRegime identifies the execution schedule regime for a dynamic M bucket.
type BucketRegime string

const (
	// RegimeSkinnyDecode targets low-acceptance or single-token decode (M = 1..2)
	// using a skinny split-K / vector GEMV schedule with collapsed draft depth.
	RegimeSkinnyDecode BucketRegime = "skinny_decode"

	// RegimeMTPTreeVerify targets medium-to-high acceptance speculative tree
	// verification (M = 4..8) using [DraftTreeTopology] ancestor masks and
	// [hopper.MTPPagedAttention].
	RegimeMTPTreeVerify BucketRegime = "mtp_tree_verify"

	// RegimeTilePrefill targets large effective token batches (M >= 16, where
	// ActiveBatch * QTokensPerSeq >= 16) using tensor-core TileGEMM / WGMMA tiles.
	RegimeTilePrefill BucketRegime = "tile_prefill"
)

// Candidate packages a compiled or compilable kernel source and its 3D launch
// geometry for hot-swapping a CUDA graph node across dynamic M buckets via
// [cudahotswap.Service].
type Candidate struct {
	Name        string
	Source      cudahotswap.Source
	Grid        [3]int
	Block       [3]int
	SharedBytes int
	Version     uint64
}

// BucketCandidate is an alias for [Candidate].
type BucketCandidate = Candidate

// LaunchGeometry converts c's 3D Grid, Block, and SharedBytes into a
// [cudahotswap.LaunchGeometry].
func (c Candidate) LaunchGeometry() cudahotswap.LaunchGeometry {
	gx, gy, gz := max(c.Grid[0], 1), max(c.Grid[1], 1), max(c.Grid[2], 1)
	bx, by, bz := max(c.Block[0], 1), max(c.Block[1], 1), max(c.Block[2], 1)
	smem := max(c.SharedBytes, 0)
	return cudahotswap.LaunchGeometry{
		Grid: cudashape.Grid3D{
			X: uint32(gx),
			Y: uint32(gy),
			Z: uint32(gz),
		},
		Block: driver.Dim3{
			X: uint32(bx),
			Y: uint32(by),
			Z: uint32(bz),
		},
		SharedMem: uint32(smem),
	}
}

// BucketSpec defines the speculative draft tree shape, attention kernel
// configuration, schedule regime, and hot-swap candidate for one dynamic M
// bucket (e.g., M in {1, 2, 4, 8, 16, 64}).
type BucketSpec struct {
	BucketM           int
	Regime            BucketRegime
	DraftDepth        int
	DraftBranchFactor int
	Topology          *DraftTreeTopology
	MTPAttention      *hopper.MTPPagedAttention
	Candidate         Candidate
}

// QTokensPerSeq returns the number of query/draft tokens verified per sequence
// in a single step for b.
func (b BucketSpec) QTokensPerSeq() int {
	if b.MTPAttention != nil && b.MTPAttention.QTokensPerSeq > 0 {
		return b.MTPAttention.QTokensPerSeq
	}
	if b.Topology != nil && b.Topology.NumNodes() > 0 {
		return b.Topology.NumNodes()
	}
	if b.DraftDepth <= 0 {
		return 1
	}
	if b.DraftBranchFactor <= 1 {
		return min(b.DraftDepth+1, 64)
	}
	nodes := 1
	level := 1
	for range b.DraftDepth {
		level *= b.DraftBranchFactor
		nodes += level
		if nodes >= 64 {
			return 64
		}
	}
	if b.BucketM > 0 && nodes > b.BucketM {
		return min(b.BucketM, 64)
	}
	return nodes
}

// LaunchGeometry returns the 3D [cudahotswap.LaunchGeometry] for b.
func (b BucketSpec) LaunchGeometry() cudahotswap.LaunchGeometry {
	return b.Candidate.LaunchGeometry()
}

// BuildRegularDraftTree constructs a topological [DraftTreeTopology] with root
// node 0, tree depth up to depth, branching factor branchFactor, and at most
// maxNodes total nodes (1..64).
func BuildRegularDraftTree(depth, branchFactor, maxNodes int) (DraftTreeTopology, error) {
	if depth < 0 {
		return DraftTreeTopology{}, fmt.Errorf("build regular draft tree: negative depth %d", depth)
	}
	if maxNodes <= 0 {
		maxNodes = 64
	}
	maxNodes = min(maxNodes, 64)
	if depth == 0 || maxNodes == 1 {
		return BuildDraftTreeTopology([]int{-1})
	}
	if branchFactor <= 1 {
		n := min(depth+1, maxNodes)
		parent := make([]int, n)
		parent[0] = -1
		for i := 1; i < n; i++ {
			parent[i] = i - 1
		}
		return BuildDraftTreeTopology(parent)
	}
	parent := []int{-1}
	nodeDepth := []int{0}
	for cur := 0; cur < len(parent) && len(parent) < maxNodes; cur++ {
		if nodeDepth[cur] >= depth {
			continue
		}
		for b := 0; b < branchFactor && len(parent) < maxNodes; b++ {
			parent = append(parent, cur)
			nodeDepth = append(nodeDepth, nodeDepth[cur]+1)
		}
	}
	return BuildDraftTreeTopology(parent)
}

// DefaultBucketSpecs returns the canonical dynamic M bucket table spanning
// [RegimeSkinnyDecode] (M=1, 2), [RegimeMTPTreeVerify] (M=4, 8), and
// [RegimeTilePrefill] (M=16, 64).
func DefaultBucketSpecs() ([]BucketSpec, error) {
	raw := []BucketSpec{
		{
			BucketM:           1,
			Regime:            RegimeSkinnyDecode,
			DraftDepth:        0,
			DraftBranchFactor: 1,
			Candidate: Candidate{
				Name:  "bucket_m1_skinny",
				Grid:  [3]int{1, 8, 1},
				Block: [3]int{128, 1, 1},
			},
		},
		{
			BucketM:           2,
			Regime:            RegimeSkinnyDecode,
			DraftDepth:        1,
			DraftBranchFactor: 1,
			Candidate: Candidate{
				Name:  "bucket_m2_skinny",
				Grid:  [3]int{2, 8, 1},
				Block: [3]int{128, 1, 1},
			},
		},
		{
			BucketM:           4,
			Regime:            RegimeMTPTreeVerify,
			DraftDepth:        2,
			DraftBranchFactor: 2,
			Candidate: Candidate{
				Name:  "bucket_m4_mtp_tree",
				Grid:  [3]int{1, 8, 1},
				Block: [3]int{128, 1, 1},
			},
		},
		{
			BucketM:           8,
			Regime:            RegimeMTPTreeVerify,
			DraftDepth:        3,
			DraftBranchFactor: 2,
			Candidate: Candidate{
				Name:  "bucket_m8_mtp_tree",
				Grid:  [3]int{1, 8, 1},
				Block: [3]int{128, 1, 1},
			},
		},
		{
			BucketM:           16,
			Regime:            RegimeTilePrefill,
			DraftDepth:        3,
			DraftBranchFactor: 2,
			Candidate: Candidate{
				Name:        "bucket_m16_tile_prefill",
				Grid:        [3]int{2, 8, 1},
				Block:       [3]int{128, 2, 1},
				SharedBytes: 32768,
			},
		},
		{
			BucketM:           64,
			Regime:            RegimeTilePrefill,
			DraftDepth:        4,
			DraftBranchFactor: 2,
			Candidate: Candidate{
				Name:        "bucket_m64_tile_prefill",
				Grid:        [3]int{8, 8, 1},
				Block:       [3]int{128, 2, 1},
				SharedBytes: 49152,
			},
		},
	}
	out := make([]BucketSpec, len(raw))
	for i, b := range raw {
		nb, err := normalizeBucketSpec(b, "specdecode_bucket_kernel")
		if err != nil {
			return nil, err
		}
		out[i] = nb
	}
	return out, nil
}

func normalizeBucketSpec(b BucketSpec, defaultEntry string) (BucketSpec, error) {
	if b.BucketM < 1 {
		return b, fmt.Errorf("adaptive bucket spec: BucketM %d < 1", b.BucketM)
	}
	if b.Regime == "" {
		switch {
		case b.BucketM <= 2:
			b.Regime = RegimeSkinnyDecode
		case b.BucketM <= 8:
			b.Regime = RegimeMTPTreeVerify
		default:
			b.Regime = RegimeTilePrefill
		}
	}
	if b.DraftBranchFactor < 1 {
		if b.Regime == RegimeSkinnyDecode {
			b.DraftBranchFactor = 1
		} else {
			b.DraftBranchFactor = 2
		}
	}
	if b.DraftDepth < 0 {
		return b, fmt.Errorf("adaptive bucket spec M=%d: negative DraftDepth %d", b.BucketM, b.DraftDepth)
	}
	if b.DraftDepth == 0 && b.BucketM > 1 && b.Topology == nil {
		switch {
		case b.BucketM == 2:
			b.DraftDepth = 1
		case b.BucketM <= 4:
			b.DraftDepth = 2
		default:
			b.DraftDepth = 3
		}
	}

	maxTreeNodes := min(b.BucketM, 8)
	if b.MTPAttention != nil && b.MTPAttention.QTokensPerSeq > 0 {
		maxTreeNodes = b.MTPAttention.QTokensPerSeq
	} else if b.Regime == RegimeSkinnyDecode {
		maxTreeNodes = min(b.BucketM, 2)
	}
	if b.Topology == nil {
		tree, err := BuildRegularDraftTree(b.DraftDepth, b.DraftBranchFactor, maxTreeNodes)
		if err != nil {
			return b, fmt.Errorf("adaptive bucket spec M=%d: %w", b.BucketM, err)
		}
		b.Topology = &tree
	} else if len(b.Topology.AncestorMask) == 0 && len(b.Topology.ParentIdx) > 0 {
		tree, err := BuildDraftTreeTopology(b.Topology.ParentIdx)
		if err != nil {
			return b, fmt.Errorf("adaptive bucket spec M=%d: %w", b.BucketM, err)
		}
		b.Topology = &tree
	}

	qTokens := b.Topology.NumNodes()
	if qTokens < 1 {
		qTokens = 1
	}
	if b.MTPAttention == nil && (b.Regime == RegimeMTPTreeVerify || b.Regime == RegimeTilePrefill) {
		wgmma := qTokens >= 2
		attn := &hopper.MTPPagedAttention{
			Name:          fmt.Sprintf("mtp_paged_attn_m%d", b.BucketM),
			NumHeads:      32,
			KVHeads:       8,
			HeadDim:       128,
			PageSize:      16,
			MaxPages:      64,
			QTokensPerSeq: qTokens,
			KVFP8:         wgmma,
			Swizzle:       true,
			WGMMA:         wgmma,
		}
		b.MTPAttention = attn
	} else if b.MTPAttention != nil {
		attnCopy := *b.MTPAttention
		if attnCopy.QTokensPerSeq <= 0 {
			attnCopy.QTokensPerSeq = qTokens
		}
		if attnCopy.NumHeads <= 0 {
			attnCopy.NumHeads = 32
		}
		if attnCopy.KVHeads <= 0 {
			attnCopy.KVHeads = 8
		}
		if attnCopy.HeadDim <= 0 {
			attnCopy.HeadDim = 128
		}
		if attnCopy.PageSize <= 0 {
			attnCopy.PageSize = 16
		}
		if attnCopy.MaxPages <= 0 {
			attnCopy.MaxPages = 64
		}
		b.MTPAttention = &attnCopy
	}

	if b.Candidate.Name == "" {
		b.Candidate.Name = fmt.Sprintf("bucket_m%d_%s", b.BucketM, b.Regime)
	}
	if b.Candidate.Grid == ([3]int{}) || b.Candidate.Block == ([3]int{}) {
		if b.MTPAttention != nil {
			batch := max(b.BucketM/max(b.QTokensPerSeq(), 1), 1)
			g, blk := b.MTPAttention.Grid(batch)
			if b.Candidate.Grid == ([3]int{}) {
				b.Candidate.Grid = g
			}
			if b.Candidate.Block == ([3]int{}) {
				b.Candidate.Block = blk
			}
			if b.Candidate.SharedBytes == 0 {
				b.Candidate.SharedBytes = b.MTPAttention.SharedBytes()
			}
		} else {
			if b.Candidate.Grid == ([3]int{}) {
				b.Candidate.Grid = [3]int{max(b.BucketM, 1), 1, 1}
			}
			if b.Candidate.Block == ([3]int{}) {
				b.Candidate.Block = [3]int{128, 1, 1}
			}
		}
	}
	if b.MTPAttention != nil && b.Candidate.SharedBytes == 0 {
		b.Candidate.SharedBytes = b.MTPAttention.SharedBytes()
	}

	entry := b.Candidate.Source.Entry
	if entry == "" {
		entry = defaultEntry
	}
	if entry == "" {
		entry = "specdecode_bucket_kernel"
	}
	if b.Candidate.Source.Code == "" {
		b.Candidate.Source.Code = fmt.Sprintf(
			"// bucket M=%d regime=%s q_tokens=%d\nextern \"C\" __global__ void %s(float* out) { if (out) out[0] = %d.0f; }\n",
			b.BucketM, b.Regime, b.QTokensPerSeq(), entry, b.BucketM,
		)
	}
	b.Candidate.Source.Entry = entry
	if b.Candidate.Source.Filename == "" {
		b.Candidate.Source.Filename = b.Candidate.Name + ".cu"
	}
	return b, nil
}

// AdaptiveBucketConfig configures an [AdaptiveBucketController].
type AdaptiveBucketConfig struct {
	// Buckets lists the available dynamic M buckets. When empty,
	// [DefaultBucketSpecs] is used.
	Buckets []BucketSpec

	// EMAAlpha is the exponential moving average smoothing factor in (0, 1] for
	// live draft acceptance-rate tracking (defaults to 0.25).
	EMAAlpha float64

	// HysteresisSteps is the number of consecutive steps that must recommend the
	// same new target bucket before triggering a hot-swap (defaults to 3).
	HysteresisSteps int

	// LowAcceptanceThreshold is the acceptance-rate EMA threshold below which
	// draft tree depth collapses to M=1..2 [RegimeSkinnyDecode] (defaults to 0.30).
	LowAcceptanceThreshold float64

	// HighAcceptanceThreshold is the acceptance-rate EMA threshold at or above
	// which draft tree depth expands to M=4..8 [RegimeMTPTreeVerify], or M>=16
	// [RegimeTilePrefill] when ActiveBatch * QTokensPerSeq >= 16 (defaults to 0.65).
	HighAcceptanceThreshold float64

	// InitialAcceptanceRate optionally seeds the acceptance-rate EMA before the
	// first [AdaptiveBucketController.RecordStep] call when in (0, 1].
	InitialAcceptanceRate float64

	// InitialBucketM optionally selects the starting bucket by BucketM.
	InitialBucketM int

	// Service is the optional [cudahotswap.Service] used to perform live bucket
	// kernel and 3D launch geometry switches.
	Service *cudahotswap.Service

	// KernelName is the registered kernel name in Service (defaults to
	// "specdecode_bucket_kernel" when Service is non-nil).
	KernelName string

	// Checker is an optional post-publication check installed on Service for Push.
	Checker cudahotswap.Checker
}

// Norm validates c and populates default parameters and normalized buckets.
func (c AdaptiveBucketConfig) Norm() (AdaptiveBucketConfig, error) {
	if c.EMAAlpha == 0 {
		c.EMAAlpha = 0.25
	}
	if c.EMAAlpha <= 0 || c.EMAAlpha > 1 || math.IsNaN(c.EMAAlpha) {
		return c, fmt.Errorf("adaptive bucket config: EMAAlpha %v outside (0, 1]", c.EMAAlpha)
	}
	if c.HysteresisSteps == 0 {
		c.HysteresisSteps = 3
	}
	if c.HysteresisSteps < 1 {
		return c, fmt.Errorf("adaptive bucket config: HysteresisSteps %d < 1", c.HysteresisSteps)
	}
	if c.LowAcceptanceThreshold == 0 {
		c.LowAcceptanceThreshold = 0.30
	}
	if c.HighAcceptanceThreshold == 0 {
		c.HighAcceptanceThreshold = 0.65
	}
	if c.LowAcceptanceThreshold <= 0 || c.LowAcceptanceThreshold >= c.HighAcceptanceThreshold || c.HighAcceptanceThreshold > 1 {
		return c, fmt.Errorf("adaptive bucket config: invalid thresholds low=%v high=%v", c.LowAcceptanceThreshold, c.HighAcceptanceThreshold)
	}
	if c.KernelName == "" && c.Service != nil {
		c.KernelName = "specdecode_bucket_kernel"
	}
	defaultEntry := c.KernelName
	if defaultEntry == "" {
		defaultEntry = "specdecode_bucket_kernel"
	}

	var buckets []BucketSpec
	if len(c.Buckets) == 0 {
		var err error
		buckets, err = DefaultBucketSpecs()
		if err != nil {
			return c, err
		}
		for i := range buckets {
			buckets[i].Candidate.Source.Entry = defaultEntry
		}
	} else {
		buckets = make([]BucketSpec, len(c.Buckets))
		for i, b := range c.Buckets {
			nb, err := normalizeBucketSpec(b, defaultEntry)
			if err != nil {
				return c, err
			}
			buckets[i] = nb
		}
	}
	slices.SortFunc(buckets, func(a, b BucketSpec) int {
		return cmp.Compare(a.BucketM, b.BucketM)
	})
	for i := 1; i < len(buckets); i++ {
		if buckets[i].BucketM == buckets[i-1].BucketM {
			return c, fmt.Errorf("adaptive bucket config: duplicate BucketM %d", buckets[i].BucketM)
		}
	}
	c.Buckets = buckets
	return c, nil
}

// BucketSnapshot records a promoted bucket, its [cudahotswap.State], and its 3D
// [cudahotswap.LaunchGeometry] on the controller's rollback stack.
type BucketSnapshot struct {
	Step          int
	Bucket        BucketSpec
	State         cudahotswap.State
	Geometry      cudahotswap.LaunchGeometry
	AcceptanceEMA float64
}

// BucketTransition reports the outcome of a step or explicit bucket switch.
type BucketTransition struct {
	Swapped     bool
	PriorBucket BucketSpec
	NewBucket   BucketSpec
	State       cudahotswap.State
	Geometry    cudahotswap.LaunchGeometry
	UsedSelect  bool
	PushResult  *cudahotswap.PushResult
}

// AdaptiveBucketController tracks the live speculative draft acceptance-rate
// EMA, recommends dynamic M buckets and speculative draft tree shapes across
// [RegimeSkinnyDecode], [RegimeMTPTreeVerify], and [RegimeTilePrefill], applies
// hysteresis anti-thrashing, and performs atomic hot-swaps and rollbacks
// through a [cudahotswap.Service].
type AdaptiveBucketController struct {
	cfg            AdaptiveBucketConfig
	service        *cudahotswap.Service
	kernelName     string
	acceptanceEMA  float64
	emaInitialized bool
	lastBatchSize  int
	totalSteps     int
	totalDrafted   int
	totalAccepted  int

	currentBucket   BucketSpec
	currentState    cudahotswap.State
	currentGeometry cudahotswap.LaunchGeometry

	pendingBucketM int
	pendingSteps   int
	transitions    int

	bucketVersions map[int]uint64
	history        []BucketSnapshot
	lastErr        error
}

// NewAdaptiveBucketController creates an [AdaptiveBucketController] from cfg.
// When cfg.Service is non-nil, the initial bucket is bound to cfg.KernelName
// and its 3D launch geometry is configured on the service.
func NewAdaptiveBucketController(cfg AdaptiveBucketConfig) (*AdaptiveBucketController, error) {
	normCfg, err := cfg.Norm()
	if err != nil {
		return nil, err
	}
	initial := normCfg.Buckets[0]
	if normCfg.InitialBucketM > 0 {
		for _, b := range normCfg.Buckets {
			if b.BucketM == normCfg.InitialBucketM {
				initial = b
				break
			}
		}
	}

	ctrl := &AdaptiveBucketController{
		cfg:             normCfg,
		service:         normCfg.Service,
		kernelName:      normCfg.KernelName,
		lastBatchSize:   1,
		currentBucket:   initial,
		currentGeometry: initial.LaunchGeometry(),
		bucketVersions:  make(map[int]uint64, len(normCfg.Buckets)),
	}
	if normCfg.InitialAcceptanceRate > 0 && normCfg.InitialAcceptanceRate <= 1 {
		ctrl.acceptanceEMA = normCfg.InitialAcceptanceRate
		ctrl.emaInitialized = true
	}
	for _, b := range normCfg.Buckets {
		if b.Candidate.Version != 0 {
			ctrl.bucketVersions[b.BucketM] = b.Candidate.Version
		}
	}

	if ctrl.service != nil && ctrl.kernelName != "" {
		if _, err := ctrl.BindService(context.Background(), ctrl.service, ctrl.kernelName); err != nil {
			return nil, err
		}
	} else {
		ctrl.history = []BucketSnapshot{
			{
				Step:          0,
				Bucket:        initial,
				Geometry:      initial.LaunchGeometry(),
				AcceptanceEMA: ctrl.acceptanceEMA,
			},
		}
	}
	return ctrl, nil
}

// BindService binds ctrl to kernelName in svc, registering the current bucket's
// kernel if kernelName is not yet registered, applying its 3D launch geometry,
// and seeding the rollback history.
func (c *AdaptiveBucketController) BindService(ctx context.Context, svc *cudahotswap.Service, kernelName string) (cudahotswap.State, error) {
	if c == nil {
		return cudahotswap.State{}, errors.New("bind service: nil controller")
	}
	if svc == nil || kernelName == "" {
		return cudahotswap.State{}, errors.New("bind service: service and kernel name are required")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	c.service = svc
	c.kernelName = kernelName

	cur := c.currentBucket
	src := cur.Candidate.Source
	if src.Entry == "" {
		src.Entry = kernelName
	}
	label := cur.Candidate.Name
	if label == "" {
		label = fmt.Sprintf("bucket_m%d", cur.BucketM)
	}

	state, err := svc.Status(kernelName)
	if err != nil {
		state, err = svc.Register(ctx, kernelName, src, label)
		if err != nil {
			return cudahotswap.State{}, fmt.Errorf("bind service register %s: %w", kernelName, err)
		}
	}
	if c.cfg.Checker != nil {
		if _, err := svc.SetCheck(kernelName, cudahotswap.CheckConfig{Checker: c.cfg.Checker}, label+": check"); err != nil {
			return state, fmt.Errorf("bind service set check %s: %w", kernelName, err)
		}
	}
	geom := cur.LaunchGeometry()
	state, err = svc.SetGeometry(kernelName, &geom, label+": geometry")
	if err != nil {
		return state, fmt.Errorf("bind service set geometry %s: %w", kernelName, err)
	}
	c.bucketVersions[cur.BucketM] = state.Version
	c.currentState = state
	c.currentGeometry = geom
	c.history = []BucketSnapshot{
		{
			Step:          c.totalSteps,
			Bucket:        cur,
			State:         state,
			Geometry:      geom,
			AcceptanceEMA: c.acceptanceEMA,
		},
	}
	return state, nil
}

// AcceptanceRate returns the current exponentially smoothed draft token
// acceptance rate in [0, 1].
func (c *AdaptiveBucketController) AcceptanceRate() float64 {
	if c == nil {
		return 0
	}
	return c.acceptanceEMA
}

// CurrentBucket returns the currently active [BucketSpec].
func (c *AdaptiveBucketController) CurrentBucket() BucketSpec {
	if c == nil {
		return BucketSpec{}
	}
	return c.currentBucket
}

// CurrentGeometry returns the currently active 3D [cudahotswap.LaunchGeometry].
func (c *AdaptiveBucketController) CurrentGeometry() cudahotswap.LaunchGeometry {
	if c == nil {
		return cudahotswap.LaunchGeometry{}
	}
	return c.currentGeometry
}

// CurrentState returns the last observed [cudahotswap.State] from the service.
func (c *AdaptiveBucketController) CurrentState() cudahotswap.State {
	if c == nil {
		return cudahotswap.State{}
	}
	return c.currentState
}

// PendingBucket returns the candidate BucketM currently accumulating hysteresis
// steps and the number of consecutive steps observed so far.
func (c *AdaptiveBucketController) PendingBucket() (bucketM, consecutiveSteps int) {
	if c == nil {
		return 0, 0
	}
	return c.pendingBucketM, c.pendingSteps
}

// Transitions returns the total number of completed bucket hot-swaps.
func (c *AdaptiveBucketController) Transitions() int {
	if c == nil {
		return 0
	}
	return c.transitions
}

// History returns a copy of the promoted bucket snapshot stack from oldest to
// current.
func (c *AdaptiveBucketController) History() []BucketSnapshot {
	if c == nil {
		return nil
	}
	return slices.Clone(c.history)
}

// LastError returns the most recent error encountered during an automatic
// hot-swap in [AdaptiveBucketController.RecordStep], if any.
func (c *AdaptiveBucketController) LastError() error {
	if c == nil {
		return nil
	}
	return c.lastErr
}

// RecordStep updates the live acceptance-rate EMA from a verification step with
// batchSize active sequences, draftedTokens proposed draft tokens, and
// acceptedTokens accepted draft tokens. When the recommended bucket differs
// from the current bucket for HysteresisSteps consecutive steps, RecordStep
// triggers an atomic bucket transition (and hot-swaps the bound
// [cudahotswap.Service] when configured). It returns the updated EMA.
func (c *AdaptiveBucketController) RecordStep(batchSize, draftedTokens, acceptedTokens int) float64 {
	if c == nil {
		return 0
	}
	_, err := c.Step(context.Background(), batchSize, draftedTokens, acceptedTokens)
	c.lastErr = err
	return c.acceptanceEMA
}

// Step updates the live acceptance-rate EMA, evaluates [AdaptiveBucketController.RecommendBucket],
// applies the hysteresis anti-thrashing gate, and switches buckets when
// HysteresisSteps consecutive steps agree on a new target bucket.
func (c *AdaptiveBucketController) Step(ctx context.Context, batchSize, draftedTokens, acceptedTokens int) (BucketTransition, error) {
	if c == nil {
		return BucketTransition{}, errors.New("adaptive bucket step: nil controller")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if batchSize > 0 {
		c.lastBatchSize = batchSize
	} else if c.lastBatchSize <= 0 {
		c.lastBatchSize = 1
	}

	if draftedTokens > 0 {
		acc := min(max(acceptedTokens, 0), draftedTokens)
		rate := float64(acc) / float64(draftedTokens)
		if !c.emaInitialized {
			c.acceptanceEMA = rate
			c.emaInitialized = true
		} else {
			alpha := c.cfg.EMAAlpha
			c.acceptanceEMA = alpha*rate + (1.0-alpha)*c.acceptanceEMA
		}
		c.totalDrafted += draftedTokens
		c.totalAccepted += acc
	}
	c.totalSteps++

	rec := c.RecommendBucket(c.lastBatchSize)
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

// RecommendBucket computes the recommended [BucketSpec] (including its
// [BucketRegime], [DraftTreeTopology], [hopper.MTPPagedAttention], and 3D
// launch geometry) for batchSize active sequences and the current
// acceptance-rate EMA.
func (c *AdaptiveBucketController) RecommendBucket(batchSize int) BucketSpec {
	if c == nil || len(c.cfg.Buckets) == 0 {
		return BucketSpec{}
	}
	activeBatch := max(batchSize, 1)
	ema := c.acceptanceEMA
	lowThresh := c.cfg.LowAcceptanceThreshold
	highThresh := c.cfg.HighAcceptanceThreshold

	// Determine the target query/draft tokens per sequence from the live
	// acceptance rate EMA.
	var targetQTokens int
	var targetRegime BucketRegime
	switch {
	case ema < lowThresh:
		// Low acceptance (< LowAcceptanceThreshold): collapse speculation to
		// skinny decode (M = 1..2).
		targetRegime = RegimeSkinnyDecode
		if activeBatch == 1 && ema >= lowThresh*0.5 {
			targetQTokens = 2
		} else {
			targetQTokens = 1
		}
	case ema < highThresh:
		// Medium acceptance ([LowAcceptanceThreshold, HighAcceptanceThreshold)):
		// verify a moderate draft tree (4 nodes per sequence).
		targetRegime = RegimeMTPTreeVerify
		targetQTokens = 4
	default:
		// High acceptance (>= HighAcceptanceThreshold): expand draft tree depth
		// to 8 nodes per sequence (or largest MTP tree bucket).
		targetRegime = RegimeMTPTreeVerify
		targetQTokens = 8
		for _, b := range c.cfg.Buckets {
			if b.Regime == RegimeMTPTreeVerify && b.QTokensPerSeq() > targetQTokens {
				targetQTokens = b.QTokensPerSeq()
			}
		}
	}

	effectiveM := activeBatch * targetQTokens
	if activeBatch >= 16 || effectiveM >= 16 {
		targetRegime = RegimeTilePrefill
		if effectiveM < 16 {
			effectiveM = activeBatch
		}
	}

	// Select the best matching bucket within targetRegime first.
	var regimeBuckets []BucketSpec
	for _, b := range c.cfg.Buckets {
		if b.Regime == targetRegime {
			regimeBuckets = append(regimeBuckets, b)
		}
	}
	if len(regimeBuckets) > 0 {
		for _, b := range regimeBuckets {
			if b.BucketM >= effectiveM {
				return b
			}
		}
		return regimeBuckets[len(regimeBuckets)-1]
	}

	// Fallback if the caller configured Buckets without explicit regime tags
	// covering targetRegime: pick the smallest bucket with BucketM >= effectiveM.
	for _, b := range c.cfg.Buckets {
		if b.BucketM >= effectiveM {
			return b
		}
	}
	return c.cfg.Buckets[len(c.cfg.Buckets)-1]
}

// SwitchBucket atomically switches the controller to target. When a
// [cudahotswap.Service] is bound, SwitchBucket calls SetGeometry followed by
// Select (when target's compiled version is already retained in the service)
// or Push (when target's kernel has not yet been compiled and loaded), and
// records the transition on the rollback stack.
func (c *AdaptiveBucketController) SwitchBucket(ctx context.Context, target BucketSpec) (BucketTransition, error) {
	if c == nil {
		return BucketTransition{}, errors.New("switch bucket: nil controller")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	defaultEntry := c.kernelName
	if defaultEntry == "" {
		defaultEntry = "specdecode_bucket_kernel"
	}
	normTarget, err := normalizeBucketSpec(target, defaultEntry)
	if err != nil {
		return BucketTransition{}, err
	}

	priorBucket := c.currentBucket
	priorGeometry := c.currentGeometry
	newGeometry := normTarget.LaunchGeometry()

	trans := BucketTransition{
		Swapped:     true,
		PriorBucket: priorBucket,
		NewBucket:   normTarget,
		Geometry:    newGeometry,
	}

	if c.service != nil && c.kernelName != "" {
		label := normTarget.Candidate.Name
		if label == "" {
			label = fmt.Sprintf("bucket_m%d", normTarget.BucketM)
		}
		if _, err := c.service.SetGeometry(c.kernelName, &newGeometry, label+": geometry"); err != nil {
			return BucketTransition{}, fmt.Errorf("switch bucket M=%d set geometry: %w", normTarget.BucketM, err)
		}

		version := c.lookupRetainedVersion(normTarget)
		if version != 0 {
			state, selErr := c.service.Select(ctx, c.kernelName, version, label)
			if selErr != nil {
				_, _ = c.service.SetGeometry(c.kernelName, &priorGeometry, priorBucket.Candidate.Name+": rollback_geometry")
				return BucketTransition{}, fmt.Errorf("switch bucket M=%d select v%d: %w", normTarget.BucketM, version, selErr)
			}
			trans.State = state
			trans.UsedSelect = true
			c.currentState = state
		} else {
			src := normTarget.Candidate.Source
			if src.Entry == "" {
				src.Entry = c.kernelName
			}
			pushRes, pushErr := c.pushCandidate(ctx, src, label)
			trans.PushResult = pushRes
			if pushErr != nil {
				_, _ = c.service.SetGeometry(c.kernelName, &priorGeometry, priorBucket.Candidate.Name+": rollback_geometry")
				if pushRes != nil && pushRes.Selected.Version != 0 {
					c.currentState = pushRes.Selected
				}
				return trans, fmt.Errorf("switch bucket M=%d push: %w", normTarget.BucketM, pushErr)
			}
			if pushRes == nil || !pushRes.Published || pushRes.RolledBack {
				_, _ = c.service.SetGeometry(c.kernelName, &priorGeometry, priorBucket.Candidate.Name+": rollback_geometry")
				return trans, fmt.Errorf("switch bucket M=%d: candidate was not published", normTarget.BucketM)
			}
			trans.State = pushRes.Selected
			c.currentState = pushRes.Selected
			c.bucketVersions[normTarget.BucketM] = pushRes.Selected.Version
		}
	}

	c.currentBucket = normTarget
	c.currentGeometry = newGeometry
	c.transitions++
	c.history = append(c.history, BucketSnapshot{
		Step:          c.totalSteps,
		Bucket:        normTarget,
		State:         c.currentState,
		Geometry:      newGeometry,
		AcceptanceEMA: c.acceptanceEMA,
	})
	return trans, nil
}

func (c *AdaptiveBucketController) lookupRetainedVersion(target BucketSpec) uint64 {
	if v := c.bucketVersions[target.BucketM]; v != 0 {
		if c.isVersionRetained(v) {
			return v
		}
	}
	if target.Candidate.Version != 0 && c.isVersionRetained(target.Candidate.Version) {
		return target.Candidate.Version
	}
	if c.service != nil && c.kernelName != "" && target.Candidate.Source.Code != "" {
		wantDigest := fmt.Sprintf("%x", sha256.Sum256([]byte(target.Candidate.Source.Code)))
		if versions, err := c.service.Versions(c.kernelName); err == nil {
			for _, v := range versions {
				if v.SourceDigest == wantDigest {
					c.bucketVersions[target.BucketM] = v.Number
					return v.Number
				}
			}
		}
	}
	return 0
}

func (c *AdaptiveBucketController) isVersionRetained(version uint64) bool {
	if c.service == nil || c.kernelName == "" || version == 0 {
		return false
	}
	versions, err := c.service.Versions(c.kernelName)
	if err != nil {
		return false
	}
	for _, v := range versions {
		if v.Number == version {
			return true
		}
	}
	return false
}

func (c *AdaptiveBucketController) pushCandidate(ctx context.Context, src cudahotswap.Source, label string) (*cudahotswap.PushResult, error) {
	res, err := c.service.Push(ctx, c.kernelName, src, label)
	if err != nil && strings.Contains(err.Error(), "no replay or application check configured") {
		checker := c.cfg.Checker
		if checker == nil {
			checker = func(context.Context, cudahotswap.State) error { return nil }
		}
		if _, sErr := c.service.SetCheck(c.kernelName, cudahotswap.CheckConfig{Checker: checker}, label+": auto_check"); sErr == nil {
			return c.service.Push(ctx, c.kernelName, src, label)
		}
	}
	return res, err
}

// RestorePrevious rolls the bound [cudahotswap.Service] and controller state
// back to the immediate prior bucket snapshot in History, restoring both the
// prior kernel version via [cudahotswap.Service.Select] and its exact 3D
// [cudahotswap.LaunchGeometry] via [cudahotswap.Service.SetGeometry].
func (c *AdaptiveBucketController) RestorePrevious(ctx context.Context) (cudahotswap.State, error) {
	_, state, err := c.RollbackBucket(ctx)
	return state, err
}

// RollbackBucket rolls the controller and bound [cudahotswap.Service] back to
// the immediate prior bucket on the promotion stack (via RestorePrevious
// semantics), restoring the exact prior [BucketSpec], kernel version, and 3D
// launch geometry.
func (c *AdaptiveBucketController) RollbackBucket(ctx context.Context) (BucketSpec, cudahotswap.State, error) {
	if c == nil {
		return BucketSpec{}, cudahotswap.State{}, errors.New("rollback bucket: nil controller")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if len(c.history) < 2 {
		return BucketSpec{}, cudahotswap.State{}, errors.New("rollback bucket: no prior bucket in history")
	}
	prior := c.history[len(c.history)-2]
	var state cudahotswap.State
	if c.service != nil && c.kernelName != "" {
		if prior.State.Version == 0 {
			return BucketSpec{}, cudahotswap.State{}, errors.New("rollback bucket: prior snapshot has no service version")
		}
		label := prior.Bucket.Candidate.Name
		if label == "" {
			label = fmt.Sprintf("bucket_m%d", prior.Bucket.BucketM)
		}
		geom := prior.Geometry
		if _, err := c.service.SetGeometry(c.kernelName, &geom, label+": restore_previous_geometry"); err != nil {
			return BucketSpec{}, cudahotswap.State{}, fmt.Errorf("rollback bucket set geometry: %w", err)
		}
		var err error
		state, err = c.service.Select(ctx, c.kernelName, prior.State.Version, label+": restore_previous")
		if err != nil {
			return BucketSpec{}, cudahotswap.State{}, fmt.Errorf("rollback bucket select v%d: %w", prior.State.Version, err)
		}
	}
	c.history = c.history[:len(c.history)-1]
	if state.Version != 0 {
		c.history[len(c.history)-1].State = state
	} else {
		state = prior.State
	}
	c.currentBucket = prior.Bucket
	c.currentGeometry = prior.Geometry
	c.currentState = state
	c.pendingBucketM = 0
	c.pendingSteps = 0
	return prior.Bucket, state, nil
}

// VerifyAndRollbackIfFailed executes post-switch verification fn (such as a
// latency guard or greedy token verification check) and, if fn returns a
// non-nil error, automatically calls [AdaptiveBucketController.RollbackBucket]
// to restore the immediate prior bucket and 3D launch geometry.
func (c *AdaptiveBucketController) VerifyAndRollbackIfFailed(ctx context.Context, fn func() error) (rolledBack bool, err error) {
	if c == nil {
		return false, errors.New("verify bucket: nil controller")
	}
	if fn == nil {
		return false, nil
	}
	verifyErr := fn()
	if verifyErr == nil {
		return false, nil
	}
	if _, _, rbErr := c.RollbackBucket(ctx); rbErr != nil {
		return false, errors.Join(verifyErr, rbErr)
	}
	return true, verifyErr
}
