package cudagraph

import (
	"fmt"
	"sort"
	"sync"

	"github.com/tmc/lib/nvidia/x/cudaerrors"
)

const (
	defaultSmallStep = 4
	defaultMedStep   = 8
	defaultLargeStep = 16

	smallRegionMax = 16
	medRegionMax   = 64
)

// BucketPolicy configures piecewise CUDA Graph batch-size bucket generation.
//
// When ExplicitBuckets is empty, [BuildBatchBuckets] generates buckets up to
// MaxBsz using power-of-two warmup sizes (1, 2) followed by three piecewise
// step regimes:
//   - SmallStep (default 4) up to 16: [4, 8, 12, 16]
//   - MedStep (default 8) from 16 up to 64: [24, 32, 40, 48, 56, 64]
//   - LargeStep (default 16) above 64: [80, 96, ...]
//
// MaxBsz is always included when MaxBsz > 0.
type BucketPolicy struct {
	MaxBsz          int
	SmallStep       int
	MedStep         int
	LargeStep       int
	ExplicitBuckets []int
}

// BuildBatchBuckets generates a sorted, deduplicated slice of positive batch-size
// buckets according to policy.
func BuildBatchBuckets(policy BucketPolicy) ([]int, error) {
	if policy.MaxBsz < 0 {
		return nil, fmt.Errorf("cudagraph: negative MaxBsz %d: %w", policy.MaxBsz, cudaerrors.ErrUnsupported)
	}
	if policy.SmallStep < 0 || policy.MedStep < 0 || policy.LargeStep < 0 {
		return nil, fmt.Errorf("cudagraph: negative step in bucket policy (small=%d, med=%d, large=%d): %w",
			policy.SmallStep, policy.MedStep, policy.LargeStep, cudaerrors.ErrUnsupported)
	}

	if len(policy.ExplicitBuckets) > 0 {
		seen := make(map[int]struct{}, len(policy.ExplicitBuckets)+1)
		out := make([]int, 0, len(policy.ExplicitBuckets)+1)
		for _, b := range policy.ExplicitBuckets {
			if b <= 0 {
				return nil, fmt.Errorf("cudagraph: non-positive explicit bucket %d: %w", b, cudaerrors.ErrUnsupported)
			}
			if policy.MaxBsz > 0 && b > policy.MaxBsz {
				continue
			}
			if _, ok := seen[b]; !ok {
				seen[b] = struct{}{}
				out = append(out, b)
			}
		}
		if policy.MaxBsz > 0 {
			if _, ok := seen[policy.MaxBsz]; !ok {
				out = append(out, policy.MaxBsz)
			}
		}
		if len(out) == 0 {
			return nil, fmt.Errorf("cudagraph: no valid buckets in policy: %w", cudaerrors.ErrUnsupported)
		}
		sort.Ints(out)
		return out, nil
	}

	if policy.MaxBsz <= 0 {
		return nil, fmt.Errorf("cudagraph: MaxBsz must be > 0 when ExplicitBuckets is empty (got %d): %w",
			policy.MaxBsz, cudaerrors.ErrUnsupported)
	}

	smallStep := policy.SmallStep
	if smallStep == 0 {
		smallStep = defaultSmallStep
	}
	medStep := policy.MedStep
	if medStep == 0 {
		medStep = defaultMedStep
	}
	largeStep := policy.LargeStep
	if largeStep == 0 {
		largeStep = defaultLargeStep
	}

	seen := make(map[int]struct{})
	var raw []int
	add := func(v int) {
		if v <= 0 || v > policy.MaxBsz {
			return
		}
		if _, ok := seen[v]; !ok {
			seen[v] = struct{}{}
			raw = append(raw, v)
		}
	}

	// Power-of-two micro-batches before smallStep.
	for p := 1; p < smallStep && p <= policy.MaxBsz; p *= 2 {
		add(p)
	}

	// Small step regime up to smallRegionMax (16).
	smallLimit := min(policy.MaxBsz, smallRegionMax)
	for v := smallStep; v <= smallLimit; v += smallStep {
		add(v)
	}

	// Medium step regime above smallRegionMax up to medRegionMax (64).
	if policy.MaxBsz > smallRegionMax {
		medLimit := min(policy.MaxBsz, medRegionMax)
		for v := smallRegionMax + medStep; v <= medLimit; v += medStep {
			add(v)
		}
	}

	// Large step regime above medRegionMax (64).
	if policy.MaxBsz > medRegionMax {
		for v := medRegionMax + largeStep; v <= policy.MaxBsz; v += largeStep {
			add(v)
		}
	}

	// Always include MaxBsz.
	add(policy.MaxBsz)
	sort.Ints(raw)
	return raw, nil
}

// BucketPadding describes the selected CUDA Graph bucket and required dummy
// token/sequence padding for an actual batch size.
type BucketPadding struct {
	ActualBsz int
	BucketBsz int
	PadCount  int
}

// ResolveBucket finds the smallest bucket in buckets that is >= actualBsz and
// returns the corresponding [BucketPadding]. It returns an error wrapping
// [cudaerrors.ErrUnsupported] if buckets is empty, actualBsz <= 0, or actualBsz
// exceeds the largest bucket.
func ResolveBucket(buckets []int, actualBsz int) (BucketPadding, error) {
	if len(buckets) == 0 {
		return BucketPadding{}, fmt.Errorf("cudagraph: empty bucket list: %w", cudaerrors.ErrUnsupported)
	}
	if actualBsz <= 0 {
		return BucketPadding{}, fmt.Errorf("cudagraph: invalid actualBsz %d: must be > 0: %w", actualBsz, cudaerrors.ErrUnsupported)
	}

	best := -1
	maxBucket := -1
	for _, b := range buckets {
		if b <= 0 {
			return BucketPadding{}, fmt.Errorf("cudagraph: invalid bucket size %d: %w", b, cudaerrors.ErrUnsupported)
		}
		if b > maxBucket {
			maxBucket = b
		}
		if b >= actualBsz && (best == -1 || b < best) {
			best = b
		}
	}
	if best == -1 {
		return BucketPadding{}, fmt.Errorf("cudagraph: actualBsz %d exceeds max bucket %d: %w",
			actualBsz, maxBucket, cudaerrors.ErrUnsupported)
	}
	return BucketPadding{
		ActualBsz: actualBsz,
		BucketBsz: best,
		PadCount:  best - actualBsz,
	}, nil
}

// BucketRunner maps piecewise batch-size buckets to captured graph handles of
// type T and provides descending warmup ordering and padded bucket lookup.
type BucketRunner[T any] struct {
	mu      sync.RWMutex
	buckets []int
	handles map[int]T
}

// NewBucketRunner constructs a [BucketRunner] over a non-empty list of positive
// bucket sizes. The input slice is copied, sorted, and deduplicated.
func NewBucketRunner[T any](buckets []int) (*BucketRunner[T], error) {
	if len(buckets) == 0 {
		return nil, fmt.Errorf("cudagraph: empty bucket list: %w", cudaerrors.ErrUnsupported)
	}
	seen := make(map[int]struct{}, len(buckets))
	sorted := make([]int, 0, len(buckets))
	for _, b := range buckets {
		if b <= 0 {
			return nil, fmt.Errorf("cudagraph: invalid bucket size %d: %w", b, cudaerrors.ErrUnsupported)
		}
		if _, ok := seen[b]; !ok {
			seen[b] = struct{}{}
			sorted = append(sorted, b)
		}
	}
	sort.Ints(sorted)
	return &BucketRunner[T]{
		buckets: sorted,
		handles: make(map[int]T, len(sorted)),
	}, nil
}

// NewBucketRunnerFromPolicy constructs a [BucketRunner] using [BuildBatchBuckets].
func NewBucketRunnerFromPolicy[T any](policy BucketPolicy) (*BucketRunner[T], error) {
	buckets, err := BuildBatchBuckets(policy)
	if err != nil {
		return nil, err
	}
	return NewBucketRunner[T](buckets)
}

// Register binds handle to bucketBsz. It returns an error wrapping
// [cudaerrors.ErrUnsupported] if bucketBsz is not a configured bucket in r.
func (r *BucketRunner[T]) Register(bucketBsz int, handle T) error {
	if r == nil {
		return fmt.Errorf("cudagraph: nil BucketRunner: %w", cudaerrors.ErrUnsupported)
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	idx := sort.SearchInts(r.buckets, bucketBsz)
	if idx >= len(r.buckets) || r.buckets[idx] != bucketBsz {
		return fmt.Errorf("cudagraph: bucket %d is not in configured buckets: %w", bucketBsz, cudaerrors.ErrUnsupported)
	}
	r.handles[bucketBsz] = handle
	return nil
}

// Lookup resolves actualBsz to the smallest configured bucket >= actualBsz and
// returns its registered handle and [BucketPadding].
func (r *BucketRunner[T]) Lookup(actualBsz int) (handle T, pad BucketPadding, err error) {
	if r == nil {
		return handle, BucketPadding{}, fmt.Errorf("cudagraph: nil BucketRunner: %w", cudaerrors.ErrUnsupported)
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	pad, err = ResolveBucket(r.buckets, actualBsz)
	if err != nil {
		return handle, BucketPadding{}, err
	}
	h, ok := r.handles[pad.BucketBsz]
	if !ok {
		return handle, BucketPadding{}, fmt.Errorf("cudagraph: no handle registered for bucket %d: %w",
			pad.BucketBsz, cudaerrors.ErrUnsupported)
	}
	return h, pad, nil
}

// Select is an alias for [BucketRunner.Lookup] that resolves totalQ to the
// smallest configured bucket >= totalQ.
func (r *BucketRunner[T]) Select(totalQ int) (handle T, pad BucketPadding, err error) {
	return r.Lookup(totalQ)
}

// Buckets returns the configured bucket sizes in ascending order.
func (r *BucketRunner[T]) Buckets() []int {
	if r == nil {
		return nil
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]int, len(r.buckets))
	copy(out, r.buckets)
	return out
}

// WarmupOrder returns the configured bucket sizes in descending order (from
// MaxBsz down to the smallest bucket), matching DeepInfer's warmup and graph
// capture order so the shared allocator high-water mark is established first.
func (r *BucketRunner[T]) WarmupOrder() []int {
	if r == nil {
		return nil
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	n := len(r.buckets)
	out := make([]int, n)
	for i, b := range r.buckets {
		out[n-1-i] = b
	}
	return out
}
