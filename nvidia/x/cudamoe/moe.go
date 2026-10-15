// SPDX-License-Identifier: Apache-2.0

package cudamoe

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"unsafe"
)

// MoEConfig defines dimensional configuration for DeepSeek-V3 and Qwen MoE architectures.
type MoEConfig struct {
	NumRoutedExperts int // e.g. 256 for DeepSeek-V3, 64 for Qwen
	TopK             int // e.g. 8 routed experts
	NumSharedExperts int // e.g. 1 shared expert
	HiddenDim        int
	IntermediateDim  int
}

// RoutingDecision holds selected expert indices and normalized softmax weights for a single token.
type RoutingDecision struct {
	SelectedExperts []int
	Weights         []float32
	HasShared       bool
}

// Gather16 gathers expert-major rows using checked scalar indices. Vectorized
// uint4 lowering is admitted only when the byte alignment contract holds;
// tails use scalar loads.
func Gather16(experts [][]float32, indices []int, width int) ([]float32, error) {
	if width <= 0 || width%16 != 0 {
		return nil, fmt.Errorf("cudamoe: width must be a positive multiple of 16")
	}
	out := make([]float32, len(indices)*width)
	for r, idx := range indices {
		if idx < 0 || idx >= len(experts) || len(experts[idx]) < width {
			return nil, fmt.Errorf("cudamoe: expert index out of range")
		}
		copy(out[r*width:], experts[idx][:width])
	}
	return out, nil
}

// Gather16Checked adds the routing-table bound required by generated kernels.
func Gather16Checked(experts [][]float32, indices []int, width, expertCount int) ([]float32, error) {
	if expertCount < 0 || expertCount > len(experts) {
		return nil, fmt.Errorf("cudamoe: invalid expert count")
	}
	for _, i := range indices {
		if i < 0 || i >= expertCount {
			return nil, fmt.Errorf("cudamoe: route expert out of range")
		}
	}
	return Gather16(experts, indices, width)
}

// GatherSource emits a bounds-checked expert-major gather. uint4 loads are
// used only when both input and output groups are aligned. Each output has
// one writer; unaligned groups and tails use scalar loads. Each group of tile
// threads handles tile adjacent columns; the final group of each row may have
// inactive threads. Use GatherGeometry for the padded launch dimensions.
// Input and output must not overlap. Invalid expert routes leave output untouched.
func GatherSource(tile int) (string, error) {
	if tile != 16 && tile != 32 {
		return "", fmt.Errorf("cudamoe: admitted tiles are 16 or 32")
	}
	return fmt.Sprintf(`#include <cuda_runtime.h>
extern "C" __global__ void moe_gather(const float* experts, float* out,
    const int* route, int tokens, int width, int expert_count) {
  if (tokens <= 0 || width <= 0 || expert_count <= 0) return;
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t tile_width = %d;
  size_t tiles_per_row = ((size_t)width + tile_width - 1) / tile_width;
  size_t tile = i / tile_width;
  size_t t = tile / tiles_per_row;
  if (t >= (size_t)tokens) return;
  size_t column = (tile %% tiles_per_row) * tile_width + i %% tile_width;
  if (column >= (size_t)width) return;
  int c = (int)column;
  int e = route[t];
  if (e < 0 || e >= expert_count) return;
  int group = c - c %% 4;
  const float* src = experts + (size_t)e * width + group;
  float* dst = out + t * width + group;
  if (width - group >= 4 &&
      (((unsigned long long)src | (unsigned long long)dst) & 15) == 0) {
    if (c == group) *(uint4*)dst = *(const uint4*)src;
  } else {
    out[t * width + c] = experts[(size_t)e * width + c];
  }
}`, tile), nil
}

// PrefillW256Source emits a 256-row FP16-input, FP32-output Tensor Core GEMM.
// Inputs are row-major A[256,K] and B[K,N]; output is row-major C[256,N].
// K and N must be positive multiples of 16, pointers 32-byte aligned, and
// buffers nonoverlapping. Launch grid (N/16,4,1) with block (128,1,1).
// The caller must group independent tokens for one expert before launch.
// It includes cuda_fp16.h and mma.h and uses NVRTC's predefined size_t.
func PrefillW256Source() (string, error) {
	return prefillW256Source, nil
}

// GroupedTopKRouter computes bias-balanced grouped top-K routing.
//
// Selection ranks experts by gate logit plus a per-expert load-balancing bias,
// but the returned weights are the softmax of the unbiased logits, so the bias
// steers which experts run without distorting how much each contributes.
type GroupedTopKRouter struct {
	cfg    MoEConfig
	biases []float32
}

// NewGroupedTopKRouter creates a new MoE router. It requires at least one
// routed expert, 0 < TopK <= NumRoutedExperts, and NumSharedExperts >= 0,
// and it rejects sizes whose routing slices would exceed 1<<31 - 1 bytes.
func NewGroupedTopKRouter(cfg MoEConfig) (*GroupedTopKRouter, error) {
	if cfg.NumRoutedExperts <= 0 {
		return nil, fmt.Errorf("cudamoe: %d routed experts, want at least 1", cfg.NumRoutedExperts)
	}
	if cfg.TopK <= 0 || cfg.TopK > cfg.NumRoutedExperts {
		return nil, fmt.Errorf("cudamoe: top-k %d outside [1, %d]", cfg.TopK, cfg.NumRoutedExperts)
	}
	if cfg.NumSharedExperts < 0 {
		return nil, fmt.Errorf("cudamoe: %d shared experts, want at least 0", cfg.NumSharedExperts)
	}
	if err := checkRouterAlloc(cfg.NumRoutedExperts, cfg.TopK); err != nil {
		return nil, err
	}
	return &GroupedTopKRouter{
		cfg:    cfg,
		biases: make([]float32, cfg.NumRoutedExperts),
	}, nil
}

// SetBiases replaces the per-expert routing biases. Balancing schemes nudge
// these between steps to spread load across experts. len(biases) must equal
// NumRoutedExperts.
func (r *GroupedTopKRouter) SetBiases(biases []float32) error {
	if len(biases) != r.cfg.NumRoutedExperts {
		return fmt.Errorf("cudamoe: %d biases for %d routed experts", len(biases), r.cfg.NumRoutedExperts)
	}
	copy(r.biases, biases)
	return nil
}

// Biases returns a copy of the per-expert routing biases.
func (r *GroupedTopKRouter) Biases() []float32 {
	return append([]float32(nil), r.biases...)
}

type expertScore struct {
	index int
	score float32 // gate logit plus bias, used for ranking
	logit float32 // unbiased gate logit, used for the weight
}

// maxRouterAlloc bounds the bytes in any one slice the router allocates.
// It is the smallest runtime allocation limit of any Go port (maxAlloc in
// runtime/malloc.go is 1<<31 - 1 on mips and mipsle, 1<<32 - 1 on other
// 32-bit ports and wasm, 1<<40 on ios/arm64, and 1<<47 or more elsewhere),
// so a slice within it never makes make panic with "len out of range".
const maxRouterAlloc = math.MaxInt32

// checkRouterAlloc reports an error if any slice a router with n routed
// experts and top-k k allocates, in NewGroupedTopKRouter or Route, would
// exceed maxRouterAlloc bytes.
func checkRouterAlloc(n, k int) error {
	allocs := []struct {
		what  string
		count int
		size  uintptr
	}{
		{"routing biases", n, unsafe.Sizeof(float32(0))},
		{"expert scores", n, unsafe.Sizeof(expertScore{})},
		{"selected experts", k, unsafe.Sizeof(int(0))},
		{"routing weights", k, unsafe.Sizeof(float32(0))},
	}
	for _, a := range allocs {
		if uint64(a.count) > maxRouterAlloc/uint64(a.size) {
			return fmt.Errorf("cudamoe: %d %s of %d bytes exceed the %d-byte allocation limit", a.count, a.what, a.size, maxRouterAlloc)
		}
	}
	return nil
}

// Route computes top-K expert selection with softmax weights. Experts with
// equal scores are ordered by index, so routing is reproducible across runs.
// len(gateLogits) must equal NumRoutedExperts, and every logit and bias must
// be finite. The router must come from NewGroupedTopKRouter; Route on the zero
// value returns an error.
func (r *GroupedTopKRouter) Route(gateLogits []float32) (RoutingDecision, error) {
	n := r.cfg.NumRoutedExperts
	if n <= 0 || r.cfg.TopK <= 0 {
		return RoutingDecision{}, fmt.Errorf("cudamoe: router has %d routed experts and top-k %d; use NewGroupedTopKRouter", n, r.cfg.TopK)
	}
	if len(gateLogits) != n {
		return RoutingDecision{}, fmt.Errorf("cudamoe: %d gate logits for %d routed experts", len(gateLogits), n)
	}

	// NewGroupedTopKRouter bounded n and k with checkRouterAlloc.
	scores := make([]expertScore, n)
	for i := 0; i < n; i++ {
		if !finite(gateLogits[i]) {
			return RoutingDecision{}, fmt.Errorf("cudamoe: gate logit %d is %v", i, gateLogits[i])
		}
		if !finite(r.biases[i]) {
			return RoutingDecision{}, fmt.Errorf("cudamoe: routing bias %d is %v", i, r.biases[i])
		}
		scores[i] = expertScore{
			index: i,
			score: gateLogits[i] + r.biases[i],
			logit: gateLogits[i],
		}
	}

	// Sort descending by score, breaking ties on the expert index. sort.Slice
	// is not stable, so without the tiebreak equal scores route arbitrarily.
	sort.Slice(scores, func(i, j int) bool {
		if scores[i].score != scores[j].score {
			return scores[i].score > scores[j].score
		}
		return scores[i].index < scores[j].index
	})

	k := r.cfg.TopK
	selected := make([]int, k)
	maxLogit := scores[0].logit
	for i := 0; i < k; i++ {
		selected[i] = scores[i].index
		if scores[i].logit > maxLogit {
			maxLogit = scores[i].logit
		}
	}

	// Softmax over the selected experts' unbiased logits. The largest term
	// is exp(0) = 1, so sumExp >= 1 for any finite logits.
	var sumExp float32
	weights := make([]float32, k)
	for i := 0; i < k; i++ {
		e := float32(math.Exp(float64(scores[i].logit - maxLogit)))
		weights[i] = e
		sumExp += e
	}
	for i := 0; i < k; i++ {
		weights[i] /= sumExp
	}

	return RoutingDecision{
		SelectedExperts: selected,
		Weights:         weights,
		HasShared:       r.cfg.NumSharedExperts > 0,
	}, nil
}

func finite(f float32) bool {
	return !math.IsNaN(float64(f)) && !math.IsInf(float64(f), 0)
}

// ExpertCacheManager manages hot in-memory experts and spills cold experts to NVMe SSD.
type ExpertCacheManager struct {
	dir         string
	maxHot      int
	hotExperts  map[int][]byte // expertID -> weight bytes
	coldFiles   map[int]string // expertID -> file path
	accessCount map[int]uint64
	mu          sync.RWMutex
}

// NewExpertCacheManager creates an expert offloader and memory manager.
func NewExpertCacheManager(dir string, maxHot int) (*ExpertCacheManager, error) {
	if dir == "" {
		dir = filepath.Join(os.TempDir(), "cudamoe-expert-cache")
	}
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return nil, fmt.Errorf("cudamoe: create expert cache dir: %w", err)
	}
	return &ExpertCacheManager{
		dir:         dir,
		maxHot:      maxHot,
		hotExperts:  make(map[int][]byte),
		coldFiles:   make(map[int]string),
		accessCount: make(map[int]uint64),
	}, nil
}

// RegisterExpert registers an expert's weight buffer into the cache.
func (m *ExpertCacheManager) RegisterExpert(expertID int, weights []byte) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.hotExperts[expertID] = weights
	m.accessCount[expertID]++
}

// GetExpert retrieves an expert weight buffer, paging in from SSD if cold.
func (m *ExpertCacheManager) GetExpert(ctx context.Context, expertID int) ([]byte, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.accessCount[expertID]++

	if data, ok := m.hotExperts[expertID]; ok {
		return data, nil
	}

	path, ok := m.coldFiles[expertID]
	if !ok {
		return nil, fmt.Errorf("cudamoe: expert %d not found in memory or disk", expertID)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("cudamoe: paging in expert %d from SSD: %w", expertID, err)
	}

	m.hotExperts[expertID] = data
	return data, nil
}

// EvictColdExpert spills an in-memory expert to secondary SSD storage.
func (m *ExpertCacheManager) EvictColdExpert(expertID int) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	data, ok := m.hotExperts[expertID]
	if !ok {
		return nil
	}

	filePath := filepath.Join(m.dir, fmt.Sprintf("cuda_expert_%d.bin", expertID))
	if err := os.WriteFile(filePath, data, 0o600); err != nil {
		return fmt.Errorf("cudamoe: write expert file: %w", err)
	}

	delete(m.hotExperts, expertID)
	m.coldFiles[expertID] = filePath
	return nil
}

// Close releases resources.
func (m *ExpertCacheManager) Close() error {
	return nil
}
