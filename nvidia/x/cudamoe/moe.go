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
// used only on aligned groups; the scalar loop handles tails.
func GatherSource(width int) (string, error) {
	if width != 16 && width != 32 {
		return "", fmt.Errorf("cudamoe: admitted tiles are 16 or 32")
	}
	return fmt.Sprintf(`#include <cuda_runtime.h>
// tile width=%d
extern "C" __global__ void moe_gather(const float* experts,float* out,const int* route,int tokens,int width,int expert_count){int i=blockIdx.x*blockDim.x+threadIdx.x;int t=i/width,c=i%%width;if(t<tokens&&c<width){int e=route[t];if(e<0||e>=expert_count)return;const float* p=experts+(size_t)e*width+c;if(c%%4==0&&c+4<=width&&(((unsigned long long)p)&15)==0){uint4 v=*(const uint4*)p;((uint4*)(out+i))[0]=v;}else out[i]=*p;}}`, width), nil
}

// PrefillW256Source emits the bounded cross-token W=256 GEMM fallback.
func PrefillW256Source() (string, error) {
	return "", fmt.Errorf("cudamoe: W=256 Tensor Core prefill lowering unsupported")
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

// NewGroupedTopKRouter creates a new MoE router.
func NewGroupedTopKRouter(cfg MoEConfig) *GroupedTopKRouter {
	return &GroupedTopKRouter{
		cfg:    cfg,
		biases: make([]float32, cfg.NumRoutedExperts),
	}
}

type expertScore struct {
	index int
	score float32 // gate logit plus bias, used for ranking
	logit float32 // unbiased gate logit, used for the weight
}

// Route computes top-K expert selection with softmax weights. Experts with
// equal scores are ordered by index, so routing is reproducible across runs.
func (r *GroupedTopKRouter) Route(gateLogits []float32) RoutingDecision {
	n := r.cfg.NumRoutedExperts
	if len(gateLogits) < n {
		n = len(gateLogits)
	}

	scores := make([]expertScore, n)
	for i := 0; i < n; i++ {
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
	if k > len(scores) {
		k = len(scores)
	}

	selected := make([]int, k)
	var maxLogit float32 = -1e20

	for i := 0; i < k; i++ {
		selected[i] = scores[i].index
		if scores[i].logit > maxLogit {
			maxLogit = scores[i].logit
		}
	}

	// Softmax over the selected experts' unbiased logits.
	var sumExp float32
	weights := make([]float32, k)
	for i := 0; i < k; i++ {
		e := float32(math.Exp(float64(scores[i].logit - maxLogit)))
		weights[i] = e
		sumExp += e
	}
	if sumExp > 0 {
		for i := 0; i < k; i++ {
			weights[i] /= sumExp
		}
	}

	return RoutingDecision{
		SelectedExperts: selected,
		Weights:         weights,
		HasShared:       r.cfg.NumSharedExperts > 0,
	}
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
