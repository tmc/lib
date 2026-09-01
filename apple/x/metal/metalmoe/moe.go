// SPDX-License-Identifier: Apache-2.0

package metalmoe

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

// GroupedTopKRouter computes bias-balanced top-K routing.
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

// SetBiases replaces the per-expert routing biases. Balancing schemes nudge
// these between steps to spread load across experts. len(biases) must equal
// NumRoutedExperts.
func (r *GroupedTopKRouter) SetBiases(biases []float32) error {
	if len(biases) != r.cfg.NumRoutedExperts {
		return fmt.Errorf("metalmoe: %d biases for %d routed experts", len(biases), r.cfg.NumRoutedExperts)
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
	if k < 0 {
		k = 0
	}

	selected := make([]int, k)
	weights := make([]float32, k)
	if k == 0 {
		return RoutingDecision{
			SelectedExperts: selected,
			Weights:         weights,
			HasShared:       r.cfg.NumSharedExperts > 0,
		}
	}

	maxLogit := scores[0].logit
	for i := 0; i < k; i++ {
		selected[i] = scores[i].index
		if scores[i].logit > maxLogit {
			maxLogit = scores[i].logit
		}
	}

	// Softmax over the selected experts' unbiased logits.
	var sumExp float32
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

// ExpertCacheManager manages hot in-memory experts and spills cold experts to
// NVMe SSD, holding at most maxHot experts resident. When admitting an expert
// would exceed that, the least recently used resident expert is spilled first.
type ExpertCacheManager struct {
	dir        string
	maxHot     int
	hotExperts map[int][]byte // expertID -> weight bytes
	coldFiles  map[int]string // expertID -> file path
	lastUsed   map[int]uint64 // expertID -> access clock at last use
	clock      uint64
	mu         sync.Mutex
}

// NewExpertCacheManager creates an expert offloader and memory manager holding
// at most maxHot experts in memory.
func NewExpertCacheManager(dir string, maxHot int) (*ExpertCacheManager, error) {
	if maxHot <= 0 {
		return nil, fmt.Errorf("metalmoe: maxHot must be positive, got %d", maxHot)
	}
	if dir == "" {
		dir = filepath.Join(os.TempDir(), "metalmoe-expert-cache")
	}
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return nil, fmt.Errorf("metalmoe: create expert cache dir: %w", err)
	}
	return &ExpertCacheManager{
		dir:        dir,
		maxHot:     maxHot,
		hotExperts: make(map[int][]byte),
		coldFiles:  make(map[int]string),
		lastUsed:   make(map[int]uint64),
	}, nil
}

// RegisterExpert registers an expert's weight buffer into the cache, spilling
// the least recently used expert if the cache is already full.
func (m *ExpertCacheManager) RegisterExpert(expertID int, weights []byte) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.hotExperts[expertID] = weights
	m.touch(expertID)
	return m.trim()
}

// GetExpert retrieves an expert weight buffer, paging in from SSD if cold.
func (m *ExpertCacheManager) GetExpert(ctx context.Context, expertID int) ([]byte, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if data, ok := m.hotExperts[expertID]; ok {
		m.touch(expertID)
		return data, nil
	}

	path, ok := m.coldFiles[expertID]
	if !ok {
		return nil, fmt.Errorf("metalmoe: expert %d not found in memory or disk", expertID)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("metalmoe: paging in expert %d from SSD: %w", expertID, err)
	}

	m.hotExperts[expertID] = data
	m.touch(expertID)
	if err := m.trim(); err != nil {
		return nil, err
	}
	return data, nil
}

// HotExperts returns the number of experts currently resident in memory.
func (m *ExpertCacheManager) HotExperts() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.hotExperts)
}

// touch records an access. It requires m.mu to be held.
func (m *ExpertCacheManager) touch(expertID int) {
	m.clock++
	m.lastUsed[expertID] = m.clock
}

// trim spills experts until the cache is within maxHot. It requires m.mu to be
// held. The most recently touched expert is never chosen, so admitting an
// expert never immediately spills it again.
func (m *ExpertCacheManager) trim() error {
	for len(m.hotExperts) > m.maxHot {
		victim, oldest := -1, uint64(math.MaxUint64)
		for id := range m.hotExperts {
			if use := m.lastUsed[id]; use < oldest {
				victim, oldest = id, use
			}
		}
		if victim < 0 {
			return nil
		}
		if err := m.evict(victim); err != nil {
			return err
		}
	}
	return nil
}

// EvictColdExpert spills an in-memory expert to secondary SSD storage.
func (m *ExpertCacheManager) EvictColdExpert(expertID int) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.evict(expertID)
}

// evict requires m.mu to be held.
func (m *ExpertCacheManager) evict(expertID int) error {
	data, ok := m.hotExperts[expertID]
	if !ok {
		return nil
	}

	filePath := filepath.Join(m.dir, fmt.Sprintf("expert_%d.bin", expertID))
	if err := os.WriteFile(filePath, data, 0o600); err != nil {
		return fmt.Errorf("metalmoe: write expert file: %w", err)
	}

	delete(m.hotExperts, expertID)
	m.coldFiles[expertID] = filePath
	return nil
}

// Close removes the expert files the cache spilled and forgets them.
func (m *ExpertCacheManager) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	var firstErr error
	for id, path := range m.coldFiles {
		if err := os.Remove(path); err != nil && !os.IsNotExist(err) && firstErr == nil {
			firstErr = fmt.Errorf("metalmoe: remove expert %d file: %w", id, err)
		}
		delete(m.coldFiles, id)
	}
	return firstErr
}
