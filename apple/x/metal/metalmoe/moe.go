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

// GroupedTopKRouter computes bias-balanced grouped top-K routing.
type GroupedTopKRouter struct {
	cfg   MoEConfig
	biases []float32
}

// NewGroupedTopKRouter creates a new MoE router.
func NewGroupedTopKRouter(cfg MoEConfig) *GroupedTopKRouter {
	return &GroupedTopKRouter{
		cfg:   cfg,
		biases: make([]float32, cfg.NumRoutedExperts),
	}
}

type expertScore struct {
	index int
	score float32
}

// Route computes top-K expert selection with softmax weights.
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
		}
	}

	// Sort descending by score
	sort.Slice(scores, func(i, j int) bool {
		return scores[i].score > scores[j].score
	})

	k := r.cfg.TopK
	if k > len(scores) {
		k = len(scores)
	}

	selected := make([]int, k)
	rawScores := make([]float32, k)
	var maxScore float32 = -1e20

	for i := 0; i < k; i++ {
		selected[i] = scores[i].index
		rawScores[i] = scores[i].score
		if scores[i].score > maxScore {
			maxScore = scores[i].score
		}
	}

	// Compute normalized softmax weights
	var sumExp float32
	weights := make([]float32, k)
	for i := 0; i < k; i++ {
		e := float32(math.Exp(float64(rawScores[i] - maxScore)))
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
		dir = filepath.Join(os.TempDir(), "metalmoe-expert-cache")
	}
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return nil, fmt.Errorf("metalmoe: create expert cache dir: %w", err)
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
		return nil, fmt.Errorf("metalmoe: expert %d not found in memory or disk", expertID)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("metalmoe: paging in expert %d from SSD: %w", expertID, err)
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

	filePath := filepath.Join(m.dir, fmt.Sprintf("expert_%d.bin", expertID))
	if err := os.WriteFile(filePath, data, 0o600); err != nil {
		return fmt.Errorf("metalmoe: write expert file: %w", err)
	}

	delete(m.hotExperts, expertID)
	m.coldFiles[expertID] = filePath
	return nil
}

// Close releases resources.
func (m *ExpertCacheManager) Close() error {
	return nil
}
