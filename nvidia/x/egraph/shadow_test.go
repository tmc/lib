package egraph

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudagraph"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudatest"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

type mockShadowExecutor struct {
	mu        sync.Mutex
	latencies map[string]time.Duration
	calls     []string
	err       error
}

func (m *mockShadowExecutor) ExecuteShadow(ctx context.Context, s *cuda.Stream, cand TuningCandidate) (time.Duration, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.calls = append(m.calls, cand.Key())
	if m.err != nil {
		return 0, m.err
	}
	if dur, ok := m.latencies[cand.Key()]; ok {
		return dur, nil
	}
	return 50 * time.Microsecond, nil
}

type mockGraphUpdater struct {
	mu           sync.Mutex
	gridUpdates  []cudalaunch.Dim3
	paramUpdates []driver.KernelNodeParams
	argUpdates   [][]any
	nodeUpdates  []cudagraph.KernelNodeUpdate
}

func (m *mockGraphUpdater) SetKernelNodeGrid(node *cudagraph.Node, grid cudalaunch.Dim3) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.gridUpdates = append(m.gridUpdates, grid)
	return nil
}

func (m *mockGraphUpdater) SetKernelNodeParams(node *cudagraph.Node, params *driver.KernelNodeParams) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if params != nil {
		m.paramUpdates = append(m.paramUpdates, *params)
	}
	return nil
}

func (m *mockGraphUpdater) UpdateKernelNode(node *cudagraph.Node, grid cudalaunch.Dim3, args ...any) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.gridUpdates = append(m.gridUpdates, grid)
	m.argUpdates = append(m.argUpdates, args)
	return nil
}

func (m *mockGraphUpdater) ApplyKernelNodeUpdate(node *cudagraph.Node, update cudagraph.KernelNodeUpdate) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.nodeUpdates = append(m.nodeUpdates, update)
	return nil
}

func TestShadowStreamExecutionAndLatency(t *testing.T) {
	tests := []struct {
		name          string
		candidates    []TuningCandidate
		latencies     map[string]time.Duration
		repeatCount   int
		wantEvaluated int
		wantScores    map[string]time.Duration
	}{
		{
			name: "single candidate evaluation",
			candidates: []TuningCandidate{
				{
					Expr:       &Expr{Op: "fused_rmsnorm_gemv"},
					BlockDim:   128,
					GridDim:    4,
					KernelName: "Kernel_RMSNorm_Block128",
				},
			},
			latencies: map[string]time.Duration{
				"fused_rmsnorm_gemv::Kernel_RMSNorm_Block128::b128::g4": 42 * time.Microsecond,
			},
			repeatCount:   1,
			wantEvaluated: 1,
			wantScores: map[string]time.Duration{
				"fused_rmsnorm_gemv::Kernel_RMSNorm_Block128::b128::g4": 42 * time.Microsecond,
			},
		},
		{
			name: "multiple block geometries (block 128 vs block 256)",
			candidates: []TuningCandidate{
				{
					Expr:       &Expr{Op: "fused_swiglu_gemv"},
					BlockDim:   128,
					GridDim:    8,
					KernelName: "Kernel_SwiGLU_Block128",
				},
				{
					Expr:       &Expr{Op: "fused_swiglu_gemv"},
					BlockDim:   256,
					GridDim:    4,
					KernelName: "Kernel_SwiGLU_Block256",
				},
			},
			latencies: map[string]time.Duration{
				"fused_swiglu_gemv::Kernel_SwiGLU_Block128::b128::g8": 60 * time.Microsecond,
				"fused_swiglu_gemv::Kernel_SwiGLU_Block256::b256::g4": 38 * time.Microsecond,
			},
			repeatCount:   1,
			wantEvaluated: 2,
			wantScores: map[string]time.Duration{
				"fused_swiglu_gemv::Kernel_SwiGLU_Block128::b128::g8": 60 * time.Microsecond,
				"fused_swiglu_gemv::Kernel_SwiGLU_Block256::b256::g4": 38 * time.Microsecond,
			},
		},
		{
			name: "sub-byte dequantization fusion candidate",
			candidates: []TuningCandidate{
				{
					Expr:       &Expr{Op: "iq2_xxs_matvec"},
					BlockDim:   128,
					GridDim:    16,
					KernelName: "Kernel_IQ2XXS_Block128",
				},
			},
			latencies: map[string]time.Duration{
				"iq2_xxs_matvec::Kernel_IQ2XXS_Block128::b128::g16": 25 * time.Microsecond,
			},
			repeatCount:   1,
			wantEvaluated: 1,
			wantScores: map[string]time.Duration{
				"iq2_xxs_matvec::Kernel_IQ2XXS_Block128::b128::g16": 25 * time.Microsecond,
			},
		},
		{
			name: "tile sizing candidate",
			candidates: []TuningCandidate{
				{
					Expr:       &Expr{Op: "tile_swiglu"},
					BlockDim:   256,
					GridDim:    1,
					KernelName: "Kernel_TileSwiGLU_Block256",
				},
			},
			latencies: map[string]time.Duration{
				"tile_swiglu::Kernel_TileSwiGLU_Block256::b256::g1": 31 * time.Microsecond,
			},
			repeatCount:   1,
			wantEvaluated: 1,
			wantScores: map[string]time.Duration{
				"tile_swiglu::Kernel_TileSwiGLU_Block256::b256::g1": 31 * time.Microsecond,
			},
		},
		{
			name: "repeated evaluations track average latency",
			candidates: []TuningCandidate{
				{
					Expr:       &Expr{Op: "add"},
					BlockDim:   128,
					GridDim:    8,
					KernelName: "Kernel_Add",
				},
			},
			latencies: map[string]time.Duration{
				"add::Kernel_Add::b128::g8": 10 * time.Microsecond,
			},
			repeatCount:   4,
			wantEvaluated: 1,
			wantScores: map[string]time.Duration{
				"add::Kernel_Add::b128::g8": 10 * time.Microsecond,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockExec := &mockShadowExecutor{
				latencies: tt.latencies,
			}
			tuner, err := NewShadowTuner(nil,
				WithShadowExecutor(mockExec),
			)
			if err != nil {
				t.Fatalf("NewShadowTuner: %v", err)
			}
			defer tuner.Close()

			ctx := context.Background()
			repeats := tt.repeatCount
			if repeats <= 0 {
				repeats = 1
			}

			for i := 0; i < repeats; i++ {
				for _, cand := range tt.candidates {
					dur, err := tuner.Evaluate(ctx, cand)
					if err != nil {
						t.Fatalf("Evaluate(%s): %v", cand.Key(), err)
					}
					if dur <= 0 {
						t.Fatalf("Evaluate duration = %v, want > 0", dur)
					}
				}
			}

			pgo := tuner.PGO()
			if pgo == nil {
				t.Fatal("tuner.PGO() returned nil")
			}

			for key, wantScore := range tt.wantScores {
				gotScore, ok := pgo.Score(key)
				if !ok {
					t.Fatalf("pgo.Score(%q) not found", key)
				}
				if gotScore != wantScore {
					t.Errorf("pgo.Score(%q) = %v, want %v", key, gotScore, wantScore)
				}
				count := pgo.Count(key)
				if count != repeats {
					t.Errorf("pgo.Count(%q) = %d, want %d", key, count, repeats)
				}
			}
		})
	}
}

func TestMemoryThrottleGating(t *testing.T) {
	tests := []struct {
		name         string
		threshold    float64
		saturation   float64
		nilThrottle  bool
		nilSampler   bool
		wantErr      error
		wantExecuted bool
	}{
		{
			name:         "high DRAM saturation blocks execution",
			threshold:    0.80,
			saturation:   0.95,
			wantErr:      ErrThrottled,
			wantExecuted: false,
		},
		{
			name:         "exact threshold saturation blocks execution",
			threshold:    0.80,
			saturation:   0.80,
			wantErr:      ErrThrottled,
			wantExecuted: false,
		},
		{
			name:         "low DRAM saturation allows execution",
			threshold:    0.80,
			saturation:   0.40,
			wantErr:      nil,
			wantExecuted: true,
		},
		{
			name:         "zero saturation allows execution",
			threshold:    0.80,
			saturation:   0.0,
			wantErr:      nil,
			wantExecuted: true,
		},
		{
			name:         "nil throttle allows execution",
			nilThrottle:  true,
			wantErr:      nil,
			wantExecuted: true,
		},
		{
			name:         "nil sampler allows execution",
			threshold:    0.80,
			nilSampler:   true,
			wantErr:      nil,
			wantExecuted: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var throttle ThrottleGate
			if !tt.nilThrottle {
				if tt.nilSampler {
					throttle = &MemoryThrottle{Threshold: tt.threshold, Sampler: nil}
				} else {
					sat := tt.saturation
					throttle = NewMemoryThrottle(tt.threshold, func() float64 {
						return sat
					})
				}
			}

			mockExec := &mockShadowExecutor{}
			tuner, err := NewShadowTuner(nil,
				WithThrottle(throttle),
				WithShadowExecutor(mockExec),
			)
			if err != nil {
				t.Fatalf("NewShadowTuner: %v", err)
			}
			defer tuner.Close()

			cand := TuningCandidate{
				Expr:       &Expr{Op: "fused_rmsnorm_gemv"},
				BlockDim:   128,
				GridDim:    2,
				KernelName: "Kernel_Test",
			}

			ctx := context.Background()
			_, err = tuner.Evaluate(ctx, cand)

			if tt.wantErr != nil {
				if !errors.Is(err, tt.wantErr) {
					t.Fatalf("Evaluate() error = %v, want %v", err, tt.wantErr)
				}
			} else if err != nil {
				t.Fatalf("Evaluate() unexpected error: %v", err)
			}

			mockExec.mu.Lock()
			callCount := len(mockExec.calls)
			mockExec.mu.Unlock()

			if tt.wantExecuted && callCount != 1 {
				t.Errorf("callCount = %d, want 1", callCount)
			}
			if !tt.wantExecuted && callCount != 0 {
				t.Errorf("callCount = %d, want 0", callCount)
			}
		})
	}

	t.Run("dynamic transition from saturated to idle", func(t *testing.T) {
		currentSat := 0.95
		throttle := NewMemoryThrottle(0.80, func() float64 {
			return currentSat
		})

		mockExec := &mockShadowExecutor{}
		tuner, err := NewShadowTuner(nil,
			WithThrottle(throttle),
			WithShadowExecutor(mockExec),
		)
		if err != nil {
			t.Fatal(err)
		}
		defer tuner.Close()

		cand := TuningCandidate{
			Expr:       &Expr{Op: "fused_swiglu_gemv"},
			BlockDim:   256,
			KernelName: "Kernel_SwiGLU",
		}

		ctx := context.Background()
		// First attempt: bus saturated (0.95) -> should be throttled
		_, err = tuner.Evaluate(ctx, cand)
		if !errors.Is(err, ErrThrottled) {
			t.Fatalf("expected ErrThrottled at 0.95 saturation, got %v", err)
		}

		// Transition to idle (0.30) -> should succeed
		currentSat = 0.30
		dur, err := tuner.Evaluate(ctx, cand)
		if err != nil {
			t.Fatalf("expected success at 0.30 saturation, got %v", err)
		}
		if dur <= 0 {
			t.Fatalf("duration = %v, want > 0", dur)
		}

		if count := tuner.PGO().Count(cand.Key()); count != 1 {
			t.Errorf("PGO observation count = %d, want 1", count)
		}
	})

	t.Run("EvaluateCandidates throttles early", func(t *testing.T) {
		throttle := NewMemoryThrottle(0.80, func() float64 {
			return 0.90
		})
		tuner, err := NewShadowTuner(nil,
			WithThrottle(throttle),
			WithShadowExecutor(&mockShadowExecutor{}),
		)
		if err != nil {
			t.Fatal(err)
		}
		defer tuner.Close()

		cands := []TuningCandidate{
			{Expr: &Expr{Op: "add"}, BlockDim: 128, KernelName: "K1"},
			{Expr: &Expr{Op: "add"}, BlockDim: 256, KernelName: "K2"},
		}

		results, err := tuner.EvaluateCandidates(context.Background(), cands)
		if !errors.Is(err, ErrThrottled) {
			t.Fatalf("EvaluateCandidates err = %v, want ErrThrottled", err)
		}
		if len(results) != 0 {
			t.Errorf("len(results) = %d, want 0", len(results))
		}
	})
}

func TestPGOTrackerPlanPromotion(t *testing.T) {
	tests := []struct {
		name          string
		currentDur    time.Duration
		currentCount  int
		candDur       time.Duration
		candCount     int
		policy        PromotionPolicy
		wantPromotion bool
	}{
		{
			name:          "faster candidate promoted",
			currentDur:    100 * time.Microsecond,
			currentCount:  5,
			candDur:       70 * time.Microsecond,
			candCount:     3,
			policy:        PromotionPolicy{MinObservations: 1},
			wantPromotion: true,
		},
		{
			name:          "slower candidate rejected",
			currentDur:    100 * time.Microsecond,
			currentCount:  5,
			candDur:       120 * time.Microsecond,
			candCount:     3,
			policy:        PromotionPolicy{MinObservations: 1},
			wantPromotion: false,
		},
		{
			name:          "equal latency candidate rejected",
			currentDur:    100 * time.Microsecond,
			currentCount:  5,
			candDur:       100 * time.Microsecond,
			candCount:     3,
			policy:        PromotionPolicy{MinObservations: 1},
			wantPromotion: false,
		},
		{
			name:          "insufficient observations rejected",
			currentDur:    100 * time.Microsecond,
			currentCount:  5,
			candDur:       50 * time.Microsecond,
			candCount:     2,
			policy:        PromotionPolicy{MinObservations: 3},
			wantPromotion: false,
		},
		{
			name:          "sufficient observations promoted",
			currentDur:    100 * time.Microsecond,
			currentCount:  5,
			candDur:       50 * time.Microsecond,
			candCount:     3,
			policy:        PromotionPolicy{MinObservations: 3},
			wantPromotion: true,
		},
		{
			name:          "improvement below absolute threshold rejected",
			currentDur:    100 * time.Microsecond,
			currentCount:  5,
			candDur:       98 * time.Microsecond,
			candCount:     3,
			policy:        PromotionPolicy{MinObservations: 1, MinImprovement: 5 * time.Microsecond},
			wantPromotion: false,
		},
		{
			name:          "improvement meeting absolute threshold promoted",
			currentDur:    100 * time.Microsecond,
			currentCount:  5,
			candDur:       94 * time.Microsecond,
			candCount:     3,
			policy:        PromotionPolicy{MinObservations: 1, MinImprovement: 5 * time.Microsecond},
			wantPromotion: true,
		},
		{
			name:          "improvement below fractional threshold rejected",
			currentDur:    100 * time.Microsecond,
			currentCount:  5,
			candDur:       97 * time.Microsecond, // 3% improvement
			candCount:     3,
			policy:        PromotionPolicy{MinObservations: 1, ImprovementFrac: 0.05}, // requires 5%
			wantPromotion: false,
		},
		{
			name:          "improvement meeting fractional threshold promoted",
			currentDur:    100 * time.Microsecond,
			currentCount:  5,
			candDur:       90 * time.Microsecond, // 10% improvement
			candCount:     3,
			policy:        PromotionPolicy{MinObservations: 1, ImprovementFrac: 0.05},
			wantPromotion: true,
		},
		{
			name:          "unprofiled baseline candidate promoted with sufficient observations",
			currentDur:    0, // unprofiled
			currentCount:  0,
			candDur:       80 * time.Microsecond,
			candCount:     2,
			policy:        PromotionPolicy{MinObservations: 2},
			wantPromotion: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tracker := &PGOTracker{}
			current := TuningCandidate{
				Expr:       &Expr{Op: "fused_rmsnorm_gemv"},
				BlockDim:   128,
				GridDim:    4,
				KernelName: "Current",
			}
			candidate := TuningCandidate{
				Expr:       &Expr{Op: "fused_rmsnorm_gemv"},
				BlockDim:   256,
				GridDim:    2,
				KernelName: "Candidate",
			}

			for i := 0; i < tt.currentCount; i++ {
				tracker.Record(current.Key(), tt.currentDur)
			}
			for i := 0; i < tt.candCount; i++ {
				tracker.Record(candidate.Key(), tt.candDur)
			}

			tuner, err := NewShadowTuner(nil, WithPGOTracker(tracker))
			if err != nil {
				t.Fatalf("NewShadowTuner: %v", err)
			}
			defer tuner.Close()

			got := tuner.ShouldPromote(current, candidate, tt.policy)
			if got != tt.wantPromotion {
				t.Errorf("ShouldPromote() = %v, want %v", got, tt.wantPromotion)
			}
		})
	}
}

func TestInPlaceGraphNodeUpdate(t *testing.T) {
	node := &cudagraph.Node{}

	t.Run("HotSwapNode grid update", func(t *testing.T) {
		updater := &mockGraphUpdater{}
		cand := TuningCandidate{
			BlockDim: 256,
			GridDim:  8,
		}

		if err := HotSwapNode(updater, node, cand); err != nil {
			t.Fatalf("HotSwapNode: %v", err)
		}
		if len(updater.gridUpdates) != 1 {
			t.Fatalf("len(gridUpdates) = %d, want 1", len(updater.gridUpdates))
		}
		if updater.gridUpdates[0].X != 8 {
			t.Errorf("grid.X = %d, want 8", updater.gridUpdates[0].X)
		}
	})

	t.Run("HotSwapNode with args", func(t *testing.T) {
		updater := &mockGraphUpdater{}
		cand := TuningCandidate{
			BlockDim: 128,
			GridDim:  4,
		}
		arg1 := int32(1024)
		arg2 := float32(1e-5)

		if err := HotSwapNode(updater, node, cand, arg1, arg2); err != nil {
			t.Fatalf("HotSwapNode: %v", err)
		}
		if len(updater.argUpdates) != 1 {
			t.Fatalf("len(argUpdates) = %d, want 1", len(updater.argUpdates))
		}
		if len(updater.argUpdates[0]) != 2 {
			t.Fatalf("len(args) = %d, want 2", len(updater.argUpdates[0]))
		}
	})

	t.Run("HotSwapKernelNodeUpdates", func(t *testing.T) {
		updater := &mockGraphUpdater{}
		u1 := cudagraph.NewGridDimUpdate(0, cudalaunch.Dims(16))
		u2 := cudagraph.NewEnabledUpdate(0, true)

		if err := HotSwapKernelNodeUpdates(updater, node, u1, u2); err != nil {
			t.Fatalf("HotSwapKernelNodeUpdates: %v", err)
		}
		if len(updater.nodeUpdates) != 2 {
			t.Fatalf("len(nodeUpdates) = %d, want 2", len(updater.nodeUpdates))
		}
		if updater.nodeUpdates[0].Field != cudagraph.KernelNodeFieldGridDim {
			t.Errorf("update[0].Field = %v, want GridDim", updater.nodeUpdates[0].Field)
		}
		if updater.nodeUpdates[1].Field != cudagraph.KernelNodeFieldEnabled {
			t.Errorf("update[1].Field = %v, want Enabled", updater.nodeUpdates[1].Field)
		}
	})

	t.Run("PromoteAndApply promoted", func(t *testing.T) {
		updater := &mockGraphUpdater{}
		tracker := &PGOTracker{}
		current := TuningCandidate{BlockDim: 128, GridDim: 8, KernelName: "K_Slow"}
		candidate := TuningCandidate{BlockDim: 256, GridDim: 4, KernelName: "K_Fast"}

		tracker.Record(current.Key(), 100*time.Microsecond)
		tracker.Record(candidate.Key(), 60*time.Microsecond)

		tuner, err := NewShadowTuner(nil, WithPGOTracker(tracker))
		if err != nil {
			t.Fatal(err)
		}
		defer tuner.Close()

		promoted, err := tuner.PromoteAndApply(updater, node, current, candidate, PromotionPolicy{MinObservations: 1})
		if err != nil {
			t.Fatalf("PromoteAndApply error: %v", err)
		}
		if !promoted {
			t.Fatal("PromoteAndApply returned false, want true")
		}
		if len(updater.gridUpdates) != 1 || updater.gridUpdates[0].X != 4 {
			t.Errorf("gridUpdates = %v, want [{X:4 Y:1 Z:1}]", updater.gridUpdates)
		}
	})

	t.Run("PromoteAndApply rejected when candidate is slower", func(t *testing.T) {
		updater := &mockGraphUpdater{}
		tracker := &PGOTracker{}
		current := TuningCandidate{BlockDim: 128, GridDim: 8, KernelName: "K_Fast"}
		candidate := TuningCandidate{BlockDim: 256, GridDim: 4, KernelName: "K_Slow"}

		tracker.Record(current.Key(), 50*time.Microsecond)
		tracker.Record(candidate.Key(), 90*time.Microsecond)

		tuner, err := NewShadowTuner(nil, WithPGOTracker(tracker))
		if err != nil {
			t.Fatal(err)
		}
		defer tuner.Close()

		promoted, err := tuner.PromoteAndApply(updater, node, current, candidate, PromotionPolicy{MinObservations: 1})
		if err != nil {
			t.Fatalf("PromoteAndApply error: %v", err)
		}
		if promoted {
			t.Fatal("PromoteAndApply returned true, want false")
		}
		if len(updater.gridUpdates) != 0 {
			t.Errorf("gridUpdates count = %d, want 0", len(updater.gridUpdates))
		}
	})

	t.Run("nil error handling", func(t *testing.T) {
		cand := TuningCandidate{BlockDim: 128}
		if err := HotSwapNode(nil, node, cand); err == nil {
			t.Error("HotSwapNode(nil exec) expected error, got nil")
		}
		if err := HotSwapNode(&mockGraphUpdater{}, nil, cand); err == nil {
			t.Error("HotSwapNode(nil node) expected error, got nil")
		}
		if err := HotSwapKernelNodeUpdates(nil, node); err == nil {
			t.Error("HotSwapKernelNodeUpdates(nil exec) expected error, got nil")
		}
		if err := HotSwapKernelNodeUpdates(&mockGraphUpdater{}, nil); err == nil {
			t.Error("HotSwapKernelNodeUpdates(nil node) expected error, got nil")
		}
	})
}

func TestExtractCandidatePlans(t *testing.T) {
	eg := New()
	x := eg.Add(NewLeaf("x"))
	wNorm := eg.Add(NewLeaf("w_norm"))
	eps := eg.Add(NewLeaf("1e-5"))
	wQKV := eg.Add(NewLeaf("w_qkv"))

	norm := eg.Add(NewNode("rmsnorm", x, wNorm, eps))
	qkvProj := eg.Add(NewNode("gemv", norm, wQKV))

	rule := MustRewrite("fuse_rmsnorm_gemv",
		"(gemv (rmsnorm ?x ?w_n ?e) ?w_q)",
		"(fused_rmsnorm_gemv ?x ?w_n ?e ?w_q)",
	)
	Saturate(eg, []*Rewrite{rule}, 5)

	tests := []struct {
		name       string
		blockSizes []int
		wantMinLen int
	}{
		{
			name:       "specified block sizes 128 and 256",
			blockSizes: []int{128, 256},
			wantMinLen: 2,
		},
		{
			name:       "default block sizes when empty",
			blockSizes: nil,
			wantMinLen: 2,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			plans, err := ExtractCandidatePlans(eg, qkvProj, tt.blockSizes)
			if err != nil {
				t.Fatalf("ExtractCandidatePlans: %v", err)
			}
			if len(plans) < tt.wantMinLen {
				t.Fatalf("len(plans) = %d, want at least %d", len(plans), tt.wantMinLen)
			}
			hasBlock128 := false
			hasBlock256 := false
			for _, p := range plans {
				if p.BlockDim == 128 {
					hasBlock128 = true
				}
				if p.BlockDim == 256 {
					hasBlock256 = true
				}
			}
			if !hasBlock128 || !hasBlock256 {
				t.Errorf("missing expected block sizes: has128=%v, has256=%v", hasBlock128, hasBlock256)
			}
		})
	}

	if _, err := ExtractCandidatePlans(nil, qkvProj, nil); err == nil {
		t.Error("ExtractCandidatePlans(nil EGraph) expected error, got nil")
	}
}

func TestDeviceShadowExecutionIntegration(t *testing.T) {
	dev := cudatest.Require(t)
	defer dev.Close()

	const kernelSource = `
extern "C" __global__ void vectorAdd(float *a, float *b, float *c, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		c[i] = a[i] + b[i];
	}
}
`
	compiled, err := nvrtcx.Compile(kernelSource, nvrtcx.WithDeviceArch(dev), nvrtcx.WithNoCache())
	if err != nil {
		t.Fatal(err)
	}
	mod, err := compiled.Load(dev)
	if err != nil {
		t.Fatal(err)
	}
	defer dev.WithContext(mod.Unload)

	fn, err := mod.GetFunction("vectorAdd")
	if err != nil {
		t.Fatal(err)
	}

	stream, err := dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer dev.WithContext(stream.Destroy)

	// Test real ShadowTuner with isolated shadow stream and DeviceShadowExecutor
	tuner, err := NewShadowTuner(dev)
	if err != nil {
		t.Fatalf("NewShadowTuner: %v", err)
	}
	defer tuner.Close()

	cand := TuningCandidate{
		Expr:       &Expr{Op: "add"},
		BlockDim:   128,
		GridDim:    8,
		KernelName: "vectorAdd",
		CUDASource: kernelSource,
	}

	ctx := context.Background()
	dur, err := tuner.Evaluate(ctx, cand)
	if err != nil {
		t.Fatalf("tuner.Evaluate on real GPU: %v", err)
	}
	t.Logf("Measured GPU shadow latency for %s: %v", cand.Key(), dur)

	if score, ok := tuner.PGO().Score(cand.Key()); !ok || score <= 0 {
		t.Errorf("PGO Score = %v, %v, want > 0, true", score, ok)
	}

	// Test in-place graph update on real CUDA GraphExec
	const n = 1024
	bufA, err := cudamem.Alloc[float32](dev, n)
	if err != nil {
		t.Fatal(err)
	}
	defer dev.WithContext(bufA.Free)

	bufB, err := cudamem.Alloc[float32](dev, n)
	if err != nil {
		t.Fatal(err)
	}
	defer dev.WithContext(bufB.Free)

	bufC, err := cudamem.Alloc[float32](dev, n)
	if err != nil {
		t.Fatal(err)
	}
	defer dev.WithContext(bufC.Free)

	// Create and instantiate CUDA graph
	g, err := cudagraph.New(dev)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Destroy()

	count := int32(n)
	kernelArgs := []any{bufA, bufB, bufC, count}
	packed, err := cudalaunch.PackArgs(kernelArgs...)
	if err != nil {
		t.Fatal(err)
	}

	var kp driver.KernelNodeParams
	kp.Func = fn
	kp.GridDimX = n / 128
	kp.GridDimY = 1
	kp.GridDimZ = 1
	kp.BlockDimX = 128
	kp.BlockDimY = 1
	kp.BlockDimZ = 1
	if len(packed) > 0 {
		kp.KernelParams = uintptr(unsafe.Pointer(&packed[0]))
	}

	node, err := g.AddKernelNode(&kp)
	if err != nil {
		t.Fatal(err)
	}

	exec, err := g.Instantiate()
	if err != nil {
		t.Fatal(err)
	}
	defer exec.Destroy()

	// Hot-swap grid dimension in-place
	candNew := TuningCandidate{
		BlockDim: 256,
		GridDim:  n / 256,
	}
	if err := HotSwapNode(exec, node, candNew); err != nil {
		t.Fatalf("HotSwapNode on real GraphExec: %v", err)
	}
}
