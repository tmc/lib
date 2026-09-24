package cudaref

import (
	"math"
	"testing"
)

// TestAdamWMatchesTorch replays three steps of torch.optim.AdamW with
// clip_grad_norm_(max_norm=2) over two groups, in float64. The expected
// values were printed by torch 2.10:
//
//	p = Parameter([1.0, -2.0, 0.5, 0.0]); q = Parameter([3.0, -0.25])
//	AdamW([{"params": [p], "weight_decay": 0.1},
//	       {"params": [q], "weight_decay": 0.0, "lr": 5e-3}],
//	      lr=1e-2, betas=(0.9, 0.999), eps=1e-8)
//
// followed, for each gradient pair, by clip_grad_norm_ and step.
func TestAdamWMatchesTorch(t *testing.T) {
	p := []float64{1.0, -2.0, 0.5, 0.0}
	q := []float64{3.0, -0.25}
	mp, vp := make([]float64, 4), make([]float64, 4)
	mq, vq := make([]float64, 2), make([]float64, 2)
	cp := AdamW{LR: 1e-2, Beta1: 0.9, Beta2: 0.999, Eps: 1e-8, WeightDecay: 0.1}
	cq := AdamW{LR: 5e-3, Beta1: 0.9, Beta2: 0.999, Eps: 1e-8}
	grads := []struct {
		p, q []float64
		norm float64
	}{
		{[]float64{0.5, -1.0, 2.0, 0.0}, []float64{1.0, -3.0}, 3.9051248379533274},
		{[]float64{0.1, 0.2, -0.3, 1e-4}, []float64{-2.0, 0.5}, 2.095232686362066},
		{[]float64{-4.0, 1.0, 0.0, 2.0}, []float64{0.25, 0.75}, 4.650268809434569},
	}
	for i, g := range grads {
		norm := GlobalNorm(g.p, g.q)
		if !near(norm, g.norm, 1e-15) {
			t.Fatalf("step %d: norm %v, want %v", i+1, norm, g.norm)
		}
		clip := ClipScale(norm, 2)
		cp.Step(p, mp, vp, g.p, i+1, clip)
		cq.Step(q, mq, vq, g.q, i+1, clip)
	}
	wantP := []float64{0.983386807122355, -1.98207095634356, 0.48063891085494664, -0.013821599772762674}
	wantQ := []float64{2.9996566254057435, -0.24194553015591613}
	for i := range wantP {
		if !near(p[i], wantP[i], 1e-13) {
			t.Errorf("p[%d] = %v, want %v", i, p[i], wantP[i])
		}
	}
	for i := range wantQ {
		if !near(q[i], wantQ[i], 1e-13) {
			t.Errorf("q[%d] = %v, want %v", i, q[i], wantQ[i])
		}
	}
}

func near(got, want, rel float64) bool {
	return math.Abs(got-want) <= rel*math.Max(math.Abs(want), 1e-300)
}

func TestClipScale(t *testing.T) {
	tests := []struct{ norm, max, want float64 }{
		{1, 0, 1},
		{1, -1, 1},
		{1, 2, 1},
		{4, 2, 2 / (4 + 1e-6)},
		{math.Inf(1), 2, 0},
	}
	for _, tt := range tests {
		if got := ClipScale(tt.norm, tt.max); got != tt.want {
			t.Errorf("ClipScale(%v, %v) = %v, want %v", tt.norm, tt.max, got, tt.want)
		}
	}
}
