//go:build linux

package main

import (
	"encoding/binary"
	"math"
	"sort"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudamem"
)

// Frozen before the edge-case device run. Pinned ds4_cuda.cu:12019 uses
// exp(x) below -20 and log1p(exp(x)) in the central interval. The old log(1+
// exp(x)) loses negative-tail scores to float32 cancellation. Normalized
// mixture weights expose that loss even when raw scores are below 1e-4.
const cudaRouterStabilityATol = 1e-4
const cudaRouterStabilityRTol = 1e-4

func TestCUDARouterStability(t *testing.T) {
	r := cudaSynthFoundation(t)
	dl, ds, dw := cudaTestAlloc(t, r, 256), cudaTestAlloc(t, r, 256), cudaTestAlloc(t, r, 6)
	selected, err := cudamem.Alloc[int32](r.device, 6)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := selected.Free(); err != nil {
			t.Error(err)
		}
	})
	hashIDs := []int32{255, 199, 128, 63, 7, 0}
	table := make([]byte, 2*6*4)
	// Distinct token rows ensure the hash lookup uses its token offset.
	for i := range 6 {
		binary.LittleEndian.PutUint32(table[4*i:], uint32(i+1))
		binary.LittleEndian.PutUint32(table[4*(6+i):], uint32(hashIDs[i]))
	}
	hash := cudaSynthTensor(t, r, "router hash", 26, []uint64{6, 2}, table)
	for _, name := range []string{"negative_tail", "below_denominator_floor", "branch_edges"} {
		t.Run(name, func(t *testing.T) {
			logits, bias := cudaRouterStabilityInputs(name)
			cudaCompressedUpload(t, dl, logits)
			bt := cudaCompressedTensor(t, r, "router bias", bias)
			for _, route := range []string{"biased", "hash"} {
				t.Run(route, func(t *testing.T) {
					scores := cudaRouterStabilityScores(logits, false)
					ids := append([]int32(nil), hashIDs...)
					if route == "biased" {
						ids = cudaRouterStabilitySelect(scores, bias)
					}
					want := cudaRouterStabilityWeights(scores, ids)
					poison := make([]float32, 256)
					for i := range poison {
						poison[i] = float32(math.NaN())
					}
					cudaCompressedUpload(t, ds, poison)
					cudaCompressedUpload(t, dw, poison[:6])
					if err := selected.CopyFromHost([]int32{-1, -1, -1, -1, -1, -1}); err != nil {
						t.Fatal(err)
					}
					cudaQualificationMeasure(t, r, func() error {
						if route == "hash" {
							return r.encodeRouterHash(dl, hash, ds, selected, dw, 1)
						}
						return r.encodeRouterBiased(dl, bt, ds, selected, dw)
					})
					gotIDs := make([]int32, 6)
					if err := selected.CopyToHost(gotIDs); err != nil {
						t.Fatal(err)
					}
					for i, id := range gotIDs {
						if id != ids[i] {
							t.Fatalf("selected[%d]=%d want%d", i, id, ids[i])
						}
					}
					gotScores, gotWeights := cudaTestRead(t, ds), cudaTestRead(t, dw)
					if err := cudaQualificationClose(gotScores, scores, cudaRouterStabilityATol, cudaRouterStabilityRTol); err != nil {
						t.Fatalf("scores: %v", err)
					}
					if err := cudaQualificationClose(gotWeights, want, cudaRouterStabilityATol, cudaRouterStabilityRTol); err != nil {
						t.Fatalf("weights: %v", err)
					}
					// All fixture logits are finite and >=-30, so every probability is
					// representable and strictly positive. A loose absolute gate alone
					// would not catch score cancellation in this interval.
					for i, v := range gotScores {
						if v <= 0 {
							t.Fatalf("score[%d]=%g lost representable negative tail", i, v)
						}
					}
					if name != "branch_edges" {
						wrong := cudaRouterStabilityWeights(cudaRouterStabilityScores(logits, true), ids)
						if cudaQualificationClose(wrong, want, cudaRouterStabilityATol, cudaRouterStabilityRTol) == nil {
							t.Fatal("old canceled softplus weights passed the fixture")
						}
						t.Logf("negative-tail normalized weights: got=%v old=%v", gotWeights, wrong)
					}
				})
			}
		})
	}
}

func cudaRouterStabilityInputs(name string) ([]float32, []float32) {
	logits, bias := make([]float32, 256), make([]float32, 256)
	edges := []float32{-30, math.Nextafter32(-20, float32(math.Inf(-1))), -20, math.Nextafter32(-20, 0), -17, -16, -.125, 0, .125, math.Nextafter32(20, 0), 20, math.Nextafter32(20, float32(math.Inf(1))), 30}
	for i := range logits {
		switch name {
		case "negative_tail":
			logits[i] = -30 + 10*float32(i)/255
		case "below_denominator_floor":
			logits[i] = -30 + float32(i)/255
		default:
			logits[i] = edges[i%len(edges)]
		}
	}
	for j, id := range []int{5, 37, 81, 129, 177, 223} {
		bias[id] = float32(6 - j)
	}
	return logits, bias
}
func cudaRouterStabilityScores(logits []float32, old bool) []float32 {
	out := make([]float32, len(logits))
	for i, v := range logits {
		x := float64(v)
		softplus := math.Log1p(math.Exp(x))
		if old && x <= 20 {
			softplus = math.Log(float64(float32(1 + float32(math.Exp(x)))))
		}
		if old && x > 20 {
			softplus = x
		}
		out[i] = float32(math.Sqrt(softplus))
	}
	return out
}
func cudaRouterStabilitySelect(scores, bias []float32) []int32 {
	order := make([]int, len(scores))
	for i := range order {
		order[i] = i
	}
	// Inputs to the rank comparison are float32, as in the pinned kernel;
	// sorting is independent of its iterative top-six implementation.
	sort.SliceStable(order, func(i, j int) bool { return scores[order[i]]+bias[order[i]] > scores[order[j]]+bias[order[j]] })
	out := make([]int32, 6)
	for i := range out {
		out[i] = int32(order[i])
	}
	return out
}
func cudaRouterStabilityWeights(scores []float32, ids []int32) []float32 {
	sum := 0.0
	for _, id := range ids {
		sum += float64(scores[id])
	}
	sum = math.Max(sum, math.Ldexp(1, -14))
	out := make([]float32, len(ids))
	for i, id := range ids {
		out[i] = float32(float64(scores[id]) / sum * 1.5)
	}
	return out
}
func TestCUDARouterStabilityReference(t *testing.T) {
	for _, name := range []string{"negative_tail", "below_denominator_floor"} {
		logits, bias := cudaRouterStabilityInputs(name)
		scores := cudaRouterStabilityScores(logits, false)
		ids := cudaRouterStabilitySelect(scores, bias)
		want := cudaRouterStabilityWeights(scores, ids)
		wrong := cudaRouterStabilityWeights(cudaRouterStabilityScores(logits, true), ids)
		if cudaQualificationClose(wrong, want, cudaRouterStabilityATol, cudaRouterStabilityRTol) == nil {
			t.Fatalf("%s does not reject lost negative-tail weights", name)
		}
	}
}
