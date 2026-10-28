package nvl72

import (
	"fmt"
	"math"
)

// EPAllToAll describes one MoE layer's expert-parallel dispatch and combine
// inside one NVLink domain, for [EstimateEPAllToAll].
type EPAllToAll struct {
	Gen           string // GenGB200 or GenGB300
	Ranks         int    // EP ranks, at most 72
	TokensPerRank int    // tokens each rank routes per layer step
	Hidden        int
	TopK          int
	// Experts counts expert slots, including redundant copies of hot
	// experts; a multiple of Ranks.
	Experts int
	// ExpertInter is the routed expert intermediate width, for the expert
	// stage time the collectives overlap with.
	ExpertInter int

	// DispatchBytes and CombineBytes are bytes per element on the wire:
	// 2 for BF16, 1 for FP8, 0.5625 for NVFP4 with one E4M3 scale per 16
	// values. Zero means 2.
	DispatchBytes, CombineBytes float64

	// RankDedup sends one copy of a token to each destination rank that
	// owns any of its experts, and combine returns one partial sum per
	// rank, as DeepEP and MoonEP do. Without it every route is a copy.
	RankDedup bool

	// LinkEfficiency is the achieved fraction of unidirectional NVLink
	// bandwidth; zero means 0.8.
	LinkEfficiency float64
	// CollectiveLatencyNs is the fixed cost of one all-to-all: launch,
	// rank barrier and flag polling. Zero means 3000.
	CollectiveLatencyNs float64
	// ExpertTFLOPS is the expert GEMM throughput per GPU; zero means the
	// generation's TensorTFLOPS (a dense BF16/FP8 figure, conservative
	// for NVFP4).
	ExpertTFLOPS float64
}

// EPAllToAllEstimate is a modelled layer step. Byte counts are per GPU and
// per direction; under uniform routing every GPU sends and receives the
// same amount.
type EPAllToAllEstimate struct {
	Copies       float64 // expected copies of a token sent, local included
	RemoteCopies float64 // expected copies sent to other ranks

	DispatchBytes float64 // remote dispatch payload plus route metadata
	CombineBytes  float64 // remote combine payload

	DispatchNs float64
	CombineNs  float64

	ExpertFLOPs       float64 // grouped W13 and W2 GEMMs on received routes
	ExpertWeightBytes float64 // local NVFP4 expert weights streamed once
	ExpertNs          float64 // max of compute and weight streaming

	// SequentialNs is dispatch, expert and combine back to back.
	SequentialNs float64
	// OverlappedNs splits the tokens into two microbatches and overlaps
	// one's collectives with the other's experts (two-batch overlap). Each
	// microbatch streams the expert weights again.
	OverlappedNs float64
	// CommFraction is the collectives' share of SequentialNs.
	CommFraction float64
}

// EstimateEPAllToAll models c's dispatch and combine all-to-alls under
// uniform routing on an NVSwitch domain, where each GPU's injection
// bandwidth, not the switch, bounds an all-to-all. The expected number of
// destination ranks per token under RankDedup is R·(1 − C(E−L, K)/C(E, K))
// for L = E/R experts per rank. It is a cost model only; nothing runs.
func EstimateEPAllToAll(c EPAllToAll) (EPAllToAllEstimate, error) {
	spec, err := LookupPodSpec(c.Gen)
	if err != nil {
		return EPAllToAllEstimate{}, err
	}
	R, S, H, K, E, I := c.Ranks, c.TokensPerRank, c.Hidden, c.TopK, c.Experts, c.ExpertInter
	switch {
	case R < 1 || R > NVL72MaxRanks:
		return EPAllToAllEstimate{}, fmt.Errorf("nvl72: %d EP ranks outside 1..%d", R, NVL72MaxRanks)
	case S < 1 || H < 1 || I < 1 || K < 1 || E < K || E%R != 0:
		return EPAllToAllEstimate{}, fmt.Errorf("nvl72: bad EP layer: %d tokens, hidden %d, inter %d, top-%d of %d experts over %d ranks", S, H, I, K, E, R)
	}
	dispatchB, combineB := orDefault(c.DispatchBytes, 2), orDefault(c.CombineBytes, 2)
	eff, lat := orDefault(c.LinkEfficiency, 0.8), orDefault(c.CollectiveLatencyNs, 3000)
	tflops := orDefault(c.ExpertTFLOPS, spec.TensorTFLOPS)
	L := E / R

	var est EPAllToAllEstimate
	if c.RankDedup {
		// Probability a given rank owns none of a token's K distinct experts.
		miss := math.Exp(lnChoose(E-L, K) - lnChoose(E, K))
		est.Copies = float64(R) * (1 - miss)
		est.RemoteCopies = float64(R-1) * (1 - miss)
	} else {
		est.Copies = float64(K)
		est.RemoteCopies = float64(K) * float64(R-1) / float64(R)
	}
	// Each copy carries its row plus K expert ids and weights.
	est.DispatchBytes = float64(S) * est.RemoteCopies * (float64(H)*dispatchB + 8*float64(K))
	est.CombineBytes = float64(S) * est.RemoteCopies * float64(H) * combineB
	bw := spec.Tier2NVLink5UniGBps * eff // bytes per ns
	est.DispatchNs = lat + est.DispatchBytes/bw
	est.CombineNs = lat + est.CombineBytes/bw

	expert := func(tokens int) (flops, bytes, ns float64) {
		flops = 6 * float64(tokens) * float64(K) * float64(H) * float64(I)
		bytes = float64(L) * 3 * float64(H) * float64(I) * 0.5625
		return flops, bytes, math.Max(flops/(tflops*1e3), bytes/spec.Tier1HBM3eGBps)
	}
	est.ExpertFLOPs, est.ExpertWeightBytes, est.ExpertNs = expert(S)
	est.SequentialNs = est.DispatchNs + est.ExpertNs + est.CombineNs
	est.CommFraction = (est.DispatchNs + est.CombineNs) / est.SequentialNs

	// Two microbatches: D0 | E0 ∥ D1 | E1 ∥ C0 | C1.
	half := max(S/2, 1)
	scale := float64(half) / float64(S)
	d := lat + est.DispatchBytes*scale/bw
	cmb := lat + est.CombineBytes*scale/bw
	_, _, e := expert(half)
	est.OverlappedNs = d + math.Max(e, d) + math.Max(e, cmb) + cmb
	return est, nil
}

func orDefault(v, def float64) float64 {
	if v == 0 {
		return def
	}
	return v
}

// lnChoose returns ln C(n, k), or -Inf when k > n.
func lnChoose(n, k int) float64 {
	if k > n {
		return math.Inf(-1)
	}
	a, _ := math.Lgamma(float64(n + 1))
	b, _ := math.Lgamma(float64(k + 1))
	c, _ := math.Lgamma(float64(n - k + 1))
	return a - b - c
}
