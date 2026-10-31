package hopper

import (
	"math"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"
)

func TestQuestKVVQFastWalshHadamardInvolutionAndOutlierSuppression(t *testing.T) {
	for _, headDim := range []int{16, 32, 64, 128, 256} {
		rng := rand.New(rand.NewPCG(uint64(headDim), 42))
		x := make([]float32, headDim)
		for i := range x {
			x[i] = float32(rng.NormFloat64()) * 0.25
		}
		// Inject a massive activation outlier spike on one channel (typical of LLM K/V outliers).
		outlierIdx := headDim / 3
		x[outlierIdx] = 32.0

		var origNormSq, origMaxAbs float64
		for _, v := range x {
			av := math.Abs(float64(v))
			if av > origMaxAbs {
				origMaxAbs = av
			}
			origNormSq += float64(v) * float64(v)
		}

		rot, err := FastWalshHadamardHost(x)
		if err != nil {
			t.Fatalf("FastWalshHadamardHost(d=%d): %v", headDim, err)
		}

		var rotNormSq, rotMaxAbs float64
		for _, v := range rot {
			av := math.Abs(float64(v))
			if av > rotMaxAbs {
				rotMaxAbs = av
			}
			rotNormSq += float64(v) * float64(v)
		}

		// Verify orthogonality (energy preservation ||H_d x||_2 == ||x||_2).
		if diff := math.Abs(rotNormSq - origNormSq); diff > 1e-3 {
			t.Fatalf("d=%d energy not preserved: got %v, want %v (diff=%v)", headDim, rotNormSq, origNormSq, diff)
		}

		// Verify outlier suppression: a spike of magnitude M is spread by ~1/sqrt(d),
		// so rotMaxAbs must be substantially smaller than origMaxAbs.
		if rotMaxAbs >= origMaxAbs*0.65 {
			t.Fatalf("d=%d outlier not suppressed by Hadamard rotation: rotMaxAbs=%v vs origMaxAbs=%v", headDim, rotMaxAbs, origMaxAbs)
		}

		// Verify exact involutive property: H_d(H_d(x)) == x.
		recon, err := FastWalshHadamardHost(rot)
		if err != nil {
			t.Fatalf("inverse FastWalshHadamardHost(d=%d): %v", headDim, err)
		}
		for i := range x {
			if diff := math.Abs(float64(recon[i] - x[i])); diff > 1e-4 {
				t.Fatalf("d=%d involution mismatch at %d: got %v, want %v (diff=%v)", headDim, i, recon[i], x[i], diff)
			}
		}
	}

	// Non-power-of-two must return an error.
	if _, err := FastWalshHadamardHost(make([]float32, 48)); err == nil {
		t.Fatal("expected error for non-power-of-two length 48")
	}
}

func TestQuestKVVQQuantizePageHadamardAndBoundingBox(t *testing.T) {
	const (
		headDim  = 64
		pageSize = 16
	)
	rng := rand.New(rand.NewPCG(101, 202))
	kPage := make([]float32, pageSize*headDim)
	vPage := make([]float32, pageSize*headDim)
	for i := range kPage {
		kPage[i] = float32(rng.NormFloat64()) * 0.5
		vPage[i] = float32(rng.NormFloat64()) * 0.5
	}
	// Inject outlier channels into K and V.
	for tok := range pageSize {
		kPage[tok*headDim+7] += 12.0
		vPage[tok*headDim+19] -= 9.5
	}

	for _, kvBits := range []int{2, 4} {
		packedK, packedV, scalesK, scalesV, kMin, kMax, err := QuantizeKVPageHadamard(kPage, vPage, headDim, kvBits)
		if err != nil {
			t.Fatalf("QuantizeKVPageHadamard(%d-bit): %v", kvBits, err)
		}

		wantBytes := (pageSize * headDim * kvBits) / 8
		if len(packedK) != wantBytes || len(packedV) != wantBytes {
			t.Fatalf("%d-bit packed bytes = (%d, %d), want %d", kvBits, len(packedK), len(packedV), wantBytes)
		}
		if len(scalesK) != pageSize || len(scalesV) != pageSize {
			t.Fatalf("%d-bit scales len = (%d, %d), want %d", kvBits, len(scalesK), len(scalesV), pageSize)
		}
		if len(kMin) != headDim || len(kMax) != headDim {
			t.Fatalf("kMin/kMax len = (%d, %d), want %d", len(kMin), len(kMax), headDim)
		}

		// Verify bounding box encloses all tokens in the page.
		for d := range headDim {
			wantMin := kPage[d]
			wantMax := kPage[d]
			for tok := 1; tok < pageSize; tok++ {
				v := kPage[tok*headDim+d]
				if v < wantMin {
					wantMin = v
				}
				if v > wantMax {
					wantMax = v
				}
			}
			if kMin[d] != wantMin || kMax[d] != wantMax {
				t.Fatalf("dim %d bbox = [%v, %v], want [%v, %v]", d, kMin[d], kMax[d], wantMin, wantMax)
			}
		}

		// Verify dequantization in the Hadamard domain preserves high cosine similarity.
		kRecon, vRecon, err := DequantizeKVPageHadamard(packedK, packedV, scalesK, scalesV, headDim, kvBits)
		if err != nil {
			t.Fatalf("DequantizeKVPageHadamard(%d-bit): %v", kvBits, err)
		}
		cosK := cosineSimilarityF32(kPage, kRecon)
		cosV := cosineSimilarityF32(vPage, vRecon)
		minCos := 0.85
		if kvBits == 4 {
			minCos = 0.98
		}
		if cosK < minCos || cosV < minCos {
			t.Fatalf("%d-bit reconstruction cosine similarity too low: cosK=%.4f, cosV=%.4f (want >= %.2f)", kvBits, cosK, cosV, minCos)
		}
	}
}

func TestQuestKVVQTopKPageRecall(t *testing.T) {
	const (
		headDim    = 64
		pageSize   = 16
		totalPages = 32
		topK       = 4
	)
	rng := rand.New(rand.NewPCG(777, 888))
	q := make([]float32, headDim)
	for d := range headDim {
		q[d] = float32(rng.NormFloat64())
	}

	// Plant 4 high-relevance needle pages aligned with q among 32 pages.
	needlePages := []int{3, 11, 19, 27}
	pageKMin := make([][]float32, totalPages)
	pageKMax := make([][]float32, totalPages)
	exactPageMaxDot := make([]float32, totalPages)

	for p := range totalPages {
		kPage := make([]float32, pageSize*headDim)
		vPage := make([]float32, pageSize*headDim)
		for i := range kPage {
			kPage[i] = float32(rng.NormFloat64()) * 0.2
			vPage[i] = float32(rng.NormFloat64()) * 0.2
		}
		if slices.Contains(needlePages, p) {
			// Inject a needle token strongly aligned with q.
			needleTok := p % pageSize
			for d := range headDim {
				kPage[needleTok*headDim+d] = q[d] * 3.0
			}
		}
		maxDot := float32(-1e30)
		for tok := range pageSize {
			var dot float32
			for d := range headDim {
				dot += q[d] * kPage[tok*headDim+d]
			}
			if dot > maxDot {
				maxDot = dot
			}
		}
		exactPageMaxDot[p] = maxDot

		_, _, _, _, kMin, kMax, err := QuantizeKVPageHadamard(kPage, vPage, headDim, 2)
		if err != nil {
			t.Fatalf("QuantizeKVPageHadamard page %d: %v", p, err)
		}
		pageKMin[p] = kMin
		pageKMax[p] = kMax
	}

	gotTopIdx, gotScores := SelectQuestTopKPages(q, pageKMin, pageKMax, topK)
	if len(gotTopIdx) != topK || len(gotScores) != topK {
		t.Fatalf("SelectQuestTopKPages len = (%d, %d), want %d", len(gotTopIdx), len(gotScores), topK)
	}

	// Verify upper-bound property: Quest score(p) >= exact max_{t in p} <q, k_t>.
	for i, p := range gotTopIdx {
		if gotScores[i]+1e-4 < exactPageMaxDot[p] {
			t.Fatalf("page %d Quest upper bound %v < exact max dot %v", p, gotScores[i], exactPageMaxDot[p])
		}
	}

	// Verify 100% recall of the planted needle pages.
	hits := 0
	for _, needle := range needlePages {
		if slices.Contains(gotTopIdx, needle) {
			hits++
		}
	}
	if hits != len(needlePages) {
		t.Fatalf("Quest Top-%d recall = %d/%d (got pages %v, want needles %v)", topK, hits, len(needlePages), gotTopIdx, needlePages)
	}
}

func TestQuestKVVQCUDAGeneration(t *testing.T) {
	for _, kvBits := range []int{2, 4} {
		cfg := QuestKVVQConfig{
			Name:       "quest_hadamard_attn",
			Batch:      2,
			NumQHeads:  32,
			NumKVHeads: 8,
			HeadDim:    64,
			PageSize:   16,
			TotalPages: 64,
			TopKPages:  8,
			KVBits:     kvBits,
			Arch:       TargetSM121A,
		}
		src := cfg.CUDA()
		wantSubstrings := []string{
			"__global__ void __launch_bounds__(128) quest_hadamard_attn(",
			"__shfl_xor_sync(0xffffffff, val, lane_mask)",
			"eg_fwht_warp32",
			"page_k_min",
			"page_k_max",
			"k_vq_cache",
			"v_vq_cache",
			"TOPK_PAGES = 8",
		}
		for _, sub := range wantSubstrings {
			if !strings.Contains(src, sub) {
				t.Fatalf("%d-bit CUDA source missing %q\n--- source ---\n%s", kvBits, sub, src)
			}
		}
	}
}

func cosineSimilarityF32(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		x, y := float64(a[i]), float64(b[i])
		dot += x * y
		na += x * x
		nb += y * y
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}
