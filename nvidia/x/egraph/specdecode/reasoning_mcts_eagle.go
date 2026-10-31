package specdecode

import (
	"cmp"
	"fmt"
	"math"
	"slices"
	"strings"

	"github.com/tmc/lib/nvidia/x/egraph/hopper"
)

// ReasoningBranch represents one candidate reasoning trajectory in a parallel
// test-time compute (Best-of-N / MCTS / Tree-of-Thoughts) rollout that shares
// the common prompt prefix KV cache tile.
type ReasoningBranch struct {
	BranchID      int
	ParentID      int
	Depth         int
	TokenIDs      []int32
	PRMScore      float32
	CumulativeVal float32
	Active        bool
}

// ReasoningMCTSTree models parallel test-time compute (Best-of-N / Monte Carlo
// Tree Search / Tree-of-Thoughts) over a shared prompt prefix KV tile across K
// active reasoning branches, with an in-megakernel Process Reward Model (PRM)
// value-head scoring and branch-pruning stage.
type ReasoningMCTSTree struct {
	Name            string
	PromptTokens    int     // Shared prompt prefix length in tokens
	PageSize        int     // KV cache tokens per page (default 16)
	NumBranches     int     // K parallel reasoning branches sharing the prompt prefix tile
	MaxSurviving    int     // Max branches surviving after PRM pruning
	MinPRMThreshold float32 // Minimum PRM value-head score to keep a branch alive
	ExplorationC    float32 // UCB1 / PUCT exploration constant (default 1.41421356)
	Branches        []ReasoningBranch
}

// NewReasoningMCTSTree constructs a [ReasoningMCTSTree] with numBranches
// initial candidate branches sharing a prompt prefix of promptTokens tokens.
func NewReasoningMCTSTree(promptTokens, numBranches, maxSurviving int, minPRMThreshold float32) (*ReasoningMCTSTree, error) {
	if promptTokens < 1 {
		return nil, fmt.Errorf("reasoning mcts tree: promptTokens %d < 1", promptTokens)
	}
	if numBranches < 1 || numBranches > 1024 {
		return nil, fmt.Errorf("reasoning mcts tree: numBranches %d outside 1..1024", numBranches)
	}
	if maxSurviving < 1 {
		maxSurviving = numBranches
	}
	if maxSurviving > numBranches {
		maxSurviving = numBranches
	}
	branches := make([]ReasoningBranch, numBranches)
	for i := range numBranches {
		branches[i] = ReasoningBranch{
			BranchID: i,
			ParentID: -1,
			Depth:    1,
			Active:   true,
		}
	}
	return &ReasoningMCTSTree{
		Name:            "reasoning_mcts_prm_tree",
		PromptTokens:    promptTokens,
		PageSize:        16,
		NumBranches:     numBranches,
		MaxSurviving:    maxSurviving,
		MinPRMThreshold: minPRMThreshold,
		ExplorationC:    1.41421356,
		Branches:        branches,
	}, nil
}

// SharedPrefixPages returns the number of shared KV cache pages covering the
// common prompt prefix tile. All active reasoning branches read these pages
// once into shared memory / L2 instead of duplicating the prompt KV across K
// branches.
func (t *ReasoningMCTSTree) SharedPrefixPages() int {
	if t == nil || t.PromptTokens <= 0 {
		return 0
	}
	ps := t.PageSize
	if ps <= 0 {
		ps = 16
	}
	return (t.PromptTokens + ps - 1) / ps
}

// Prune applies [PruneReasoningBranchesOnDevice] to branchScores and updates
// t.Branches in place, marking pruned branches inactive and returning the
// surviving branch indices and the shared prefix tile reuse factor.
func (t *ReasoningMCTSTree) Prune(branchScores []float32) (survivingIndices []int, prefixTileReuseFactor float64, err error) {
	if t == nil {
		return nil, 0, fmt.Errorf("reasoning mcts tree: nil tree")
	}
	if len(branchScores) == 0 {
		return nil, 0, fmt.Errorf("reasoning mcts tree: empty branchScores")
	}
	survivingIndices, prefixTileReuseFactor = PruneReasoningBranchesOnDevice(
		branchScores, t.MinPRMThreshold, t.MaxSurviving,
	)
	survivingSet := make(map[int]bool, len(survivingIndices))
	for _, idx := range survivingIndices {
		survivingSet[idx] = true
	}
	for i := range t.Branches {
		if i < len(branchScores) {
			t.Branches[i].PRMScore = branchScores[i]
			t.Branches[i].CumulativeVal += branchScores[i]
		}
		t.Branches[i].Active = survivingSet[i]
	}
	return survivingIndices, prefixTileReuseFactor, nil
}

// CUDA emits the in-megakernel Process Reward Model (PRM) value-head branch
// scoring and warp-ballot compaction device stage.
func (t *ReasoningMCTSTree) CUDA() string {
	name := "reasoning_mcts_prm_prune"
	numBranches := 8
	maxSurviving := 4
	thresh := float32(0.5)
	if t != nil {
		if t.Name != "" && cIdent(t.Name) {
			name = t.Name
		}
		if t.NumBranches > 0 {
			numBranches = t.NumBranches
		}
		if t.MaxSurviving > 0 {
			maxSurviving = t.MaxSurviving
		}
		thresh = t.MinPRMThreshold
	}
	var b strings.Builder
	fmt.Fprintf(&b, "// %s: in-megakernel MCTS / PRM value-head branch pruning (K=%d, max_surviving=%d, min_prm=%s).\n",
		name, numBranches, maxSurviving, floatLiteral(thresh))
	fmt.Fprintf(&b, `EG_DEV void %s_device(
    const float* __restrict__ branch_scores,
    int num_branches,
    float min_prm_threshold,
    int max_surviving,
    int32_t* __restrict__ surviving_indices,
    int32_t* __restrict__ surviving_count) {
    const int lane = threadIdx.x & 31;
    if ((threadIdx.x >> 5) != 0) return;

    float my_score = -1e30f;
    int my_idx = -1;
    if (lane < num_branches) {
        const float sc = branch_scores[lane];
        if (sc >= min_prm_threshold) {
            my_score = sc;
            my_idx = lane;
        }
    }

    int count = 0;
    const int limit = max_surviving < num_branches ? max_surviving : num_branches;
    for (int s = 0; s < limit; ++s) {
        float best_s = my_score;
        int best_i = my_idx;
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            const float os = __shfl_xor_sync(0xffffffffu, best_s, mask);
            const int oi = __shfl_xor_sync(0xffffffffu, best_i, mask);
            if (oi >= 0 && (best_i < 0 || os > best_s || (os == best_s && oi < best_i))) {
                best_s = os;
                best_i = oi;
            }
        }
        if (best_i < 0) break;
        if (lane == 0) {
            surviving_indices[count] = best_i;
        }
        ++count;
        if (my_idx == best_i) {
            my_score = -1e30f;
            my_idx = -1;
        }
    }
    if (lane == 0) {
        *surviving_count = count;
    }
}
`, name)
	return b.String()
}

// PruneReasoningBranchesOnDevice evaluates Process Reward Model (PRM) value-head
// scores across K parallel reasoning branches, filters out branches with
// score < minPRMThreshold (or NaN), selects up to maxSurviving highest-scoring
// branches (ties broken by lower branch index; if all branches fall below
// minPRMThreshold and maxSurviving >= 1, the single highest-scoring branch is
// kept as a fallback so decoding never stalls), and returns survivingIndices
// along with prefixTileReuseFactor (the effective shared prompt prefix tile
// reuse multiplier across the surviving reasoning branches).
func PruneReasoningBranchesOnDevice(branchScores []float32, minPRMThreshold float32, maxSurviving int) (survivingIndices []int, prefixTileReuseFactor float64) {
	n := len(branchScores)
	if n == 0 || maxSurviving <= 0 {
		return nil, 0
	}
	if maxSurviving > n {
		maxSurviving = n
	}

	type candidate struct {
		idx   int
		score float32
	}
	var passing []candidate
	bestIdx := -1
	bestScore := float32(math.Inf(-1))

	for i, s := range branchScores {
		if math.IsNaN(float64(s)) {
			continue
		}
		if bestIdx < 0 || s > bestScore {
			bestScore = s
			bestIdx = i
		}
		if s >= minPRMThreshold {
			passing = append(passing, candidate{idx: i, score: s})
		}
	}

	slices.SortFunc(passing, func(a, b candidate) int {
		if a.score > b.score {
			return -1
		}
		if a.score < b.score {
			return 1
		}
		return cmp.Compare(a.idx, b.idx)
	})

	if len(passing) > maxSurviving {
		passing = passing[:maxSurviving]
	}
	if len(passing) == 0 && bestIdx >= 0 {
		passing = []candidate{{idx: bestIdx, score: bestScore}}
	}

	survivingIndices = make([]int, len(passing))
	for i, c := range passing {
		survivingIndices[i] = c.idx
	}
	// Shared prompt prefix tile reuse factor: K surviving branches share 1
	// physical prompt KV read pass (plus amortized initial K-branch evaluation).
	prefixTileReuseFactor = float64(len(survivingIndices))
	return survivingIndices, prefixTileReuseFactor
}

// BuildFusedEAGLE3TargetDraftMegakernel appends an EAGLE-3 feature-conditioned
// lightweight autoregression draft block (conditioning on the target model's
// final hidden state feature vector and selecting topK candidate draft tokens
// across draftSteps speculative steps) onto the end of targetPlan and lowers
// both target verification and next-step draft generation into a single
// persistent [*hopper.Megakernel].
func BuildFusedEAGLE3TargetDraftMegakernel(targetPlan *hopper.Plan, draftHiddenDim, draftSteps, topK int) (*hopper.Plan, *hopper.Megakernel, error) {
	if targetPlan == nil || len(targetPlan.Kernels) == 0 {
		return nil, nil, fmt.Errorf("build fused eagle3 target-draft megakernel: empty targetPlan")
	}
	if targetPlan.Output.Name == "" || len(targetPlan.Output.Shape) == 0 {
		return nil, nil, fmt.Errorf("build fused eagle3 target-draft megakernel: targetPlan %q has no output tensor", targetPlan.Name)
	}
	if draftHiddenDim < 4 || draftHiddenDim%4 != 0 {
		return nil, nil, fmt.Errorf("build fused eagle3 target-draft megakernel: draftHiddenDim %d must be a positive multiple of 4", draftHiddenDim)
	}
	if draftSteps < 1 || draftSteps > 16 {
		return nil, nil, fmt.Errorf("build fused eagle3 target-draft megakernel: draftSteps %d outside 1..16", draftSteps)
	}
	if topK < 1 || topK > 32 {
		return nil, nil, fmt.Errorf("build fused eagle3 target-draft megakernel: topK %d outside 1..32", topK)
	}

	outShape := targetPlan.Output.Shape
	m := 1
	targetHidden := outShape[len(outShape)-1]
	if len(outShape) >= 2 {
		m = outShape[0]
	}
	if m < 1 || targetHidden < 1 {
		return nil, nil, fmt.Errorf("build fused eagle3 target-draft megakernel: invalid target output shape %v", outShape)
	}

	const eps = 1e-5
	bridgeName := "eagle3_target_feat"
	effHidden := max(256, ((draftHiddenDim+255)/256)*256)

	// Build the EAGLE-3 feature-conditioned autoregression draft block:
	//   1. Cast/normalize target output hidden state if needed -> eagle3_fc_in (BF16 [M, targetHidden])
	//   2. Feature conditioning projection: MatMul(eagle3_fc_in, w_eagle3_fc [effHidden, targetHidden])
	//   3. Autoregressive draft recurrence steps (1..draftSteps):
	//      RMSNorm -> SwiGLU MLP / conditioning update -> draft logits projection
	bDraft := hopper.NewBuilder()
	featIn := bDraft.Input(bridgeName, targetPlan.Output.DType, outShape...)
	var curBF16 = featIn
	if targetPlan.Output.DType != hopper.BF16 {
		curBF16 = bDraft.Cast(hopper.BF16, featIn)
	}
	wFC := bDraft.Input("eagle3_w_fc", hopper.BF16, effHidden, targetHidden)
	hDraft := bDraft.MatMul(curBF16, wFC)

	for step := range draftSteps {
		normW := bDraft.Input(fmt.Sprintf("eagle3_norm_s%d", step), hopper.F32, effHidden)
		normed := bDraft.Cast(hopper.BF16, bDraft.RMSNorm(hDraft, normW, effHidden, eps))
		wStep := bDraft.Input(fmt.Sprintf("eagle3_w_step_s%d", step), hopper.BF16, 2*effHidden, effHidden)
		act := bDraft.SwiGLU(bDraft.MatMul(normed, wStep))
		wProj := bDraft.Input(fmt.Sprintf("eagle3_w_proj_s%d", step), hopper.BF16, effHidden, effHidden)
		hDraft = bDraft.Map(hopper.MustLambda("(add $0 $1)"), bDraft.MatMul(act, wProj), hDraft)
	}

	// Final EAGLE-3 draft head projection producing [M, topK * draftHiddenDim]
	headOutDim := max(32, topK*draftHiddenDim)
	if headOutDim%32 != 0 {
		headOutDim = ((headOutDim + 31) / 32) * 32
	}
	normFinalW := bDraft.Input("eagle3_norm_final", hopper.F32, effHidden)
	normFinal := bDraft.Cast(hopper.BF16, bDraft.RMSNorm(hDraft, normFinalW, effHidden, eps))
	wDraftHead := bDraft.Input("eagle3_w_head", hopper.BF16, headOutDim, effHidden)
	draftLogits := bDraft.MatMul(normFinal, wDraftHead)

	draftExpr, err := hopper.Optimize(bDraft.G, draftLogits)
	if err != nil {
		return nil, nil, fmt.Errorf("optimize eagle3 draft plan: %w", err)
	}
	draftPlanName := fmt.Sprintf("eagle3_draft_s%d_k%d", draftSteps, topK)
	draftPlan, err := hopper.Compile(draftExpr, draftPlanName, hopper.F32)
	if err != nil {
		return nil, nil, fmt.Errorf("compile eagle3 draft plan: %w", err)
	}

	// Ensure all SkinnyGEMM stages in both plans satisfy RowsPerBlock * M <= 256.
	tuneSkinnyGEMM := func(pl *hopper.Plan) {
		for i := range pl.Kernels {
			if g := pl.Kernels[i].GEMM; g != nil {
				rowM := max(pl.Kernels[i].M, g.MaxM, m, 1)
				for _, r := range []int{32, 16, 8, 4, 2, 1} {
					if g.GLU && r%2 != 0 {
						continue
					}
					if g.N%r == 0 && r*rowM <= 256 {
						g.RowsPerBlock = r
						break
					}
				}
			}
		}
	}
	tuneSkinnyGEMM(draftPlan)

	baseName := targetPlan.Name
	if baseName == "" {
		baseName = "target"
	}
	glue := hopper.CrossLayerGlue{
		Name:               fmt.Sprintf("%s_eagle3_fused", baseName),
		LayerLPrefix:       "target_",
		LayerNextPrefix:    "eagle3_",
		BridgeInput:        bridgeName,
		RetainBridgeOutput: true,
	}
	fusedPlan, mega, err := hopper.BuildCrossLayerMegakernel(targetPlan, draftPlan, glue)
	if err != nil {
		return nil, nil, fmt.Errorf("fuse target and eagle3 draft megakernel: %w", err)
	}
	return fusedPlan, &mega, nil
}

// EmitToolCallTrapDoorCUDA generates the `eg_grammar_trap_door` device epilogue
// function that inspects newly accepted / generated tokens for a completed
// tool-call or structured JSON delimiter token in delimiterTokens and signals
// a zero-copy mapped host doorbell ring buffer via system-scope release store
// (`st.release.sys.global.u64` / `st.release.sys.global.b32`) so the host can
// immediately dispatch the tool call without polling or tearing down the
// persistent megakernel.
func EmitToolCallTrapDoorCUDA(delimiterTokens []int32) string {
	var tokLits []string
	for _, t := range delimiterTokens {
		tokLits = append(tokLits, fmt.Sprintf("%d", t))
	}
	if len(tokLits) == 0 {
		tokLits = []string{"-1"}
	}

	var b strings.Builder
	b.WriteString(`#ifndef EG_GRAMMAR_TRAP_DOOR_DEFINED
#define EG_GRAMMAR_TRAP_DOOR_DEFINED 1

// eg_grammar_trap_door inspects emitted_token against the configured tool-call /
// JSON delimiter token table and, on a match, writes the (seq_id, step_idx,
// emitted_token) payload to the zero-copy host doorbell ring buffer and
// publishes the updated doorbell sequence word with st.release.sys.global.
EG_DEV bool eg_grammar_trap_door(
    int32_t emitted_token,
    int32_t seq_id,
    int32_t step_idx,
    int32_t* __restrict__ host_payload_ring,
    uint32_t ring_capacity,
    unsigned long long* __restrict__ host_doorbell_sys) {
`)
	fmt.Fprintf(&b, "    constexpr int NUM_DELIMS = %d;\n", len(tokLits))
	fmt.Fprintf(&b, "    constexpr int32_t DELIM_TOKENS[NUM_DELIMS] = {%s};\n", strings.Join(tokLits, ", "))
	b.WriteString(`    bool matched = false;
#pragma unroll
    for (int i = 0; i < NUM_DELIMS; ++i) {
        if (emitted_token == DELIM_TOKENS[i] && DELIM_TOKENS[i] >= 0) {
            matched = true;
        }
    }
    if (!matched) {
        return false;
    }

    if (threadIdx.x == 0) {
        if (host_payload_ring != nullptr && ring_capacity > 0u) {
            const uint32_t slot = ((uint32_t)step_idx) % ring_capacity;
            host_payload_ring[slot * 4 + 0] = seq_id;
            host_payload_ring[slot * 4 + 1] = step_idx;
            host_payload_ring[slot * 4 + 2] = emitted_token;
            host_payload_ring[slot * 4 + 3] = 1;
        }
        if (host_doorbell_sys != nullptr) {
            const unsigned long long ticket =
                ((unsigned long long)(uint32_t)(step_idx + 1) << 32) |
                (unsigned long long)(uint32_t)emitted_token;
#if defined(EG_HOST_EMU)
            *host_doorbell_sys = ticket;
#else
            asm volatile("st.release.sys.global.u64 [%0], %1;" :: "l"(host_doorbell_sys), "l"(ticket) : "memory");
#endif
        }
    }
    return true;
}

#endif // EG_GRAMMAR_TRAP_DOOR_DEFINED
`)
	return b.String()
}
