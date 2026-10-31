package specdecode

import (
	"context"
	"encoding/binary"
	"fmt"
	"maps"
	"math"
	"math/bits"
	"slices"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudahostemu"
	"github.com/tmc/lib/nvidia/x/egraph/hopper"
)

// A DraftTreeTopology is a speculative draft tree of at most 64 nodes in
// topological order. Node 0 is the root: the token the previous step
// emitted, which the target model has not yet seen. Every other node is a
// draft token whose parent precedes it.
type DraftTreeTopology struct {
	ParentIdx    []int    // ParentIdx[0] == -1; 0 <= ParentIdx[i] < i for i > 0
	AncestorMask []uint64 // bit j of AncestorMask[i] is set iff j is i or an ancestor of i
	Depth        []int    // Depth[0] == 0
}

// BuildDraftTreeTopology validates parentIdx and computes the ancestor masks
// that [hopper.MTPPagedAttention] takes as its draft mask. Unlike
// [hopper.TreeDraftMask] it requires a single tree: only node 0 may have
// parent -1.
func BuildDraftTreeTopology(parentIdx []int) (DraftTreeTopology, error) {
	mask, err := hopper.TreeDraftMask(parentIdx)
	if err != nil {
		return DraftTreeTopology{}, fmt.Errorf("build draft tree: %w", err)
	}
	depth := make([]int, len(parentIdx))
	for i := 1; i < len(parentIdx); i++ {
		p := parentIdx[i]
		if p < 0 {
			return DraftTreeTopology{}, fmt.Errorf("build draft tree: node %d has no parent; only node 0 may be a root", i)
		}
		depth[i] = depth[p] + 1
	}
	return DraftTreeTopology{
		ParentIdx:    slices.Clone(parentIdx),
		AncestorMask: mask,
		Depth:        depth,
	}, nil
}

// NumNodes returns the number of nodes in t.
func (t DraftTreeTopology) NumNodes() int { return len(t.ParentIdx) }

// A TreeAccept is the outcome of verifying one draft tree.
type TreeAccept struct {
	// Path lists the accepted nodes from the root: Path[0] == 0 and
	// Path[j+1] is a child of Path[j]. Its tokens are committed to the
	// sequence and their KV entries kept.
	Path []int
	// Bonus is the token the target model emits after the last node of
	// Path: its greedy choice there, or a sample from the residual
	// distribution. It becomes the root of the next tree.
	Bonus int32
}

// AcceptDraftTreeGreedy verifies tree against greedy target decoding.
// tokens[i] is node i's token and targetArgmax[i] the target model's argmax
// next token after the root-to-i path. Starting from the root, it moves to
// the first child (in node order) whose token equals the target's argmax and
// stops when no child does, so Path is the longest root-to-node path that
// greedy decoding of the target would itself produce.
func AcceptDraftTreeGreedy(tree DraftTreeTopology, tokens, targetArgmax []int32) (TreeAccept, error) {
	n := tree.NumNodes()
	if len(tokens) != n || len(targetArgmax) != n {
		return TreeAccept{}, fmt.Errorf("accept draft tree: %d nodes, %d tokens, %d target tokens", n, len(tokens), len(targetArgmax))
	}
	path := []int{0}
	cur := 0
	for {
		next := -1
		for c := cur + 1; c < n; c++ {
			if tree.ParentIdx[c] == cur && tokens[c] == targetArgmax[cur] {
				next = c
				break
			}
		}
		if next < 0 {
			break
		}
		path = append(path, next)
		cur = next
	}
	return TreeAccept{Path: path, Bonus: targetArgmax[cur]}, nil
}

// AcceptDraftTreeSampled verifies tree by multi-candidate speculative
// sampling (SpecInfer, Miao et al. 2024). draftProbs[i] is the draft
// distribution node i's children were drawn from, independently, and
// targetProbs[i] the target's next-token distribution after the root-to-i
// path. At each node the children are tried in order: child x is accepted
// with probability min(1, r(x)/q(x)), where r starts as the target
// distribution and becomes norm(max(0, r-q)) after each rejection; if every
// child is rejected, or there are none, the bonus token is drawn from r.
// If each node's children were drawn independently from q, the emitted
// tokens are distributed exactly as target sampling.
//
// The k-th uniform drawn in the walk is word 0 of
// Philox4x32({step, seq, k, 0}, {seed, seed>>32}), and the arithmetic is
// float32 in the order [TreeSpecVerify]'s kernel uses, so the two agree bit
// for bit.
func AcceptDraftTreeSampled(tree DraftTreeTopology, tokens []int32, draftProbs, targetProbs [][]float32, seed uint64, step, seq uint32) (TreeAccept, error) {
	n := tree.NumNodes()
	if len(tokens) != n || len(draftProbs) != n || len(targetProbs) != n {
		return TreeAccept{}, fmt.Errorf("accept draft tree: %d nodes, %d tokens, %d draft and %d target distributions",
			n, len(tokens), len(draftProbs), len(targetProbs))
	}
	v := len(targetProbs[0])
	for i := range n {
		if len(draftProbs[i]) != v || len(targetProbs[i]) != v {
			return TreeAccept{}, fmt.Errorf("accept draft tree: node %d distributions have %d and %d entries, want %d",
				i, len(draftProbs[i]), len(targetProbs[i]), v)
		}
		if t := tokens[i]; i > 0 && (t < 0 || int(t) >= v) {
			return TreeAccept{}, fmt.Errorf("accept draft tree: node %d token %d outside vocabulary of %d", i, t, v)
		}
	}
	key := [2]uint32{uint32(seed), uint32(seed >> 32)}
	draws := uint32(0)
	uniform := func() float32 {
		w := Philox4x32([4]uint32{step, seq, draws, 0}, key)
		draws++
		return philoxUniform(w[0])
	}
	r := make([]float32, v)
	path := []int{0}
	cur := 0
	for {
		q := draftProbs[cur]
		copy(r, targetProbs[cur])
		next := -1
		for c := cur + 1; c < n; c++ {
			if tree.ParentIdx[c] != cur {
				continue
			}
			x := tokens[c]
			if uniform()*q[x] <= r[x] {
				next = c
				break
			}
			var sum float32
			for i := range r {
				r[i] = max(r[i]-q[i], 0)
				sum += r[i]
			}
			if sum > 0 {
				for i := range r {
					r[i] /= sum
				}
			}
		}
		if next < 0 {
			break
		}
		path = append(path, next)
		cur = next
	}
	return TreeAccept{Path: path, Bonus: sampleResidual(r, uniform())}, nil
}

// sampleResidual draws from the unnormalized distribution r using u in
// [0, 1), scanning in token order as the kernel does.
func sampleResidual(r []float32, u float32) int32 {
	var total float32
	for _, x := range r {
		total += x
	}
	target := u * total
	var cum float32
	last := int32(-1)
	for i, x := range r {
		if x <= 0 {
			continue
		}
		last = int32(i)
		cum += x
		if cum > target {
			return int32(i)
		}
	}
	return last
}

// TreeSpecVerify configures the device-resident tree verification kernels
// that form the tail of a CUDA graph conditional WHILE loop. Each iteration
// of the loop runs
//
//	Name_reserve   pop free KV pages so every active sequence can hold a tree
//	(draft, target forward)  write the tree tokens, target outputs and tree KV
//	Name           walk each tree, compact the accepted KV, free the pages
//	               the accepted prefix no longer needs, set the condition
//
// with no host round trip. Name_reserve and Name are emitted here; the draft
// and target forward passes are the caller's. The target forward writes
// node i's KV at sequence position seqlen+i, using the tree mask from
// [BuildDraftTreeTopology] in [hopper.MTPPagedAttention].
//
// Both kernels run as one warp; lane b handles sequence b and lane 0
// updates the shared free-page stack in sequence order, so the page
// assignment is deterministic and matches [TreeSpecVerify.Reference]. The
// KV copy is plain 32-bit loads and stores, not TMA; this is a control
// kernel, not a bandwidth-tuned one.
//
// The kernels use no sm_90+ instructions. The conditional is set through
// cudaGraphSetConditional (CUDA 12.4+). They were checked on the host
// emulator ([TreeSpecVerify.RunEmulatedLoop]); a native graph harness for
// sm_121a is in the CUDA_EGRAPH_HOPPER_NATIVE tests.
type TreeSpecVerify struct {
	Name      string
	Nodes     int  // tree nodes per sequence per step, 1..64
	Batch     int  // sequences, 1..32
	KVWords   int  // 32-bit words of K and V per token, all layers (1..4096)
	PageSize  int  // tokens per KV page
	MaxPages  int  // page_table stride per sequence
	MaxSteps  int  // WHILE iteration cap (0 means 64)
	Sampled   bool // speculative sampling instead of greedy verification
	VocabSize int  // with Sampled: distribution length
}

func (s TreeSpecVerify) norm() (TreeSpecVerify, error) {
	if s.Name == "" {
		s.Name = "tree_spec_verify"
	}
	if !cIdent(s.Name) {
		return s, fmt.Errorf("tree spec verify: bad name %q", s.Name)
	}
	if s.MaxSteps == 0 {
		s.MaxSteps = 64
	}
	switch {
	case s.Nodes < 1 || s.Nodes > 64:
		return s, fmt.Errorf("tree spec verify %s: Nodes %d outside 1..64", s.Name, s.Nodes)
	case s.Batch < 1 || s.Batch > 32:
		return s, fmt.Errorf("tree spec verify %s: Batch %d outside 1..32", s.Name, s.Batch)
	case s.KVWords < 1 || s.KVWords > 4096:
		return s, fmt.Errorf("tree spec verify %s: KVWords %d outside 1..4096", s.Name, s.KVWords)
	case s.PageSize < 1:
		return s, fmt.Errorf("tree spec verify %s: PageSize %d < 1", s.Name, s.PageSize)
	case s.MaxPages*s.PageSize < s.Nodes+1:
		return s, fmt.Errorf("tree spec verify %s: MaxPages %d * PageSize %d cannot hold a %d-node tree", s.Name, s.MaxPages, s.PageSize, s.Nodes)
	case s.MaxSteps < 1 || s.MaxSteps > 4096:
		return s, fmt.Errorf("tree spec verify %s: MaxSteps %d outside 1..4096", s.Name, s.MaxSteps)
	case s.Sampled && (s.VocabSize < 2 || s.VocabSize > 1<<20):
		return s, fmt.Errorf("tree spec verify %s: VocabSize %d outside 2..%d", s.Name, s.VocabSize, 1<<20)
	}
	return s, nil
}

// Grid returns the launch shape of both kernels: one block of one warp.
func (s TreeSpecVerify) Grid() (grid, block [3]int) {
	return [3]int{1, 1, 1}, [3]int{32, 1, 1}
}

func (s TreeSpecVerify) reserveParams() []string {
	return []string{
		"int32_t* __restrict__ page_table",
		"int32_t* __restrict__ num_pages",
		"int32_t* __restrict__ free_pages",
		"int32_t* __restrict__ free_top",
		"const int32_t* __restrict__ seqlens",
		"int32_t* __restrict__ active",
	}
}

func (s TreeSpecVerify) params() []string {
	p := []string{
		"const int32_t* __restrict__ parent",
		"const int32_t* __restrict__ tree_tokens",
	}
	if s.Sampled {
		p = append(p,
			"const float* __restrict__ draft_probs",
			"const float* __restrict__ target_probs",
			"float* __restrict__ resid",
			"uint64_t seed",
		)
	} else {
		p = append(p, "const int32_t* __restrict__ target_argmax")
	}
	return append(p,
		"uint32_t* __restrict__ kv",
		"int32_t* __restrict__ page_table",
		"int32_t* __restrict__ num_pages",
		"int32_t* __restrict__ free_pages",
		"int32_t* __restrict__ free_top",
		"int32_t* __restrict__ seqlens",
		"int32_t* __restrict__ active",
		"int32_t* __restrict__ out_tokens",
		"int32_t* __restrict__ root_tokens",
		"int32_t* __restrict__ accepted",
		"int32_t* __restrict__ step_idx",
		"int32_t eos_token_id",
		"cudaGraphConditionalHandle cond_handle",
	)
}

// CUDA returns both kernels (without [hopper.Prelude]).
func (s TreeSpecVerify) CUDA() (string, error) {
	s, err := s.norm()
	if err != nil {
		return "", err
	}
	var b strings.Builder
	b.WriteString(specDecodeCondPrelude)
	if s.Sampled {
		b.WriteString(philoxDevice)
	}
	fmt.Fprintf(&b, "// %s: device-resident draft tree verification, %d nodes x %d sequences, %d KV words/token, page %d.\n",
		s.Name, s.Nodes, s.Batch, s.KVWords, s.PageSize)
	fmt.Fprintf(&b, "__global__ void __launch_bounds__(32) %s_reserve(\n    %s) {\n", s.Name, strings.Join(s.reserveParams(), ",\n    "))
	fmt.Fprintf(&b, "    constexpr int N = %d, B = %d, PAGE = %d, MAX_PAGES = %d;\n", s.Nodes, s.Batch, s.PageSize, s.MaxPages)
	b.WriteString(`    if (threadIdx.x != 0) return;
    // Lane 0 pops pages in sequence order so the assignment is deterministic.
    for (int b = 0; b < B; ++b) {
        if (!active[b]) continue;
        const int need = (seqlens[b] + N + PAGE - 1) / PAGE;
        if (need > MAX_PAGES) {
            active[b] = 0;
            continue;
        }
        while (num_pages[b] < need && *free_top > 0) {
            page_table[b * MAX_PAGES + num_pages[b]] = free_pages[--*free_top];
            ++num_pages[b];
        }
        if (num_pages[b] < need) active[b] = 0;
    }
}

`)
	fmt.Fprintf(&b, "__global__ void __launch_bounds__(32) %s(\n    %s) {\n", s.Name, strings.Join(s.params(), ",\n    "))
	fmt.Fprintf(&b, "    constexpr int N = %d, B = %d, KVW = %d, PAGE = %d, MAX_PAGES = %d, MAX_STEPS = %d;\n",
		s.Nodes, s.Batch, s.KVWords, s.PageSize, s.MaxPages, s.MaxSteps)
	if s.Sampled {
		fmt.Fprintf(&b, "    constexpr int V = %d;\n", s.VocabSize)
	}
	b.WriteString(`    __shared__ int32_t keep_pages[B];
    const int b = threadIdx.x;
    const int step = *step_idx;
    if (b < B) {
        keep_pages[b] = (seqlens[b] + PAGE - 1) / PAGE;
        if (step < MAX_STEPS) accepted[step * B + b] = 0;
    }
    if (b < B && active[b] && step < MAX_STEPS) {
        const int32_t* par = parent + b * N;
        const int32_t* tok = tree_tokens + b * N;
        int32_t path[N];
        int len = 1, cur = 0;
        path[0] = 0;
`)
	if s.Sampled {
		b.WriteString(`        float* r = resid + (size_t)b * V;
        uint32_t draws = 0;
        const uint32_t k0 = (uint32_t)seed, k1 = (uint32_t)(seed >> 32);
        for (;;) {
            const float* q = draft_probs + (size_t)(b * N + cur) * V;
            const float* p = target_probs + (size_t)(b * N + cur) * V;
            for (int v = 0; v < V; ++v) r[v] = p[v];
            int next = -1;
            for (int c = cur + 1; c < N; ++c) {
                if (par[c] != cur) continue;
                const int32_t x = tok[c];
                uint32_t w[4];
                eg_philox4x32((uint32_t)step, (uint32_t)b, draws++, 0u, k0, k1, w);
                const float u = eg_philox_uniform(w[0]);
                if (u * q[x] <= r[x]) {
                    next = c;
                    break;
                }
                float sum = 0.0f;
                for (int v = 0; v < V; ++v) {
                    r[v] = fmaxf(r[v] - q[v], 0.0f);
                    sum += r[v];
                }
                if (sum > 0.0f) {
                    for (int v = 0; v < V; ++v) r[v] = r[v] / sum;
                }
            }
            if (next < 0) break;
            path[len++] = next;
            cur = next;
        }
        int32_t bonus = -1;
        {
            uint32_t w[4];
            eg_philox4x32((uint32_t)step, (uint32_t)b, draws++, 0u, k0, k1, w);
            const float u = eg_philox_uniform(w[0]);
            float total = 0.0f;
            for (int v = 0; v < V; ++v) total += r[v];
            const float target = u * total;
            float cum = 0.0f;
            for (int v = 0; v < V; ++v) {
                if (r[v] <= 0.0f) continue;
                bonus = v;
                cum += r[v];
                if (cum > target) break;
            }
        }
`)
	} else {
		b.WriteString(`        const int32_t* am = target_argmax + b * N;
        for (;;) {
            int next = -1;
            for (int c = cur + 1; c < N; ++c) {
                if (par[c] == cur && tok[c] == am[cur]) {
                    next = c;
                    break;
                }
            }
            if (next < 0) break;
            path[len++] = next;
            cur = next;
        }
        int32_t bonus = am[cur];
`)
	}
	b.WriteString(`        // Truncate at EOS: the sequence ends with it and nothing follows.
        bool done = false;
        for (int j = 1; j < len; ++j) {
            if (tok[path[j]] == eos_token_id) {
                len = j + 1;
                bonus = -1;
                done = true;
                break;
            }
        }
        if (bonus == eos_token_id) done = true;

        // Compact the accepted nodes' KV from slot L+path[j] to L+j. Since
        // path[j] >= j and path is increasing, a forward pass never
        // overwrites a source it has yet to read.
        const int L = seqlens[b];
        const int32_t* pt = page_table + b * MAX_PAGES;
        for (int j = 1; j < len; ++j) {
            const int src = L + path[j], dst = L + j;
            if (src == dst) continue;
            uint32_t* s = kv + ((size_t)pt[src / PAGE] * PAGE + src % PAGE) * KVW;
            uint32_t* d = kv + ((size_t)pt[dst / PAGE] * PAGE + dst % PAGE) * KVW;
            for (int w = 0; w < KVW; ++w) d[w] = s[w];
        }
        for (int j = 0; j < len; ++j) out_tokens[b * MAX_PAGES * PAGE + L + j] = tok[path[j]];
        const int newL = L + len;
        seqlens[b] = newL;
        root_tokens[b] = bonus;
        accepted[step * B + b] = len;
        keep_pages[b] = (newL + PAGE - 1) / PAGE;
        if (done || newL + N > MAX_PAGES * PAGE) active[b] = 0;
    }
    __syncwarp();
    if (b == 0) {
        // Roll back: return the pages past the committed prefix, last first.
        bool any = false;
        for (int s = 0; s < B; ++s) {
            for (int p = num_pages[s] - 1; p >= keep_pages[s]; --p) {
                free_pages[(*free_top)++] = page_table[s * MAX_PAGES + p];
                page_table[s * MAX_PAGES + p] = -1;
            }
            num_pages[s] = keep_pages[s];
            any = any || active[s] != 0;
        }
        const int next_step = step + 1;
        *step_idx = next_step;
        cudaGraphSetConditional(cond_handle, (any && next_step < MAX_STEPS) ? 1u : 0u);
    }
}
`)
	return b.String(), nil
}

// Launcher returns host functions Name_reserve_launch and Name_launch that
// launch the two kernels with their parameters followed by a cudaStream_t.
func (s TreeSpecVerify) Launcher() (string, error) {
	s, err := s.norm()
	if err != nil {
		return "", err
	}
	var b strings.Builder
	for _, k := range []struct {
		name   string
		params []string
	}{
		{s.Name + "_reserve", s.reserveParams()},
		{s.Name, s.params()},
	} {
		fmt.Fprintf(&b, "inline void %s_launch(%s, cudaStream_t stream) {\n", k.name, strings.Join(k.params, ", "))
		b.WriteString("    const dim3 grid(1, 1, 1), block(32, 1, 1);\n")
		fmt.Fprintf(&b, "    EG_LAUNCH(%s, grid, block, 0, stream, %s);\n}\n", k.name, strings.Join(paramNames(k.params), ", "))
	}
	return b.String(), nil
}

// Source returns [hopper.Prelude], the kernels and their launchers.
func (s TreeSpecVerify) Source() (string, error) {
	src, err := s.CUDA()
	if err != nil {
		return "", err
	}
	l, err := s.Launcher()
	if err != nil {
		return "", err
	}
	return hopper.Prelude + "\n" + src + "\n" + l, nil
}

// TreeSpecSeqStep is one sequence's draft tree for one step: what the draft
// model and the target forward pass hand the verifier. Tokens[0] is
// ignored; the root is the sequence's previous bonus token.
type TreeSpecSeqStep struct {
	Parent       []int
	Tokens       []int32
	TargetArgmax []int32     // greedy
	DraftProbs   [][]float32 // sampled: per node
	TargetProbs  [][]float32 // sampled: per node
}

// TreeSpecInput is the starting state and per-step trees for
// [TreeSpecVerify.Reference] and [TreeSpecVerify.RunEmulatedLoop].
type TreeSpecInput struct {
	Prefix     [][]int32           // per sequence: committed tokens; KV for them is [TreeSpecPrefixKV]
	Roots      []int32             // per sequence: first root token
	Steps      [][]TreeSpecSeqStep // [step][sequence]
	TotalPages int                 // KV pool size in pages
	EOSTokenID int32               // < 0 disables EOS
	Seed       uint64              // sampled: Philox key
}

// TreeSpecState is the device state after the loop ends.
type TreeSpecState struct {
	Tokens    [][]int32  // per sequence: committed tokens
	Roots     []int32    // per sequence: pending root token (-1 after EOS)
	Accepted  [][]int32  // [step][sequence]: tokens committed (0 if inactive)
	Steps     int        // iterations run
	PageTable [][]int32  // per sequence: pages in use
	FreePages []int32    // free stack, bottom first
	KV        [][]uint32 // per sequence: KVWords words per committed token, read through the page table
}

// TreeSpecPrefixKV is the KV word w that the tests and fixtures store for
// prefix position pos of sequence seq.
func TreeSpecPrefixKV(seq, pos, w int) uint32 {
	return treeSpecMix(0xffffffff, uint32(seq), uint32(pos), uint32(w))
}

// TreeSpecNodeKV is the KV word w that the emulated target forward pass of
// [TreeSpecVerify.RunEmulatedLoop] writes for node of sequence seq at step.
func TreeSpecNodeKV(step, seq, node, w int) uint32 {
	return treeSpecMix(uint32(step), uint32(seq), uint32(node), uint32(w))
}

func treeSpecMix(a, b, c, d uint32) uint32 {
	h := a*0x9E3779B1 ^ bits.RotateLeft32(b*0x85EBCA77, 7) ^ bits.RotateLeft32(c*0xC2B2AE3D, 13) ^ bits.RotateLeft32(d*0x27D4EB2F, 19)
	h ^= h >> 15
	h *= 0x2C1B3C6D
	return h ^ h>>12
}

const treeSpecMixDevice = `#ifndef EG_TREE_SPEC_MIX_DEFINED
#define EG_TREE_SPEC_MIX_DEFINED 1
EG_DEV uint32_t eg_tree_spec_rotl(uint32_t x, int r) { return (x << r) | (x >> (32 - r)); }
EG_DEV uint32_t eg_tree_spec_mix(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    uint32_t h = (a * 0x9E3779B1u) ^ eg_tree_spec_rotl(b * 0x85EBCA77u, 7) ^
                 eg_tree_spec_rotl(c * 0xC2B2AE3Du, 13) ^ eg_tree_spec_rotl(d * 0x27D4EB2Fu, 19);
    h ^= h >> 15;
    h *= 0x2C1B3C6Du;
    return h ^ (h >> 12);
}
#endif
`

func (s TreeSpecVerify) validateInput(in TreeSpecInput) error {
	if len(in.Prefix) != s.Batch || len(in.Roots) != s.Batch {
		return fmt.Errorf("tree spec verify %s: %d prefixes and %d roots for batch %d", s.Name, len(in.Prefix), len(in.Roots), s.Batch)
	}
	if len(in.Steps) == 0 {
		return fmt.Errorf("tree spec verify %s: no steps", s.Name)
	}
	need := 0
	for _, p := range in.Prefix {
		if len(p) > s.MaxPages*s.PageSize {
			return fmt.Errorf("tree spec verify %s: prefix of %d tokens exceeds %d", s.Name, len(p), s.MaxPages*s.PageSize)
		}
		need += (len(p) + s.PageSize - 1) / s.PageSize
	}
	if in.TotalPages < need {
		return fmt.Errorf("tree spec verify %s: %d pages cannot hold the prefixes (%d)", s.Name, in.TotalPages, need)
	}
	for si, st := range in.Steps {
		if len(st) != s.Batch {
			return fmt.Errorf("tree spec verify %s: step %d has %d sequences, want %d", s.Name, si, len(st), s.Batch)
		}
		for b, t := range st {
			tree, err := BuildDraftTreeTopology(t.Parent)
			if err != nil {
				return fmt.Errorf("tree spec verify %s: step %d sequence %d: %w", s.Name, si, b, err)
			}
			if tree.NumNodes() != s.Nodes || len(t.Tokens) != s.Nodes {
				return fmt.Errorf("tree spec verify %s: step %d sequence %d: %d nodes and %d tokens, want %d",
					s.Name, si, b, tree.NumNodes(), len(t.Tokens), s.Nodes)
			}
			if s.Sampled {
				if len(t.DraftProbs) != s.Nodes || len(t.TargetProbs) != s.Nodes {
					return fmt.Errorf("tree spec verify %s: step %d sequence %d: want %d distributions", s.Name, si, b, s.Nodes)
				}
				for i := range s.Nodes {
					if len(t.DraftProbs[i]) != s.VocabSize || len(t.TargetProbs[i]) != s.VocabSize {
						return fmt.Errorf("tree spec verify %s: step %d sequence %d node %d: want %d-entry distributions", s.Name, si, b, i, s.VocabSize)
					}
				}
			} else if len(t.TargetArgmax) != s.Nodes {
				return fmt.Errorf("tree spec verify %s: step %d sequence %d: %d target tokens, want %d", s.Name, si, b, len(t.TargetArgmax), s.Nodes)
			}
		}
	}
	return nil
}

// Reference runs the loop on the host: the same page pool, tree walk, KV
// compaction and rollback as the kernels, with the emulated target forward
// pass of [TreeSpecVerify.RunEmulatedLoop] writing [TreeSpecNodeKV].
func (s TreeSpecVerify) Reference(in TreeSpecInput) (*TreeSpecState, error) {
	s, err := s.norm()
	if err != nil {
		return nil, err
	}
	if err := s.validateInput(in); err != nil {
		return nil, err
	}
	maxSteps := min(s.MaxSteps, len(in.Steps))
	free := make([]int32, in.TotalPages)
	for i := range free {
		free[i] = int32(in.TotalPages - 1 - i)
	}
	pop := func() int32 {
		p := free[len(free)-1]
		free = free[:len(free)-1]
		return p
	}
	// kv[pos] is the KV a sequence position holds. Only its logical
	// content matters here; pages are tracked separately.
	type seqState struct {
		pages  []int32
		tokens []int32
		kv     [][]uint32
		root   int32
		active bool
	}
	seqs := make([]*seqState, s.Batch)
	for b := range s.Batch {
		st := &seqState{tokens: slices.Clone(in.Prefix[b]), root: in.Roots[b], active: true}
		for range (len(in.Prefix[b]) + s.PageSize - 1) / s.PageSize {
			st.pages = append(st.pages, pop())
		}
		for pos := range in.Prefix[b] {
			st.kv = append(st.kv, treeSpecKVRow(s.KVWords, func(w int) uint32 { return TreeSpecPrefixKV(b, pos, w) }))
		}
		seqs[b] = st
	}
	out := &TreeSpecState{}
	for step := 0; step < maxSteps; step++ {
		// Reserve.
		for _, st := range seqs {
			if !st.active {
				continue
			}
			need := (len(st.tokens) + s.Nodes + s.PageSize - 1) / s.PageSize
			if need > s.MaxPages {
				st.active = false
				continue
			}
			for len(st.pages) < need && len(free) > 0 {
				st.pages = append(st.pages, pop())
			}
			if len(st.pages) < need {
				st.active = false
			}
		}
		acc := make([]int32, s.Batch)
		any := false
		for b, st := range seqs {
			if !st.active {
				continue
			}
			t := in.Steps[step][b]
			tree, _ := BuildDraftTreeTopology(t.Parent)
			tokens := slices.Clone(t.Tokens)
			tokens[0] = st.root
			var a TreeAccept
			if s.Sampled {
				a, err = AcceptDraftTreeSampled(tree, tokens, t.DraftProbs, t.TargetProbs, in.Seed, uint32(step), uint32(b))
			} else {
				a, err = AcceptDraftTreeGreedy(tree, tokens, t.TargetArgmax)
			}
			if err != nil {
				return nil, fmt.Errorf("tree spec verify %s: step %d sequence %d: %w", s.Name, step, b, err)
			}
			path, bonus := a.Path, a.Bonus
			done := false
			for j := 1; j < len(path); j++ {
				if tokens[path[j]] == in.EOSTokenID {
					path, bonus, done = path[:j+1], -1, true
					break
				}
			}
			if bonus == in.EOSTokenID {
				done = true
			}
			for _, node := range path {
				st.tokens = append(st.tokens, tokens[node])
				st.kv = append(st.kv, treeSpecKVRow(s.KVWords, func(w int) uint32 { return TreeSpecNodeKV(step, b, node, w) }))
			}
			st.root = bonus
			acc[b] = int32(len(path))
			if done || len(st.tokens)+s.Nodes > s.MaxPages*s.PageSize {
				st.active = false
			}
		}
		// Roll back pages past each committed prefix, in sequence order.
		for _, st := range seqs {
			keep := (len(st.tokens) + s.PageSize - 1) / s.PageSize
			for p := len(st.pages) - 1; p >= keep; p-- {
				free = append(free, st.pages[p])
			}
			st.pages = st.pages[:min(keep, len(st.pages))]
			any = any || st.active
		}
		out.Accepted = append(out.Accepted, acc)
		out.Steps++
		if !any {
			break
		}
	}
	for _, st := range seqs {
		out.Tokens = append(out.Tokens, st.tokens)
		out.Roots = append(out.Roots, st.root)
		out.PageTable = append(out.PageTable, st.pages)
		out.KV = append(out.KV, slices.Concat(st.kv...))
	}
	out.FreePages = free
	return out, nil
}

func treeSpecKVRow(n int, f func(int) uint32) []uint32 {
	row := make([]uint32, n)
	for w := range row {
		row[w] = f(w)
	}
	return row
}

// RunEmulatedLoop runs the loop on the host emulator through
// [cudahostemu.RunConditionalWhile]. Each iteration launches Name_reserve, a
// fixture kernel standing in for the draft and target forward passes (it
// copies the step's tree, sets node 0 to the root and writes
// [TreeSpecNodeKV] at positions seqlen..seqlen+Nodes-1), and Name.
func (s TreeSpecVerify) RunEmulatedLoop(ctx context.Context, in TreeSpecInput) (*TreeSpecState, error) {
	s, err := s.norm()
	if err != nil {
		return nil, err
	}
	s.MaxSteps = min(s.MaxSteps, len(in.Steps))
	src, err := s.Source()
	if err != nil {
		return nil, err
	}
	return s.runEmulatedLoop(ctx, in, src)
}

func (s TreeSpecVerify) runEmulatedLoop(ctx context.Context, in TreeSpecInput, src string) (*TreeSpecState, error) {
	if err := s.validateInput(in); err != nil {
		return nil, err
	}
	maxSteps := min(s.MaxSteps, len(in.Steps))
	B, P, MP := s.Batch, s.PageSize, s.MaxPages
	capTokens := MP * P
	bufs := s.loopBuffers(in, maxSteps)
	stage := s.Name + "_stage"
	var full strings.Builder
	full.WriteString(src)
	full.WriteString(s.stageCUDA(stage))

	ctypes := map[string]string{
		"all_dprobs": "float", "all_tprobs": "float", "draft_probs": "float",
		"target_probs": "float", "resid": "float", "kv": "uint32_t",
	}
	var args []cudahostemu.Arg
	for _, name := range slices.Sorted(maps.Keys(bufs)) {
		ct := ctypes[name]
		if ct == "" {
			ct = "int32_t"
		}
		args = append(args, cudahostemu.BufferArg(name, ct, bufs[name]))
	}
	args = append(args,
		cudahostemu.Arg{Name: "eos_token_id", Expr: fmt.Sprintf("%d", in.EOSTokenID)},
		cudahostemu.Arg{Name: "seed", Expr: fmt.Sprintf("%dULL", in.Seed)},
		cudahostemu.CondHandleArg("cond_handle", 1),
	)
	block := cudahostemu.Dim3{X: 32}
	body := []cudahostemu.CondGraphStep{
		{Kernel: s.Name + "_reserve", Grid: cudahostemu.Dim3{X: 1}, Block: block},
		{Kernel: stage, Grid: cudahostemu.Dim3{X: 1}, Block: block},
		{Kernel: s.Name, Grid: cudahostemu.Dim3{X: 1}, Block: block},
	}
	res, err := cudahostemu.RunConditionalWhile(ctx, full.String(), cudahostemu.CondWhileConfig{
		Handle:                1,
		InitialCondition:      1,
		MaxIterations:         maxSteps + 2,
		RequireExplicitUpdate: true,
		DetectDataRace:        true,
	}, body, args...)
	if err != nil {
		return nil, err
	}

	steps := int(specBytesI32(bufs["step_idx"])[0])
	if steps != res.IterationsExecuted || steps > maxSteps {
		return nil, fmt.Errorf("tree spec verify %s: step_idx %d after %d iterations", s.Name, steps, res.IterationsExecuted)
	}
	outTokens := specBytesI32(bufs["out_tokens"])
	seqlens := specBytesI32(bufs["seqlens"])
	pageTable := specBytesI32(bufs["page_table"])
	numPages := specBytesI32(bufs["num_pages"])
	kv := le32u(bufs["kv"])
	top := int(specBytesI32(bufs["free_top"])[0])
	accepted := specBytesI32(bufs["accepted"])
	out := &TreeSpecState{
		Roots:     specBytesI32(bufs["root_tokens"]),
		Steps:     steps,
		FreePages: specBytesI32(bufs["free_pages"])[:top],
	}
	for step := range steps {
		out.Accepted = append(out.Accepted, accepted[step*B:(step+1)*B])
	}
	for b := range B {
		L := int(seqlens[b])
		if L > capTokens || int(numPages[b]) > MP {
			return nil, fmt.Errorf("tree spec verify %s: sequence %d has length %d and %d pages", s.Name, b, L, numPages[b])
		}
		pages := pageTable[b*MP : b*MP+int(numPages[b])]
		out.Tokens = append(out.Tokens, outTokens[b*capTokens:b*capTokens+L])
		out.PageTable = append(out.PageTable, pages)
		var rows []uint32
		for pos := range L {
			page := int(pages[pos/P])
			if page < 0 || page >= in.TotalPages {
				return nil, fmt.Errorf("tree spec verify %s: sequence %d position %d maps to page %d", s.Name, b, pos, page)
			}
			slot := page*P + pos%P
			rows = append(rows, kv[slot*s.KVWords:(slot+1)*s.KVWords]...)
		}
		out.KV = append(out.KV, rows)
	}
	return out, nil
}

// loopBuffers returns the initial device buffers of a loop over
// in.Steps[:maxSteps]: the per-step fixtures, the page pool with the
// prefixes' pages popped and their KV written, and zeroed outputs.
func (s TreeSpecVerify) loopBuffers(in TreeSpecInput, maxSteps int) map[string][]byte {
	N, B, P, MP := s.Nodes, s.Batch, s.PageSize, s.MaxPages
	capTokens := MP * P

	var parents, tokens, argmax []int32
	var dprobs, tprobs []float32
	for _, st := range in.Steps[:maxSteps] {
		for _, t := range st {
			for _, p := range t.Parent {
				parents = append(parents, int32(p))
			}
			tokens = append(tokens, t.Tokens...)
			if s.Sampled {
				for i := range N {
					dprobs = append(dprobs, t.DraftProbs[i]...)
					tprobs = append(tprobs, t.TargetProbs[i]...)
				}
			} else {
				argmax = append(argmax, t.TargetArgmax...)
			}
		}
	}

	// Initial pool and prefix KV, as Reference sets them up.
	free := make([]int32, in.TotalPages)
	for i := range free {
		free[i] = int32(in.TotalPages - 1 - i)
	}
	top := in.TotalPages
	pageTable := slices.Repeat([]int32{-1}, B*MP)
	numPages := make([]int32, B)
	seqlens := make([]int32, B)
	outTokens := make([]int32, B*capTokens)
	kv := make([]uint32, in.TotalPages*P*s.KVWords)
	for b, pre := range in.Prefix {
		for range (len(pre) + P - 1) / P {
			top--
			pageTable[b*MP+int(numPages[b])] = free[top]
			numPages[b]++
		}
		for pos, t := range pre {
			outTokens[b*capTokens+pos] = t
			slot := int(pageTable[b*MP+pos/P])*P + pos%P
			for w := range s.KVWords {
				kv[slot*s.KVWords+w] = TreeSpecPrefixKV(b, pos, w)
			}
		}
		seqlens[b] = int32(len(pre))
	}

	v := max(s.VocabSize, 1)
	nonEmpty := func(x []float32, n int) []float32 {
		if len(x) == 0 {
			return make([]float32, n)
		}
		return x
	}
	nonEmptyI := func(x []int32, n int) []int32 {
		if len(x) == 0 {
			return make([]int32, n)
		}
		return x
	}
	return map[string][]byte{
		"all_parent":    specI32Bytes(parents),
		"all_tokens":    specI32Bytes(tokens),
		"all_argmax":    specI32Bytes(nonEmptyI(argmax, 1)),
		"all_dprobs":    f32le(nonEmpty(dprobs, 1)),
		"all_tprobs":    f32le(nonEmpty(tprobs, 1)),
		"parent":        specI32Bytes(make([]int32, B*N)),
		"tree_tokens":   specI32Bytes(make([]int32, B*N)),
		"target_argmax": specI32Bytes(make([]int32, B*N)),
		"draft_probs":   f32le(make([]float32, B*N*v)),
		"target_probs":  f32le(make([]float32, B*N*v)),
		"resid":         f32le(make([]float32, B*v)),
		"kv":            u32le(kv),
		"page_table":    specI32Bytes(pageTable),
		"num_pages":     specI32Bytes(numPages),
		"free_pages":    specI32Bytes(free),
		"free_top":      specI32Bytes([]int32{int32(top)}),
		"seqlens":       specI32Bytes(seqlens),
		"active":        specI32Bytes(slices.Repeat([]int32{1}, B)),
		"out_tokens":    specI32Bytes(outTokens),
		"root_tokens":   specI32Bytes(in.Roots),
		"accepted":      specI32Bytes(make([]int32, maxSteps*B)),
		"step_idx":      specI32Bytes([]int32{0}),
	}
}

// stageCUDA returns the fixture kernel that stands in for the draft and
// target forward passes in emulated and native loop tests.
func (s TreeSpecVerify) stageCUDA(name string) string {
	return treeSpecMixDevice + fmt.Sprintf(`
__global__ void __launch_bounds__(32) %s(
    const int32_t* __restrict__ all_parent,
    const int32_t* __restrict__ all_tokens,
    const int32_t* __restrict__ all_argmax,
    const float* __restrict__ all_dprobs,
    const float* __restrict__ all_tprobs,
    const int32_t* __restrict__ step_idx,
    const int32_t* __restrict__ root_tokens,
    const int32_t* __restrict__ seqlens,
    const int32_t* __restrict__ page_table,
    const int32_t* __restrict__ active,
    int32_t* __restrict__ parent,
    int32_t* __restrict__ tree_tokens,
    int32_t* __restrict__ target_argmax,
    float* __restrict__ draft_probs,
    float* __restrict__ target_probs,
    uint32_t* __restrict__ kv) {
    constexpr int N = %d, B = %d, KVW = %d, PAGE = %d, MAX_PAGES = %d, SAMPLED = %d, V = %d;
    const int b = threadIdx.x, s = *step_idx;
    if (b >= B) return;
    const size_t off = (size_t)(s * B + b) * N;
    for (int i = 0; i < N; ++i) {
        parent[b * N + i] = all_parent[off + i];
        tree_tokens[b * N + i] = i == 0 ? root_tokens[b] : all_tokens[off + i];
        if (!SAMPLED) target_argmax[b * N + i] = all_argmax[off + i];
        for (int v = 0; SAMPLED && v < V; ++v) {
            draft_probs[(size_t)(b * N + i) * V + v] = all_dprobs[(off + i) * V + v];
            target_probs[(size_t)(b * N + i) * V + v] = all_tprobs[(off + i) * V + v];
        }
    }
    if (!active[b]) return;
    const int L = seqlens[b];
    for (int i = 0; i < N; ++i) {
        const int pos = L + i;
        const size_t slot = (size_t)page_table[b * MAX_PAGES + pos / PAGE] * PAGE + pos %% PAGE;
        for (int w = 0; w < KVW; ++w) kv[slot * KVW + w] = eg_tree_spec_mix((uint32_t)s, (uint32_t)b, (uint32_t)i, (uint32_t)w);
    }
}
`, name, s.Nodes, s.Batch, s.KVWords, s.PageSize, s.MaxPages, boolInt(s.Sampled), max(s.VocabSize, 1))
}

func boolInt(b bool) int {
	if b {
		return 1
	}
	return 0
}

func f32le(v []float32) []byte {
	b := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(x))
	}
	return b
}

func u32le(v []uint32) []byte {
	b := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[4*i:], x)
	}
	return b
}

func le32u(b []byte) []uint32 {
	v := make([]uint32, len(b)/4)
	for i := range v {
		v[i] = binary.LittleEndian.Uint32(b[4*i:])
	}
	return v
}
