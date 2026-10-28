package hopper

import (
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudahostemu"
)

func TestPhilox4x32(t *testing.T) {
	// Known-answer vectors from Random123 (kat_vectors, philox4x32_10).
	tests := []struct {
		ctr  [4]uint32
		key  [2]uint32
		want [4]uint32
	}{
		{[4]uint32{}, [2]uint32{}, [4]uint32{0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8}},
		{
			[4]uint32{0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff},
			[2]uint32{0xffffffff, 0xffffffff},
			[4]uint32{0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd},
		},
		{
			[4]uint32{0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344},
			[2]uint32{0xa4093822, 0x299f31d0},
			[4]uint32{0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1},
		},
	}
	for _, tt := range tests {
		if got := Philox4x32(tt.ctr, tt.key); got != tt.want {
			t.Errorf("Philox4x32(%#x, %#x) = %#x, want %#x", tt.ctr, tt.key, got, tt.want)
		}
	}
}

func TestPhiloxDeviceMatchesGo(t *testing.T) {
	const n = 64
	kernel := philoxDevice + `
__global__ void philox_kat(uint32_t* out, float* u, float* ou) {
    const uint32_t i = threadIdx.x;
    uint32_t w[4];
    eg_philox4x32(i, i * 7u, ~i, 3u, 0xa4093822u, 0x299f31d0u, w);
    for (int k = 0; k < 4; ++k) out[4 * i + k] = w[k];
    u[i] = eg_philox_uniform(w[0]);
    ou[i] = eg_philox_open_uniform(w[1]);
}
`
	res := runEmu(t, kernel, fmt.Sprintf("eg_emu_launch(dim3(1), dim3(%d), 0, philox_kat, out, u, ou)", n),
		emuBuf{"out", "uint32_t", poison(16 * n)},
		emuBuf{"u", "float", poison(4 * n)},
		emuBuf{"ou", "float", poison(4 * n)},
	)
	out, u, ou := le32u(res["out"]), bytesF32(res["u"]), bytesF32(res["ou"])
	for i := range uint32(n) {
		want := Philox4x32([4]uint32{i, i * 7, ^i, 3}, [2]uint32{0xa4093822, 0x299f31d0})
		if got := [4]uint32(out[4*i : 4*i+4]); got != want {
			t.Fatalf("thread %d: device philox %#x, want %#x", i, got, want)
		}
		if u[i] != philoxUniform(want[0]) || ou[i] != philoxOpenUniform(want[1]) {
			t.Fatalf("thread %d: uniforms %v %v, want %v %v", i, u[i], ou[i], philoxUniform(want[0]), philoxOpenUniform(want[1]))
		}
	}
}

func TestBuildDraftTreeTopology(t *testing.T) {
	tests := []struct {
		name      string
		parent    []int
		wantMask  []uint64
		wantDepth []int
		wantErr   string
	}{
		{"root", []int{-1}, []uint64{1}, []int{0}, ""},
		{"chain", []int{-1, 0, 1}, []uint64{1, 3, 7}, []int{0, 1, 2}, ""},
		{"branch", []int{-1, 0, 0, 1, 2}, []uint64{1, 3, 5, 0b1011, 0b10101}, []int{0, 1, 1, 2, 2}, ""},
		{"empty", nil, nil, nil, "outside 1..64"},
		{"too big", append([]int{-1}, make([]int, 64)...), nil, nil, "outside 1..64"},
		{"rootless", []int{0}, nil, nil, "want -1"},
		{"forward parent", []int{-1, 2, 0}, nil, nil, "outside -1..0"},
		{"self parent", []int{-1, 1}, nil, nil, "outside -1..0"},
		{"forest", []int{-1, 0, -1}, nil, nil, "only node 0"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := BuildDraftTreeTopology(tt.parent)
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("BuildDraftTreeTopology(%v) error = %v, want %q", tt.parent, err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(got.AncestorMask, tt.wantMask) || !slices.Equal(got.Depth, tt.wantDepth) {
				t.Fatalf("BuildDraftTreeTopology(%v) = masks %b depths %v, want %b %v", tt.parent, got.AncestorMask, got.Depth, tt.wantMask, tt.wantDepth)
			}
		})
	}
}

func TestAcceptDraftTreeGreedy(t *testing.T) {
	// Node:    0  1  2  3  4  5  6
	// Parent: -1  0  0  1  2  2  4
	parent := []int{-1, 0, 0, 1, 2, 2, 4}
	tokens := []int32{100, 10, 20, 11, 21, 22, 23}
	tests := []struct {
		name      string
		argmax    []int32
		wantPath  []int
		wantBonus int32
	}{
		{"reject all", []int32{99, 0, 0, 0, 0, 0, 0}, []int{0}, 99},
		{"first branch", []int32{10, 11, 0, 50, 0, 0, 0}, []int{0, 1, 3}, 50},
		{"second branch deep", []int32{20, 0, 21, 0, 23, 0, 60}, []int{0, 2, 4, 6}, 60},
		{"sibling of match", []int32{20, 0, 22, 0, 0, 70, 0}, []int{0, 2, 5}, 70},
		{"mismatch at depth 2", []int32{20, 0, 24, 0, 0, 0, 0}, []int{0, 2}, 24},
	}
	tree, err := BuildDraftTreeTopology(parent)
	if err != nil {
		t.Fatal(err)
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := AcceptDraftTreeGreedy(tree, tokens, tt.argmax)
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(got.Path, tt.wantPath) || got.Bonus != tt.wantBonus {
				t.Fatalf("got path %v bonus %d, want %v %d", got.Path, got.Bonus, tt.wantPath, tt.wantBonus)
			}
		})
	}
}

// treeToyModel is a one-layer attention language model over a small
// vocabulary: query, key and value of a token are its embedding, and the
// next token is the argmax of out·Wᵀ. It has no positional signal, so a
// node's prediction depends only on which tokens it may attend to: exactly
// what a tree mask must get right.
type treeToyModel struct {
	vocab, dim int
	emb        []uint16  // [vocab][dim] BF16
	head       []float64 // [vocab][dim]
}

func newTreeToyModel(r *rand.Rand, vocab, dim int) *treeToyModel {
	m := &treeToyModel{vocab: vocab, dim: dim, emb: make([]uint16, vocab*dim), head: make([]float64, vocab*dim)}
	for i := range m.emb {
		m.emb[i] = BF16Bits(float32(r.NormFloat64()))
		m.head[i] = r.NormFloat64()
	}
	return m
}

// forward returns the argmax next token for each of the last q positions of
// seq, which attend under mask (nil: causal).
func (m *treeToyModel) forward(t *testing.T, seq []int32, q int, mask []uint64) []int32 {
	t.Helper()
	const page = 4
	pages := (len(seq) + page - 1) / page
	// A small softmax scale keeps attention close to an average of the
	// visible tokens, so the context, not the query token, decides.
	a := MTPPagedAttention{NumHeads: 1, KVHeads: 1, HeadDim: m.dim, PageSize: page, MaxPages: pages, QTokensPerSeq: q, Scale: 0.05}
	kv := make([]uint16, pages*page*m.dim)
	pt := make([]int32, pages)
	for i := range pt {
		pt[i] = int32(pages - 1 - i) // scrambled page order
	}
	for pos, tok := range seq {
		slot := int(pt[pos/page])*page + pos%page
		copy(kv[slot*m.dim:], m.emb[int(tok)*m.dim:int(tok+1)*m.dim])
	}
	var qv []uint16
	for _, tok := range seq[len(seq)-q:] {
		qv = append(qv, m.emb[int(tok)*m.dim:int(tok+1)*m.dim]...)
	}
	out, err := a.Reference(qv, MTPPagedCache{KBF16: kv, VBF16: kv, PageTable: pt, SeqLens: []int32{int32(len(seq))}, DraftMask: mask})
	if err != nil {
		t.Fatal(err)
	}
	next := make([]int32, q)
	for i := range q {
		o := out[i*m.dim : (i+1)*m.dim]
		best := math.Inf(-1)
		for v := range m.vocab {
			var s float64
			for d, x := range o {
				s += x * m.head[v*m.dim+d]
			}
			if s > best {
				best, next[i] = s, int32(v)
			}
		}
	}
	return next
}

func (m *treeToyModel) greedy(t *testing.T, seq []int32, n int) []int32 {
	seq = slices.Clone(seq)
	var out []int32
	for range n {
		next := m.forward(t, seq, 1, nil)[0]
		out = append(out, next)
		seq = append(seq, next)
	}
	return out
}

// TestTreeVerifyMatchesGreedyDecoding checks the tree mask and the greedy
// walk together: verifying a draft tree in one masked forward pass accepts
// exactly the longest root path that token-by-token greedy decoding of the
// same model produces, and its bonus token is greedy decoding's next token.
// Flipping one non-ancestor bit of the mask (a sibling leak) must break it.
func TestTreeVerifyMatchesGreedyDecoding(t *testing.T) {
	r := rand.New(rand.NewPCG(3, 1))
	m := newTreeToyModel(r, 12, 16)
	type trial struct {
		prefix []int32
		tree   DraftTreeTopology
		tokens []int32
	}
	var trials []trial
	for range 40 {
		prefix := make([]int32, 3+r.IntN(4))
		for i := range prefix {
			prefix[i] = int32(r.IntN(m.vocab))
		}
		n := 4 + r.IntN(9)
		parent := []int{-1}
		for i := 1; i < n; i++ {
			parent = append(parent, r.IntN(i))
		}
		tree, err := BuildDraftTreeTopology(parent)
		if err != nil {
			t.Fatal(err)
		}
		tokens := make([]int32, n)
		tokens[0] = int32(r.IntN(m.vocab))
		for i := 1; i < n; i++ {
			// Propose the greedy continuation of the node's path half the
			// time, keeping sibling tokens distinct.
			path := append(slices.Clone(prefix), treePathTokens(tree, tokens, parent[i])...)
			want := m.greedy(t, path, 1)[0]
			used := map[int32]bool{}
			for j := 1; j < i; j++ {
				if parent[j] == parent[i] {
					used[tokens[j]] = true
				}
			}
			tokens[i] = want
			if used[want] || r.IntN(2) == 0 {
				for tokens[i] = int32(r.IntN(m.vocab)); used[tokens[i]]; tokens[i] = int32(r.IntN(m.vocab)) {
				}
			}
		}
		trials = append(trials, trial{prefix, tree, tokens})
	}

	verify := func(tr trial, mask []uint64) TreeAccept {
		seq := append(slices.Clone(tr.prefix), tr.tokens...)
		argmax := m.forward(t, seq, tr.tree.NumNodes(), mask)
		a, err := AcceptDraftTreeGreedy(tr.tree, tr.tokens, argmax)
		if err != nil {
			t.Fatal(err)
		}
		return a
	}
	accepted := 0
	for i, tr := range trials {
		a := verify(tr, tr.tree.AncestorMask)
		// The deepest node whose root path greedy decoding reproduces.
		ctx := append(slices.Clone(tr.prefix), tr.tokens[0])
		greedy := m.greedy(t, ctx, tr.tree.NumNodes())
		best := 0
		for node := range tr.tree.NumNodes() {
			path := treePathTokens(tr.tree, tr.tokens, node)[1:]
			if slices.Equal(path, greedy[:len(path)]) && tr.tree.Depth[node] > tr.tree.Depth[best] {
				best = node
			}
		}
		depth := len(a.Path) - 1
		if a.Path[depth] != best {
			t.Errorf("trial %d: accepted path %v ends at %d, longest greedy path ends at %d", i, a.Path, a.Path[depth], best)
		}
		if a.Bonus != greedy[depth] {
			t.Errorf("trial %d: bonus %d, greedy decoding emits %d", i, a.Bonus, greedy[depth])
		}
		accepted += depth
	}
	if accepted == 0 {
		t.Fatal("no trial accepted a draft token; the test exercises nothing")
	}

	t.Run("sibling leak mutant", func(t *testing.T) {
		// Let the last accepted node attend to one node that is not its
		// ancestor.
		caught := 0
		for _, tr := range trials {
			a := verify(tr, tr.tree.AncestorMask)
			end := a.Path[len(a.Path)-1]
			mask := slices.Clone(tr.tree.AncestorMask)
			for j := range tr.tree.NumNodes() {
				if mask[end]&(1<<j) == 0 {
					mask[end] |= 1 << j
					break
				}
			}
			if b := verify(tr, mask); !slices.Equal(a.Path, b.Path) || a.Bonus != b.Bonus {
				caught++
			}
		}
		t.Logf("%d of %d trials changed", caught, len(trials))
		if caught == 0 {
			t.Fatal("a sibling-leaking mask changed no verification outcome")
		}
	})
}

func treePathTokens(tree DraftTreeTopology, tokens []int32, node int) []int32 {
	var path []int32
	for ; node >= 0; node = tree.ParentIdx[node] {
		path = append(path, tokens[node])
	}
	slices.Reverse(path)
	return path
}

// TestAcceptDraftTreeSampledDistribution checks that speculative tree
// sampling is lossless: the first emitted token follows the target
// distribution at the root, though the draft distribution differs from it
// and each root child is an independent draft sample.
func TestAcceptDraftTreeSampledDistribution(t *testing.T) {
	const (
		vocab  = 6
		trials = 100000
	)
	p := []float32{0.05, 0.40, 0.10, 0.25, 0.15, 0.05}
	q := []float32{0.30, 0.05, 0.30, 0.05, 0.10, 0.20}
	pChild := []float32{0.2, 0.2, 0.2, 0.2, 0.1, 0.1}
	// Root with three draft children, one of which has a child.
	tree, err := BuildDraftTreeTopology([]int{-1, 0, 0, 0, 1})
	if err != nil {
		t.Fatal(err)
	}
	draft := [][]float32{q, q, q, q, q}
	target := [][]float32{p, pChild, pChild, pChild, pChild}
	r := rand.New(rand.NewPCG(1, 2))
	sample := func(dist []float32) int32 {
		u := r.Float32()
		for i, x := range dist {
			if u < x {
				return int32(i)
			}
			u -= x
		}
		return int32(len(dist) - 1)
	}
	var hist [vocab]float64
	accepted := 0
	for i := range trials {
		tokens := []int32{0, sample(q), sample(q), sample(q), sample(q)}
		a, err := AcceptDraftTreeSampled(tree, tokens, draft, target, 7, uint32(i), 0)
		if err != nil {
			t.Fatal(err)
		}
		first := a.Bonus
		if len(a.Path) > 1 {
			first = tokens[a.Path[1]]
			accepted++
		}
		hist[first]++
	}
	var tvd float64
	for v := range vocab {
		tvd += math.Abs(hist[v]/trials - float64(p[v]))
	}
	tvd /= 2
	t.Logf("TVD %.4f over %d trials, %d with an accepted draft token", tvd, trials, accepted)
	if tvd > 5e-3 {
		t.Fatalf("first-token TVD from target = %.4f, want < 5e-3; histogram %v", tvd, hist)
	}
	if accepted == 0 || accepted == trials {
		t.Fatalf("%d of %d trials accepted a draft token; both outcomes must occur", accepted, trials)
	}
}

func TestTreeSpecVerifyErrors(t *testing.T) {
	tests := []struct {
		name string
		s    TreeSpecVerify
		want string
	}{
		{"bad name", TreeSpecVerify{Name: "a-b", Nodes: 4, Batch: 1, KVWords: 1, PageSize: 4, MaxPages: 4}, "bad name"},
		{"nodes", TreeSpecVerify{Nodes: 65, Batch: 1, KVWords: 1, PageSize: 4, MaxPages: 64}, "Nodes"},
		{"batch", TreeSpecVerify{Nodes: 4, Batch: 33, KVWords: 1, PageSize: 4, MaxPages: 4}, "Batch"},
		{"capacity", TreeSpecVerify{Nodes: 8, Batch: 1, KVWords: 1, PageSize: 4, MaxPages: 2}, "cannot hold"},
		{"vocab", TreeSpecVerify{Nodes: 4, Batch: 1, KVWords: 1, PageSize: 4, MaxPages: 4, Sampled: true}, "VocabSize"},
	}
	for _, tt := range tests {
		if _, err := tt.s.CUDA(); err == nil || !strings.Contains(err.Error(), tt.want) {
			t.Errorf("%s: CUDA() error = %v, want %q", tt.name, err, tt.want)
		}
	}
}

// treeSpecGreedyFixture builds steps of random draft trees with distinct
// sibling tokens whose target argmax follows a random child 70% of the
// time and emits EOS rarely.
func treeSpecGreedyFixture(r *rand.Rand, s TreeSpecVerify, steps int, eos int32) [][]TreeSpecSeqStep {
	var out [][]TreeSpecSeqStep
	for range steps {
		var st []TreeSpecSeqStep
		for range s.Batch {
			parent := []int{-1}
			for i := 1; i < s.Nodes; i++ {
				parent = append(parent, r.IntN(i))
			}
			tokens := make([]int32, s.Nodes)
			argmax := make([]int32, s.Nodes)
			kids := make([]int32, s.Nodes)
			for i := 1; i < s.Nodes; i++ {
				// Distinct among siblings; cousins share tokens.
				tokens[i] = 10 + kids[parent[i]]
				kids[parent[i]]++
			}
			for i := range s.Nodes {
				var kids []int
				for c := i + 1; c < s.Nodes; c++ {
					if parent[c] == i {
						kids = append(kids, c)
					}
				}
				switch {
				case r.IntN(40) == 0:
					argmax[i] = eos
				case len(kids) > 0 && r.IntN(10) < 7:
					argmax[i] = tokens[kids[r.IntN(len(kids))]]
				default:
					argmax[i] = int32(5000 + r.IntN(1000))
				}
			}
			st = append(st, TreeSpecSeqStep{Parent: parent, Tokens: tokens, TargetArgmax: argmax})
		}
		out = append(out, st)
	}
	return out
}

func treeSpecStateDiff(got, want *TreeSpecState) string {
	var d []string
	if got.Steps != want.Steps {
		d = append(d, fmt.Sprintf("steps %d, want %d", got.Steps, want.Steps))
	}
	if !slices.Equal(got.Roots, want.Roots) {
		d = append(d, fmt.Sprintf("roots %v, want %v", got.Roots, want.Roots))
	}
	if !slices.Equal(got.FreePages, want.FreePages) {
		d = append(d, fmt.Sprintf("free pages %v, want %v", got.FreePages, want.FreePages))
	}
	for i := range min(len(got.Accepted), len(want.Accepted)) {
		if !slices.Equal(got.Accepted[i], want.Accepted[i]) {
			d = append(d, fmt.Sprintf("step %d accepted %v, want %v", i, got.Accepted[i], want.Accepted[i]))
			break
		}
	}
	for b := range min(len(got.Tokens), len(want.Tokens)) {
		if !slices.Equal(got.Tokens[b], want.Tokens[b]) {
			d = append(d, fmt.Sprintf("sequence %d tokens %v, want %v", b, got.Tokens[b], want.Tokens[b]))
		}
		if !slices.Equal(got.PageTable[b], want.PageTable[b]) {
			d = append(d, fmt.Sprintf("sequence %d pages %v, want %v", b, got.PageTable[b], want.PageTable[b]))
		}
		if !slices.Equal(got.KV[b], want.KV[b]) {
			d = append(d, fmt.Sprintf("sequence %d KV differs", b))
		}
	}
	return strings.Join(d, "\n")
}

// checkPagesConserved checks that every pool page is owned by exactly one
// sequence or the free stack.
func checkPagesConserved(t *testing.T, st *TreeSpecState, total int) {
	t.Helper()
	seen := make([]int, total)
	for _, pt := range st.PageTable {
		for _, p := range pt {
			seen[p]++
		}
	}
	for _, p := range st.FreePages {
		seen[p]++
	}
	for p, n := range seen {
		if n != 1 {
			t.Fatalf("page %d owned %d times", p, n)
		}
	}
}

func TestTreeSpecVerifyEmulatedLoop(t *testing.T) {
	if _, err := cudahostemu.Compiler(); errors.Is(err, cudahostemu.ErrNoCompiler) {
		t.Skip(err)
	}
	const steps = 32
	s := TreeSpecVerify{Name: "tree_verify", Nodes: 8, Batch: 3, KVWords: 3, PageSize: 4, MaxPages: 24, MaxSteps: steps}
	r := rand.New(rand.NewPCG(5, 8))
	in := TreeSpecInput{
		Prefix:     [][]int32{{1, 2, 3}, {4, 5, 6, 7, 8}, {9}},
		Roots:      []int32{10, 11, 12},
		Steps:      treeSpecGreedyFixture(r, s, steps, 999),
		TotalPages: 48,
		EOSTokenID: 999,
	}
	want, err := s.Reference(in)
	if err != nil {
		t.Fatal(err)
	}
	checkPagesConserved(t, want, in.TotalPages)
	var compacted, deactivated bool
	for _, acc := range want.Accepted {
		for _, a := range acc {
			compacted = compacted || a > 1
			deactivated = deactivated || a == 0
		}
	}
	if !compacted || !deactivated || want.Steps < 4 {
		t.Fatalf("fixture too weak: compacted=%v deactivated=%v steps=%d", compacted, deactivated, want.Steps)
	}
	got, err := s.RunEmulatedLoop(t.Context(), in)
	if err != nil {
		t.Fatal(err)
	}
	if d := treeSpecStateDiff(got, want); d != "" {
		t.Fatalf("emulated loop differs from Reference:\n%s", d)
	}
	checkPagesConserved(t, got, in.TotalPages)

	src, err := s.Source()
	if err != nil {
		t.Fatal(err)
	}
	mutants := []struct{ name, old, new string }{
		{"reverse compaction", "for (int j = 1; j < len; ++j) {\n            const int src", "for (int j = len - 1; j >= 1; --j) {\n            const int src"},
		{"no rollback", "free_pages[(*free_top)++] = page_table[s * MAX_PAGES + p];", "(void)0;"},
		{"keep sibling", "if (par[c] == cur && tok[c] == am[cur])", "if (par[c] <= cur && tok[c] == am[cur])"},
		{"no EOS stop", "if (bonus == eos_token_id) done = true;", ""},
	}
	for _, m := range mutants {
		t.Run(m.name, func(t *testing.T) {
			if !strings.Contains(src, m.old) {
				t.Fatalf("mutant site %q not in source", m.old)
			}
			got, err := s.runEmulatedLoop(t.Context(), in, strings.Replace(src, m.old, m.new, 1))
			if err == nil && treeSpecStateDiff(got, want) == "" {
				t.Fatal("mutant survived")
			}
		})
	}
}

// TestTreeSpecVerifyCompactionOracle accepts path 0, 2, 5, 11 of a 16-node
// tree and checks the KV slots L..L+3 against nodes 0, 2, 5 and 11 and that
// the pages past L+4 return to the pool.
func TestTreeSpecVerifyCompactionOracle(t *testing.T) {
	if _, err := cudahostemu.Compiler(); errors.Is(err, cudahostemu.ErrNoCompiler) {
		t.Skip(err)
	}
	s := TreeSpecVerify{Name: "tree_oracle", Nodes: 16, Batch: 1, KVWords: 2, PageSize: 4, MaxPages: 8, MaxSteps: 1}
	parent := []int{-1, 0, 0, 1, 2, 2, 3, 4, 4, 5, 5, 5, 6, 7, 9, 10}
	tokens := make([]int32, 16)
	argmax := make([]int32, 16)
	for i := range tokens {
		tokens[i] = int32(20 + i)
		argmax[i] = 77
	}
	argmax[0], argmax[2], argmax[5], argmax[11] = 22, 25, 31, 88
	const L = 6
	in := TreeSpecInput{
		Prefix:     [][]int32{{1, 2, 3, 4, 5, 6}},
		Roots:      []int32{20},
		Steps:      [][]TreeSpecSeqStep{{{Parent: parent, Tokens: tokens, TargetArgmax: argmax}}},
		TotalPages: 8,
		EOSTokenID: -1,
	}
	got, err := s.RunEmulatedLoop(t.Context(), in)
	if err != nil {
		t.Fatal(err)
	}
	if want := []int32{1, 2, 3, 4, 5, 6, 20, 22, 25, 31}; !slices.Equal(got.Tokens[0], want) || got.Roots[0] != 88 {
		t.Fatalf("tokens %v root %d, want %v 88", got.Tokens[0], got.Roots[0], want)
	}
	for j, node := range []int{0, 2, 5, 11} {
		row := got.KV[0][(L+j)*s.KVWords : (L+j+1)*s.KVWords]
		for w, x := range row {
			if x != TreeSpecNodeKV(0, 0, node, w) {
				t.Fatalf("slot L+%d word %d = %#x, want node %d's %#x", j, w, x, node, TreeSpecNodeKV(0, 0, node, w))
			}
		}
	}
	// L+4 = 10 tokens need 3 pages; the tree reserved ceil(22/4) = 6.
	if len(got.PageTable[0]) != 3 || len(got.FreePages) != 5 {
		t.Fatalf("pages %v, free %v; want 3 in use and 5 free", got.PageTable[0], got.FreePages)
	}
}

func TestTreeSpecVerifySampledEmulatedLoop(t *testing.T) {
	if _, err := cudahostemu.Compiler(); errors.Is(err, cudahostemu.ErrNoCompiler) {
		t.Skip(err)
	}
	const (
		steps = 8
		vocab = 8
	)
	s := TreeSpecVerify{Name: "tree_verify_sampled", Nodes: 6, Batch: 2, KVWords: 2, PageSize: 4, MaxPages: 16, MaxSteps: steps, Sampled: true, VocabSize: vocab}
	r := rand.New(rand.NewPCG(9, 9))
	dist := func() []float32 {
		d := make([]float32, vocab)
		var sum float32
		for i := range d {
			d[i] = r.Float32() + 0.05
			sum += d[i]
		}
		for i := range d {
			d[i] /= sum
		}
		return d
	}
	in := TreeSpecInput{
		Prefix:     [][]int32{{1, 2}, {3}},
		Roots:      []int32{4, 5},
		TotalPages: 32,
		EOSTokenID: -1,
		Seed:       0x1234_5678_9abc,
	}
	for range steps {
		var st []TreeSpecSeqStep
		for range s.Batch {
			parent := []int{-1, 0, 0, 1, 1, 3}
			t := TreeSpecSeqStep{Parent: parent, Tokens: make([]int32, s.Nodes)}
			for i := range s.Nodes {
				t.DraftProbs = append(t.DraftProbs, dist())
				t.TargetProbs = append(t.TargetProbs, dist())
				if i > 0 {
					t.Tokens[i] = int32(r.IntN(vocab))
				}
			}
			st = append(st, t)
		}
		in.Steps = append(in.Steps, st)
	}
	want, err := s.Reference(in)
	if err != nil {
		t.Fatal(err)
	}
	got, err := s.RunEmulatedLoop(t.Context(), in)
	if err != nil {
		t.Fatal(err)
	}
	if d := treeSpecStateDiff(got, want); d != "" {
		t.Fatalf("emulated sampled loop differs from Reference:\n%s", d)
	}
	var drafts int32
	for _, acc := range want.Accepted {
		for _, a := range acc {
			drafts += a - 1
		}
	}
	if drafts == 0 {
		t.Fatal("no draft token accepted; the test exercises nothing")
	}
}

func ExampleAcceptDraftTreeGreedy() {
	// The root (node 0) has drafts 1 and 2; node 3 continues node 2.
	tree, err := BuildDraftTreeTopology([]int{-1, 0, 0, 2})
	if err != nil {
		panic(err)
	}
	tokens := []int32{7, 41, 42, 43}
	// The target's argmax after each node's path.
	argmax := []int32{42, 0, 43, 50}
	a, err := AcceptDraftTreeGreedy(tree, tokens, argmax)
	if err != nil {
		panic(err)
	}
	fmt.Printf("masks %b\n", tree.AncestorMask)
	fmt.Println("path", a.Path, "bonus", a.Bonus)
	// Output:
	// masks [1 11 101 1101]
	// path [0 2 3] bonus 50
}
