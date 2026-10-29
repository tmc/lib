package tensorparallel

import (
	"context"
	"math"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudacollective"
	"github.com/tmc/lib/nvidia/x/egraph"
	"github.com/tmc/lib/nvidia/x/egraph/hopper"
)

var testLayer = Layer{Hidden: 512, Heads: 16, KVHeads: 4, HeadDim: 64, Intermediate: 1024, Eps: 1e-5, QKVBias: true}

// weights is a dense layer's checkpoint, row-major, in float64.
type weights struct {
	attnNorm, ffnNorm  []float64
	wq, wk, wv, bq, bk []float64
	bv                 []float64
	wo, wg, wu, wd     []float64
}

func randWeights(r *rand.Rand, l Layer) weights {
	v := func(n int, s float64) []float64 {
		out := make([]float64, n)
		for i := range out {
			out[i] = r.NormFloat64() * s
		}
		return out
	}
	h, q, kv, in := l.Hidden, l.Heads*l.HeadDim, l.KVHeads*l.HeadDim, l.Intermediate
	return weights{
		attnNorm: v(h, 1), ffnNorm: v(h, 1),
		wq: v(q*h, 0.05), wk: v(kv*h, 0.05), wv: v(kv*h, 0.05),
		bq: v(q, 0.1), bk: v(kv, 0.1), bv: v(kv, 0.1),
		wo: v(h*q, 0.05), wg: v(in*h, 0.05), wu: v(in*h, 0.05), wd: v(h*in, 0.05),
	}
}

func rmsnorm(x, g []float64, eps float64) []float64 {
	var ss float64
	for _, v := range x {
		ss += v * v
	}
	s := 1 / math.Sqrt(ss/float64(len(x))+eps)
	out := make([]float64, len(x))
	for i := range x {
		out[i] = x[i] * s * g[i]
	}
	return out
}

// matvec returns w·x for w [len(w)/len(x), len(x)].
func matvec(w, x []float64) []float64 {
	out := make([]float64, len(w)/len(x))
	for i := range out {
		for j, v := range x {
			out[i] += w[i*len(x)+j] * v
		}
	}
	return out
}

func add(a, b []float64) []float64 {
	out := slices.Clone(a)
	for i := range b {
		out[i] += b[i]
	}
	return out
}

// rankForward runs rank rp's share of the layer for one token at position
// 0, where attention returns each query head's v: it returns the o_proj and
// MLP partials given x and the reduced h. Rank 0 adds the residuals.
func rankForward(l Layer, rp RankPlan, w weights, x []float64, reduceH func([]float64) []float64) (oPart, ffnPart []float64) {
	h := l.Hidden
	xn := rmsnorm(x, w.attnNorm, float64(l.Eps))
	qkv := add(matvec(QKV(rp, w.wq, w.wk, w.wv, h), xn), QKV(rp, w.bq, w.bk, w.bv, 1))
	hd := l.HeadDim
	heads, kvHeads := rp.QHeads.Len(), rp.KVHeads.Len()
	v := qkv[(heads+kvHeads)*hd:]
	attn := make([]float64, heads*hd)
	for i := range heads {
		g := i / (heads / kvHeads)
		copy(attn[i*hd:(i+1)*hd], v[g*hd:(g+1)*hd])
	}
	oPart = matvec(O(rp, w.wo), attn)
	if rp.Rank == 0 {
		oPart = add(oPart, x)
	}
	hid := reduceH(oPart)
	hn := rmsnorm(hid, w.ffnNorm, float64(l.Eps))
	gu := matvec(GateUp(rp, w.wg, w.wu), hn)
	n := rp.Inter.Len()
	act := make([]float64, n)
	for i := range act {
		g := gu[i]
		act[i] = g / (1 + math.Exp(-g)) * gu[n+i]
	}
	ffnPart = matvec(Down(rp, w.wd), act)
	if rp.Rank == 0 {
		ffnPart = add(ffnPart, hid)
	}
	return oPart, ffnPart
}

// forward runs the layer sharded by p, summing partials across ranks for
// each all-reduce.
func forward(t *testing.T, p *Plan, w weights, x []float64) []float64 {
	t.Helper()
	// Run the ranks in lock step: every rank's o_proj partial is needed
	// before any rank's MLP.
	oParts := make([][]float64, p.TP)
	for r, rp := range p.Ranks {
		oParts[r], _ = rankForward(p.Layer, rp, w, x, func(o []float64) []float64 { return o })
	}
	h := make([]float64, p.Layer.Hidden)
	for _, o := range oParts {
		h = add(h, o)
	}
	out := make([]float64, p.Layer.Hidden)
	for _, rp := range p.Ranks {
		_, f := rankForward(p.Layer, rp, w, x, func([]float64) []float64 { return h })
		out = add(out, f)
	}
	return out
}

func TestShardMatchesTP1(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	w := randWeights(r, testLayer)
	x := make([]float64, testLayer.Hidden)
	for i := range x {
		x[i] = r.NormFloat64()
	}
	p1, err := Shard(testLayer, Options{TP: 1})
	if err != nil {
		t.Fatal(err)
	}
	want := forward(t, p1, w, x)
	for _, tp := range []int{2, 4} {
		p, err := Shard(testLayer, Options{TP: tp})
		if err != nil {
			t.Fatal(err)
		}
		got := forward(t, p, w, x)
		for i := range want {
			if d := math.Abs(got[i] - want[i]); d > 1e-9*(1+math.Abs(want[i])) {
				t.Fatalf("tp %d: out[%d] = %g, want %g", tp, i, got[i], want[i])
			}
		}
	}
}

// TestShardHelpers pins the shard layout on a small matrix.
func TestShardHelpers(t *testing.T) {
	w := []int{
		0, 1, 2, 3,
		4, 5, 6, 7,
		8, 9, 10, 11,
	}
	tests := []struct {
		name string
		got  []int
		want []int
	}{
		{"rows", ShardRows(w, 4, Range{1, 3}), []int{4, 5, 6, 7, 8, 9, 10, 11}},
		{"cols", ShardCols(w, 4, Range{2, 4}), []int{2, 3, 6, 7, 10, 11}},
		{"empty", ShardCols(w, 4, Range{1, 1}), []int{}},
	}
	for _, tt := range tests {
		if !slices.Equal(tt.got, tt.want) {
			t.Errorf("%s = %v, want %v", tt.name, tt.got, tt.want)
		}
	}
}

func inputNames(p *hopper.Plan) []string {
	var names []string
	for _, in := range p.Inputs {
		names = append(names, in.Name)
	}
	slices.Sort(names)
	return names
}

func TestShardPlans(t *testing.T) {
	l := testLayer
	l.QKVBias = false
	for _, tt := range []struct {
		tp     int
		fusion Fusion
		target string
		store  hopper.GEMMStore
	}{
		{1, FuseNone, "", hopper.StoreWrite},
		{2, FuseNone, "", hopper.StoreWrite},
		{4, FuseReduceAdd, "", hopper.StoreReduceAdd},
		{2, FuseMultimem, "sm_103a", hopper.StoreMultimemReduceAdd},
	} {
		p, err := Shard(l, Options{TP: tt.tp, Fusion: tt.fusion, Target: tt.target})
		if err != nil {
			t.Fatal(err)
		}
		if p.Local.Heads != l.Heads/tt.tp || p.Local.KVHeads != l.KVHeads/tt.tp || p.Local.Intermediate != l.Intermediate/tt.tp {
			t.Errorf("tp %d: local layer %+v", tt.tp, p.Local)
		}
		for _, rp := range p.Ranks {
			wantO, wantF := []string{"attn", "w_o"}, []string{"ffn_norm", "h", "w_down", "w_gu"}
			if rp.Rank == 0 {
				wantO = append(wantO, "x")
			}
			slices.Sort(wantO)
			if got := inputNames(rp.OProj); !slices.Equal(got, wantO) {
				t.Errorf("tp %d rank %d: o_proj inputs %v, want %v", tt.tp, rp.Rank, got, wantO)
			}
			if got := inputNames(rp.FFN); !slices.Equal(got, wantF) {
				t.Errorf("tp %d rank %d: ffn inputs %v, want %v", tt.tp, rp.Rank, got, wantF)
			}
			if rp.Rank > 0 && (rp.OProj == p.Ranks[0].OProj || rp.FFN == p.Ranks[0].FFN) {
				t.Errorf("tp %d rank %d shares rank 0's residual plans", tt.tp, rp.Rank)
			}
			for _, pl := range []*hopper.Plan{rp.OProj, rp.FFN} {
				last := pl.Kernels[len(pl.Kernels)-1]
				if last.GEMM == nil || last.GEMM.Store != tt.store {
					t.Errorf("tp %d rank %d %s: last kernel %+v, want a gemm storing %v", tt.tp, rp.Rank, pl.Name, last, tt.store)
				}
				if tt.store != hopper.StoreWrite {
					src, err := last.GEMM.CUDA()
					if err != nil {
						t.Fatal(err)
					}
					if !strings.Contains(src, "atomicAdd") {
						t.Errorf("%s: fused store emits no reduction", pl.Name)
					}
				}
			}
			if rp.QHeads.Len()/rp.KVHeads.Len() != l.Heads/l.KVHeads {
				t.Errorf("tp %d rank %d: %v heads over %v kv heads splits a GQA group", tt.tp, rp.Rank, rp.QHeads, rp.KVHeads)
			}
		}
	}
}

func TestShardErrors(t *testing.T) {
	for _, tt := range []struct {
		name string
		l    Layer
		opt  Options
	}{
		{"tp0", testLayer, Options{TP: 0}},
		{"kv", testLayer, Options{TP: 8}},
		{"inter", Layer{Hidden: 512, Heads: 8, KVHeads: 8, HeadDim: 64, Intermediate: 1000}, Options{TP: 8}},
		{"gqa", Layer{Hidden: 512, Heads: 6, KVHeads: 4, HeadDim: 64, Intermediate: 1024}, Options{TP: 2}},
		{"multimem_sm80", testLayer, Options{TP: 2, Fusion: FuseMultimem, Target: "sm_80"}},
		{"multimem_target", testLayer, Options{TP: 2, Fusion: FuseMultimem}},
		{"fuse_rows", Layer{Hidden: 512, Heads: 8, KVHeads: 4, HeadDim: 64, Intermediate: 1024, Rows: 64}, Options{TP: 2, Fusion: FuseReduceAdd}},
		{"fusion", testLayer, Options{TP: 2, Fusion: 7}},
	} {
		if _, err := Shard(tt.l, tt.opt); err == nil {
			t.Errorf("%s: Shard succeeded", tt.name)
		}
	}
}

// TestCollectiveSchedule lowers every layer's all-reduces through the
// collective runtime's validator and host replay.
func TestCollectiveSchedule(t *testing.T) {
	const tp, layers = 2, 3
	p, err := Shard(testLayer, Options{TP: tp})
	if err != nil {
		t.Fatal(err)
	}
	m, err := cudacollective.NewMembership([]cudacollective.Rank{{ID: 0}, {ID: 1}})
	if err != nil {
		t.Fatal(err)
	}
	q, err := cudacollective.NewClique(m)
	if err != nil {
		t.Fatal(err)
	}
	tr := cudacollective.NewDeterministicTestTransport(tp)
	var comms []*cudacollective.Communicator
	for r := range tp {
		c, err := cudacollective.NewCommunicator(cudacollective.CommunicatorConfig{Rank: r, Clique: q, Generation: 1, Transport: tr})
		if err != nil {
			t.Fatal(err)
		}
		comms = append(comms, c)
	}
	calls := make([][]egraph.CollectiveCall, tp)
	var inputs [][][]float32
	for layer := range layers {
		for _, node := range p.Collectives() {
			step := make([][]float32, tp)
			for r, in := range node.Intents("tp", 1, layer, tp) {
				bytes := uint64(4 * node.Count)
				base := uint64(1<<32*(r+1)) + uint64(len(calls[r]))*2*bytes
				pred := uint64(0)
				if n := len(calls[r]); n > 0 {
					pred = calls[r][n-1].Completion
				}
				calls[r] = append(calls[r], egraph.CollectiveCall{
					CollectiveIntent: in,
					Reduction:        "sum",
					Layout:           "contiguous",
					Partition:        1,
					Send:             egraph.CollectiveBuffer{Address: base, Bytes: bytes},
					Recv:             egraph.CollectiveBuffer{Address: base + bytes, Bytes: bytes},
					Predecessor:      pred,
					Completion:       uint64(1000*(r+1) + len(calls[r]) + 1),
					StreamState:      egraph.CollectiveStreamOrdered,
				})
				step[r] = make([]float32, node.Count)
				step[r][0] = float32(r + 1)
			}
			inputs = append(inputs, step)
		}
	}
	plan, err := egraph.CollectiveSchedule{Communicators: comms, Calls: calls, Quarantine: new(egraph.CollectiveQuarantine)}.Plan()
	if err != nil {
		t.Fatal(err)
	}
	defer plan.Close()
	rec, err := plan.Replay(context.Background(), inputs)
	if err != nil {
		t.Fatal(err)
	}
	if len(rec.Steps) != 2*layers {
		t.Fatalf("%d steps, want %d", len(rec.Steps), 2*layers)
	}
	for _, s := range rec.Steps {
		if s.Outputs[1][0] != 3 {
			t.Fatalf("effect %d: rank 1 got %v, want 3", s.EffectToken, s.Outputs[1][0])
		}
	}
}
