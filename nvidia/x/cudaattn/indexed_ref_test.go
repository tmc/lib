package cudaattn

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"math"
	"sort"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudatest"
)

type indexedFixture struct {
	Name     string
	Config   IndexedConfig
	Request  IndexedRequest
	Q, K, V  []float32
	Pipeline bool
}

func indexedFixtures() []indexedFixture {
	c := IndexedConfig{Batch: 2, Queries: 3, QueryHeads: 4, KVHeads: 2, HeadDim: 7, BlockSize: 3, MaxSequence: 8, PhysicalBlocks: 7, SelectedBlocks: 4, Selection: IndexedQueryHeads, Causal: true, Scale: 0.4}
	e, _ := c.Extents()
	r := IndexedRequest{Sequences: []int32{0, 1, 0}, Positions: []int32{6, 3, -1}, Lengths: []int32{8, 5}, Pages: []int32{6, 2, 4, 1, 5, -1}, IDs: make([]int32, e.IDs), Counts: make([]int32, e.Sets)}
	for i := range r.IDs {
		r.IDs[i] = -1
	}
	for row := range r.Counts {
		n := 2
		if row >= 4 && row < 8 {
			n = 1
		}
		r.Counts[row] = int32(n)
		r.IDs[row*c.SelectedBlocks] = int32(row % 2)
		if n == 2 {
			r.IDs[row*c.SelectedBlocks+1] = 2
		}
	}
	base := indexedFixture{Name: "subset-paged-asymmetric-tail-offset", Config: c, Request: r, Q: make([]float32, e.Query), K: make([]float32, e.KV), V: make([]float32, e.KV)}
	for i := range base.Q {
		base.Q[i] = float32(i%13-6) / 5
	}
	for i := range base.K {
		base.K[i] = float32((i*7)%17-8) / 3
		base.V[i] = float32((i*11)%19-9) / 4
	}
	full := indexedCopyFixture(base)
	full.Name = "all-blocks"
	for row := range full.Request.Counts {
		seq := full.Request.Sequences[row/c.QueryHeads]
		n := (int(full.Request.Lengths[seq]) + c.BlockSize - 1) / c.BlockSize
		full.Request.Counts[row] = int32(n)
		for j := 0; j < c.SelectedBlocks; j++ {
			full.Request.IDs[row*c.SelectedBlocks+j] = -1
			if j < n {
				full.Request.IDs[row*c.SelectedBlocks+j] = int32(j)
			}
		}
	}
	empty := indexedCopyFixture(base)
	empty.Name = "empty-selection"
	for i := range empty.Request.Counts {
		empty.Request.Counts[i] = 0
	}
	for i := range empty.Request.IDs {
		empty.Request.IDs[i] = -1
	}
	shared := indexedCopyFixture(base)
	shared.Name = "explicit-shared-kv-head"
	shared.Config.Selection = IndexedKVHeads
	se, _ := shared.Config.Extents()
	shared.Request.IDs = make([]int32, se.IDs)
	shared.Request.Counts = make([]int32, se.Sets)
	for i := range shared.Request.IDs {
		shared.Request.IDs[i] = -1
	}
	for row := range shared.Request.Counts {
		shared.Request.Counts[row] = 1
		shared.Request.IDs[row*c.SelectedBlocks] = int32(row % 2)
	}
	zero := indexedCopyFixture(empty)
	zero.Name = "zero-length-sequence"
	zero.Request.Lengths[0] = 0
	zero.Request.Positions[0] = -1
	for i := 0; i < 3; i++ {
		zero.Request.Pages[i] = -1
	}
	pipe := indexedCopyFixture(base)
	pipe.Name = "proxy-topk-short-and-masked"
	pipe.Pipeline = true
	pipe.Config.SelectedBlocks = 2
	pe, _ := pipe.Config.Extents()
	pipe.Request.IDs = make([]int32, pe.IDs)
	for i := range pipe.Request.IDs {
		pipe.Request.IDs[i] = -1
	}
	pipe.Request.Counts = make([]int32, pe.Sets)
	pipe.Request.Lengths[1] = 2
	pipe.Request.Positions[1] = 1
	pipe.Request.Pages[4] = -1
	return []indexedFixture{base, full, empty, shared, zero, pipe}
}
func indexedCopyFixture(f indexedFixture) indexedFixture {
	f.Request.Sequences = append([]int32(nil), f.Request.Sequences...)
	f.Request.Positions = append([]int32(nil), f.Request.Positions...)
	f.Request.Lengths = append([]int32(nil), f.Request.Lengths...)
	f.Request.Pages = append([]int32(nil), f.Request.Pages...)
	f.Request.IDs = append([]int32(nil), f.Request.IDs...)
	f.Request.Counts = append([]int32(nil), f.Request.Counts...)
	return f
}

// Independent restricted dense oracle: enumerate all actual positions first,
// apply membership/causal masks, then materialize FP64 scores and normalize once.
// It neither visits compacted IDs in order nor uses the production recurrence.
func indexedOracle(f indexedFixture) ([]float32, []float32) {
	c, r := f.Config, f.Request
	e, _ := c.Extents()
	out, lse := make([]float32, e.Query), make([]float32, e.Rows)
	heads := e.Sets / c.Queries
	for row := 0; row < e.Rows; row++ {
		query, head := row/c.QueryHeads, row%c.QueryHeads
		kv := head / (c.QueryHeads / c.KVHeads)
		sethead := head
		if c.Selection == IndexedKVHeads {
			sethead = kv
		}
		set := query*heads + sethead
		seq := int(r.Sequences[query])
		selected := map[int]bool{}
		for j := 0; j < int(r.Counts[set]); j++ {
			selected[int(r.IDs[set*c.SelectedBlocks+j])] = true
		}
		var scores []float64
		var bases []int
		m := math.Inf(-1)
		for actual := 0; actual < int(r.Lengths[seq]); actual++ {
			if !selected[actual/c.BlockSize] || c.Causal && actual > int(r.Positions[query]) {
				continue
			}
			physical := int(r.Pages[seq*e.MaxBlocks+actual/c.BlockSize])
			base := ((physical*c.BlockSize+actual%c.BlockSize)*c.KVHeads + kv) * c.HeadDim
			dot := 0.0
			for d := 0; d < c.HeadDim; d++ {
				dot += float64(f.Q[row*c.HeadDim+d]) * float64(f.K[base+d])
			}
			dot *= float64(c.Scale)
			scores = append(scores, dot)
			bases = append(bases, base)
			m = math.Max(m, dot)
		}
		if len(scores) == 0 {
			lse[row] = float32(math.Inf(-1))
			continue
		}
		weights := make([]float64, len(scores))
		sum := 0.0
		for i, s := range scores {
			weights[i] = math.Exp(s - m)
			sum += weights[i]
		}
		for d := 0; d < c.HeadDim; d++ {
			acc := 0.0
			for i, w := range weights {
				acc += w / sum * float64(f.V[bases[i]+d])
			}
			out[row*c.HeadDim+d] = float32(acc)
		}
		lse[row] = float32(m + math.Log(sum))
	}
	return out, lse
}

func indexedProxyOracle(f indexedFixture) ([]float32, []int32, []int32) {
	c, r := f.Config, f.Request
	e, _ := c.Extents()
	scores := make([]float32, e.Scores)
	ids, counts := make([]int32, e.IDs), make([]int32, e.Rows)
	for i := range ids {
		ids[i] = -1
	}
	for i := range scores {
		scores[i] = float32(math.Inf(-1))
	}
	for row := 0; row < e.Rows; row++ {
		query, head := row/c.QueryHeads, row%c.QueryHeads
		seq := int(r.Sequences[query])
		n := (int(r.Lengths[seq]) + c.BlockSize - 1) / c.BlockSize
		kv := head / (c.QueryHeads / c.KVHeads)
		for actual := 0; actual < int(r.Lengths[seq]); actual++ {
			if c.Causal && actual > int(r.Positions[query]) {
				continue
			}
			block := actual / c.BlockSize
			p := int(r.Pages[seq*e.MaxBlocks+block])
			base := ((p*c.BlockSize+actual%c.BlockSize)*c.KVHeads + kv) * c.HeadDim
			dot := 0.0
			for d := 0; d < c.HeadDim; d++ {
				dot += float64(f.Q[row*c.HeadDim+d]) * float64(f.K[base+d])
			}
			score := float32(dot * float64(c.Scale))
			scores[row*e.MaxBlocks+block] = max(scores[row*e.MaxBlocks+block], score)
		}
		order := make([]int, n)
		for i := range order {
			order[i] = i
		}
		sort.Slice(order, func(i, j int) bool {
			a, b := order[i], order[j]
			x, y := scores[row*e.MaxBlocks+a], scores[row*e.MaxBlocks+b]
			return x > y || x == y && a < b
		})
		count := min(n, c.SelectedBlocks)
		counts[row] = int32(count)
		for j := 0; j < count; j++ {
			ids[row*c.SelectedBlocks+j] = int32(order[j])
		}
	}
	return scores, ids, counts
}
func indexedInputDigest(f indexedFixture) string {
	h := sha256.New()
	h.Write([]byte("indexed-input-v1\x00"))
	c := f.Config
	causal := int64(0)
	if c.Causal {
		causal = 1
	}
	for _, x := range []int64{int64(c.Batch), int64(c.Queries), int64(c.QueryHeads), int64(c.KVHeads), int64(c.HeadDim), int64(c.BlockSize), int64(c.MaxSequence), int64(c.PhysicalBlocks), int64(c.SelectedBlocks), int64(c.Selection), causal, int64(math.Float32bits(c.Scale))} {
		var b [8]byte
		binary.LittleEndian.PutUint64(b[:], uint64(x))
		h.Write(b[:])
	}
	for _, v := range [][]float32{f.Q, f.K, f.V} {
		var size [8]byte
		binary.LittleEndian.PutUint64(size[:], uint64(len(v)))
		h.Write(size[:])
		for _, x := range v {
			var b [4]byte
			binary.LittleEndian.PutUint32(b[:], math.Float32bits(x))
			h.Write(b[:])
		}
	}
	for _, v := range [][]int32{f.Request.Sequences, f.Request.Positions, f.Request.Lengths, f.Request.Pages, f.Request.IDs, f.Request.Counts} {
		var size [8]byte
		binary.LittleEndian.PutUint64(size[:], uint64(len(v)))
		h.Write(size[:])
		for _, x := range v {
			var b [4]byte
			binary.LittleEndian.PutUint32(b[:], uint32(x))
			h.Write(b[:])
		}
	}
	return hex.EncodeToString(h.Sum(nil))
}
func indexedCompare(f indexedFixture, source, target string, actual, want []float32, lse bool) error {
	sum := sha256.Sum256([]byte(source))
	infinity := "reject"
	if lse {
		infinity = "same-sign"
	}
	return cudatest.CompareCUDAOutputFP32(actual, want, cudatest.CUDAOutputPolicy{Target: target, DType: "fp32", MathMode: "fp32-rn fmad=false ftz=false online-softmax", SourceDigest: hex.EncodeToString(sum[:]), InputDigest: indexedInputDigest(f), Elements: len(want), Absolute: 2e-5, Relative: 2e-5, NaN: "reject", Infinity: infinity, Zero: "equal"})
}
func indexedCheckResult(t *testing.T, f indexedFixture, source, target string, out, lse []float32) {
	t.Helper()
	want, wl := indexedOracle(f)
	if err := indexedCompare(f, source, target, out, want, false); err != nil {
		t.Fatal(err)
	}
	if err := indexedCompare(f, source, target, lse, wl, true); err != nil {
		t.Fatal(err)
	}
}
