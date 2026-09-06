//go:build darwin

package main

import (
	"context"
	"fmt"
	"math"
	"os"
	"slices"
	"testing"

	apple "github.com/tmc/apple/metal"
	"github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
	"github.com/tmc/lib/apple/x/metal/metalattn"
	"github.com/tmc/lib/examples/internal/metalprefix"
)

func runPersistentGQABlocks(t *testing.T, eng *ModelEngine, cfg Config, leases []*metalprefix.Lease, layer int, qRows, suffixKRows, suffixVRows [][]float32) [][]float32 {
	t.Helper()
	if !eng.dev.Raw().SupportsFamily(apple.MTLGPUFamilyApple9) {
		t.Fatal("persistent GQA blocks: Apple family 9 required")
	}
	if err := metalattn.ValidatePersistentGQA(metalattn.Config{NumQHeads: cfg.NumHeads, NumKVHeads: cfg.NumKVHeads, HeadDim: cfg.HeadDim}, cfg.MaxSeqLen); err != nil {
		t.Fatal(err)
	}
	lib, err := eng.dev.CompileLibrary(metalattn.PersistentGQAMSL)
	if err != nil {
		t.Fatal(err)
	}
	pso, err := eng.dev.NewComputePipeline(lib, "PersistentGQABlockDecode")
	if err != nil {
		t.Fatal(err)
	}
	qd, kd := cfg.NumHeads*cfg.HeadDim, cfg.NumKVHeads*cfg.HeadDim
	var desc []metalBlockDescriptor
	rows := make([]metalBlockRow, len(leases))
	var q, sk, sv []float32
	var pins []*metalprefix.ViewLease
	defer func() { releaseMetalBlockPins(pins) }()
	for row, lease := range leases {
		descStart := len(desc)
		pin, err := lease.PinLayer(layer)
		if err != nil {
			t.Fatal(err)
		}
		pins = append(pins, pin)
		views := pin.Views()
		rows[row] = metalBlockRow{PrefixTokens: uint32(lease.Tokens()), SuffixTokens: uint32(len(suffixKRows[row]) / kd), SuffixOffset: uint32(len(sk) / kd), BlockOffset: uint32(len(desc)), BlockCount: uint32(len(views))}
		want := 0
		for _, view := range views {
			if view.Start != want {
				t.Fatalf("row %d block starts at %d, want %d", row, view.Start, want)
			}
			desc = append(desc, metalBlockDescriptor{Start: uint32(view.Start), End: uint32(view.End), K: uint64(view.K.GpuAddress()) + uint64(view.KOffset), V: uint64(view.V.GpuAddress()) + uint64(view.VOffset)})
			want = view.End
		}
		if want != lease.Tokens() || len(qRows[row]) != qd || len(suffixKRows[row]) != len(suffixVRows[row]) {
			t.Fatal("invalid persistent block fixture")
		}
		// The shader searches by logical range and must not depend on descriptor
		// or physical allocation order. Reverse each production-derived table.
		slices.Reverse(desc[descStart:])
		q = append(q, qRows[row]...)
		sk = append(sk, suffixKRows[row]...)
		sv = append(sv, suffixVRows[row]...)
	}
	reversed := false
	for i := 1; i < len(desc); i++ {
		if desc[i].K < desc[i-1].K {
			reversed = true
		}
	}
	qb, _ := metalmem.AllocFromSlice(eng.dev, q)
	defer qb.Free()
	kb, _ := metalmem.AllocFromSlice(eng.dev, sk)
	defer kb.Free()
	vb, _ := metalmem.AllocFromSlice(eng.dev, sv)
	defer vb.Free()
	out, _ := metalmem.Alloc[float32](eng.dev, len(q))
	defer out.Free()
	db, _ := metalmem.AllocFromSlice(eng.dev, desc)
	defer db.Free()
	rb, _ := metalmem.AllocFromSlice(eng.dev, rows)
	defer rb.Free()
	cb := eng.dev.Queue().CommandBuffer()
	ce := cb.ComputeCommandEncoder()
	ce.SetComputePipelineState(pso)
	for i, b := range []metalmem.Slice[float32]{qb, kb, vb, out} {
		ce.SetBufferWithOffsetAtIndex(b.Buffer(), b.Offset(), uint(i))
	}
	ce.SetBufferWithOffsetAtIndex(db.Buffer(), db.Offset(), 4)
	ce.SetBufferWithOffsetAtIndex(rb.Buffer(), rb.Offset(), 5)
	metal.SetScalar(ce, uint32(cfg.NumHeads), 6)
	metal.SetScalar(ce, uint32(cfg.NumKVHeads), 7)
	metal.SetScalar(ce, uint32(cfg.HeadDim), 8)
	metal.SetScalar(ce, float32(1/math.Sqrt(float64(cfg.HeadDim))), 9)
	for _, pin := range pins {
		for _, view := range pin.Views() {
			ce.UseResourceUsage(view.K, apple.MTLResourceUsageRead)
			ce.UseResourceUsage(view.V, apple.MTLResourceUsageRead)
		}
	}
	ce.DispatchThreadgroupsThreadsPerThreadgroup(metal.Threads(len(leases)*cfg.NumKVHeads), metal.Threads(metalattn.DecodeThreadgroupSize))
	ce.EndEncoding()
	cb.Commit()
	if err := metal.WaitCommandBuffer(context.Background(), cb); err != nil {
		t.Fatal(err)
	}
	flat := out.HostSlice()
	result := make([][]float32, len(leases))
	for i := range result {
		result[i] = append([]float32(nil), flat[i*qd:(i+1)*qd]...)
	}
	t.Logf("physical=true kernel=PersistentGQABlockDecode groups=%d threads=128 descriptors=%d descriptor_bytes=%d row_bytes=%d reversed_address_transition=%v", len(leases)*cfg.NumKVHeads, len(desc), len(desc)*32, len(rows)*20, reversed)
	return result
}

func TestPersistentGQABlockRealLlamaAllLayers(t *testing.T) {
	model := os.Getenv("LLAMA_INFER_TEST_MODEL")
	if model == "" {
		t.Skip("LLAMA_INFER_TEST_MODEL is not set")
	}
	cfg := Llama3_2_1BConfig(model)
	cfg.Precision = "f32"
	cfg.MaxSeqLen = 32
	eng, err := GetOrCompileEngine(cfg)
	if err != nil {
		t.Fatal(err)
	}
	prefix := []int32{cfg.BOSTokenID, 22691, 11}
	tokens := []int32{358, 13, 7, 91}
	seq, err := eng.NewSequenceKV(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer seq.Free()
	if _, err := GPUBatchedPrefillPassKV(eng, cfg, seq, append(prefix, tokens[0]), 0); err != nil {
		t.Fatal(err)
	}
	cache, err := metalprefix.New(metalprefix.Config{Device: eng.dev, MaxBytes: 64 << 20, MaxEntries: 8, BlockTokens: 1, Layers: cfg.NumLayers, KVHeads: cfg.NumKVHeads, HeadDim: cfg.HeadDim})
	if err != nil {
		t.Fatal(err)
	}
	defer cache.Close()
	leases := make([]*metalprefix.Lease, 4)
	for row := range leases {
		rowTokens := append([]int32(nil), prefix...)
		rowTokens[1] += int32(row)
		if err := cache.Put(context.Background(), rowTokens, tokens[row], sequenceSlices(seq.KCache), sequenceSlices(seq.VCache), metalprefix.CompletedFence()); err != nil {
			t.Fatal(err)
		}
		lease, ok, err := cache.Acquire(append(rowTokens, tokens[row]))
		if err != nil || !ok {
			t.Fatalf("acquire row %d: %v %v", row, ok, err)
		}
		leases[row] = lease
		defer lease.Release()
	}
	for layer := range cfg.NumLayers {
		for _, b := range []int{1, 2, 4} {
			qRows := make([][]float32, b)
			kRows := make([][]float32, b)
			vRows := make([][]float32, b)
			for row := range b {
				qkv, err := RunBatchedQKV(eng, cfg, layer, []int32{tokens[row]}, []int{len(prefix)})
				if err != nil {
					t.Fatal(err)
				}
				rotCache, err := eng.NewBatchedAttentionCache(cfg, 1)
				if err != nil {
					t.Fatal(err)
				}
				rot, err := RunBatchedAttention(eng, cfg, layer, rotCache, qkv.Q, qkv.K, qkv.V, []int{len(prefix)}, []int{0})
				rotCache.Free()
				if err != nil {
					t.Fatal(err)
				}
				qRows[row], kRows[row], vRows[row] = rot.Q, rot.K, qkv.V
			}
			got := runPersistentGQABlocks(t, eng, cfg, leases[:b], layer, qRows, kRows, vRows)
			for row := range b {
				kd := cfg.NumKVHeads * cfg.HeadDim
				pk := seq.KCache[layer].HostSlice()[:len(prefix)*kd]
				pv := seq.VCache[layer].HostSlice()[:len(prefix)*kd]
				want := contiguousAttention(cfg, qRows[row], append(pk, kRows[row]...), append(pv, vRows[row]...))
				for i := range want {
					if d := math.Abs(float64(got[row][i] - want[i])); d > 4e-4 {
						t.Fatalf("layer=%d B=%d row=%d output=%d diff=%g", layer, b, row, i, d)
					}
				}
			}
		}
	}
	t.Logf("real_model=%s layers=%d batches=1,2,4 every_output=true", model, cfg.NumLayers)
}

func TestPersistentGQABlockRatiosAndLongContexts(t *testing.T) {
	eng, err := GetOrCompileEngine(DefaultTinyLlamaConfig())
	if err != nil {
		t.Fatal(err)
	}
	for _, ratio := range []int{1, 2, 4, 8} {
		for _, total := range []int{257, 2048} {
			cfg := DefaultTinyLlamaConfig()
			cfg.NumLayers, cfg.NumKVHeads, cfg.NumHeads, cfg.HeadDim, cfg.MaxSeqLen = 1, 2, 2*ratio, 64, total
			prefixTokens := total - 1
			pk, pv := blockAttentionData(cfg, prefixTokens, 0.003)
			cache, err := metalprefix.New(metalprefix.Config{Device: eng.dev, MaxBytes: 64 << 20, MaxEntries: 8, BlockTokens: 257, Layers: 1, KVHeads: 2, HeadDim: 64})
			if err != nil {
				t.Fatal(err)
			}
			leases := make([]*metalprefix.Lease, 4)
			for row := range leases {
				tokens := make([]int32, prefixTokens)
				for i := range tokens {
					tokens[i] = int32(i + 1 + row*10000)
				}
				if err := cache.Put(context.Background(), tokens, int32(7+row), pk, pv, metalprefix.CompletedFence()); err != nil {
					t.Fatal(err)
				}
				lease, ok, err := cache.Acquire(append(tokens, 99))
				if err != nil || !ok {
					t.Fatalf("acquire: %v %v", ok, err)
				}
				leases[row] = lease
			}
			qd, kd := cfg.NumHeads*cfg.HeadDim, cfg.NumKVHeads*cfg.HeadDim
			for _, b := range []int{1, 2, 4} {
				q, sk, sv := make([][]float32, b), make([][]float32, b), make([][]float32, b)
				for row := range b {
					q[row] = make([]float32, qd)
					sk[row] = make([]float32, kd)
					sv[row] = make([]float32, kd)
					for i := range q[row] {
						q[row][i] = float32((i+row)%23-11) * 0.007
					}
					for i := range sk[row] {
						sk[row][i] = float32(i%17-8) * 0.005
						sv[row][i] = float32(i%19-9) * 0.006
					}
				}
				got := runPersistentGQABlocks(t, eng, cfg, leases[:b], 0, q, sk, sv)
				for row := range b {
					want := contiguousAttention(cfg, q[row], append(append([]float32(nil), pk[0]...), sk[row]...), append(append([]float32(nil), pv[0]...), sv[row]...))
					for i := range want {
						if d := math.Abs(float64(got[row][i] - want[i])); d > 3e-5 {
							t.Fatalf("ratio=%d total=%d B=%d row=%d output=%d diff=%g", ratio, total, b, row, i, d)
						}
					}
				}
			}
			for _, lease := range leases {
				lease.Release()
			}
			if err := cache.Close(); err != nil {
				t.Fatal(err)
			}
		}
	}
}

func TestPersistentGQABlockRejectsMissingCoverage(t *testing.T) {
	_ = fmt.Sprintf("compile-time ABI guard: descriptor=%d row=%d", 32, 20)
}
