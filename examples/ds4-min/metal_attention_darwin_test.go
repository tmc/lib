//go:build darwin && arm64

package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"
	"testing"

	libmetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
)

func TestMetalRatio4CompressorOracle(t *testing.T) {
	model := os.Getenv("DS4_GGUF_PROFILE_MODEL")
	dirs := strings.Split(os.Getenv("DS4_LAYER2_COMPRESSOR_ORACLE_DIRS"), ":")
	if model == "" || len(dirs) != 4 || dirs[0] == "" {
		t.Skip("model and four compressor oracle paths not set")
	}
	r, err := newMetalRuntime(model, 1024)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	state := &r.state.layers[2].attn
	layer := &r.layers[2]
	for position, dir := range dirs {
		kv := readOracleF32(t, dir+fmt.Sprintf("/stage_attn_comp_kv_raw-2_pos%d.bin", position))
		score := readOracleF32(t, dir+fmt.Sprintf("/stage_attn_comp_score_raw-2_pos%d.bin", position))
		got, emitted, err := r.ops.CompressorAttention(kv, score, layer.attnCompressorAPE, layer.attnCompressorNorm, state, position)
		if err != nil {
			t.Fatal(err)
		}
		if emitted != (position == 3) {
			t.Fatalf("position %d emitted = %t", position, emitted)
		}
		if emitted {
			compareOracleFP8F32(t, "KVcompress", got, dir+"/stage_KVcompress-2_pos3.bin")
		}
	}
}

func TestMetalMixedAttentionIncludesRawCompressedAndSink(t *testing.T) {
	dev, err := libmetal.NewDevice()
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	ops, err := newMetalPinnedOps(dev)
	if err != nil {
		t.Fatal(err)
	}
	defer ops.Close()
	a := metalmem.NewArena(dev)
	defer a.Free()
	raw := metalmem.AllocIn[float32](a, "raw", 4*512)
	comp := metalmem.AllocIn[float32](a, "compressed", 512)
	for i := range comp.HostSlice() {
		comp.HostSlice()[i] = 2
	}
	sinkData := metalmem.UploadIn(a, "sinks", make([]byte, 64*4))
	if err := a.Err(); err != nil {
		t.Fatal(err)
	}
	sinks := boundMetalTensor{name: "sinks", typeID: 0, dimensions: []uint64{64}, data: sinkData}
	kv := make([]float32, 512)
	for i := range kv {
		kv[i] = 1
	}
	_, gotKV, heads, err := ops.RopeAttentionMixed(make([]float32, 64*512), kv, sinks, raw, comp, 1, []uint32{1}, 0, 2)
	if err != nil {
		t.Fatal(err)
	}
	for i, value := range gotKV {
		if value != 1 {
			t.Fatalf("kv[%d] = %g, want 1", i, value)
		}
	}
	for i, value := range heads {
		if math.Abs(float64(value-1)) > 1e-6 {
			t.Fatalf("heads[%d] = %g, want 1", i, value)
		}
	}
}

func TestMetalMixedAttentionRatio4RopeOracle(t *testing.T) {
	model := os.Getenv("DS4_GGUF_PROFILE_MODEL")
	dirs := strings.Split(os.Getenv("DS4_LAYER2_COMPRESSOR_ORACLE_DIRS"), ":")
	if model == "" || len(dirs) != 4 || dirs[0] == "" {
		t.Skip("model and four compressor oracle paths not set")
	}
	r, err := newMetalRuntime(model, 1024)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	raw := r.state.layers[2].rawKV
	comp, err := r.state.layers[2].attn.cache.Slice(0, 512)
	if err != nil {
		t.Fatal(err)
	}
	copy(comp.HostSlice(), readOracleF32(t, dirs[3]+"/stage_KVcompress-2_pos3.bin"))
	q, kv, heads, err := r.ops.RopeAttentionMixed(
		readOracleF32(t, dirs[3]+"/stage_Qnorm-2_pos3.bin"),
		readOracleF32(t, dirs[3]+"/stage_KVnorm-2_pos3.bin"),
		r.layers[2].attnSinks,
		raw,
		comp,
		1,
		[]uint32{1},
		3,
		2,
	)
	if err != nil {
		t.Fatal(err)
	}
	compareOracleF32(t, "Qcur", q, dirs[3]+"/stage_Qcur-2_pos3.bin")
	compareOracleFP8F32(t, "KVcur", kv, dirs[3]+"/stage_KVcur-2_pos3.bin")
	for i, value := range heads {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			t.Fatalf("heads[%d] = %g", i, value)
		}
	}
}

func TestMetalRatio4CompressorEmitsFourthToken(t *testing.T) {
	dev, err := libmetal.NewDevice()
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	ops, err := newMetalPinnedOps(dev)
	if err != nil {
		t.Fatal(err)
	}
	defer ops.Close()
	a := metalmem.NewArena(dev)
	defer a.Free()
	state := metalCompressedState{
		cache:      metalmem.AllocIn[float32](a, "cache", 6*512),
		stateKV:    metalmem.AllocIn[float32](a, "state-kv", 8*1024),
		stateScore: metalmem.AllocIn[float32](a, "state-score", 8*1024),
		pooled:     metalmem.AllocIn[float32](a, "pooled", 512),
		headDim:    512,
		width:      1024,
		ratio:      4,
		cap:        6,
	}
	if err := a.Err(); err != nil {
		t.Fatal(err)
	}
	resetCompressedState(&state)
	apeData := metalmem.UploadIn(a, "ape", make([]byte, 4*1024*2))
	normBytes := make([]byte, 512*4)
	for i := range 512 {
		binary.LittleEndian.PutUint32(normBytes[4*i:], math.Float32bits(1))
	}
	normData := metalmem.UploadIn(a, "norm", normBytes)
	if err := a.Err(); err != nil {
		t.Fatal(err)
	}
	ape := boundMetalTensor{name: "ape", typeID: 1, dimensions: []uint64{1024, 4}, data: apeData}
	norm := boundMetalTensor{name: "norm", typeID: 0, dimensions: []uint64{512}, data: normData}
	kv := make([]float32, 1024)
	score := make([]float32, 1024)
	for i := range kv {
		kv[i] = 1
	}
	for position := range 4 {
		got, emitted, err := ops.CompressorAttention(kv, score, ape, norm, &state, position)
		if err != nil {
			t.Fatal(err)
		}
		if emitted != (position == 3) {
			t.Fatalf("position %d emitted = %t", position, emitted)
		}
		if !emitted {
			continue
		}
		for i, value := range got {
			if math.Abs(float64(value-1)) > 1e-5 {
				t.Fatalf("compressed[%d] = %g, want 1", i, value)
			}
		}
	}
	if state.n != 1 {
		t.Fatalf("compressed rows = %d, want 1", state.n)
	}
}

func TestMetalRatio128CompressorEmits128thToken(t *testing.T) {
	dev, err := libmetal.NewDevice()
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	ops, err := newMetalPinnedOps(dev)
	if err != nil {
		t.Fatal(err)
	}
	defer ops.Close()
	a := metalmem.NewArena(dev)
	defer a.Free()
	state := metalCompressedState{
		cache:      metalmem.AllocIn[float32](a, "cache", 4*512),
		stateKV:    metalmem.AllocIn[float32](a, "state-kv", 128*512),
		stateScore: metalmem.AllocIn[float32](a, "state-score", 128*512),
		pooled:     metalmem.AllocIn[float32](a, "pooled", 512),
		headDim:    512,
		width:      512,
		ratio:      128,
		cap:        4,
	}
	if err := a.Err(); err != nil {
		t.Fatal(err)
	}
	resetCompressedState(&state)
	apeData := metalmem.UploadIn(a, "ape", make([]byte, 128*512*2))
	normBytes := make([]byte, 512*4)
	for i := range 512 {
		binary.LittleEndian.PutUint32(normBytes[4*i:], math.Float32bits(1))
	}
	normData := metalmem.UploadIn(a, "norm", normBytes)
	if err := a.Err(); err != nil {
		t.Fatal(err)
	}
	ape := boundMetalTensor{name: "ape", typeID: 1, dimensions: []uint64{512, 128}, data: apeData}
	norm := boundMetalTensor{name: "norm", typeID: 0, dimensions: []uint64{512}, data: normData}
	kv := make([]float32, 512)
	score := make([]float32, 512)
	for i := range kv {
		kv[i] = 1
	}
	for position := range 128 {
		got, emitted, err := ops.CompressorAttention(kv, score, ape, norm, &state, position)
		if err != nil {
			t.Fatal(err)
		}
		if emitted != (position == 127) {
			t.Fatalf("position %d emitted = %t", position, emitted)
		}
		if emitted {
			for i, value := range got {
				if math.Abs(float64(value-1)) > 1e-5 {
					t.Fatalf("compressed[%d] = %g, want 1", i, value)
				}
			}
		}
	}
}

func TestMetalIndexerCompressorAppliesQAT(t *testing.T) {
	dev, err := libmetal.NewDevice()
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	ops, err := newMetalPinnedOps(dev)
	if err != nil {
		t.Fatal(err)
	}
	defer ops.Close()
	a := metalmem.NewArena(dev)
	defer a.Free()
	state := metalCompressedState{
		cache:      metalmem.AllocIn[float32](a, "cache", 6*128),
		stateKV:    metalmem.AllocIn[float32](a, "state-kv", 8*256),
		stateScore: metalmem.AllocIn[float32](a, "state-score", 8*256),
		pooled:     metalmem.AllocIn[float32](a, "pooled", 128),
		headDim:    128,
		width:      256,
		ratio:      4,
		cap:        6,
	}
	if err := a.Err(); err != nil {
		t.Fatal(err)
	}
	resetCompressedState(&state)
	apeData := metalmem.UploadIn(a, "ape", make([]byte, 4*256*2))
	normBytes := make([]byte, 128*4)
	for i := range 128 {
		binary.LittleEndian.PutUint32(normBytes[4*i:], math.Float32bits(1))
	}
	normData := metalmem.UploadIn(a, "norm", normBytes)
	if err := a.Err(); err != nil {
		t.Fatal(err)
	}
	ape := boundMetalTensor{name: "ape", typeID: 1, dimensions: []uint64{256, 4}, data: apeData}
	norm := boundMetalTensor{name: "norm", typeID: 0, dimensions: []uint64{128}, data: normData}
	kv := make([]float32, 256)
	score := make([]float32, 256)
	for i := range kv {
		kv[i] = 1
	}
	for position := range 4 {
		got, emitted, err := ops.CompressorIndexer(kv, score, ape, norm, &state, position)
		if err != nil {
			t.Fatal(err)
		}
		if emitted != (position == 3) {
			t.Fatalf("position %d emitted = %t", position, emitted)
		}
		if !emitted {
			continue
		}
		if got[0] != 12 {
			t.Fatalf("QAT[0] = %g, want 12", got[0])
		}
		for i, value := range got[1:] {
			if value != 0 {
				t.Fatalf("QAT[%d] = %g, want 0", i+1, value)
			}
		}
	}
}

func TestSelectIndexerRowsTiesPreferLowerIndex(t *testing.T) {
	scores := make([]float32, 513)
	allowed := selectIndexerRows(scores, 512)
	for i := range 512 {
		if allowed[i] != 1 {
			t.Fatalf("allowed[%d] = %d, want 1", i, allowed[i])
		}
	}
	if allowed[512] != 0 {
		t.Fatalf("allowed[512] = %d, want 0", allowed[512])
	}
}

func TestMetalIndexerAllowsAllEarlyRowsWithoutWeights(t *testing.T) {
	dev, err := libmetal.NewDevice()
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	ops, err := newMetalPinnedOps(dev)
	if err != nil {
		t.Fatal(err)
	}
	defer ops.Close()
	a := metalmem.NewArena(dev)
	defer a.Free()
	cache := metalmem.AllocIn[float32](a, "index-cache", 512*128)
	if err := a.Err(); err != nil {
		t.Fatal(err)
	}
	allowed, err := ops.IndexerAllowed(make([]float32, 1024), make([]float32, ds4Width), boundMetalTensor{}, boundMetalTensor{}, cache, 512, 2047, 2)
	if err != nil {
		t.Fatal(err)
	}
	for i, value := range allowed {
		if value != 1 {
			t.Fatalf("allowed[%d] = %d, want 1", i, value)
		}
	}
}

func TestMetalCompressedIndexer(t *testing.T) {
	dev, err := libmetal.NewDevice()
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	ops, err := newMetalPinnedOps(dev)
	if err != nil {
		t.Fatal(err)
	}
	defer ops.Close()

	arena := metalmem.NewArena(dev)
	defer arena.Free()

	const position = 2051

	q, weights := make([]float32, 64*128), make([]float32, 64)
	for i := range q {
		q[i] = float32((i*13)%61-30) / 32
	}
	for h := range weights {
		weights[h] = float32(h%5+1) / 8
	}

	transformed := make([]float32, len(q))
	for h := range 64 {
		row := append([]float32(nil), q[h*128:(h+1)*128]...)
		ropeCompressedTail(row, position, false)
		indexerQAT(row)
		copy(transformed[h*128:], row)
	}

	// Each cache row is a positive scalar multiple of one quantized query.
	// Scores have a strict ordering: row 0 is rejected at 513 rows.
	cache := make([]float32, 513*128)
	for c := range 513 {
		for i := range 128 {
			cache[c*128+i] = transformed[i] * float32(c+1) / 512
		}
	}

	// Build qWeight [1024, 8192] and headWeight [4096, 64] such that
	// matvec with unit-vector inputs reproduces q and weights exactly.
	qData := make([]byte, 2*1024*8192)
	for r, v := range q {
		binary.LittleEndian.PutUint16(qData[2*r*1024:], testFloat32ToFloat16(v))
	}
	qWeight := boundMetalTensor{
		name:       "indexer.attn_q_b.weight",
		typeID:     1,
		dimensions: []uint64{1024, 8192},
		data:       metalmem.UploadIn(arena, "indexer-qb", qData),
	}

	hData := make([]byte, 2*4096*64)
	for h, v := range weights {
		binary.LittleEndian.PutUint16(hData[2*h*4096:], testFloat32ToFloat16(v))
	}
	headWeight := boundMetalTensor{
		name:       "indexer.proj.weight",
		typeID:     1,
		dimensions: []uint64{4096, 64},
		data:       metalmem.UploadIn(arena, "indexer-proj", hData),
	}

	cacheGPU := metalmem.UploadIn(arena, "cache", cache)
	if err := arena.Err(); err != nil {
		t.Fatal(err)
	}

	qrNorm := make([]float32, 1024)
	qrNorm[0] = 1
	current := make([]float32, ds4Width)
	current[0] = 1

	for _, count := range []int{511, 512, 513} {
		t.Run(fmt.Sprintf("rows%d", count), func(t *testing.T) {
			allowed, err := ops.IndexerAllowed(qrNorm, current, qWeight, headWeight, cacheGPU, count, position, 2)
			if err != nil {
				t.Fatal(err)
			}
			if len(allowed) != count {
				t.Fatalf("len(allowed) = %d, want %d", len(allowed), count)
			}
			mask := make([]uint32, 513)
			for i := range mask {
				mask[i] = 99
			}
			copy(mask, allowed)
			for i, v := range mask {
				want := uint32(1)
				if i >= count {
					want = 99
				}
				if count == 513 && i == 0 {
					want = 0
				}
				if v != want {
					t.Fatalf("allowed[%d]=%d want%d", i, v, want)
				}
			}
			if count > 512 {
				// Assert the fixture has a strictly ordered score array where row 0 is the single loser.
				refScores := make([]float32, count)
				for c := range refScores {
					row := cache[c*128 : (c+1)*128]
					var score float32
					for h := range 64 {
						var dot float32
						for i, value := range row {
							dot += value * transformed[h*128+i]
						}
						if dot > 0 {
							score += dot * (weights[h] / float32(math.Sqrt(64*128)))
						}
					}
					refScores[c] = score
				}
				order := make([]int, count)
				for i := range order {
					order[i] = i
				}
				sort.SliceStable(order, func(i, j int) bool { return refScores[order[i]] > refScores[order[j]] })
				if order[len(order)-1] != 0 {
					t.Fatal("fixture has no strict excluded row")
				}
			}
		})
	}

	// Equal scores exercise stable lower-index tie breaking independently.
	t.Run("tie_breaking", func(t *testing.T) {
		zeroCurrent := make([]float32, ds4Width)
		allowed, err := ops.IndexerAllowed(qrNorm, zeroCurrent, qWeight, headWeight, cacheGPU, 513, position, 2)
		if err != nil {
			t.Fatal(err)
		}
		if len(allowed) != 513 {
			t.Fatalf("len(allowed) = %d, want 513", len(allowed))
		}
		for i, v := range allowed {
			want := uint32(1)
			if i == 512 {
				want = 0
			}
			if v != want {
				t.Fatalf("tie allowed[%d]=%d want%d", i, v, want)
			}
		}
	})
}

func testFloat32ToFloat16(f float32) uint16 {
	b := math.Float32bits(f)
	sign := uint16(b >> 16 & 0x8000)
	exp := int32(b>>23&0xFF) - 127
	mant := b & 0x7FFFFF

	switch {
	case exp == 128:
		if mant != 0 {
			return sign | 0x7C00 | uint16(mant>>13) | 0x0200
		}
		return sign | 0x7C00
	case exp > 15:
		return sign | 0x7C00
	case exp >= -14:
		half := sign | uint16(exp+15)<<10 | uint16(mant>>13)
		rest := mant & (1<<13 - 1)
		halfway := uint32(1) << 12
		if rest > halfway || (rest == halfway && half&1 == 1) {
			half++
		}
		return half
	case exp >= -24:
		m := mant | 0x800000
		shift := uint(-exp - 14 + 13)
		half := sign | uint16(m>>shift)
		rest := m & (1<<shift - 1)
		halfway := uint32(1) << (shift - 1)
		if rest > halfway || (rest == halfway && half&1 == 1) {
			half++
		}
		return half
	default:
		return sign
	}
}

func TestIndexerQAT(t *testing.T) {
	x := make([]float32, 128)
	for i := range x {
		x[i] = 1
	}
	indexerQAT(x)
	if x[0] != 12 {
		t.Fatalf("QAT[0] = %g, want 12", x[0])
	}
	for i, value := range x[1:] {
		if value != 0 {
			t.Fatalf("QAT[%d] = %g, want 0", i+1, value)
		}
	}
}

func TestMetalRawAttentionPositionOneOracle(t *testing.T) {
	model := os.Getenv("DS4_GGUF_PROFILE_MODEL")
	oracle := os.Getenv("DS4_LAYER0_POS1_ORACLE_DIR")
	if model == "" || oracle == "" {
		t.Skip("model and position 1 oracle paths not set")
	}
	r, err := newMetalRuntime(model, 1024)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	cache := r.state.layers[0].rawKV
	raw := readOracleF32(t, oracle+"/stage_raw_cache-0_pos1.bin")
	if len(raw) != 2*ds4KVWidth {
		t.Fatalf("raw cache length = %d, want %d", len(raw), 2*ds4KVWidth)
	}
	copy(cache.HostSlice(), raw[:ds4KVWidth])
	q, kv, heads, err := r.ops.RopeAttentionRaw(
		readOracleF32(t, oracle+"/stage_Qnorm-0_pos1.bin"),
		readOracleF32(t, oracle+"/stage_KVnorm-0_pos1.bin"),
		r.layers[0].attnSinks,
		cache,
		1,
	)
	if err != nil {
		t.Fatal(err)
	}
	compareOracleF32(t, "Qcur", q, oracle+"/stage_Qcur-0_pos1.bin")
	compareOracleFP8F32(t, "KVcur", kv, oracle+"/stage_KVcur-0_pos1.bin")
	compareOracleFP8F32(t, "kqv_back", heads, oracle+"/stage_kqv_back-0_pos1.bin")
}

func TestMetalRawAttentionKeepsHistory(t *testing.T) {
	dev, err := libmetal.NewDevice()
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	ops, err := newMetalPinnedOps(dev)
	if err != nil {
		t.Fatal(err)
	}
	defer ops.Close()
	cache, err := metalmem.Alloc[float32](dev, 2*ds4KVWidth)
	if err != nil {
		t.Fatal(err)
	}
	defer cache.Free()
	sinkData, err := metalmem.AllocFromSlice(dev, make([]byte, 64*4))
	if err != nil {
		t.Fatal(err)
	}
	defer sinkData.Free()
	sinks := boundMetalTensor{name: "sinks", typeID: 0, dimensions: []uint64{64}, data: sinkData}
	q := make([]float32, 64*ds4KVWidth)
	for position, value := range []float32{1, 2} {
		kv := make([]float32, ds4KVWidth)
		for i := 0; i < 448; i++ {
			kv[i] = value
		}
		_, gotKV, heads, err := ops.RopeAttentionRaw(q, kv, sinks, cache, position)
		if err != nil {
			t.Fatal(err)
		}
		if got := gotKV[0]; got != value {
			t.Fatalf("position %d KV = %g, want %g", position, got, value)
		}
		want := value / 2
		if position == 1 {
			want = 1
		}
		for head := range 64 {
			if got := heads[head*ds4KVWidth]; math.Abs(float64(got-want)) > 1e-6 {
				t.Fatalf("position %d head %d = %g, want %g", position, head, got, want)
			}
		}
	}
}
