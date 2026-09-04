//go:build darwin && arm64

package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
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
	compareOracleFP8F32(t, "kqv_out", heads, oracle+"/stage_kqv_out-0_pos1.bin")
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
