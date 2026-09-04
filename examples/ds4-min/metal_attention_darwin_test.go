//go:build darwin && arm64

package main

import (
	"math"
	"os"
	"testing"

	libmetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
)

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
