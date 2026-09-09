//go:build linux

package main

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/tmc/lib/lm/quant"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// Exercise compress itself across the first indexer-selection boundary. Earlier
// short-prefix gates never projected indexer queries: 512 rows need no query,
// and position 2051 emits row 513. No model mapping is needed for this fixture.
func TestCUDAIndexerProjectionBoundary(t *testing.T) {
	f := cudaSynthFoundation(t)
	r := &cudaRuntime{cudaFoundation: f}
	alloc := func(n int) cudamem.Slice[float32] { return cudaTestAlloc(t, f, n) }
	zeros := func(name string, cols, rows int) cudaModelTensor {
		return cudaSynthTensor(t, f, name, 1, []uint64{uint64(cols), uint64(rows)}, make([]byte, 2*cols*rows))
	}
	norm := func(name string, n int) cudaModelTensor {
		x := make([]float32, n)
		for i := range x {
			x[i] = 1
		}
		return cudaCompressedTensor(t, f, name, x)
	}
	state := func(head int) cudaCompressedState {
		width := 2 * head
		s := cudaCompressedState{headDim: head, width: width, ratio: 4, cap: 514, n: 512,
			cache: alloc(514 * head), stateKV: alloc(8 * width), stateScore: alloc(8 * width), pooled: alloc(head)}
		cudaCompressedUpload(t, s.cache, make([]float32, s.cache.Len()))
		cudaCompressedUpload(t, s.stateKV, make([]float32, s.stateKV.Len()))
		v := make([]float32, s.stateScore.Len())
		for i := range v {
			v[i] = float32(math.Inf(-1))
		}
		cudaCompressedUpload(t, s.stateScore, v)
		return s
	}
	l := cudaLayer{
		attnCompressorKV: zeros("attention kv", 4096, 1024), attnCompressorGate: zeros("attention gate", 4096, 1024),
		attnCompressorAPE: zeros("attention ape", 1024, 4), attnCompressorNorm: norm("attention norm", 512),
		indexerCompressorKV: zeros("indexer kv", 4096, 256), indexerCompressorGate: zeros("indexer gate", 4096, 256),
		indexerCompressorAPE: zeros("indexer ape", 256, 4), indexerCompressorNorm: norm("indexer norm", 128),
	}
	// A single exactly representable weight per row makes each projection's
	// expected value independent of the reduction tree. Inputs deliberately need
	// F16 rounding; substituting the raw-F32-input primitive must change bits.
	projection := func(name string, cols, rows int) (cudaModelTensor, []float32, []float32) {
		x := make([]float32, cols)
		for i := range x {
			x[i] = float32(1+i%17)/16 + 1.0/32768
		}
		data, want := make([]byte, 2*cols*rows), make([]float32, rows)
		for row := range want {
			col := (row*37 + 11) % cols
			binary.LittleEndian.PutUint16(data[2*(row*cols+col):], 0x3800) // 0.5
			want[row] = quant.Float16ToFloat32(cudaQ81HalfReference(x[col])) * 0.5
		}
		return cudaSynthTensor(t, f, name, 1, []uint64{uint64(cols), uint64(rows)}, data), x, want
	}
	var qin, qwant, hin, hwant []float32
	l.indexerQB, qin, qwant = projection("blk.2.indexer.attn_q_b.weight", 1024, 8192)
	l.indexerProj, hin, hwant = projection("blk.2.indexer.proj.weight", 4096, 64)
	s := cudaTokenState{attnNorm: cudaSynthUpload(t, f, hin), qLora: cudaSynthUpload(t, f, qin),
		compressorKV: alloc(1024), compressorGate: alloc(1024), indexerKV: alloc(256), indexerGate: alloc(256),
		indexerQ: alloc(8192), indexerWeights: alloc(64), indexerScratch: alloc(8192), indexerScores: alloc(514)}
	allowed, err := cudamem.Alloc[uint32](f.device, 514)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := allowed.Free(); err != nil {
			t.Error(err)
		}
	})
	s.allowed = allowed
	// Retain production's scratch precondition even though the correct F16
	// dispatch does not consume it. Restoring the old Q8_0 dispatch must reach
	// the tensor-type refusal rather than fail earlier on missing scratch.
	q80Input, err := cudamem.Alloc[byte](f.device, 1024/32*36)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := q80Input.Free(); err != nil {
			t.Error(err)
		}
	})
	r.state.q80Input = q80Input
	ls := cudaSessionLayer{attn: state(512), indexer: state(128)}
	poison := make([]float32, 8192)
	for i := range poison {
		poison[i] = math.Float32frombits(0x7fc01234)
	}
	cudaCompressedUpload(t, s.indexerQ, poison)
	for _, position := range []int{2050, 2051, 2052} {
		if position == 2052 {
			for i := range qin {
				qin[i] *= 2
			}
			for i := range qwant {
				qwant[i] *= 2
			}
			cudaCompressedUpload(t, s.qLora, qin)
		}
		if err := r.compress(&l, &ls, &s, position); err != nil {
			t.Fatalf("position %d: compress: %v", position, err)
		}
		if err := f.device.WithContext(func() error { return syncStream(f.device.DefaultStream()) }); err != nil {
			t.Fatal(err)
		}
		wantN := 513
		if position == 2050 {
			wantN = 512
		}
		if ls.indexer.n != wantN || ls.attn.n != wantN {
			t.Fatalf("position %d: row counts attention=%d indexer=%d want%d", position, ls.attn.n, ls.indexer.n, wantN)
		}
		q := cudaTestRead(t, s.indexerQ)
		if position == 2050 {
			for i, v := range q {
				if math.Float32bits(v) != math.Float32bits(poison[i]) {
					t.Fatalf("inactive query overwritten at %d", i)
				}
			}
		} else {
			for i, v := range q {
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || math.Float32bits(v) != math.Float32bits(qwant[i]) {
					t.Fatalf("position %d: query bits at %d: %08x want %08x", position, i, math.Float32bits(v), math.Float32bits(qwant[i]))
				}
			}
			for i, v := range cudaTestRead(t, s.indexerWeights) {
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || math.Float32bits(v) != math.Float32bits(hwant[i]) {
					t.Fatalf("position %d: head weight bits at %d", position, i)
				}
			}
		}
		mask := make([]uint32, 514)
		if err := allowed.CopyToHost(mask); err != nil {
			t.Fatal(err)
		}
		for i := 0; i < wantN; i++ {
			want := uint32(1)
			if i == 512 {
				want = 0
			}
			if mask[i] != want {
				t.Fatalf("position %d: allowed[%d]=%d want%d", position, i, mask[i], want)
			}
		}
		t.Logf("position=%d compressed_rows=%d projections=%t", position, wantN, position != 2050)
	}
}
