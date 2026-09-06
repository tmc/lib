//go:build darwin

package metalattn

import (
	"math"
	"testing"
	"unsafe"

	apple "github.com/tmc/apple/metal"
	"github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
)

func TestPersistentGQADecodePhysicalParity(t *testing.T) {
	dev, err := metal.NewDevice()
	if err != nil {
		t.Skipf("Metal device unavailable: %v", err)
	}
	lib, err := dev.CompileLibrary(PersistentGQAMSL)
	if err != nil {
		t.Fatalf("compile persistent GQA: %v", err)
	}
	pso, err := dev.NewComputePipeline(lib, "PersistentGQADecode")
	if err != nil {
		t.Fatalf("create persistent GQA pipeline: %v", err)
	}

	for _, ratio := range []int{1, 2, 4, 8} {
		for _, batch := range []int{1, 2, 4} {
			for _, seqLen := range []int{2, 257, 2048} {
				name := fmtName(ratio, batch, seqLen)
				t.Run(name, func(t *testing.T) {
					const kvHeads, headDim = 2, 64
					cfg := DefaultConfig(kvHeads*ratio, kvHeads, headDim)
					q := make([]float32, batch*cfg.NumQHeads*headDim)
					kv := batch * seqLen * kvHeads * headDim
					k, v := make([]float32, kv), make([]float32, kv)
					fillPseudoRandom(q, uint32(5+ratio+batch))
					fillPseudoRandom(k, uint32(11+seqLen))
					fillPseudoRandom(v, uint32(17+ratio))
					positions := make([]uint32, batch)
					for i := range positions {
						positions[i] = uint32(seqLen - 1 - i%(min(seqLen, 7)))
					}
					got := runPersistentGQA(t, dev, pso, cfg, seqLen, positions, q, k, v)
					rowQ := cfg.NumQHeads * headDim
					rowKV := seqLen * kvHeads * headDim
					for row := range batch {
						want := referenceDecodeAttention(q[row*rowQ:(row+1)*rowQ], k[row*rowKV:(row+1)*rowKV], v[row*rowKV:(row+1)*rowKV], cfg, int(positions[row]))
						checkPersistentGQA(t, got[row*rowQ:(row+1)*rowQ], want, 2e-5)
					}
					t.Logf("physical=true dispatch_groups=%d threads_per_group=%d q_bytes=%d kv_bytes=%d out_bytes=%d", batch*kvHeads, DecodeThreadgroupSize, len(q)*4, (len(k)+len(v))*4, len(got)*4)
				})
			}
		}
	}
}

func TestPersistentGQADecodeAdversarial(t *testing.T) {
	dev, err := metal.NewDevice()
	if err != nil {
		t.Skip(err)
	}
	lib, err := dev.CompileLibrary(PersistentGQAMSL)
	if err != nil {
		t.Fatal(err)
	}
	pso, err := dev.NewComputePipeline(lib, "PersistentGQADecode")
	if err != nil {
		t.Fatal(err)
	}
	cfg := DefaultConfig(8, 2, 64)
	for _, special := range []float32{1e10, float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
		q := make([]float32, cfg.NumQHeads*cfg.HeadDim)
		k := make([]float32, 4*cfg.NumKVHeads*cfg.HeadDim)
		v := make([]float32, len(k))
		fillPseudoRandom(q, 3)
		fillPseudoRandom(k, 5)
		fillPseudoRandom(v, 7)
		q[0] = special
		got := runPersistentGQA(t, dev, pso, cfg, 4, []uint32{3}, q, k, v)
		want := referenceDecodeAttention(q, k, v, cfg, 3)
		if math.IsNaN(float64(special)) || math.IsInf(float64(special), 0) {
			for i := 0; i < cfg.HeadDim; i++ {
				want[i] = 0
			}
		}
		checkPersistentGQA(t, got, want, 2e-5)
	}
}

func TestValidatePersistentGQA(t *testing.T) {
	tests := []struct {
		name     string
		cfg      Config
		sequence int
		wantErr  bool
	}{
		{"supported", DefaultConfig(32, 8, 64), 2048, false},
		{"fractional ratio", DefaultConfig(7, 2, 64), 2, true},
		{"ratio too large", DefaultConfig(18, 2, 64), 2, true},
		{"head dimension too large", DefaultConfig(8, 2, 257), 2, true},
		{"sequence too long", DefaultConfig(8, 2, 64), PersistentGQAMaxSequence + 1, true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := ValidatePersistentGQA(test.cfg, test.sequence)
			if (err != nil) != test.wantErr {
				t.Fatalf("ValidatePersistentGQA error = %v, wantErr %v", err, test.wantErr)
			}
		})
	}
}

func runPersistentGQA(t *testing.T, dev *metal.Device, pso apple.MTLComputePipelineState, cfg Config, maxSeq int, positions []uint32, q, k, v []float32) []float32 {
	t.Helper()
	if !dev.Raw().SupportsFamily(apple.MTLGPUFamilyApple9) {
		t.Fatal("persistent GQA: Apple family 9 required")
	}
	if err := ValidatePersistentGQA(cfg, maxSeq); err != nil {
		t.Fatal(err)
	}
	alloc := func(src []float32) metalmem.Slice[float32] {
		b, err := metalmem.Alloc[float32](dev, len(src))
		if err != nil {
			t.Fatal(err)
		}
		copy(b.HostSlice(), src)
		return b
	}
	qb, kb, vb := alloc(q), alloc(k), alloc(v)
	out, err := metalmem.Alloc[float32](dev, len(q))
	if err != nil {
		t.Fatal(err)
	}
	pos, err := metalmem.Alloc[uint32](dev, len(positions))
	if err != nil {
		t.Fatal(err)
	}
	copy(pos.HostSlice(), positions)
	newScalar := func(p unsafe.Pointer) apple.MTLBuffer {
		return dev.Raw().NewBufferWithBytesLengthOptions(p, 4, apple.MTLResourceStorageModeShared)
	}
	maxSeqV, qhV, kvhV, dimV, scaleV := uint32(maxSeq), uint32(cfg.NumQHeads), uint32(cfg.NumKVHeads), uint32(cfg.HeadDim), cfg.Scale
	cb := dev.Queue().CommandBuffer()
	enc := cb.ComputeCommandEncoder()
	enc.SetComputePipelineState(pso)
	for i, b := range []apple.MTLBuffer{qb.Buffer(), kb.Buffer(), vb.Buffer(), out.Buffer(), pos.Buffer(), newScalar(unsafe.Pointer(&maxSeqV)), newScalar(unsafe.Pointer(&qhV)), newScalar(unsafe.Pointer(&kvhV)), newScalar(unsafe.Pointer(&dimV)), newScalar(unsafe.Pointer(&scaleV))} {
		enc.SetBufferWithOffsetAtIndex(b, 0, uint(i))
	}
	enc.DispatchThreadgroupsThreadsPerThreadgroup(apple.MTLSize{Width: uint(cfg.NumKVHeads * len(positions)), Height: 1, Depth: 1}, apple.MTLSize{Width: DecodeThreadgroupSize, Height: 1, Depth: 1})
	enc.EndEncoding()
	cb.Commit()
	if err := metal.WaitCommandBuffer(t.Context(), cb); err != nil {
		t.Fatalf("physical dispatch: %v", err)
	}
	return append([]float32(nil), out.HostSlice()...)
}

func checkPersistentGQA(t *testing.T, got, want []float32, tol float64) {
	t.Helper()
	for i := range want {
		if math.IsNaN(float64(want[i])) {
			if !math.IsNaN(float64(got[i])) {
				t.Fatalf("out[%d]=%v, want NaN", i, got[i])
			}
			continue
		}
		if math.IsInf(float64(want[i]), 0) {
			if got[i] != want[i] {
				t.Fatalf("out[%d]=%v, want %v", i, got[i], want[i])
			}
			continue
		}
		if d := math.Abs(float64(got[i] - want[i])); d > tol {
			t.Fatalf("out[%d]=%v want=%v diff=%g", i, got[i], want[i], d)
		}
	}
}

func fmtName(ratio, batch, seq int) string {
	return "GQA" + itoa(ratio) + "/B" + itoa(batch) + "/L" + itoa(seq)
}
func itoa(v int) string {
	if v == 0 {
		return "0"
	}
	b := [20]byte{}
	i := len(b)
	for v > 0 {
		i--
		b[i] = byte('0' + v%10)
		v /= 10
	}
	return string(b[i:])
}
