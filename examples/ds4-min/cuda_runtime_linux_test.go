//go:build linux

package main

import (
	"bufio"
	"context"
	"encoding/binary"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
	"math"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"
)

func TestCUDAFoundation(t *testing.T) {
	path := os.Getenv("DS4_MODEL")
	if path == "" {
		t.Skip("DS4_MODEL is not set")
	}
	r, err := newCUDAFoundation(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := r.Close(); err != nil {
			t.Error(err)
		}
	})
	if len(r.model) != pinnedModelBytes {
		t.Fatalf("mapped bytes = %d, want %d", len(r.model), pinnedModelBytes)
	}
	if r.name == "" || r.major == 0 {
		t.Fatalf("invalid CUDA identity %q sm_%d%d", r.name, r.major, r.minor)
	}
	if r.hmm {
		if r.mapped.bytes != 0 {
			t.Fatalf("HMM registered bytes = %d, want 0", r.mapped.bytes)
		}
	} else {
		if r.mapped.host == 0 || r.mapped.device == 0 || r.mapped.bytes == 0 {
			t.Fatalf("model range was not registered: %+v", r.mapped)
		}
		if r.mapped.bytes > cudaMaxRegisteredModelRange+uint(os.Getpagesize()) {
			t.Fatalf("registered bytes = %d, want a bounded range", r.mapped.bytes)
		}
	}
	if r.probeTensor == "" || r.probeDevice != r.probeHost {
		t.Fatalf("CUDA read of %q = %#04x, host = %#04x", r.probeTensor, r.probeDevice, r.probeHost)
	}
	t.Logf("CUDA read of %q = host %#04x, device %#04x; HMM=%t registered=%d bytes", r.probeTensor, r.probeHost, r.probeDevice, r.hmm, r.mapped.bytes)
}

func TestCUDAHMMTensorViews(t *testing.T) {
	path := os.Getenv("DS4_MODEL")
	if path == "" {
		t.Skip("DS4_MODEL is not set")
	}
	beforeRSS := cudaTestProcKB(t, "/proc/self/status", "VmRSS:")
	beforeAvailable := cudaTestProcKB(t, "/proc/meminfo", "MemAvailable:")
	r, err := newCUDAFoundation(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := r.Close(); err != nil {
			t.Error(err)
		}
	})
	if !r.hmm {
		t.Skipf("device does not provide integrated UVA pageable host-page-table access; registered=%d", r.mapped.bytes)
	}
	if r.mapped.bytes != 0 {
		t.Fatalf("registered bytes = %d, want 0", r.mapped.bytes)
	}

	// Choose tensors whose file offsets span the model, then sample separate
	// pages at each tensor's beginning, middle, and end.
	wantAt := []int{0, len(r.gguf.Tensors) / 2, len(r.gguf.Tensors) - 1}
	compiled, err := nvrtcx.Compile(cudaProbeSource,
		nvrtcx.WithName("ds4_hmm_views.cu"),
		nvrtcx.WithArch(r.major, r.minor),
	)
	if err != nil {
		t.Fatal(err)
	}
	err = r.device.WithContext(func() error {
		mod, err := compiled.Load(r.device)
		if err != nil {
			return err
		}
		defer mod.Unload()
		fn, err := mod.GetFunction("ds4_read_u8")
		if err != nil {
			return err
		}
		out, err := cudamem.Alloc[uint32](r.device, 1)
		if err != nil {
			return err
		}
		defer out.Free()
		for _, index := range wantAt {
			tensor, err := r.tensor(r.gguf.Tensors[index].Name)
			if err != nil {
				return err
			}
			for _, sample := range []uint64{0, tensor.bytes / 2, tensor.bytes - 1} {
				if err := cudalaunch.Kernel(fn).Launch(tensor.device, sample, out); err != nil {
					return err
				}
				if err := driver.CtxSynchronize(); err != nil {
					return err
				}
				var got [1]uint32
				if err := out.CopyToHost(got[:]); err != nil {
					return err
				}
				want := r.model[tensor.fileOffset+sample]
				if got[0] != uint32(want) {
					t.Errorf("%s byte %d = %#02x, want %#02x", tensor.name, sample, got[0], want)
				}
			}
			t.Logf("read %q at [%d,%d), %d bytes", tensor.name, tensor.fileOffset, tensor.fileOffset+tensor.bytes, tensor.bytes)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	afterRSS := cudaTestProcKB(t, "/proc/self/status", "VmRSS:")
	afterAvailable := cudaTestProcKB(t, "/proc/meminfo", "MemAvailable:")
	t.Logf("memory kB: RSS %d -> %d; available %d -> %d; registered=0", beforeRSS, afterRSS, beforeAvailable, afterAvailable)
}

func cudaTestProcKB(t *testing.T, path, key string) uint64 {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Logf("read memory receipt %s: %v", path, err)
		return 0
	}
	defer f.Close()
	s := bufio.NewScanner(f)
	for s.Scan() {
		fields := strings.Fields(s.Text())
		if len(fields) < 2 || fields[0] != key {
			continue
		}
		value, err := strconv.ParseUint(fields[1], 10, 64)
		if err == nil {
			return value
		}
	}
	return 0
}

func TestCUDAQ80Row(t *testing.T) {
	path := os.Getenv("DS4_MODEL")
	if path == "" {
		t.Skip("DS4_MODEL is not set")
	}
	r, err := newCUDAFoundation(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := r.Close(); err != nil {
			t.Error(err)
		}
	})

	x := make([]float32, r.probeCols)
	for i := range x {
		x[i] = float32(i%17-8) / 64
	}
	dx, err := cudamem.Alloc[float32](r.device, len(x))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = dx.Free() })
	dout, err := cudamem.Alloc[float32](r.device, 1)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = dout.Free() })
	if err := dx.CopyFromHost(x); err != nil {
		t.Fatal(err)
	}
	if err := r.q80Row(dx, dout); err != nil {
		t.Fatal(err)
	}
	var got [1]float32
	if err := dout.CopyToHost(got[:]); err != nil {
		t.Fatal(err)
	}

	p := r.model[r.probeOffset : r.probeOffset+r.probeBytes]
	var want float32
	for block := 0; block < len(x)/32; block++ {
		scale := cudaTestFloat16(binary.LittleEndian.Uint16(p[block*34:]))
		for i := range 32 {
			want += scale * float32(int8(p[block*34+2+i])) * x[block*32+i]
		}
	}
	delta := math.Abs(float64(got[0] - want))
	limit := 1e-4 + 2e-5*math.Abs(float64(want))
	if delta > limit {
		t.Fatalf("Q8_0 row = %g, want %g (delta %g > %g)", got[0], want, delta, limit)
	}
	t.Logf("Q8_0 tensor %q row 0: cols=%d bytes=%d CUDA=%g CPU=%g delta=%g limit=%g",
		r.probeTensor, r.probeCols, r.probeBytes, got[0], want, delta, limit)
}

func TestCUDAQ80Matvec(t *testing.T) {
	path := os.Getenv("DS4_MODEL")
	if path == "" {
		t.Skip("DS4_MODEL is not set")
	}
	r, err := newCUDAFoundation(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := r.Close(); err != nil {
			t.Error(err)
		}
	})

	x := make([]float32, r.probeCols)
	for i := range x {
		x[i] = float32((i*37+11)%257-128) / 128
	}
	dx, err := cudamem.Alloc[float32](r.device, len(x))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = dx.Free() })
	dout, err := cudamem.Alloc[float32](r.device, int(r.probeRows))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = dout.Free() })
	if err := dx.CopyFromHost(x); err != nil {
		t.Fatal(err)
	}
	start := time.Now()
	if err := r.q80Matvec(dx, dout); err != nil {
		t.Fatal(err)
	}
	setupElapsed := time.Since(start)
	start = time.Now()
	if err := r.q80Matvec(dx, dout); err != nil {
		t.Fatal(err)
	}
	warmElapsed := time.Since(start)
	got := make([]float32, r.probeRows)
	if err := dout.CopyToHost(got); err != nil {
		t.Fatal(err)
	}

	p := r.model[r.probeOffset : r.probeOffset+r.probeBytes]
	rowBytes := int(r.probeCols) / 32 * 34
	var maxDelta, maxLimit float64
	for row := range got {
		want := cudaTestQ80Row(p[row*rowBytes:], x)
		delta := math.Abs(float64(got[row] - want))
		limit := 5e-4 + 5e-5*math.Abs(float64(want))
		if delta > maxDelta {
			maxDelta = delta
			maxLimit = limit
		}
		if delta > limit {
			t.Fatalf("Q8_0 row %d = %g, want %g (delta %g > %g)", row, got[row], want, delta, limit)
		}
	}
	t.Logf("Q8_0 tensor %q: %dx%d, %d bytes, all %d rows matched; max delta=%g limit=%g; setup=%s warm launch+sync=%s",
		r.probeTensor, r.probeRows, r.probeCols, r.probeBytes, r.probeRows, maxDelta, maxLimit, setupElapsed, warmElapsed)
}

func TestCUDALayer0AttentionProjectionChain(t *testing.T) {
	path := os.Getenv("DS4_MODEL")
	if path == "" {
		t.Skip("DS4_MODEL is not set")
	}
	r, err := newCUDAFoundation(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := r.Close(); err != nil {
			t.Error(err)
		}
	})
	if !r.hmm {
		t.Skip("layer projection chain requires HMM tensor views")
	}

	hcFn := cudaTestTensor(t, r, "blk.0.hc_attn_fn.weight")
	hcScale := cudaTestTensor(t, r, "blk.0.hc_attn_scale.weight")
	hcBase := cudaTestTensor(t, r, "blk.0.hc_attn_base.weight")
	attnNorm := cudaTestTensor(t, r, "blk.0.attn_norm.weight")
	qa := cudaTestTensor(t, r, "blk.0.attn_q_a.weight")
	qaNorm := cudaTestTensor(t, r, "blk.0.attn_q_a_norm.weight")
	qb := cudaTestTensor(t, r, "blk.0.attn_q_b.weight")
	kv := cudaTestTensor(t, r, "blk.0.attn_kv.weight")
	kvNorm := cudaTestTensor(t, r, "blk.0.attn_kv_a_norm.weight")
	state := make([]float32, 4*4096)
	for i := range state {
		state[i] = float32((i*37+11)%257-128) / 128
	}

	dState := cudaTestAlloc(t, r, len(state))
	dStateNorm := cudaTestAlloc(t, r, len(state))
	mix := cudaTestAlloc(t, r, 24)
	hcOut := cudaTestAlloc(t, r, 2*4096)
	x := hcOut.Slice(4096, 8192)
	qRaw := cudaTestAlloc(t, r, int(qa.dimensions[1]))
	qNorm := cudaTestAlloc(t, r, int(qa.dimensions[1]))
	qHeadsRaw := cudaTestAlloc(t, r, int(qb.dimensions[1]))
	qHeads := cudaTestAlloc(t, r, int(qb.dimensions[1]))
	kvRaw := cudaTestAlloc(t, r, int(kv.dimensions[1]))
	kvOut := cudaTestAlloc(t, r, int(kv.dimensions[1]))
	if err := dState.CopyFromHost(state); err != nil {
		t.Fatal(err)
	}
	wantStateNorm := cudaTestRMS(state, nil, 1)
	if err := dStateNorm.CopyFromHost(wantStateNorm); err != nil {
		t.Fatal(err)
	}
	if err := r.loadMathKernels(); err != nil {
		t.Fatal(err)
	}

	start := time.Now()
	if err := r.encodeF16Matvec(hcFn, dStateNorm, mix); err != nil {
		t.Fatal(err)
	}
	if err := r.encodeHCPre(dState, hcFn, hcScale, hcBase, attnNorm, hcOut); err != nil {
		t.Fatal(err)
	}
	if err := r.encodeQ80Matvec(qa, x, qRaw); err != nil {
		t.Fatal(err)
	}
	if err := r.encodeQ80Matvec(kv, x, kvRaw); err != nil {
		t.Fatal(err)
	}
	if err := r.encodeRMSPair(qRaw, qaNorm, qNorm, kvRaw, kvNorm, kvOut); err != nil {
		t.Fatal(err)
	}
	if err := r.encodeQ80Matvec(qb, qNorm, qHeadsRaw); err != nil {
		t.Fatal(err)
	}
	if err := r.encodeRMS(qHeadsRaw, cudaModelTensor{}, qHeads, 64, false); err != nil {
		t.Fatal(err)
	}
	if err := r.device.DefaultStream().Synchronize(context.Background()); err != nil {
		t.Fatal(err)
	}
	elapsed := time.Since(start)

	gotMix := cudaTestRead(t, mix)
	gotHC := cudaTestRead(t, hcOut)
	gotQRaw := cudaTestRead(t, qRaw)
	gotQNorm := cudaTestRead(t, qNorm)
	gotQHeadsRaw := cudaTestRead(t, qHeadsRaw)
	gotQHeads := cudaTestRead(t, qHeads)
	gotKVRaw := cudaTestRead(t, kvRaw)
	gotKV := cudaTestRead(t, kvOut)
	wantMix := cudaTestF16Matvec(t, r, hcFn, wantStateNorm)
	wantCur, wantX := cudaTestHCPre(state, wantMix, cudaTestF32Tensor(t, r, hcScale), cudaTestF32Tensor(t, r, hcBase), cudaTestF32Tensor(t, r, attnNorm))
	wantHC := append(wantCur, wantX...)
	wantQRaw := cudaTestQ80Tensor(t, r, qa, wantX)
	wantQNorm := cudaTestRMS(wantQRaw, cudaTestF32Tensor(t, r, qaNorm), 1)
	wantQHeadsRaw := cudaTestQ80Tensor(t, r, qb, wantQNorm)
	wantQHeads := cudaTestRMS(wantQHeadsRaw, nil, 64)
	wantKVRaw := cudaTestQ80Tensor(t, r, kv, wantX)
	wantKV := cudaTestRMS(wantKVRaw, cudaTestF32Tensor(t, r, kvNorm), 1)
	cudaTestClose(t, "HC F16 projection", gotMix, wantMix)
	cudaTestClose(t, "HC pre", gotHC, wantHC)
	cudaTestClose(t, "attnQA", gotQRaw, wantQRaw)
	cudaTestClose(t, "attnQA RMS", gotQNorm, wantQNorm)
	cudaTestClose(t, "attnQB", gotQHeadsRaw, wantQHeadsRaw)
	cudaTestClose(t, "query head RMS", gotQHeads, wantQHeads)
	cudaTestClose(t, "attnKV", gotKVRaw, wantKVRaw)
	cudaTestClose(t, "attnKV RMS", gotKV, wantKV)
	t.Logf("layer 0 HC and attention projection chain: 7 launches, 1 stream synchronization, %s", elapsed)
}

func cudaTestTensor(t *testing.T, r *cudaFoundation, name string) cudaModelTensor {
	t.Helper()
	tensor, err := r.tensor(name)
	if err != nil {
		t.Fatal(err)
	}
	return tensor
}

func cudaTestAlloc(t *testing.T, r *cudaFoundation, n int) cudamem.Slice[float32] {
	t.Helper()
	s, err := cudamem.Alloc[float32](r.device, n)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = s.Free() })
	return s
}

func cudaTestRead(t *testing.T, s cudamem.Slice[float32]) []float32 {
	t.Helper()
	v := make([]float32, s.Len())
	if err := s.CopyToHost(v); err != nil {
		t.Fatal(err)
	}
	return v
}

func cudaTestQ80Tensor(t *testing.T, r *cudaFoundation, tensor cudaModelTensor, x []float32) []float32 {
	t.Helper()
	if tensor.typeID != 8 || len(tensor.dimensions) != 2 || uint64(len(x)) != tensor.dimensions[0] {
		t.Fatalf("invalid Q8_0 test tensor %q", tensor.name)
	}
	p := r.model[tensor.fileOffset : tensor.fileOffset+tensor.bytes]
	rowBytes := len(x) / 32 * 34
	out := make([]float32, tensor.dimensions[1])
	for row := range out {
		out[row] = cudaTestQ80Row(p[row*rowBytes:], x)
	}
	return out
}

func cudaTestF32Tensor(t *testing.T, r *cudaFoundation, tensor cudaModelTensor) []float32 {
	t.Helper()
	if tensor.typeID != 0 || len(tensor.dimensions) != 1 {
		t.Fatalf("invalid F32 test tensor %q", tensor.name)
	}
	p := r.model[tensor.fileOffset : tensor.fileOffset+tensor.bytes]
	out := make([]float32, tensor.dimensions[0])
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(p[i*4:]))
	}
	return out
}

func cudaTestF16Matvec(t *testing.T, r *cudaFoundation, tensor cudaModelTensor, x []float32) []float32 {
	t.Helper()
	if tensor.typeID != 1 || len(tensor.dimensions) != 2 || uint64(len(x)) != tensor.dimensions[0] {
		t.Fatalf("invalid F16 test tensor %q", tensor.name)
	}
	p := r.model[tensor.fileOffset : tensor.fileOffset+tensor.bytes]
	return cudaTestF16MatvecBytes(p, x, int(tensor.dimensions[1]))
}

// cudaTestF16MatvecBytes is the row-major F16 matrix-vector reference over raw
// tensor bytes, so it serves both mapped and synthetic weights.
