//go:build linux

package main

import (
	"bufio"
	"context"
	"encoding/binary"
	"math"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
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
	wantAt := []int{0, len(r.gguf.tensors) / 2, len(r.gguf.tensors) - 1}
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
			tensor, err := r.tensor(r.gguf.tensors[index].name)
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

	qa := cudaTestTensor(t, r, "blk.0.attn_q_a.weight")
	qaNorm := cudaTestTensor(t, r, "blk.0.attn_q_a_norm.weight")
	qb := cudaTestTensor(t, r, "blk.0.attn_q_b.weight")
	kv := cudaTestTensor(t, r, "blk.0.attn_kv.weight")
	kvNorm := cudaTestTensor(t, r, "blk.0.attn_kv_a_norm.weight")
	x := make([]float32, 4096)
	for i := range x {
		x[i] = float32((i*37+11)%257-128) / 128
	}

	dx := cudaTestAlloc(t, r, len(x))
	qRaw := cudaTestAlloc(t, r, int(qa.dimensions[1]))
	qNorm := cudaTestAlloc(t, r, int(qa.dimensions[1]))
	qHeadsRaw := cudaTestAlloc(t, r, int(qb.dimensions[1]))
	qHeads := cudaTestAlloc(t, r, int(qb.dimensions[1]))
	kvRaw := cudaTestAlloc(t, r, int(kv.dimensions[1]))
	kvOut := cudaTestAlloc(t, r, int(kv.dimensions[1]))
	if err := dx.CopyFromHost(x); err != nil {
		t.Fatal(err)
	}

	start := time.Now()
	if err := r.encodeQ80Matvec(qa, dx, qRaw); err != nil {
		t.Fatal(err)
	}
	if err := r.encodeRMS(qRaw, qaNorm, qNorm, 1, true); err != nil {
		t.Fatal(err)
	}
	if err := r.encodeQ80Matvec(qb, qNorm, qHeadsRaw); err != nil {
		t.Fatal(err)
	}
	if err := r.encodeRMS(qHeadsRaw, cudaModelTensor{}, qHeads, 64, false); err != nil {
		t.Fatal(err)
	}
	if err := r.encodeQ80Matvec(kv, dx, kvRaw); err != nil {
		t.Fatal(err)
	}
	if err := r.encodeRMS(kvRaw, kvNorm, kvOut, 1, true); err != nil {
		t.Fatal(err)
	}
	if err := r.device.DefaultStream().Synchronize(context.Background()); err != nil {
		t.Fatal(err)
	}
	elapsed := time.Since(start)

	gotQRaw := cudaTestRead(t, qRaw)
	gotQNorm := cudaTestRead(t, qNorm)
	gotQHeadsRaw := cudaTestRead(t, qHeadsRaw)
	gotQHeads := cudaTestRead(t, qHeads)
	gotKVRaw := cudaTestRead(t, kvRaw)
	gotKV := cudaTestRead(t, kvOut)
	wantQRaw := cudaTestQ80Tensor(t, r, qa, x)
	wantQNorm := cudaTestRMS(wantQRaw, cudaTestF32Tensor(t, r, qaNorm), 1)
	wantQHeadsRaw := cudaTestQ80Tensor(t, r, qb, wantQNorm)
	wantQHeads := cudaTestRMS(wantQHeadsRaw, nil, 64)
	wantKVRaw := cudaTestQ80Tensor(t, r, kv, x)
	wantKV := cudaTestRMS(wantKVRaw, cudaTestF32Tensor(t, r, kvNorm), 1)
	cudaTestClose(t, "attnQA", gotQRaw, wantQRaw)
	cudaTestClose(t, "attnQA RMS", gotQNorm, wantQNorm)
	cudaTestClose(t, "attnQB", gotQHeadsRaw, wantQHeadsRaw)
	cudaTestClose(t, "query head RMS", gotQHeads, wantQHeads)
	cudaTestClose(t, "attnKV", gotKVRaw, wantKVRaw)
	cudaTestClose(t, "attnKV RMS", gotKV, wantKV)
	t.Logf("layer 0 attention projection chain: 6 launches, 1 stream synchronization, %s", elapsed)
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

func cudaTestRMS(x, weight []float32, groups int) []float32 {
	out := make([]float32, len(x))
	width := len(x) / groups
	for group := range groups {
		var sum float32
		for _, v := range x[group*width : (group+1)*width] {
			sum += v * v
		}
		inv := float32(1 / math.Sqrt(float64(sum/float32(width)+1e-6)))
		for i, v := range x[group*width : (group+1)*width] {
			w := float32(1)
			if weight != nil {
				w = weight[i]
			}
			out[group*width+i] = v * inv * w
		}
	}
	return out
}

func cudaTestClose(t *testing.T, name string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s length = %d, want %d", name, len(got), len(want))
	}
	var maxDelta float64
	for i := range got {
		delta := math.Abs(float64(got[i] - want[i]))
		limit := 2e-3 + 2e-4*math.Abs(float64(want[i]))
		if delta > maxDelta {
			maxDelta = delta
		}
		if delta > limit {
			t.Fatalf("%s[%d] = %g, want %g (delta %g > %g)", name, i, got[i], want[i], delta, limit)
		}
	}
	t.Logf("%s: %d values matched, max delta=%g", name, len(got), maxDelta)
}

func cudaTestQ80Row(p []byte, x []float32) float32 {
	var sum float32
	for block := 0; block < len(x)/32; block++ {
		scale := cudaTestFloat16(binary.LittleEndian.Uint16(p[block*34:]))
		for i := range 32 {
			sum += scale * float32(int8(p[block*34+2+i])) * x[block*32+i]
		}
	}
	return sum
}

func cudaTestFloat16(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := uint32(h>>10) & 0x1f
	frac := uint32(h & 0x03ff)
	switch exp {
	case 0:
		if frac == 0 {
			return math.Float32frombits(sign)
		}
		e := int32(-14)
		for frac&0x0400 == 0 {
			frac <<= 1
			e--
		}
		frac &= 0x03ff
		return math.Float32frombits(sign | uint32(e+127)<<23 | frac<<13)
	case 0x1f:
		return math.Float32frombits(sign | 0x7f800000 | frac<<13)
	default:
		return math.Float32frombits(sign | uint32(int32(exp)-15+127)<<23 | frac<<13)
	}
}
