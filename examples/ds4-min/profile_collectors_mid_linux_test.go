//go:build linux

package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

func TestCUDACollectorComparisonRejectsNonFinite(t *testing.T) {
	for _, v := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
		if err := cudaQualificationClose([]float32{v}, []float32{1}, 1e-4, 2e-3); err == nil {
			t.Fatal("non-finite actual accepted")
		}
		if err := cudaQualificationClose([]float32{1}, []float32{v}, 1e-4, 2e-3); err == nil {
			t.Fatal("non-finite reference accepted")
		}
	}
}

func TestCUDACollectorD2RMidValues(t *testing.T) {
	if os.Getenv("DS4_CUDA_COLLECTOR_GATE") == "" {
		t.Skip("DS4_CUDA_COLLECTOR_GATE is not set")
	}
	if os.Getenv("DS4_CUDA_COLLECTOR_GATE") != "1" || os.Getenv("GPU_LOCK_HELD") == "" {
		t.Fatal("explicit collector gate and gpu-lock required")
	}
	dev, err := cuda.NewDevice(0)
	if err != nil {
		t.Fatal(err)
	}
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		if closeErr := dev.Close(); closeErr != nil {
			t.Error(closeErr)
		}
		t.Fatal(err)
	}
	if major != 12 || minor != 1 {
		if err := dev.Close(); err != nil {
			t.Error(err)
		}
		t.Fatalf("collector gate requires sm_121, got sm_%d%d", major, minor)
	}
	r := &cudaFoundation{device: dev, major: major, minor: minor}
	t.Cleanup(func() {
		if err := r.Close(); err != nil {
			t.Error(err)
		}
	})
	baselineCompiled, err := nvrtcx.Compile(cudaMathCompilationSource(), cudaMathCompileOptions(major, minor)...)
	if err != nil {
		t.Fatal(err)
	}
	baselineImage := baselineCompiled.CUBIN()
	if len(baselineImage) == 0 {
		t.Fatal("baseline CUBIN unavailable")
	}
	baselineMod, err := baselineCompiled.LoadOutput(dev, nvrtcx.OutputCUBIN)
	if err != nil {
		t.Fatal(err)
	}
	r.matrixMod = baselineMod
	r.mathCompiled = baselineCompiled
	for _, binding := range r.cudaKernels() {
		fn, err := baselineMod.Function(binding.name)
		if err != nil {
			t.Fatal(err)
		}
		*binding.dst = fn
	}
	sourceHash := sha256.New()
	for _, name := range cudaMathFiles {
		data, err := cudaSourceFS.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		_, _ = sourceHash.Write([]byte(name + "\x00"))
		_, _ = sourceHash.Write(data)
	}
	t.Logf("baseline image=CUBIN bytes=%d image_sha256=%x source_sha256=%x", len(baselineImage), sha256.Sum256(baselineImage), sourceHash.Sum(nil))
	const m, k, experts, tokens, assign = 128, 256, 2, 2, 4
	gateRaw := cudaExpertFixture(16, k, m, experts, 17)
	upRaw := cudaExpertFixture(16, k, m, experts, 29)
	repack := func(raw []byte) cudamem.Slice[byte] {
		src := cudaExpertAllocBytes(t, r, len(raw))
		if err := src.CopyFromHost(raw); err != nil {
			t.Fatal(err)
		}
		dst := cudaExpertAllocBytes(t, r, experts*m*66)
		if err := r.repackIQ2(dst, src.Ptr(), experts*m); err != nil {
			t.Fatal(err)
		}
		return dst
	}
	gate, up := repack(gateRaw), repack(upRaw)
	floats := func(v []float32) cudamem.Slice[float32] {
		d, err := cudamem.Alloc[float32](r.device, len(v))
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := d.Free(); err != nil {
				t.Error(err)
			}
		})
		if err := d.CopyFromHost(v); err != nil {
			t.Fatal(err)
		}
		return d
	}
	x := make([]float32, tokens*k)
	for i := range x {
		x[i] = float32((i*37+i/17)%257-128) / 128
	}
	q := cudaExpertAllocBytes(t, r, (k/128)*tokens*144)
	if err := r.encodeQuantMMQ(floats(x), q, k, k, tokens); err != nil {
		t.Fatal(err)
	}
	qraw := make([]byte, q.Len())
	if err := q.CopyToHost(qraw); err != nil {
		t.Fatal(err)
	}
	decoded := make([]float32, len(x))
	for token := 0; token < tokens; token++ {
		for col := 0; col < k; col++ {
			off := ((col/128)*tokens + token) * 144
			scale := math.Float32frombits(binary.LittleEndian.Uint32(qraw[off+(col%128)/32*4:]))
			decoded[token*k+col] = scale * float32(int8(qraw[off+16+col%128]))
		}
	}
	ids := []int32{0, 1, 1, 0}
	slotInput := make([]float32, assign*k)
	for pair := 0; pair < assign; pair++ {
		copy(slotInput[pair*k:], decoded[(pair/2)*k:(pair/2+1)*k])
	}
	gref := cudaExpertReference(t, gateRaw, 16, k, m, ids, slotInput, true)
	uref := cudaExpertReference(t, upRaw, 16, k, m, ids, slotInput, true)
	weights := []float32{0.1, 0.2, 0.3, 0.4}
	want := make([]float32, assign*m)
	for pair := 0; pair < assign; pair++ {
		for row := 0; row < m; row++ {
			i := pair*m + row
			g, u := float64(gref[i]), float64(uref[i])
			want[i] = float32(g / (1 + math.Exp(-g)) * u * float64(weights[pair]))
		}
	}
	ints := func(v []int32) cudamem.Slice[int32] {
		d, err := cudamem.Alloc[int32](r.device, len(v))
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := d.Free(); err != nil {
				t.Error(err)
			}
		})
		if err := d.CopyFromHost(v); err != nil {
			t.Fatal(err)
		}
		return d
	}
	pairs := ints([]int32{3, 0, 2, 1})
	src := ints([]int32{1, 0, 1, 0})
	bounds := ints([]int32{0, 2, 4})
	work := ints([]int32{0, 1 << 16, 0})
	nitems := ints([]int32{2})
	route := floats(weights)
	launch := func(fn driver.Function, emit bool) ([]byte, []float32) {
		out := cudaExpertAllocBytes(t, r, (m/128)*assign*144)
		sentinel := make([]float32, assign*m+2)
		for i := range sentinel {
			sentinel[i] = -123.25
		}
		storage := floats(sentinel)
		args := []any{gate, up, q, src, pairs, bounds, route, work, nitems, out, int32(m), int32(k), int32(assign), int32(tokens), int32(experts), float32(0)}
		if emit {
			args = append(args, storage.Slice(1, 1+assign*m))
		}
		if err := r.device.WithContext(func() error {
			return cudalaunch.Kernel(fn).Grid(1, 3, 1).Block(32, 8, 1).On(r.stream()).Launch(args...)
		}); err != nil {
			t.Fatal(err)
		}
		if err := syncStream(r.stream()); err != nil {
			t.Fatal(err)
		}
		got := cudaTestRead(t, storage)
		if got[0] != -123.25 || got[len(got)-1] != -123.25 {
			t.Fatal("adjacent emission guard changed")
		}
		raw := make([]byte, out.Len())
		if err := out.CopyToHost(raw); err != nil {
			t.Fatal(err)
		}
		return raw, got[1 : len(got)-1]
	}
	if err := r.loadMathKernels(); err != nil {
		t.Fatal(err)
	}
	base, _ := launch(r.gateupD2RFn, false)
	gotQ, got := launch(r.gateupD2RMidFn, true)
	if !bytes.Equal(base, gotQ) {
		t.Fatal("optional emission changed consumer Q8 bytes")
	}
	const atol, rtol = 1e-4, 2e-3
	if err := cudaQualificationClose(got, want, atol, rtol); err != nil {
		t.Fatalf("prequant value oracle: %v", err)
	}
	// Change the exact emission site, not the comparator or a host snapshot.
	opts := cudaMathCompileOptions(r.major, r.minor)[len(cudaMathFiles):]
	mutantSourceHash := sha256.New()
	for _, name := range cudaMathFiles {
		data, err := cudaSourceFS.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		source := string(data)
		if name == "cuda_moe_d2r.cu" {
			old := "emitted_mid[(uint64_t)pair * M + cta_row0 + row] = s_smem.mid[col][row];"
			if strings.Count(source, old) != 1 {
				t.Fatal("same-site mutant target missing/ambiguous")
			}
			source = strings.Replace(source, old, strings.TrimSuffix(old, ";")+" + 1.0f;", 1)
		}
		opts = append(opts, nvrtcx.WithHeader(cudaSourceRoot+name, source))
		_, _ = mutantSourceHash.Write([]byte(name + "\x00"))
		_, _ = mutantSourceHash.Write([]byte(source))
	}
	compiled, err := nvrtcx.Compile(cudaMathCompilationSource(), opts...)
	if err != nil {
		t.Fatal(err)
	}
	mutantImage := compiled.CUBIN()
	if len(mutantImage) == 0 {
		t.Fatal("mutant CUBIN unavailable")
	}
	t.Logf("mutant image=CUBIN bytes=%d image_sha256=%x source_sha256=%x", len(mutantImage), sha256.Sum256(mutantImage), mutantSourceHash.Sum(nil))
	mod, err := compiled.LoadOutput(r.device, nvrtcx.OutputCUBIN)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := r.device.WithContext(func() error { return mod.Close() }); err != nil {
			t.Error(err)
		}
	}()
	fn, err := mod.Function("ds4_gateup_iq2_swiglu_q8_d2r_mid")
	if err != nil {
		t.Fatal(err)
	}
	mutantQ, mutant := launch(fn, true)
	if !bytes.Equal(base, mutantQ) {
		t.Fatal("emission-only mutant changed consumer bytes")
	}
	if err := cudaQualificationClose(mutant, want, atol, rtol); err == nil {
		t.Fatal("same-site device mutant escaped comparator")
	}
	t.Logf("%d emitted values; abs=%g rel=%g; original/optional Q8 byte equality; same-site mutant rejected; adjacent guards unchanged", len(got), atol, rtol)
}
