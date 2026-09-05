//go:build darwin && arm64

package main

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	libmetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestMetalHCIntegratedLayer13Regression(t *testing.T) {
	f := metalHCLayer13Load(t)
	outdir := os.Getenv("DS4_METAL_HC_LAYER13_OUTPUT_DIR")
	if outdir == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			t.Fatal(err)
		}
		outdir, err = os.MkdirTemp(filepath.Join(home, "tmp"), "metal-hc-layer13-")
		if err != nil {
			t.Fatal(err)
		}
	}
	if err := os.MkdirAll(outdir, 0755); err != nil {
		t.Fatal(err)
	}
	dev, err := libmetal.NewDevice()
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := dev.Close(); err != nil {
			t.Error(err)
		}
	}()
	ops, err := newMetalPinnedOps(dev)
	if err != nil {
		t.Fatal(err)
	}
	defer ops.Close()
	parallel := ops.hc
	scalar, err := newMetalHCScalarOps(dev)
	if err != nil {
		t.Fatal(err)
	}
	defer scalar.Close()
	arena := metalmem.NewArena(dev)
	defer arena.Free()
	tensor := func(name string, kind uint32, dims ...uint64) boundMetalTensor {
		return boundMetalTensor{name: name, typeID: kind, dimensions: dims, data: metalmem.UploadIn(arena, name, f.files[name])}
	}
	fn := tensor("hc-fn.bin", 1, 16384, 24)
	scale := tensor("hc-scale.bin", 0, 3)
	base := tensor("hc-base.bin", 0, 24)
	norm := tensor("attn-norm-weight.bin", 0, 4096)
	kv := tensor("kv-weight.bin", 8, 4096, 512)
	kvnorm := tensor("norm-weight.bin", 0, 512)
	sinks := boundMetalTensor{name: "zero sinks", typeID: 0, dimensions: []uint64{64}, data: metalmem.UploadIn(arena, "zero sinks", make([]byte, 256))}
	if err := arena.Err(); err != nil {
		t.Fatal(err)
	}
	t.Logf("candidate device=%s; retained oracle device=Apple M4 Max; artifacts=%s; HCnorm/mix/preweight oracle targets UNMEASURED", dev.Name(), outdir)
	state := f.floats(t, "hc_ffn_post.bin")
	stateGPU := metalmem.UploadIn(arena, "integration-state", state)
	outGPU := metalmem.AllocIn[float32](arena, "integration-out", 8192)
	if err := arena.Err(); err != nil {
		t.Fatal(err)
	}
	for _, buf := range []metalmem.Slice[float32]{parallel.mix, parallel.split, outGPU} {
		for i := range buf.HostSlice() {
			buf.HostSlice()[i] = float32(math.NaN())
		}
	}
	opening, err := newMetalBatch(ops)
	if err != nil {
		t.Fatal(err)
	}
	if err := opening.EncodeHCPre(stateGPU, fn, scale, base, norm, outGPU); err != nil {
		t.Fatal(err)
	}
	if err := opening.Wait(t.Context()); err != nil {
		t.Fatal(err)
	}
	if parallel.pending != nil {
		t.Fatal("completed pre left a reusable cross-batch split")
	}
	cur := append([]float32(nil), outGPU.HostSlice()[:4096]...)
	normalized := append([]float32(nil), outGPU.HostSlice()[4096:]...)
	files := make(map[string]string)
	anchored := false
	defer func() {
		receipt := map[string]any{"candidate_device": dev.Name(), "oracle_device": "Apple M4 Max", "fixture_manifest_sha256": metalHCLayer13Manifest,
			"candidate_intermediates_only": true, "oracle_intermediates": "UNMEASURED", "instrumentation_bitwise_anchor": anchored,
			"atol": metalLayer13ATol, "rtol": metalLayer13RTol, "files_sha256": files}
		b, err := json.MarshalIndent(receipt, "", "  ")
		if err == nil {
			err = os.WriteFile(filepath.Join(outdir, "capture.json"), append(b, '\n'), 0644)
		}
		if err != nil {
			t.Errorf("capture receipt: %v", err)
		}
	}()
	write := func(name string, v []float32) {
		t.Helper()
		b := make([]byte, 4*len(v))
		for i, x := range v {
			binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(x))
		}
		files[name+".bin"] = fmt.Sprintf("%x", sha256.Sum256(b))
		if err := os.WriteFile(filepath.Join(outdir, name+".bin"), b, 0644); err != nil {
			t.Fatal(err)
		}
	}
	write("production-attn-cur", cur)
	write("production-attn-norm", normalized)
	// Instrumentation must preserve both actual production outputs bitwise before
	// its added intermediate stores can be used even as candidate diagnostics.
	traced, trace := metalHCLayer13Trace(t, dev, state, fn, scale, base, norm)
	scalarCur, scalarNorm, err := scalar.HCPre(state, fn, scale, base, norm)
	if err != nil {
		t.Fatal(err)
	}
	actual := append(append([]float32(nil), scalarCur...), scalarNorm...)
	for i, v := range traced {
		if math.Float32bits(v) != math.Float32bits(actual[i]) {
			t.Fatalf("instrumentation changes production output %d: %g vs %g", i, v, actual[i])
		}
	}
	anchored = true
	write("parallel-mix24", parallel.mix.HostSlice())
	write("parallel-preweights", parallel.split.HostSlice()[:4])
	write("scalar-hc-normalized", trace[:16384])
	write("scalar-mix24", trace[16384:16408])
	write("scalar-preweights", trace[16408:])
	compare := func(name string, got []float32, wantName string) {
		t.Helper()
		want := f.floats(t, wantName)
		var maximum float64
		for i, v := range got {
			maximum = max(maximum, math.Abs(float64(v)-float64(want[i])))
		}
		t.Logf("%s max absolute delta=%g", name, maximum)
		if err := metalLayer13Close(got, want); err != nil {
			t.Errorf("%s: %v", name, err)
		}
	}
	compare("HC current", cur, "hc_attn_pre.bin")
	compare("HC normalized current", normalized, "attn_norm.bin")
	raw, err := ops.MatvecQ80(kv, normalized)
	if err != nil {
		t.Fatal(err)
	}
	write("production-KVraw", raw)
	compare("downstream KV raw", raw, "KVraw.bin")
	kn, err := ops.RMS(raw, kvnorm, 1, true)
	if err != nil {
		t.Fatal(err)
	}
	write("production-KVnorm", kn)
	compare("downstream KV norm", kn, "KVnorm.bin")
	_, kc, _, err := ops.RopeAttentionOne(make([]float32, 64*512), kn, sinks, 0)
	if err != nil {
		t.Fatal(err)
	}
	write("production-KVcur", kc)
	wantKV := f.floats(t, "KVcur.bin")
	compare("downstream KV stored", kc, "KVcur.bin")
	for i, v := range kc {
		if v != wantKV[i] {
			t.Errorf("exact downstream KVcur[%d]=%g want%g", i, v, wantKV[i])
			break
		}
	}
	t.Logf("KVnorm[362]=%.10g midpoint distance=%.10g; KVcur[362]=%.10g oracle=%.10g", kn[362], float64(kn[362])+.033203125, kc[362], wantKV[362])
	// Retain the old scalar route as a physical negative control of the SAME
	// full-vector exact KV gate. A tail-only discrepancy remains a failure.
	oldRaw, err := ops.MatvecQ80(kv, scalarNorm)
	if err != nil {
		t.Fatal(err)
	}
	oldNorm, err := ops.RMS(oldRaw, kvnorm, 1, true)
	if err != nil {
		t.Fatal(err)
	}
	_, oldKV, _, err := ops.RopeAttentionOne(make([]float32, 64*512), oldNorm, sinks, 0)
	if err != nil {
		t.Fatal(err)
	}
	write("scalar-KVcur", oldKV)
	if len(oldKV) != 512 {
		t.Fatalf("old scalar KV length %d, want 512", len(oldKV))
	}
	changed := false
	for i, v := range oldKV {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("old scalar KV element %d is non-finite", i)
		}
		if v != wantKV[i] {
			changed = true
		}
	}
	if !changed {
		t.Error("old scalar route no longer fails the frozen exact downstream gate; negative control unavailable")
	}
	// The post arithmetic receives a bounded synthetic block. This is
	// independent float64 validation of expansion, not a missing oracle stage.
	split := append([]float32(nil), parallel.split.HostSlice()...)
	block := make([]float32, 4096)
	for i := range block {
		block[i] = float32(i%31-15) / 16
	}
	for i := range parallel.mix.HostSlice() {
		parallel.mix.HostSlice()[i] = float32(math.NaN())
	}
	blockGPU := metalmem.UploadIn(arena, "integration-block", block)
	postGPU := metalmem.AllocIn[float32](arena, "integration-post", 16384)
	if err := arena.Err(); err != nil {
		t.Fatal(err)
	}
	for i := range postGPU.HostSlice() {
		postGPU.HostSlice()[i] = float32(math.NaN())
	}
	closing, err := newMetalBatch(ops)
	if err != nil {
		t.Fatal(err)
	}
	if err := closing.EncodeHCPost(stateGPU, blockGPU, fn, scale, base, postGPU); err != nil {
		t.Fatal(err)
	}
	if err := closing.Wait(t.Context()); err != nil {
		t.Fatal(err)
	}
	post := append([]float32(nil), postGPU.HostSlice()...)
	expected := make([]float32, 16384)
	for dst := range 4 {
		for d, v := range block {
			sum := float64(v) * float64(split[4+dst])
			for src := range 4 {
				sum += float64(state[src*4096+d]) * float64(split[8+dst+4*src])
			}
			expected[dst*4096+d] = float32(sum)
		}
	}
	if err := metalLayer13Close(post, expected); err != nil {
		t.Errorf("cached post mathematical reference: %v", err)
	}
	if parallel.pending != nil {
		t.Fatal("post left a reusable split")
	}
	fallback, err := ops.HCPost(state, block, fn, scale, base)
	if err != nil {
		t.Fatal(err)
	}
	if err := metalLayer13Close(fallback, post); err != nil {
		t.Errorf("standalone post differs from cached post: %v", err)
	}
	// A different base tensor must trigger the standalone projection/split path.
	changedBase := append([]byte(nil), f.files["hc-base.bin"]...)
	old := math.Float32frombits(binary.LittleEndian.Uint32(changedBase[16:]))
	binary.LittleEndian.PutUint32(changedBase[16:], math.Float32bits(old+1))
	base2 := base
	base2.data = metalmem.UploadIn(arena, "changed-base", changedBase)
	opening, err = newMetalBatch(ops)
	if err != nil {
		t.Fatal(err)
	}
	if err := opening.EncodeHCPre(stateGPU, fn, scale, base, norm, outGPU); err != nil {
		t.Fatal(err)
	}
	if err := opening.Wait(t.Context()); err != nil {
		t.Fatal(err)
	}
	closing, err = newMetalBatch(ops)
	if err != nil {
		t.Fatal(err)
	}
	if err := closing.EncodeHCPost(stateGPU, blockGPU, fn, scale, base2, postGPU); err != nil {
		t.Fatal(err)
	}
	if err := closing.Wait(t.Context()); err != nil {
		t.Fatal(err)
	}
	changedPost := append([]float32(nil), postGPU.HostSlice()...)
	fresh, err := ops.HCPost(state, block, fn, scale, base2)
	if err != nil {
		t.Fatal(err)
	}
	if err := metalLayer13Close(changedPost, fresh); err != nil {
		t.Errorf("wrong-base cache reuse: %v", err)
	}
	if metalLayer13Close(changedPost, post) == nil {
		t.Error("base mutation does not expose stale split reuse")
	}

	// Reproducer: A pre completes; its post is encoded but not submitted;
	// B pre with another base completes before A post is submitted. A post
	// must rebuild A's split in its own command buffer, not read B's split.
	for _, mutant := range []bool{false, true} {
		aPre, err := newMetalBatch(ops)
		if err != nil {
			t.Fatal(err)
		}
		if err := aPre.EncodeHCPre(stateGPU, fn, scale, base, norm, outGPU); err != nil {
			t.Fatal(err)
		}
		if err := aPre.Wait(t.Context()); err != nil {
			t.Fatal(err)
		}
		aPost, err := newMetalBatch(ops)
		if err != nil {
			t.Fatal(err)
		}
		if !mutant {
			if err := aPost.EncodeHCPost(stateGPU, blockGPU, fn, scale, base, postGPU); err != nil {
				t.Fatal(err)
			}
		} else {
			// Deliberately recreate the rejected cross-batch shared-split read.
			enc := aPost.buffer.ComputeCommandEncoder()
			enc.SetComputePipelineState(parallel.kernels.Post)
			for i, v := range []metalmem.Slice[float32]{stateGPU, blockGPU, parallel.split, postGPU} {
				enc.SetBufferWithOffsetAtIndex(v.Buffer(), v.Offset(), uint(i))
			}
			enc.DispatchThreadsThreadsPerThreadgroup(libmetal.Threads(4096), libmetal.Threads(256))
			enc.EndEncoding()
		}
		bPre, err := newMetalBatch(ops)
		if err != nil {
			t.Fatal(err)
		}
		if err := bPre.EncodeHCPre(stateGPU, fn, scale, base2, norm, outGPU); err != nil {
			t.Fatal(err)
		}
		if err := bPre.Wait(t.Context()); err != nil {
			t.Fatal(err)
		}
		for i := range postGPU.HostSlice() {
			postGPU.HostSlice()[i] = float32(math.NaN())
		}
		if err := aPost.Wait(t.Context()); err != nil {
			t.Fatal(err)
		}
		got := postGPU.HostSlice()
		for i, v := range got {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("interleaved post element%d non-finite", i)
			}
		}
		err = metalLayer13Close(got, post)
		if mutant {
			if err == nil {
				t.Error("stale-split mutant passed A reference")
			}
			if err := metalLayer13Close(got, fresh); err != nil {
				t.Errorf("stale-split mutant did not read B split: %v", err)
			}
		} else if err != nil {
			t.Errorf("interleaved batch split overwrite: %v", err)
		}
	}
	// Same-buffer pre/post remains an ordered cache hit with no interleaving.
	paired, err := newMetalBatch(ops)
	if err != nil {
		t.Fatal(err)
	}
	if err := paired.EncodeHCPre(stateGPU, fn, scale, base, norm, outGPU); err != nil {
		t.Fatal(err)
	}
	if err := paired.EncodeHCPost(stateGPU, blockGPU, fn, scale, base, postGPU); err != nil {
		t.Fatal(err)
	}
	if err := paired.Wait(t.Context()); err != nil {
		t.Fatal(err)
	}
	if err := metalLayer13Close(postGPU.HostSlice(), post); err != nil {
		t.Errorf("same-buffer cached post: %v", err)
	}

}
