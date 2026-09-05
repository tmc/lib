//go:build darwin && arm64

package main

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	apple "github.com/tmc/apple/metal"
	"github.com/tmc/apple/objectivec"
	libmetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
	"github.com/tmc/lib/apple/x/metal/metalx"
)

const metalHCLayer13Manifest = "46e305624d7971cee16ea759d6f8a63e1fc5b731acd419d9e3f1bc4b5d568e88"

// The original retained oracle ran on M4 Max with hc_stable=on,
// norm_unify=on, math_safe=off, tensor_matmul=off. Pinned ds4.c:22960-23054
// admits the pre-M5 fused HC producer, falling back to fused norm/mix if the
// compound pipeline is unavailable (ds4_metal.m:44499-44510). Both reproduce
// the 1024-thread RMS and eight-simdgroup F16 projection rather than the local
// serial HC kernel. Fused sigmoid sites intentionally retain exp, despite
// hc_stable (metal/dsv4_hc.metal:86-112). No pipeline trace was retained, so the
// precise compound-versus-fallback dispatch is not asserted here.
//
// These artifacts contain actual incoming HC and outgoing current/norm values.
// Oracle HC-normalized input, mix24, and preweights were NOT retained. The
// corresponding candidate captures below are diagnostics, not oracle targets.
func metalHCLayer13Load(t *testing.T) metalLayer13Fixture {
	t.Helper()
	dir := os.Getenv("DS4_METAL_HC_LAYER13_FIXTURE_DIR")
	if dir == "" {
		t.Skip("DS4_METAL_HC_LAYER13_FIXTURE_DIR is not set")
	}
	raw, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		t.Fatal(err)
	}
	if fmt.Sprintf("%x", sha256.Sum256(raw)) != metalHCLayer13Manifest {
		t.Fatal("HC layer13 manifest digest mismatch")
	}
	var m struct {
		Files map[string]struct {
			Bytes int    `json:"bytes"`
			Hash  string `json:"sha256"`
		} `json:"files"`
	}
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatal(err)
	}
	f := metalLayer13Fixture{files: make(map[string][]byte)}
	for name, size := range map[string]int{"hc-fn.bin": 786432, "hc-scale.bin": 12, "hc-base.bin": 96, "attn-norm-weight.bin": 16384, "hc_ffn_post.bin": 65536, "hc_attn_pre.bin": 16384, "attn_norm.bin": 16384, "kv-weight.bin": 2228224, "norm-weight.bin": 2048, "KVraw.bin": 2048, "KVnorm.bin": 2048, "KVcur.bin": 2048} {
		d, ok := m.Files[name]
		if !ok || d.Bytes != size {
			t.Fatalf("invalid descriptor %s", name)
		}
		b, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			t.Fatal(err)
		}
		if len(b) != size || fmt.Sprintf("%x", sha256.Sum256(b)) != d.Hash {
			t.Fatalf("invalid bytes %s", name)
		}
		f.files[name] = b
	}
	return f
}

func TestMetalHCLayer13Fixture(t *testing.T) {
	f := metalHCLayer13Load(t)
	for _, name := range []string{"hc_ffn_post.bin", "hc_attn_pre.bin", "attn_norm.bin", "KVraw.bin", "KVnorm.bin", "KVcur.bin"} {
		f.floats(t, name)
	}
	want := f.floats(t, "attn_norm.bin")
	wrong := append([]float32(nil), want...)
	wrong[0] += 1
	if metalLayer13Close(wrong, want) == nil {
		t.Fatal("strict comparator accepts injected output error")
	}
}

// Historical scalar control, kept explicitly after the parallel integration.
// This runs only bounded device allocations. It never opens or maps the model.
// Keep the established layer13 1e-4 absolute/relative gate and exact FP8 gate;
// a changed KV cell is a failure even if upstream ordinary tolerances pass.
func TestMetalHCLayer13Regression(t *testing.T) {
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
	t.Logf("historical scalar device=%s; retained oracle device=Apple M4 Max; artifacts=%s; HCnorm/mix/preweight oracle targets UNMEASURED", dev.Name(), outdir)
	state := f.floats(t, "hc_ffn_post.bin")
	cur, normalized, err := scalar.HCPre(state, fn, scale, base, norm)
	if err != nil {
		t.Fatal(err)
	}
	files := make(map[string]string)
	anchored := false
	defer func() {
		receipt := map[string]any{"candidate_device": dev.Name(), "oracle_device": "Apple M4 Max", "fixture_manifest_sha256": metalHCLayer13Manifest,
			"candidate_intermediates_only": true, "route": "historical scalar control", "oracle_intermediates": "UNMEASURED", "instrumentation_bitwise_anchor": anchored,
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
	actual := append(append([]float32(nil), cur...), normalized...)
	for i, v := range traced {
		if math.Float32bits(v) != math.Float32bits(actual[i]) {
			t.Fatalf("instrumentation changes production output %d: %g vs %g", i, v, actual[i])
		}
	}
	anchored = true
	write("candidate-hc-normalized", trace[:16384])
	write("candidate-mix24", trace[16384:16408])
	write("candidate-preweights", trace[16408:])
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
}

// Append diagnostic stores to the exact historical scalar HC text. This is not a
// reference implementation of the pinned parallel Metal HC path.
func metalHCLayer13Trace(t *testing.T, dev *libmetal.Device, state []float32, weights ...boundMetalTensor) ([]float32, []float32) {
	t.Helper()
	start := strings.Index(metalHCScalarMSL, "kernel void hc_pre(")
	end := strings.Index(metalHCScalarMSL, "\nkernel void hc_post(")
	if start < 0 || end <= start {
		t.Fatal("historical HC source anchors missing")
	}
	source := metalHCScalarMSL[start:end]
	source = strings.Replace(source, "kernel void hc_pre(", "kernel void hc_pre_trace(", 1)
	before := "device float *out [[buffer(5)]])"
	if strings.Count(source, before) != 1 {
		t.Fatal("historical HC signature changed")
	}
	source = strings.Replace(source, before, "device float *out [[buffer(5)]], device float *trace [[buffer(6)]])", 1)
	last := strings.LastIndex(source, "}")
	source = source[:last] + "for(uint i=0;i<16384;i++)trace[i]=state[i]*inv; for(uint i=0;i<24;i++)trace[16384+i]=mix[i]; for(uint i=0;i<4;i++)trace[16408+i]=pre[i];\n" + source[last:]
	library, err := dev.CompileLibrary(metalPinnedMSL + source)
	if err != nil {
		t.Fatal(err)
	}
	defer objectivec.ObjectFromID(library.GetID()).Release()
	var kernels struct {
		Trace apple.MTLComputePipelineState `metal:"hc_pre_trace"`
	}
	if err := metalx.Bind(dev, library, &kernels); err != nil {
		t.Fatal(err)
	}
	defer objectivec.ObjectFromID(kernels.Trace.GetID()).Release()
	in, err := metalmem.AllocFromSlice(dev, state)
	if err != nil {
		t.Fatal(err)
	}
	defer in.Free()
	output, err := metalmem.Alloc[float32](dev, 8192)
	if err != nil {
		t.Fatal(err)
	}
	defer output.Free()
	trace, err := metalmem.Alloc[float32](dev, 16412)
	if err != nil {
		t.Fatal(err)
	}
	defer trace.Free()
	for i := range output.HostSlice() {
		output.HostSlice()[i] = float32(math.NaN())
	}
	for i := range trace.HostSlice() {
		trace.HostSlice()[i] = float32(math.NaN())
	}
	cb := dev.Queue().CommandBuffer()
	enc := cb.ComputeCommandEncoder()
	enc.SetComputePipelineState(kernels.Trace)
	enc.SetBufferWithOffsetAtIndex(in.Buffer(), in.Offset(), 0)
	for i, w := range weights {
		enc.SetBufferWithOffsetAtIndex(w.data.Buffer(), w.data.Offset(), uint(i+1))
	}
	enc.SetBufferWithOffsetAtIndex(output.Buffer(), output.Offset(), 5)
	enc.SetBufferWithOffsetAtIndex(trace.Buffer(), trace.Offset(), 6)
	enc.DispatchThreadsThreadsPerThreadgroup(libmetal.Threads(1), libmetal.Threads(1))
	enc.EndEncoding()
	cb.Commit()
	if err := libmetal.WaitCommandBuffer(context.Background(), cb); err != nil {
		t.Fatal(err)
	}
	for i, v := range trace.HostSlice() {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("invalid candidate trace %d", i)
		}
	}
	return append([]float32(nil), output.HostSlice()...), append([]float32(nil), trace.HostSlice()...)
}
