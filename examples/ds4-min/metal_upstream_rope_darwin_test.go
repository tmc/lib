//go:build darwin && arm64

package main

import (
	"bytes"
	"crypto/sha1"
	"crypto/sha256"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"unsafe"

	"github.com/tmc/apple/foundation"
	apple "github.com/tmc/apple/metal"
	"github.com/tmc/apple/objectivec"
	libmetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
)

// The upstream compatibility gate compares the production inverse RoPE with
// upstream ds4 11e8cad's own Metal kernel_dsv4_rope_tail_f32, run on this
// device on the same pre-inverse heads. The oracle is the full upstream
// library, assembled as ds4_gpu_full_source assembles it from the pinned
// files and compiled with the options ds4_metal.m uses on a device without
// the Metal 4 tensor API. A pass establishes compatibility with that shader
// and compile profile only, not exact arithmetic: the CPU discrepancy
// TestMetalMixedAttentionCapacity reports is logged here, not waived.

const upstreamRopeCommit = "11e8cad4473716a41baf4d62e92ed9764a00fce6"

// upstreamRopeFiles is ds4_gpu_full_source's required_sources order with the
// pinned git blob id of each file.
var upstreamRopeFiles = []struct{ path, blob string }{
	{"metal/flash_attn.metal", "e3232a72d83188002bfa1230bf7a5a3c14f0d4fd"},
	{"metal/dense.metal", "1d54f6e371dc7e2605f3d0702bc4f565daa5d2f9"},
	{"metal/glm53_bf16.metal", "a46dd81cd8e2325f1617feac5eac367e81516470"},
	{"metal/glm53_vision.metal", "9d5ed37192cd0b90d04bc6a62a806ebcc02b0c04"},
	{"metal/deepseek4_vision.metal", "76992cdd29e44b6059d0c089b74de0d6575f580c"},
	{"metal/glm53_kda.metal", "c21d30ecf5059543c153d666c5c366a7a01342e9"},
	{"metal/moe.metal", "b24e27a90d6291098ded083faf99f719727600c1"},
	{"metal/dsv4_hc.metal", "c161a03a8b618751b4f6f6f35fba142bd4580cf1"},
	{"metal/unary.metal", "717fb350b1b0f6981173f3b24662d177b906a8f4"},
	{"metal/dsv4_kv.metal", "dd602ce3cfe25a2bb0ce5a22cd48e7cbb69d8bbb"},
	{"metal/dsv41.metal", "03afd310ddbd41e72e52d55fabaf9646ebd5a507"},
	{"metal/dsv4_rope.metal", "279365c8f15464429a916e3978b5358b5b799c6c"},
	{"metal/dsv4_misc.metal", "20069d26327f5eecdb3f5ed1f747609a50e6749d"},
	{"metal/argsort.metal", "276cce621b06548658ac8ab2ecc5516adc48735a"},
	{"metal/cpy.metal", "feeb9230172dec40c747a563e3a9050eccf46949"},
	{"metal/concat.metal", "bfe957fc5b69e260e7fbf29990fa1004ea6c14e1"},
	{"metal/get_rows.metal", "836225f6c03000f0ddf3abd5dd92c9012f2b69e2"},
	{"metal/sum_rows.metal", "8efea9b39588f8766ce2ec50f5b67ce80a5e225c"},
	{"metal/softmax.metal", "7634705c63cac0b84bcc2396dd6a4f95b344d5b0"},
	{"metal/repeat.metal", "98f68e9afe86c0bb1fad6c6ae4758ebb57c0f70f"},
	{"metal/glu.metal", "fd5c6fd0967ec046eaed1ba8914b777cf11e3da3"},
	{"metal/norm.metal", "db7c57600842e5bdf3034cc00cf5a9e492fc447f"},
	{"metal/bin.metal", "ca7def1dd895374636185ec4ae8560b34f06a77b"},
	{"metal/set_rows.metal", "d3df6e0ba4808a7a4f2f775db2ea7419d5178a75"},
}

const upstreamRopeMetalM = "b61cbee8d7119c5eb3911b2f3d222f1271718ecd"

// upstreamRopeArgs mirrors ds4_gpu_rope_tail_batch_args, 144 bytes.
type upstreamRopeArgs struct {
	ne00, ne01, ne02, ne03                     int64
	nb00, nb01, nb02, nb03, nb0, nb1, nb2, nb3 uint64
	nDims, mode, nCtxOrig, inverse             int32
	freqBase, freqScale, extFactor, attnFactor float32
	betaFast, betaSlow                         float32
	src2                                       bool
}

func upstreamRopeBlob(t *testing.T, repo, path, blob string) string {
	t.Helper()
	data, err := exec.Command("git", "-C", repo, "show", upstreamRopeCommit+":"+path).Output()
	if err != nil {
		t.Fatalf("read upstream %s: %v", path, err)
	}
	h := sha1.New()
	fmt.Fprintf(h, "blob %d\x00", len(data))
	h.Write(data)
	if got := fmt.Sprintf("%x", h.Sum(nil)); got != blob {
		t.Fatalf("upstream %s blob %s, want %s", path, got, blob)
	}
	return string(data)
}

// upstreamRopePrelude decodes ds4_metal.m's ds4_gpu_source string literal.
func upstreamRopePrelude(t *testing.T, metalM string) string {
	t.Helper()
	const head = "static const char *ds4_gpu_source =\n"
	if strings.Count(metalM, head) != 1 {
		t.Fatal("ds4_gpu_source declaration not unique")
	}
	var b strings.Builder
	for _, line := range strings.Split(metalM[strings.Index(metalM, head)+len(head):], "\n") {
		last := strings.HasSuffix(line, ";")
		s, err := strconv.Unquote(strings.TrimSpace(strings.TrimSuffix(line, ";")))
		if err != nil {
			t.Fatalf("ds4_gpu_source literal %q: %v", line, err)
		}
		b.WriteString(s)
		if last {
			return b.String()
		}
	}
	t.Fatal("ds4_gpu_source literal unterminated")
	return ""
}

// upstreamRopeSource returns the pinned upstream library source, with the
// named oracle mutation applied to dsv4_rope.metal after verification.
func upstreamRopeSource(t *testing.T, repo, mutant string) string {
	t.Helper()
	var b strings.Builder
	b.WriteString(upstreamRopePrelude(t, upstreamRopeBlob(t, repo, "ds4_metal.m", upstreamRopeMetalM)))
	for _, f := range upstreamRopeFiles {
		text := upstreamRopeBlob(t, repo, f.path, f.blob)
		if f.path == "metal/dsv4_rope.metal" && mutant != "" {
			const anchor = "mscale *= 1.0f + 0.1f * log(1.0f / freq_scale);"
			if mutant != "mscale_0.2" {
				t.Fatalf("unknown oracle mutant %q", mutant)
			}
			if strings.Count(text, anchor) != 1 {
				t.Fatal("oracle mutant anchor changed")
			}
			text = strings.Replace(text, anchor, "mscale *= 1.0f + 0.2f * log(1.0f / freq_scale);", 1)
		}
		fmt.Fprintf(&b, "\n// appended %s\n%s\n", f.path, text)
	}
	return b.String()
}

func upstreamRopeOptions(t *testing.T) apple.MTLCompileOptions {
	t.Helper()
	opts := apple.NewMTLCompileOptions()
	keys := []objectivec.IObject{foundation.NewStringWithString("DS4_METAL_HC_STABLE"), foundation.NewStringWithString("DS4_METAL_NORM_RSQRT_DISABLE")}
	values := []objectivec.IObject{foundation.NewStringWithString("1"), foundation.NewStringWithString("1")}
	opts.SetPreprocessorMacros(foundation.NewDictionaryWithObjectsForKeys(values, keys))
	t.Logf("upstream_profile mathMode=%v fpFunctions=%v fastMathEnabled=%t languageVersion=%v macros=DS4_METAL_HC_STABLE=1,DS4_METAL_NORM_RSQRT_DISABLE=1",
		opts.MathMode(), opts.MathFloatingPointFunctions(), opts.FastMathEnabled(), opts.LanguageVersion())
	return opts
}

// upstreamRopePipeline compiles its own library and returns the
// kernel_dsv4_rope_tail_f32 pipeline, logging its identity.
func upstreamRopePipeline(t *testing.T, dev *libmetal.Device, source string, opts apple.MTLCompileOptions, name string) apple.MTLComputePipelineState {
	t.Helper()
	lib, err := dev.Raw().NewLibraryWithSourceOptionsError(source, opts)
	if err != nil {
		t.Fatalf("compile upstream library %s: %v", name, err)
	}
	defer objectivec.ObjectFromID(lib.GetID()).Release()
	fn := lib.NewFunctionWithName("kernel_dsv4_rope_tail_f32")
	if fn == nil || fn.GetID() == 0 {
		t.Fatalf("upstream %s: kernel_dsv4_rope_tail_f32 missing", name)
	}
	defer objectivec.ObjectFromID(fn.GetID()).Release()
	pipeline, err := dev.Raw().NewComputePipelineStateWithFunctionError(fn)
	if err != nil {
		t.Fatalf("upstream %s pipeline: %v", name, err)
	}
	t.Cleanup(func() { objectivec.ObjectFromID(pipeline.GetID()).Release() })
	if pipeline.MaxTotalThreadsPerThreadgroup() < 256 {
		t.Fatalf("upstream %s pipeline max threads %d < 256", name, pipeline.MaxTotalThreadsPerThreadgroup())
	}
	archive, err := dev.Raw().NewBinaryArchiveWithDescriptorError(apple.NewMTLBinaryArchiveDescriptor())
	if err != nil {
		t.Fatalf("upstream %s binary archive: %v", name, err)
	}
	defer objectivec.ObjectFromID(archive.GetID()).Release()
	desc := apple.NewMTLComputePipelineDescriptor()
	defer objectivec.ObjectFromID(desc.GetID()).Release()
	desc.SetComputeFunction(fn)
	if _, err := archive.AddComputePipelineFunctionsWithDescriptorError(desc); err != nil {
		t.Fatalf("upstream %s binary archive add: %v", name, err)
	}
	path := filepath.Join(t.TempDir(), name+".metallib")
	if _, err := archive.SerializeToURLError(foundation.NewURLFileURLWithPath(path)); err != nil {
		t.Fatalf("upstream %s binary archive serialize: %v", name, err)
	}
	binary, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("upstream_identity library=%s threadExecutionWidth=%d maxTotalThreadsPerThreadgroup=%d staticThreadgroupMemory=%d binary_archive_sha256=%x binary_archive_bytes=%d",
		name, pipeline.ThreadExecutionWidth(), pipeline.MaxTotalThreadsPerThreadgroup(), pipeline.StaticThreadgroupMemoryLength(), sha256.Sum256(binary), len(binary))
	return pipeline
}

// upstreamRopeRun applies the oracle in place to a copy of heads, launched
// as ds4_gpu_encode_rope_tail_inplace launches g_rope_tail_batch_pipeline.
func upstreamRopeRun(t *testing.T, ops *metalPinnedOps, pipeline apple.MTLComputePipelineState, heads []float32, position int) []float32 {
	t.Helper()
	arena := metalmem.NewArena(ops.dev)
	defer arena.Free()
	x := metalmem.UploadIn(arena, "heads", heads)
	if err := arena.Err(); err != nil {
		t.Fatal(err)
	}
	if err := rawCapacityExact(x.HostSlice(), heads); err != nil {
		t.Fatalf("oracle operand: %v", err)
	}
	freqScale := float32(1) / 16
	args := upstreamRopeArgs{
		ne00: 512, ne01: 64, ne02: 1, ne03: 1,
		nb00: 4, nb01: 2048, nb02: 64 * 2048, nb03: 64 * 2048,
		nb0: 4, nb1: 2048, nb2: 64 * 2048, nb3: 64 * 2048,
		nDims: 64, mode: 0, nCtxOrig: 65536, inverse: 1,
		freqBase: 160000, freqScale: freqScale, extFactor: 1,
		attnFactor: 1 / (1 + float32(0.1)*float32(math.Log(float64(1/freqScale)))),
		betaFast:   32, betaSlow: 1,
	}
	if position == 0 {
		t.Logf("oracle_args attn_factor_bits=%08x", math.Float32bits(args.attnFactor))
	}
	if unsafe.Sizeof(args) != 144 {
		t.Fatalf("oracle args size %d", unsafe.Sizeof(args))
	}
	pos := int32(position)
	batch, err := newMetalBatch(ops)
	if err != nil {
		t.Fatal(err)
	}
	enc := batch.encoder()
	enc.SetComputePipelineState(pipeline)
	enc.SetBytesLengthAtIndex(unsafe.Slice((*byte)(unsafe.Pointer(&args)), unsafe.Sizeof(args)), 0)
	enc.SetBufferWithOffsetAtIndex(x.Buffer(), x.Offset(), 1)
	enc.SetBytesLengthAtIndex(unsafe.Slice((*byte)(unsafe.Pointer(&pos)), 4), 2)
	enc.SetBufferWithOffsetAtIndex(x.Buffer(), x.Offset(), 3)
	enc.SetBufferWithOffsetAtIndex(x.Buffer(), x.Offset(), 4)
	enc.DispatchThreadgroupsThreadsPerThreadgroup(apple.MTLSize{Width: 64, Height: 1, Depth: 1}, apple.MTLSize{Width: 256, Height: 1, Depth: 1})
	batch.encoded++
	if err := batch.Wait(t.Context()); err != nil {
		t.Fatal(err)
	}
	return append([]float32(nil), x.HostSlice()...)
}

// upstreamRopeHeads compares 512-wide heads under mixedCapacityClose, the
// unrotated body [0,448) of every head before the rotary tail [448,512).
func upstreamRopeHeads(got, want []float32) error {
	if len(got) != len(want) || len(want)%512 != 0 {
		return fmt.Errorf("heads length %d, want %d", len(got), len(want))
	}
	for _, part := range []struct {
		name   string
		lo, hi int
	}{{"heads_body", 0, 448}, {"heads_tail", 448, 512}} {
		for head := 0; head < len(want); head += 512 {
			if err := mixedCapacityClose(got[head+part.lo:head+part.hi], want[head+part.lo:head+part.hi]); err != nil {
				return fmt.Errorf("%s head %d: %v", part.name, head/512, err)
			}
		}
	}
	return nil
}

func upstreamRopeOver(got, want []float32) (over int, maxAbs float64) {
	for i := range got {
		d := math.Abs(float64(got[i] - want[i]))
		maxAbs = max(maxAbs, d)
		if d > metalMixedCapacityATol+metalMixedCapacityRTol*math.Abs(float64(want[i])) {
			over++
		}
	}
	return
}

func f32Digest(values []float32) string {
	return fmt.Sprintf("%x", sha256.Sum256(unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(values))), 4*len(values))))
}

func TestMetalUpstreamRopeCompat(t *testing.T) {
	if os.Getenv("DS4_METAL_UPSTREAM_ROPE_GATE") != "1" {
		t.Skip("opt-in upstream Metal RoPE compatibility gate")
	}
	if os.Getenv("GPU_LOCK_HELD") == "" {
		t.Fatal("gpu-lock required")
	}
	repo := os.Getenv("DS41_ORACLE")
	if repo == "" {
		t.Fatal("DS41_ORACLE required")
	}
	mutant := os.Getenv("DS4_METAL_UPSTREAM_ROPE_ORACLE_MUTANT")
	source := upstreamRopeSource(t, repo, mutant)
	t.Logf("upstream_source commit=%s oracle_mutant=%q assembled_sha256=%x bytes=%d production_source_sha256=%x",
		upstreamRopeCommit, mutant, sha256.Sum256([]byte(source)), len(source), sha256.Sum256([]byte(metalPinnedMSL)))

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
	osVersion, _ := syscall.Sysctl("kern.osversion")
	arch := ""
	if a := dev.Raw().Architecture(); a != nil && a.GetID() != 0 {
		arch = a.Name()
	}
	t.Logf("device name=%q registryID=%#x architecture=%q kern.osversion=%s", dev.Name(), dev.Raw().RegistryID(), arch, osVersion)

	opts := upstreamRopeOptions(t)
	defer objectivec.ObjectFromID(opts.GetID()).Release()
	oracle := [2]apple.MTLComputePipelineState{
		upstreamRopePipeline(t, dev, source, opts, "oracle_1"),
		upstreamRopePipeline(t, dev, source, opts, "oracle_2"),
	}

	const noop = `
kernel void upstream_rope_no_unrotate(device float *out [[buffer(0)]],
  constant uint &position [[buffer(1)]], uint head [[thread_position_in_threadgroup]]) {
  (void)out; (void)position; (void)head;
}
`
	library, err := dev.CompileLibrary(metalPinnedMSL + noop)
	if err != nil {
		t.Fatal(err)
	}
	defer objectivec.ObjectFromID(library.GetID()).Release()
	split, err := dev.NewComputePipeline(library, "upstream_rope_no_unrotate")
	if err != nil {
		t.Fatal(err)
	}
	defer objectivec.ObjectFromID(split.GetID()).Release()

	for _, position := range []int{0, 127, 128, 129, 2048} {
		for _, count := range []int{0, 1, 512, 513} {
			if !t.Run(fmt.Sprintf("position_%d/compressed_%d", position, count), func(t *testing.T) {
				upstreamRopeCase(t, ops, split, oracle, position, count)
			}) {
				return
			}
		}
	}
}

func upstreamRopeCase(t *testing.T, ops *metalPinnedOps, split apple.MTLComputePipelineState, oracle [2]apple.MTLComputePipelineState, position, count int) {
	query, current, sinks := mixedCapacityInputs(position)
	currentRotated := append([]float32(nil), current...)
	mixedCapacityRotate(currentRotated, position, false)
	history := rawCapacityHistory(position, 128)
	cache := make([]float32, 128*512)
	for i := range cache {
		cache[i] = 13
	}
	for i, row := range history {
		copy(cache[((position-len(history)+i)%128)*512:], row)
	}
	compressed, mask := mixedCapacityRows(count)
	flat := make([]float32, max(1, count)*512)
	for i, row := range compressed {
		copy(flat[i*512:], row)
	}
	if count == 0 {
		mask = []uint32{0}
	}
	production := ops.kernels.RopeOutUnrotate
	defer func() { ops.kernels.RopeOutUnrotate = production }()
	arena := metalmem.NewArena(ops.dev)
	defer arena.Free()
	q := metalmem.UploadIn(arena, "query", query)
	kv := metalmem.UploadIn(arena, "current", current)
	comp := metalmem.UploadIn(arena, "compressed", flat)
	allowed := metalmem.UploadIn(arena, "allowed", mask)
	sink := testBatchF32Tensor(arena, "sinks", sinks)
	run := func(unrotate apple.MTLComputePipelineState) metalmem.Slice[float32] {
		raw := metalmem.UploadIn(arena, "raw", cache)
		scores := metalmem.AllocIn[float32](arena, "scores", 64*(128+count))
		out := metalmem.AllocIn[float32](arena, "out", 32768+512+32768)
		if err := arena.Err(); err != nil {
			t.Fatal(err)
		}
		for _, values := range [][]float32{scores.HostSlice(), out.HostSlice()} {
			for i := range values {
				values[i] = float32(math.NaN())
			}
		}
		ops.kernels.RopeOutUnrotate = unrotate
		batch, err := newMetalBatch(ops)
		if err != nil {
			t.Fatal(err)
		}
		if err := batch.EncodeRopeAttentionMixed(q, kv, sink, raw, comp, count, allowed, scores, out, position, 2); err != nil {
			t.Fatal(err)
		}
		if err := batch.Wait(t.Context()); err != nil {
			t.Fatal(err)
		}
		ops.kernels.RopeOutUnrotate = production
		return out
	}
	full := append([]float32(nil), run(production).HostSlice()...)
	saved := run(split)
	heads := append([]float32(nil), saved.HostSlice()[33280:]...)

	batch, err := newMetalBatch(ops)
	if err != nil {
		t.Fatal(err)
	}
	enc := batch.encoder()
	enc.SetComputePipelineState(production)
	enc.SetBufferWithOffsetAtIndex(saved.Buffer(), saved.Offset(), 0)
	libmetal.SetScalar(enc, uint32(position), 1)
	enc.DispatchThreadgroupsThreadsPerThreadgroup(libmetal.Threads(1), libmetal.Threads(64))
	batch.encoded++
	if err := batch.Wait(t.Context()); err != nil {
		t.Fatal(err)
	}
	if err := rawCapacityExact(saved.HostSlice(), full); err != nil {
		t.Fatalf("extraction: %v", err)
	}

	up := [2][]float32{
		upstreamRopeRun(t, ops, oracle[0], heads, position),
		upstreamRopeRun(t, ops, oracle[1], heads, position),
	}
	if !bytes.Equal(unsafe.Slice((*byte)(unsafe.Pointer(&up[0][0])), 4*len(up[0])), unsafe.Slice((*byte)(unsafe.Pointer(&up[1][0])), 4*len(up[1]))) {
		t.Fatalf("upstream_aa: %v", rawCapacityExact(up[1], up[0]))
	}
	got := full[33280:]
	cpu := append([]float32(nil), heads...)
	for head := range 64 {
		mixedCapacityRotate(cpu[head*512:(head+1)*512], position, true)
	}
	upOver, upMax := upstreamRopeOver(up[0], cpu)
	prodOver, prodMax := upstreamRopeOver(got, cpu)
	candOver, candMax := upstreamRopeOver(got, up[0])
	t.Logf("operands position=%d count=%d pre_heads_sha256=%s production_heads_sha256=%s oracle_heads_sha256=%s candidate_over=%d candidate_max_abs=%.6g",
		position, count, f32Digest(heads), f32Digest(got), f32Digest(up[0]), candOver, candMax)
	t.Logf("cpu_retained position=%d count=%d oracle_vs_cpu_over=%d max_abs=%.6g production_vs_cpu_over=%d max_abs=%.6g",
		position, count, upOver, upMax, prodOver, prodMax)
	if err := upstreamRopeHeads(got, up[0]); err != nil {
		t.Fatal(err)
	}
}
