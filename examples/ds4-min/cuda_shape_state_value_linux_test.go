//go:build linux

package main

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/tmc/lib/examples/ds4-min/dsv4shape"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// This gate exercises the production State owners and dispatch methods. It is
// not a checkpoint/complete-forward gate. The outer reviewed runner supplies
// the sealed library manifest and owns the exclusive L4 admission.
func TestShapeStateNativeValues(t *testing.T) {
	if os.Getenv("DS4_PRO_STATE_NATIVE") != "1" {
		t.Skip("native State gate not requested")
	}
	dir := os.Getenv("DS4_PRO_STATE_RECEIPT")
	if dir == "" || os.Getenv("DS4_PRO_STATE_ADMISSION") == "" {
		t.Fatal("missing reviewed native admission or receipt directory")
	}
	if os.Getenv("GPU_LOCK_HELD") == "" || os.Getenv("XDG_CACHE_HOME") != filepath.Join(dir, "cache") {
		t.Fatal("missing canonical outer lock or scoped cold cache")
	}
	cache, err := os.ReadDir(filepath.Join(dir, "cache"))
	if err != nil || len(cache) != 0 {
		t.Fatalf("native cache is not initially empty: %v", err)
	}
	var pins map[string]string
	data, err := os.ReadFile(os.Getenv("DS4_PRO_STATE_LIBRARIES"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &pins); err != nil {
		t.Fatal(err)
	}
	const compiler = "/usr/local/cuda-12.9/targets/x86_64-linux/lib/libnvrtc.so"
	const builtin = "/usr/local/cuda-12.9/targets/x86_64-linux/lib/libnvrtc-builtins.so"
	if pins[compiler] != "b0064dcb18b3c41384a03dd70a1a41a5130fee3b3d4f728bc4bbd661693718f5" || pins[builtin] != "6507cf33450cfb22970c2c1e7aa8e7c0a33e39d864720277b05f61a92791e2e8" || os.Getenv("NVRTC_LIB_PATH") != compiler {
		t.Fatal("unqualified compiler closure")
	}
	check := func() {
		t.Helper()
		for name, want := range pins {
			b, err := os.ReadFile(name)
			if err != nil {
				t.Fatal(err)
			}
			if len(want) != 64 || fmt.Sprintf("%x", sha256.Sum256(b)) != want {
				t.Fatalf("library seal differs: %s", name)
			}
		}
	}
	check()
	if err := driver.Init(0); err != nil {
		t.Fatal(err)
	}
	if pins[driver.Path()] == "" {
		t.Fatal("loaded driver absent from reviewed library manifest")
	}
	version, err := driver.DriverGetVersion()
	if err != nil || version != 13000 {
		t.Fatalf("unqualified driver API version %d: %v", version, err)
	}
	rawDevice, err := driver.DeviceGet(0)
	if err != nil {
		t.Fatal(err)
	}
	uuid, err := rawDevice.Uuid()
	if err != nil || fmt.Sprintf("%x", uuid.Bytes) != "3ac20ff35caf479fb816d367fd966fdf" {
		t.Fatalf("unqualified device identity: %x: %v", uuid.Bytes, err)
	}
	flags, active, err := rawDevice.PrimaryCtxGetState()
	if err != nil || flags != 0 || active != 0 {
		t.Fatalf("primary context before: flags=%d active=%d error=%v", flags, active, err)
	}
	d, err := cuda.NewDevice(0)
	r := &cudaFoundation{device: d, major: 8, minor: 9, gguf: shapeTestRopeMetadata()}
	owner := &shapeStateNativeOwner{parent: r.Close, drain: func() error {
		if r.device == nil {
			return nil // Only error-owned constructor resources can exist here.
		}
		return r.device.WithContext(driver.CtxSynchronize)
	}}
	if d == nil {
		owner.parent = nil
	}
	t.Cleanup(func() {
		if err := owner.Close(); err != nil {
			shapeStateNativeRetained = append(shapeStateNativeRetained, owner)
			t.Errorf("retain native State owner and device: %v", err)
			return
		}
		flags, active, err := rawDevice.PrimaryCtxGetState()
		if err != nil || flags != 0 || active != 0 {
			t.Errorf("primary context after: flags=%d active=%d error=%v", flags, active, err)
		}
		check()
	})
	if err != nil {
		owner.retainError(err)
		t.Fatal(err)
	}
	if d == nil {
		t.Fatal("missing successful device owner")
	}
	major, minor, err := d.ComputeCapability()
	if err != nil || major != 8 || minor != 9 {
		t.Fatalf("unqualified architecture %d.%d: %v", major, minor, err)
	}
	c, err := dsv4shape.For(dsv4shape.Pro)
	if err != nil {
		t.Fatal(err)
	}
	binding := forwardTestBinding(t, c)
	const context = 4
	plan, err := planShapeForward(c, binding, context)
	if err != nil {
		t.Fatal(err)
	}
	_, consumerBytes, err := planShapeConsumerBuffers(c, context)
	if err != nil {
		t.Fatal(err)
	}
	cachePlan, err := planShapeCache(c, context, 1<<30, plan.bytes+consumerBytes)
	if err != nil {
		t.Fatal(err)
	}
	const limit = 512 << 20
	if cachePlan.bytes+plan.bytes+consumerBytes+uint64(c.Embed+c.Heads)*4+(360<<20) > limit {
		t.Fatal("State fixture exceeds bounded 512MiB device allocation")
	}
	upload := func(values []float32) cudamem.Slice[float32] {
		t.Helper()
		v, err := cudamem.Alloc[float32](d, len(values))
		owner.retainAllocation(v.Ptr() != 0, v.Free, err)
		if err != nil {
			t.Fatal(err)
		}
		if err := v.CopyFromHost(values); err != nil {
			t.Fatal(err)
		}
		return v
	}
	weights := make(map[string]cudamem.Slice[float32])
	ones := make([]float32, c.Embed)
	for i := range ones {
		ones[i] = 1
	}
	norm := upload(ones)
	for name := range plan.norms {
		weights[name] = norm
	}
	state, err := newCUDAShapeForward(r, c, binding, weights, context, consumerBytes)
	if err != nil {
		owner.retainError(err)
		t.Fatal(err)
	}
	owner.children = append(owner.children, state)
	consumers, err := newCUDAShapeConsumers(r, c, context, 0)
	if err != nil {
		owner.retainError(err)
		t.Fatal(err)
	}
	owner.children = append(owner.children, consumers)
	compilerMajor, compilerMinor, err := nvrtc.Version()
	if err != nil || compilerMajor != 12 || compilerMinor != 9 || nvrtc.Path() != compiler {
		t.Fatalf("loaded compiler differs: path=%s version=%d.%d error=%v", nvrtc.Path(), compilerMajor, compilerMinor, err)
	}
	_, _, _, sinks, _ := shapeAttentionFixture(c.Heads)
	sink := upload(sinks)
	read := func(v cudamem.Slice[float32]) []float32 {
		t.Helper()
		if err := d.WithContext(driver.CtxSynchronize); err != nil {
			t.Fatal(err)
		}
		out := make([]float32, v.Len())
		if err := v.CopyToHost(out); err != nil {
			t.Fatal(err)
		}
		return out
	}
	var progress shapeRawProgress
	attention, qkvReference, outputReference := shapeStateAttentionFixture(t, r, c, binding, norm, owner)
	quant, err := consumers.attentionQuant()
	if err != nil {
		t.Fatal(err)
	}
	var compressed, indexed shapeCompressorProgress
	compressor, compressorProject := shapeStateCompressorFixture(t, r, c, binding, false, owner)
	indexer, indexerProject := shapeStateCompressorFixture(t, r, c, binding, true, owner)
	var tokens, gates, indexTokens, indexGates [][]float32
	var compressedValues []float32
	var chronological []float32
	for position := uint32(0); position < context; position++ {
		input := make([]float32, c.Embed)
		for i := range input {
			input[i] = float32((i*11+int(position)*7)%31-15) / 32
		}
		if err := state.scratch["attn_norm"].CopyFromHost(input); err != nil {
			t.Fatal(err)
		}
		if err := encodeShapeQKV(state, binding, 2, attention, quant); err != nil {
			t.Fatal(err)
		}
		q, kv := qkvReference(input)
		shapeStateCompare(t, "Q8 bound QKV Q", read(state.scratch["q"]), q, 3e-5)
		shapeStateCompare(t, "Q8 bound QKV KV", read(state.scratch["kv"]), kv, 3e-5)
		if err := encodeShapeRoPE(state, consumers.entries["ds4_shape_rope_tail"], 2, position); err != nil {
			t.Fatal(err)
		}
		wantQ, wantKV := append([]float32(nil), q...), append([]float32(nil), kv...)
		for h := range c.Heads {
			cudaCompressedRotateReference(wantQ[h*c.HeadDim:(h+1)*c.HeadDim], int(position))
		}
		cudaCompressedRotateReference(wantKV, int(position))
		shapeStateCompare(t, "RoPE Q", read(state.scratch["q"]), wantQ, 3e-6)
		shapeStateCompare(t, "RoPE KV", read(state.scratch["kv"]), wantKV, 3e-6)
		if err := commitShapeRaw(state, consumers.entries["ds4_shape_raw_commit"], 2, position, &progress); err != nil {
			t.Fatal(err)
		}
		cudaCompressedFP8Reference(wantKV)
		shapeStateCompare(t, "post FP8 current KV", read(state.scratch["kv"]), wantKV, 3e-6)
		stored := append([]float32(nil), wantKV...)
		for i := range stored {
			stored[i] = cudaTestFloat16Round(stored[i])
		}
		chronological = append(chronological, stored...)
		raw, err := shapeCacheLayerBuffer(state.cache, 2, "raw")
		if err != nil {
			t.Fatal(err)
		}
		shapeStateCompare(t, "raw cache", read(raw)[:len(chronological)], chronological, 0)
		for _, e := range []struct {
			indexer  bool
			weights  shapeCompressorWeights
			progress *shapeCompressorProgress
		}{{false, compressor, &compressed}, {true, indexer, &indexed}} {
			if err := encodeShapeCompressor(state, binding, 2, position, e.indexer, e.weights, e.progress); err != nil {
				t.Fatal(err)
			}
		}
		token, gate := compressorProject(input, int(position))
		tokens, gates = append(tokens, token), append(gates, gate)
		token, gate = indexerProject(input, int(position))
		indexTokens, indexGates = append(indexTokens, token), append(indexGates, gate)
		if compressed.rows != int((position+1)/4) || indexed.rows != compressed.rows {
			t.Fatalf("compressor/indexer emission count: attention=%d indexer=%d", compressed.rows, indexed.rows)
		}
		if compressed.rows != 0 {
			pooled := cudaCompressedPoolReference(tokens, gates, 4, c.HeadDim)
			compressedValues = cudaCompressedFinishReference(pooled, ones[:c.HeadDim], int(position+1-4), false)
			buffer, err := shapeCacheLayerBuffer(state.cache, 2, "attention-cache")
			if err != nil {
				t.Fatal(err)
			}
			shapeStateCompare(t, "emitted attention cache", read(buffer)[:c.HeadDim], compressedValues, 3e-4)
			pooled = cudaCompressedPoolReference(indexTokens, indexGates, 4, c.IndexerHeadDim)
			wantIndex := cudaCompressedFinishReference(pooled, ones[:c.IndexerHeadDim], int(position+1-4), true)
			buffer, err = shapeCacheLayerBuffer(state.cache, 2, "indexer-cache")
			if err != nil {
				t.Fatal(err)
			}
			shapeStateCompare(t, "emitted indexer cache", read(buffer)[:c.IndexerHeadDim], wantIndex, 3e-4)
		}
		query, err := shapeConsumerView[float32](consumers, "indexer-query")
		if err != nil {
			t.Fatal(err)
		}
		invalid, err := shapeConsumerView[uint32](consumers, "indexer-invalid")
		if err != nil {
			t.Fatal(err)
		}
		mask := state.scratch["attn_score"]
		if err := invalid.CopyFromHost([]uint32{0xfeed}); err != nil {
			t.Fatal(err)
		}
		if err := encodeShapeIndexer(state, consumers.entries["ds4_shape_indexer_select"], 2, position, indexed.rows, query, mask, invalid); err != nil {
			t.Fatal(err)
		}
		mask = mask.Slice(0, indexed.rows)
		scores, err := cudamem.Reinterpret[float32](consumers.buffers["attention-scores"])
		if err != nil {
			t.Fatal(err)
		}
		req := shapeAttentionRequest{layer: 2, position: position, rawRows: progress.rows, rawStart: progress.start, compressedRows: uint32(compressed.rows), useMask: shapeAttentionUseMask(true, uint32(compressed.rows))}
		if err := encodeShapeAttentionExternal(state, consumers.entries["ds4_shape_attention_decode_external"], req, sink, mask, scores); err != nil {
			t.Fatal(err)
		}
		operand := chronological
		if position == 0 {
			operand = wantKV // Production zero-prefix path reads float KV, not half ring.
		}
		reference := shapeAttentionReference(wantQ, operand, compressedValues, sinks, nil, c.Heads, c.HeadDim)
		wantHeads := make([]float32, len(reference))
		for i, v := range reference {
			wantHeads[i] = float32(v)
		}
		shapeStateCompare(t, "external attention", read(state.scratch["heads"]), wantHeads, 3e-5)
		if err := encodeShapeAttentionInverseRoPE(state, consumers.entries["ds4_shape_rope_tail"], 2, position); err != nil {
			t.Fatal(err)
		}
		for h := range c.Heads {
			cudaCompressedRotateReference(wantHeads[h*c.HeadDim:(h+1)*c.HeadDim], -int(position))
		}
		shapeStateCompare(t, "inverse RoPE", read(state.scratch["heads"]), wantHeads, 3e-5)
		if err := encodeShapeAttentionOutput(state, binding, 2, attention, quant); err != nil {
			t.Fatal(err)
		}
		shapeStateCompare(t, "Q8 grouped attention output", read(state.scratch["attn_out"]), outputReference(wantHeads), 3e-5)
		t.Logf("ACTUAL_STATE_CHAIN_PASS position=%d raw_rows=%d heads=%d", position, progress.rows, c.Heads)
	}
	if err := os.WriteFile(filepath.Join(dir, "state-values.json"), []byte(fmt.Sprintf("{\"positions\":4,\"heads\":128,\"allocation_limit\":%d,\"checkpoint_qualified\":false}\n", limit)), 0600); err != nil {
		t.Fatal(err)
	}
}

func shapeStateRMSReference(input []float32, groups int) []float32 {
	output := make([]float32, len(input))
	width := len(input) / groups
	for group := range groups {
		var sum float64
		for _, v := range input[group*width : (group+1)*width] {
			sum += float64(v) * float64(v)
		}
		scale := 1 / math.Sqrt(sum/float64(width)+1e-6)
		for i := range width {
			output[group*width+i] = float32(float64(input[group*width+i]) * scale)
		}
	}
	return output
}

func shapeStateAttentionFixture(t *testing.T, r *cudaFoundation, c dsv4shape.Config, binding *dsv4shape.Binding, norm cudamem.Slice[float32], owner *shapeStateNativeOwner) (shapeAttentionWeights, func([]float32) ([]float32, []float32), func([]float32) []float32) {
	t.Helper()
	weight := func(suffix string, cols, rows, groups int) (cudaModelTensor, func([]float32) []float32) {
		t.Helper()
		name := "blk.2." + suffix
		var view dsv4shape.WeightView
		for _, v := range binding.Weights {
			if v.Name == name {
				view = v
			}
		}
		rowBytes := cols / 32 * 34
		values := make([]byte, rows*rowBytes)
		for row := range rows {
			for j, col := range []int{row % cols, (row*17 + 13) % cols} {
				p := values[row*rowBytes+col/32*34:]
				binary.LittleEndian.PutUint16(p, 0x2400) // Exact F16 scale 1/64.
				if j == 0 {
					p[2+col%32] = 1
				} else {
					p[2+col%32] = 255 // -1, signed Q8.
				}
			}
		}
		v, err := cudamem.Alloc[byte](r.device, len(values))
		owner.retainAllocation(v.Ptr() != 0, v.Free, err)
		if err != nil {
			t.Fatal(err)
		}
		if err := v.CopyFromHost(values); err != nil {
			t.Fatal(err)
		}
		project := func(input []float32) []float32 {
			activation := cudaQ81ActivationReference(input)
			output := make([]float32, rows)
			for row := range rows {
				start := row / (rows / groups) * cols
				var sum float64
				for j, col := range []int{row % cols, (row*17 + 13) % cols} {
					index := start + col
					p := activation[index/32*36:]
					d := cudaTestFloat16(binary.LittleEndian.Uint16(p))
					q := float64(int8(p[4+index%32]))
					if j == 1 {
						q = -q
					}
					sum += float64(d) * q / 64
				}
				output[row] = float32(sum)
			}
			return output
		}
		return cudaModelTensor{name: name, typeID: 8, dimensions: []uint64{uint64(cols), uint64(rows)}, device: v.Ptr(), bytes: uint64(len(values)), fileOffset: view.Offset}, project
	}
	w := shapeAttentionWeights{}
	var qa, qb, kv, outA, outB func([]float32) []float32
	w.qa, qa = weight("attn_q_a.weight", c.Embed, c.QLoRA, 1)
	w.qb, qb = weight("attn_q_b.weight", c.QLoRA, c.Heads*c.HeadDim, 1)
	w.kv, kv = weight("attn_kv.weight", c.Embed, c.HeadDim, 1)
	w.outA, outA = weight("attn_output_a.weight", c.HeadDim*c.Heads/c.OutGroups, c.OutGroups*c.OutLoRA, c.OutGroups)
	w.outB, outB = weight("attn_output_b.weight", c.OutGroups*c.OutLoRA, c.Embed, 1)
	w.qaNorm = cudaModelTensor{name: "blk.2.attn_q_a_norm.weight", typeID: 0, dimensions: []uint64{uint64(c.QLoRA)}, device: norm.Ptr(), bytes: uint64(c.QLoRA) * 4}
	w.kvNorm = cudaModelTensor{name: "blk.2.attn_kv_a_norm.weight", typeID: 0, dimensions: []uint64{uint64(c.HeadDim)}, device: norm.Ptr(), bytes: uint64(c.HeadDim) * 4}
	qkv := func(input []float32) ([]float32, []float32) {
		q := qb(shapeStateRMSReference(qa(input), 1))
		return shapeStateRMSReference(q, c.Heads), shapeStateRMSReference(kv(input), 1)
	}
	output := func(input []float32) []float32 { return outB(outA(input)) }
	return w, qkv, output
}

// Matrices are complete, nonzero F16 tensors at the admitted 7168-wide layout.
// Two distinct nonzero columns in every row keep an independent exact sparse
// projection oracle cheap without replacing any production projection route.
func shapeStateCompressorFixture(t *testing.T, r *cudaFoundation, c dsv4shape.Config, binding *dsv4shape.Binding, indexer bool, owner *shapeStateNativeOwner) (shapeCompressorWeights, func([]float32, int) ([]float32, []float32)) {
	t.Helper()
	head, prefix := c.HeadDim, "attn_compressor_"
	if indexer {
		head, prefix = c.IndexerHeadDim, "indexer_compressor_"
	}
	width := 2 * head
	upload := func(name string, values []uint16, dims []uint64) cudaModelTensor {
		t.Helper()
		v, err := cudamem.Alloc[uint16](r.device, len(values))
		owner.retainAllocation(v.Ptr() != 0, v.Free, err)
		if err != nil {
			t.Fatal(err)
		}
		if err := v.CopyFromHost(values); err != nil {
			t.Fatal(err)
		}
		var view dsv4shape.WeightView
		for _, w := range binding.Weights {
			if w.Name == name {
				view = w
			}
		}
		return cudaModelTensor{name: name, typeID: 1, dimensions: dims, device: v.Ptr(), bytes: uint64(v.Len()) * 2, fileOffset: view.Offset}
	}
	name := "blk.2." + prefix
	a, b := make([]uint16, c.Embed*width), make([]uint16, c.Embed*width)
	for row := range width {
		i, j := row%c.Embed, (row*17+13)%c.Embed
		a[row*c.Embed+i], a[row*c.Embed+j] = 0x3400, 0x3000 // .25, .125
		b[row*c.Embed+i], b[row*c.Embed+j] = 0x3000, 0xac00 // .125, -.0625
	}
	w := shapeCompressorWeights{
		kv:   upload(name+"kv.weight", a, []uint64{uint64(c.Embed), uint64(width)}),
		gate: upload(name+"gate.weight", b, []uint64{uint64(c.Embed), uint64(width)}),
	}
	ape := make([]uint16, width*4)
	for position := range 4 {
		for i := range width {
			ape[position*width+i] = cudaQ81HalfReference(float32((i*3+position)%7-3) / 16)
		}
	}
	w.ape = upload(name+"ape.weight", ape, []uint64{uint64(width), 4})
	normValues := make([]float32, head)
	for i := range normValues {
		normValues[i] = 1
	}
	v, err := cudamem.Alloc[float32](r.device, head)
	owner.retainAllocation(v.Ptr() != 0, v.Free, err)
	if err != nil {
		t.Fatal(err)
	}
	if err := v.CopyFromHost(normValues); err != nil {
		t.Fatal(err)
	}
	w.norm = cudaModelTensor{name: name + "norm.weight", typeID: 0, dimensions: []uint64{uint64(head)}, device: v.Ptr(), bytes: uint64(head) * 4}
	project := func(input []float32, position int) ([]float32, []float32) {
		x, g := make([]float32, width), make([]float32, width)
		for row := range width {
			i, j := row%c.Embed, (row*17+13)%c.Embed
			x[row] = .25*input[i] + .125*input[j]
			g[row] = .125*input[i] - .0625*input[j] + float32((row*3+position)%7-3)/16
		}
		return x, g
	}
	return w, project
}

// One ordered owner, rather than independent testing cleanup callbacks, keeps
// all borrowed fixture operands and the device alive after any child failure.
// A retained entry remains reachable until the process ends or Close succeeds.
type shapeStateNativeOwner struct {
	children []io.Closer
	buffers  []func() error
	drain    func() error
	parent   func() error
}

var shapeStateNativeRetained []*shapeStateNativeOwner

type shapeStateNativeCloseFunc func() error

func (f shapeStateNativeCloseFunc) Close() error { return f() }

type shapeNativeConstructorTestError struct {
	name    string
	blocked *bool
	calls   *[]string
}

func (e *shapeNativeConstructorTestError) Error() string { return e.name }
func (e *shapeNativeConstructorTestError) Close() error {
	*e.calls = append(*e.calls, e.name)
	if e.blocked != nil && *e.blocked {
		return errors.New("partial cleanup blocked")
	}
	return nil
}

func TestShapeStateNativeDeviceErrorOwner(t *testing.T) {
	blocked := true
	var calls []string
	partial := &shapeNativeConstructorTestError{"device", &blocked, &calls}
	owner := shapeNativeDeviceOwner(nil, fmt.Errorf("construct device: %w", partial), nil)
	if len(owner.children) != 1 || owner.parent != nil {
		t.Fatal("partial device owner disappeared")
	}
	if err := owner.Close(); err == nil || !slices.Equal(calls, []string{"device"}) {
		t.Fatalf("partial device failure: %v calls %v", err, calls)
	}
	blocked = false
	if err := owner.Close(); err != nil {
		t.Fatal(err)
	}
	if err := owner.Close(); err != nil || !slices.Equal(calls, []string{"device", "device"}) {
		t.Fatal("partial device retry lost or repeated owner")
	}
}

func TestShapeStateNativeAllocationErrorOwners(t *testing.T) {
	blocked := true
	var calls []string
	first := &shapeNativeConstructorTestError{"first", nil, &calls}
	second := &shapeNativeConstructorTestError{"second", &blocked, &calls}
	owner := &shapeStateNativeOwner{
		drain:  func() error { calls = append(calls, "drain"); return nil },
		parent: func() error { calls = append(calls, "parent"); return nil },
	}
	owner.retainAllocation(true, func() error { calls = append(calls, "buffer"); return nil }, errors.Join(fmt.Errorf("first allocation: %w", first), second))
	owner.retainAllocation(false, func() error { t.Fatal("null allocation freed"); return nil }, nil)
	if len(owner.children) != 2 || len(owner.buffers) != 1 {
		t.Fatal("partial allocation owners disappeared")
	}
	if err := owner.Close(); err == nil || !slices.Equal(calls, []string{"drain", "second"}) {
		t.Fatalf("partial allocation released dependencies: %v calls %v", err, calls)
	}
	blocked = false
	calls = nil
	if err := owner.Close(); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(calls, []string{"drain", "second", "first", "buffer", "parent"}) {
		t.Fatalf("partial allocation retry order: %v", calls)
	}
	calls = nil
	if err := owner.Close(); err != nil || len(calls) != 0 {
		t.Fatal("consumed allocation owner reused")
	}
}

func TestShapeStateNativeOwnerFailure(t *testing.T) {
	for _, fail := range []string{"drain", "child", "buffer", "parent"} {
		t.Run(fail, func(t *testing.T) {
			var calls []string
			fault := true
			call := func(name string) error {
				calls = append(calls, name)
				if fault && name == fail {
					return errors.New("injected " + name)
				}
				return nil
			}
			s := &shapeStateNativeOwner{
				children: []io.Closer{shapeStateNativeCloseFunc(func() error { return call("child") })},
				buffers:  []func() error{func() error { return call("buffer") }},
				drain:    func() error { return call("drain") },
				parent:   func() error { return call("parent") },
			}
			if err := s.Close(); err == nil {
				t.Fatal("fault admitted")
			}
			order := []string{"drain", "child", "buffer", "parent"}
			stop := slices.Index(order, fail) + 1
			if !slices.Equal(calls, order[:stop]) || s.parent == nil {
				t.Fatalf("failure consumed dependent owner: calls %v, parent retained %v", calls, s.parent != nil)
			}
			fault = false
			calls = nil
			if err := s.Close(); err != nil {
				t.Fatal(err)
			}
			want := append([]string(nil), order[stop-1:]...)
			if fail == "child" || fail == "buffer" {
				want = append([]string{"drain"}, want...)
			}
			if !slices.Equal(calls, want) {
				t.Fatalf("retry: calls %v want %v", calls, want)
			}
			calls = nil
			if err := s.Close(); err != nil || len(calls) != 0 {
				t.Fatalf("consumed owner reused: %v, calls %v", err, calls)
			}
		})
	}
}

func TestShapeStateNativeOwnerMissingDrain(t *testing.T) {
	called := false
	s := &shapeStateNativeOwner{buffers: []func() error{func() error { called = true; return nil }}, parent: func() error { called = true; return nil }}
	if err := s.Close(); err == nil || called || s.parent == nil || len(s.buffers) != 1 {
		t.Fatalf("missing drain consumed resources: %v", err)
	}
}

func (s *shapeStateNativeOwner) retainError(err error) {
	s.children = append(s.children, backendErrorOwners(err)...)
}

func (s *shapeStateNativeOwner) retainAllocation(live bool, free func() error, err error) {
	if live {
		s.buffers = append(s.buffers, free)
	}
	s.retainError(err)
}

func (s *shapeStateNativeOwner) Close() error {
	if s == nil {
		return nil
	}
	if len(s.children) != 0 || len(s.buffers) != 0 {
		if s.drain == nil {
			return errors.New("native State owner: missing drain")
		}
		if err := s.drain(); err != nil {
			return err
		}
	}
	for len(s.children) != 0 {
		i := len(s.children) - 1
		if err := s.children[i].Close(); err != nil {
			return err
		}
		s.children = s.children[:i]
	}
	for len(s.buffers) != 0 {
		i := len(s.buffers) - 1
		if err := s.buffers[i](); err != nil {
			return err
		}
		s.buffers = s.buffers[:i]
	}
	if s.parent != nil {
		if err := s.parent(); err != nil {
			return err
		}
		s.parent = nil
	}
	return nil
}

func shapeStateCompare(t *testing.T, name string, got, want []float32, tolerance float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length %d want %d", name, len(got), len(want))
	}
	for i, v := range got {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || math.Abs(float64(v)-float64(want[i])) > tolerance*(1+math.Abs(float64(want[i]))) {
			t.Fatalf("%s[%d]: got %.9g want %.9g", name, i, v, want[i])
		}
	}
}
