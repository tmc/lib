//go:build darwin && arm64

package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"testing"

	libmetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
)

func TestFP8ComparisonLimit(t *testing.T) {
	tests := []struct {
		name      string
		got, want float32
		accept    bool
	}{
		{"equal zero", 0, 0, true},
		{"adjacent bins", -0.03125, -0.03515625, true},
		{"two bins", -0.02734375, -0.03515625, false},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			delta := math.Abs(float64(test.got - test.want))
			if got := delta <= fp8ComparisonLimit(test.got, test.want); got != test.accept {
				t.Fatalf("accepted = %t, want %t", got, test.accept)
			}
		})
	}
}

func TestMetalPinnedTensorViews(t *testing.T) {
	path := os.Getenv("DS4_GGUF_PROFILE_MODEL")
	if path == "" {
		t.Skip("profile model path not set")
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		t.Fatal(err)
	}
	g, err := readGGUF(f, info.Size())
	if err != nil {
		t.Fatal(err)
	}
	var maxTensor uint64
	for _, tensor := range g.tensors {
		n, err := tensorBytes(tensor)
		if err != nil {
			t.Fatal(err)
		}
		maxTensor = max(maxTensor, n)
	}
	dev, err := libmetal.NewDevice()
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	views, err := mapMetalModel(dev, f, uint64(info.Size()), maxTensor)
	if err != nil {
		t.Fatal(err)
	}
	defer views.Close()
	for _, tensor := range g.tensors {
		n, err := tensorBytes(tensor)
		if err != nil {
			t.Fatal(err)
		}
		view, err := views.Tensor(g.dataOffset+tensor.offset, n)
		if err != nil {
			t.Fatalf("tensor %q: %v", tensor.name, err)
		}
		if uint64(view.Len()) != n {
			t.Fatalf("tensor %q view length = %d, want %d", tensor.name, view.Len(), n)
		}
	}
	t.Logf("device=%s views=%d tensors=%d max_tensor=%d", dev.Name(), len(views.views), len(g.tensors), maxTensor)
	ops, err := newMetalPinnedOps(dev)
	if err != nil {
		t.Fatal(err)
	}
	defer ops.Close()
	embedding, err := views.Bind(g, "token_embd.weight")
	if err != nil {
		t.Fatal(err)
	}
	x, err := ops.GetRowF16(embedding, 0)
	if err != nil {
		t.Fatal(err)
	}
	for i, got := range x {
		bits := binary.LittleEndian.Uint16(embedding.data.HostSlice()[2*i:])
		if want := testFloat16(bits); got != want {
			t.Fatalf("embedding[%d] = %g, want %g", i, got, want)
		}
	}
	q, err := views.Bind(g, "blk.0.attn_q_a.weight")
	if err != nil {
		t.Fatal(err)
	}
	got, err := ops.MatvecQ80(q, x)
	if err != nil {
		t.Fatal(err)
	}
	want := testQ80Matvec(q.data.HostSlice(), x, int(q.dimensions[1]))
	for i := range got {
		delta := math.Abs(float64(got[i] - want[i]))
		limit := 1e-3 + 1e-4*math.Abs(float64(want[i]))
		if delta > limit {
			t.Fatalf("attn_q_a[%d] = %g, want %g (delta %g > %g)", i, got[i], want[i], delta, limit)
		}
	}
	oracleDir := os.Getenv("DS4_LAYER0_ORACLE_DIR")
	if oracleDir == "" {
		t.Skip("layer 0 oracle directory not set")
	}
	bind := func(name string) boundMetalTensor {
		t.Helper()
		tensor, err := views.Bind(g, name)
		if err != nil {
			t.Fatal(err)
		}
		return tensor
	}
	pre, attnNorm, err := ops.Layer0AttentionPre(
		embedding,
		bind("blk.0.hc_attn_fn.weight"),
		bind("blk.0.hc_attn_scale.weight"),
		bind("blk.0.hc_attn_base.weight"),
		bind("blk.0.attn_norm.weight"),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	compareOracleF32(t, "hc_attn_pre", pre, oracleDir+"/stage_hc_attn_pre-0_pos0.bin")
	compareOracleF32(t, "attn_norm", attnNorm, oracleDir+"/stage_attn_norm-0_pos0.bin")
	qRaw, err := ops.MatvecQ80(q, attnNorm)
	if err != nil {
		t.Fatal(err)
	}
	compareOracleF32(t, "q_lora", qRaw, oracleDir+"/stage_q_lora-0_pos0.bin")
	qNormWeight := bind("blk.0.attn_q_a_norm.weight")
	qNorm, err := ops.RMS(qRaw, qNormWeight, 1, true)
	if err != nil {
		t.Fatal(err)
	}
	compareOracleF32(t, "q_lora_norm", qNorm, oracleDir+"/stage_q_lora_norm-0_pos0.bin")
	qHeadsRaw, err := ops.MatvecQ80(bind("blk.0.attn_q_b.weight"), qNorm)
	if err != nil {
		t.Fatal(err)
	}
	compareOracleF32(t, "Qraw", qHeadsRaw, oracleDir+"/stage_Qraw-0_pos0.bin")
	qHeads, err := ops.RMS(qHeadsRaw, qNormWeight, 64, false)
	if err != nil {
		t.Fatal(err)
	}
	compareOracleF32(t, "Qnorm", qHeads, oracleDir+"/stage_Qnorm-0_pos0.bin")
	kvRaw, err := ops.MatvecQ80(bind("blk.0.attn_kv.weight"), attnNorm)
	if err != nil {
		t.Fatal(err)
	}
	compareOracleF32(t, "KVraw", kvRaw, oracleDir+"/stage_KVraw-0_pos0.bin")
	kvNorm, err := ops.RMS(kvRaw, bind("blk.0.attn_kv_a_norm.weight"), 1, true)
	if err != nil {
		t.Fatal(err)
	}
	compareOracleF32(t, "KVnorm", kvNorm, oracleDir+"/stage_KVnorm-0_pos0.bin")
	compareOracleF32(t, "KVrope", kvNorm, oracleDir+"/stage_KVrope-0_pos0.bin")
	qCur, kvCur, heads, err := ops.RopeAttentionOne(qHeads, kvNorm, bind("blk.0.attn_sinks.weight"), 0)
	if err != nil {
		t.Fatal(err)
	}
	compareOracleF32(t, "Qcur", qCur, oracleDir+"/stage_Qcur-0_pos0.bin")
	compareOracleFP8F32(t, "KVcur", kvCur, oracleDir+"/stage_KVcur-0_pos0.bin")
	if len(heads) != len(qHeads) {
		t.Fatalf("attention heads length = %d, want %d", len(heads), len(qHeads))
	}
	compareOracleF32(t, "kqv_out", heads, oracleDir+"/stage_kqv_out-0_pos0.bin")
	rawCache, err := metalmem.Alloc[float32](dev, ds4RawCap*ds4KVWidth)
	if err != nil {
		t.Fatal(err)
	}
	defer rawCache.Free()
	rawQ, rawKV, rawHeads, err := ops.RopeAttentionRaw(qHeads, kvNorm, bind("blk.0.attn_sinks.weight"), rawCache, 0)
	if err != nil {
		t.Fatal(err)
	}
	compareOracleF32(t, "raw Qcur", rawQ, oracleDir+"/stage_Qcur-0_pos0.bin")
	compareOracleFP8F32(t, "raw KVcur", rawKV, oracleDir+"/stage_KVcur-0_pos0.bin")
	compareOracleF32(t, "raw kqv_out", rawHeads, oracleDir+"/stage_kqv_out-0_pos0.bin")
	projectionA := bind("blk.0.attn_output_a.weight")
	low := make([]float32, 0, 8192)
	for group := range 8 {
		rows, err := projectionA.Rows(group*1024, 1024)
		if err != nil {
			t.Fatal(err)
		}
		part, err := ops.MatvecQ80(rows, heads[group*4096:(group+1)*4096])
		if err != nil {
			t.Fatal(err)
		}
		low = append(low, part...)
	}
	compareOracleF32(t, "attn_low", low, oracleDir+"/stage_attn_low-0_pos0.bin")
	attnOut, err := ops.MatvecQ80(bind("blk.0.attn_output_b.weight"), low)
	if err != nil {
		t.Fatal(err)
	}
	compareOracleF32(t, "attn_out", attnOut, oracleDir+"/stage_attn_out-0_pos0.bin")
	hc, err := ops.Layer0AttentionPost(
		embedding, attnOut,
		bind("blk.0.hc_attn_fn.weight"),
		bind("blk.0.hc_attn_scale.weight"),
		bind("blk.0.hc_attn_base.weight"),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	compareOracleF32(t, "hc_attn_post", hc, oracleDir+"/stage_hc_attn_post-0_pos0.bin")
	ffnCur, ffnNorm, err := ops.HCPre(
		hc,
		bind("blk.0.hc_ffn_fn.weight"),
		bind("blk.0.hc_ffn_scale.weight"),
		bind("blk.0.hc_ffn_base.weight"),
		bind("blk.0.ffn_norm.weight"),
	)
	if err != nil {
		t.Fatal(err)
	}
	compareOracleF32(t, "hc_ffn_pre", ffnCur, oracleDir+"/stage_hc_ffn_pre-0_pos0.bin")
	compareOracleF32(t, "ffn_norm", ffnNorm, oracleDir+"/stage_ffn_norm-0_pos0.bin")
	routerLogits, err := ops.MatvecF16(bind("blk.0.ffn_gate_inp.weight"), ffnNorm)
	if err != nil {
		t.Fatal(err)
	}
	compareOracleF32(t, "ffn_moe_logits", routerLogits, oracleDir+"/stage_ffn_moe_logits-0_pos0.bin")
	probs, selected, weights, err := ops.RouterHash(routerLogits, bind("blk.0.ffn_gate_tid2eid.weight"), 0)
	if err != nil {
		t.Fatal(err)
	}
	compareOracleF32(t, "ffn_moe_probs", probs, oracleDir+"/stage_ffn_moe_probs-0_pos0.bin")
	compareOracleI32(t, "ffn_moe_topk", selected, oracleDir+"/stage_ffn_moe_topk-0_pos0.i32")
	compareOracleF32(t, "ffn_moe_weights_scaled", weights, oracleDir+"/stage_ffn_moe_weights_scaled-0_pos0.bin")
	sharedGate, err := ops.MatvecQ80(bind("blk.0.ffn_gate_shexp.weight"), ffnNorm)
	if err != nil {
		t.Fatal(err)
	}
	sharedUp, err := ops.MatvecQ80(bind("blk.0.ffn_up_shexp.weight"), ffnNorm)
	if err != nil {
		t.Fatal(err)
	}
	sharedMid, err := ops.SwiGLU(sharedGate, sharedUp)
	if err != nil {
		t.Fatal(err)
	}
	sharedOut, err := ops.MatvecQ80(bind("blk.0.ffn_down_shexp.weight"), sharedMid)
	if err != nil {
		t.Fatal(err)
	}
	compareOracleF32(t, "ffn_shexp", sharedOut, oracleDir+"/stage_ffn_shexp-0_pos0.bin")
	gateExperts := bind("blk.0.ffn_gate_exps.weight")
	upExperts := bind("blk.0.ffn_up_exps.weight")
	var routedGate, routedUp, routedMid []float32
	for slot, id := range selected {
		gateExpert, err := gateExperts.Expert(id)
		if err != nil {
			t.Fatal(err)
		}
		upExpert, err := upExperts.Expert(id)
		if err != nil {
			t.Fatal(err)
		}
		gate, up, err := ops.MatvecIQ2PairF32(gateExpert, upExpert, ffnNorm)
		if err != nil {
			t.Fatal(err)
		}
		gate, err = ops.ClampGate10(gate)
		if err != nil {
			t.Fatal(err)
		}
		up, err = ops.Clamp10(up)
		if err != nil {
			t.Fatal(err)
		}
		mid, err := ops.SwiGLUWeighted(gate, up, weights[slot])
		if err != nil {
			t.Fatal(err)
		}
		routedGate = append(routedGate, gate...)
		routedUp = append(routedUp, up...)
		routedMid = append(routedMid, mid...)
	}
	// The fused one-token prefill path reuses the gate and up scratch buffers,
	// so those dumps are partial rather than six logical expert rows. The
	// weighted SwiGLU dump is the complete six-row routed intermediate.
	compareOracleF32(t, "ffn_moe_weighted_swiglu", routedMid, oracleDir+"/stage_ffn_moe_weighted_swiglu-0_pos0.bin")
	downExperts := bind("blk.0.ffn_down_exps.weight")
	var routedDown []float32
	for slot, id := range selected {
		downExpert, err := downExperts.Expert(id)
		if err != nil {
			t.Fatal(err)
		}
		down, err := ops.MatvecQ2KF32(downExpert, routedMid[slot*2048:(slot+1)*2048])
		if err != nil {
			t.Fatal(err)
		}
		routedDown = append(routedDown, down...)
	}
	routedOut, err := ops.SumVectors(routedDown, 6)
	if err != nil {
		t.Fatal(err)
	}
	compareOracleF32(t, "ffn_moe_out", routedOut, oracleDir+"/stage_ffn_moe_out-0_pos0.bin")
	ffnOut, err := ops.SumVectors(append(append([]float32(nil), routedOut...), sharedOut...), 2)
	if err != nil {
		t.Fatal(err)
	}
	compareOracleF32(t, "ffn_out", ffnOut, oracleDir+"/stage_ffn_out-0_pos0.bin")
	hcFFN, err := ops.HCPost(
		hc, ffnOut,
		bind("blk.0.hc_ffn_fn.weight"),
		bind("blk.0.hc_ffn_scale.weight"),
		bind("blk.0.hc_ffn_base.weight"),
	)
	if err != nil {
		t.Fatal(err)
	}
	compareOracleF32(t, "hc_ffn_post", hcFFN, oracleDir+"/stage_hc_ffn_post-0_pos0.bin")
	transitionRoot := os.Getenv("DS4_TRANSITION_ORACLE_ROOT")
	allLayersOracle := os.Getenv("DS4_ALL_LAYERS_ORACLE_DIR")
	if allLayersOracle != "" {
		state := hcFFN
		for layer := 1; layer < 43; layer++ {
			t.Run(fmt.Sprintf("layer-%d", layer), func(t *testing.T) {
				state = runPinnedHashLayer(t, ops, bind, state, layer, allLayersOracle)
			})
			if t.Failed() {
				break
			}
		}
	} else if transitionRoot != "" {
		layer1 := runPinnedHashLayer(t, ops, bind, hcFFN, 1, transitionRoot+"/layer-1")
		layer2 := runPinnedHashLayer(t, ops, bind, layer1, 2, transitionRoot+"/layer-2")
		runPinnedHashLayer(t, ops, bind, layer2, 3, transitionRoot+"/layer-3")
		validatePinnedLateTransition(t, ops, bind, 42, transitionRoot+"/layer-42")
	}
}

func TestMetalRuntimePositionZeroExecution(t *testing.T) {
	model := os.Getenv("DS4_GGUF_PROFILE_MODEL")
	if model == "" {
		t.Skip("model path not set")
	}
	r, err := newMetalRuntime(model, 1024)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	logits, err := r.evalToken(0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(logits) != 129280 || r.pos != 1 {
		t.Fatalf("logits/position = %d/%d, want 129280/1", len(logits), r.pos)
	}
	if got := r.lastOpeningBatch; got != (metalBatchStats{Encoded: 7, Commits: 1, Waits: 1}) {
		t.Fatalf("last opening batch = %+v, want {Encoded:7 Commits:1 Waits:1}", got)
	}
	if got := r.lastRoutedBatch; got != (metalBatchStats{Encoded: 11, Commits: 1, Waits: 1}) {
		t.Fatalf("last routed batch = %+v, want {Encoded:11 Commits:1 Waits:1}", got)
	}
	if got := r.lastFFNBatches; got != (metalBatchStats{Encoded: 19, Commits: 4, Waits: 4}) {
		t.Fatalf("last FFN batches = %+v, want {Encoded:19 Commits:4 Waits:4}", got)
	}
	for i, value := range logits {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			t.Fatalf("logits[%d] = %g", i, value)
		}
	}
}

func TestMetalRuntimePromptAGreedyTokens(t *testing.T) {
	model := os.Getenv("DS4_GGUF_PROFILE_MODEL")
	if model == "" {
		t.Skip("model path not set")
	}
	r, err := newMetalRuntime(model, 128)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	tok, err := newTokenizer(r.model)
	if err != nil {
		t.Fatal(err)
	}
	prompt, err := tok.RenderNonThinking("You are a helpful assistant", "a")
	if err != nil {
		t.Fatal(err)
	}
	s, err := newSession(r, 128, tok.eos)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Sync(prompt); err != nil {
		t.Fatal(err)
	}
	want := []int{19923, 3, 1730, 588, 342}
	for step := range len(want) {
		id := greedyToken(s.logits)
		text, err := tok.Decode([]int{id})
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("step %d: %d %q", step, id, text)
		if id != want[step] {
			t.Fatalf("step %d token = %d, want %d", step, id, want[step])
		}
		if id == tok.eos {
			break
		}
		if _, err := s.Accept(id); err != nil {
			t.Fatal(err)
		}
	}
}

func validatePinnedLateTransition(t *testing.T, ops *metalPinnedOps, bind func(string) boundMetalTensor, layer int, oracleDir string) {
	t.Helper()
	attnNorm := readOracleF32(t, fmt.Sprintf("%s/stage_attn_norm-%d_pos0.bin", oracleDir, layer))
	validatePinnedCompressors(t, ops, bind, attnNorm, layer, oracleDir)
	ffnNorm := readOracleF32(t, fmt.Sprintf("%s/stage_ffn_norm-%d_pos0.bin", oracleDir, layer))
	name := func(s string) string { return fmt.Sprintf("blk.%d.%s", layer, s) }
	logits, err := ops.MatvecF16(bind(name("ffn_gate_inp.weight")), ffnNorm)
	if err != nil {
		t.Fatal(err)
	}
	compareOracleF32(t, "late ffn_moe_logits", logits, fmt.Sprintf("%s/stage_ffn_moe_logits-%d_pos0.bin", oracleDir, layer))
	probs, selected, weights, err := ops.RouterBiased(logits, bind(name("exp_probs_b.bias")))
	if err != nil {
		t.Fatal(err)
	}
	compareOracleF32(t, "late ffn_moe_probs", probs, fmt.Sprintf("%s/stage_ffn_moe_probs-%d_pos0.bin", oracleDir, layer))
	compareOracleI32(t, "late ffn_moe_topk", selected, fmt.Sprintf("%s/stage_ffn_moe_topk-%d_pos0.i32", oracleDir, layer))
	compareOracleF32(t, "late ffn_moe_weights_scaled", weights, fmt.Sprintf("%s/stage_ffn_moe_weights_scaled-%d_pos0.bin", oracleDir, layer))
}

func readOracleF32(t *testing.T, path string) []float32 {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(b)%4 != 0 {
		t.Fatalf("oracle %q has invalid byte length %d", path, len(b))
	}
	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return out
}

func runPinnedHashLayer(t *testing.T, ops *metalPinnedOps, bind func(string) boundMetalTensor, state []float32, layer int, oracleDir string) []float32 {
	t.Helper()
	name := func(s string) string { return fmt.Sprintf("blk.%d.%s", layer, s) }
	oracle := func(stage string) string {
		return fmt.Sprintf("%s/stage_%s-%d_pos0.bin", oracleDir, stage, layer)
	}
	pre, attnNorm, err := ops.HCPre(state, bind(name("hc_attn_fn.weight")), bind(name("hc_attn_scale.weight")), bind(name("hc_attn_base.weight")), bind(name("attn_norm.weight")))
	if err != nil {
		t.Fatal(err)
	}
	compareOracleF32(t, "hc_attn_pre", pre, oracle("hc_attn_pre"))
	compareOracleF32(t, "attn_norm", attnNorm, oracle("attn_norm"))
	if layer >= 2 {
		validatePinnedCompressors(t, ops, bind, attnNorm, layer, oracleDir)
	}
	qRaw, err := ops.MatvecQ80(bind(name("attn_q_a.weight")), attnNorm)
	if err != nil {
		t.Fatal(err)
	}
	compareOracleF32(t, "q_lora", qRaw, oracle("q_lora"))
	qNormWeight := bind(name("attn_q_a_norm.weight"))
	qNorm, err := ops.RMS(qRaw, qNormWeight, 1, true)
	if err != nil {
		t.Fatal(err)
	}
	compareOracleF32(t, "q_lora_norm", qNorm, oracle("q_lora_norm"))
	qHeadsRaw, err := ops.MatvecQ80(bind(name("attn_q_b.weight")), qNorm)
	if err != nil {
		t.Fatal(err)
	}
	compareOracleF32(t, "Qraw", qHeadsRaw, oracle("Qraw"))
	qHeads, err := ops.RMS(qHeadsRaw, qNormWeight, 64, false)
	if err != nil {
		t.Fatal(err)
	}
	compareOracleF32(t, "Qnorm", qHeads, oracle("Qnorm"))
	kvRaw, err := ops.MatvecQ80(bind(name("attn_kv.weight")), attnNorm)
	if err != nil {
		t.Fatal(err)
	}
	compareOracleF32(t, "KVraw", kvRaw, oracle("KVraw"))
	kvNorm, err := ops.RMS(kvRaw, bind(name("attn_kv_a_norm.weight")), 1, true)
	if err != nil {
		t.Fatal(err)
	}
	compareOracleF32(t, "KVnorm", kvNorm, oracle("KVnorm"))
	qCur, kvCur, heads, err := ops.RopeAttentionOne(qHeads, kvNorm, bind(name("attn_sinks.weight")), 0)
	if err != nil {
		t.Fatal(err)
	}
	compareOracleF32(t, "Qcur", qCur, oracle("Qcur"))
	compareOracleFP8F32(t, "KVcur", kvCur, oracle("KVcur"))
	compareOracleFP8F32(t, "kqv_out", heads, oracle("kqv_out"))
	projectionA := bind(name("attn_output_a.weight"))
	low := make([]float32, 0, 8192)
	for group := range 8 {
		rows, err := projectionA.Rows(group*1024, 1024)
		if err != nil {
			t.Fatal(err)
		}
		part, err := ops.MatvecQ80(rows, heads[group*4096:(group+1)*4096])
		if err != nil {
			t.Fatal(err)
		}
		low = append(low, part...)
	}
	compareOracleF32(t, "attn_low", low, oracle("attn_low"))
	attnOut, err := ops.MatvecQ80(bind(name("attn_output_b.weight")), low)
	if err != nil {
		t.Fatal(err)
	}
	compareOracleF32(t, "attn_out", attnOut, oracle("attn_out"))
	hc, err := ops.HCPost(state, attnOut, bind(name("hc_attn_fn.weight")), bind(name("hc_attn_scale.weight")), bind(name("hc_attn_base.weight")))
	if err != nil {
		t.Fatal(err)
	}
	compareOracleF32(t, "hc_attn_post", hc, oracle("hc_attn_post"))
	_, ffnNorm, err := ops.HCPre(hc, bind(name("hc_ffn_fn.weight")), bind(name("hc_ffn_scale.weight")), bind(name("hc_ffn_base.weight")), bind(name("ffn_norm.weight")))
	if err != nil {
		t.Fatal(err)
	}
	compareOracleF32(t, "ffn_norm", ffnNorm, oracle("ffn_norm"))
	logits, err := ops.MatvecF16(bind(name("ffn_gate_inp.weight")), ffnNorm)
	if err != nil {
		t.Fatal(err)
	}
	compareOracleF32(t, "ffn_moe_logits", logits, oracle("ffn_moe_logits"))
	var probs, weights []float32
	var selected []int32
	if layer < 3 {
		probs, selected, weights, err = ops.RouterHash(logits, bind(name("ffn_gate_tid2eid.weight")), 0)
	} else {
		probs, selected, weights, err = ops.RouterBiased(logits, bind(name("exp_probs_b.bias")))
	}
	if err != nil {
		t.Fatal(err)
	}
	compareOracleF32(t, "ffn_moe_probs", probs, oracle("ffn_moe_probs"))
	compareOracleI32(t, "ffn_moe_topk", selected, fmt.Sprintf("%s/stage_ffn_moe_topk-%d_pos0.i32", oracleDir, layer))
	compareOracleF32(t, "ffn_moe_weights_scaled", weights, oracle("ffn_moe_weights_scaled"))
	sharedGate, err := ops.MatvecQ80(bind(name("ffn_gate_shexp.weight")), ffnNorm)
	if err != nil {
		t.Fatal(err)
	}
	sharedUp, err := ops.MatvecQ80(bind(name("ffn_up_shexp.weight")), ffnNorm)
	if err != nil {
		t.Fatal(err)
	}
	sharedMid, err := ops.SwiGLU(sharedGate, sharedUp)
	if err != nil {
		t.Fatal(err)
	}
	sharedOut, err := ops.MatvecQ80(bind(name("ffn_down_shexp.weight")), sharedMid)
	if err != nil {
		t.Fatal(err)
	}
	compareOracleF32(t, "ffn_shexp", sharedOut, oracle("ffn_shexp"))
	gateExperts, upExperts, downExperts := bind(name("ffn_gate_exps.weight")), bind(name("ffn_up_exps.weight")), bind(name("ffn_down_exps.weight"))
	var mids []float32
	for slot, id := range selected {
		gateExpert, err := gateExperts.Expert(id)
		if err != nil {
			t.Fatal(err)
		}
		upExpert, err := upExperts.Expert(id)
		if err != nil {
			t.Fatal(err)
		}
		gate, up, err := ops.MatvecIQ2PairF32(gateExpert, upExpert, ffnNorm)
		if err != nil {
			t.Fatal(err)
		}
		gate, err = ops.ClampGate10(gate)
		if err != nil {
			t.Fatal(err)
		}
		up, err = ops.Clamp10(up)
		if err != nil {
			t.Fatal(err)
		}
		mid, err := ops.SwiGLUWeighted(gate, up, weights[slot])
		if err != nil {
			t.Fatal(err)
		}
		mids = append(mids, mid...)
	}
	compareOracleF32(t, "ffn_moe_weighted_swiglu", mids, oracle("ffn_moe_weighted_swiglu"))
	var downs []float32
	for slot, id := range selected {
		downExpert, err := downExperts.Expert(id)
		if err != nil {
			t.Fatal(err)
		}
		down, err := ops.MatvecQ2KF32(downExpert, mids[slot*2048:(slot+1)*2048])
		if err != nil {
			t.Fatal(err)
		}
		downs = append(downs, down...)
	}
	routedOut, err := ops.SumVectors(downs, 6)
	if err != nil {
		t.Fatal(err)
	}
	compareOracleF32(t, "ffn_moe_out", routedOut, oracle("ffn_moe_out"))
	ffnOut, err := ops.SumVectors(append(append([]float32(nil), routedOut...), sharedOut...), 2)
	if err != nil {
		t.Fatal(err)
	}
	compareOracleF32(t, "ffn_out", ffnOut, oracle("ffn_out"))
	out, err := ops.HCPost(hc, ffnOut, bind(name("hc_ffn_fn.weight")), bind(name("hc_ffn_scale.weight")), bind(name("hc_ffn_base.weight")))
	if err != nil {
		t.Fatal(err)
	}
	compareOracleF32(t, "hc_ffn_post", out, oracle("hc_ffn_post"))
	return out
}

func validatePinnedCompressors(t *testing.T, ops *metalPinnedOps, bind func(string) boundMetalTensor, x []float32, layer int, oracleDir string) {
	t.Helper()
	ratio := 4
	lanes := []struct {
		name  string
		width int
	}{
		{"attn", 1024},
		{"indexer", 256},
	}
	if layer%2 != 0 {
		ratio = 128
		lanes = lanes[:1]
		lanes[0].width = 512
	}
	for _, lane := range lanes {
		prefix := fmt.Sprintf("blk.%d.%s_compressor_", layer, lane.name)
		kv, err := ops.MatvecF16(bind(prefix+"kv.weight"), x)
		if err != nil {
			t.Fatal(err)
		}
		score, err := ops.MatvecF16(bind(prefix+"gate.weight"), x)
		if err != nil {
			t.Fatal(err)
		}
		ape, err := ops.GetRowF16(bind(prefix+"ape.weight"), 0)
		if err != nil {
			t.Fatal(err)
		}
		scoreAPE, err := ops.SumVectors(append(append([]float32(nil), score...), ape...), 2)
		if err != nil {
			t.Fatal(err)
		}
		compareOracleF32(t, lane.name+"_comp_kv_raw", kv, fmt.Sprintf("%s/stage_%s_comp_kv_raw-%d_pos0.bin", oracleDir, lane.name, layer))
		compareOracleF32(t, lane.name+"_comp_score_raw", score, fmt.Sprintf("%s/stage_%s_comp_score_raw-%d_pos0.bin", oracleDir, lane.name, layer))
		rows := ratio
		row := 0
		if ratio == 4 {
			rows = 8
			row = 4
		}
		stateKV := make([]float32, rows*lane.width)
		stateScore := make([]float32, rows*lane.width)
		for i := range stateScore {
			stateScore[i] = float32(math.Inf(-1))
		}
		copy(stateKV[row*lane.width:], kv)
		copy(stateScore[row*lane.width:], scoreAPE)
		compareOracleF32(t, lane.name+"_state_kv", stateKV, fmt.Sprintf("%s/stage_%s_state_kv-%d_pos0.bin", oracleDir, lane.name, layer))
		compareOracleF32(t, lane.name+"_state_score", stateScore, fmt.Sprintf("%s/stage_%s_state_score-%d_pos0.bin", oracleDir, lane.name, layer))
	}
}

func compareOracleI32(t *testing.T, name string, got []int32, path string) {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(b) < 4*len(got) {
		t.Fatalf("%s oracle has %d bytes, need %d", name, len(b), 4*len(got))
	}
	for i, value := range got {
		want := int32(binary.LittleEndian.Uint32(b[4*i:]))
		if value != want {
			t.Fatalf("%s[%d] = %d, want %d", name, i, value, want)
		}
	}
}

func compareOracleF32(t *testing.T, name string, got []float32, path string) {
	compareOracleF32Tolerance(t, name, got, path, 3e-3, 5e-4)
}

// compareOracleFP8F32 allows the implementation and oracle to choose adjacent
// E4M3 values when pre-quantization rounding crosses a cell boundary. It is
// used only at the dequantized FP8 boundary; downstream results retain the
// ordinary float tolerance.
func compareOracleFP8F32(t *testing.T, name string, got []float32, path string) {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(b) < 4*len(got) {
		t.Fatalf("%s oracle has %d bytes, need %d", name, len(b), 4*len(got))
	}
	for i := range got {
		want := math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
		delta := math.Abs(float64(got[i] - want))
		limit := fp8ComparisonLimit(got[i], want)
		if delta > limit {
			t.Fatalf("%s[%d] = %g, want %g (delta %g > one FP8 ULP %g)", name, i, got[i], want, delta, limit)
		}
	}
}

func fp8ComparisonLimit(got, want float32) float64 {
	magnitude := max(math.Abs(float64(got)), math.Abs(float64(want)))
	ulp := 0.0
	if magnitude != 0 {
		_, exponent := math.Frexp(magnitude)
		ulp = math.Ldexp(1, exponent-4)
	}
	return 2e-3 + 2e-4*math.Abs(float64(want)) + ulp
}

func compareOracleF32Tolerance(t *testing.T, name string, got []float32, path string, atol, rtol float64) {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(b) < 4*len(got) {
		t.Fatalf("%s oracle has %d bytes, need %d", name, len(b), 4*len(got))
	}
	for i := range got {
		want := math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
		delta := math.Abs(float64(got[i] - want))
		limit := atol + rtol*math.Abs(float64(want))
		if delta > limit {
			t.Fatalf("%s[%d] = %g, want %g (delta %g > %g)", name, i, got[i], want, delta, limit)
		}
	}
}

func testQ80Matvec(weights []byte, x []float32, rows int) []float32 {
	blocks := len(x) / 32
	rowBytes := blocks * 34
	out := make([]float32, rows)
	for row := range rows {
		p := weights[row*rowBytes : (row+1)*rowBytes]
		var sum float32
		for block := range blocks {
			scale := testFloat16(binary.LittleEndian.Uint16(p[block*34:]))
			for i := range 32 {
				sum += float32(int8(p[block*34+2+i])) * scale * x[block*32+i]
			}
		}
		out[row] = sum
	}
	return out
}

func testFloat16(h uint16) float32 {
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
		return math.Float32frombits(sign | (exp+112)<<23 | frac<<13)
	}
}
