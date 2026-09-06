//go:build darwin && arm64

package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
	"testing"

	libmetal "github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
	"github.com/tmc/lib/gguf"
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
	prof := newPhaseTimer(t)
	defer prof.report()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		t.Fatal(err)
	}
	g, err := gguf.Read(f, info.Size())
	if err != nil {
		t.Fatal(err)
	}
	prof.mark("gguf.Read")
	var maxTensor uint64
	for _, tensor := range g.Tensors {
		n, err := tensor.Bytes()
		if err != nil {
			t.Fatal(err)
		}
		maxTensor = max(maxTensor, n)
	}
	prof.mark("tensor size scan")
	dev, err := libmetal.NewDevice()
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	prof.mark("NewDevice")
	views, err := mapMetalModel(dev, f, uint64(info.Size()), maxTensor)
	if err != nil {
		t.Fatal(err)
	}
	defer views.Close()
	prof.mark("mapMetalModel")
	for _, tensor := range g.Tensors {
		n, err := tensor.Bytes()
		if err != nil {
			t.Fatal(err)
		}
		view, err := views.Tensor(g.DataOffset+tensor.Offset, n)
		if err != nil {
			t.Fatalf("tensor %q: %v", tensor.Name, err)
		}
		if uint64(view.Len()) != n {
			t.Fatalf("tensor %q view length = %d, want %d", tensor.Name, view.Len(), n)
		}
	}
	t.Logf("device=%s views=%d tensors=%d max_tensor=%d", dev.Name(), len(views.views), len(g.Tensors), maxTensor)
	prof.mark("tensor view check")
	ops, err := newMetalPinnedOps(dev)
	if err != nil {
		t.Fatal(err)
	}
	defer ops.Close()
	prof.mark("newMetalPinnedOps (MSL compile)")
	embedding, err := views.Bind(g, "token_embd.weight")
	if err != nil {
		t.Fatal(err)
	}
	prof.mark("bind token_embd")
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
	prof.mark("GetRowF16 (first GPU touch)")
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
		requireNoDependentOracles(t)
		t.Skip("layer 0 oracle directory not set")
	}
	requireOraclePopulated(t, "DS4_LAYER0_ORACLE_DIR", oracleDir)
	bind := func(name string) boundMetalTensor {
		t.Helper()
		tensor, err := views.Bind(g, name)
		if err != nil {
			t.Fatal(err)
		}
		return tensor
	}
	prof.mark("bind loop + first MatvecQ80")
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
	prof.mark("attn projections")
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
	prof.mark("attention")
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
	prof.mark("HCPre")
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
	prof.mark("router + experts + shared")
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
		// Each layer is checked twice. The teacher-forced pass feeds every
		// layer the oracle's own input state, so a strict per-stage tolerance
		// isolates that one layer's arithmetic. The chained pass then runs the
		// layers free-running from layer 0 and bounds how fast the residual
		// stream may diverge; a fixed per-stage tolerance cannot survive 42
		// chained layers, but an amplification bound can.
		prof.mark("HCPost + layer0 checks")
		pos := allLayersOraclePosition(t)
		for layer := 1; layer < 43; layer++ {
			t.Run(fmt.Sprintf("forced-layer-%d", layer), func(t *testing.T) {
				prior := fmt.Sprintf("%s/stage_hc_ffn_post-%d_pos%d.bin", allLayersOracle, layer-1, pos)
				runPinnedHashLayerAt(t, ops, bind, readOracleF32(t, prior), layer, allLayersOracle, pos)
			})
		}
		prof.mark("forced layers 1-42")
		t.Run("chained-drift", func(t *testing.T) {
			state := hcFFN
			for layer := 1; layer < 43; layer++ {
				state = runPinnedHashLayerNoCompare(t, ops, bind, state, layer)
				want := readOracleF32(t, fmt.Sprintf("%s/stage_hc_ffn_post-%d_pos0.bin", allLayersOracle, layer))
				drift := relativeDrift(t, state, want)
				measured, err := measureOracleDrift(state, want)
				if err != nil {
					t.Fatal(err)
				}
				t.Logf("layer %d diagnostic only: count=%d max_abs=%g rms=%g small_count=%d small_max_abs=%g", layer, measured.count, measured.maxAbs, measured.rms, measured.smallCount, measured.smallMaxAbs)
				t.Logf("layer %d chained drift %.3e", layer, drift)
				if drift > chainedDriftBudget {
					t.Fatalf("layer %d chained drift %.3e exceeds budget %.3e", layer, drift, chainedDriftBudget)
				}
			}
		})
		t.Run("output-head", func(t *testing.T) {
			runPinnedOutputHead(t, ops, bind, allLayersOracle)
		})
	}
	if posOracle := os.Getenv("DS4_POS_ORACLE_DIR"); posOracle != "" {
		t.Run("rope-nonzero-position", func(t *testing.T) {
			runPinnedRopeAtPosition(t, ops, bind, posOracle, pinnedOraclePosition(t))
		})
	}
	if dir := os.Getenv("DS4_COMPRESSOR_ORACLE_DIR"); dir != "" {
		t.Run("compressor-state", func(t *testing.T) {
			runPinnedCompressorState(t, ops, bind, dir, pinnedCompressorRun(t))
		})
	}
	if transitionRoot != "" && allLayersOracle == "" {
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
	if got := r.lastOpeningBatch; got != (metalBatchStats{Encoded: 14, Commits: 1, Waits: 1}) {
		t.Fatalf("last opening batch = %+v, want {Encoded:14 Commits:1 Waits:1}", got)
	}
	if got := r.lastRawOpeningBatch; got != (metalBatchStats{Encoded: 8, Commits: 1, Waits: 1}) {
		t.Fatalf("last raw opening batch = %+v, want {Encoded:8 Commits:1 Waits:1}", got)
	}
	if got := r.lastCompressedOdd; got != (metalBatchStats{Encoded: 11, Commits: 1, Waits: 1}) {
		t.Fatalf("last odd compressed batch = %+v, want {Encoded:11 Commits:1 Waits:1}", got)
	}
	if got := r.lastCompressedEven; got != (metalBatchStats{Encoded: 14, Commits: 1, Waits: 1}) {
		t.Fatalf("last even compressed batch = %+v, want {Encoded:14 Commits:1 Waits:1}", got)
	}
	if got := r.lastAttentionBatch; got != (metalBatchStats{Encoded: 10, Commits: 1, Waits: 1}) {
		t.Fatalf("last attention batch = %+v, want {Encoded:10 Commits:1 Waits:1}", got)
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
	s, err := newSession(r, 128, tok.stops)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Sync(prompt); err != nil {
		t.Fatal(err)
	}
	// The registered oracle trace for prompt A, from the capture under
	// evidence/oracle-provenance/capture-prompt-a-01: ten selections
	// decoding to "Hello! How can I help you today?" and stopping
	// naturally on token 1. Earlier revisions checked only the first
	// five, which miss the later divergence: the CUDA candidate failed
	// at selection 5, choosing 8233 (" assist") over 1694 (" help").
	want := []int{19923, 3, 1730, 588, 342, 1694, 440, 4316, 33, 1}
	steps := 0
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
		steps++
		if tok.IsStop(id) {
			break
		}
		if _, err := s.Accept(id); err != nil {
			t.Fatal(err)
		}
	}
	if steps != len(want) {
		t.Fatalf("stopped after %d selections, want %d", steps, len(want))
	}
}

func validatePinnedLateTransition(t *testing.T, ops *metalPinnedOps, bind func(string) boundMetalTensor, layer int, oracleDir string) {
	t.Helper()
	attnNorm := readOracleF32(t, fmt.Sprintf("%s/stage_attn_norm-%d_pos0.bin", oracleDir, layer))
	validatePinnedCompressors(t, ops, bind, attnNorm, layer, oracleDir, 0)
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
	return runPinnedHashLayerAt(t, ops, bind, state, layer, oracleDir, 0)
}

// runPinnedHashLayerAt runs one layer against the capture's stages for a chosen
// token position. Every rotation is the identity at position 0, so a sweep that
// only ever reads _pos0 cannot see a position-dependent defect.
func runPinnedHashLayerAt(t *testing.T, ops *metalPinnedOps, bind func(string) boundMetalTensor, state []float32, layer int, oracleDir string, pos int) []float32 {
	t.Helper()
	return runPinnedHashLayerStrict(t, ops, bind, state, layer, oracleDir, true, pos)
}

// runPinnedHashLayerNoCompare runs one layer without any oracle comparison. The
// chained drift check uses it so that only the residual stream is measured.
func runPinnedHashLayerNoCompare(t *testing.T, ops *metalPinnedOps, bind func(string) boundMetalTensor, state []float32, layer int) []float32 {
	t.Helper()
	return runPinnedHashLayerStrict(t, ops, bind, state, layer, "", false, 0)
}

func runPinnedHashLayerStrict(t *testing.T, ops *metalPinnedOps, bind func(string) boundMetalTensor, state []float32, layer int, oracleDir string, strict bool, pos int) []float32 {
	t.Helper()
	// The chained drift pass reuses this body with strict comparison off, so
	// every stage check goes through these guards.
	checkF32 := func(name string, got []float32, path string) {
		if strict {
			compareOracleF32(t, name, got, path)
		}
	}
	checkFP8 := func(name string, got []float32, path string) {
		if strict {
			compareOracleFP8F32(t, name, got, path)
		}
	}
	checkI32 := func(name string, got []int32, path string) {
		if strict {
			compareOracleI32(t, name, got, path)
		}
	}
	// hc_ffn_pre is compared at the frozen acceptance limit rather than the
	// exploratory one the other stages use. It has never been compared beyond
	// layer 0, so there is no exploratory history to preserve, and the FFN HCPre
	// is the boundary this sweep was leaving unmeasured.
	checkFrozenF32 := func(name string, got []float32, path string) {
		if strict {
			compareOracleF32Tolerance(t, name, got, path, frozenAtol, frozenRtol)
		}
	}
	checkResidual := func(name string, got []float32, path string) {
		if strict {
			compareOracleResidual(t, name, got, path)
		}
	}
	name := func(s string) string { return fmt.Sprintf("blk.%d.%s", layer, s) }
	oracle := func(stage string) string {
		return fmt.Sprintf("%s/stage_%s-%d_pos%d.bin", oracleDir, stage, layer, pos)
	}
	pre, attnNorm, err := ops.HCPre(state, bind(name("hc_attn_fn.weight")), bind(name("hc_attn_scale.weight")), bind(name("hc_attn_base.weight")), bind(name("attn_norm.weight")))
	if err != nil {
		t.Fatal(err)
	}
	checkF32("hc_attn_pre", pre, oracle("hc_attn_pre"))
	checkF32("attn_norm", attnNorm, oracle("attn_norm"))
	// validatePinnedCompressors hand-builds the compressor state assuming the
	// window holds a single position, so it is only meaningful at position 0.
	if strict && layer >= 2 && pos == 0 {
		validatePinnedCompressors(t, ops, bind, attnNorm, layer, oracleDir, pos)
	}
	qRaw, err := ops.MatvecQ80(bind(name("attn_q_a.weight")), attnNorm)
	if err != nil {
		t.Fatal(err)
	}
	checkF32("q_lora", qRaw, oracle("q_lora"))
	qNormWeight := bind(name("attn_q_a_norm.weight"))
	qNorm, err := ops.RMS(qRaw, qNormWeight, 1, true)
	if err != nil {
		t.Fatal(err)
	}
	checkF32("q_lora_norm", qNorm, oracle("q_lora_norm"))
	qHeadsRaw, err := ops.MatvecQ80(bind(name("attn_q_b.weight")), qNorm)
	if err != nil {
		t.Fatal(err)
	}
	checkF32("Qraw", qHeadsRaw, oracle("Qraw"))
	qHeads, err := ops.RMS(qHeadsRaw, qNormWeight, 64, false)
	if err != nil {
		t.Fatal(err)
	}
	checkF32("Qnorm", qHeads, oracle("Qnorm"))
	kvRaw, err := ops.MatvecQ80(bind(name("attn_kv.weight")), attnNorm)
	if err != nil {
		t.Fatal(err)
	}
	checkF32("KVraw", kvRaw, oracle("KVraw"))
	kvNorm, err := ops.RMS(kvRaw, bind(name("attn_kv_a_norm.weight")), 1, true)
	if err != nil {
		t.Fatal(err)
	}
	checkF32("KVnorm", kvNorm, oracle("KVnorm"))
	qCur, kvCur, heads, err := ops.RopeAttentionOne(qHeads, kvNorm, bind(name("attn_sinks.weight")), 0)
	if err != nil {
		t.Fatal(err)
	}
	checkF32("Qcur", qCur, oracle("Qcur"))
	checkFP8("KVcur", kvCur, oracle("KVcur"))
	// kqv_out was gated at the FP8 limit, which masked a systematic divergence:
	// the attention dot rounded only KV to f16 and left Q at f32, where the
	// oracle rounds both operands (dsv4_misc.metal:5748-5751, 5779-5782). That
	// held kqv_out near 3.4e-4 at every layer and propagated into hc_ffn_pre.
	// With both operands rounded the stage sits at 1.5e-5, so it is pinned at
	// the frozen limit here: restoring the unrounded Q turns this red.
	checkFrozenF32("kqv_out", heads, oracle("kqv_out"))
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
	checkF32("attn_low", low, oracle("attn_low"))
	attnOut, err := ops.MatvecQ80(bind(name("attn_output_b.weight")), low)
	if err != nil {
		t.Fatal(err)
	}
	checkF32("attn_out", attnOut, oracle("attn_out"))
	hc, err := ops.HCPost(state, attnOut, bind(name("hc_attn_fn.weight")), bind(name("hc_attn_scale.weight")), bind(name("hc_attn_base.weight")))
	if err != nil {
		t.Fatal(err)
	}
	checkF32("hc_attn_post", hc, oracle("hc_attn_post"))
	ffnPre, ffnNorm, err := ops.HCPre(hc, bind(name("hc_ffn_fn.weight")), bind(name("hc_ffn_scale.weight")), bind(name("hc_ffn_base.weight")), bind(name("ffn_norm.weight")))
	if err != nil {
		t.Fatal(err)
	}
	// The attention HCPre above binds its first return and checks it; this one
	// discarded the identical return, so the FFN half of the hyper-connection
	// pre-mix went unmeasured for every layer in this sweep.
	checkFrozenF32("hc_ffn_pre", ffnPre, oracle("hc_ffn_pre"))
	checkF32("ffn_norm", ffnNorm, oracle("ffn_norm"))
	logits, err := ops.MatvecF16(bind(name("ffn_gate_inp.weight")), ffnNorm)
	if err != nil {
		t.Fatal(err)
	}
	checkF32("ffn_moe_logits", logits, oracle("ffn_moe_logits"))
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
	checkF32("ffn_moe_probs", probs, oracle("ffn_moe_probs"))
	checkI32("ffn_moe_topk", selected, fmt.Sprintf("%s/stage_ffn_moe_topk-%d_pos%d.i32", oracleDir, layer, pos))
	checkF32("ffn_moe_weights_scaled", weights, oracle("ffn_moe_weights_scaled"))
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
	checkF32("ffn_shexp", sharedOut, oracle("ffn_shexp"))
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
	checkF32("ffn_moe_weighted_swiglu", mids, oracle("ffn_moe_weighted_swiglu"))
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
	checkF32("ffn_moe_out", routedOut, oracle("ffn_moe_out"))
	ffnOut, err := ops.SumVectors(append(append([]float32(nil), routedOut...), sharedOut...), 2)
	if err != nil {
		t.Fatal(err)
	}
	checkF32("ffn_out", ffnOut, oracle("ffn_out"))
	out, err := ops.HCPost(hc, ffnOut, bind(name("hc_ffn_fn.weight")), bind(name("hc_ffn_scale.weight")), bind(name("hc_ffn_base.weight")))
	if err != nil {
		t.Fatal(err)
	}
	checkResidual("hc_ffn_post", out, oracle("hc_ffn_post"))
	return out
}

func validatePinnedCompressors(t *testing.T, ops *metalPinnedOps, bind func(string) boundMetalTensor, x []float32, layer int, oracleDir string, pos int) {
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
		compareOracleF32(t, lane.name+"_comp_kv_raw", kv, fmt.Sprintf("%s/stage_%s_comp_kv_raw-%d_pos%d.bin", oracleDir, lane.name, layer, pos))
		compareOracleF32(t, lane.name+"_comp_score_raw", score, fmt.Sprintf("%s/stage_%s_comp_score_raw-%d_pos%d.bin", oracleDir, lane.name, layer, pos))
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
		compareOracleF32(t, lane.name+"_state_kv", stateKV, fmt.Sprintf("%s/stage_%s_state_kv-%d_pos%d.bin", oracleDir, lane.name, layer, pos))
		compareOracleMaskedF32(t, lane.name+"_state_score", stateScore, fmt.Sprintf("%s/stage_%s_state_score-%d_pos%d.bin", oracleDir, lane.name, layer, pos))
	}
}

func compareOracleI32(t *testing.T, name string, got []int32, path string) {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(b)%4 != 0 {
		t.Fatalf("%s oracle has incomplete value in %d bytes", name, len(b))
	}
	if err := oracleComparisonShape(len(got), len(b)/4); err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	for i, value := range got {
		want := int32(binary.LittleEndian.Uint32(b[4*i:]))
		if value != want {
			t.Fatalf("%s[%d] = %d, want %d", name, i, value, want)
		}
	}
}

// compareOracleMaskedF32 compares an attention or indexer score state, the one
// stage kind whose oracle carries the -Inf mask sentinel in every slot outside
// the live window. In the pinned position-0 capture the sentinel fills 7168 of
// 8192 slots on layer 2 and 65024 of 65536 on layer 3, the latter leaving
// exactly the indexer's top 512, so rejecting it outright leaves those layers
// unmeasured.
func compareOracleMaskedF32(t *testing.T, name string, got []float32, path string) {
	t.Helper()
	compareOracleMaskedF32Tolerance(t, name, got, path, oracleAtol, oracleRtol)
}

// frozenAtol and frozenRtol are the project's acceptance tolerance. They are
// pre-registered: a gate that fails at these numbers is a finding, never a
// reason to move them.
const (
	frozenAtol = 1e-4
	frozenRtol = 1e-4
)

func compareOracleMaskedF32Tolerance(t *testing.T, name string, got []float32, path string, atol, rtol float64) {
	t.Helper()
	live, err := maskedComparison(got, readOracleF32(t, path), atol, rtol)
	if err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	t.Logf("%s: compared %d live of %d slots", name, live, len(got))
}

// maskedComparison compares got against a want that carries the -Inf mask
// sentinel, and reports how many slots were live.
//
// The sentinel is data, so both sides must agree on the mask as well as on the
// finite values: the mask pattern is compared element for element, and the live
// population must be non-empty so a fully masked comparison cannot pass by
// having nothing left to compare. Every other comparator keeps its finite-only
// contract.
func maskedComparison(got, want []float32, atol, rtol float64) (int, error) {
	if err := oracleComparisonShape(len(got), len(want)); err != nil {
		return 0, err
	}
	live := 0
	for i := range got {
		masked := math.IsInf(float64(want[i]), -1)
		if masked != math.IsInf(float64(got[i]), -1) {
			return 0, fmt.Errorf("element %d: got %g, want %g: mask pattern differs", i, got[i], want[i])
		}
		if masked {
			continue
		}
		live++
		if !comparableValue(got[i]) || !comparableValue(want[i]) {
			return 0, fmt.Errorf("element %d: non-finite values %g, %g", i, got[i], want[i])
		}
		delta := math.Abs(float64(got[i] - want[i]))
		limit := atol + rtol*math.Abs(float64(want[i]))
		if delta > limit {
			return 0, fmt.Errorf("element %d: got %g, want %g (delta %g > %g)", i, got[i], want[i], delta, limit)
		}
	}
	if live == 0 {
		return 0, fmt.Errorf("every slot is masked, so the comparison measures nothing")
	}
	return live, nil
}

// oracleAtol and oracleRtol are the pre-registered tolerance for an ordinary
// float32 oracle comparison.
const (
	oracleAtol = 3e-3
	oracleRtol = 5e-4
)

func compareOracleF32(t *testing.T, name string, got []float32, path string) {
	compareOracleF32Tolerance(t, name, got, path, oracleAtol, oracleRtol)
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
	if len(b)%4 != 0 {
		t.Fatalf("%s oracle has incomplete value in %d bytes", name, len(b))
	}
	if err := oracleComparisonShape(len(got), len(b)/4); err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	for i := range got {
		want := math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
		requireOracleFinite(t, name, i, got[i], want)
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
	if len(b)%4 != 0 {
		t.Fatalf("%s oracle has incomplete value in %d bytes", name, len(b))
	}
	if err := oracleComparisonShape(len(got), len(b)/4); err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	for i := range got {
		want := math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
		requireOracleFinite(t, name, i, got[i], want)
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

// chainedDriftBudget bounds the relative divergence of the free-running
// residual stream from the teacher-forced oracle. Measured growth is smooth and
// roughly geometric -- 5.2e-4 at layer 1 rising to 6.6e-2 at layer 42, about
// 1.12x per layer with no step at any single layer. This is a flat ceiling, not
// an amplification bound: it catches a layer that drives drift past the ceiling,
// but a sharp jump that stays underneath it -- 5e-4 to 1.4e-1 at one layer, say
// -- passes. Bounding per-layer growth would need a per-layer budget derived
// from that curve, which is not yet registered as a contract.
// Measured against the pinned DeepSeek-V4-Flash IQ2_XXS checkpoint at prefill
// position 0, the oracle DS4_ALL_LAYERS_ORACLE_DIR captures.
const chainedDriftBudget = 1.5e-1

// relativeDrift reports the largest relative deviation of got from want,
// ignoring want values too small for a relative measure to be meaningful.
func relativeDrift(t *testing.T, got, want []float32) float64 {
	t.Helper()
	if err := oracleComparisonShape(len(got), len(want)); err != nil {
		t.Fatal(err)
	}
	worst := 0.0
	for i := range got {
		requireOracleFinite(t, "relative drift", i, got[i], want[i])
		w := math.Abs(float64(want[i]))
		if w <= 0.1 {
			continue
		}
		worst = max(worst, math.Abs(float64(got[i]-want[i]))/w)
	}
	return worst
}

// residualULPBudget is how many float32 ULPs of a residual tensor's own peak
// magnitude the comparison allows. The hyper-connection residual stream grows
// with depth, peaking near 6.3e4 by layer 42, where one float32 ULP is already
// 3.9e-3 -- larger than the fixed 3e-3 absolute tolerance the earlier stages
// use. An element's error is set by the scale of the accumulation it came out
// of, not by its own magnitude, so a mid-magnitude element in a tensor with
// that peak cannot be resolved to 3e-3 in float32 by any implementation.
// Measured worst deviations against the same pinned checkpoint and oracle as
// chainedDriftBudget are 0.7 ULP at layer 40 and 4.2 ULP at layer 42; every
// other layer is exact within the fixed tolerance.

// comparableValue reports whether v can take part in a numeric comparison.
// Every comparison below is of the form delta > limit, which is false when
// delta is NaN, so a non-finite value must be rejected explicitly.
func comparableValue(v float32) bool {
	return !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0)
}

// requireOracleFinite fails the test on a value that no comparison can accept. The
// -Inf sentinel is legitimate data in exactly one place, the attention and
// indexer score state, and compareOracleMaskedF32 handles it there; everywhere
// else a non-finite value on either side is a defect.
func requireOracleFinite(t *testing.T, name string, i int, got, want float32) {
	t.Helper()
	if !comparableValue(got) {
		t.Fatalf("%s[%d] = %g, want %g", name, i, got, want)
	}
	if !comparableValue(want) {
		t.Fatalf("%s[%d] = %g, oracle %g", name, i, got, want)
	}
}

const (
	residualAtol = 3e-3
	residualRtol = 5e-4

	residualULPBudget = 8
)

// compareOracleResidual compares a hyper-connection residual tensor. It applies
// the same absolute and relative tolerances as compareOracleF32 plus a term for
// the tensor's dynamic range. Stages that are not residual accumulations keep
// the fixed tolerance alone.
func compareOracleResidual(t *testing.T, name string, got []float32, path string) {
	t.Helper()
	want := readOracleF32(t, path)
	if err := oracleComparisonShape(len(got), len(want)); err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	peak := 0.0
	for i := range got {
		peak = max(peak, math.Abs(float64(want[i])))
	}
	ulp := 0.0
	if peak != 0 {
		_, exponent := math.Frexp(peak)
		ulp = math.Ldexp(1, exponent-24)
	}
	scale := residualAtol + residualULPBudget*ulp
	for i := range got {
		requireOracleFinite(t, name, i, got[i], want[i])
		limit := scale + residualRtol*math.Abs(float64(want[i]))
		if delta := math.Abs(float64(got[i] - want[i])); delta > limit {
			t.Fatalf("%s[%d] = %g, want %g (delta %g > %g at peak %g)", name, i, got[i], want[i], delta, limit, peak)
		}
	}
}

// runPinnedOutputHead compares the tail of the output head against the oracle.
// The head is not part of a layer, so the per-layer sweep stops before it, yet
// it is where logits are produced and therefore where a full-model divergence
// is measured.
//
// It starts from the oracle's own collapsed embedding rather than from a layer
// state. The capture labels every head stage pos0, but that 0 is a literal
// argument at ds4.c:26293 rather than a position: the head reads
// metal_graph_cur_hc(g), which after prefill holds the last prompt token, not
// position 0. The head stages are therefore mutually consistent but share no
// position with the layer dumps, so feeding one a layer-42 state compares two
// different tokens. Independent double-precision evaluation of
// output_hc_fn @ rmsnorm(hc_ffn_post-42) confirms this: it reproduces the Go
// projection to seven digits and matches no head dump, and no other
// 16384-float tensor in the capture reproduces result_hc_pre either.
//
// What remains pinned is the chain from result_hc onward, which is where the
// vocabulary projection lives. The arithmetic mirrors the tail of
// metalRuntime.outputLogits.
func runPinnedOutputHead(t *testing.T, ops *metalPinnedOps, bind func(string) boundMetalTensor, oracleDir string) {
	t.Helper()
	oracle := func(stage string) string {
		return fmt.Sprintf("%s/stage_result_%s-43_pos0.bin", oracleDir, stage)
	}
	collapsed := readOracleF32(t, oracle("hc"))
	normed, err := ops.RMS(collapsed, bind("output_norm.weight"), 1, true)
	if err != nil {
		t.Fatal(err)
	}
	compareOracleF32(t, "result_norm", normed, oracle("norm"))
	logits, err := ops.MatvecQ80(bind("output.weight"), normed)
	if err != nil {
		t.Fatal(err)
	}
	compareOracleF32(t, "result_output", logits, oracle("output"))
}

// pinnedOraclePosition reports the token position the DS4_POS_ORACLE_DIR
// capture was taken at. The capture is named for it and the value must match,
// because position is an input to the rotation under test: comparing a capture
// taken at one position against a rotation computed for another would fail for
// the right reason but tell us nothing about the implementation.
func pinnedOraclePosition(t *testing.T) int {
	t.Helper()
	raw := os.Getenv("DS4_POS_ORACLE_POS")
	if raw == "" {
		t.Fatal("DS4_POS_ORACLE_DIR set without DS4_POS_ORACLE_POS")
	}
	pos, err := strconv.Atoi(raw)
	if err != nil {
		t.Fatalf("DS4_POS_ORACLE_POS=%q: %v", raw, err)
	}
	if pos <= 0 {
		t.Fatalf("DS4_POS_ORACLE_POS=%d is not a nonzero position", pos)
	}
	return pos
}

// runPinnedRopeAtPosition compares the query and key-value path at a nonzero
// token position, which the position-0 sweep cannot reach. Rotary embedding is
// a function of position, so every stage from the projections through Qcur and
// KVcur is a different computation at position 1 than at position 0, and none
// of it was measured before.
//
// The sweep stops at KVcur deliberately. kqv_out attends over the cache, and
// at a nonzero position that cache holds earlier tokens the teacher-forced
// harness does not reconstruct. Comparing it here would measure the harness
// rather than the model. The compressor state that would settle the cache is
// not capturable at a nonzero position at all: the oracle emits attn_state_kv
// and attn_state_score once per run tagged pos 0 whatever the real position,
// established by DS4_METAL_GRAPH_DUMP_TRACE showing 41 offers all at pos 0
// against 43 offers at each of pos 0..9 for ordinary stages.
//
// Each layer is teacher-forced from the capture's own attn_norm, so a failure
// names the layer that first diverges rather than the first layer downstream of
// a divergence.
func runPinnedRopeAtPosition(t *testing.T, ops *metalPinnedOps, bind func(string) boundMetalTensor, oracleDir string, pos int) {
	t.Helper()
	for layer := range 43 {
		t.Run(fmt.Sprintf("layer-%d", layer), func(t *testing.T) {
			// RopeAttentionOne takes no layer and applies the uncompressed
			// rotation. ds4.c:1159 gives layers 0 and 1 a compress ratio of 0
			// and every later layer a nonzero one, and a compressed layer uses
			// the long-context frequency base and scale, so this helper is the
			// right rotation only for the first two layers. At position 0 the
			// distinction is invisible because every rotation is the identity
			// there, which is why the existing sweep never noticed.
			//
			// Running it anyway on a compressed layer produces a confident
			// failure that measures the harness, not the model, so a compressed
			// layer goes through RopeAttentionMixed instead, which does take a
			// layer and picks the rotation from it.
			name := func(s string) string { return fmt.Sprintf("blk.%d.%s", layer, s) }
			oracle := func(stage string) string {
				return fmt.Sprintf("%s/stage_%s-%d_pos%d.bin", oracleDir, stage, layer, pos)
			}
			attnNorm := readOracleF32(t, oracle("attn_norm"))
			qRaw, err := ops.MatvecQ80(bind(name("attn_q_a.weight")), attnNorm)
			if err != nil {
				t.Fatal(err)
			}
			compareOracleF32Tolerance(t, "q_lora", qRaw, oracle("q_lora"), frozenAtol, frozenRtol)
			qNormWeight := bind(name("attn_q_a_norm.weight"))
			qNorm, err := ops.RMS(qRaw, qNormWeight, 1, true)
			if err != nil {
				t.Fatal(err)
			}
			compareOracleF32Tolerance(t, "q_lora_norm", qNorm, oracle("q_lora_norm"), frozenAtol, frozenRtol)
			qHeadsRaw, err := ops.MatvecQ80(bind(name("attn_q_b.weight")), qNorm)
			if err != nil {
				t.Fatal(err)
			}
			compareOracleF32Tolerance(t, "Qraw", qHeadsRaw, oracle("Qraw"), frozenAtol, frozenRtol)
			qHeads, err := ops.RMS(qHeadsRaw, qNormWeight, 64, false)
			if err != nil {
				t.Fatal(err)
			}
			compareOracleF32Tolerance(t, "Qnorm", qHeads, oracle("Qnorm"), frozenAtol, frozenRtol)
			kvRaw, err := ops.MatvecQ80(bind(name("attn_kv.weight")), attnNorm)
			if err != nil {
				t.Fatal(err)
			}
			compareOracleF32Tolerance(t, "KVraw", kvRaw, oracle("KVraw"), frozenAtol, frozenRtol)
			kvNorm, err := ops.RMS(kvRaw, bind(name("attn_kv_a_norm.weight")), 1, true)
			if err != nil {
				t.Fatal(err)
			}
			compareOracleF32Tolerance(t, "KVnorm", kvNorm, oracle("KVnorm"), frozenAtol, frozenRtol)
			sinks := bind(name("attn_sinks.weight"))
			var qCur, kvCur []float32
			if layer < 2 {
				qCur, kvCur, _, err = ops.RopeAttentionOne(qHeads, kvNorm, sinks, uint32(pos))
			} else {
				// The rotation and the FP8 round of the current token do not
				// read the caches; they only write into them. Empty caches are
				// therefore enough to reach Qcur and KVcur, and leaving the
				// compressed cache empty keeps this a rotation measurement
				// rather than an attention one.
				a := metalmem.NewArena(ops.dev)
				defer a.Free()
				raw := metalmem.AllocIn[float32](a, "raw-kv", (pos+1)*512)
				comp := metalmem.AllocIn[float32](a, "comp-kv", 512)
				if e := a.Err(); e != nil {
					t.Fatal(e)
				}
				qCur, kvCur, _, err = ops.RopeAttentionMixed(qHeads, kvNorm, sinks, raw, comp, 0, nil, pos, layer)
			}
			if err != nil {
				t.Fatal(err)
			}
			compareOracleF32Tolerance(t, "Qcur", qCur, oracle("Qcur"), frozenAtol, frozenRtol)
			// KVcur is left unasserted on purpose. compareOracleFP8F32 accepts
			// an adjacent E4M3 code, which is a real rounding allowance and not
			// a tolerance, so it cannot stand in for the exact stored-cell
			// identity this gate would need. Asserting it at the ordinary
			// tolerance instead would be inventing an acceptance rule here.
			_ = kvCur
		})
	}
}

// allLayersOraclePosition reports the token position the all-layers sweep runs
// at. It defaults to 0, the position every existing capture was taken at.
func allLayersOraclePosition(t *testing.T) int {
	t.Helper()
	raw := os.Getenv("DS4_ALL_LAYERS_ORACLE_POS")
	if raw == "" {
		return 0
	}
	pos, err := strconv.Atoi(raw)
	if err != nil || pos < 0 {
		t.Fatalf("DS4_ALL_LAYERS_ORACLE_POS = %q, want a non-negative position", raw)
	}
	return pos
}

// pinnedCompressorRun reports the last position of the compressor-state sweep.
// The sweep needs a capture that dumped every position of one invocation, so it
// is gated on its own variable rather than on the single-position oracle dir.
func pinnedCompressorRun(t *testing.T) int {
	t.Helper()
	raw := os.Getenv("DS4_COMPRESSOR_ORACLE_LAST_POS")
	if raw == "" {
		t.Fatal("DS4_COMPRESSOR_ORACLE_LAST_POS must name the last position of the capture")
	}
	pos, err := strconv.Atoi(raw)
	if err != nil || pos <= 0 {
		t.Fatalf("DS4_COMPRESSOR_ORACLE_LAST_POS = %q, want a positive position", raw)
	}
	return pos
}

// runPinnedCompressorState drives the production compressor across positions 0
// through last and compares its state against the oracle after every position.
// It is the only gate that measures compressor state past position 0: the state
// is carried in a Metal buffer across calls, so a position-0 comparison cannot
// reach the pooling that happens when a window closes.
//
// Each position is teacher-forced from that position's own captured
// comp_kv_raw and comp_score_raw, so a divergence here is the compressor's and
// not an inherited one from the projections that feed it. Every input and every
// expectation comes from one invocation of the instrumented oracle.
func runPinnedCompressorState(t *testing.T, ops *metalPinnedOps, bind func(string) boundMetalTensor, oracleDir string, last int) {
	t.Helper()
	for layer := 2; layer < 43; layer++ {
		ratio, width, headDim := 4, 1024, 512
		if layer%2 != 0 {
			ratio, width, headDim = 128, 512, 512
		}
		lanes := []struct {
			name    string
			width   int
			headDim int
			indexer bool
		}{{"attn", width, headDim, false}}
		if ratio == 4 {
			lanes = append(lanes, struct {
				name    string
				width   int
				headDim int
				indexer bool
			}{"indexer", 256, 128, true})
		}
		for _, lane := range lanes {
			t.Run(fmt.Sprintf("layer-%d-%s", layer, lane.name), func(t *testing.T) {
				rows := ratio
				if ratio == 4 {
					rows = 2 * ratio
				}
				a := metalmem.NewArena(ops.dev)
				defer a.Free()
				state := metalCompressedState{
					cache:      metalmem.AllocIn[float32](a, "cache", (last/ratio+2)*lane.headDim),
					stateKV:    metalmem.AllocIn[float32](a, "state-kv", rows*lane.width),
					stateScore: metalmem.AllocIn[float32](a, "state-score", rows*lane.width),
					pooled:     metalmem.AllocIn[float32](a, "pooled", lane.headDim),
					headDim:    lane.headDim,
					width:      lane.width,
					ratio:      ratio,
					cap:        last/ratio + 2,
				}
				if err := a.Err(); err != nil {
					t.Fatal(err)
				}
				resetCompressedState(&state)
				prefix := fmt.Sprintf("blk.%d.%s_compressor_", layer, lane.name)
				ape := bind(prefix + "ape.weight")
				norm := bind(prefix + "norm.weight")
				stage := func(name string, pos int) string {
					return fmt.Sprintf("%s/stage_%s_%s-%d_pos%d.bin", oracleDir, lane.name, name, layer, pos)
				}
				for pos := 0; pos <= last; pos++ {
					kv := readOracleF32(t, stage("comp_kv_raw", pos))
					score := readOracleF32(t, stage("comp_score_raw", pos))
					_, emitted, err := ops.compressor(kv, score, ape, norm, &state, pos, lane.indexer)
					if err != nil {
						t.Fatalf("position %d: %v", pos, err)
					}
					if emitted {
						state.n++
					}
					what := fmt.Sprintf("%s_state_kv@%d", lane.name, pos)
					compareOracleF32Tolerance(t, what, append([]float32(nil), state.stateKV.HostSlice()...), stage("state_kv", pos), frozenAtol, frozenRtol)
					what = fmt.Sprintf("%s_state_score@%d", lane.name, pos)
					compareOracleMaskedF32Tolerance(t, what, append([]float32(nil), state.stateScore.HostSlice()...), stage("state_score", pos), frozenAtol, frozenRtol)
				}
			})
		}
	}
}

// dependentOracleVars name the oracle roots whose gates are layered on the
// layer-0 comparison. Setting one of them expresses an intent to measure, so
// reaching the layer-0 skip with one of them set means the run is silently
// weaker than the operator asked for.
var dependentOracleVars = []string{
	"DS4_ALL_LAYERS_ORACLE_DIR",
	"DS4_COMPRESSOR_ORACLE_DIR",
	"DS4_POS_ORACLE_DIR",
	"DS4_TRANSITION_ORACLE_ROOT",
}

// requireNoDependentOracles fails a partially supplied oracle environment
// rather than skipping it. A skip here reports success for a run that compared
// nothing, which is the shape that produced three false greens earlier in this
// project.
func requireNoDependentOracles(t *testing.T) {
	t.Helper()
	var set []string
	for _, name := range dependentOracleVars {
		if os.Getenv(name) != "" {
			set = append(set, name)
		}
	}
	if len(set) > 0 {
		t.Fatalf("partial oracle environment: %s set without DS4_LAYER0_ORACLE_DIR; "+
			"refusing to report a pass for a run that compares nothing", strings.Join(set, ", "))
	}
}

// requireOraclePopulated rejects an oracle root that is missing or empty. The
// per-file reads would fail anyway, but they fail one comparison at a time and
// name a single path; naming the root is what an operator can act on.
func requireOraclePopulated(t *testing.T, name, dir string) {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("%s = %q: %v", name, dir, err)
	}
	if len(entries) == 0 {
		t.Fatalf("%s = %q is empty", name, dir)
	}
}
