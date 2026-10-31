package model

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudahostemu"
	"github.com/tmc/lib/nvidia/x/egraph/hopper"
)

func TestParseArchConfig(t *testing.T) {
	TestArchParseConfig(t)
}

func TestCompileLayerSchedules(t *testing.T) {
	moeCfg := ArchConfig{
		HiddenSize:          1024,
		IntermediateSize:    2048,
		MoEIntermediateSize: 512,
		NumHiddenLayers:     24,
		NumAttentionHeads:   16,
		NumKeyValueHeads:    2,
		HeadDim:             64,
		NumLocalExperts:     4,
		NumExperts:          16,
		NumExpertsPerTok:    4,
		RMSNormEps:          1e-5,
		AttentionGate:       true,
		WeightDType:         NVFP4,
		KVDType:             E4M3,
	}

	t.Run("h200_hopper_schedules", func(t *testing.T) {
		clDecode, err := CompileLayer(moeCfg, 4, 1024, "NVIDIA H200")
		if err != nil {
			t.Fatalf("CompileLayer H200 decode: %v", err)
		}
		if clDecode.Mode != ModePersistentMegakernel {
			t.Errorf("Mode = %q, want %q", clDecode.Mode, ModePersistentMegakernel)
		}
		if clDecode.MegakernelSource == "" || len(clDecode.Megakernels) == 0 {
			t.Errorf("expected non-empty MegakernelSource for batchM=4")
		}
		if clDecode.CUDASource == "" || clDecode.TVMFFISource == "" {
			t.Errorf("expected non-empty CUDASource and TVMFFISource")
		}

		var (
			sawGatedAttn    bool
			sawWGMMADequant bool
			sawExpertSum    bool
		)
		for _, k := range clDecode.Plan.Kernels {
			if k.Attention != nil && k.Attention.Gate && k.Attention.Swizzle {
				sawGatedAttn = true
			}
			if k.Tile != nil && k.Tile.Weight == hopper.WeightNVFP4 {
				if k.Tile.WGMMA && k.Tile.Repacked && k.Tile.DequantWords == 2 && !k.Tile.BlackwellFP4 {
					sawWGMMADequant = true
				}
			}
			if k.ExpertSum != nil {
				sawExpertSum = true
			}
		}
		if !sawGatedAttn {
			t.Errorf("H200 plan did not fuse gated PagedAttention{Gate: true, Swizzle: true}")
		}
		if !sawWGMMADequant {
			t.Errorf("H200 plan did not select WGMMA + Repacked + DequantWords=2 on NVFP4 TileGEMM")
		}
		if !sawExpertSum {
			t.Errorf("H200 MoE plan did not fuse ExpertSum")
		}

		clPrefill, err := CompileLayer(moeCfg, 128, 1024, "NVIDIA H200")
		if err != nil {
			t.Fatalf("CompileLayer H200 prefill: %v", err)
		}
		if clPrefill.Mode != ModeStreamKWarpSpec {
			t.Errorf("prefill Mode = %q, want %q", clPrefill.Mode, ModeStreamKWarpSpec)
		}
		for _, k := range clPrefill.Plan.Kernels {
			if k.Tile != nil && (!k.Tile.WGMMA || k.Tile.DequantWords != 2) {
				t.Errorf("H200 prefill TileGEMM %s: WGMMA=%v DequantWords=%d, want true, 2",
					k.Tile.Name, k.Tile.WGMMA, k.Tile.DequantWords)
			}
		}
	})

	t.Run("gb10_blackwell_schedules", func(t *testing.T) {
		clGB10, err := CompileLayer(moeCfg, 4, 1024, "NVIDIA GB10")
		if err != nil {
			t.Fatalf("CompileLayer GB10 decode: %v", err)
		}
		var (
			sawSkinnyBlackwellFP4 bool
			sawTileBlackwellFP4   bool
		)
		for _, k := range clGB10.Plan.Kernels {
			if k.GEMM != nil && k.GEMM.Weight == hopper.WeightNVFP4 {
				if !k.GEMM.BlackwellFP4 {
					t.Errorf("GB10 SkinnyGEMM %s: BlackwellFP4 = false, want true", k.GEMM.Name)
				}
				sawSkinnyBlackwellFP4 = true
			}
			if k.Tile != nil && k.Tile.Weight == hopper.WeightNVFP4 {
				if !k.Tile.BlackwellFP4 || k.Tile.WGMMA {
					t.Errorf("GB10 TileGEMM %s: BlackwellFP4=%v WGMMA=%v, want true, false",
						k.Tile.Name, k.Tile.BlackwellFP4, k.Tile.WGMMA)
				}
				sawTileBlackwellFP4 = true
			}
		}
		if !sawSkinnyBlackwellFP4 || !sawTileBlackwellFP4 {
			t.Errorf("GB10 schedules: sawSkinnyBlackwellFP4=%v sawTileBlackwellFP4=%v, want both true",
				sawSkinnyBlackwellFP4, sawTileBlackwellFP4)
		}
	})
}

func TestCompileMultiBucketLayer(t *testing.T) {
	cfg := ArchConfig{
		HiddenSize:        1024,
		IntermediateSize:  2048,
		NumHiddenLayers:   16,
		NumAttentionHeads: 16,
		NumKeyValueHeads:  4,
		HeadDim:           64,
		RMSNormEps:        1e-5,
	}
	buckets := []int{1, 16, 128}
	bundle, err := CompileMultiBucketLayer(cfg, buckets, 1024, "NVIDIA H200")
	if err != nil {
		t.Fatalf("CompileMultiBucketLayer: %v", err)
	}
	if len(bundle.Layers) != 3 {
		t.Fatalf("len(Layers) = %d, want 3", len(bundle.Layers))
	}
	if bundle.Layers[1].Mode != ModePersistentMegakernel {
		t.Errorf("M=1 mode = %q, want %q", bundle.Layers[1].Mode, ModePersistentMegakernel)
	}
	if bundle.Layers[16].Mode != ModeMultiLaunch {
		t.Errorf("M=16 mode = %q, want %q", bundle.Layers[16].Mode, ModeMultiLaunch)
	}
	if bundle.Layers[128].Mode != ModeStreamKWarpSpec {
		t.Errorf("M=128 mode = %q, want %q", bundle.Layers[128].Mode, ModeStreamKWarpSpec)
	}
	for _, want := range []string{
		"inline void dispatch_layer_m(int M,",
		"extern \"C\" void dispatch_layer_m_c(int M,",
		"if (M <= 1) {",
		"layer_m1_run(",
		"if (M <= 16) {",
		"layer_m16_run(",
		"layer_m128_run(",
	} {
		if !strings.Contains(bundle.CUDASource, want) {
			t.Errorf("CUDASource missing %q", want)
		}
	}
	if !strings.Contains(bundle.TVMFFISource, "namespace eg {") {
		t.Errorf("TVMFFISource missing namespace eg")
	}
}

// TestCompileLayerDenseEmulated compiles a complete dense GQA transformer
// decoder layer on small dimensions (M=2, H=64, I=128, G=4, KVH=2, D=16),
// executes every step on the CPU via cudahostemu.Run, and verifies finite,
// accurate outputs against both TermProgram.Eval and an independent Go
// reference implementation.
func TestCompileLayerDenseEmulated(t *testing.T) {
	const (
		M      = 2
		H      = 64
		I      = 128
		G      = 4
		KVH    = 2
		D      = 16
		seqLen = 32
		eps    = 1e-5
	)
	cfg := ArchConfig{
		HiddenSize:        H,
		IntermediateSize:  I,
		NumHiddenLayers:   2,
		NumAttentionHeads: G,
		NumKeyValueHeads:  KVH,
		HeadDim:           D,
		RMSNormEps:        eps,
		WeightDType:       BF16,
		KVDType:           BF16,
	}

	prog, err := BuildLayerProgram(cfg, M, seqLen)
	if err != nil {
		t.Fatalf("BuildLayerProgram: %v", err)
	}
	cl, err := CompileLayer(cfg, M, seqLen, "NVIDIA H200")
	if err != nil {
		t.Fatalf("CompileLayer: %v", err)
	}
	if cl.Mode != ModePersistentMegakernel {
		t.Fatalf("Mode = %q, want %q", cl.Mode, ModePersistentMegakernel)
	}
	if len(cl.Estimates) != len(cl.Plan.Kernels) || cl.TotalNs <= 0 {
		t.Fatalf("Estimates len=%d (plan=%d) TotalNs=%v", len(cl.Estimates), len(cl.Plan.Kernels), cl.TotalNs)
	}

	// Construct deterministic inputs for all layer tensors.
	r := rand.New(rand.NewPCG(501, 502))
	env := make(map[string]hopper.Value)
	for _, in := range cl.Plan.Inputs {
		switch in.Name {
		case "k_cache", "v_cache":
			data := make([]float64, tensorSize(in.Shape))
			for i := range data {
				v := float32(r.NormFloat64() * 0.25)
				data[i] = float64(hopper.E4M3Bits(v))
			}
			env[in.Name] = hopper.Value{Shape: in.Shape, Data: data}
		case "page_table":
			maxPages := in.Shape[1]
			data := make([]float64, M*maxPages)
			for i := range data {
				data[i] = float64(i)
			}
			env[in.Name] = hopper.Value{Shape: in.Shape, Data: data}
		case "seqlens":
			env[in.Name] = hopper.Value{Shape: in.Shape, Data: []float64{24, 19}}
		case "k_scale", "v_scale":
			env[in.Name] = hopper.Value{Shape: in.Shape, Data: []float64{0.125}}
		default:
			data := make([]float64, tensorSize(in.Shape))
			for i := range data {
				v := float32(r.NormFloat64() * 0.2)
				data[i] = float64(in.DType.Round(v))
			}
			env[in.Name] = hopper.Value{Shape: in.Shape, Data: data}
		}
	}

	// 1. Evaluate via TermProgram.Eval.
	evalVal, err := prog.Eval(env)
	if err != nil {
		t.Fatalf("prog.Eval: %v", err)
	}
	wantEval := roundAll(hopper.BF16, evalVal.Data)

	// 2. Evaluate via standalone Go reference function.
	wantRef := denseLayerGoReference(t, cfg, M, env)
	if e := relErr(wantEval, wantRef); e > 1e-4 || math.IsNaN(e) {
		t.Fatalf("TermProgram.Eval vs Go reference relative L2 error = %.2e > 1e-4", e)
	}

	// 3. Execute all compiled layer kernels on CPU via cudahostemu.Run (runPlan).
	got := runPlan(t, cl.Plan, env)
	if len(got) != M*H {
		t.Fatalf("len(got) = %d, want %d", len(got), M*H)
	}
	var normSq float64
	for i, v := range got {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Fatalf("emulated output[%d] = %v is not finite", i, v)
		}
		normSq += v * v
	}
	if normSq == 0 {
		t.Fatalf("emulated output is all zeros")
	}
	if e := relErr(got, wantRef); e > 5e-3 || math.IsNaN(e) {
		t.Fatalf("emulated layer vs Go reference relative L2 error = %.2e > 5e-3\ngot:  %v\nwant: %v",
			e, got[:8], wantRef[:8])
	}
	if e := relErr(got, wantEval); e > 5e-3 || math.IsNaN(e) {
		t.Fatalf("emulated layer vs Eval relative L2 error = %.2e > 5e-3", e)
	}
}

func denseLayerGoReference(t *testing.T, cfg ArchConfig, batchM int, env map[string]hopper.Value) []float64 {
	t.Helper()
	H := cfg.HiddenSize
	I := cfg.IntermediateSize
	hq := cfg.NumAttentionHeads
	hkv := cfg.NumKeyValueHeads
	qDim := hq * hopper.HeadDim
	eps := float64(cfg.RMSNormEps)

	rmsNormRef := func(x []float64, w []float64, rows, cols int) []float64 {
		out := make([]float64, rows*cols)
		for r := range rows {
			var ss float64
			for c := range cols {
				v := float64(float32(x[r*cols+c]))
				ss += v * v
			}
			inv := float64(1.0 / float32(math.Sqrt(ss/float64(cols)+eps)))
			for c := range cols {
				val := float32(x[r*cols+c]) * float32(inv) * float32(w[c])
				out[r*cols+c] = float64(hopper.BF16.Round(val))
			}
		}
		return out
	}

	matmulRef := func(x, w []float64, m, n, k int) []float64 {
		out := make([]float64, m*n)
		for r := range m {
			for col := range n {
				var sum float64
				for c := range k {
					sum += float64(hopper.BF16.Round(float32(x[r*k+c]))) * float64(hopper.BF16.Round(float32(w[col*k+c])))
				}
				out[r*n+col] = float64(float32(sum))
			}
		}
		return out
	}

	x := env["x"].Data
	norm1 := rmsNormRef(x, env["attn_norm_w"].Data, batchM, H)
	qRaw := matmulRef(norm1, env["wq"].Data, batchM, qDim, H)
	qBF16 := make([]float32, len(qRaw))
	for i, v := range qRaw {
		qBF16[i] = hopper.BF16.Round(float32(v))
	}

	// PagedAttention reference.
	kv := env["k_cache"].Data
	vv := env["v_cache"].Data
	pt := env["page_table"]
	sl := env["seqlens"].Data
	kb := make([]uint8, len(kv))
	vb := make([]uint8, len(vv))
	for i := range kv {
		kb[i] = uint8(kv[i])
		vb[i] = uint8(vv[i])
	}
	pti := make([]int32, len(pt.Data))
	for i, v := range pt.Data {
		pti[i] = int32(v)
	}
	sli := make([]int32, len(sl))
	for i, v := range sl {
		sli[i] = int32(v)
	}
	attnCfg := hopper.PagedAttention{
		Name: "ref_attn",
		Hq:   hq,
		Hkv:  hkv,
		QLen: 1,
		Page: 16,
		Q:    hopper.BF16,
		Out:  hopper.F32,
	}
	attnOutF64, err := attnCfg.Reference(qBF16, hopper.PagedCache{
		K:         kb,
		V:         vb,
		PageTable: pti,
		MaxPages:  pt.Shape[1],
		SeqLens:   sli,
		KScale:    float32(env["k_scale"].Data[0]),
		VScale:    float32(env["v_scale"].Data[0]),
	})
	if err != nil {
		t.Fatalf("PagedAttention.Reference: %v", err)
	}
	attnBF16 := roundAll(hopper.BF16, attnOutF64)

	// OProj + residual add.
	oProj := matmulRef(attnBF16, env["wo"].Data, batchM, H, qDim)
	h1 := make([]float64, batchM*H)
	for i := range h1 {
		h1[i] = float64(float32(oProj[i]) + float32(x[i]))
	}

	// Post-attention RMSNorm + Dense MLP (GateUp -> SwiGLU -> Down).
	norm2 := rmsNormRef(h1, env["ffn_norm_w"].Data, batchM, H)
	gateUp := matmulRef(norm2, env["w_gate_up"].Data, batchM, 2*I, H)
	silu := hopper.MustLambda("(mul (silu $0) $1)")
	act := make([]float64, batchM*I)
	for r := range batchM {
		for c := range I {
			g := float32(gateUp[r*2*I+c])
			u := float32(gateUp[r*2*I+I+c])
			act[r*I+c] = float64(hopper.BF16.Round(silu.Eval(g, u)))
		}
	}
	down := matmulRef(act, env["w_down"].Data, batchM, H, I)

	// ResidualRMSNorm.
	res := make([]float64, batchM*H)
	for i := range res {
		res[i] = float64(float32(down[i]) + float32(h1[i]))
	}
	return rmsNormRef(res, env["post_norm_w"].Data, batchM, H)
}

// TestCompileLayerMoEEmulated compiles a complete routed-MoE + gated-attention
// transformer decoder layer on small dimensions (M=2, H=64, MoEI=64, G=4, KVH=2,
// Local=2, Experts=32, TopK=2), executes every step on the CPU via cudahostemu.Run,
// and verifies outputs against TermProgram.Eval.
func TestCompileLayerMoEEmulated(t *testing.T) {
	const (
		M      = 2
		H      = 64
		MoEI   = 64
		G      = 4
		KVH    = 2
		D      = 16
		seqLen = 32
		eps    = 1e-5
	)
	cfg := ArchConfig{
		HiddenSize:          H,
		IntermediateSize:    128,
		MoEIntermediateSize: MoEI,
		NumHiddenLayers:     2,
		NumAttentionHeads:   G,
		NumKeyValueHeads:    KVH,
		HeadDim:             D,
		NumLocalExperts:     2,
		NumExperts:          32,
		NumExpertsPerTok:    2,
		RMSNormEps:          eps,
		AttentionGate:       true,
		WeightDType:         BF16,
		KVDType:             BF16,
	}

	prog, err := BuildLayerProgram(cfg, M, seqLen)
	if err != nil {
		t.Fatalf("BuildLayerProgram: %v", err)
	}
	cl, err := CompileLayer(cfg, M, seqLen, "NVIDIA H200")
	if err != nil {
		t.Fatalf("CompileLayer: %v", err)
	}

	r := rand.New(rand.NewPCG(601, 602))
	env := make(map[string]hopper.Value)
	for _, in := range cl.Plan.Inputs {
		switch in.Name {
		case "k_cache", "v_cache":
			data := make([]float64, tensorSize(in.Shape))
			for i := range data {
				v := float32(r.NormFloat64() * 0.25)
				data[i] = float64(hopper.E4M3Bits(v))
			}
			env[in.Name] = hopper.Value{Shape: in.Shape, Data: data}
		case "page_table":
			maxPages := in.Shape[1]
			data := make([]float64, M*maxPages)
			for i := range data {
				data[i] = float64(i)
			}
			env[in.Name] = hopper.Value{Shape: in.Shape, Data: data}
		case "seqlens":
			env[in.Name] = hopper.Value{Shape: in.Shape, Data: []float64{20, 16}}
		case "k_scale", "v_scale":
			env[in.Name] = hopper.Value{Shape: in.Shape, Data: []float64{0.125}}
		case "local_start":
			env[in.Name] = hopper.Value{Shape: in.Shape, Data: []float64{0}}
		case "router_logits":
			data := make([]float64, tensorSize(in.Shape))
			for i := range data {
				data[i] = float64(float32(r.NormFloat64() * 0.5))
			}
			// Boost local experts 0 and 1 so top-2 routes hit local experts.
			for tok := range M {
				data[tok*in.Shape[1]+0] = 4.0
				data[tok*in.Shape[1]+1] = 3.5
			}
			env[in.Name] = hopper.Value{Shape: in.Shape, Data: data}
		case "router_bias":
			env[in.Name] = hopper.Value{Shape: in.Shape, Data: make([]float64, tensorSize(in.Shape))}
		default:
			data := make([]float64, tensorSize(in.Shape))
			for i := range data {
				v := float32(r.NormFloat64() * 0.2)
				data[i] = float64(in.DType.Round(v))
			}
			env[in.Name] = hopper.Value{Shape: in.Shape, Data: data}
		}
	}

	evalVal, err := prog.Eval(env)
	if err != nil {
		t.Fatalf("prog.Eval: %v", err)
	}
	wantEval := roundAll(hopper.BF16, evalVal.Data)

	got := runPlan(t, cl.Plan, env)
	if len(got) != M*H {
		t.Fatalf("len(got) = %d, want %d", len(got), M*H)
	}
	for i, v := range got {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Fatalf("emulated MoE output[%d] = %v is not finite", i, v)
		}
	}
	if e := relErr(got, wantEval); e > 5e-3 || math.IsNaN(e) {
		t.Fatalf("emulated MoE layer vs Eval relative L2 error = %.2e > 5e-3\ngot:  %v\nwant: %v",
			e, got[:8], wantEval[:8])
	}
}

func roundAll(d hopper.DType, v []float64) []float64 {
	out := make([]float64, len(v))
	for i, f := range v {
		out[i] = float64(d.Round(float32(f)))
	}
	return out
}

func relErr(got, want []float64) float64 {
	var num, den float64
	for i := range want {
		num += (got[i] - want[i]) * (got[i] - want[i])
		den += want[i] * want[i]
	}
	return math.Sqrt(num / den)
}

func u16Bytes(v []uint16) []byte {
	return EncodeU16LE(v)
}

func bytesU16(b []byte) []uint16 {
	v := make([]uint16, len(b)/2)
	for i := range v {
		v[i] = binary.LittleEndian.Uint16(b[2*i:])
	}
	return v
}

func i32Bytes(v []int32) []byte {
	return EncodeI32LE(v)
}

func i64Bytes(v []int64) []byte {
	b := make([]byte, 8*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint64(b[8*i:], uint64(x))
	}
	return b
}

func u64Bytes(v []uint64) []byte {
	b := make([]byte, 8*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint64(b[8*i:], x)
	}
	return b
}

func f32Bytes(v []float32) []byte {
	return EncodeF32LE(v)
}

func bytesF32(b []byte) []float32 {
	v := make([]float32, len(b)/4)
	for i := range v {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return v
}

func encodeTyped(dt hopper.DType, vals []float64) []byte {
	switch dt {
	case hopper.BF16:
		v := make([]uint16, len(vals))
		for i, f := range vals {
			v[i] = hopper.BF16Bits(float32(f))
		}
		return u16Bytes(v)
	case hopper.E4M3:
		v := make([]byte, len(vals))
		for i, f := range vals {
			v[i] = hopper.E4M3Bits(float32(f))
		}
		return v
	case hopper.U8:
		v := make([]byte, len(vals))
		for i, f := range vals {
			v[i] = byte(f)
		}
		return v
	case hopper.I64:
		v := make([]int64, len(vals))
		for i, f := range vals {
			v[i] = int64(f)
		}
		return i64Bytes(v)
	case hopper.I32:
		v := make([]int32, len(vals))
		for i, f := range vals {
			v[i] = int32(f)
		}
		return i32Bytes(v)
	default:
		v := make([]float32, len(vals))
		for i, f := range vals {
			v[i] = float32(f)
		}
		return f32Bytes(v)
	}
}

func decodeTyped(dt hopper.DType, raw []byte) []float64 {
	var out []float64
	switch dt {
	case hopper.BF16:
		for _, b := range bytesU16(raw) {
			out = append(out, float64(hopper.FromBF16Bits(b)))
		}
	case hopper.E4M3:
		for _, b := range raw {
			out = append(out, float64(hopper.FromE4M3Bits(b)))
		}
	default:
		for _, f := range bytesF32(raw) {
			out = append(out, float64(f))
		}
	}
	return out
}

func agree(t *testing.T, name string, d hopper.DType, got, want []float64) {
	t.Helper()
	rel := map[hopper.DType]float64{hopper.F32: 1e-4, hopper.BF16: 1.0 / 64, hopper.E4M3: 1.0 / 4}[d]
	bad := 0
	for i := range want {
		if diff := math.Abs(got[i] - want[i]); diff > rel*math.Abs(want[i])+1e-6 || math.IsNaN(got[i]) {
			if bad < 3 {
				t.Errorf("%s[%d] = %g, want %g", name, i, got[i], want[i])
			}
			bad++
		}
	}
	if bad > 0 {
		t.Errorf("%s: %d of %d values differ", name, bad, len(want))
	}
}

func randEnv(r *rand.Rand, ts ...hopper.Tensor) map[string]hopper.Value {
	env := make(map[string]hopper.Value)
	for _, t := range ts {
		v := make([]float64, tensorSize(t.Shape))
		for i := range v {
			v[i] = float64(t.DType.Round(float32(r.NormFloat64())))
		}
		env[t.Name] = hopper.Value{Shape: t.Shape, Data: v}
	}
	return env
}

func attnCounterBytes(a hopper.PagedAttention, batch, splits int) int {
	if a.ClusterCombine {
		splits /= hopper.ClusterSize
	}
	const attnGroup = 8
	ng := (splits + attnGroup - 1) / attnGroup
	return 4 * batch * a.Hkv / max(a.KVHeadsPerBlock, 1) * (1 + ng)
}

func runPlan(t *testing.T, p *hopper.Plan, env map[string]hopper.Value) []float64 {
	t.Helper()
	var src strings.Builder
	src.WriteString(hopper.Prelude)
	seenHelpers := make(map[string]bool)
	var launches []string
	wsZero := make(map[string]int)
	for _, k := range p.Kernels {
		if k.GEMM != nil {
			s, err := k.GEMM.CUDA()
			if err != nil {
				t.Fatal(err)
			}
			src.WriteString(hopper.DeduplicateCUDAHelpers(s, seenHelpers))
			grid, block, _ := k.GEMM.Grid()
			launches = append(launches, fmt.Sprintf("eg_emu_launch(dim3(%d, %d), dim3(%d), 0, %s<%d>, %s)",
				grid[0], grid[1], block[0], k.GEMM.Name, k.M, strings.Join(k.Args, ", ")))
			continue
		}
		if k.Tile != nil {
			s, err := k.Tile.CUDA()
			if err != nil {
				t.Fatal(err)
			}
			src.WriteString(hopper.DeduplicateCUDAHelpers(s, seenHelpers))
			if k.Tile.Grouped {
				grid, block := k.Tile.Grid(k.MaxRows, k.Experts)
				launches = append(launches, fmt.Sprintf("eg_emu_launch(dim3(%d, %d, %d), dim3(%d), %d, %s, %d, %s)",
					grid[0], grid[1], grid[2], block[0], k.Tile.SharedBytes(), k.Tile.Name, k.M, strings.Join(k.Args, ", ")))
			} else {
				grid, block := k.Tile.Grid(k.M, 1)
				launches = append(launches, fmt.Sprintf("eg_emu_launch(dim3(%d, %d), dim3(%d), %d, %s, %d, %s)",
					grid[0], grid[1], block[0], k.Tile.SharedBytes(), k.Tile.Name, k.M, strings.Join(k.Args, ", ")))
			}
			continue
		}
		if k.Attention != nil {
			s, err := k.Attention.CUDA()
			if err != nil {
				t.Fatal(err)
			}
			l, err := k.Attention.Launcher()
			if err != nil {
				t.Fatal(err)
			}
			src.WriteString(s + "\n" + l)
			args := append([]string(nil), k.Args[:4]...)
			args = append(args, fmt.Sprint(k.MaxPages))
			args = append(args, k.Args[4:]...)
			launches = append(launches, fmt.Sprintf("eg_emu_sms = 16; eg_emu_blocks_per_sm = 1; %s_launch(%d, %d, %s, nullptr)",
				k.Attention.Name, k.Batch, k.MaxKeys, strings.Join(args, ", ")))
			wsName := k.Args[len(k.Args)-1]
			wsZero[wsName] = attnCounterBytes(*k.Attention, k.Batch, k.Attention.Splits(k.MaxKeys))
			continue
		}
		if k.TopK != nil {
			s, err := k.TopK.CUDA()
			if err != nil {
				t.Fatal(err)
			}
			src.WriteString(s)
			grid, block := k.TopK.Grid(k.Rows)
			launches = append(launches, fmt.Sprintf("eg_emu_launch(dim3(%d), dim3(%d), 0, %s, %d, %s)",
				grid[0], block[0], k.TopK.Name, k.Rows, strings.Join(k.Args, ", ")))
			continue
		}
		if k.Align != nil {
			s, err := k.Align.CUDA()
			if err != nil {
				t.Fatal(err)
			}
			src.WriteString(s)
			al, _ := k.Align.Norm()
			launches = append(launches, fmt.Sprintf("eg_emu_launch(dim3(1), dim3(%d), 0, %s, %d, %s)",
				32*al.Warps, k.Align.Name, k.Rows, strings.Join(k.Args, ", ")))
			continue
		}
		if k.ExpertSum != nil {
			s, err := k.ExpertSum.CUDA()
			if err != nil {
				t.Fatal(err)
			}
			src.WriteString(s)
			grid, block := k.ExpertSum.Grid(k.Rows)
			launches = append(launches, fmt.Sprintf("eg_emu_launch(dim3(%d), dim3(%d), 0, %s, %d, %s)",
				grid[0], block[0], k.ExpertSum.Name, k.Rows, strings.Join(k.Args, ", ")))
			continue
		}
		s, err := k.Row.CUDA()
		if err != nil {
			t.Fatal(err)
		}
		src.WriteString(s)
		if k.Row.Cluster {
			l, err := k.Row.Launcher()
			if err != nil {
				t.Fatal(err)
			}
			src.WriteString("\n" + l)
			launches = append(launches, fmt.Sprintf("%s_launch(%d, %s, nullptr)",
				k.Row.Name, k.Rows, strings.Join(k.Args, ", ")))
			continue
		}
		grid, block := k.Row.Grid(k.Rows)
		launches = append(launches, fmt.Sprintf("eg_emu_launch(dim3(%d), dim3(%d), 0, %s, %d, %s)",
			grid[0], block[0], k.Row.Name, k.Rows, strings.Join(k.Args, ", ")))
	}

	var cbufs []cudahostemu.Buffer
	for _, in := range p.Inputs {
		cbufs = append(cbufs, cudahostemu.Buffer{
			Name:  in.Name,
			CType: in.DType.CType(),
			Data:  encodeTyped(in.DType, env[in.Name].Data),
		})
	}
	for _, tmp := range append(p.Temps, p.Output) {
		raw := cudahostemu.Poison(tensorSize(tmp.Shape) * tmp.DType.Bits() / 8)
		if z, ok := wsZero[tmp.Name]; ok && z <= len(raw) {
			clear(raw[:z])
		}
		if tmp.DType == hopper.I32 && strings.HasSuffix(tmp.Name, "_count") {
			clear(raw)
		}
		cbufs = append(cbufs, cudahostemu.Buffer{
			Name:  tmp.Name,
			CType: tmp.DType.CType(),
			Data:  raw,
		})
	}

	res, err := cudahostemu.Run(t.Context(), t.TempDir(), cudahostemu.Program{
		Source:  src.String(),
		Launch:  strings.Join(launches, ";\n    "),
		Buffers: cbufs,
		Flags:   strings.Fields(os.Getenv("EG_EMU_FLAGS")),
	})
	if errors.Is(err, cudahostemu.ErrNoCompiler) {
		t.Skip("no host C++ compiler; emulated kernel not run")
	}
	if err != nil {
		t.Fatalf("emulated kernel: %v", err)
	}
	return decodeTyped(p.Output.DType, res[p.Output.Name])
}
