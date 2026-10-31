package main

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudahostemu"
	"github.com/tmc/lib/nvidia/x/egraph/hopper"
)

func encodeFullLayerTestTensor(dt hopper.DType, vals []float32) []byte {
	switch dt {
	case hopper.BF16:
		b := make([]byte, 2*len(vals))
		for i, v := range vals {
			binary.LittleEndian.PutUint16(b[2*i:], hopper.BF16Bits(v))
		}
		return b
	case hopper.F32:
		b := make([]byte, 4*len(vals))
		for i, v := range vals {
			binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(v))
		}
		return b
	case hopper.I32:
		b := make([]byte, 4*len(vals))
		for i, v := range vals {
			binary.LittleEndian.PutUint32(b[4*i:], uint32(int32(v)))
		}
		return b
	default:
		b := make([]byte, len(vals))
		for i, v := range vals {
			b[i] = hopper.E4M3Bits(v)
		}
		return b
	}
}

func makeFullLayerTestEnv(r *rand.Rand, p *hopper.Plan, batchM, maxPages int) map[string][]float32 {
	env := make(map[string][]float32, len(p.Inputs))
	for _, in := range p.Inputs {
		n := crossLayerTensorSize(in.Shape)
		vals := make([]float32, n)
		for i := range vals {
			v := float32((r.Float64()*2 - 1) * 0.25)
			vals[i] = in.DType.Round(v)
		}
		env[in.Name] = vals
	}
	setMeta := func(prefix string) {
		pos := make([]float32, batchM)
		seqIDs := make([]float32, batchM)
		seqLens := make([]float32, batchM)
		kvPos := make([]float32, batchM)
		pt := make([]float32, batchM*maxPages)
		for b := range batchM {
			pos[b] = 0
			seqIDs[b] = float32(b)
			seqLens[b] = 1
			kvPos[b] = 0
			for pg := range maxPages {
				pt[b*maxPages+pg] = float32(b*maxPages + pg)
			}
		}
		if _, ok := env[prefix+"positions"]; ok {
			env[prefix+"positions"] = pos
		}
		if _, ok := env[prefix+"seq_ids"]; ok {
			env[prefix+"seq_ids"] = seqIDs
		}
		if _, ok := env[prefix+"seqlens"]; ok {
			env[prefix+"seqlens"] = seqLens
		}
		if _, ok := env[prefix+"kv_positions"]; ok {
			env[prefix+"kv_positions"] = kvPos
		}
		if _, ok := env[prefix+"page_table"]; ok {
			env[prefix+"page_table"] = pt
		}
		if _, ok := env[prefix+"k_scale"]; ok {
			env[prefix+"k_scale"] = []float32{1.0}
		}
		if _, ok := env[prefix+"v_scale"]; ok {
			env[prefix+"v_scale"] = []float32{1.0}
		}
	}
	setMeta("")
	for l := range 8 {
		setMeta(fmt.Sprintf("l%d_", l))
	}
	return env
}

func runFullLayerPlanEmulated(t *testing.T, p *hopper.Plan, env map[string][]float32) []float32 {
	t.Helper()
	src, err := p.Source()
	if err != nil {
		t.Fatalf("%s.Source(): %v", p.Name, err)
	}
	var bufs []cudahostemu.Buffer
	var args []string
	for _, in := range p.Inputs {
		vals, ok := env[in.Name]
		if !ok {
			t.Fatalf("plan %s missing input %q", p.Name, in.Name)
		}
		bufs = append(bufs, cudahostemu.Buffer{
			Name:  in.Name,
			CType: crossLayerCType(in.DType),
			Data:  encodeFullLayerTestTensor(in.DType, vals),
		})
		args = append(args, in.Name)
	}
	for _, tmp := range p.Temps {
		nbytes := crossLayerTensorSize(tmp.Shape) * tmp.DType.Bits() / 8
		bufs = append(bufs, cudahostemu.Buffer{
			Name:  tmp.Name,
			CType: crossLayerCType(tmp.DType),
			Data:  make([]byte, nbytes),
		})
		args = append(args, tmp.Name)
	}
	outBytes := crossLayerTensorSize(p.Output.Shape) * p.Output.DType.Bits() / 8
	bufs = append(bufs, cudahostemu.Buffer{
		Name:  p.Output.Name,
		CType: crossLayerCType(p.Output.DType),
		Data:  cudahostemu.Poison(outBytes),
	})
	args = append(args, p.Output.Name, "nullptr")

	res, err := cudahostemu.Run(t.Context(), t.TempDir(), cudahostemu.Program{
		Source:  src,
		Launch:  fmt.Sprintf("%s_run(%s)", p.Name, strings.Join(args, ", ")),
		Buffers: bufs,
	})
	if errors.Is(err, cudahostemu.ErrNoCompiler) {
		t.Skip("no host C++ compiler; emulated plan not run")
	}
	if err != nil {
		t.Fatalf("emulate plan %s: %v", p.Name, err)
	}
	return decodeCrossLayerTensor(p.Output.DType, res[p.Output.Name])
}

func runFullLayerMegaEmulated(t *testing.T, m *hopper.Megakernel, src string, env map[string][]float32) ([]float32, []uint64) {
	t.Helper()
	var bufs []cudahostemu.Buffer
	var args []string
	for _, in := range m.Inputs {
		bufs = append(bufs, cudahostemu.Buffer{
			Name:  in.Name,
			CType: crossLayerCType(in.DType),
			Data:  encodeFullLayerTestTensor(in.DType, env[in.Name]),
		})
		args = append(args, in.Name)
	}
	for _, tmp := range m.Temps {
		nbytes := crossLayerTensorSize(tmp.Shape) * tmp.DType.Bits() / 8
		bufs = append(bufs, cudahostemu.Buffer{
			Name:  tmp.Name,
			CType: crossLayerCType(tmp.DType),
			Data:  cudahostemu.Poison(nbytes),
		})
		args = append(args, tmp.Name)
	}
	outBytes := crossLayerTensorSize(m.Output.Shape) * m.Output.DType.Bits() / 8
	bufs = append(bufs, cudahostemu.Buffer{
		Name:  m.Output.Name,
		CType: crossLayerCType(m.Output.DType),
		Data:  cudahostemu.Poison(outBytes),
	})
	bufs = append(bufs, cudahostemu.Buffer{
		Name:  "ready",
		CType: "unsigned long long",
		Data:  make([]byte, m.ReadyBytes()),
	})
	args = append(args, m.Output.Name, "ready", "1u", "nullptr")

	emuSrc := strings.ReplaceAll(src, "__nanosleep(32);", "__nanosleep(32); std::this_thread::yield();")
	res, err := cudahostemu.Run(t.Context(), t.TempDir(), cudahostemu.Program{
		Source:  emuSrc,
		Launch:  fmt.Sprintf("%s_run(%s)", m.Name, strings.Join(args, ", ")),
		Buffers: bufs,
	})
	if errors.Is(err, cudahostemu.ErrNoCompiler) {
		t.Skip("no host C++ compiler; emulated megakernel not run")
	}
	if err != nil {
		t.Fatalf("emulate megakernel %s: %v", m.Name, err)
	}
	out := decodeCrossLayerTensor(m.Output.DType, res[m.Output.Name])
	rawReady := res["ready"]
	ready := make([]uint64, len(rawReady)/8)
	for i := range ready {
		ready[i] = binary.LittleEndian.Uint64(rawReady[8*i:])
	}
	return out, ready
}

func TestFullLayerLlamaMegakernel(t *testing.T) {
	modelCfg := &ModelConfig{
		ModelType:         "llama",
		HiddenSize:        256,
		IntermediateSize:  256,
		NumHiddenLayers:   2,
		NumAttentionHeads: 4,
		NumKeyValueHeads:  1,
		VocabSize:         256,
		RmsNormEps:        1e-5,
		RopeTheta:         10000,
		MaxSeqLen:         64,
	}

	// 1-layer full megakernel from ModelConfig.
	plan1, mega1, err := BuildLlamaFullLayerMegakernel(FullLayerMegakernelConfig{
		Model:      modelCfg,
		StartLayer: 0,
		NumLayers:  1,
		BatchM:     1,
		Arch:       hopper.TargetSM121A,
	})
	if err != nil {
		t.Fatalf("BuildLlamaFullLayerMegakernel(1L): %v", err)
	}
	if mega1.Name != "llama_full_l0_mega" {
		t.Fatalf("mega1.Name = %q, want llama_full_l0_mega", mega1.Name)
	}
	if len(plan1.Kernels) != 9 || len(mega1.Stages) != 9 {
		t.Fatalf("1L kernels=%d stages=%d, want 9 and 9", len(plan1.Kernels), len(mega1.Stages))
	}

	// 2-layer full megakernel with tiny decoder dimensions.
	cfg2 := FullLayerMegakernelConfig{
		Name:       "llama_tiny_2l_mega",
		StartLayer: 0,
		NumLayers:  2,
		BatchM:     1,
		HiddenDim:  16,
		IntermDim:  32,
		NumQHeads:  2,
		NumKVHeads: 1,
		HeadDim:    8,
		PageSize:   16,
		MaxPages:   4,
		RoPETheta:  10000.0,
		RMSEps:     1e-5,
		Arch:       hopper.TargetSM121A,
	}
	plan2, mega2, err := BuildLlamaFullLayerMegakernel(cfg2)
	if err != nil {
		t.Fatalf("BuildLlamaFullLayerMegakernel(2L): %v", err)
	}
	if len(plan2.Kernels) != 18 || len(mega2.Stages) != 18 {
		t.Fatalf("2L kernels=%d stages=%d, want 18 and 18", len(plan2.Kernels), len(mega2.Stages))
	}

	cudaSrc, err := mega2.CUDA()
	if err != nil {
		t.Fatalf("mega2.CUDA(): %v", err)
	}
	if !strings.Contains(cudaSrc, `asm volatile("bar.sync 1, 128;")`) {
		t.Fatal("missing bar.sync 1, 128; in mega2 CUDA source")
	}
	if !strings.Contains(cudaSrc, "eg_ready_resolve_epoch") {
		t.Fatal("missing eg_ready_resolve_epoch in mega2 CUDA source")
	}

	lineSrc, spans, err := mega2.LineInfoSource()
	if err != nil {
		t.Fatalf("mega2.LineInfoSource(): %v", err)
	}
	if len(spans) != 18 {
		t.Fatalf("len(spans) = %d, want 18", len(spans))
	}
	flightStages, flightRanges, err := StageAttributionMap(*mega2, nil)
	if err != nil {
		t.Fatalf("StageAttributionMap: %v", err)
	}
	if len(flightStages) != 18 || len(flightRanges) != 18 {
		t.Fatalf("flight StageAttributionMap returned (%d, %d), want (18, 18)",
			len(flightStages), len(flightRanges))
	}

	// Compare 2-layer monolithic megakernel output against sequential execution of layer 0 and layer 1 plans.
	specs, err := cfg2.LayerSpecs()
	if err != nil {
		t.Fatal(err)
	}
	l0Plan, err := hopper.BuildFullDecoderLayerPlan(specs[0])
	if err != nil {
		t.Fatal(err)
	}
	l1Plan, err := hopper.BuildFullDecoderLayerPlan(specs[1])
	if err != nil {
		t.Fatal(err)
	}

	r := rand.New(rand.NewPCG(301, 302))
	env := makeFullLayerTestEnv(r, plan2, cfg2.BatchM, cfg2.MaxPages)
	megaOut, readyWords := runFullLayerMegaEmulated(t, mega2, lineSrc, env)

	curX := env["x"]
	for i, lp := range []*hopper.Plan{l0Plan, l1Plan} {
		pfx := fmt.Sprintf("l%d_", i)
		lEnv := make(map[string][]float32, len(lp.Inputs))
		for _, in := range lp.Inputs {
			if in.Name == "x" {
				lEnv["x"] = curX
				continue
			}
			if v, ok := env[pfx+in.Name]; ok {
				lEnv[in.Name] = v
			} else {
				lEnv[in.Name] = env[in.Name]
			}
		}
		curX = runFullLayerPlanEmulated(t, lp, lEnv)
	}

	if len(megaOut) != len(curX) {
		t.Fatalf("len(megaOut) = %d, want %d", len(megaOut), len(curX))
	}
	for i := range curX {
		diff := math.Abs(float64(megaOut[i] - curX[i]))
		if math.IsNaN(float64(megaOut[i])) || diff > 1e-4 {
			t.Fatalf("out[%d] = %g, sequential want %g (diff %g)", i, megaOut[i], curX[i], diff)
		}
	}
	for i := 0; i < len(mega2.Stages)-1; i++ {
		wantWord := hopper.PackReadyWord(1, uint32(mega2.Stages[i].Blocks))
		if readyWords[i] != wantWord {
			t.Fatalf("ready[%d] = %#x, want %#x", i, readyWords[i], wantWord)
		}
	}
	if !slices.Contains(mega2.Stages[9].WaitSlots, 8) {
		t.Fatalf("stage 9 WaitSlots = %v, want stage 8", mega2.Stages[9].WaitSlots)
	}
}

func TestFullLayerLlamaMegakernelErrors(t *testing.T) {
	modelCfg := llama3_8B
	if _, _, err := BuildLlamaFullLayerMegakernel(FullLayerMegakernelConfig{StartLayer: -1}); err == nil {
		t.Error("negative StartLayer: expected error")
	}
	if _, _, err := BuildLlamaFullLayerMegakernel(FullLayerMegakernelConfig{NumLayers: -1}); err == nil {
		t.Error("negative NumLayers: expected error")
	}
	if _, _, err := BuildLlamaFullLayerMegakernel(FullLayerMegakernelConfig{
		Model:      &modelCfg,
		StartLayer: modelCfg.NumHiddenLayers,
		NumLayers:  1,
	}); err == nil {
		t.Error("out-of-range StartLayer: expected error")
	}
	if _, _, err := BuildLlamaFullLayerMegakernel(FullLayerMegakernelConfig{}); err == nil {
		t.Error("empty config without model or dims: expected error")
	}
}
