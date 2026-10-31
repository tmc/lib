package main

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudahostemu"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/egraph/hopper"
)

func crossLayerTensorSize(shape []int) int {
	n := 1
	for _, d := range shape {
		n *= d
	}
	return n
}

func encodeCrossLayerTensor(dt hopper.DType, vals []float32) []byte {
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
	default:
		panic(fmt.Sprintf("unsupported dtype %v", dt))
	}
}

func decodeCrossLayerTensor(dt hopper.DType, raw []byte) []float32 {
	switch dt {
	case hopper.BF16:
		out := make([]float32, len(raw)/2)
		for i := range out {
			out[i] = hopper.FromBF16Bits(binary.LittleEndian.Uint16(raw[2*i:]))
		}
		return out
	case hopper.F32:
		out := make([]float32, len(raw)/4)
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
		}
		return out
	default:
		panic(fmt.Sprintf("unsupported dtype %v", dt))
	}
}

func crossLayerCType(dt hopper.DType) string {
	switch dt {
	case hopper.BF16:
		return "uint16_t"
	case hopper.F32:
		return "float"
	case hopper.I32:
		return "int32_t"
	default:
		return "uint8_t"
	}
}

func runPlanEmulated(t *testing.T, p *hopper.Plan, env map[string][]float32) []float32 {
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
			Data:  encodeCrossLayerTensor(in.DType, vals),
		})
		args = append(args, in.Name)
	}
	for _, tmp := range p.Temps {
		nbytes := crossLayerTensorSize(tmp.Shape) * tmp.DType.Bits() / 8
		raw := cudahostemu.Poison(nbytes)
		if tmp.DType == hopper.I32 && strings.HasSuffix(tmp.Name, "_count") {
			clear(raw)
		}
		bufs = append(bufs, cudahostemu.Buffer{
			Name:  tmp.Name,
			CType: crossLayerCType(tmp.DType),
			Data:  raw,
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

	launch := fmt.Sprintf("%s_run(%s)", p.Name, strings.Join(args, ", "))
	res, err := cudahostemu.Run(t.Context(), t.TempDir(), cudahostemu.Program{
		Source:  src,
		Launch:  launch,
		Buffers: bufs,
	})
	if errors.Is(err, cudahostemu.ErrNoCompiler) {
		t.Skip("no host C++ compiler; emulated kernel not run")
	}
	if err != nil {
		t.Fatalf("emulate plan %s: %v", p.Name, err)
	}
	return decodeCrossLayerTensor(p.Output.DType, res[p.Output.Name])
}

func runMegakernelEmulatedEpochs(t *testing.T, m hopper.Megakernel, src string, envs []map[string][]float32) ([][]float32, []uint64) {
	t.Helper()
	var bufs []cudahostemu.Buffer
	var launches []string

	for _, tmp := range m.Temps {
		nbytes := crossLayerTensorSize(tmp.Shape) * tmp.DType.Bits() / 8
		bufs = append(bufs, cudahostemu.Buffer{
			Name:  tmp.Name,
			CType: crossLayerCType(tmp.DType),
			Data:  cudahostemu.Poison(nbytes),
		})
	}
	bufs = append(bufs, cudahostemu.Buffer{
		Name:  "ready",
		CType: "unsigned long long",
		Data:  make([]byte, m.ReadyBytes()),
	})

	for epIdx, env := range envs {
		epoch := epIdx + 1
		var args []string
		for _, in := range m.Inputs {
			bname := fmt.Sprintf("%s_ep%d", in.Name, epoch)
			bufs = append(bufs, cudahostemu.Buffer{
				Name:  bname,
				CType: crossLayerCType(in.DType),
				Data:  encodeCrossLayerTensor(in.DType, env[in.Name]),
			})
			args = append(args, bname)
		}
		for _, tmp := range m.Temps {
			args = append(args, tmp.Name)
		}
		outName := fmt.Sprintf("%s_ep%d", m.Output.Name, epoch)
		outBytes := crossLayerTensorSize(m.Output.Shape) * m.Output.DType.Bits() / 8
		bufs = append(bufs, cudahostemu.Buffer{
			Name:  outName,
			CType: crossLayerCType(m.Output.DType),
			Data:  cudahostemu.Poison(outBytes),
		})
		args = append(args, outName, "ready", fmt.Sprintf("%du", epoch), "nullptr")
		launches = append(launches, fmt.Sprintf("%s_run(%s)", m.Name, strings.Join(args, ", ")))
	}

	res, err := cudahostemu.Run(t.Context(), t.TempDir(), cudahostemu.Program{
		Source:  src,
		Launch:  strings.Join(launches, ";\n    "),
		Buffers: bufs,
	})
	if errors.Is(err, cudahostemu.ErrNoCompiler) {
		t.Skip("no host C++ compiler; emulated megakernel not run")
	}
	if err != nil {
		t.Fatalf("emulate megakernel %s: %v", m.Name, err)
	}

	outs := make([][]float32, len(envs))
	for epIdx := range envs {
		outName := fmt.Sprintf("%s_ep%d", m.Output.Name, epIdx+1)
		outs[epIdx] = decodeCrossLayerTensor(m.Output.DType, res[outName])
	}
	rawReady := res["ready"]
	readyWords := make([]uint64, len(rawReady)/8)
	for i := range readyWords {
		readyWords[i] = binary.LittleEndian.Uint64(rawReady[8*i:])
	}
	return outs, readyWords
}

func TestCrossLayerDecodeMegakernel(t *testing.T) {
	cfg := ModelConfig{
		ModelType:         "llama",
		HiddenSize:        256,
		IntermediateSize:  256,
		NumHiddenLayers:   2,
		NumAttentionHeads: 4,
		NumKeyValueHeads:  1,
		VocabSize:         256,
		RmsNormEps:        1e-5,
		RopeTheta:         10000,
		MaxSeqLen:         16,
	}

	fusedPlan, mega, err := buildCrossLayerDecodeMegakernel(&cfg, true, 0, 1, 1, "sm_121a")
	if err != nil {
		t.Fatalf("buildCrossLayerDecodeMegakernel: %v", err)
	}
	if mega.Name != "llama_cross_l0_l1" {
		t.Fatalf("mega.Name = %q, want llama_cross_l0_l1", mega.Name)
	}
	// Layer 0 post unfuses into 5 stages (o_proj, ffn_norm, gate_up, down_proj, next_norm)
	// plus Layer 1 attnIn (qkv_proj) = 6 persistent megakernel stages.
	if len(mega.Stages) != 6 {
		t.Fatalf("len(mega.Stages) = %d, want 6", len(mega.Stages))
	}
	for i := 0; i < 5; i++ {
		if !strings.HasPrefix(mega.Stages[i].Name, "l0_") {
			t.Errorf("stage %d name %q does not have l0_ prefix", i, mega.Stages[i].Name)
		}
	}
	if !strings.HasPrefix(mega.Stages[5].Name, "l1_") {
		t.Errorf("stage 5 name %q does not have l1_ prefix", mega.Stages[5].Name)
	}
	if !slices.Contains(mega.Stages[5].WaitSlots, 4) {
		t.Fatalf("stage 5 (%s) WaitSlots = %v, want to wait on stage 4", mega.Stages[5].Name, mega.Stages[5].WaitSlots)
	}

	// Verify #line-attributed CUDA C++ source and both hopper and flight-loop StageAttributionMap.
	lineSrc, spans, err := mega.LineInfoSource()
	if err != nil {
		t.Fatalf("LineInfoSource: %v", err)
	}
	hopperAttr := mega.StageAttributionMap()
	if len(hopperAttr) != len(mega.Stages) {
		t.Fatalf("len(mega.StageAttributionMap()) = %d, want %d", len(hopperAttr), len(mega.Stages))
	}
	flightStages, flightRanges, err := StageAttributionMap(mega, nil)
	if err != nil {
		t.Fatalf("StageAttributionMap: %v", err)
	}
	if len(flightStages) != len(mega.Stages) || len(flightRanges) != len(mega.Stages) {
		t.Fatalf("flight StageAttributionMap returned %d stages and %d ranges, want %d",
			len(flightStages), len(flightRanges), len(mega.Stages))
	}
	lines := strings.Split(lineSrc, "\n")
	for i, st := range mega.Stages {
		sp, ok := hopperAttr[st.Name]
		if !ok {
			t.Fatalf("StageAttributionMap missing stage %q", st.Name)
		}
		if sp != spans[i] {
			t.Fatalf("span[%d] = %+v, want %+v", i, sp, spans[i])
		}
		wantLine := fmt.Sprintf("#line 1 %q", st.Name+".cu")
		if got := lines[sp.StartLine-2]; got != wantLine {
			t.Errorf("stage %d line directive = %q, want %q", i, got, wantLine)
		}
	}

	// Build sequential reference plans for layer 0 post and layer 1 attnIn.
	_, post0, err := buildLayerPlans(&cfg, true, false, "l0", 1, hopper.F32, "sm_121a", true)
	if err != nil {
		t.Fatalf("buildLayerPlans l0: %v", err)
	}
	attnIn1, _, err := buildLayerPlans(&cfg, true, false, "l1", 1, hopper.F32, "sm_121a", true)
	if err != nil {
		t.Fatalf("buildLayerPlans l1: %v", err)
	}
	for _, pl := range []*hopper.Plan{post0, attnIn1} {
		for i := range pl.Kernels {
			if pl.Kernels[i].GEMM != nil {
				pl.Kernels[i].GEMM.RowsPerBlock = 32
			}
		}
	}

	r := rand.New(rand.NewPCG(101, 102))
	envs := make([]map[string][]float32, 2)
	for ep := range envs {
		env := make(map[string][]float32, len(fusedPlan.Inputs))
		for _, in := range fusedPlan.Inputs {
			vals := make([]float32, crossLayerTensorSize(in.Shape))
			for i := range vals {
				v := float32((r.Float64()*2 - 1) * 0.25)
				vals[i] = in.DType.Round(v)
			}
			env[in.Name] = vals
		}
		envs[ep] = env
	}

	megaOuts, readyWords := runMegakernelEmulatedEpochs(t, mega, lineSrc, envs)
	for ep, env := range envs {
		postOut := runPlanEmulated(t, post0, env)
		nextEnv := make(map[string][]float32, len(env)+1)
		for k, v := range env {
			nextEnv[k] = v
		}
		nextEnv["xn"] = postOut
		seqOut := runPlanEmulated(t, attnIn1, nextEnv)

		if len(megaOuts[ep]) != len(seqOut) {
			t.Fatalf("epoch %d len(megaOut) = %d, want %d", ep+1, len(megaOuts[ep]), len(seqOut))
		}
		for i := range seqOut {
			diff := math.Abs(float64(megaOuts[ep][i] - seqOut[i]))
			if math.IsNaN(float64(megaOuts[ep][i])) || diff > 1e-4 {
				t.Fatalf("epoch %d out[%d] = %g, sequential want %g (diff %g)",
					ep+1, i, megaOuts[ep][i], seqOut[i], diff)
			}
		}
	}
	for i := 0; i < len(mega.Stages)-1; i++ {
		wantWord := hopper.PackReadyWord(uint32(len(envs)), uint32(mega.Stages[i].Blocks))
		if readyWords[i] != wantWord {
			t.Fatalf("ready[%d] = %#x, want %#x", i, readyWords[i], wantWord)
		}
	}
}

func TestCrossLayerDecodeMegakernelErrors(t *testing.T) {
	cfg := llama3_8B
	if _, _, err := buildCrossLayerDecodeMegakernel(nil, false, 0, 1, 1, "sm_121a"); err == nil {
		t.Error("nil config: expected error")
	}
	if _, _, err := buildCrossLayerDecodeMegakernel(&cfg, false, -1, 1, 1, "sm_121a"); err == nil {
		t.Error("negative layerL: expected error")
	}
	if _, _, err := buildCrossLayerDecodeMegakernel(&cfg, false, 0, 0, 1, "sm_121a"); err == nil {
		t.Error("identical layers: expected error")
	}
	if _, _, err := buildCrossLayerDecodeMegakernel(&cfg, false, 0, cfg.NumHiddenLayers, 1, "sm_121a"); err == nil {
		t.Error("out-of-range layerNext: expected error")
	}
	if _, _, err := buildCrossLayerDecodeMegakernel(&cfg, false, 0, 1, 0, "sm_121a"); err == nil {
		t.Error("zero m: expected error")
	}
}

func TestEgraphPlansAndDecoderCrossLayer(t *testing.T) {
	cfg := llama3_8B
	cfg.NumHiddenLayers = 4
	prof := mustProfile(t, "GB10")

	basePlans, err := buildEgraphPlans(&cfg, true, egraphOptions{
		Profile:    prof,
		Attn:       true,
		AttnSplits: 2,
		Mega:       true,
	})
	if err != nil {
		t.Fatalf("buildEgraphPlans base: %v", err)
	}
	if basePlans.cross != nil {
		t.Fatal("expected basePlans.cross == nil when CrossLayer is false")
	}

	crossPlans, err := buildEgraphPlans(&cfg, true, egraphOptions{
		Profile:    prof,
		Attn:       true,
		AttnSplits: 2,
		Mega:       true,
		CrossLayer: true,
	})
	if err != nil {
		t.Fatalf("buildEgraphPlans cross: %v", err)
	}
	if crossPlans.cross == nil {
		t.Fatal("expected crossPlans.cross != nil when CrossLayer is true")
	}
	if crossPlans.cross.Name != "llama_cross" {
		t.Fatalf("crossPlans.cross.Name = %q, want llama_cross", crossPlans.cross.Name)
	}
	genOps, _ := crossPlans.ops()
	if !slices.Contains(genOps, "cross_layer_post_qkv") {
		t.Fatalf("crossPlans.ops() = %v, missing cross_layer_post_qkv", genOps)
	}

	// Emit with CrossLayer: true and Mega: true and check llama_egraph.cu and names.txt.
	dir := t.TempDir()
	if err := emitEgraph(dir, "GB10", &cfg, egraphOptions{
		Mega:       true,
		Attn:       true,
		AttnSplits: 2,
		CrossLayer: true,
	}); err != nil {
		t.Fatalf("emitEgraph CrossLayer+Mega: %v", err)
	}
	cuBytes, err := os.ReadFile(filepath.Join(dir, "sm_121a", "llama_egraph.cu"))
	if err != nil {
		t.Fatal(err)
	}
	namesBytes, err := os.ReadFile(filepath.Join(dir, "sm_121a", "names.txt"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(cuBytes), "llama_cross_mega") {
		t.Fatal("llama_egraph.cu missing llama_cross_mega megakernel")
	}
	if !slices.Contains(strings.Fields(string(namesBytes)), "llama_cross_mega") {
		t.Fatalf("names.txt = %q, missing llama_cross_mega", string(namesBytes))
	}

	// Bind both decoders (base vs cross-layer) and verify launch counts and buffer bindings.
	bindTestDecoder := func(t *testing.T, plans *egraphPlans) (*egraphDecoder, egraphBuffers) {
		t.Helper()
		eg := &egraphDecoder{
			plans:   plans,
			megas:   make(map[*hopper.Plan]hopper.Megakernel),
			fns:     make(map[string]driver.Function),
			ready:   make(map[*hopper.Plan]cudamem.Slice[uint64]),
			tempPtr: make(map[string]driver.DevicePtr),
			posKind: make(map[driver.Function]string),
			posArgs: make(map[driver.Function][]egraphPosArg),
		}
		for _, p := range eg.decodePlans() {
			m, err := p.Megakernel("")
			if err != nil {
				t.Fatalf("Megakernel(%s): %v", p.Name, err)
			}
			eg.megas[p] = m
		}
		_, names, err := eg.source()
		if err != nil {
			t.Fatalf("eg.source(): %v", err)
		}
		for i, name := range names {
			eg.fns[name] = driver.Function(i + 1)
		}
		nextPtr := driver.DevicePtr(0x1000)
		allocPtr := func() driver.DevicePtr {
			p := nextPtr
			nextPtr += 0x1000
			return p
		}
		for _, p := range eg.planList() {
			for _, tmp := range p.Temps {
				if _, ok := eg.tempPtr[tmp.Name]; !ok {
					eg.tempPtr[tmp.Name] = allocPtr()
				}
			}
		}
		b := egraphBuffers{
			x: allocPtr(), x2: allocPtr(), xn: allocPtr(), qkv: allocPtr(), attnOut: allocPtr(),
			norm: allocPtr(), lmHead: allocPtr(), logits: allocPtr(),
			qbf: allocPtr(), cs: allocPtr(), ropePage: allocPtr(),
			seqLen1: allocPtr(), pages: allocPtr(), ws: allocPtr(),
		}
		for range cfg.NumHiddenLayers {
			b.layers = append(b.layers, egraphLayerBuffers{
				attnNorm: allocPtr(),
				wqkv:     allocPtr(),
				bqkv:     allocPtr(),
				wo:       allocPtr(),
				ffnNorm:  allocPtr(),
				wgu:      allocPtr(),
				wdown:    allocPtr(),
				k:        allocPtr(),
				v:        allocPtr(),
			})
		}
		if err := eg.bindDecode(&cfg, b); err != nil {
			t.Fatalf("bindDecode: %v", err)
		}
		return eg, b
	}

	baseEG, _ := bindTestDecoder(t, basePlans)
	crossEG, bufs := bindTestDecoder(t, crossPlans)

	// Without CrossLayer: each of the L=4 layers launches 1 attnIn megakernel + 1 post megakernel = 2L = 8 projection/MLP launches.
	// With CrossLayer: layer 0 attnIn (1) + 3 cross boundaries (3) + layer 3 post (1) = L+1 = 5 projection/MLP launches (saving L-1 = 3 launches).
	wantSaving := cfg.NumHiddenLayers - 1
	if gotSaving := baseEG.Kernels() - crossEG.Kernels(); gotSaving != wantSaving {
		t.Fatalf("baseEG.Kernels()=%d, crossEG.Kernels()=%d; saving=%d, want %d",
			baseEG.Kernels(), crossEG.Kernels(), gotSaving, wantSaving)
	}
	if len(crossEG.cross) != cfg.NumHiddenLayers-1 {
		t.Fatalf("len(crossEG.cross) = %d, want %d", len(crossEG.cross), cfg.NumHiddenLayers-1)
	}

	// Verify layerLaunches(i) across all layers:
	// i=0: attnIn[0] (1) + rope (1) + sdpa (1) + combine (1) + cross[0] (1) = 5 launches
	// i=1,2: rope (1) + sdpa (1) + combine (1) + cross[i] (1) = 4 launches
	// i=3: rope (1) + sdpa (1) + combine (1) + post[3] (1) = 4 launches
	totalLayerLaunches := 0
	for i := range cfg.NumHiddenLayers {
		ls := crossEG.layerLaunches(i)
		totalLayerLaunches += len(ls)
		wantLen := 4
		if i == 0 {
			wantLen = 5
		}
		if len(ls) != wantLen {
			t.Fatalf("layer %d: len(layerLaunches) = %d, want %d", i, len(ls), wantLen)
		}
		lastFn := ls[len(ls)-1].fn
		if i < cfg.NumHiddenLayers-1 {
			if lastFn != crossEG.fns["llama_cross_mega"] {
				t.Fatalf("layer %d last launch fn = %v, want llama_cross_mega (%v)", i, lastFn, crossEG.fns["llama_cross_mega"])
			}
		} else {
			if lastFn != crossEG.fns["llama_post_mega"] {
				t.Fatalf("final layer %d last launch fn = %v, want llama_post_mega (%v)", i, lastFn, crossEG.fns["llama_post_mega"])
			}
		}
	}
	if gotTotal := len(crossEG.norm) + totalLayerLaunches + len(crossEG.head); gotTotal != crossEG.Kernels() {
		t.Fatalf("sum of launches = %d, want crossEG.Kernels() = %d", gotTotal, crossEG.Kernels())
	}

	// Verify cross[0] binds layer 0 post weights/residual and layer 1 attnIn weights/bias/norm.
	cross0Args := crossEG.cross[0][0].args
	for _, wantPtr := range []driver.DevicePtr{
		bufs.x,                  // layer 0 input residual cur
		bufs.x2,                 // layer 0 output residual next (postRes)
		bufs.layers[0].wo,       // l0 w_o
		bufs.layers[0].ffnNorm,  // l0 ffn_norm
		bufs.layers[0].wgu,      // l0 w_gu
		bufs.layers[0].wdown,    // l0 w_down
		bufs.layers[1].attnNorm, // l0 next_norm (layer 1 input norm)
		bufs.layers[1].wqkv,     // l1 w_qkv
		bufs.layers[1].bqkv,     // l1 b_qkv
		bufs.xn,                 // bridge tensor xn
		bufs.qkv,                // output qkv
	} {
		if !slices.ContainsFunc(cross0Args, func(a any) bool {
			p, ok := a.(driver.DevicePtr)
			return ok && p == wantPtr
		}) {
			t.Fatalf("cross[0] launch args %v missing bound pointer %#x", cross0Args, wantPtr)
		}
	}
}
