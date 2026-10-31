package hopper

import (
	"fmt"
	"slices"
	"strings"
)

// CrossLayerGlue configures how layer L's post-attention plan (post) connects
// across the inter-layer boundary to layer (L+1)'s pre-attention input plan
// (nextAttnIn).
type CrossLayerGlue struct {
	// Name is the combined plan and megakernel base name. When empty, it
	// defaults to post.Name + "_" + nextAttnIn.Name.
	Name string

	// LayerLPrefix prefixes layer L's tensor and stage names when disambiguating
	// collisions with layer L+1. When empty, it defaults to "l0_".
	LayerLPrefix string

	// LayerNextPrefix prefixes layer (L+1)'s tensor and stage names when
	// disambiguating collisions with layer L. When empty, it defaults to "l1_".
	LayerNextPrefix string

	// BridgeInput is the input tensor name in nextAttnIn that receives
	// post.Output across the inter-layer boundary. When empty, it defaults
	// to "xn".
	BridgeInput string

	// RetainBridgeOutput preserves the intermediate bridge tensor (post.Output)
	// in the fused plan's Temps buffer list so downstream residual or attention
	// consumers can inspect or reuse the normalized inter-layer activation.
	RetainBridgeOutput bool
}

func (g CrossLayerGlue) norm(postName, nextName string) (CrossLayerGlue, error) {
	if g.BridgeInput == "" {
		g.BridgeInput = "xn"
	}
	if !cIdent(g.BridgeInput) {
		return g, fmt.Errorf("cross-layer fuse: bad bridge input name %q", g.BridgeInput)
	}
	if g.LayerLPrefix == "" {
		g.LayerLPrefix = "l0_"
	} else if !strings.HasSuffix(g.LayerLPrefix, "_") {
		g.LayerLPrefix += "_"
	}
	if g.LayerNextPrefix == "" {
		g.LayerNextPrefix = "l1_"
	} else if !strings.HasSuffix(g.LayerNextPrefix, "_") {
		g.LayerNextPrefix += "_"
	}
	if g.LayerLPrefix == g.LayerNextPrefix {
		return g, fmt.Errorf("cross-layer fuse: identical layer prefixes %q", g.LayerLPrefix)
	}
	if g.Name == "" {
		switch {
		case postName != "" && nextName != "":
			g.Name = postName + "_" + nextName
		case postName != "":
			g.Name = postName + "_cross"
		case nextName != "":
			g.Name = nextName + "_cross"
		default:
			g.Name = "cross_layer"
		}
	}
	if !cIdent(g.Name) {
		return g, fmt.Errorf("cross-layer fuse: bad name %q", g.Name)
	}
	return g, nil
}

// StageAttributionMap returns the 1-based #line span ([StageLineSpan]) for each
// stage of m in the translation unit emitted by [Megakernel.LineInfoSource],
// keyed by stage name.
func (m Megakernel) StageAttributionMap() map[string]StageLineSpan {
	_, spans, err := m.LineInfoSource()
	if err != nil {
		return nil
	}
	out := make(map[string]StageLineSpan, len(spans))
	for _, sp := range spans {
		out[sp.StageName] = sp
	}
	return out
}

// FuseCrossLayerPlans fuses layer L's post-attention plan (post) with layer
// (L+1)'s pre-attention projection plan (nextAttnIn) across the inter-layer
// boundary.
//
// It validates that post.Output matches the shape and dtype of nextAttnIn's
// bridge input (defaulting to "xn"), unfuses any single-CTA GEMM tails into
// explicit megakernel stages, disables incompatible cluster and
// warp-specialized flags, renames colliding input/temp/kernel names using
// glue's layer prefixes when not already prefixed, wires post.Output.Name into
// nextAttnIn's bridge input (placing post.Output into Temps of the combined
// Plan), and sets the combined plan's Output to nextAttnIn.Output.
func FuseCrossLayerPlans(post, nextAttnIn *Plan, glue CrossLayerGlue) (*Plan, error) {
	if post == nil || len(post.Kernels) == 0 {
		return nil, fmt.Errorf("cross-layer fuse: empty post plan")
	}
	if nextAttnIn == nil || len(nextAttnIn.Kernels) == 0 {
		return nil, fmt.Errorf("cross-layer fuse: empty nextAttnIn plan")
	}
	if post.Output.Name == "" || len(post.Output.Shape) == 0 {
		return nil, fmt.Errorf("cross-layer fuse: post plan %q has no output tensor", post.Name)
	}
	if nextAttnIn.Output.Name == "" || len(nextAttnIn.Output.Shape) == 0 {
		return nil, fmt.Errorf("cross-layer fuse: nextAttnIn plan %q has no output tensor", nextAttnIn.Name)
	}

	normGlue, err := glue.norm(post.Name, nextAttnIn.Name)
	if err != nil {
		return nil, err
	}

	p0 := preparePlanForMegakernel(post)
	p1 := preparePlanForMegakernel(nextAttnIn)

	bridgeIdx := -1
	var bridgeIn Tensor
	for i, in := range p1.Inputs {
		if in.Name == normGlue.BridgeInput || strings.TrimPrefix(in.Name, normGlue.LayerNextPrefix) == normGlue.BridgeInput {
			bridgeIdx = i
			bridgeIn = in
			break
		}
	}
	if bridgeIdx < 0 {
		return nil, fmt.Errorf("cross-layer fuse: bridge input %q not found in nextAttnIn (%s)", normGlue.BridgeInput, p1.Name)
	}
	if p0.Output.DType != bridgeIn.DType {
		return nil, fmt.Errorf("cross-layer fuse: post output %s dtype %v != nextAttnIn bridge input %s dtype %v",
			p0.Output.Name, p0.Output.DType, bridgeIn.Name, bridgeIn.DType)
	}
	if !slices.Equal(p0.Output.Shape, bridgeIn.Shape) {
		return nil, fmt.Errorf("cross-layer fuse: post output %s shape %v != nextAttnIn bridge input %s shape %v",
			p0.Output.Name, p0.Output.Shape, bridgeIn.Name, bridgeIn.Shape)
	}

	p0Names := collectPlanTensorNames(p0, "")
	p1Names := collectPlanTensorNames(p1, bridgeIn.Name)

	rename0 := make(map[string]string)
	rename1 := make(map[string]string)
	usedTensors := make(map[string]bool)

	// Disambiguate colliding tensor names between layer L and layer L+1.
	for name := range p0Names {
		target := name
		if p1Names[name] {
			target = ensureLayerPrefix(name, normGlue.LayerLPrefix)
		}
		rename0[name] = target
		usedTensors[target] = true
	}
	for name := range p1Names {
		target := name
		if p0Names[name] || usedTensors[target] {
			target = ensureLayerPrefix(name, normGlue.LayerNextPrefix)
		}
		for usedTensors[target] {
			target = normGlue.LayerNextPrefix + target
		}
		rename1[name] = target
		usedTensors[target] = true
	}

	bridgeTensor := cloneTensor(p0.Output)
	bridgeTensor.Name = rename0[p0.Output.Name]
	rename1[bridgeIn.Name] = bridgeTensor.Name

	applyPlanTensorRename(p0, rename0)
	applyPlanTensorRename(p1, rename1)

	// Disambiguate stage kernel names across both plans.
	p0KernelNames := make(map[string]bool, len(p0.Kernels))
	for _, k := range p0.Kernels {
		p0KernelNames[planKernelName(k)] = true
	}
	p1KernelNames := make(map[string]bool, len(p1.Kernels))
	for _, k := range p1.Kernels {
		p1KernelNames[planKernelName(k)] = true
	}

	usedKernels := map[string]bool{normGlue.Name: true}
	for i := range p0.Kernels {
		kname := planKernelName(p0.Kernels[i])
		if p1KernelNames[kname] || usedKernels[kname] || glue.LayerLPrefix != "" {
			kname = ensureLayerPrefix(kname, normGlue.LayerLPrefix)
		}
		for usedKernels[kname] {
			kname = fmt.Sprintf("%s%s_%d", normGlue.LayerLPrefix, kname, i)
		}
		setPlanKernelName(&p0.Kernels[i], kname)
		usedKernels[kname] = true
	}
	for i := range p1.Kernels {
		kname := planKernelName(p1.Kernels[i])
		if p0KernelNames[kname] || usedKernels[kname] || glue.LayerNextPrefix != "" {
			kname = ensureLayerPrefix(kname, normGlue.LayerNextPrefix)
		}
		for usedKernels[kname] {
			kname = fmt.Sprintf("%s%s_%d", normGlue.LayerNextPrefix, kname, i)
		}
		setPlanKernelName(&p1.Kernels[i], kname)
		usedKernels[kname] = true
	}

	var inputs []Tensor
	seenInputs := make(map[string]bool)
	for _, in := range p0.Inputs {
		if !seenInputs[in.Name] {
			seenInputs[in.Name] = true
			inputs = append(inputs, in)
		}
	}
	for i, in := range p1.Inputs {
		if i == bridgeIdx {
			continue
		}
		if !seenInputs[in.Name] {
			seenInputs[in.Name] = true
			inputs = append(inputs, in)
		}
	}

	var temps []Tensor
	seenTemps := make(map[string]bool)
	addTemp := func(t Tensor) {
		if !seenTemps[t.Name] {
			seenTemps[t.Name] = true
			temps = append(temps, t)
		}
	}
	for _, t := range p0.Temps {
		addTemp(t)
	}
	// Wire post.Output into Temps so post's final stage publishes it on ready
	// and nextAttnIn's stage waits on and reads it (also retained when
	// glue.RetainBridgeOutput is set).
	addTemp(bridgeTensor)
	for _, t := range p1.Temps {
		addTemp(t)
	}

	kernels := make([]PlanKernel, 0, len(p0.Kernels)+len(p1.Kernels))
	kernels = append(kernels, p0.Kernels...)
	kernels = append(kernels, p1.Kernels...)

	return &Plan{
		Name:    normGlue.Name,
		Inputs:  inputs,
		Temps:   temps,
		Output:  p1.Output,
		Kernels: kernels,
	}, nil
}

// BuildCrossLayerMegakernel fuses layer L's post plan and layer (L+1)'s
// nextAttnIn plan via [FuseCrossLayerPlans] and lowers the combined plan to a
// single persistent [Megakernel] whose [Megakernel.StageAttributionMap] spans
// both layers' stages with #line directives.
func BuildCrossLayerMegakernel(post, nextAttnIn *Plan, glue CrossLayerGlue) (*Plan, Megakernel, error) {
	fused, err := FuseCrossLayerPlans(post, nextAttnIn, glue)
	if err != nil {
		return nil, Megakernel{}, err
	}
	megaName := glue.Name
	if megaName == "" {
		megaName = fused.Name + "_mega"
	}
	mega, err := fused.Megakernel(megaName)
	if err != nil {
		return nil, Megakernel{}, err
	}
	return fused, mega, nil
}

func ensureLayerPrefix(name, prefix string) string {
	if strings.HasPrefix(name, prefix) {
		return name
	}
	trimmed := strings.TrimSuffix(prefix, "_")
	if trimmed != "" && strings.HasPrefix(name, trimmed+"_") {
		return name
	}
	return prefix + name
}

func collectPlanTensorNames(p *Plan, skipInput string) map[string]bool {
	names := make(map[string]bool, len(p.Inputs)+len(p.Temps)+1)
	for _, in := range p.Inputs {
		if in.Name != skipInput {
			names[in.Name] = true
		}
	}
	for _, t := range p.Temps {
		names[t.Name] = true
	}
	if p.Output.Name != "" && p.Output.Name != skipInput {
		names[p.Output.Name] = true
	}
	return names
}

func cloneTensor(t Tensor) Tensor {
	return Tensor{
		Name:  t.Name,
		DType: t.DType,
		Shape: slices.Clone(t.Shape),
	}
}

// preparePlanForMegakernel deep-clones p, unfuses any SkinnyGEMM tails into
// standalone RowKernel stages, and disables cluster / warp-specialized flags
// that are incompatible with [Plan.Megakernel].
func preparePlanForMegakernel(p *Plan) *Plan {
	out := &Plan{
		Name:   p.Name,
		Inputs: make([]Tensor, len(p.Inputs)),
		Output: cloneTensor(p.Output),
	}
	for i, in := range p.Inputs {
		out.Inputs[i] = cloneTensor(in)
	}

	dropTemps := make(map[string]bool)
	for _, k := range p.Kernels {
		ck := clonePlanKernel(k)
		if ck.UMMA != nil && ck.UMMA.Weight == WeightBF16 {
			u := ck.UMMA
			if ck.M <= 16 {
				g := SkinnyGEMM{
					Name:     u.Name,
					N:        u.N,
					K:        u.K,
					MaxM:     max(ck.M, 1),
					Weight:   u.Weight,
					Out:      u.Out,
					Epilogue: u.Epilogue,
					Extras:   slices.Clone(u.Extras),
					X:        BF16,
				}
				ck.UMMA = nil
				ck.GEMM = &g
			} else {
				t := TileGEMM{
					Name:     u.Name,
					N:        u.N,
					K:        u.K,
					Weight:   u.Weight,
					Out:      u.Out,
					Epilogue: u.Epilogue,
					Extras:   slices.Clone(u.Extras),
				}
				if t.N%128 != 0 {
					t.BN = 64
				}
				if ck.M <= 64 {
					t.BM = 64
				}
				ck.UMMA = nil
				ck.Tile = &t
			}
		}
		if ck.MMA != nil && ck.M <= 16 {
			m := ck.MMA
			g := SkinnyGEMM{
				Name:     m.Name,
				N:        m.N,
				K:        m.K,
				MaxM:     max(ck.M, 1),
				Weight:   WeightBF16,
				Out:      m.Out,
				Epilogue: m.Epilogue,
				Extras:   slices.Clone(m.Extras),
				X:        BF16,
				GLU:      m.GLU,
			}
			ck.MMA = nil
			ck.GEMM = &g
		}
		if ck.GEMM != nil {
			if ck.GEMM.Warps != 0 && ck.GEMM.Warps != 8 {
				ck.GEMM.Warps = 8
			}
			if ck.GEMM.RowsPerBlock == 0 && ck.GEMM.N <= 512 && ck.GEMM.Weight != WeightNVFP4 {
				for _, r := range []int{32, 16, 8, 4} {
					if ck.GEMM.N%r == 0 && r*max(ck.M, ck.GEMM.MaxM, 1) <= 256 {
						ck.GEMM.RowsPerBlock = r
						break
					}
				}
			}
			if ck.GEMM.Tail != nil {
				gk, rk, countName := unfuseGEMMTail(ck)
				if countName != "" {
					dropTemps[countName] = true
				}
				out.Kernels = append(out.Kernels, gk, rk)
				continue
			}
		}
		if ck.Tile != nil {
			if ck.Tile.WarpSpecialized {
				ck.Tile.WarpSpecialized = false
				ck.Tile.CompactScales = false
				ck.Tile.VectorActivationLoads = false
			}
		}
		if ck.Row != nil {
			if ck.Row.Cluster {
				ck.Row.Cluster = false
				ck.Row.ClusterMaxRows = 0
			}
		}
		out.Kernels = append(out.Kernels, ck)
	}

	for _, t := range p.Temps {
		if !dropTemps[t.Name] {
			out.Temps = append(out.Temps, cloneTensor(t))
		}
	}
	return out
}

func unfuseGEMMTail(k PlanKernel) (gk, rk PlanKernel, countName string) {
	g := *k.GEMM
	tailRow := cloneRowKernel(*g.Tail)
	tailRow.Cluster = false
	tailRow.ClusterMaxRows = 0
	tailIn := g.TailIn

	g.Tail = nil
	g.TailIn = ""
	baseParams := len(g.params())
	if baseParams > len(k.Args) {
		baseParams = len(k.Args)
	}
	outIdx := baseParams - 1
	if g.SplitK > 1 && outIdx > 0 {
		outIdx--
	}
	gemmOutBuf := ""
	if outIdx >= 0 && outIdx < len(k.Args) {
		gemmOutBuf = k.Args[outIdx]
	}

	var extraArgs []string
	if len(k.Args) > baseParams {
		countName = k.Args[len(k.Args)-1]
		extraArgs = k.Args[baseParams : len(k.Args)-1]
	}

	for i := range tailRow.Inputs {
		if tailRow.Inputs[i].buffer() == tailIn && gemmOutBuf != "" {
			tailRow.Inputs[i].Buffer = gemmOutBuf
		}
	}

	var rkArgs []string
	extraIdx := 0
	for _, tp := range tailRow.params()[1:] {
		pname := paramName(tp)
		if pname == tailIn || pname == gemmOutBuf {
			rkArgs = append(rkArgs, gemmOutBuf)
		} else if extraIdx < len(extraArgs) {
			rkArgs = append(rkArgs, extraArgs[extraIdx])
			extraIdx++
		}
	}

	gk = k
	gk.GEMM = &g
	gk.Args = slices.Clone(k.Args[:baseParams])
	rk = PlanKernel{
		Row:  &tailRow,
		Rows: k.M,
		Args: rkArgs,
	}
	return gk, rk, countName
}

func cloneRowKernel(r RowKernel) RowKernel {
	out := r
	out.Inputs = slices.Clone(r.Inputs)
	out.Values = make([]RowValue, len(r.Values))
	for i, v := range r.Values {
		v.Args = slices.Clone(v.Args)
		out.Values[i] = v
	}
	out.Outputs = slices.Clone(r.Outputs)
	return out
}

func clonePlanKernel(k PlanKernel) PlanKernel {
	ck := k
	ck.Args = slices.Clone(k.Args)
	if k.GEMM != nil {
		g := *k.GEMM
		g.Extras = slices.Clone(k.GEMM.Extras)
		if k.GEMM.Tail != nil {
			tr := cloneRowKernel(*k.GEMM.Tail)
			g.Tail = &tr
		}
		ck.GEMM = &g
	}
	if k.Tile != nil {
		t := *k.Tile
		t.Extras = slices.Clone(k.Tile.Extras)
		ck.Tile = &t
	}
	if k.MMA != nil {
		m := *k.MMA
		m.Extras = slices.Clone(k.MMA.Extras)
		ck.MMA = &m
	}
	if k.UMMA != nil {
		u := *k.UMMA
		u.Extras = slices.Clone(k.UMMA.Extras)
		ck.UMMA = &u
	}
	if k.Quant != nil {
		q := *k.Quant
		ck.Quant = &q
	}
	if k.Row != nil {
		r := cloneRowKernel(*k.Row)
		ck.Row = &r
	}
	if k.Attention != nil {
		a := *k.Attention
		ck.Attention = &a
	}
	if k.TopK != nil {
		tk := *k.TopK
		ck.TopK = &tk
	}
	if k.Align != nil {
		al := *k.Align
		ck.Align = &al
	}
	if k.ExpertSum != nil {
		es := *k.ExpertSum
		ck.ExpertSum = &es
	}
	return ck
}

func applyPlanTensorRename(p *Plan, rename map[string]string) {
	mapName := func(s string) string {
		if r, ok := rename[s]; ok && r != "" {
			return r
		}
		return s
	}
	for i := range p.Inputs {
		p.Inputs[i].Name = mapName(p.Inputs[i].Name)
	}
	for i := range p.Temps {
		p.Temps[i].Name = mapName(p.Temps[i].Name)
	}
	if p.Output.Name != "" {
		p.Output.Name = mapName(p.Output.Name)
	}
	for i := range p.Kernels {
		k := &p.Kernels[i]
		for j, a := range k.Args {
			k.Args[j] = mapName(a)
		}
		if k.Row != nil {
			for j := range k.Row.Inputs {
				buf := k.Row.Inputs[j].buffer()
				mapped := mapName(buf)
				if k.Row.Inputs[j].Buffer != "" || mapped != k.Row.Inputs[j].Name {
					k.Row.Inputs[j].Buffer = mapped
				}
			}
			for j := range k.Row.Outputs {
				k.Row.Outputs[j].Name = mapName(k.Row.Outputs[j].Name)
				if k.Row.Outputs[j].Index != "" {
					k.Row.Outputs[j].Index = mapName(k.Row.Outputs[j].Index)
				}
			}
			if k.Row.SlotIDs != "" {
				k.Row.SlotIDs = mapName(k.Row.SlotIDs)
			}
		}
	}
}

func planKernelName(k PlanKernel) string {
	switch {
	case k.GEMM != nil:
		return k.GEMM.Name
	case k.Tile != nil:
		return k.Tile.Name
	case k.MMA != nil:
		return k.MMA.Name
	case k.UMMA != nil:
		return k.UMMA.Name
	case k.Quant != nil:
		return k.Quant.Name
	case k.Row != nil:
		return k.Row.Name
	case k.Attention != nil:
		return k.Attention.Name
	case k.TopK != nil:
		return k.TopK.Name
	case k.Align != nil:
		return k.Align.Name
	case k.ExpertSum != nil:
		return k.ExpertSum.Name
	default:
		return ""
	}
}

func setPlanKernelName(k *PlanKernel, name string) {
	switch {
	case k.GEMM != nil:
		k.GEMM.Name = name
	case k.Tile != nil:
		k.Tile.Name = name
	case k.MMA != nil:
		k.MMA.Name = name
	case k.UMMA != nil:
		k.UMMA.Name = name
	case k.Quant != nil:
		k.Quant.Name = name
	case k.Row != nil:
		k.Row.Name = name
	case k.Attention != nil:
		k.Attention.Name = name
	case k.TopK != nil:
		k.TopK.Name = name
	case k.Align != nil:
		k.Align.Name = name
	case k.ExpertSum != nil:
		k.ExpertSum.Name = name
	}
}
