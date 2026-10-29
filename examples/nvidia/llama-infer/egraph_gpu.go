package main

import (
	"errors"
	"fmt"
	"math"
	"slices"
	"strings"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudaattn"
	"github.com/tmc/lib/nvidia/x/cudadevicetable"
	"github.com/tmc/lib/nvidia/x/cudakv"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/egraph/hopper"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// An egraphLaunch is one generated kernel launch with its arguments bound.
// A megakernel launch also carries the ready-counter epoch, which must grow
// with every launch: epoch is the index of that argument in args, or -1.
type egraphLaunch struct {
	fn          driver.Function
	grid, block [3]uint32
	smem        uint32
	args        []any
	epoch       int
	pos         []egraphPosArg     // arguments that follow the position
	kern        *hopper.PlanKernel // the plan kernel, for row-count changes
}

// withRows returns l over the first n rows of its plan's M: a tile, tcgen05
// or quantize kernel, or a row kernel, takes the row count as its first
// argument and sizes its grid by it.
func (l egraphLaunch) withRows(n int) egraphLaunch {
	k := l.kern
	var grid, block [3]int
	switch {
	case k == nil || k.GEMM != nil:
		return l
	case k.MMA != nil:
		// The grid covers MaxM rows whatever the count.
		l.args = slices.Clone(l.args)
		l.args[0] = int32(n)
		return l
	case k.Tile != nil:
		grid, block = k.Tile.Grid(n, 1)
	case k.Row != nil:
		grid, block = k.Row.Grid(n)
	case k.UMMA != nil:
		// The tensor maps cover the plan's M rows; the kernel stores
		// only the first n.
		grid, block = k.UMMA.Grid(n)
	case k.Quant != nil:
		grid, block = k.Quant.Grid(n)
	}
	l.args = slices.Clone(l.args)
	l.args[0] = int32(n)
	for i := range 3 {
		l.grid[i], l.block[i] = uint32(grid[i]), uint32(block[i])
	}
	return l
}

// An egraphPosArg is a launch argument that depends on the token position:
// with stride > 0 a device pointer, advanced stride bytes per position from
// its position-0 value; with stride 0 the sequence length pos+1.
type egraphPosArg struct {
	idx    int
	stride driver.DevicePtr
}

// setPos sets args' position-dependent arguments for pos from base, the
// arguments at position 0.
func setPos(args, base []any, pa []egraphPosArg, pos int) {
	for _, a := range pa {
		if a.stride == 0 {
			args[a.idx] = int32(pos + 1)
			continue
		}
		args[a.idx] = base[a.idx].(driver.DevicePtr) + driver.DevicePtr(pos)*a.stride
	}
}

// An egraphDecoder runs the decode step's projections, norms, SwiGLU and
// residuals as kernels the e-graph generator emits ([buildEgraphPlans]).
// It owns the generated module, the fused QKV and gate·up weights those
// kernels read, and their intermediates.
//
// With mega set, each decode plan runs as one persistent
// [hopper.Megakernel] launch whose stages wait on ready counters instead
// of on the stream.
type egraphDecoder struct {
	plans *egraphPlans
	megas map[*hopper.Plan]hopper.Megakernel
	mod   driver.Module
	fns   map[string]driver.Function // kernel name -> function
	// pdl holds the functions that wait for their predecessor grid at
	// entry ([hopper.AddPDLEntry]), which may launch programmatically.
	pdl   map[driver.Function]bool
	ready map[*hopper.Plan]cudamem.Slice[uint64]
	epoch uint32

	bqkv     []cudamem.Slice[float32]
	qkv, x2  cudamem.Slice[float32]
	xn       cudamem.Slice[uint16]             // the normalized residual a layer reads
	temps32  map[string]cudamem.Slice[float32] // by name
	temps16  map[string]cudamem.Slice[uint16]
	temps8   map[string]cudamem.Slice[uint8] // NVFP4 activations and scales
	tempsI32 map[string]cudamem.Slice[int32] // gemm tail counters
	tempPtr  map[string]driver.DevicePtr

	attnIn, post [][]egraphLaunch // per layer
	norm, head   []egraphLaunch

	// With plans.attn: the BF16 KV cache, the rotary cos|sin table, the
	// page tables and seqlens the emitted kernels read, q in BF16, and the
	// per-layer rotary and attention launches.
	kc, vc     []cudamem.Slice[uint16]
	cs         cudamem.Slice[float32]
	ropePage   cudamem.Slice[int32] // {0}
	seqLen1    cudamem.Slice[int32] // {1}
	pages      cudamem.Slice[int32] // identity page table
	qbf        cudamem.Slice[uint16]
	rope, sdpa []egraphLaunch
	ws         cudamem.Slice[float32] // split attention partials
	combine    []egraphLaunch         // with splits: merges ws into sp.AttnOut
	posKind    map[driver.Function]string
	posArgs    map[driver.Function][]egraphPosArg

	pf *egraphPrefill // with plans.pf
	// pfWS is the Stream-K workspace the prefill tile gemms share, when
	// any of them uses Stream-K.
	pfWS cudamem.Slice[float32]
	// mmaWS holds a workspace for each split-K skinny mma, at mmaOff[name]
	// floats. Every layer's launch of a kernel shares its workspace: they
	// run one after another, and each leaves the counters zero.
	mmaWS  cudamem.Slice[float32]
	mmaOff map[string]int

	// tmaps are the tensor maps of the tcgen05 gemms ([hopper.UMMAGEMM]),
	// encoded by encode (nil means [encodeTensorMap]) once per buffer and
	// shape when the launches are bound. Each launch passes its maps by
	// pointer, so cuLaunchKernel copies them into the parameter space.
	tmaps  map[egraphTensorMap]*driver.TensorMap
	encode func(hopper.TensorMap2D, driver.DevicePtr) (*driver.TensorMap, error)
}

// An egraphTensorMap is a tensor map's shape and the buffer it covers.
type egraphTensorMap struct {
	hopper.TensorMap2D // Operand zero
	base               driver.DevicePtr
}

// egraphBuffers are the device buffers the decode step's launches read and
// write, other than the plans' temporaries.
type egraphBuffers struct {
	x, x2        driver.DevicePtr // the residual stream, alternating by layer
	xn           driver.DevicePtr // the normalized residual a layer reads
	qkv          driver.DevicePtr
	attnOut      driver.DevicePtr
	norm, lmHead driver.DevicePtr
	logits       driver.DevicePtr
	layers       []egraphLayerBuffers

	// With generated attention: q in BF16, the rotary cos|sin table, the
	// rotary page table {0}, seqlens {1}, the attention page table, and
	// the split attention workspace (0 without splits).
	qbf, cs, ropePage, seqLen1, pages, ws driver.DevicePtr
}

type egraphLayerBuffers struct {
	attnNorm, wqkv, bqkv, wo, ffnNorm, wgu, wdown driver.DevicePtr // bqkv 0 without biases
	k, v                                          driver.DevicePtr // with generated attention: the BF16 cache
}

func newEgraphDecoder(dev *cuda.Device, cfg *ModelConfig, gpuW *GPUModelWeights, sp *GPUScratchpad, opt egraphOptions) (eg *egraphDecoder, err error) {
	if gpuW.DType != weightBF16 {
		return nil, errors.New("-egraph needs -weight-dtype bf16")
	}
	for i, l := range gpuW.Layers {
		if l.WQKV.buf == nil || l.WGateUp.buf == nil {
			return nil, fmt.Errorf("layer %d has no fused qkv and gate_up weights: load the model with streamModel", i)
		}
	}
	if (opt.PrefillRows != 0 || opt.AttnSplits != 0) && !opt.Attn {
		return nil, errors.New("e-graph prefill and attention splits need generated attention")
	}
	if opt.SMs == 0 {
		if opt.SMs, err = dev.Attribute(driver.DevAttrMultiprocessorCount); err != nil {
			return nil, fmt.Errorf("read SM count: %w", err)
		}
	}
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		return nil, fmt.Errorf("read compute capability: %w", err)
	}
	if err := checkEgraphTarget(fmt.Sprintf("sm_%d%da", major, minor), opt.Profile); err != nil {
		return nil, err
	}
	bias := len(gpuW.Layers) > 0 && gpuW.Layers[0].QBias.Len() > 0
	plans, err := buildEgraphPlans(cfg, bias, opt)
	if err != nil {
		return nil, fmt.Errorf("build e-graph plans: %w", err)
	}
	eg = &egraphDecoder{
		plans:    plans,
		megas:    make(map[*hopper.Plan]hopper.Megakernel),
		fns:      make(map[string]driver.Function),
		pdl:      make(map[driver.Function]bool),
		ready:    make(map[*hopper.Plan]cudamem.Slice[uint64]),
		temps32:  make(map[string]cudamem.Slice[float32]),
		temps16:  make(map[string]cudamem.Slice[uint16]),
		temps8:   make(map[string]cudamem.Slice[uint8]),
		tempsI32: make(map[string]cudamem.Slice[int32]),
		tempPtr:  make(map[string]driver.DevicePtr),
		posKind:  make(map[driver.Function]string),
		posArgs:  make(map[driver.Function][]egraphPosArg),
		mmaOff:   make(map[string]int),
		tmaps:    make(map[egraphTensorMap]*driver.TensorMap),
	}
	defer func() {
		if err != nil {
			if cerr := eg.Close(); cerr != nil {
				err = errors.Join(err, cerr)
			}
			eg = nil
		}
	}()
	if opt.Mega {
		for _, p := range eg.decodePlans() {
			m, err := p.Megakernel("")
			if err != nil {
				return nil, err
			}
			eg.megas[p] = m
			if eg.ready[p], err = cudamem.Alloc[uint64](dev, m.ReadyWords()); err != nil {
				return nil, fmt.Errorf("alloc %s ready counters: %w", m.Name, err)
			}
			if err := eg.ready[p].CopyFromHost(make([]uint64, m.ReadyWords())); err != nil {
				return nil, fmt.Errorf("zero %s ready counters: %w", m.Name, err)
			}
		}
	}
	if err := eg.compile(dev, opt.PDL); err != nil {
		return nil, err
	}

	if bias {
		eg.bqkv = make([]cudamem.Slice[float32], len(gpuW.Layers))
		for i := range gpuW.Layers {
			l := &gpuW.Layers[i]
			q, k, v := l.QBias.Len(), l.KBias.Len(), l.VBias.Len()
			if eg.bqkv[i], err = cudamem.Alloc[float32](dev, q+k+v); err != nil {
				return nil, fmt.Errorf("alloc layer %d qkv bias: %w", i, err)
			}
			b := eg.bqkv[i]
			if err := errors.Join(
				b.Slice(0, q).CopyDeviceToDevice(l.QBias),
				b.Slice(q, q+k).CopyDeviceToDevice(l.KBias),
				b.Slice(q+k, q+k+v).CopyDeviceToDevice(l.VBias)); err != nil {
				return nil, fmt.Errorf("copy layer %d qkv bias: %w", i, err)
			}
		}
	}
	if eg.qkv, err = cudamem.Alloc[float32](dev, plans.attnIn.Output.Shape[1]); err != nil {
		return nil, fmt.Errorf("alloc qkv: %w", err)
	}
	if eg.x2, err = cudamem.Alloc[float32](dev, cfg.HiddenSize); err != nil {
		return nil, fmt.Errorf("alloc residual: %w", err)
	}
	if eg.xn, err = cudamem.Alloc[uint16](dev, cfg.HiddenSize); err != nil {
		return nil, fmt.Errorf("alloc normalized residual: %w", err)
	}
	for _, p := range eg.planList() {
		for _, t := range p.Temps {
			n := 1
			for _, d := range t.Shape {
				n *= d
			}
			switch t.DType {
			case hopper.F32:
				if eg.temps32[t.Name], err = cudamem.Alloc[float32](dev, n); err != nil {
					return nil, fmt.Errorf("alloc %s: %w", t.Name, err)
				}
				eg.tempPtr[t.Name] = eg.temps32[t.Name].Ptr()
			case hopper.BF16:
				if eg.temps16[t.Name], err = cudamem.Alloc[uint16](dev, n); err != nil {
					return nil, fmt.Errorf("alloc %s: %w", t.Name, err)
				}
				eg.tempPtr[t.Name] = eg.temps16[t.Name].Ptr()
			case hopper.U8:
				if eg.temps8[t.Name], err = cudamem.Alloc[uint8](dev, n); err != nil {
					return nil, fmt.Errorf("alloc %s: %w", t.Name, err)
				}
				eg.tempPtr[t.Name] = eg.temps8[t.Name].Ptr()
			case hopper.I32:
				// A gemm tail's counter: zero before the first launch.
				if eg.tempsI32[t.Name], err = cudamem.Alloc[int32](dev, n); err != nil {
					return nil, fmt.Errorf("alloc %s: %w", t.Name, err)
				}
				if err := eg.tempsI32[t.Name].CopyFromHost(make([]int32, n)); err != nil {
					return nil, fmt.Errorf("zero %s: %w", t.Name, err)
				}
				eg.tempPtr[t.Name] = eg.tempsI32[t.Name].Ptr()
			default:
				return nil, fmt.Errorf("temporary %s has dtype %v", t.Name, t.DType)
			}
		}
	}

	if opt.Attn {
		hd := cfg.HeadDim()
		slots := cfg.MaxSeqLen
		if plans.pf != nil {
			slots = plans.pf.cacheSlots
		}
		n := slots * cfg.NumKeyValueHeads * hd
		eg.kc = make([]cudamem.Slice[uint16], cfg.NumHiddenLayers)
		eg.vc = make([]cudamem.Slice[uint16], cfg.NumHiddenLayers)
		for i := range cfg.NumHiddenLayers {
			if eg.kc[i], err = cudamem.Alloc[uint16](dev, n); err != nil {
				return nil, fmt.Errorf("alloc layer %d k cache: %w", i, err)
			}
			if eg.vc[i], err = cudamem.Alloc[uint16](dev, n); err != nil {
				return nil, fmt.Errorf("alloc layer %d v cache: %w", i, err)
			}
		}
		if eg.cs, err = cudamem.Alloc[float32](dev, cfg.MaxSeqLen*hd); err != nil {
			return nil, fmt.Errorf("alloc rope table: %w", err)
		}
		if eg.ropePage, err = cudamem.Alloc[int32](dev, 1); err != nil {
			return nil, fmt.Errorf("alloc rope page table: %w", err)
		}
		if eg.seqLen1, err = cudamem.Alloc[int32](dev, 1); err != nil {
			return nil, fmt.Errorf("alloc seqlens: %w", err)
		}
		if eg.pages, err = cudamem.Alloc[int32](dev, plans.sdpa.MaxPages); err != nil {
			return nil, fmt.Errorf("alloc attention page table: %w", err)
		}
		if eg.qbf, err = cudamem.Alloc[uint16](dev, cfg.NumAttentionHeads*hd); err != nil {
			return nil, fmt.Errorf("alloc q: %w", err)
		}
		if n := plans.sdpa.WorkspaceFloats(1); n > 0 {
			if eg.ws, err = cudamem.Alloc[float32](dev, n); err != nil {
				return nil, fmt.Errorf("alloc attention workspace: %w", err)
			}
		}
		if err := eg.uploadAttention(cfg); err != nil {
			return nil, err
		}
	}
	b := egraphBuffers{
		x: sp.X.Ptr(), x2: eg.x2.Ptr(), xn: eg.xn.Ptr(), qkv: eg.qkv.Ptr(), attnOut: sp.AttnOut.Ptr(),
		norm: gpuW.Norm.Ptr(), lmHead: gpuW.LMHead.Ptr(), logits: sp.Logits.Ptr(),
		qbf: eg.qbf.Ptr(), cs: eg.cs.Ptr(), ropePage: eg.ropePage.Ptr(),
		seqLen1: eg.seqLen1.Ptr(), pages: eg.pages.Ptr(), ws: eg.ws.Ptr(),
	}
	for i := range gpuW.Layers {
		lw := &gpuW.Layers[i]
		lb := egraphLayerBuffers{
			attnNorm: lw.InputNorm.Ptr(), wqkv: lw.WQKV.Ptr(), wo: lw.WO.Ptr(),
			ffnNorm: lw.PostNorm.Ptr(), wgu: lw.WGateUp.Ptr(), wdown: lw.WDown.Ptr(),
		}
		if bias {
			lb.bqkv = eg.bqkv[i].Ptr()
		}
		if opt.Attn {
			lb.k, lb.v = eg.kc[i].Ptr(), eg.vc[i].Ptr()
		}
		b.layers = append(b.layers, lb)
	}
	if err := eg.bindDecode(cfg, b); err != nil {
		return nil, err
	}
	if pl := plans.pf; pl != nil {
		h := cfg.HiddenSize
		qDim := cfg.NumAttentionHeads * cfg.HeadDim()
		kvDim := cfg.NumKeyValueHeads * cfg.HeadDim()
		groups := (pl.rows + pl.qtok - 1) / pl.qtok
		// Every chunk but the last fills its groups, so the prompt needs
		// at most one partial group per chunk beyond MaxSeqLen/qtok, and
		// a narrow last chunk at most its narrowQTok-row groups.
		maxGroups := (cfg.MaxSeqLen+pl.qtok-1)/pl.qtok + (cfg.MaxSeqLen+pl.rows-1)/pl.rows + (pl.narrow+pl.narrowQTok-1)/pl.narrowQTok
		eg.pf = &egraphPrefill{rows: pl.rows, qtok: pl.qtok, narrow: pl.narrow, narrowQTok: pl.narrowQTok, gather: eg.fns[embedGatherName]}
		if eg.pf.tokens, err = cudamem.Alloc[int32](dev, cfg.MaxSeqLen); err != nil {
			return nil, fmt.Errorf("alloc prefill tokens: %w", err)
		}
		if eg.pf.positions, err = cudamem.Alloc[int32](dev, cfg.MaxSeqLen); err != nil {
			return nil, fmt.Errorf("alloc prefill positions: %w", err)
		}
		if eg.pf.seqLens, err = cudamem.Alloc[int32](dev, maxGroups); err != nil {
			return nil, fmt.Errorf("alloc prefill seq lens: %w", err)
		}
		if eg.pf.seqIDs, err = cudamem.Alloc[int32](dev, pl.rows); err != nil {
			return nil, fmt.Errorf("alloc prefill seq ids: %w", err)
		}
		if eg.pf.pages, err = cudamem.Alloc[int32](dev, groups*pl.attn.MaxPages); err != nil {
			return nil, fmt.Errorf("alloc prefill page table: %w", err)
		}
		for i := range eg.pf.x {
			if eg.pf.x[i], err = cudamem.Alloc[float32](dev, pl.rows*h); err != nil {
				return nil, fmt.Errorf("alloc prefill residual: %w", err)
			}
		}
		if eg.pf.qkv, err = cudamem.Alloc[uint16](dev, pl.rows*(qDim+2*kvDim)); err != nil {
			return nil, fmt.Errorf("alloc prefill qkv: %w", err)
		}
		// Attention reads and writes whole groups.
		if eg.pf.q, err = cudamem.Alloc[uint16](dev, groups*pl.qtok*qDim); err != nil {
			return nil, fmt.Errorf("alloc prefill q: %w", err)
		}
		if eg.pf.attnOut, err = cudamem.Alloc[uint16](dev, groups*pl.qtok*qDim); err != nil {
			return nil, fmt.Errorf("alloc prefill attention: %w", err)
		}
		n := 0
		for _, p := range []*hopper.Plan{pl.narrowIn, pl.narrowPost} {
			for _, k := range p.Kernels {
				if k.MMA != nil && k.MMA.WorkspaceBytes() > 0 {
					eg.mmaOff[k.MMA.Name] = n
					n += k.MMA.WorkspaceBytes() / 4
				}
			}
		}
		if n > 0 {
			if eg.mmaWS, err = cudamem.Alloc[float32](dev, n); err != nil {
				return nil, fmt.Errorf("alloc prefill skinny mma workspace: %w", err)
			}
			if err := eg.mmaWS.CopyFromHost(make([]float32, n)); err != nil {
				return nil, fmt.Errorf("zero prefill skinny mma workspace: %w", err)
			}
		}
		if n := prefillWorkspaceFloats(pl); n > 0 {
			if eg.pfWS, err = cudamem.Alloc[float32](dev, n); err != nil {
				return nil, fmt.Errorf("alloc prefill stream-k workspace: %w", err)
			}
			if err := eg.pfWS.CopyFromHost(make([]float32, n)); err != nil {
				return nil, fmt.Errorf("zero prefill stream-k workspace: %w", err)
			}
		}
		if err := eg.bindPrefill(dev, cfg, gpuW); err != nil {
			return nil, err
		}
	}
	return eg, nil
}

// uploadAttention fills the BF16 KV cache with zeros and uploads the tables
// the emitted rotary and attention kernels read.
func (eg *egraphDecoder) uploadAttention(cfg *ModelConfig) error {
	hd := cfg.HeadDim()
	maxSeq := cfg.MaxSeqLen
	// Prefill attention loads, and masks, cache rows past a chunk's last
	// token; a NaN there would still poison the output (0·NaN), so start
	// from zeros.
	for i := range eg.kc {
		for _, c := range []cudamem.Slice[uint16]{eg.kc[i], eg.vc[i]} {
			if err := c.CopyFromHost(make([]uint16, c.Len())); err != nil {
				return fmt.Errorf("zero layer %d kv cache: %w", i, err)
			}
		}
	}

	// cos_sin_table row pos holds cos(pos·f_i) then sin(pos·f_i), i < hd/2,
	// the NeoX (split-half) layout llama-infer's RoPE uses.
	half := hd / 2
	table := make([]float32, maxSeq*hd)
	for pos := range maxSeq {
		for i := range half {
			a := float64(pos) * float64(ropeInvFreq(i, hd, cfg.RopeTheta, cfg.RoPEScaling))
			table[pos*hd+i] = float32(math.Cos(a))
			table[pos*hd+half+i] = float32(math.Sin(a))
		}
	}
	if err := eg.cs.CopyFromHost(table); err != nil {
		return fmt.Errorf("upload rope table: %w", err)
	}
	// The rotary kernel writes cache slot page_table[0]·1 + (seqlens[0]-1)
	// = 0 of the pointers it is given, which the decoder offsets to the
	// token's row. The attention kernel reads page i at row 16i.
	pages := make([]int32, eg.pages.Len())
	for i := range pages {
		pages[i] = int32(i)
	}
	for _, t := range []struct {
		s cudamem.Slice[int32]
		v []int32
	}{{eg.ropePage, []int32{0}}, {eg.seqLen1, []int32{1}}, {eg.pages, pages}} {
		if err := t.s.CopyFromHost(t.v); err != nil {
			return fmt.Errorf("upload attention tables: %w", err)
		}
	}
	return nil
}

// bindDecode binds the decode step's launches to b: every layer's plans,
// the first layer's norm, the head and, with generated attention, the
// rotary and attention launches.
func (eg *egraphDecoder) bindDecode(cfg *ModelConfig, b egraphBuffers) error {
	plans := eg.plans
	var err error
	if eg.norm, err = eg.bind(plans.norm, map[string]driver.DevicePtr{
		"x":                    b.x,
		"norm":                 b.layers[0].attnNorm,
		plans.norm.Output.Name: b.xn,
	}); err != nil {
		return err
	}
	// The residual stream alternates between b.x and b.x2, one layer each,
	// and each layer's post normalizes it into xn for the next layer, or
	// for the head after the last.
	cur, next := b.x, b.x2
	for i, lb := range b.layers {
		nextNorm := b.norm
		if i+1 < len(b.layers) {
			nextNorm = b.layers[i+1].attnNorm
		}
		bufs := map[string]driver.DevicePtr{
			"xn":                     b.xn,
			"x":                      cur,
			"w_qkv":                  lb.wqkv,
			plans.attnIn.Output.Name: b.qkv,
			"attn":                   b.attnOut,
			"w_o":                    lb.wo,
			"ffn_norm":               lb.ffnNorm,
			"w_gu":                   lb.wgu,
			"w_down":                 lb.wdown,
			"next_norm":              nextNorm,
			plans.postRes:            next,
			plans.post.Output.Name:   b.xn,
		}
		if lb.bqkv != 0 {
			bufs["b_qkv"] = lb.bqkv
		}
		in, err := eg.bind(plans.attnIn, bufs)
		if err != nil {
			return err
		}
		post, err := eg.bind(plans.post, bufs)
		if err != nil {
			return err
		}
		eg.attnIn = append(eg.attnIn, in)
		eg.post = append(eg.post, post)
		cur, next = next, cur
	}
	eg.head, err = eg.bind(plans.head, map[string]driver.DevicePtr{
		"xn":                   b.xn,
		"lm_head":              b.lmHead,
		plans.head.Output.Name: b.logits,
	})
	if err != nil || !plans.attn {
		return err
	}

	hd := cfg.HeadDim()
	kvDim := cfg.NumKeyValueHeads * hd

	ropeFn, sdpaFn := eg.fns[plans.rope.Name], eg.fns[plans.sdpa.Name]
	row := driver.DevicePtr(2 * kvDim)
	eg.posKind[ropeFn], eg.posKind[sdpaFn] = "rope_kv", "attn"
	eg.posArgs[ropeFn] = []egraphPosArg{{idx: 1, stride: driver.DevicePtr(4 * hd)}, {idx: 8, stride: row}, {idx: 9, stride: row}}
	eg.posArgs[sdpaFn] = []egraphPosArg{{idx: 7}}
	rg, rb := plans.rope.Grid(1)
	for _, lb := range b.layers {
		// qkv, cos_sin_table, positions, seq_ids, seqlens, kv_positions,
		// page_table, q_out, k_cache, v_cache
		eg.rope = append(eg.rope, egraphLaunch{
			fn: ropeFn, grid: [3]uint32{uint32(rg[0]), uint32(rg[1]), uint32(rg[2])}, block: [3]uint32{uint32(rb[0]), uint32(rb[1]), uint32(rb[2])}, epoch: -1,
			args: []any{b.qkv, b.cs, driver.DevicePtr(0), driver.DevicePtr(0), b.seqLen1, driver.DevicePtr(0),
				b.ropePage, b.qbf, lb.k, lb.v},
			pos: eg.posArgs[ropeFn],
		})
		// q, k_pages, v_pages, k_scales, v_scales, page_table, out (ws
		// with splits), seq_len
		g := plans.sdpa.Grid(1)
		blk := plans.sdpa.Block()
		out := b.attnOut
		if b.ws != 0 {
			out = b.ws
		}
		eg.sdpa = append(eg.sdpa, egraphLaunch{
			fn: sdpaFn, grid: [3]uint32{uint32(g[0]), uint32(g[1]), uint32(g[2])}, block: [3]uint32{uint32(blk[0]), uint32(blk[1]), uint32(blk[2])}, epoch: -1,
			args: []any{b.qbf, lb.k, lb.v, driver.DevicePtr(0), driver.DevicePtr(0), b.pages, out, int32(1)},
			pos:  eg.posArgs[sdpaFn],
		})
	}
	if b.ws != 0 {
		// ws, out
		g, blk := plans.sdpa.CombineGrid(1)
		eg.combine = []egraphLaunch{{
			fn: eg.fns[plans.sdpa.Name+"_combine"], epoch: -1,
			grid: [3]uint32{uint32(g[0]), uint32(g[1]), uint32(g[2])}, block: [3]uint32{uint32(blk[0]), uint32(blk[1]), uint32(blk[2])},
			args: []any{b.ws, b.attnOut},
		}}
	}
	return nil
}

// layerLaunches returns layer i's generated launches in order, with
// generated attention: the QKV projection, rotary and cache store,
// attention and its combine, and the rest of the layer.
func (eg *egraphDecoder) layerLaunches(i int) []egraphLaunch {
	return slices.Concat(eg.attnIn[i], eg.rope[i:i+1], eg.sdpa[i:i+1], eg.combine, eg.post[i])
}

// positionArgs reports how a DecodeGraph must update a node running fn
// when the position changes: its kind and its position-dependent arguments,
// or ok false when fn's arguments do not depend on the position.
func (eg *egraphDecoder) positionArgs(fn driver.Function) (kind string, pa []egraphPosArg, ok bool) {
	pa, ok = eg.posArgs[fn]
	return eg.posKind[fn], pa, ok
}

// decodePlans returns the decode step's plans, which -egraph-mega runs as
// megakernels.
func (eg *egraphDecoder) decodePlans() []*hopper.Plan {
	return []*hopper.Plan{eg.plans.norm, eg.plans.attnIn, eg.plans.post, eg.plans.head}
}

func (eg *egraphDecoder) planList() []*hopper.Plan {
	ps := eg.decodePlans()
	if pf := eg.plans.pf; pf != nil {
		ps = append(ps, pf.attnIn, pf.post, pf.narrowIn, pf.narrowPost)
	}
	return ps
}

// kernelName returns the NVRTC name expression for k.
func kernelName(k hopper.PlanKernel) (string, error) {
	switch {
	case k.GEMM != nil:
		return fmt.Sprintf("%s<%d>", k.GEMM.Name, k.M), nil
	case k.Tile != nil:
		return k.Tile.Name, nil
	case k.MMA != nil:
		return k.MMA.Name, nil
	case k.Row != nil:
		return k.Row.Name, nil
	case k.UMMA != nil:
		return k.UMMA.Name, nil
	case k.Quant != nil:
		return k.Quant.Name, nil
	}
	return "", fmt.Errorf("plan kernel %+v: only gemm, row, tcgen05 and quantize kernels are supported", k)
}

// source returns the CUDA source of eg's plans, with eg.megas in place of
// the plans they run, and the name expressions of its kernels. It needs no
// device.
func (eg *egraphDecoder) source() (string, []string, error) {
	var src strings.Builder
	src.WriteString(hopper.Prelude)
	src.WriteString(hopper.PDLHelpers())
	var names []string
	for _, p := range eg.planList() {
		if m, ok := eg.megas[p]; ok {
			cu, err := m.CUDA()
			if err != nil {
				return "", nil, fmt.Errorf("emit %s: %w", m.Name, err)
			}
			src.WriteString("\n" + cu + "\n")
			names = append(names, m.Name)
			continue
		}
		for _, k := range p.Kernels {
			name, err := kernelName(k)
			if err != nil {
				return "", nil, err
			}
			var cu string
			switch {
			case k.GEMM != nil:
				cu, err = k.GEMM.CUDA()
			case k.Tile != nil:
				cu, err = k.Tile.CUDA()
			case k.MMA != nil:
				cu, err = k.MMA.CUDA()
			case k.UMMA != nil:
				cu, err = k.UMMA.CUDA()
			case k.Quant != nil:
				cu, err = k.Quant.CUDA()
			default:
				cu, err = k.Row.CUDA()
			}
			if err != nil {
				return "", nil, fmt.Errorf("emit %s: %w", name, err)
			}
			src.WriteString("\n" + cu + "\n")
			names = append(names, name)
		}
	}
	if eg.plans.attn {
		for _, k := range []interface {
			CUDA() (string, error)
		}{eg.plans.rope, eg.plans.sdpa} {
			cu, err := k.CUDA()
			if err != nil {
				return "", nil, fmt.Errorf("emit attention: %w", err)
			}
			src.WriteString("\n" + cu + "\n")
		}
		names = append(names, eg.plans.rope.Name, eg.plans.sdpa.Name)
		if eg.plans.sdpa.Splits > 0 {
			names = append(names, eg.plans.sdpa.Name+"_combine")
		}
	}
	if pf := eg.plans.pf; pf != nil {
		for _, k := range []interface {
			CUDA() (string, error)
		}{pf.rope, pf.attn, pf.narrowAttn} {
			cu, err := k.CUDA()
			if err != nil {
				return "", nil, fmt.Errorf("emit prefill attention: %w", err)
			}
			src.WriteString("\n" + cu + "\n")
		}
		src.WriteString("\n" + embedGatherCUDA + "\n")
		names = append(names, pf.rope.Name, pf.attn.Name, pf.narrowAttn.Name, embedGatherName)
		if pf.flash != nil {
			cu, err := pf.flash.CUDA()
			if err != nil {
				return "", nil, fmt.Errorf("emit prefill attention: %w", err)
			}
			src.WriteString("\n" + cu + "\n")
			names = append(names, pf.flash.Name)
		}
	}
	return src.String(), names, nil
}

// compile builds eg's module. With pdl, every kernel in it waits for its
// predecessor grid at entry.
func (eg *egraphDecoder) compile(dev *cuda.Device, pdl bool) error {
	cu, names, err := eg.source()
	if err != nil {
		return err
	}
	waits := make(map[string]bool)
	if pdl {
		var rewritten []string
		if cu, rewritten, err = hopper.AddPDLEntry(cu); err != nil {
			return err
		}
		for _, name := range rewritten {
			waits[name] = true
		}
	}
	cm, err := nvrtcx.Compile(cu,
		nvrtcx.WithName("llama_egraph.cu"),
		nvrtcx.WithDeviceArch(dev),
		nvrtcx.WithStandard("c++20"),
		nvrtcx.WithNameExpressions(names...))
	if err != nil {
		return fmt.Errorf("compile e-graph kernels: %w", releaseErrorOwner(err))
	}
	eg.mod, err = cm.Load(dev)
	if err != nil {
		return fmt.Errorf("load e-graph kernels: %w", releaseErrorOwner(err))
	}
	for _, name := range names {
		lowered, ok := cm.LoweredName(name)
		if !ok {
			return fmt.Errorf("e-graph kernel %s: no lowered name", name)
		}
		fn, err := eg.mod.Function(lowered)
		if err != nil {
			return fmt.Errorf("lookup e-graph kernel %s: %w", name, err)
		}
		eg.fns[name] = fn
		base := name
		if i := strings.IndexByte(name, '<'); i >= 0 {
			base = name[:i]
		}
		if waits[base] {
			eg.pdl[fn] = true
		}
	}
	if pdl && len(eg.pdl) != len(names) {
		return fmt.Errorf("e-graph kernels: %d of %d wait at entry", len(eg.pdl), len(names))
	}
	return nil
}

// programmatic reports whether a launch of fn may be the programmatic
// dependent of the launch before it: fn waits for its predecessor grid at
// entry. It is false for every function without -egraph-pdl.
func (eg *egraphDecoder) programmatic(fn driver.Function) bool {
	return eg != nil && eg.pdl[fn]
}

// bind resolves p's launches against bufs (inputs and output) and eg's
// temporaries.
func (eg *egraphDecoder) bind(p *hopper.Plan, bufs map[string]driver.DevicePtr) ([]egraphLaunch, error) {
	lookup := func(kernel, name string) (driver.DevicePtr, error) {
		ptr, ok := bufs[name]
		if !ok {
			ptr, ok = eg.tempPtr[name]
		}
		if !ok || ptr == 0 {
			return 0, fmt.Errorf("%s: unbound buffer %s", kernel, name)
		}
		return ptr, nil
	}
	if m, ok := eg.megas[p]; ok {
		grid, block := m.Grid()
		l := egraphLaunch{
			fn:    eg.fns[m.Name],
			grid:  [3]uint32{uint32(grid[0]), uint32(grid[1]), uint32(grid[2])},
			block: [3]uint32{uint32(block[0]), uint32(block[1]), uint32(block[2])},
			smem:  uint32(m.SharedBytes()),
		}
		for _, t := range append(append(append([]hopper.Tensor(nil), m.Inputs...), m.Temps...), m.Output) {
			ptr, err := lookup(m.Name, t.Name)
			if err != nil {
				return nil, err
			}
			l.args = append(l.args, ptr)
		}
		l.epoch = len(l.args) + 1
		l.args = append(l.args, eg.ready[p].Ptr(), uint32(0))
		return []egraphLaunch{l}, nil
	}
	var out []egraphLaunch
	for i := range p.Kernels {
		k := &p.Kernels[i]
		name, err := kernelName(*k)
		if err != nil {
			return nil, err
		}
		l := egraphLaunch{fn: eg.fns[name], epoch: -1, kern: k}
		var grid, block [3]int
		args := k.Args
		switch {
		case k.GEMM != nil:
			if grid, block, err = k.GEMM.Grid(); err != nil {
				return nil, fmt.Errorf("%s: %w", name, err)
			}
		case k.Tile != nil:
			grid, block = k.Tile.Grid(k.M, 1)
			l.smem = uint32(k.Tile.SharedBytes())
			l.args = append(l.args, int32(k.M))
		case k.MMA != nil:
			if grid, block, err = k.MMA.Grid(); err != nil {
				return nil, fmt.Errorf("%s: %w", name, err)
			}
			l.smem = uint32(k.MMA.SharedBytes())
			l.args = append(l.args, int32(k.M))
		case k.UMMA != nil:
			// m, the operands' tensor maps, then the other buffers.
			grid, block = k.UMMA.Grid(k.M)
			l.smem = uint32(k.UMMA.SharedBytes())
			l.args = append(l.args, int32(k.M))
			maps, err := k.UMMA.TensorMaps(k.M)
			if err != nil {
				return nil, err
			}
			last := -1
			for _, tm := range maps {
				ptr, err := lookup(name, k.Args[tm.Operand])
				if err != nil {
					return nil, err
				}
				m, err := eg.tensorMap(tm, ptr)
				if err != nil {
					return nil, fmt.Errorf("%s: %w", name, err)
				}
				l.args = append(l.args, m)
				last = max(last, tm.Operand)
			}
			args = k.Args[last+1:]
		case k.Quant != nil:
			grid, block = k.Quant.Grid(k.M)
			l.args = append(l.args, int32(k.M))
		default:
			grid, block = k.Row.Grid(k.Rows)
			l.args = append(l.args, int32(k.Rows))
		}
		for i := range 3 {
			l.grid[i], l.block[i] = uint32(grid[i]), uint32(block[i])
		}
		for _, a := range args {
			ptr, err := lookup(name, a)
			if err != nil {
				return nil, err
			}
			l.args = append(l.args, ptr)
		}
		if k.Tile != nil && k.Tile.StreamK > 0 {
			if eg.pfWS.Len() == 0 {
				return nil, fmt.Errorf("%s: no Stream-K workspace", name)
			}
			l.args = append(l.args, eg.pfWS.Ptr())
		}
		if k.MMA != nil && k.MMA.WorkspaceBytes() > 0 {
			off, ok := eg.mmaOff[name]
			if !ok {
				return nil, fmt.Errorf("%s: no split-K workspace", name)
			}
			l.args = append(l.args, eg.mmaWS.Ptr()+driver.DevicePtr(4*off))
		}
		out = append(out, l)
	}
	return out, nil
}

// tensorMap returns the tensor map tm over base, encoding it on first use.
func (eg *egraphDecoder) tensorMap(tm hopper.TensorMap2D, base driver.DevicePtr) (*driver.TensorMap, error) {
	tm.Operand = 0
	key := egraphTensorMap{tm, base}
	if m, ok := eg.tmaps[key]; ok {
		return m, nil
	}
	encode := eg.encode
	if encode == nil {
		encode = encodeTensorMap
	}
	m, err := encode(tm, base)
	if err != nil {
		return nil, err
	}
	if eg.tmaps == nil {
		eg.tmaps = make(map[egraphTensorMap]*driver.TensorMap)
	}
	eg.tmaps[key] = m
	return m, nil
}

// encodeTensorMap encodes tm over base with cuTensorMapEncodeTiled, as
// [hopper.UMMAGEMM]'s launcher does.
func encodeTensorMap(tm hopper.TensorMap2D, base driver.DevicePtr) (*driver.TensorMap, error) {
	dt := driver.TensorMapDataTypeUint8
	if tm.ElemBytes == 2 {
		dt = driver.TensorMapDataTypeBfloat16
	}
	sw := driver.TensorMapSwizzleNone
	if tm.Swizzle128 {
		sw = driver.TensorMapSwizzle128B
	}
	dims := [2]uint64{tm.Inner, tm.Outer}
	strides := [1]uint64{tm.Stride}
	box := [2]uint32{tm.BoxInner, tm.BoxOuter}
	elems := [2]uint32{1, 1}
	m := driver.NewTensorMap()
	if err := driver.TensorMapEncodeTiled(m, dt, 2, base, &dims[0], &strides[0], &box[0], &elems[0],
		driver.TensorMapInterleaveNone, sw, driver.TensorMapL2PromotionL2256B, driver.TensorMapFloatOobFillNone); err != nil {
		return nil, fmt.Errorf("encode tensor map: %w", err)
	}
	return m, nil
}

// checkEgraphTarget reports whether plans for p can run on a device whose
// target is dev: NVRTC compiles them for the device, so p must name it.
func checkEgraphTarget(dev string, p cudadevicetable.Profile) error {
	if p.Target == dev {
		return nil
	}
	if hopper.UMMATarget(p.Target) && !hopper.UMMATarget(dev) {
		return fmt.Errorf("e-graph target %s (%s) lowers its gemms to tcgen05, which this %s device lacks; use -egraph-emit to compile for it", p.Target, p.Product, dev)
	}
	return fmt.Errorf("e-graph target %s (%s) is not this device's %s", p.Target, p.Product, dev)
}

// Kernels returns the number of generated launches per decode step.
func (eg *egraphDecoder) Kernels() int {
	n := len(eg.norm) + len(eg.attnIn)*(len(eg.attnIn[0])+len(eg.post[0])) + len(eg.head)
	if eg.plans.attn {
		n += len(eg.rope) + len(eg.sdpa) + len(eg.attnIn)*len(eg.combine)
	}
	return n
}

// Close releases eg's module and buffers.
func (eg *egraphDecoder) Close() error {
	var errs []error
	for i := range eg.bqkv {
		errs = append(errs, freeOwned(&eg.bqkv[i]))
	}
	for name, s := range eg.temps32 {
		errs = append(errs, freeOwned(&s))
		delete(eg.temps32, name)
	}
	for name, s := range eg.temps16 {
		errs = append(errs, freeOwned(&s))
		delete(eg.temps16, name)
	}
	for name, s := range eg.temps8 {
		errs = append(errs, freeOwned(&s))
		delete(eg.temps8, name)
	}
	for name, s := range eg.tempsI32 {
		errs = append(errs, freeOwned(&s))
		delete(eg.tempsI32, name)
	}
	for i := range eg.kc {
		errs = append(errs, freeOwned(&eg.kc[i]))
	}
	for i := range eg.vc {
		errs = append(errs, freeOwned(&eg.vc[i]))
	}
	errs = append(errs, freeOwned(&eg.ropePage), freeOwned(&eg.seqLen1), freeOwned(&eg.pages))
	errs = append(errs, freeOwned(&eg.cs), freeOwned(&eg.qbf), freeOwned(&eg.ws))
	if eg.pf != nil {
		errs = append(errs, eg.pf.free(), freeOwned(&eg.pfWS))
	}
	errs = append(errs, freeOwned(&eg.mmaWS))
	for p, r := range eg.ready {
		errs = append(errs, freeOwned(&r))
		delete(eg.ready, p)
	}
	errs = append(errs, freeOwned(&eg.qkv), freeOwned(&eg.x2), freeOwned(&eg.xn))
	if eg.mod != 0 {
		errs = append(errs, eg.mod.Close())
	}
	return errors.Join(errs...)
}

// emitStep is emitForwardStep with the e-graph kernels in place of
// llama-infer's norms, projections, SwiGLU and residual adds. RoPE, the
// KV-cache store and attention keep their kernels and argument order, so a
// DecodeGraph finds and updates their position arguments as before.
func (eg *egraphDecoder) emitStep(
	gpuW *GPUModelWeights,
	cfg *ModelConfig,
	layerKV func(int) (cudakv.LayerCache, error),
	sp *GPUScratchpad,
	gm *GPUModules,
	attnFn driver.Function,
	sampleFn driver.Function,
	tokenSrc cudamem.Slice[int32],
	pos int,
	attnShared uint32,
	emit func(l *cudalaunch.Launcher, args ...any) error,
) error {
	if err := checkPos(pos, cfg.MaxSeqLen); err != nil {
		return err
	}
	headDim := cfg.HeadDim()
	qHeads, kvHeads := cfg.NumAttentionHeads, cfg.NumKeyValueHeads
	q := eg.qkv.Ptr()
	k := q + driver.DevicePtr(4*qHeads*headDim)
	v := k + driver.DevicePtr(4*kvHeads*headDim)
	launch := func(ls []egraphLaunch) error {
		for _, l := range ls {
			if l.epoch >= 0 {
				eg.epoch++
				l.args[l.epoch] = eg.epoch
			}
			args := l.args
			if l.pos != nil {
				args = slices.Clone(l.args)
				setPos(args, l.args, l.pos, pos)
			}
			if err := emit(cudalaunch.Kernel(l.fn).Grid(l.grid[0], l.grid[1], l.grid[2]).Block(l.block[0], l.block[1], l.block[2]).SharedMem(l.smem), args...); err != nil {
				return err
			}
		}
		return nil
	}

	if err := emit(cudalaunch.Kernel(gm.embedBF16Fn).Grid(1, 1, 1).Block(256, 1, 1),
		gpuW.EmbedTokens, tokenSrc, sp.X, int32(cfg.HiddenSize)); err != nil {
		return err
	}
	if err := launch(eg.norm); err != nil {
		return err
	}
	attnCfg := cudaattn.DefaultConfig(qHeads, kvHeads, headDim)
	for i := range cfg.NumHiddenLayers {
		lkv, err := layerKV(i)
		if err != nil {
			return err
		}
		if eg.plans.attn {
			if err := launch(eg.layerLaunches(i)); err != nil {
				return err
			}
			continue
		}
		if err := launch(eg.attnIn[i]); err != nil {
			return err
		}
		if cfg.RoPEScaling.hasLlama3Params() {
			rs := cfg.RoPEScaling
			if err := emit(cudalaunch.Kernel(gm.ropeScaledFn).Grid(uint32(qHeads), 1, 1).Block(uint32(headDim/2), 1, 1),
				q, k, int32(pos), int32(qHeads), int32(kvHeads), int32(headDim),
				cfg.RopeTheta, rs.Factor, rs.LowFreqFactor, rs.HighFreqFactor, int32(rs.OriginalMaxPositionEmbeddings)); err != nil {
				return err
			}
		} else {
			if err := emit(cudalaunch.Kernel(gm.ropeFn).Grid(uint32(qHeads), 1, 1).Block(uint32(headDim/2), 1, 1),
				q, k, int32(pos), int32(qHeads), int32(kvHeads), int32(headDim), cfg.RopeTheta); err != nil {
				return err
			}
		}
		if err := emit(cudalaunch.Kernel(gm.storeKVFn).Grid(1, 1, 1).Block(128, 1, 1),
			k, v, lkv.K, lkv.V, int32(pos), int32(kvHeads), int32(headDim)); err != nil {
			return err
		}
		if err := emit(cudalaunch.Kernel(attnFn).Grid(uint32(qHeads), 1, 1).Block(attnThreads, 1, 1).SharedMem(attnShared),
			q, lkv.K, lkv.V, sp.AttnOut, int32(pos), int32(cfg.MaxSeqLen), int32(qHeads), int32(kvHeads), int32(headDim), attnCfg.Scale); err != nil {
			return err
		}
		if err := launch(eg.post[i]); err != nil {
			return err
		}
	}
	if err := launch(eg.head); err != nil {
		return err
	}
	return emit(cudalaunch.Kernel(sampleFn).Grid(1, 1, 1).Block(256, 1, 1),
		sp.Logits, sp.TokenOut, int32(cfg.VocabSize))
}
