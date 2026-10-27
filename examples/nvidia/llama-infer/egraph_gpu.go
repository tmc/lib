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

// withRows returns l over the first n rows of its plan's M: a tile gemm or
// row kernel takes the row count as its first argument and sizes its grid
// by it.
func (l egraphLaunch) withRows(n int) egraphLaunch {
	k := l.kern
	var grid [3]int
	switch {
	case k == nil || k.GEMM != nil:
		return l
	case k.Tile != nil:
		grid, _ = k.Tile.Grid(n, 1)
	case k.Row != nil:
		grid, _ = k.Row.Grid(n)
	}
	l.args = slices.Clone(l.args)
	l.args[0] = int32(n)
	l.grid = [3]uint32{uint32(grid[0]), uint32(grid[1]), uint32(grid[2])}
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
	ready map[*hopper.Plan]cudamem.Slice[uint64]
	epoch uint32

	wqkv, wgu []deviceWeight
	bqkv      []cudamem.Slice[float32]
	qkv, x2   cudamem.Slice[float32]
	temps32   []cudamem.Slice[float32]
	temps16   []cudamem.Slice[uint16]
	tempPtr   map[string]driver.DevicePtr

	attnIn, post [][]egraphLaunch // per layer
	head         []egraphLaunch

	// With plans.attn: the BF16 KV cache, the rotary cos|sin table, the
	// page tables and seqlens the emitted kernels read, q in BF16, and the
	// per-layer rotary and attention launches.
	kc, vc     []cudamem.Slice[uint16]
	cs         cudamem.Slice[float32]
	ints       []cudamem.Slice[int32]
	qbf        cudamem.Slice[uint16]
	rope, sdpa []egraphLaunch
	qcast      []egraphLaunch
	posKind    map[driver.Function]string
	posArgs    map[driver.Function][]egraphPosArg
	pageTable  driver.DevicePtr
	one        driver.DevicePtr

	pf *egraphPrefill // with plans.pf
}

func newEgraphDecoder(dev *cuda.Device, w *ModelWeights, cfg *ModelConfig, gpuW *GPUModelWeights, sp *GPUScratchpad, mega, attn bool, prefillRows int) (eg *egraphDecoder, err error) {
	if gpuW.DType != weightBF16 {
		return nil, errors.New("-egraph needs -weight-dtype bf16")
	}
	if prefillRows != 0 && !attn {
		return nil, errors.New("e-graph prefill needs generated attention")
	}
	bias := len(w.Layers) > 0 && len(w.Layers[0].QBias) > 0
	plans, err := buildEgraphPlans(cfg, bias, attn, prefillRows)
	if err != nil {
		return nil, fmt.Errorf("build e-graph plans: %w", err)
	}
	eg = &egraphDecoder{
		plans:   plans,
		megas:   make(map[*hopper.Plan]hopper.Megakernel),
		fns:     make(map[string]driver.Function),
		ready:   make(map[*hopper.Plan]cudamem.Slice[uint64]),
		tempPtr: make(map[string]driver.DevicePtr),
		posKind: make(map[driver.Function]string),
		posArgs: make(map[driver.Function][]egraphPosArg),
	}
	defer func() {
		if err != nil {
			err = errors.Join(err, eg.Close())
			eg = nil
		}
	}()
	if mega {
		for _, p := range eg.decodePlans() {
			m, err := p.Megakernel("")
			if err != nil {
				return nil, err
			}
			eg.megas[p] = m
			r, err := cudamem.Alloc[uint64](dev, m.ReadyWords())
			eg.ready[p] = r
			if err != nil {
				return nil, fmt.Errorf("alloc %s ready counters: %w", m.Name, err)
			}
			if err := r.CopyFromHost(make([]uint64, m.ReadyWords())); err != nil {
				return nil, fmt.Errorf("zero %s ready counters: %w", m.Name, err)
			}
		}
	}
	if err := eg.compile(dev); err != nil {
		return nil, err
	}

	for i := range w.Layers {
		l := &w.Layers[i]
		qkv, err := uploadWeight(dev, concat(l.WQ, l.WK, l.WV), weightBF16)
		eg.wqkv = append(eg.wqkv, qkv)
		if err != nil {
			return nil, fmt.Errorf("upload layer %d qkv: %w", i, err)
		}
		gu, err := uploadWeight(dev, concat(l.WGate, l.WUp), weightBF16)
		eg.wgu = append(eg.wgu, gu)
		if err != nil {
			return nil, fmt.Errorf("upload layer %d gate_up: %w", i, err)
		}
		if bias {
			b, err := cudamem.Alloc[float32](dev, len(l.QBias)+len(l.KBias)+len(l.VBias))
			eg.bqkv = append(eg.bqkv, b)
			if err != nil {
				return nil, fmt.Errorf("alloc layer %d qkv bias: %w", i, err)
			}
			if err := b.CopyFromHost(concat(l.QBias, l.KBias, l.VBias)); err != nil {
				return nil, fmt.Errorf("upload layer %d qkv bias: %w", i, err)
			}
		}
	}
	if eg.qkv, err = cudamem.Alloc[float32](dev, plans.attnIn.Output.Shape[1]); err != nil {
		return nil, fmt.Errorf("alloc qkv: %w", err)
	}
	if eg.x2, err = cudamem.Alloc[float32](dev, cfg.HiddenSize); err != nil {
		return nil, fmt.Errorf("alloc residual: %w", err)
	}
	for _, p := range eg.planList() {
		for _, t := range p.Temps {
			n := 1
			for _, d := range t.Shape {
				n *= d
			}
			switch t.DType {
			case hopper.F32:
				s, err := cudamem.Alloc[float32](dev, n)
				eg.temps32 = append(eg.temps32, s)
				if err != nil {
					return nil, fmt.Errorf("alloc %s: %w", t.Name, err)
				}
				eg.tempPtr[t.Name] = s.Ptr()
			case hopper.BF16:
				s, err := cudamem.Alloc[uint16](dev, n)
				eg.temps16 = append(eg.temps16, s)
				if err != nil {
					return nil, fmt.Errorf("alloc %s: %w", t.Name, err)
				}
				eg.tempPtr[t.Name] = s.Ptr()
			default:
				return nil, fmt.Errorf("temporary %s has dtype %v", t.Name, t.DType)
			}
		}
	}

	// The residual stream alternates between sp.X and x2, one layer each.
	cur, next := sp.X, eg.x2
	for i := range gpuW.Layers {
		lw := &gpuW.Layers[i]
		bufs := map[string]driver.DevicePtr{
			"x":                      cur.Ptr(),
			"attn_norm":              lw.InputNorm.Ptr(),
			"w_qkv":                  eg.wqkv[i].Ptr(),
			plans.attnIn.Output.Name: eg.qkv.Ptr(),
			"attn":                   sp.AttnOut.Ptr(),
			"w_o":                    lw.WO.Ptr(),
			"ffn_norm":               lw.PostNorm.Ptr(),
			"w_gu":                   eg.wgu[i].Ptr(),
			"w_down":                 lw.WDown.Ptr(),
			plans.post.Output.Name:   next.Ptr(),
		}
		if bias {
			bufs["b_qkv"] = eg.bqkv[i].Ptr()
		}
		in, err := eg.bind(plans.attnIn, bufs)
		if err != nil {
			return nil, err
		}
		post, err := eg.bind(plans.post, bufs)
		if err != nil {
			return nil, err
		}
		eg.attnIn = append(eg.attnIn, in)
		eg.post = append(eg.post, post)
		cur, next = next, cur
	}
	eg.head, err = eg.bind(plans.head, map[string]driver.DevicePtr{
		"x":                    cur.Ptr(),
		"norm":                 gpuW.Norm.Ptr(),
		"lm_head":              gpuW.LMHead.Ptr(),
		plans.head.Output.Name: sp.Logits.Ptr(),
	})
	if err != nil {
		return nil, err
	}
	if attn {
		if err := eg.bindAttention(dev, cfg, sp); err != nil {
			return nil, err
		}
	}
	if plans.pf != nil {
		if eg.pf, err = eg.bindPrefill(dev, cfg, gpuW); err != nil {
			return nil, err
		}
	}
	return eg, nil
}

// bindAttention allocates the BF16 KV cache and the tables the emitted
// rotary and attention kernels read, and binds their launches.
func (eg *egraphDecoder) bindAttention(dev *cuda.Device, cfg *ModelConfig, sp *GPUScratchpad) error {
	hd := cfg.HeadDim()
	kvDim := cfg.NumKeyValueHeads * hd
	maxSeq := cfg.MaxSeqLen
	slots := maxSeq
	if eg.plans.pf != nil {
		slots = eg.plans.pf.cacheSlots
	}
	for i := range cfg.NumHiddenLayers {
		for _, c := range []*[]cudamem.Slice[uint16]{&eg.kc, &eg.vc} {
			s, err := cudamem.Alloc[uint16](dev, slots*kvDim)
			*c = append(*c, s)
			if err != nil {
				return fmt.Errorf("alloc layer %d kv cache: %w", i, err)
			}
			// Prefill attention loads, and masks, cache rows past a
			// chunk's last token; a NaN there would still poison the
			// output (0·NaN), so start from zeros.
			if err := s.CopyFromHost(make([]uint16, slots*kvDim)); err != nil {
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
	var err error
	if eg.cs, err = cudamem.Alloc[float32](dev, len(table)); err != nil {
		return fmt.Errorf("alloc rope table: %w", err)
	}
	if err := eg.cs.CopyFromHost(table); err != nil {
		return fmt.Errorf("upload rope table: %w", err)
	}
	ints := func(v []int32) (driver.DevicePtr, error) {
		s, err := cudamem.Alloc[int32](dev, len(v))
		eg.ints = append(eg.ints, s)
		if err != nil {
			return 0, err
		}
		return s.Ptr(), s.CopyFromHost(v)
	}
	// The rotary kernel writes cache slot page_table[0]·1 + (seqlens[0]-1)
	// = 0 of the pointers it is given, which the decoder offsets to the
	// token's row. The attention kernel reads page i at row 16i.
	zero, err := ints([]int32{0})
	if err != nil {
		return fmt.Errorf("alloc rope page table: %w", err)
	}
	one, err := ints([]int32{1})
	if err != nil {
		return fmt.Errorf("alloc seqlens: %w", err)
	}
	pages := make([]int32, eg.plans.sdpa.MaxPages)
	for i := range pages {
		pages[i] = int32(i)
	}
	pageTable, err := ints(pages)
	if err != nil {
		return fmt.Errorf("alloc attention page table: %w", err)
	}
	eg.pageTable, eg.one = pageTable, one
	if eg.qbf, err = cudamem.Alloc[uint16](dev, cfg.NumAttentionHeads*hd); err != nil {
		return fmt.Errorf("alloc q: %w", err)
	}
	if eg.qcast, err = eg.bind(eg.plans.qcast, map[string]driver.DevicePtr{
		"q":                        eg.qbf.Ptr(),
		eg.plans.qcast.Output.Name: sp.Q.Ptr(),
	}); err != nil {
		return err
	}

	ropeFn, sdpaFn := eg.fns[eg.plans.rope.Name], eg.fns[eg.plans.sdpa.Name]
	row := driver.DevicePtr(2 * kvDim)
	eg.posKind[ropeFn], eg.posKind[sdpaFn] = "rope_kv", "attn"
	eg.posArgs[ropeFn] = []egraphPosArg{{idx: 1, stride: driver.DevicePtr(4 * hd)}, {idx: 8, stride: row}, {idx: 9, stride: row}}
	eg.posArgs[sdpaFn] = []egraphPosArg{{idx: 7}}
	for i := range cfg.NumHiddenLayers {
		// qkv, cos_sin_table, positions, seq_ids, seqlens, kv_positions,
		// page_table, q_out, k_cache, v_cache
		eg.rope = append(eg.rope, egraphLaunch{
			fn: ropeFn, grid: [3]uint32{1, 1, 1}, block: [3]uint32{128, 1, 1}, epoch: -1,
			args: []any{eg.qkv.Ptr(), eg.cs.Ptr(), driver.DevicePtr(0), driver.DevicePtr(0), one, driver.DevicePtr(0),
				zero, eg.qbf.Ptr(), eg.kc[i].Ptr(), eg.vc[i].Ptr()},
			pos: eg.posArgs[ropeFn],
		})
		// q, k_pages, v_pages, k_scales, v_scales, page_table, out, seq_len
		g := eg.plans.sdpa.Grid(1)
		b := eg.plans.sdpa.Block()
		eg.sdpa = append(eg.sdpa, egraphLaunch{
			fn: sdpaFn, grid: [3]uint32{uint32(g[0]), uint32(g[1]), uint32(g[2])}, block: [3]uint32{uint32(b[0]), uint32(b[1]), uint32(b[2])}, epoch: -1,
			args: []any{sp.Q.Ptr(), eg.kc[i].Ptr(), eg.vc[i].Ptr(), driver.DevicePtr(0), driver.DevicePtr(0), pageTable, sp.AttnOut.Ptr(), int32(1)},
			pos:  eg.posArgs[sdpaFn],
		})
	}
	return nil
}

// positionArgs reports how a DecodeGraph must update a node running fn
// when the position changes: its kind and its position-dependent arguments,
// or ok false when fn's arguments do not depend on the position.
func (eg *egraphDecoder) positionArgs(fn driver.Function) (kind string, pa []egraphPosArg, ok bool) {
	pa, ok = eg.posArgs[fn]
	return eg.posKind[fn], pa, ok
}

func concat(parts ...[]float32) []float32 {
	n := 0
	for _, p := range parts {
		n += len(p)
	}
	out := make([]float32, 0, n)
	for _, p := range parts {
		out = append(out, p...)
	}
	return out
}

// decodePlans returns the decode step's plans, which -egraph-mega runs as
// megakernels.
func (eg *egraphDecoder) decodePlans() []*hopper.Plan {
	ps := []*hopper.Plan{eg.plans.attnIn, eg.plans.post, eg.plans.head}
	if eg.plans.attn {
		ps = append(ps, eg.plans.qcast)
	}
	return ps
}

func (eg *egraphDecoder) planList() []*hopper.Plan {
	ps := eg.decodePlans()
	if pf := eg.plans.pf; pf != nil {
		ps = append(ps, pf.attnIn, pf.post)
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
	case k.Row != nil:
		return k.Row.Name, nil
	}
	return "", fmt.Errorf("plan kernel %+v: only gemm and row kernels are supported", k)
}

func (eg *egraphDecoder) compile(dev *cuda.Device) error {
	var src strings.Builder
	src.WriteString(hopper.Prelude)
	var names []string
	for _, p := range eg.planList() {
		if m, ok := eg.megas[p]; ok {
			cu, err := m.CUDA()
			if err != nil {
				return fmt.Errorf("emit %s: %w", m.Name, err)
			}
			src.WriteString("\n" + cu + "\n")
			names = append(names, m.Name)
			continue
		}
		for _, k := range p.Kernels {
			name, err := kernelName(k)
			if err != nil {
				return err
			}
			var cu string
			switch {
			case k.GEMM != nil:
				cu, err = k.GEMM.CUDA()
			case k.Tile != nil:
				cu, err = k.Tile.CUDA()
			default:
				cu, err = k.Row.CUDA()
			}
			if err != nil {
				return fmt.Errorf("emit %s: %w", name, err)
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
				return fmt.Errorf("emit attention: %w", err)
			}
			src.WriteString("\n" + cu + "\n")
		}
		names = append(names, eg.plans.rope.Name, eg.plans.sdpa.Name)
	}
	if pf := eg.plans.pf; pf != nil {
		for _, k := range []interface {
			CUDA() (string, error)
		}{pf.rope, pf.attn} {
			cu, err := k.CUDA()
			if err != nil {
				return fmt.Errorf("emit prefill attention: %w", err)
			}
			src.WriteString("\n" + cu + "\n")
		}
		src.WriteString("\n" + embedGatherCUDA + "\n")
		names = append(names, pf.rope.Name, pf.attn.Name, embedGatherName)
	}
	cm, err := nvrtcx.Compile(src.String(),
		nvrtcx.WithName("llama_egraph.cu"),
		nvrtcx.WithDeviceArch(dev),
		nvrtcx.WithStandard("c++20"),
		nvrtcx.WithNameExpressions(names...))
	if err != nil {
		return fmt.Errorf("compile e-graph kernels: %w", releaseErrorOwner(err))
	}
	if eg.mod, err = cm.Load(dev); err != nil {
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
	}
	return nil
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
		switch {
		case k.GEMM != nil:
			if grid, block, err = k.GEMM.Grid(); err != nil {
				return nil, fmt.Errorf("%s: %w", name, err)
			}
		case k.Tile != nil:
			grid, block = k.Tile.Grid(k.M, 1)
			l.smem = uint32(k.Tile.SharedBytes())
			l.args = append(l.args, int32(k.M))
		default:
			grid, block = k.Row.Grid(k.Rows)
			l.args = append(l.args, int32(k.Rows))
		}
		for i := range 3 {
			l.grid[i], l.block[i] = uint32(grid[i]), uint32(block[i])
		}
		for _, a := range k.Args {
			ptr, err := lookup(name, a)
			if err != nil {
				return nil, err
			}
			l.args = append(l.args, ptr)
		}
		out = append(out, l)
	}
	return out, nil
}

// Kernels returns the number of generated launches per decode step.
func (eg *egraphDecoder) Kernels() int {
	n := len(eg.attnIn)*(len(eg.attnIn[0])+len(eg.post[0])) + len(eg.head)
	if eg.plans.attn {
		n += len(eg.rope) + len(eg.sdpa) + len(eg.attnIn)*len(eg.qcast)
	}
	return n
}

// Close releases eg's module and buffers.
func (eg *egraphDecoder) Close() error {
	var errs []error
	for i := range eg.wqkv {
		errs = append(errs, eg.wqkv[i].release())
	}
	for i := range eg.wgu {
		errs = append(errs, eg.wgu[i].release())
	}
	for i := range eg.bqkv {
		errs = append(errs, freeOwned(&eg.bqkv[i]))
	}
	for i := range eg.temps32 {
		errs = append(errs, freeOwned(&eg.temps32[i]))
	}
	for i := range eg.temps16 {
		errs = append(errs, freeOwned(&eg.temps16[i]))
	}
	for i := range eg.kc {
		errs = append(errs, freeOwned(&eg.kc[i]))
	}
	for i := range eg.vc {
		errs = append(errs, freeOwned(&eg.vc[i]))
	}
	for i := range eg.ints {
		errs = append(errs, freeOwned(&eg.ints[i]))
	}
	errs = append(errs, freeOwned(&eg.cs), freeOwned(&eg.qbf))
	if eg.pf != nil {
		errs = append(errs, eg.pf.free())
	}
	for p, r := range eg.ready {
		errs = append(errs, freeOwned(&r))
		delete(eg.ready, p)
	}
	errs = append(errs, freeOwned(&eg.qkv), freeOwned(&eg.x2))
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
	attnCfg := cudaattn.DefaultConfig(qHeads, kvHeads, headDim)
	for i := range cfg.NumHiddenLayers {
		lkv, err := layerKV(i)
		if err != nil {
			return err
		}
		if err := launch(eg.attnIn[i]); err != nil {
			return err
		}
		if eg.plans.attn {
			if err := launch([]egraphLaunch{eg.rope[i]}); err != nil {
				return err
			}
			if err := launch(eg.qcast); err != nil {
				return err
			}
			if err := launch([]egraphLaunch{eg.sdpa[i]}); err != nil {
				return err
			}
			if err := launch(eg.post[i]); err != nil {
				return err
			}
			continue
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
