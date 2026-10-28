package main

import (
	"errors"
	"fmt"
	"math"
	"strings"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/egraph/hopper"
	"github.com/tmc/lib/nvidia/x/egraph/tensorparallel"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
	"github.com/tmc/lib/nvidia/x/rankgroup"
)

// A tpDecoder runs the e-graph decode step tensor-parallel over the ranks
// of a [rankgroup.Group], with the layer sharded by [tensorparallel.Shard]:
// each rank holds heads/TP query and KV heads and intermediate/TP MLP
// columns of every layer, and its own BF16 KV cache for its KV heads.
//
// Per token and layer each rank runs its attn_in plan, the generated rotary
// and attention kernels over its heads, and its o_proj plan; an all-reduce
// sums the o_proj partials into h; each rank runs its MLP plan; an
// all-reduce sums the MLP partials into the next layer's x. Rank 0 adds the
// residuals and, after the last layer, runs the head and argmax. Prompts
// prefill token by token through the same step.
//
// With fuse, the row-parallel GEMMs add their partials into one h and one x
// that every rank reads (red.global.add), which needs every rank on one
// device; otherwise an [rankgroup.AllReducer] sums per-rank buffers.
type tpDecoder struct {
	g      *rankgroup.Group
	ar     rankgroup.AllReducer // nil when fused
	plan   *tensorparallel.Plan
	head   *hopper.Plan
	rope   hopper.QKNormRoPEKVAppend
	qcast  *hopper.Plan
	sdpa   hopper.SM120PagedAttention
	fuse   bool
	hidden int
	vocab  int

	mods  map[*cuda.Device]driver.Module
	fns   []map[string]driver.Function // by rank: kernel name -> function
	ranks []*tpRank
	frees []func() error

	// Rank 0's head: the final norm, lm_head, logits and argmax.
	headLaunch []egraphLaunch
	sample     driver.Function
	logits     driver.DevicePtr
	tokenOut   cudamem.Slice[int32]
	hostOut    []int32
}

// A tpRank is one rank's weights, cache and buffers, and its launches
// bound to them.
type tpRank struct {
	embed            driver.DevicePtr
	token            driver.DevicePtr
	x                [2]driver.DevicePtr // the residual stream, alternating by layer
	h                driver.DevicePtr
	partO, partF     driver.DevicePtr // unfused all-reduce inputs
	attnIn, oproj    [][]egraphLaunch // per layer
	rope, sdpa, post [][]egraphLaunch // per layer; post is the MLP
	qcast, combine   []egraphLaunch
}

// tpOptions configure newTPDecoder.
type tpOptions struct {
	TP      int
	Mode    rankgroup.Mode
	Fuse    bool
	Backend string // "device" or "nccl"
	Splits  int    // attention splits per KV head
}

func newTPDecoder(dev *cuda.Device, w *ModelWeights, cfg *ModelConfig, gpuW *GPUModelWeights, sp *GPUScratchpad, sampleFn driver.Function, opt tpOptions) (d *tpDecoder, err error) {
	if gpuW.DType != weightBF16 {
		return nil, errors.New("-tp needs -weight-dtype bf16")
	}
	bias := len(w.Layers) > 0 && len(w.Layers[0].QBias) > 0
	d = &tpDecoder{fuse: opt.Fuse, hidden: cfg.HiddenSize, vocab: cfg.VocabSize, mods: make(map[*cuda.Device]driver.Module),
		sample: sampleFn, logits: sp.Logits.Ptr(), tokenOut: sp.TokenOut, hostOut: make([]int32, 1)}
	defer func() {
		if err != nil {
			err = errors.Join(err, d.Close())
			d = nil
		}
	}()
	switch opt.Backend {
	case "device":
		d.g, err = rankgroup.OnDevice(dev, opt.TP, opt.Mode)
	case "nccl":
		if opt.Fuse {
			return nil, errors.New("-tp-fuse needs -tp-backend device: nccl ranks do not share an output buffer")
		}
		var devs []*cuda.Device
		devs, err = tpDevices(dev, opt.TP)
		if err == nil {
			d.g, err = rankgroup.OnDevices(devs)
		}
	default:
		return nil, fmt.Errorf("unknown -tp-backend %q (want device or nccl)", opt.Backend)
	}
	if err != nil {
		return nil, err
	}
	if opt.Fuse && !d.g.SingleDevice() {
		return nil, errors.New("-tp-fuse needs every rank on one device")
	}
	fusion := tensorparallel.FuseNone
	if opt.Fuse {
		fusion = tensorparallel.FuseReduceAdd
	}
	hd := cfg.HeadDim()
	if d.plan, err = tensorparallel.Shard(tensorparallel.Layer{
		Hidden: cfg.HiddenSize, Heads: cfg.NumAttentionHeads, KVHeads: cfg.NumKeyValueHeads, HeadDim: hd,
		Intermediate: cfg.IntermediateSize, Eps: cfg.RmsNormEps, QKVBias: bias,
		QKVOut: hopper.BF16, AttnDType: hopper.F32,
	}, tensorparallel.Options{TP: opt.TP, Fusion: fusion}); err != nil {
		return nil, err
	}
	b := hopper.NewBuilder()
	x := b.Input("x", hopper.F32, 1, cfg.HiddenSize)
	norm := b.Cast(hopper.BF16, b.RMSNorm(x, b.Input("norm", hopper.F32, cfg.HiddenSize), cfg.HiddenSize, cfg.RmsNormEps))
	if d.head, err = compilePlan("tp_head", b, b.MatMul(norm, b.Input("lm_head", hopper.BF16, cfg.VocabSize, cfg.HiddenSize)), hopper.F32); err != nil {
		return nil, err
	}
	local := d.plan.Local
	b = hopper.NewBuilder()
	if d.qcast, err = compilePlan("tp_qcast", b, b.Cast(hopper.F32, b.Input("q", hopper.BF16, 1, local.Heads*hd)), hopper.F32); err != nil {
		return nil, err
	}
	d.rope = hopper.QKNormRoPEKVAppend{Name: "tp_rope_kv", NumHeads: local.Heads, KVHeads: local.KVHeads, HeadDim: hd, PageSize: 1, MaxPages: 1, DType: hopper.BF16}
	d.sdpa = hopper.SM120PagedAttention{
		Name: "tp_sdpa", NumQHeads: local.Heads, NumKVHeads: local.KVHeads, HeadDim: hd,
		PageSize: egraphPageSize, MaxPages: (cfg.MaxSeqLen + egraphPageSize - 1) / egraphPageSize, SeqLen: cfg.MaxSeqLen,
		Target: "sm_121a", KVFormat: hopper.WeightBF16, Splits: opt.Splits,
	}
	err = d.g.Do(func() error {
		if err := d.compile(); err != nil {
			return err
		}
		if !opt.Fuse {
			switch opt.Backend {
			case "device":
				d.ar, err = rankgroup.NewDeviceAllReduce(d.g)
			case "nccl":
				d.ar, err = rankgroup.NewNCCLAllReduce(d.g)
			}
			if err != nil {
				return err
			}
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	if err := d.bind(dev, w, cfg, gpuW, bias); err != nil {
		return nil, err
	}
	return d, nil
}

// tpDevices returns dev and the next tp-1 devices, one per rank.
func tpDevices(dev *cuda.Device, tp int) ([]*cuda.Device, error) {
	n, err := cuda.DeviceCount()
	if err != nil {
		return nil, err
	}
	if n < tp {
		return nil, fmt.Errorf("-tp-backend nccl needs %d gpus, %d visible", tp, n)
	}
	devs := []*cuda.Device{dev}
	for i := 1; i < tp; i++ {
		d, err := cuda.NewDevice((dev.Ordinal() + i) % n)
		if err != nil {
			return nil, err
		}
		devs = append(devs, d)
	}
	return devs, nil
}

// plans returns the decoder's distinct plans.
func (d *tpDecoder) plans() []*hopper.Plan {
	ps := []*hopper.Plan{d.plan.Ranks[0].AttnIn, d.plan.Ranks[0].OProj, d.plan.Ranks[0].FFN, d.qcast, d.head}
	if d.plan.TP > 1 {
		ps = append(ps, d.plan.Ranks[1].OProj, d.plan.Ranks[1].FFN)
	}
	return ps
}

// compile compiles every kernel once per device. It runs inside g.Do.
func (d *tpDecoder) compile() error {
	var src strings.Builder
	src.WriteString(hopper.Prelude)
	var names []string
	for _, p := range d.plans() {
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
	for _, k := range []interface{ CUDA() (string, error) }{d.rope, d.sdpa} {
		cu, err := k.CUDA()
		if err != nil {
			return fmt.Errorf("emit attention: %w", err)
		}
		src.WriteString("\n" + cu + "\n")
	}
	src.WriteString("\n" + embedGatherCUDA + "\n")
	names = append(names, d.rope.Name, d.sdpa.Name, embedGatherName)
	if d.sdpa.Splits > 0 {
		names = append(names, d.sdpa.Name+"_combine")
	}
	for r := range d.g.Size() {
		dev := d.g.Device(r)
		fns := make(map[string]driver.Function)
		if _, ok := d.mods[dev]; !ok {
			cm, err := nvrtcx.Compile(src.String(), nvrtcx.WithName("llama_tp.cu"), nvrtcx.WithDeviceArch(dev),
				nvrtcx.WithStandard("c++20"), nvrtcx.WithNameExpressions(names...))
			if err != nil {
				return fmt.Errorf("compile tp kernels: %w", releaseErrorOwner(err))
			}
			mod, err := cm.Load(dev)
			if err != nil {
				return fmt.Errorf("load tp kernels: %w", releaseErrorOwner(err))
			}
			d.mods[dev] = mod
			for _, name := range names {
				lowered, ok := cm.LoweredName(name)
				if !ok {
					return fmt.Errorf("tp kernel %s: no lowered name", name)
				}
				fn, err := mod.Function(lowered)
				if err != nil {
					return fmt.Errorf("lookup tp kernel %s: %w", name, err)
				}
				fns[name] = fn
			}
		} else {
			fns = d.fns[r-1]
		}
		d.fns = append(d.fns, fns)
	}
	return nil
}

// tpAlloc allocates n elements of T on dev, copies init into them unless
// it is nil, and records the free.
func tpAlloc[T any](d *tpDecoder, dev *cuda.Device, n int, init []T) (driver.DevicePtr, error) {
	var p driver.DevicePtr
	err := dev.WithContext(func() error {
		var err error
		if p, err = driver.MemAlloc[T](n); err != nil {
			return fmt.Errorf("alloc %d values: %w", n, err)
		}
		d.frees = append(d.frees, func() error { return dev.WithContext(p.Free) })
		if init != nil {
			return p.CopyFromHost(init)
		}
		return nil
	})
	return p, err
}

// upload converts host to BF16 on dev.
func (d *tpDecoder) upload(dev *cuda.Device, host []float32) (driver.DevicePtr, error) {
	w, err := uploadWeight(dev, host, weightBF16)
	if err != nil {
		return 0, err
	}
	d.frees = append(d.frees, w.release)
	return w.Ptr(), nil
}

func (d *tpDecoder) f32(dev *cuda.Device, host []float32) (driver.DevicePtr, error) {
	return tpAlloc(d, dev, len(host), host)
}

// bind uploads every rank's shards and binds its launches.
func (d *tpDecoder) bind(dev0 *cuda.Device, w *ModelWeights, cfg *ModelConfig, gpuW *GPUModelWeights, bias bool) error {
	hd := cfg.HeadDim()
	h := cfg.HiddenSize
	p := d.plan
	local := p.Local
	maxSeq := cfg.MaxSeqLen
	// Shared on one device: rope table, tables, and, fused, x and h.
	type devState struct {
		cs, ropePage, seqLen1, pages driver.DevicePtr
		embed                        driver.DevicePtr
		x                            [2]driver.DevicePtr
		h                            driver.DevicePtr
	}
	perDev := make(map[*cuda.Device]*devState)
	half := hd / 2
	table := make([]float32, maxSeq*hd)
	for pos := range maxSeq {
		for i := range half {
			a := float64(pos) * float64(ropeInvFreq(i, hd, cfg.RopeTheta, cfg.RoPEScaling))
			table[pos*hd+i] = float32(math.Cos(a))
			table[pos*hd+half+i] = float32(math.Sin(a))
		}
	}
	pageIDs := make([]int32, d.sdpa.MaxPages)
	for i := range pageIDs {
		pageIDs[i] = int32(i)
	}
	for r := range p.TP {
		dev := d.g.Device(r)
		ds := perDev[dev]
		if ds == nil {
			ds = new(devState)
			perDev[dev] = ds
			var err error
			if ds.cs, err = d.f32(dev, table); err != nil {
				return err
			}
			for _, t := range []struct {
				p *driver.DevicePtr
				v []int32
			}{{&ds.ropePage, []int32{0}}, {&ds.seqLen1, []int32{1}}, {&ds.pages, pageIDs}} {
				if *t.p, err = tpAlloc(d, dev, len(t.v), t.v); err != nil {
					return err
				}
			}
			if dev == dev0 {
				ds.embed = gpuW.EmbedTokens.Ptr()
			} else if ds.embed, err = d.upload(dev, w.EmbedTokens); err != nil {
				return fmt.Errorf("upload embedding: %w", err)
			}
			for i := range ds.x {
				if ds.x[i], err = tpAlloc[float32](d, dev, h, nil); err != nil {
					return err
				}
			}
			if ds.h, err = tpAlloc[float32](d, dev, h, nil); err != nil {
				return err
			}
		}
		rk := &tpRank{embed: ds.embed}
		d.ranks = append(d.ranks, rk)
		var err error
		if rk.token, err = tpAlloc[int32](d, dev, 1, nil); err != nil {
			return err
		}
		f32 := func(n int) (driver.DevicePtr, error) { return tpAlloc[float32](d, dev, n, nil) }
		if d.fuse {
			rk.x, rk.h = ds.x, ds.h
		} else {
			for i := range rk.x {
				if rk.x[i], err = f32(h); err != nil {
					return err
				}
			}
			if rk.h, err = f32(h); err != nil {
				return err
			}
			if rk.partO, err = f32(h); err != nil {
				return err
			}
			if rk.partF, err = f32(h); err != nil {
				return err
			}
		}
		rp := p.Ranks[r]
		qkvLen := (local.Heads + 2*local.KVHeads) * hd
		qkv, err := tpAlloc[uint16](d, dev, qkvLen, nil)
		if err != nil {
			return err
		}
		qbf, err := tpAlloc[uint16](d, dev, local.Heads*hd, nil)
		if err != nil {
			return err
		}
		qf, err := f32(local.Heads * hd)
		if err != nil {
			return err
		}
		attnOut, err := f32(local.Heads * hd)
		if err != nil {
			return err
		}
		var ws driver.DevicePtr
		if n := d.sdpa.WorkspaceFloats(1); n > 0 {
			if ws, err = f32(n); err != nil {
				return err
			}
		}
		temps, err := d.temps(dev, rp.AttnIn, rp.OProj, rp.FFN, d.qcast)
		if err != nil {
			return err
		}
		if rk.qcast, err = d.bindPlan(r, d.qcast, temps, map[string]driver.DevicePtr{"q": qbf, d.qcast.Output.Name: qf}); err != nil {
			return err
		}
		ropeFn, sdpaFn := d.fns[r][d.rope.Name], d.fns[r][d.sdpa.Name]
		row := driver.DevicePtr(2 * local.KVHeads * hd)
		ropePos := []egraphPosArg{{idx: 1, stride: driver.DevicePtr(4 * hd)}, {idx: 8, stride: row}, {idx: 9, stride: row}}
		sdpaPos := []egraphPosArg{{idx: 7}}
		g, bl := d.sdpa.Grid(1), d.sdpa.Block()
		out := attnOut
		if ws != 0 {
			out = ws
			cg, cb := d.sdpa.CombineGrid(1)
			rk.combine = []egraphLaunch{{fn: d.fns[r][d.sdpa.Name+"_combine"], epoch: -1,
				grid: [3]uint32{uint32(cg[0]), uint32(cg[1]), uint32(cg[2])}, block: [3]uint32{uint32(cb[0]), uint32(cb[1]), uint32(cb[2])},
				args: []any{ws, attnOut}}}
		}
		for i := range cfg.NumHiddenLayers {
			l := &w.Layers[i]
			kc, err := tpAlloc[uint16](d, dev, maxSeq*local.KVHeads*hd, make([]uint16, maxSeq*local.KVHeads*hd))
			if err != nil {
				return fmt.Errorf("alloc rank %d layer %d k cache: %w", r, i, err)
			}
			vc, err := tpAlloc[uint16](d, dev, maxSeq*local.KVHeads*hd, make([]uint16, maxSeq*local.KVHeads*hd))
			if err != nil {
				return fmt.Errorf("alloc rank %d layer %d v cache: %w", r, i, err)
			}
			bufs := map[string]driver.DevicePtr{
				"x": rk.x[i%2], "h": rk.h, "attn": attnOut,
				rp.AttnIn.Output.Name: qkv,
			}
			if bufs["w_qkv"], err = d.upload(dev, tensorparallel.QKV(rp, l.WQ, l.WK, l.WV, h)); err != nil {
				return err
			}
			if bias {
				if bufs["b_qkv"], err = d.f32(dev, tensorparallel.QKV(rp, l.QBias, l.KBias, l.VBias, 1)); err != nil {
					return err
				}
			}
			if bufs["w_o"], err = d.upload(dev, tensorparallel.O(rp, l.WO)); err != nil {
				return err
			}
			if bufs["w_gu"], err = d.upload(dev, tensorparallel.GateUp(rp, l.WGate, l.WUp)); err != nil {
				return err
			}
			if bufs["w_down"], err = d.upload(dev, tensorparallel.Down(rp, l.WDown)); err != nil {
				return err
			}
			if dev == dev0 {
				bufs["attn_norm"], bufs["ffn_norm"] = gpuW.Layers[i].InputNorm.Ptr(), gpuW.Layers[i].PostNorm.Ptr()
			} else {
				if bufs["attn_norm"], err = d.f32(dev, l.InputNorm); err != nil {
					return err
				}
				if bufs["ffn_norm"], err = d.f32(dev, l.PostNorm); err != nil {
					return err
				}
			}
			// Fused, the row-parallel GEMMs add into h and the next x;
			// otherwise into the all-reduce inputs.
			if d.fuse {
				bufs[rp.OProj.Output.Name], bufs[rp.FFN.Output.Name] = rk.h, rk.x[(i+1)%2]
			} else {
				bufs[rp.OProj.Output.Name], bufs[rp.FFN.Output.Name] = rk.partO, rk.partF
			}
			in, err := d.bindPlan(r, rp.AttnIn, temps, bufs)
			if err != nil {
				return err
			}
			o, err := d.bindPlan(r, rp.OProj, temps, bufs)
			if err != nil {
				return err
			}
			post, err := d.bindPlan(r, rp.FFN, temps, bufs)
			if err != nil {
				return err
			}
			rk.attnIn, rk.oproj, rk.post = append(rk.attnIn, in), append(rk.oproj, o), append(rk.post, post)
			// qkv, cos_sin_table, positions, seq_ids, seqlens, kv_positions,
			// page_table, q_out, k_cache, v_cache
			rk.rope = append(rk.rope, []egraphLaunch{{fn: ropeFn, grid: [3]uint32{1, 1, 1}, block: [3]uint32{128, 1, 1}, epoch: -1,
				args: []any{qkv, ds.cs, driver.DevicePtr(0), driver.DevicePtr(0), ds.seqLen1, driver.DevicePtr(0),
					ds.ropePage, qbf, kc, vc}, pos: ropePos}})
			// q, k_pages, v_pages, k_scales, v_scales, page_table, out, seq_len
			rk.sdpa = append(rk.sdpa, []egraphLaunch{{fn: sdpaFn, epoch: -1,
				grid: [3]uint32{uint32(g[0]), uint32(g[1]), uint32(g[2])}, block: [3]uint32{uint32(bl[0]), uint32(bl[1]), uint32(bl[2])},
				args: []any{qf, kc, vc, driver.DevicePtr(0), driver.DevicePtr(0), ds.pages, out, int32(1)}, pos: sdpaPos}})
		}
		if r == 0 {
			headTemps, err := d.temps(dev, d.head)
			if err != nil {
				return err
			}
			if d.headLaunch, err = d.bindPlan(0, d.head, headTemps, map[string]driver.DevicePtr{
				"x": rk.x[cfg.NumHiddenLayers%2], "norm": gpuW.Norm.Ptr(), "lm_head": gpuW.LMHead.Ptr(), d.head.Output.Name: d.logits,
			}); err != nil {
				return err
			}
		}
	}
	return nil
}

// temps allocates the temporaries of plans on dev.
func (d *tpDecoder) temps(dev *cuda.Device, plans ...*hopper.Plan) (map[string]driver.DevicePtr, error) {
	temps := make(map[string]driver.DevicePtr)
	for _, pl := range plans {
		for _, t := range pl.Temps {
			n := 1
			for _, s := range t.Shape {
				n *= s
			}
			var ptr driver.DevicePtr
			switch t.DType {
			case hopper.F32:
				var err error
				if ptr, err = tpAlloc[float32](d, dev, n, nil); err != nil {
					return nil, err
				}
			case hopper.BF16:
				var err error
				if ptr, err = tpAlloc[uint16](d, dev, n, nil); err != nil {
					return nil, err
				}
			default:
				return nil, fmt.Errorf("temporary %s has dtype %v", t.Name, t.DType)
			}
			temps[t.Name] = ptr
		}
	}
	return temps, nil
}

// bindPlan binds p's launches for rank r.
func (d *tpDecoder) bindPlan(r int, p *hopper.Plan, temps, bufs map[string]driver.DevicePtr) ([]egraphLaunch, error) {
	var out []egraphLaunch
	for i := range p.Kernels {
		k := &p.Kernels[i]
		name, err := kernelName(*k)
		if err != nil {
			return nil, err
		}
		l := egraphLaunch{fn: d.fns[r][name], epoch: -1, kern: k}
		var grid, block [3]int
		switch {
		case k.GEMM != nil:
			if grid, block, err = k.GEMM.Grid(); err != nil {
				return nil, fmt.Errorf("%s: %w", name, err)
			}
			if k.GEMM.WorkspaceBytes() > 0 {
				return nil, fmt.Errorf("%s: split-k workspace unsupported", name)
			}
		case k.Tile != nil:
			return nil, fmt.Errorf("%s: tile gemm in a decode plan", name)
		default:
			grid, block = k.Row.Grid(k.Rows)
			l.args = append(l.args, int32(k.Rows))
		}
		for i := range 3 {
			l.grid[i], l.block[i] = uint32(grid[i]), uint32(block[i])
		}
		for _, a := range k.Args {
			ptr, ok := bufs[a]
			if !ok {
				ptr, ok = temps[a]
			}
			if !ok || ptr == 0 {
				return nil, fmt.Errorf("%s: unbound buffer %s", name, a)
			}
			l.args = append(l.args, ptr)
		}
		out = append(out, l)
	}
	return out, nil
}

// launch enqueues ls on rank r at position pos.
func (d *tpDecoder) launch(r int, ls []egraphLaunch, pos int) error {
	for _, l := range ls {
		args := l.args
		if l.pos != nil {
			args = append([]any(nil), l.args...)
			setPos(args, l.args, l.pos, pos)
		}
		if err := d.g.Launch(r, l.fn, l.grid, l.block, l.smem, args...); err != nil {
			return err
		}
	}
	return nil
}

// step runs token tok at position pos through every rank and returns the
// argmax of the logits.
func (d *tpDecoder) step(tok int32, pos int, maxSeq int) (int32, error) {
	if err := checkToken(tok, d.vocab); err != nil {
		return 0, err
	}
	if err := checkPos(pos, maxSeq); err != nil {
		return 0, err
	}
	for r, rk := range d.ranks {
		if err := d.g.Device(r).WithContext(func() error { return rk.token.CopyFromHost([]int32{tok}) }); err != nil {
			return 0, err
		}
	}
	err := d.g.Do(func() error {
		g := d.g
		for r, rk := range d.ranks {
			if d.fuse && r > 0 {
				break // the ranks share x
			}
			if err := g.Launch(r, d.fns[r][embedGatherName], [3]uint32{1, 1, 1}, [3]uint32{256, 1, 1}, 0,
				rk.embed, rk.token, rk.x[0], int32(d.hidden)); err != nil {
				return err
			}
		}
		n := len(d.ranks[0].attnIn)
		for i := range n {
			if d.fuse {
				if err := d.ranks[0].h.MemsetD32Async(0, uint(d.hidden), g.Stream(0)); err != nil {
					return fmt.Errorf("zero h: %w", err)
				}
				if err := g.Barrier(); err != nil {
					return err
				}
			}
			for r, rk := range d.ranks {
				for _, ls := range [][]egraphLaunch{rk.attnIn[i], rk.rope[i], rk.qcast, rk.sdpa[i], rk.combine, rk.oproj[i]} {
					if err := d.launch(r, ls, pos); err != nil {
						return err
					}
				}
			}
			if err := d.reduce(func(rk *tpRank) (driver.DevicePtr, driver.DevicePtr) { return rk.partO, rk.h }); err != nil {
				return err
			}
			if d.fuse {
				if err := d.ranks[0].x[(i+1)%2].MemsetD32Async(0, uint(d.hidden), g.Stream(0)); err != nil {
					return fmt.Errorf("zero x: %w", err)
				}
				if err := g.Barrier(); err != nil {
					return err
				}
			}
			for r, rk := range d.ranks {
				if err := d.launch(r, rk.post[i], pos); err != nil {
					return err
				}
			}
			if err := d.reduce(func(rk *tpRank) (driver.DevicePtr, driver.DevicePtr) { return rk.partF, rk.x[(i+1)%2] }); err != nil {
				return err
			}
		}
		if err := d.launch(0, d.headLaunch, pos); err != nil {
			return err
		}
		if err := g.Launch(0, d.sample, [3]uint32{1, 1, 1}, [3]uint32{256, 1, 1}, 0, d.logits, d.tokenOut.Ptr(), int32(d.vocab)); err != nil {
			return err
		}
		return g.Synchronize()
	})
	if err != nil {
		return 0, err
	}
	if err := d.tokenOut.CopyToHost(d.hostOut); err != nil {
		return 0, err
	}
	return d.hostOut[0], nil
}

// reduce sums every rank's partial into its output, or, fused, orders
// every rank's later reads after every rank's adds.
func (d *tpDecoder) reduce(bufs func(*tpRank) (part, out driver.DevicePtr)) error {
	if d.fuse {
		return d.g.Barrier()
	}
	var send, recv []driver.DevicePtr
	for _, rk := range d.ranks {
		s, r := bufs(rk)
		send, recv = append(send, s), append(recv, r)
	}
	return d.ar.AllReduce(send, recv, d.hidden)
}

// describe returns a one-line summary of the decoder.
func (d *tpDecoder) describe() string {
	ar := "fused red.global.add epilogue"
	if d.ar != nil {
		ar = fmt.Sprintf("%T", d.ar)
	}
	l := d.plan.Local
	return fmt.Sprintf("TP=%d (%v mode, SMs/rank %d): per rank %d q heads, %d kv heads, %d intermediate; all-reduce: %s",
		d.plan.TP, d.g.Mode(), d.g.SMs(0), l.Heads, l.KVHeads, l.Intermediate, ar)
}

// Close releases the decoder's buffers, modules and group.
func (d *tpDecoder) Close() error {
	var errs []error
	if d.g != nil {
		errs = append(errs, d.g.Do(func() error {
			var errs []error
			if d.ar != nil {
				errs = append(errs, d.ar.Close())
			}
			return errors.Join(errs...)
		}))
	}
	for i := len(d.frees) - 1; i >= 0; i-- {
		errs = append(errs, d.frees[i]())
	}
	d.frees = nil
	for dev, m := range d.mods {
		errs = append(errs, dev.WithContext(m.Close))
		delete(d.mods, dev)
	}
	if d.g != nil {
		errs = append(errs, d.g.Close())
		d.g = nil
	}
	return errors.Join(errs...)
}
