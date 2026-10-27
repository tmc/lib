package main

import (
	"errors"
	"fmt"
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
type egraphLaunch struct {
	fn          driver.Function
	grid, block [3]uint32
	args        []any
}

// An egraphDecoder runs the decode step's projections, norms, SwiGLU and
// residuals as kernels the e-graph generator emits ([buildEgraphPlans]).
// It owns the generated module, the fused QKV and gate·up weights those
// kernels read, and their intermediates.
type egraphDecoder struct {
	plans *egraphPlans
	mod   driver.Module
	fns   map[string]driver.Function // kernel name -> function

	wqkv, wgu []deviceWeight
	bqkv      []cudamem.Slice[float32]
	qkv, x2   cudamem.Slice[float32]
	temps32   []cudamem.Slice[float32]
	temps16   []cudamem.Slice[uint16]
	tempPtr   map[string]driver.DevicePtr

	attnIn, post [][]egraphLaunch // per layer
	head         []egraphLaunch
	final        cudamem.Slice[float32] // the residual stream after the last layer
}

func newEgraphDecoder(dev *cuda.Device, w *ModelWeights, cfg *ModelConfig, gpuW *GPUModelWeights, sp *GPUScratchpad) (eg *egraphDecoder, err error) {
	if gpuW.DType != weightBF16 {
		return nil, errors.New("-egraph needs -weight-dtype bf16")
	}
	bias := len(w.Layers) > 0 && len(w.Layers[0].QBias) > 0
	plans, err := buildEgraphPlans(cfg, bias)
	if err != nil {
		return nil, fmt.Errorf("build e-graph plans: %w", err)
	}
	eg = &egraphDecoder{plans: plans, fns: make(map[string]driver.Function), tempPtr: make(map[string]driver.DevicePtr)}
	defer func() {
		if err != nil {
			err = errors.Join(err, eg.Close())
			eg = nil
		}
	}()
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
	for _, p := range []*hopper.Plan{plans.attnIn, plans.post, plans.head} {
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
	eg.final = cur
	eg.head, err = eg.bind(plans.head, map[string]driver.DevicePtr{
		"x":                    cur.Ptr(),
		"norm":                 gpuW.Norm.Ptr(),
		"lm_head":              gpuW.LMHead.Ptr(),
		plans.head.Output.Name: sp.Logits.Ptr(),
	})
	if err != nil {
		return nil, err
	}
	return eg, nil
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

// kernelName returns the NVRTC name expression for k.
func kernelName(k hopper.PlanKernel) (string, error) {
	switch {
	case k.GEMM != nil:
		return fmt.Sprintf("%s<%d>", k.GEMM.Name, k.M), nil
	case k.Row != nil:
		return k.Row.Name, nil
	}
	return "", fmt.Errorf("plan kernel %+v: only skinny gemm and row kernels are supported", k)
}

func (eg *egraphDecoder) compile(dev *cuda.Device) error {
	var src strings.Builder
	src.WriteString(hopper.Prelude)
	var names []string
	for _, p := range []*hopper.Plan{eg.plans.attnIn, eg.plans.post, eg.plans.head} {
		for _, k := range p.Kernels {
			name, err := kernelName(k)
			if err != nil {
				return err
			}
			var cu string
			if k.GEMM != nil {
				cu, err = k.GEMM.CUDA()
			} else {
				cu, err = k.Row.CUDA()
			}
			if err != nil {
				return fmt.Errorf("emit %s: %w", name, err)
			}
			src.WriteString("\n" + cu + "\n")
			names = append(names, name)
		}
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
	var out []egraphLaunch
	for _, k := range p.Kernels {
		name, err := kernelName(k)
		if err != nil {
			return nil, err
		}
		l := egraphLaunch{fn: eg.fns[name]}
		var grid, block [3]int
		if k.GEMM != nil {
			if grid, block, err = k.GEMM.Grid(); err != nil {
				return nil, fmt.Errorf("%s: %w", name, err)
			}
		} else {
			grid, block = k.Row.Grid(k.Rows)
			l.args = append(l.args, int32(k.Rows))
		}
		for i := range 3 {
			l.grid[i], l.block[i] = uint32(grid[i]), uint32(block[i])
		}
		for _, a := range k.Args {
			ptr, ok := bufs[a]
			if !ok {
				ptr, ok = eg.tempPtr[a]
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

// Kernels returns the number of generated launches per decode step.
func (eg *egraphDecoder) Kernels() int {
	return len(eg.attnIn)*(len(eg.plans.attnIn.Kernels)+len(eg.plans.post.Kernels)) + len(eg.plans.head.Kernels)
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
			if err := emit(cudalaunch.Kernel(l.fn).Grid(l.grid[0], l.grid[1], l.grid[2]).Block(l.block[0], l.block[1], l.block[2]), l.args...); err != nil {
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
