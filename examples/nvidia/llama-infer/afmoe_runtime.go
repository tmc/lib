package main

import (
	"errors"
	"fmt"
	"math"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// afmoeDecoder is a tokenwise eager baseline. It borrows the original BF16
// weights and owns its modules, pointer tables and scratch. KV storage is
// float32 containing BF16-rounded values; no CUDA graph is used.
type afmoeDecoder struct {
	dev               *cuda.Device
	cfg               *afmoeConfig
	weights           *afmoeWeights
	core              *afmoeCoreModule
	attn              *afmoeAttentionModule
	moe               *afmoeMoEKernels
	context           int
	wide              bool
	closed            bool
	embedding, lmHead deviceWeight
	finalNorm         cudamem.Slice[float32]
	layers            []afmoeBoundLayer
	f32               map[string]cudamem.Slice[float32]
	token             cudamem.Slice[int32]
	routes            cudamem.Slice[int32]
	fault             cudamem.Slice[int32]
}

type afmoeBoundLayer struct {
	q, k, v, gate, o                     deviceWeight
	inputNorm, postAttn, preMLP, postMLP cudamem.Slice[float32]
	qNorm, kNorm                         cudamem.Slice[float32]
	gateMLP, upMLP, downMLP              deviceWeight
	router                               deviceWeight
	bias                                 cudamem.Slice[float32]
	tables                               *afmoeExpertTables
	keys, values                         cudamem.Slice[float32]
}

func newAfmoeDecoder(dev *cuda.Device, cfg *afmoeConfig, weights *afmoeWeights, context int, wide bool) (res *afmoeDecoder, err error) {
	if dev == nil || cfg == nil || weights == nil || context < 1 || context > cfg.MaxSeqLen || context > 12032 {
		return nil, fmt.Errorf("afmoe decoder: invalid device, weights or context (baseline maximum 12032)")
	}
	if err := validateAfmoeDeviceWeights(dev, cfg, weights); err != nil {
		return nil, err
	}
	d := &afmoeDecoder{dev: dev, cfg: cfg, weights: weights, context: context, wide: wide,
		layers: make([]afmoeBoundLayer, cfg.NumHiddenLayers), f32: make(map[string]cudamem.Slice[float32])}
	d.embedding = weights.Tensors["model.embed_tokens.weight"]
	d.lmHead = weights.Tensors["lm_head.weight"]
	d.finalNorm = weights.Tensors["model.norm.weight"].buf.(cudamem.Slice[float32])
	defer func() {
		if err != nil {
			if cerr := d.Close(); cerr != nil {
				res = d
				err = errors.Join(err, cerr)
			}
		}
	}()
	if d.core, err = compileAfmoeCoreModule(dev); err != nil {
		return nil, err
	}
	if d.attn, err = compileAfmoeAttentionModule(dev); err != nil {
		return nil, err
	}
	if d.moe, err = compileAFMoEMoEKernels(dev); err != nil {
		return nil, err
	}
	h, q, kv := cfg.HiddenSize, cfg.NumAttentionHeads*cfg.HeadDim(), cfg.NumKeyValueHeads*cfg.HeadDim()
	for name, n := range map[string]int{
		"x": h, "norm": h, "q": q, "k": kv, "v": kv, "gate": q, "attention": q,
		"projected": h, "mlp": h, "post": h, "final": h, "logits": cfg.VocabSize,
		"dense_act": cfg.IntermediateSize, "route_logits": cfg.NumExperts, "route_weights": cfg.NumExpertsPerTok,
		"shared_act": cfg.MoEIntermediateSize, "shared_out": h,
		"routed_act": cfg.NumExpertsPerTok * cfg.MoEIntermediateSize, "routed_out": cfg.NumExpertsPerTok * h,
	} {
		s, aerr := cudamem.Alloc[float32](dev, n)
		d.f32[name] = s
		if aerr != nil {
			return nil, fmt.Errorf("allocate afmoe %s: %w", name, aerr)
		}
	}
	if d.token, err = cudamem.Alloc[int32](dev, 1); err != nil {
		return nil, err
	}
	if d.routes, err = cudamem.Alloc[int32](dev, cfg.NumExpertsPerTok); err != nil {
		return nil, err
	}
	if d.fault, err = cudamem.Alloc[int32](dev, 1); err != nil {
		return nil, err
	}
	vector := func(name string, n int) (cudamem.Slice[float32], error) {
		w, ok := weights.Tensors[name]
		s, valid := w.buf.(cudamem.Slice[float32])
		if !ok || !valid || s.Len() != n || s.Device() != dev {
			return cudamem.Slice[float32]{}, fmt.Errorf("afmoe decoder: invalid vector %s", name)
		}
		return s, nil
	}
	matrix := func(name string, n int) (deviceWeight, error) {
		w, ok := weights.Tensors[name]
		if !ok {
			return deviceWeight{}, fmt.Errorf("afmoe decoder: missing matrix %s", name)
		}
		if _, err := afmoeBF16Matrix(w, dev, n); err != nil {
			return deviceWeight{}, fmt.Errorf("afmoe decoder: matrix %s: %w", name, err)
		}
		return w, nil
	}
	for i := range d.layers {
		l := &d.layers[i]
		pre := fmt.Sprintf("model.layers.%d.", i)
		for _, item := range []struct {
			name string
			n    int
			dst  *deviceWeight
		}{{"self_attn.q_proj.weight", q * h, &l.q}, {"self_attn.k_proj.weight", kv * h, &l.k}, {"self_attn.v_proj.weight", kv * h, &l.v}, {"self_attn.gate_proj.weight", q * h, &l.gate}, {"self_attn.o_proj.weight", h * q, &l.o}} {
			if *item.dst, err = matrix(pre+item.name, item.n); err != nil {
				return nil, err
			}
		}
		for _, item := range []struct {
			name string
			n    int
			dst  *cudamem.Slice[float32]
		}{{"input_layernorm.weight", h, &l.inputNorm}, {"post_attention_layernorm.weight", h, &l.postAttn}, {"pre_mlp_layernorm.weight", h, &l.preMLP}, {"post_mlp_layernorm.weight", h, &l.postMLP}, {"self_attn.q_norm.weight", cfg.HeadDim(), &l.qNorm}, {"self_attn.k_norm.weight", cfg.HeadDim(), &l.kNorm}} {
			if *item.dst, err = vector(pre+item.name, item.n); err != nil {
				return nil, err
			}
		}
		prefix, inter := "mlp.", cfg.IntermediateSize
		if i >= cfg.NumDenseLayers {
			prefix, inter = "mlp.shared_experts.", cfg.MoEIntermediateSize
			if l.router, err = matrix(pre+"mlp.router.gate.weight", cfg.NumExperts*h); err != nil {
				return nil, err
			}
			if l.bias, err = vector(pre+"mlp.expert_bias", cfg.NumExperts); err != nil {
				return nil, err
			}
			gate, up, down := make([]deviceWeight, cfg.NumExperts), make([]deviceWeight, cfg.NumExperts), make([]deviceWeight, cfg.NumExperts)
			for expert := range cfg.NumExperts {
				ep := fmt.Sprintf("%smlp.experts.%d.", pre, expert)
				for _, item := range []struct {
					name string
					n    int
					dst  *deviceWeight
				}{{"gate_proj.weight", inter * h, &gate[expert]}, {"up_proj.weight", inter * h, &up[expert]}, {"down_proj.weight", h * inter, &down[expert]}} {
					if *item.dst, err = matrix(ep+item.name, item.n); err != nil {
						return nil, err
					}
				}
			}
			if l.tables, err = newAFMoEExpertTables(dev, gate, up, down, h, inter); err != nil {
				return nil, err
			}
		}
		for _, item := range []struct {
			name string
			n    int
			dst  *deviceWeight
		}{{"gate_proj.weight", inter * h, &l.gateMLP}, {"up_proj.weight", inter * h, &l.upMLP}, {"down_proj.weight", h * inter, &l.downMLP}} {
			if *item.dst, err = matrix(pre+prefix+item.name, item.n); err != nil {
				return nil, err
			}
		}
		if l.keys, err = cudamem.Alloc[float32](dev, context*kv); err != nil {
			return nil, err
		}
		if l.values, err = cudamem.Alloc[float32](dev, context*kv); err != nil {
			return nil, err
		}
	}
	return d, nil
}

func (d *afmoeDecoder) drain() error {
	return d.dev.WithContext(func() error { return driver.CtxSynchronize() })
}

func (d *afmoeDecoder) Close() error {
	if d == nil || d.closed || d.dev == nil {
		return nil
	}
	if err := d.drain(); err != nil {
		return errors.Join(errReleasesHeld, fmt.Errorf("afmoe drain before release: %w", err))
	}
	var err error
	for i := range d.layers {
		l := &d.layers[i]
		err = errors.Join(err, l.tables.Close(), freeOwned(&l.keys), freeOwned(&l.values))
	}
	for name, s := range d.f32 {
		if e := freeOwned(&s); e != nil {
			err = errors.Join(err, e)
		} else {
			delete(d.f32, name)
		}
	}
	err = errors.Join(err, freeOwned(&d.token), freeOwned(&d.routes), freeOwned(&d.fault), d.core.Close(), d.attn.Close(), d.moe.Close())
	if err == nil {
		d.closed = true
	}
	return err
}

func (d *afmoeDecoder) forward(token int32, pos int, sample bool) (int32, error) {
	if d == nil || d.closed || d.cfg == nil || d.dev == nil || pos < 0 || pos >= d.context || token < 0 || int(token) >= d.cfg.VocabSize {
		return 0, fmt.Errorf("afmoe forward: invalid token or position")
	}
	var result int32
	err := d.dev.WithContext(func() error {
		emit := func(l *cudalaunch.Launcher, args ...any) error { return l.Launch(args...) }
		c, x := d.cfg, d.f32["x"]
		h, q, kv := c.HiddenSize, c.NumAttentionHeads*c.HeadDim(), c.NumKeyValueHeads*c.HeadDim()
		if err := d.core.emitEmbedding(d.embedding, x, int(token), c.VocabSize, h, float32(math.Sqrt(float64(h))), emit); err != nil {
			return err
		}
		norm := func(src, weights, dst cudamem.Slice[float32], width int) error {
			if err := dst.CopyDeviceToDevice(src); err != nil {
				return err
			}
			return d.attn.emitHeadNorm(dst, weights, 1, width, c.RmsNormEps, emit)
		}
		for i := range d.layers {
			l := &d.layers[i]
			if err := norm(x, l.inputNorm, d.f32["norm"], h); err != nil {
				return err
			}
			for _, item := range []struct {
				w    deviceWeight
				out  string
				rows int
			}{{l.q, "q", q}, {l.k, "k", kv}, {l.v, "v", kv}, {l.gate, "gate", q}} {
				if err := d.core.emitLinear(item.w, d.f32["norm"], d.f32[item.out], item.rows, h, true, emit); err != nil {
					return err
				}
			}
			if err := d.attn.emitHeadNorm(d.f32["q"], l.qNorm, c.NumAttentionHeads, c.HeadDim(), c.RmsNormEps, emit); err != nil {
				return err
			}
			if err := d.attn.emitHeadNorm(d.f32["k"], l.kNorm, c.NumKeyValueHeads, c.HeadDim(), c.RmsNormEps, emit); err != nil {
				return err
			}
			window := 0
			if c.LayerTypes[i] == "sliding_attention" {
				window = c.SlidingWindow
				if err := d.attn.emitRoPE(d.f32["q"], c.NumAttentionHeads, c.HeadDim(), pos, c.RopeTheta, emit); err != nil {
					return err
				}
				if err := d.attn.emitRoPE(d.f32["k"], c.NumKeyValueHeads, c.HeadDim(), pos, c.RopeTheta, emit); err != nil {
					return err
				}
			}
			if err := l.keys.Slice(pos*kv, (pos+1)*kv).CopyDeviceToDevice(d.f32["k"]); err != nil {
				return err
			}
			if err := l.values.Slice(pos*kv, (pos+1)*kv).CopyDeviceToDevice(d.f32["v"]); err != nil {
				return err
			}
			if err := d.attn.emitDecode(d.f32["q"], l.keys, l.values, d.f32["attention"], afmoeAttentionShape{c.NumAttentionHeads, c.NumKeyValueHeads, c.HeadDim()}, pos, window, emit); err != nil {
				return err
			}
			if err := d.attn.emitGate(d.f32["attention"], d.f32["gate"], emit); err != nil {
				return err
			}
			if err := d.core.emitLinear(l.o, d.f32["attention"], d.f32["projected"], h, q, true, emit); err != nil {
				return err
			}
			if err := norm(d.f32["projected"], l.postAttn, d.f32["post"], h); err != nil {
				return err
			}
			if err := d.core.emitResidual(x, x, d.f32["post"], emit); err != nil {
				return err
			}
			if err := norm(x, l.preMLP, d.f32["norm"], h); err != nil {
				return err
			}
			if i < c.NumDenseLayers {
				if err := d.moe.emitDenseMLP(l.gateMLP, l.upMLP, l.downMLP, d.f32["norm"], d.f32["dense_act"], d.f32["mlp"], h, c.IntermediateSize, emit); err != nil {
					return err
				}
			} else {
				b := afmoeMoEBuffers{Input: d.f32["norm"], Logits: d.f32["route_logits"], Weights: d.f32["route_weights"], SharedAct: d.f32["shared_act"], SharedOut: d.f32["shared_out"], RoutedAct: d.f32["routed_act"], RoutedOut: d.f32["routed_out"], Output: d.f32["mlp"], IDs: d.routes, Fault: d.fault}
				if err := d.moe.emitMoE(l.tables, l.router, l.bias, l.gateMLP, l.upMLP, l.downMLP, &b, c.NumExpertsPerTok, c.RouteNorm, 1e-20, c.RouteScale, d.wide, emit); err != nil {
					return err
				}
				if err := driver.CtxSynchronize(); err != nil {
					return err
				}
				if err := b.checkFault(); err != nil {
					return err
				}
			}
			if err := norm(d.f32["mlp"], l.postMLP, d.f32["post"], h); err != nil {
				return err
			}
			if err := d.core.emitResidual(x, x, d.f32["post"], emit); err != nil {
				return err
			}
		}
		if sample {
			if err := norm(x, d.finalNorm, d.f32["final"], h); err != nil {
				return err
			}
			if err := d.core.emitLinear(d.lmHead, d.f32["final"], d.f32["logits"], c.VocabSize, h, true, emit); err != nil {
				return err
			}
			if err := d.core.emitArgmax(d.f32["logits"], d.token, emit); err != nil {
				return err
			}
		}
		if err := driver.CtxSynchronize(); err != nil {
			return err
		}
		if sample {
			var value [1]int32
			if err := d.token.CopyToHost(value[:]); err != nil {
				return err
			}
			result = value[0]
			if result < 0 || int(result) >= c.VocabSize {
				return fmt.Errorf("afmoe sampled invalid ID %d", result)
			}
		}
		return nil
	})
	return result, err
}

// validateAfmoeDeviceWeights checks all handles before compiling modules or
// allocating runtime scratch, including embedding, final norm and LM head.
func validateAfmoeDeviceWeights(dev *cuda.Device, cfg *afmoeConfig, weights *afmoeWeights) error {
	if dev == nil || cfg == nil || weights == nil {
		return fmt.Errorf("afmoe weights: missing device, config or weights")
	}
	specs := cfg.tensors()
	if len(weights.Tensors) != len(specs) {
		return fmt.Errorf("afmoe weights: tensor count %d, want %d", len(weights.Tensors), len(specs))
	}
	for _, spec := range specs {
		w, ok := weights.Tensors[spec.Name]
		if !ok {
			return fmt.Errorf("afmoe weights: missing %s", spec.Name)
		}
		n := 1
		for _, d := range spec.Shape {
			if d < 1 || n > math.MaxInt/d {
				return fmt.Errorf("afmoe weights: invalid shape for %s", spec.Name)
			}
			n *= d
		}
		if len(spec.Shape) == 1 {
			s, valid := w.buf.(cudamem.Slice[float32])
			if !valid || w.dtype != weightF32 || s.Ptr() == 0 || s.Device() != dev || s.Len() != n {
				return fmt.Errorf("afmoe weights: invalid vector %s", spec.Name)
			}
		} else if _, err := afmoeBF16Matrix(w, dev, n); err != nil {
			return fmt.Errorf("afmoe weights: %s: %w", spec.Name, err)
		}
	}
	return nil
}

// reset physically clears KV after a complete same-workload warmup. Its drain
// and clears are outside measured prefill/decode intervals.
func (d *afmoeDecoder) reset() error {
	if d == nil || d.closed || d.dev == nil {
		return fmt.Errorf("afmoe reset: decoder unavailable")
	}
	if err := d.drain(); err != nil {
		return errors.Join(errReleasesHeld, err)
	}
	return d.dev.WithContext(func() error {
		for _, l := range d.layers {
			for _, s := range []cudamem.Slice[float32]{l.keys, l.values} {
				if err := s.Ptr().MemsetD32(0, uint(s.Len())); err != nil {
					return err
				}
			}
		}
		return driver.CtxSynchronize()
	})
}
