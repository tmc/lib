package main

import (
	_ "embed"
	"errors"
	"fmt"
	"math"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

//go:embed afmoe_attention.cu
var afmoeAttentionCUDA string

// afmoeAttentionModule is separate from the dense decoder's modules. It owns
// its module until Close succeeds. Its presence does not enable AFMoE execution.
type afmoeAttentionModule struct {
	mod      driver.Module
	dev      *cuda.Device
	headNorm driver.Function
	rope     driver.Function
	decode   driver.Function
	gate     driver.Function
}

func compileAfmoeAttentionModule(dev *cuda.Device) (*afmoeAttentionModule, error) {
	compiled, err := nvrtcx.Compile(afmoeAttentionCUDA,
		nvrtcx.WithName("afmoe_attention.cu"), nvrtcx.WithDeviceArch(dev))
	if err != nil {
		return nil, fmt.Errorf("compile afmoe attention: %w", releaseErrorOwner(err))
	}
	loaded, err := compiled.Load(dev)
	if err != nil {
		return nil, fmt.Errorf("load afmoe attention: %w", releaseErrorOwner(err))
	}
	lookup := func(name string) (fn driver.Function, err error) {
		err = dev.WithContext(func() error {
			fn, err = loaded.Function(name)
			return err
		})
		return fn, err
	}
	unload := func(mod *driver.Module) error {
		return dev.WithContext(func() error { return mod.Close() })
	}
	m, err := lookupAfmoeAttentionModule(loaded, lookup, unload)
	if m != nil {
		m.dev = dev
	}
	return m, err
}

func lookupAfmoeAttentionModule(loaded driver.Module, lookup func(string) (driver.Function, error), unload func(*driver.Module) error) (*afmoeAttentionModule, error) {
	m := &afmoeAttentionModule{mod: loaded}
	for _, entry := range []struct {
		name string
		dst  *driver.Function
	}{
		{"AfmoeHeadNorm", &m.headNorm},
		{"AfmoeRoPE", &m.rope},
		{"AfmoeAttentionDecode", &m.decode},
		{"AfmoeAttentionGate", &m.gate},
	} {
		var err error
		if *entry.dst, err = lookup(entry.name); err != nil {
			err = fmt.Errorf("lookup afmoe kernel %s: %w", entry.name, err)
			if cerr := m.close(unload); cerr != nil {
				return m, errors.Join(err, fmt.Errorf("unload afmoe attention: %w", cerr))
			}
			return nil, err
		}
	}
	return m, nil
}

func (m *afmoeAttentionModule) Close() error {
	if m != nil && m.dev != nil {
		return m.close(func(mod *driver.Module) error {
			return m.dev.WithContext(func() error { return mod.Close() })
		})
	}
	return m.close((*driver.Module).Close)
}

func (m *afmoeAttentionModule) close(unload func(*driver.Module) error) error {
	if m == nil || m.mod == 0 {
		return nil
	}
	if err := unload(&m.mod); err != nil {
		return err
	}
	*m = afmoeAttentionModule{}
	return nil
}

type afmoeKernelEmitter func(*cudalaunch.Launcher, ...any) error

// emitHeadNorm requires BF16-rounded input and BF16-valued norm weights in
// float32 buffers. It rounds normalized values before multiplying the weights.
func (m *afmoeAttentionModule) emitHeadNorm(x, weight cudamem.Slice[float32], heads, headDim int, eps float32, emit afmoeKernelEmitter) error {
	if m == nil || m.mod == 0 || m.headNorm == 0 || emit == nil {
		return fmt.Errorf("afmoe head norm: module or emitter unavailable")
	}
	if heads <= 0 || headDim <= 0 || heads > math.MaxInt32/headDim || x.Len() != heads*headDim || weight.Len() != headDim || eps <= 0 || math.IsNaN(float64(eps)) || math.IsInf(float64(eps), 0) {
		return fmt.Errorf("afmoe head norm: invalid dimensions or epsilon")
	}
	return emit(cudalaunch.Kernel(m.headNorm).Grid(uint32(heads), 1, 1).Block(256, 1, 1),
		x, weight, int32(heads), int32(headDim), eps)
}

func (m *afmoeAttentionModule) emitRoPE(x cudamem.Slice[float32], heads, headDim, pos int, theta float32, emit afmoeKernelEmitter) error {
	if m == nil || m.mod == 0 || m.rope == 0 || emit == nil {
		return fmt.Errorf("afmoe rope: module or emitter unavailable")
	}
	if heads <= 0 || headDim <= 0 || headDim%2 != 0 || heads > math.MaxInt32/headDim || x.Len() != heads*headDim || pos < 0 || pos > math.MaxInt32 || theta <= 0 || math.IsNaN(float64(theta)) || math.IsInf(float64(theta), 0) {
		return fmt.Errorf("afmoe rope: invalid dimensions, position or theta")
	}
	return emit(cudalaunch.Kernel(m.rope).Grid(uint32((heads*(headDim/2)+255)/256), 1, 1).Block(256, 1, 1),
		x, int32(heads), int32(headDim), int32(pos), theta)
}

// emitDecode reads a cache containing the current token at pos. The local
// window includes that token. It uses at most 48 KiB shared memory, so longer
// contexts need a different attention implementation rather than an oversized
// launch. Q/K/V and gate projections must already have rounded to BF16.
func (m *afmoeAttentionModule) emitDecode(q, keys, values, out cudamem.Slice[float32], s afmoeAttentionShape, pos, window int, emit afmoeKernelEmitter) error {
	if m == nil || m.mod == 0 || m.decode == 0 || emit == nil {
		return fmt.Errorf("afmoe attention: module or emitter unavailable")
	}
	if err := s.validate(); err != nil {
		return err
	}
	if s.Heads > math.MaxInt32/s.HeadDim || s.KVHeads > math.MaxInt32/s.HeadDim || pos < 0 || pos >= math.MaxInt32 || window < 0 || window > math.MaxInt32 {
		return fmt.Errorf("afmoe attention: dimensions exceed kernel limits")
	}
	stride := s.KVHeads * s.HeadDim
	if q.Len() != s.Heads*s.HeadDim || out.Len() != q.Len() || keys.Len() != values.Len() || keys.Len()%stride != 0 || pos >= keys.Len()/stride || keys.Len() > math.MaxInt32 {
		return fmt.Errorf("afmoe attention: invalid query, output or cache dimensions")
	}
	count := pos - afmoeAttentionStart(pos, window) + 1
	shared := (count + 256) * 4
	if shared > 48*1024 {
		return fmt.Errorf("afmoe attention: %d shared bytes exceed baseline limit", shared)
	}
	return emit(cudalaunch.Kernel(m.decode).Grid(uint32(s.Heads), 1, 1).Block(256, 1, 1).SharedMem(uint32(shared)),
		q, keys, values, out, int32(s.Heads), int32(s.KVHeads), int32(s.HeadDim), int32(pos), int32(window))
}

func (m *afmoeAttentionModule) emitGate(out, gate cudamem.Slice[float32], emit afmoeKernelEmitter) error {
	if m == nil || m.mod == 0 || m.gate == 0 || emit == nil {
		return fmt.Errorf("afmoe attention gate: module or emitter unavailable")
	}
	if out.Len() <= 0 || out.Len() != gate.Len() || out.Len() > math.MaxInt32-255 {
		return fmt.Errorf("afmoe attention gate: invalid dimensions")
	}
	return emit(cudalaunch.Kernel(m.gate).Grid(uint32((out.Len()+255)/256), 1, 1).Block(256, 1, 1),
		out, gate, int32(out.Len()))
}
