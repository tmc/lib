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

//go:embed afmoe_core.cu
var afmoeCoreCUDA string

// afmoeCoreModule owns the common AFMoE baseline kernels. It retains its
// module when unloading fails, so the caller can retry Close.
type afmoeCoreModule struct {
	mod       driver.Module
	run       func(func() error) error
	embedding driver.Function
	linear    driver.Function
	residual  driver.Function
	argmax    driver.Function
}

func compileAfmoeCoreModule(dev *cuda.Device) (*afmoeCoreModule, error) {
	compiled, err := nvrtcx.Compile(afmoeCoreCUDA,
		nvrtcx.WithName("afmoe_core.cu"), nvrtcx.WithDeviceArch(dev))
	if err != nil {
		return nil, fmt.Errorf("compile afmoe core: %w", releaseErrorOwner(err))
	}
	loaded, err := compiled.Load(dev)
	if err != nil {
		return nil, fmt.Errorf("load afmoe core: %w", releaseErrorOwner(err))
	}
	return bindAfmoeCoreModule(loaded, dev.WithContext, loaded.Function, (*driver.Module).Close)
}

// bindAfmoeCoreModule retains the loaded owner even if entering its context
// fails before lookup. Load restores the previous thread context.
func bindAfmoeCoreModule(loaded driver.Module, run func(func() error) error, lookup func(string) (driver.Function, error), unload func(*driver.Module) error) (*afmoeCoreModule, error) {
	m := &afmoeCoreModule{mod: loaded, run: run}
	err := run(func() error {
		var err error
		m, err = lookupAfmoeCoreModule(loaded, lookup, unload)
		return err
	})
	if m != nil {
		m.run = run
	}
	if err != nil {
		return m, fmt.Errorf("initialize afmoe core functions: %w", err)
	}
	return m, nil
}

func lookupAfmoeCoreModule(loaded driver.Module, lookup func(string) (driver.Function, error), unload func(*driver.Module) error) (*afmoeCoreModule, error) {
	m := &afmoeCoreModule{mod: loaded}
	for _, entry := range []struct {
		name string
		dst  *driver.Function
	}{
		{"AfmoeEmbedding", &m.embedding},
		{"AfmoeLinear", &m.linear},
		{"AfmoeResidual", &m.residual},
		{"AfmoeArgmax", &m.argmax},
	} {
		var err error
		if *entry.dst, err = lookup(entry.name); err == nil && *entry.dst == 0 {
			err = fmt.Errorf("zero function")
		}
		if err != nil {
			err = fmt.Errorf("lookup afmoe kernel %s: %w", entry.name, err)
			if cerr := m.close(unload); cerr != nil {
				return m, errors.Join(err, fmt.Errorf("unload afmoe core: %w", cerr))
			}
			return nil, err
		}
	}
	return m, nil
}

func (m *afmoeCoreModule) Close() error { return m.close((*driver.Module).Close) }

func (m *afmoeCoreModule) close(unload func(*driver.Module) error) error {
	if m == nil || m.mod == 0 {
		return nil
	}
	close := func() error {
		if err := unload(&m.mod); err != nil {
			return err
		}
		*m = afmoeCoreModule{}
		return nil
	}
	if m.run != nil {
		return m.run(close)
	}
	return close()
}

func afmoeCoreWeight(w deviceWeight, elements int) bool {
	b, ok := w.buf.(cudamem.Slice[uint16])
	return ok && w.dtype == weightBF16 && w.Ptr() != 0 && b.Len() == elements
}

// afmoeCoreOverlap checks bounded device address intervals without adding
// lengths to the device pointers, so pointer-end arithmetic cannot wrap.
func afmoeCoreOverlap(a driver.DevicePtr, aBytes int, b driver.DevicePtr, bBytes int) bool {
	if aBytes <= 0 || bBytes <= 0 {
		return false
	}
	if a <= b {
		return uint64(b-a) < uint64(aBytes)
	}
	return uint64(a-b) < uint64(bBytes)
}

func (m *afmoeCoreModule) emitEmbedding(table deviceWeight, out cudamem.Slice[float32], token, vocab, hidden int, scale float32, emit afmoeKernelEmitter) error {
	if m == nil || m.mod == 0 || m.embedding == 0 || emit == nil {
		return fmt.Errorf("afmoe embedding: module or emitter unavailable")
	}
	if vocab <= 0 || vocab > math.MaxInt32 || hidden <= 0 || hidden > math.MaxInt32-255 || vocab > math.MaxInt/hidden || token < 0 || token >= vocab || !afmoeFinite(scale) || scale <= 0 || out.Len() != hidden || out.Ptr() == 0 {
		return fmt.Errorf("afmoe embedding: invalid dimensions, token or scale")
	}
	n := vocab * hidden
	if n > math.MaxInt/2 || hidden > math.MaxInt/4 || !afmoeCoreWeight(table, n) || afmoeCoreOverlap(table.Ptr(), n*2, out.Ptr(), hidden*4) {
		return fmt.Errorf("afmoe embedding: invalid weight storage or overlapping output")
	}
	return emit(cudalaunch.Kernel(m.embedding).Grid(uint32((hidden+255)/256), 1, 1).Block(256, 1, 1),
		table, out, int32(token), int32(hidden), scale)
}

// emitLinear reads BF16 weights and BF16-valued float32 activations. The
// official BF16 lm_head also returns BF16, so its caller passes roundOutput
// true. False is an explicit FP32 diagnostic boundary, not the reference.
func (m *afmoeCoreModule) emitLinear(weights deviceWeight, in, out cudamem.Slice[float32], rows, cols int, roundOutput bool, emit afmoeKernelEmitter) error {
	if m == nil || m.mod == 0 || m.linear == 0 || emit == nil {
		return fmt.Errorf("afmoe linear: module or emitter unavailable")
	}
	if rows <= 0 || cols <= 0 || rows > math.MaxInt32 || cols > math.MaxInt32-255 || rows > math.MaxInt/cols || rows > math.MaxInt/4 || cols > math.MaxInt/4 || in.Len() != cols || out.Len() != rows || in.Ptr() == 0 || out.Ptr() == 0 {
		return fmt.Errorf("afmoe linear: invalid dimensions or buffers")
	}
	n := rows * cols
	if n > math.MaxInt/2 || !afmoeCoreWeight(weights, n) || afmoeCoreOverlap(in.Ptr(), cols*4, out.Ptr(), rows*4) || afmoeCoreOverlap(weights.Ptr(), n*2, out.Ptr(), rows*4) {
		return fmt.Errorf("afmoe linear: invalid weight storage or overlapping output")
	}
	round := int32(0)
	if roundOutput {
		round = 1
	}
	return emit(cudalaunch.Kernel(m.linear).Grid(uint32(rows), 1, 1).Block(256, 1, 1),
		weights, in, out, int32(rows), int32(cols), round)
}

func (m *afmoeCoreModule) emitResidual(dst, x, y cudamem.Slice[float32], emit afmoeKernelEmitter) error {
	if m == nil || m.mod == 0 || m.residual == 0 || emit == nil {
		return fmt.Errorf("afmoe residual: module or emitter unavailable")
	}
	n := dst.Len()
	if n <= 0 || n > math.MaxInt32-255 || n > math.MaxInt/4 || x.Len() != n || y.Len() != n || dst.Ptr() == 0 || x.Ptr() == 0 || y.Ptr() == 0 {
		return fmt.Errorf("afmoe residual: invalid dimensions or buffers")
	}
	for _, src := range []cudamem.Slice[float32]{x, y} {
		if dst.Ptr() != src.Ptr() && afmoeCoreOverlap(dst.Ptr(), n*4, src.Ptr(), n*4) {
			return fmt.Errorf("afmoe residual: partially overlapping buffers")
		}
	}
	return emit(cudalaunch.Kernel(m.residual).Grid(uint32((n+255)/256), 1, 1).Block(256, 1, 1), dst, x, y, int32(n))
}

// emitArgmax writes -1 if any logit is nonfinite. Otherwise ties choose the
// lowest vocabulary index. The caller must validate the copied scalar ID.
func (m *afmoeCoreModule) emitArgmax(logits cudamem.Slice[float32], token cudamem.Slice[int32], emit afmoeKernelEmitter) error {
	if m == nil || m.mod == 0 || m.argmax == 0 || emit == nil {
		return fmt.Errorf("afmoe argmax: module or emitter unavailable")
	}
	n := logits.Len()
	if n <= 0 || n > math.MaxInt32-255 || n > math.MaxInt/4 || token.Len() != 1 || logits.Ptr() == 0 || token.Ptr() == 0 || afmoeCoreOverlap(logits.Ptr(), n*4, token.Ptr(), 4) {
		return fmt.Errorf("afmoe argmax: invalid dimensions or overlapping output")
	}
	return emit(cudalaunch.Kernel(m.argmax).Grid(1, 1, 1).Block(256, 1, 1), logits, token, int32(n))
}
