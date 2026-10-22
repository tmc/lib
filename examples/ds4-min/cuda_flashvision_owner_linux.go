//go:build linux

package main

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"sync"

	"github.com/tmc/lib/examples/ds4-min/dsv4vision"
	"github.com/tmc/lib/nvidia/cublas"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// flashVisionEncoder owns the encoder's stream, module, cuBLAS handle, weights
// and scratch, but borrows its device. The device must outlive Close. Operations
// are serialized. A failed constructor returns its owner with the error so a
// caller can retry Close rather than lose partially constructed CUDA resources.
type flashVisionEncoder struct {
	mu           sync.Mutex
	device       *cuda.Device
	module       driver.Module
	arena        *cudamem.Arena
	graph        flashVisionGraph
	gridH, gridW int
	ready        bool
}

func flashVisionWeightSizes() map[string]int {
	m := map[string]int{"vision.patch_embed.proj.weight": 588 * 1024, "vision.patch_embed.proj.bias": 1024, "vision.norm.weight": 1024,
		"aligner.w1.weight": 9216 * 4096, "aligner.w1.bias": 4096, "aligner.w2.weight": 4096 * 4096, "aligner.w2.bias": 4096}
	for layer := 0; layer < 32; layer++ {
		p := fmt.Sprintf("vision.blocks.%d.", layer)
		for name, count := range map[string]int{"norm1.weight": 1024, "attn.wqkv.weight": 1024 * 3072, "attn.wqkv.bias": 3072, "attn.wo.weight": 1024 * 1024,
			"attn.wo.bias": 1024, "norm2.weight": 1024, "mlp.w1.weight": 1024 * 5632, "mlp.w2.weight": 2816 * 1024} {
			m[p+name] = count
		}
	}
	return m
}

func newFlashVisionEncoder(dev *cuda.Device, sidecar *dsv4vision.Sidecar, h, w int) (*flashVisionEncoder, error) {
	if dev == nil || sidecar == nil || h < 1 || w < 1 || h > 384 || w > 384 {
		return nil, fmt.Errorf("vision encoder: invalid device, sidecar or grid")
	}
	// The actual image layout is the admission bound, including compression pads.
	if _, _, err := dsv4vision.Layout((h+2)/3, (w+2)/3, 0); err != nil {
		return nil, err
	}
	e := &flashVisionEncoder{device: dev, arena: cudamem.NewArena(dev), gridH: h, gridW: w}
	e.graph.weights = make(map[string]cudamem.Slice[uint16])
	e.graph.buffer = make(map[string]cudamem.Slice[float32])
	e.graph.kernels = make(map[string]driver.Function)
	fail := func(err error) (*flashVisionEncoder, error) {
		return e, fmt.Errorf("construct vision encoder: %w", err)
	}
	var err error
	e.graph.stream, err = dev.NewStreamWithFlags(driver.StreamNonBlocking)
	if err != nil {
		return fail(err)
	}
	compiled, err := compileFlashVisionCUDA(flashVisionCUDA, nvrtcx.WithName("ds4_flashvision.cu"), nvrtcx.WithDeviceArch(dev), nvrtcx.WithFlag("--use_fast_math"))
	if err != nil {
		return fail(err)
	}
	e.module, err = compiled.Load(dev)
	if err != nil {
		return fail(err)
	}
	if err := dev.WithContext(func() error {
		var err error
		e.graph.blas, err = cublas.Create()
		if err != nil {
			return err
		}
		for _, name := range []string{"matvec_bf16", "f32_to_bf16", "round_bf16", "bias", "rms", "qkv_rope", "attention", "swiglu", "add_residual", "aligner_reorder", "gelu_bias"} {
			fn, err := e.module.Function("ds4_vision_" + name)
			if err != nil {
				return err
			}
			e.graph.kernels[name] = fn
		}
		return nil
	}); err != nil {
		return fail(err)
	}
	// Read and upload one tensor at a time. No pageable file mapping is exposed
	// to kernels and no duplicate full-sidecar host allocation is retained.
	for name, count := range flashVisionWeightSizes() {
		t, err := sidecar.Tensor(name)
		if err != nil {
			return fail(err)
		}
		if t.TypeID != 30 || t.Bytes != uint64(count)*2 {
			return fail(fmt.Errorf("invalid %s tensor", name))
		}
		r, err := sidecar.Reader(name)
		if err != nil {
			return fail(err)
		}
		b := make([]byte, count*2)
		if _, err := io.ReadFull(r, b); err != nil {
			return fail(err)
		}
		v := make([]uint16, count)
		for i := range v {
			v[i] = binary.LittleEndian.Uint16(b[2*i:])
		}
		dst := cudamem.AllocIn[uint16](e.arena, name, count)
		if err := e.arena.Err(); err != nil {
			return fail(err)
		}
		e.graph.weights[name] = dst
		if err := dst.CopyFromHost(v); err != nil {
			return fail(err)
		}
	}
	rows, aligned := h*w, ((h+2)/3)*((w+2)/3)
	for name, count := range map[string]int{"patch": rows * 588, "a": rows * 1024, "b": rows * 1024, "qkv": rows * 3072, "q": rows * 1024, "k": rows * 1024, "v": rows * 1024,
		"attn": rows * 1024, "mlp_w1": rows * 5632, "mlp_mid": rows * 2816, "align_in": aligned * 9216, "align_a": aligned * 4096, "align_b": aligned * 4096} {
		e.graph.buffer[name] = cudamem.AllocIn[float32](e.arena, name, count)
	}
	e.graph.activation = cudamem.AllocIn[uint16](e.arena, "activation-bf16", max(rows*2816, aligned*9216))
	if err := e.arena.Err(); err != nil {
		return fail(err)
	}
	e.ready = true
	return e, nil
}

func (e *flashVisionEncoder) encode(patches dsv4vision.Patches) ([]float32, error) {
	if e == nil {
		return nil, fmt.Errorf("vision encoder: nil owner")
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if !e.ready || patches.GridHeight != e.gridH || patches.GridWidth != e.gridW || len(patches.Values) != e.gridH*e.gridW*588 {
		return nil, fmt.Errorf("vision encoder: closed or incompatible patches")
	}
	if err := e.graph.buffer["patch"].CopyFromHost(patches.Values); err != nil {
		return nil, err
	}
	var out cudamem.Slice[float32]
	err := e.device.WithContext(func() error { var err error; out, err = e.graph.encode(e.gridH, e.gridW); return err })
	if err != nil {
		// Some earlier launches may still consume scratch after a later launch
		// fails. Drain them before permitting another host write or evaluation.
		if syncErr := e.graph.stream.Synchronize(context.Background()); syncErr != nil {
			e.ready = false
			return nil, errors.Join(err, syncErr)
		}
		return nil, err
	}
	if err := e.graph.stream.Synchronize(context.Background()); err != nil {
		e.ready = false
		return nil, err
	}
	values := make([]float32, out.Len())
	if err := out.CopyToHost(values); err != nil {
		return nil, err
	}
	return values, nil
}

func (e *flashVisionEncoder) Close() error {
	return e.closeWithOperations(flashVisionCloseOperations{})
}

// Overrides are instance-local host fault controls. Zero operations use the
// native context and release calls; they do not bypass stream synchronization.
type flashVisionCloseOperations struct {
	synchronize  func() error
	withContext  func(func() error) error
	destroyBLAS  func(cublas.Handle) error
	unloadModule func(driver.Module) error
}

func (e *flashVisionEncoder) closeWithOperations(ops flashVisionCloseOperations) error {
	if e == nil {
		return nil
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	e.ready = false
	// A failed synchronization leaves all producer/consumer resources owned.
	if ops.synchronize == nil {
		ops.synchronize = func() error {
			if e.graph.stream != nil {
				return e.graph.stream.Synchronize(context.Background())
			}
			return nil
		}
	}
	if err := ops.synchronize(); err != nil {
		return err
	}
	var errs []error
	if ops.withContext == nil {
		ops.withContext = e.device.WithContext
	}
	if ops.destroyBLAS == nil {
		ops.destroyBLAS = func(h cublas.Handle) error { return h.Close() }
	}
	if ops.unloadModule == nil {
		ops.unloadModule = func(m driver.Module) error { return m.Close() }
	}
	if e.graph.blas != 0 {
		if err := ops.withContext(func() error {
			if err := ops.destroyBLAS(e.graph.blas); err != nil {
				return err
			}
			e.graph.blas = 0
			return nil
		}); err != nil {
			errs = append(errs, err)
		}
	}
	if e.module != 0 {
		if err := ops.withContext(func() error {
			if err := ops.unloadModule(e.module); err != nil {
				return err
			}
			e.module = 0
			return nil
		}); err != nil {
			errs = append(errs, err)
		}
	}
	if e.arena != nil {
		if err := e.arena.Free(); err != nil {
			errs = append(errs, err)
		} else {
			e.arena = nil
		}
	}
	if len(errs) == 0 && e.graph.stream != nil {
		if err := e.graph.stream.Close(); err != nil {
			errs = append(errs, err)
		} else {
			e.graph.stream = nil
		}
	}
	return errors.Join(errs...)
}
