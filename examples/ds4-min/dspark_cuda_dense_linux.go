//go:build linux

package main

import (
	_ "embed"
	"fmt"
	"github.com/tmc/lib/examples/ds4-min/dsv4"
	"github.com/tmc/lib/nvidia/driver"

	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

//go:embed cuda_dspark_dense.cu
var cudaDSparkDenseSource string

//go:embed cuda_dspark_attention.cu
var cudaDSparkAttentionSource string

//go:embed cuda_dspark_input.cu
var cudaDSparkInputSource string

//go:embed cuda_dspark_head.cu
var cudaDSparkHeadSource string

func dsparkRequestOpen(r *cudaRuntime) error {
	if r == nil || r.cudaFoundation == nil {
		return dsv4.ErrClosed
	}
	if r.failed != nil {
		return r.failed
	}
	return r.checkOpen()
}

// loadMathLocked publishes a complete module only after every expected
// entrypoint exists. Caller holds w.mu. Loads use CUBIN exclusively.
func (w *cudaDSparkWeights) loadMathLocked() error {
	if w.closed || len(w.moduleCleanup) != 0 {
		return fmt.Errorf("cuda dspark math: closed owner or pending module cleanup")
	}
	if w.module != 0 {
		return nil
	}
	compiled, err := nvrtcx.Compile(cudaDSparkDenseSource+"\n"+cudaDSparkAttentionSource+"\n"+cudaDSparkInputSource+"\n"+cudaDSparkHeadSource, nvrtcx.WithName("cuda_dspark_math.cu"), nvrtcx.WithDeviceArch(w.device))
	if err != nil {
		return err
	}
	module, _, err := compiled.LoadOutputWithLog(w.device, nvrtcx.OutputCUBIN)
	var matrix = w.matrix
	module, retained, err := admitDSparkHandle(module, err, func(module driver.Module) error {
		return w.device.WithContext(func() error {
			for _, name := range []string{"ds4_dspark_matvec_f32", "ds4_dspark_attention_noncausal", "ds4_dspark_rope_tail", "ds4_dspark_store_raw", "ds4_dspark_repeat_hc", "ds4_dspark_markov_row", "ds4_dspark_markov_argmax", "ds4_dspark_confidence_input", "ds4_dspark_confidence_prefix"} {
				fn, err := module.Function(name)
				if err != nil {
					return err
				}
				if name == "ds4_dspark_matvec_f32" {
					matrix = fn
				}
			}
			return nil
		})
	}, func(module driver.Module) (bool, error) {
		released := false
		err := w.device.WithContext(func() error {
			if err := module.Close(); err != nil {
				return err
			}
			released = true
			return nil
		})
		return released, err
	})
	w.moduleCleanup = append(w.moduleCleanup, retained...)
	if err != nil {
		return err
	}
	w.module, w.matrix = module, matrix
	return nil
}

// encodeDense reads only resident admitted support weights. It serializes
// module publication/close, but enqueues on the consuming request's stream.
// F32 uses its own device reduction; F16/Q8_0 use the existing native math.
// None of these paths invokes the CPU support reference.
func (w *cudaDSparkWeights) encodeDense(r *cudaRuntime, name string, input, output cudamem.Slice[float32]) error {
	return w.encodeDenseRange(r, name, 0, output.Len(), input, output)
}

// encodeDenseRange selects contiguous output rows without repacking the
// immutable weights. Grouped attention A uses one rank slice per head group.
func (w *cudaDSparkWeights) encodeDenseRange(r *cudaRuntime, name string, first, count int, input, output cudamem.Slice[float32]) error {
	if err := dsparkRequestOpen(r); err != nil {
		return err
	}
	if w == nil || r == nil || r.cudaFoundation == nil || r.device == nil {
		return fmt.Errorf("cuda dspark dense: missing owner or request")
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	tensor, ok := w.tensors[name]
	if w.closed || r.device != w.device || !ok || len(tensor.dimensions) != 2 || first < 0 || count < 1 || uint64(first) > tensor.dimensions[1] || uint64(count) > tensor.dimensions[1]-uint64(first) || tensor.bytes%tensor.dimensions[1] != 0 || input.Ptr() == 0 || output.Ptr() == 0 || uint64(input.Len()) != tensor.dimensions[0] || output.Len() != count {
		return fmt.Errorf("cuda dspark dense: closed owner or shape/device mismatch")
	}
	stride := tensor.bytes / tensor.dimensions[1]
	tensor.device += driver.DevicePtr(uint64(first) * stride)
	tensor.bytes = uint64(count) * stride
	tensor.dimensions = []uint64{tensor.dimensions[0], uint64(count)}
	switch tensor.typeID {
	case 1:
		return r.cudaFoundation.encodeF16Matvec(tensor, input, output)
	case 8:
		return r.cudaFoundation.encodeQ80Matvec(tensor, input, output)
	case 0:
	default:
		return fmt.Errorf("cuda dspark dense: unsupported type %d", tensor.typeID)
	}
	if err := w.loadMathLocked(); err != nil {
		return err
	}
	return r.withContext(func() error {
		return cudalaunch.Kernel(w.matrix).Grid(uint32(tensor.dimensions[1]), 1, 1).Block(256, 1, 1).On(r.stream()).Launch(tensor.device, input, output, uint32(tensor.dimensions[0]))
	})
}
