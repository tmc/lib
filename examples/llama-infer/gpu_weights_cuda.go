//go:build !darwin

package main

import (
	"encoding/binary"
	"fmt"
	"reflect"
	"strings"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// A WeightSlice is device storage for one weight tensor.
//
// The stored dtype is a property of the slice rather than of its Go type, so a
// half model and a float32 model differ in how many bytes each weight occupies
// and in which kernel reads it, not in the shape of the code that launches it.
// It satisfies cudalaunch.DevicePtrProvider, so a launch site passes it exactly
// as it passed a cudamem.Slice.
//
// The zero value is an absent weight: Len reports 0 and Ptr reports the null
// device pointer, which is what the kernels test for an optional bias or norm.
type WeightSlice struct {
	dtype WeightDType
	elems int
	f32   cudamem.Slice[float32]
	half  cudamem.Slice[uint16]
}

// Ptr returns the device address of the weight data.
func (w WeightSlice) Ptr() driver.DevicePtr {
	if w.dtype == WeightF32 {
		return w.f32.Ptr()
	}
	return w.half.Ptr()
}

// Len returns the number of weights, not the number of bytes.
//
// Callers use this to decide whether an optional tensor is present, so it must
// not change meaning with the stored dtype.
func (w WeightSlice) Len() int { return w.elems }

// Bytes returns the device memory the weights occupy.
func (w WeightSlice) Bytes() int { return w.elems * w.dtype.Bytes() }

// DType returns the dtype the weights are actually stored in.
func (w WeightSlice) DType() WeightDType { return w.dtype }

// uploadWeights copies host into device memory in mode d.
//
// A checkpoint tensor that is already stored in d's dtype is copied through
// without conversion; only a float32 source is rounded, and only once. host is
// the float32 view, which is what the reference loader produces.
func uploadWeights(arena *cudamem.Arena, d WeightDType, host []float32) (WeightSlice, error) {
	if len(host) == 0 {
		return WeightSlice{dtype: d}, nil
	}
	if d == WeightF32 {
		s := cudamem.AllocIn[float32](arena, "model-weight", len(host))
		if err := arena.Err(); err != nil {
			return WeightSlice{}, err
		}
		if err := s.CopyFromHost(host); err != nil {
			return WeightSlice{}, err
		}
		return WeightSlice{dtype: d, elems: len(host), f32: s}, nil
	}

	// Half storage: allocate and copy the 16-bit words directly, so the device
	// never holds a float32 copy of the tensor even transiently.
	packed := make([]uint16, len(host))
	switch d {
	case WeightF16:
		for i, v := range host {
			packed[i] = float32ToFloat16(v)
		}
	case WeightBF16:
		for i, v := range host {
			packed[i] = float32ToBfloat16(v)
		}
	}
	s := cudamem.AllocIn[uint16](arena, "model-weight", len(packed))
	if err := arena.Err(); err != nil {
		return WeightSlice{}, err
	}
	if err := s.CopyFromHost(packed); err != nil {
		return WeightSlice{}, err
	}
	return WeightSlice{dtype: d, elems: len(host), half: s}, nil
}

// WeightBytes returns the total device memory the model's weights occupy.
//
// It counts weights only. KV cache, scratchpad and any workspace are allocated
// elsewhere and reported separately, because a weight-storage change should not
// be credited with memory it did not save.
func (w *GPUModelWeights) WeightBytes() int {
	if w == nil {
		return 0
	}
	n := w.EmbedTokens.Bytes() + w.Norm.Bytes()
	if w.LMHead.Ptr() != w.EmbedTokens.Ptr() {
		n += w.LMHead.Bytes()
	}
	for i := range w.Layers {
		l := &w.Layers[i]
		for _, s := range []WeightSlice{
			l.InputNorm, l.PostNorm, l.WQ, l.WK, l.WV,
			l.QBias, l.KBias, l.VBias, l.WO, l.WGate, l.WUp, l.WDown,
		} {
			n += s.Bytes()
		}
	}
	return n
}

// DType returns the dtype the model's weights are stored in.
func (w *GPUModelWeights) DType() WeightDType {
	if w == nil {
		return WeightF32
	}
	return w.dtype
}

// bindWeightKernels resolves the Kernels struct against mod for weight dtype d.
//
// Fields tagged weights:"yes" have one entry point per dtype and resolve to the
// suffixed name; the rest read no weights and have a single entry point. A
// missing kernel is an error rather than a fallback to the float32 name, so a
// module that was compiled without half support fails the load instead of
// silently running in float32 while reporting f16.
func bindWeightKernels(mod *cuda.Module, d WeightDType, kernels *Kernels) error {
	rv := reflect.ValueOf(kernels).Elem()
	rt := rv.Type()
	for i := 0; i < rt.NumField(); i++ {
		f := rt.Field(i)
		name := f.Tag.Get("cuda")
		if name == "" {
			return fmt.Errorf("kernel field %s has no cuda tag", f.Name)
		}
		if f.Tag.Get("weights") == "yes" {
			name += d.Suffix()
		}
		fn, err := mod.Function(name)
		if err != nil {
			return fmt.Errorf("bind kernel %q (field %s, weights %s): %w", name, f.Name, d, err)
		}
		rv.Field(i).Set(reflect.ValueOf(fn))
	}
	return nil
}

// weightKernelNames returns every kernel name the module must export for d.
//
// It is what the host-side kernel source check compares against, so that a
// kernel added to Kernels without a matching device entry point is caught
// without a GPU.
func weightKernelNames(d WeightDType) []string {
	rt := reflect.TypeOf(Kernels{})
	names := make([]string, 0, rt.NumField())
	for i := 0; i < rt.NumField(); i++ {
		f := rt.Field(i)
		name := f.Tag.Get("cuda")
		if f.Tag.Get("weights") == "yes" {
			name += d.Suffix()
		}
		names = append(names, name)
	}
	return names
}

// weightKernelReport describes the selected dtype and the kernels it bound.
//
// A run reports what it actually dispatched rather than what it requested, so
// an f32 execution can never be recorded as an f16 success.
func weightKernelReport(d WeightDType) string {
	return fmt.Sprintf("weights=%s bytes/weight=%d kernels=[%s]",
		d, d.Bytes(), strings.Join(weightKernelNames(d), " "))
}

// CopyToHost reads the weights back as the float32 values the kernels see.
//
// For a half slice that is the widened value, not the original float32 the
// weight was rounded from: the rounded value is what is actually stored, so it
// is what a reader must be shown.
func (w WeightSlice) CopyToHost(dst []float32) error {
	if w.dtype == WeightF32 {
		return w.f32.CopyToHost(dst)
	}
	packed := make([]uint16, w.elems)
	if err := w.half.CopyToHost(packed); err != nil {
		return err
	}
	n := min(len(dst), len(packed))
	for i := 0; i < n; i++ {
		if w.dtype == WeightF16 {
			dst[i] = float16ToFloat32(packed[i])
		} else {
			dst[i] = bfloat16ToFloat32(packed[i])
		}
	}
	return nil
}

// CopyFromHost writes float32 values into the slice, rounding once to the
// stored dtype.
func (w WeightSlice) CopyFromHost(src []float32) error {
	if w.dtype == WeightF32 {
		return w.f32.CopyFromHost(src)
	}
	packed := make([]uint16, len(src))
	for i, v := range src {
		if w.dtype == WeightF16 {
			packed[i] = float32ToFloat16(v)
		} else {
			packed[i] = float32ToBfloat16(v)
		}
	}
	return w.half.CopyFromHost(packed)
}

// uploadRetainedWeights copies a retained tensor into device memory in mode d.
//
// When the tensor's checkpoint dtype already matches d, the source bytes are
// copied through unchanged: nothing is widened, nothing is rounded, and no
// float32 copy of the tensor exists on either side of the transfer. Otherwise
// the tensor is widened once and rounded once, which is the only conversion any
// weight undergoes.
func uploadRetainedWeights(arena *cudamem.Arena, d WeightDType, t *RetainedTensor) (WeightSlice, error) {
	if t == nil || t.Elems() == 0 {
		return WeightSlice{dtype: d}, nil
	}
	if d == WeightF32 || t.DType != d.CheckpointDType() {
		return uploadWeights(arena, d, t.DecodeFloat32())
	}
	packed := make([]uint16, t.Elems())
	for i := range packed {
		packed[i] = binary.LittleEndian.Uint16(t.Raw[i*2:])
	}
	s := cudamem.AllocIn[uint16](arena, "model-weight", len(packed))
	if err := arena.Err(); err != nil {
		return WeightSlice{}, err
	}
	if err := s.CopyFromHost(packed); err != nil {
		return WeightSlice{}, err
	}
	return WeightSlice{dtype: d, elems: t.Elems(), half: s}, nil
}
