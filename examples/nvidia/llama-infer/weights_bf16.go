package main

import (
	"fmt"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/safetensors"
)

// A weightDType is the device storage type of the matrix weights: the
// projections, the embedding table and lm_head. Norms, biases, activations
// and accumulation are float32 in every mode.
type weightDType string

const (
	weightF32  weightDType = "f32"
	weightBF16 weightDType = "bf16"
)

func parseWeightDType(s string) (weightDType, error) {
	switch d := weightDType(s); d {
	case weightF32, weightBF16:
		return d, nil
	}
	return "", fmt.Errorf("unknown weight dtype %q (want f32 or bf16)", s)
}

// A deviceBuffer is device memory a weight refers to. cudamem.Slice
// satisfies it.
type deviceBuffer interface {
	Ptr() driver.DevicePtr
	Free() error
}

// A deviceWeight is a matrix weight in device memory. A borrowed weight is a
// view of memory another weight owns, such as an lm_head tied to the
// embedding table, and releasing it never frees that memory.
type deviceWeight struct {
	buf      deviceBuffer
	dtype    weightDType
	borrowed bool
}

// Ptr returns the weight's device address, or 0 for no weight. It makes a
// deviceWeight a kernel argument.
func (w deviceWeight) Ptr() driver.DevicePtr {
	if w.buf == nil {
		return 0
	}
	return w.buf.Ptr()
}

// borrow returns a view of w that does not own its memory.
func (w deviceWeight) borrow() deviceWeight {
	w.borrowed = true
	return w
}

// release frees w's memory if w owns it, then zeroes w. If the free fails,
// w is left as it was, so that release can be called again, and the error
// is returned unwrapped.
func (w *deviceWeight) release() error {
	if w.buf == nil {
		return nil
	}
	if !w.borrowed {
		if err := w.buf.Free(); err != nil {
			return err
		}
	}
	*w = deviceWeight{}
	return nil
}

// roundBF16 rounds host weights to bfloat16, to nearest with ties to even.
// Weights read from a BF16 checkpoint were widened exactly, so rounding them
// returns the checkpoint's bits.
func roundBF16(host []float32) []uint16 {
	b := make([]uint16, len(host))
	for i, f := range host {
		b[i] = safetensors.Float32ToBFloat16(f)
	}
	return b
}

// uploadWeight copies host to the device as dtype. If the allocation or the
// copy fails, the returned weight still owns any memory allocated, for the
// caller to release.
func uploadWeight(dev *cuda.Device, host []float32, dtype weightDType) (deviceWeight, error) {
	switch dtype {
	case weightF32:
		return uploadAs(dtype, host, func(n int) (cudamem.Slice[float32], error) {
			return cudamem.Alloc[float32](dev, n)
		})
	case weightBF16:
		return uploadAs(dtype, roundBF16(host), func(n int) (cudamem.Slice[uint16], error) {
			return cudamem.Alloc[uint16](dev, n)
		})
	}
	return deviceWeight{}, fmt.Errorf("unknown weight dtype %q", dtype)
}

// uploadAs allocates len(host) elements with alloc and copies host there.
// cudamem.Alloc can return a nonzero slice, which the caller owns, along
// with its error, so the returned weight holds any nonzero allocation
// whatever the error. A zero allocation gives a zero weight.
func uploadAs[T any, S interface {
	deviceBuffer
	CopyFromHost([]T) error
}](dtype weightDType, host []T, alloc func(n int) (S, error)) (deviceWeight, error) {
	s, err := alloc(len(host))
	if err != nil {
		if s.Ptr() == 0 {
			return deviceWeight{}, err
		}
		return deviceWeight{buf: s, dtype: dtype}, err
	}
	return deviceWeight{buf: s, dtype: dtype}, s.CopyFromHost(host)
}

// storedAs reports the dtype w's buffer holds, when its type says.
func (w deviceWeight) storedAs() (weightDType, bool) {
	switch w.buf.(type) {
	case cudamem.Slice[float32]:
		return weightF32, true
	case cudamem.Slice[uint16]:
		return weightBF16, true
	}
	return "", false
}

// checkDTypes reports a matrix weight that is missing or not stored as the
// model's dtype. Mixed storage would launch a kernel on bytes of the wrong
// type.
func (w *GPUModelWeights) checkDTypes() error {
	for i, m := range w.matrices() {
		if m.buf == nil {
			return fmt.Errorf("matrix weight %d is missing", i)
		}
		if m.dtype != w.DType {
			return fmt.Errorf("matrix weight %d is %s, model is %s", i, m.dtype, w.DType)
		}
		if d, ok := m.storedAs(); ok && d != m.dtype {
			return fmt.Errorf("matrix weight %d is labelled %s but stored as %s", i, m.dtype, d)
		}
	}
	return nil
}
