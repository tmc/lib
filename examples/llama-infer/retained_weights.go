package main

import (
	"fmt"
	"math"

	"github.com/tmc/lib/nvidia/x/cudasafetensors"
)

// A RetainedTensor holds a checkpoint tensor in the dtype the checkpoint
// declared, rather than widened to float32.
//
// Raw is the source byte sequence, unmodified. DType names the interpretation
// and must be carried alongside Raw: F16 and BF16 are both two bytes per
// element and are indistinguishable from the bytes alone, so a tensor that
// loses its tag cannot be decoded correctly. Nothing in this type converts;
// see [RetainedTensor.DecodeFloat32] for that.
//
// The zero value is not usable. Build one with [NewRetainedTensor].
type RetainedTensor struct {
	DType string
	Shape []int
	Raw   []byte
}

// dtypeElemSize returns the size in bytes of one element of dtype.
//
// Only the dtypes the loader accepts are listed. An unsupported dtype is an
// error rather than a guess, so a checkpoint written in a format this package
// does not implement is rejected instead of silently misread.
func dtypeElemSize(dtype string) (int, error) {
	switch dtype {
	case "F32", "float32":
		return 4, nil
	case "F16", "float16", "BF16", "bfloat16":
		return 2, nil
	}
	return 0, fmt.Errorf("unsupported dtype %q, want F32, F16, or BF16", dtype)
}

// elemCount returns the number of elements described by shape.
//
// An empty shape describes a scalar, which is one element. The running product
// is checked at every step: checkpoint metadata is untrusted input, and a
// product that wrapped would describe an impossible tensor as a plausible one.
func elemCount(shape []int) (int, error) {
	n := 1
	for _, d := range shape {
		if d < 0 {
			return 0, fmt.Errorf("negative dimension %d in shape %v", d, shape)
		}
		if d != 0 && n > math.MaxInt/d {
			return 0, fmt.Errorf("element count overflows in shape %v", shape)
		}
		n *= d
	}
	return n, nil
}

// NewRetainedTensor returns a tensor holding raw in its source dtype.
//
// It reports an error unless dtype is supported and len(raw) is exactly the
// number of elements shape describes times the dtype's element size. A short
// or overlong buffer means the checkpoint metadata and its payload disagree,
// which is not something to interpret.
//
// raw is retained, not copied. The caller must not modify it afterwards.
func NewRetainedTensor(raw []byte, dtype string, shape []int) (*RetainedTensor, error) {
	size, err := dtypeElemSize(dtype)
	if err != nil {
		return nil, err
	}
	n, err := elemCount(shape)
	if err != nil {
		return nil, err
	}
	if n != 0 && size > math.MaxInt/n {
		return nil, fmt.Errorf("dtype %s shape %v overflows a byte count", dtype, shape)
	}
	if want := n * size; len(raw) != want {
		return nil, fmt.Errorf("dtype %s shape %v needs %d bytes, got %d", dtype, shape, want, len(raw))
	}
	return &RetainedTensor{DType: dtype, Shape: append([]int(nil), shape...), Raw: raw}, nil
}

// Elems returns the number of elements in the tensor.
func (t *RetainedTensor) Elems() int {
	size, err := dtypeElemSize(t.DType)
	if err != nil || size == 0 {
		return 0
	}
	return len(t.Raw) / size
}

// Bytes returns the number of payload bytes the tensor retains.
func (t *RetainedTensor) Bytes() int { return len(t.Raw) }

// IsHalf reports whether the tensor is retained at two bytes per element.
func (t *RetainedTensor) IsHalf() bool {
	size, err := dtypeElemSize(t.DType)
	return err == nil && size == 2
}

// DecodeFloat32 returns the tensor widened to float32.
//
// This is the explicit reference conversion. It is separate from retention on
// purpose: retaining the source bytes and converting them correctly are two
// different claims, and a caller that wants the float32 path should have to
// ask for it.
func (t *RetainedTensor) DecodeFloat32() []float32 {
	return decodeTensorToFloat32(t.Raw, t.DType)
}

// A RetainedCheckpoint holds every tensor a checkpoint provides, each in its
// source dtype.
type RetainedCheckpoint struct {
	Tensors map[string]*RetainedTensor
}

// PayloadBytes returns the total retained payload across every tensor.
func (c *RetainedCheckpoint) PayloadBytes() int {
	var n int
	for _, t := range c.Tensors {
		n += t.Bytes()
	}
	return n
}

// PayloadBytesAsFloat32 returns what the same tensors would occupy widened to
// float32, which is what the loader used to produce.
func (c *RetainedCheckpoint) PayloadBytesAsFloat32() int {
	var n int
	for _, t := range c.Tensors {
		n += t.Elems() * 4
	}
	return n
}

// tensorSource is the part of the safetensors reader this loader needs.
type tensorSource interface {
	TensorBytes(name string) ([]byte, cudasafetensors.TensorInfo, error)
}

// loadRetainedTensors reads every named tensor from src without widening it.
func loadRetainedTensors(src tensorSource, names []string) (*RetainedCheckpoint, error) {
	c := &RetainedCheckpoint{Tensors: make(map[string]*RetainedTensor, len(names))}
	for _, name := range names {
		raw, info, err := src.TensorBytes(name)
		if err != nil {
			return nil, fmt.Errorf("read %s: %w", name, err)
		}
		t, err := NewRetainedTensor(raw, info.DType, info.Shape)
		if err != nil {
			return nil, fmt.Errorf("retain %s: %w", name, err)
		}
		c.Tensors[name] = t
	}
	return c, nil
}

// LoadRetainedCheckpoint opens the safetensors checkpoint in dir and retains
// every tensor it declares in its source dtype.
//
// It does not widen anything to float32 and it does not depend on any device
// code: the result is a host-side representation that a device upload path can
// consume in its source precision, and that [RetainedTensor.DecodeFloat32]
// can widen where a float32 consumer needs it.
func LoadRetainedCheckpoint(dir string) (*RetainedCheckpoint, error) {
	sw, err := cudasafetensors.OpenSharded(dir)
	if err != nil {
		return nil, fmt.Errorf("open safetensors: %w", err)
	}
	defer sw.Close()

	names := make([]string, 0, len(sw.Tensors))
	for name := range sw.Tensors {
		names = append(names, name)
	}
	return loadRetainedTensors(sw, names)
}
