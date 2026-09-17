package cudaplan

import (
	"errors"
	"fmt"
	"math"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

var (
	// ErrNilBuffer is returned when a required binding is nil or has null pointer.
	ErrNilBuffer = errors.New("cudaplan: nil buffer pointer")

	// ErrCrossDeviceBuffer is returned when a bound buffer belongs to a different CUDA device.
	ErrCrossDeviceBuffer = errors.New("cudaplan: cross-device buffer mismatch")

	// ErrOverlappingOutputs is returned when output buffers overlap with each other or with inputs.
	ErrOverlappingOutputs = errors.New("cudaplan: overlapping buffers detected")

	// ErrShapeMismatch is returned when buffer byte length does not match required tensor size.
	ErrShapeMismatch = errors.New("cudaplan: buffer size mismatch")

	// ErrAlignmentMismatch is returned when buffer address violates hardware alignment constraints.
	ErrAlignmentMismatch = errors.New("cudaplan: buffer address not aligned")

	// ErrMissingBinding is returned when a required input or output binding is absent.
	ErrMissingBinding = errors.New("cudaplan: missing required buffer binding")

	// ErrWorkspaceBudgetExceeded is returned when required scratchpad memory exceeds configured peak budget.
	ErrWorkspaceBudgetExceeded = errors.New("cudaplan: workspace budget exceeded")

	// ErrStaleBinding is returned when an instantiated graph binding has been invalidated.
	ErrStaleBinding = errors.New("cudaplan: stale graph binding")

	// ErrClosed is returned when attempting an operation on a closed plan or resource.
	ErrClosed = errors.New("cudaplan: plan is closed")

	// ErrInvocationInFlight is returned when an invocation is submitted while another is already running.
	ErrInvocationInFlight = errors.New("cudaplan: an invocation is already in flight")
)

// BufferBinding describes a physical device memory binding for a graph node.
type BufferBinding struct {
	NodeName  string
	DevicePtr driver.DevicePtr
	Bytes     int
	Device    *cuda.Device
	Release   func() // Optional release callback to retain allocation
}

// Bindings manages the mapping from node names to buffer bindings.
type Bindings struct {
	inputs  map[string]BufferBinding
	outputs map[string]BufferBinding
}

// NewBindings initializes an empty Bindings container.
func NewBindings() *Bindings {
	return &Bindings{
		inputs:  make(map[string]BufferBinding),
		outputs: make(map[string]BufferBinding),
	}
}

// SetInput binds a device buffer for an input node.
func (b *Bindings) SetInput(name string, ptr driver.DevicePtr, bytes int, dev *cuda.Device) error {
	if b == nil {
		return errors.New("cudaplan: nil bindings")
	}
	if name == "" {
		return errors.New("cudaplan: empty input name")
	}
	if ptr == 0 {
		return fmt.Errorf("%w: input %q", ErrNilBuffer, name)
	}
	if bytes <= 0 {
		return fmt.Errorf("cudaplan: invalid byte count %d for input %q", bytes, name)
	}
	if uint64(ptr) > math.MaxUint64-uint64(bytes) {
		return errors.New("cudaplan: input address range overflows")
	}
	if b.inputs == nil {
		b.inputs = make(map[string]BufferBinding)
	}
	b.inputs[name] = BufferBinding{
		NodeName:  name,
		DevicePtr: ptr,
		Bytes:     bytes,
		Device:    dev,
	}
	return nil
}

// SetOutput binds a device buffer for an output node.
func (b *Bindings) SetOutput(name string, ptr driver.DevicePtr, bytes int, dev *cuda.Device) error {
	if b == nil {
		return errors.New("cudaplan: nil bindings")
	}
	if name == "" {
		return errors.New("cudaplan: empty output name")
	}
	if ptr == 0 {
		return fmt.Errorf("%w: output %q", ErrNilBuffer, name)
	}
	if bytes <= 0 {
		return fmt.Errorf("cudaplan: invalid byte count %d for output %q", bytes, name)
	}
	if uint64(ptr) > math.MaxUint64-uint64(bytes) {
		return errors.New("cudaplan: output address range overflows")
	}
	if b.outputs == nil {
		b.outputs = make(map[string]BufferBinding)
	}
	b.outputs[name] = BufferBinding{
		NodeName:  name,
		DevicePtr: ptr,
		Bytes:     bytes,
		Device:    dev,
	}
	return nil
}

// SetInputSlice binds a cudamem.Slice as an input.
func SetInputSlice[T any](b *Bindings, name string, s cudamem.Slice[T], dev *cuda.Device) error {
	if s.SizeBytes() > uintptr(math.MaxInt) {
		return errors.New("cudaplan: input byte length overflows int")
	}
	bytes := int(s.SizeBytes())
	return b.SetInput(name, s.Ptr(), bytes, dev)
}

// SetOutputSlice binds a cudamem.Slice as an output.
func SetOutputSlice[T any](b *Bindings, name string, s cudamem.Slice[T], dev *cuda.Device) error {
	if s.SizeBytes() > uintptr(math.MaxInt) {
		return errors.New("cudaplan: output byte length overflows int")
	}
	bytes := int(s.SizeBytes())
	return b.SetOutput(name, s.Ptr(), bytes, dev)
}

// Validate verifies all input and output bindings against the plan's requirements,
// checking device affinity, size bounds, alignment, and lack of overlapping memory ranges.
func (b *Bindings) Validate(planDev *cuda.Device, requiredInputs, requiredOutputs map[string]int) error {
	if b == nil {
		return errors.New("cudaplan: nil bindings")
	}
	// Check inputs
	for name, reqBytes := range requiredInputs {
		if reqBytes <= 0 {
			return fmt.Errorf("cudaplan: invalid required input size for %q", name)
		}
		binding, ok := b.inputs[name]
		if !ok {
			return fmt.Errorf("%w: input %q", ErrMissingBinding, name)
		}
		if binding.DevicePtr == 0 {
			return fmt.Errorf("%w: input %q", ErrNilBuffer, name)
		}
		if planDev != nil && binding.Device != planDev {
			return fmt.Errorf("%w: input %q belongs to different device", ErrCrossDeviceBuffer, name)
		}
		if binding.Bytes < reqBytes {
			return fmt.Errorf("%w: input %q has %d bytes, need %d", ErrShapeMismatch, name, binding.Bytes, reqBytes)
		}
		if uintptr(binding.DevicePtr)%8 != 0 {
			return fmt.Errorf("%w: input %q pointer 0x%x not 8-byte aligned", ErrAlignmentMismatch, name, binding.DevicePtr)
		}
	}

	// Check outputs
	for name, reqBytes := range requiredOutputs {
		if reqBytes <= 0 {
			return fmt.Errorf("cudaplan: invalid required output size for %q", name)
		}
		binding, ok := b.outputs[name]
		if !ok {
			return fmt.Errorf("%w: output %q", ErrMissingBinding, name)
		}
		if binding.DevicePtr == 0 {
			return fmt.Errorf("%w: output %q", ErrNilBuffer, name)
		}
		if planDev != nil && binding.Device != planDev {
			return fmt.Errorf("%w: output %q belongs to different device", ErrCrossDeviceBuffer, name)
		}
		if binding.Bytes < reqBytes {
			return fmt.Errorf("%w: output %q has %d bytes, need %d", ErrShapeMismatch, name, binding.Bytes, reqBytes)
		}
		if uintptr(binding.DevicePtr)%8 != 0 {
			return fmt.Errorf("%w: output %q pointer 0x%x not 8-byte aligned", ErrAlignmentMismatch, name, binding.DevicePtr)
		}
	}

	// Check for buffer overlap among outputs, and between outputs and distinct inputs
	type rangeSpan struct {
		name  string
		start uint64
		end   uint64
	}
	var outSpans []rangeSpan
	for name, binding := range b.outputs {
		if binding.Bytes <= 0 || uint64(binding.DevicePtr) > math.MaxUint64-uint64(binding.Bytes) {
			return errors.New("cudaplan: invalid output address range")
		}
		outSpans = append(outSpans, rangeSpan{
			name:  name,
			start: uint64(binding.DevicePtr),
			end:   uint64(binding.DevicePtr) + uint64(binding.Bytes),
		})
	}
	for _, out := range outSpans {
		for name, in := range b.inputs {
			if in.Bytes <= 0 || uint64(in.DevicePtr) > math.MaxUint64-uint64(in.Bytes) {
				return errors.New("cudaplan: invalid input address range")
			}
			if rangesOverlap(out.start, out.end, uint64(in.DevicePtr), uint64(in.DevicePtr)+uint64(in.Bytes)) {
				return fmt.Errorf("%w: output %q overlaps input %q", ErrOverlappingOutputs, out.name, name)
			}
		}
	}

	for i := 0; i < len(outSpans); i++ {
		for j := i + 1; j < len(outSpans); j++ {
			if rangesOverlap(outSpans[i].start, outSpans[i].end, outSpans[j].start, outSpans[j].end) {
				return fmt.Errorf("%w: output %q [0x%x, 0x%x) overlaps output %q [0x%x, 0x%x)",
					ErrOverlappingOutputs, outSpans[i].name, outSpans[i].start, outSpans[i].end,
					outSpans[j].name, outSpans[j].start, outSpans[j].end)
			}
		}
	}

	return nil
}

func rangesOverlap(start1, end1, start2, end2 uint64) bool {
	return start1 < end2 && start2 < end1
}
