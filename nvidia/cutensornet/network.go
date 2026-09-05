package cutensornet

import (
	"errors"
	"fmt"
	"runtime"
	"unsafe"
)

// A Tensor describes one operand of a tensor network.
//
// Modes carries one integer label per dimension. Two dimensions carry the same
// label exactly when they are the same index: a label shared between two input
// tensors is contracted, a label that also appears in the output tensor is
// kept. Extents carries the size of each dimension and must be the same length
// as Modes.
//
// Strides is optional and, when present, must be the same length as Modes. It
// gives the distance in elements — not bytes — between logically neighbouring
// elements along each dimension. A nil Strides means the library derives
// generalized column-major (Fortran) strides from Extents. cuTensorNet operands
// are strided, not packed: a row-major operand must state its strides
// explicitly, and so must any operand that is a view into a larger allocation.
type Tensor struct {
	Modes      []int32
	Extents    []int64
	Strides    []int64
	Qualifiers TensorQualifiers
}

// ErrNetworkInUse reports an attempt to release a [Network] while a [Plan]
// built from it is still live.
var ErrNetworkInUse = errors.New("cutensornet: network still referenced by a live plan")

// ErrNetworkClosed reports use of a [Network] after Close.
var ErrNetworkClosed = errors.New("cutensornet: network is closed")

// validate reports whether t is internally consistent. what names the operand
// in the returned error.
func (t Tensor) validate(what string) error {
	if len(t.Modes) == 0 {
		return fmt.Errorf("cutensornet: %s has no modes", what)
	}
	if len(t.Extents) != len(t.Modes) {
		return fmt.Errorf("cutensornet: %s has %d modes but %d extents", what, len(t.Modes), len(t.Extents))
	}
	if t.Strides != nil && len(t.Strides) != len(t.Modes) {
		return fmt.Errorf("cutensornet: %s has %d modes but %d strides", what, len(t.Modes), len(t.Strides))
	}
	seen := make(map[int32]bool, len(t.Modes))
	for i, m := range t.Modes {
		if t.Extents[i] < 1 {
			return fmt.Errorf("cutensornet: %s mode %d (label %d) has extent %d, want >= 1", what, i, m, t.Extents[i])
		}
		if seen[m] {
			return fmt.Errorf("cutensornet: %s repeats mode label %d; a diagonal or trace operand is not expressible this way", what, m)
		}
		seen[m] = true
	}
	return nil
}

// A Network is a validated tensor-network description together with the
// descriptor the library holds for it.
//
// The mode, extent and stride slices handed to [Handle.NewNetwork] are borrowed
// for the duration of the call only; the library copies what it needs. The
// descriptor itself is heap-allocated inside the library and must be released
// with Close.
//
// A Network is not safe for concurrent use.
type Network struct {
	h         Handle
	desc      NetworkDescriptor
	numInputs int
	dataType  DataType
	plans     int
	closed    bool
}

// NewNetwork describes a tensor network with the given input operands and
// output operand and returns the resulting descriptor.
//
// A nil output asks the library to infer the output modes from the inputs: any
// mode that appears in exactly one input survives, in the order the library
// chooses. Pass an explicit output to fix the mode order, the extents, or a
// non-default layout. An output with no modes is a full reduction to a scalar.
//
// dataType applies to every operand; the library does not support mixing
// element types within one network. computeType selects the accumulation
// precision and must be one of the combinations cuTensorNet documents for
// dataType; an unsupported pair is rejected by the library, not here.
//
// NewNetwork validates, before calling the library, that every operand is
// self-consistent and that a mode label has the same extent everywhere it
// appears. It does not validate the data-type and compute-type pairing, the
// strides against the underlying allocation, or anything that needs a device.
//
// The returned Network must be released with [Network.Close].
func (h Handle) NewNetwork(inputs []Tensor, output *Tensor, dataType DataType, computeType ComputeType) (*Network, error) {
	if len(inputs) == 0 {
		return nil, errors.New("cutensornet: network has no input tensors")
	}
	if len(inputs) > 1<<31-1 {
		return nil, fmt.Errorf("cutensornet: %d input tensors exceeds the int32 count the library takes", len(inputs))
	}

	extent := map[int32]int64{}
	for i, t := range inputs {
		if err := t.validate(fmt.Sprintf("input tensor %d", i)); err != nil {
			return nil, err
		}
		for j, m := range t.Modes {
			if got, ok := extent[m]; ok && got != t.Extents[j] {
				return nil, fmt.Errorf("cutensornet: mode label %d has extent %d in input tensor %d but %d elsewhere", m, t.Extents[j], i, got)
			}
			extent[m] = t.Extents[j]
		}
	}
	if output != nil && len(output.Modes) > 0 {
		if err := output.validate("output tensor"); err != nil {
			return nil, err
		}
		for j, m := range output.Modes {
			got, ok := extent[m]
			if !ok {
				return nil, fmt.Errorf("cutensornet: output mode label %d does not appear in any input tensor", m)
			}
			if got != output.Extents[j] {
				return nil, fmt.Errorf("cutensornet: mode label %d has extent %d in the output tensor but %d in the inputs", m, output.Extents[j], got)
			}
		}
	}

	numModesIn := make([]int32, len(inputs))
	extentsIn := make([]unsafe.Pointer, len(inputs))
	modesIn := make([]unsafe.Pointer, len(inputs))
	qualifiersIn := make([]TensorQualifiers, len(inputs))
	var stridesIn []unsafe.Pointer
	for _, t := range inputs {
		if t.Strides != nil {
			stridesIn = make([]unsafe.Pointer, len(inputs))
			break
		}
	}
	for i, t := range inputs {
		numModesIn[i] = int32(len(t.Modes))
		extentsIn[i] = unsafe.Pointer(&t.Extents[0])
		modesIn[i] = unsafe.Pointer(&t.Modes[0])
		qualifiersIn[i] = t.Qualifiers
		if stridesIn != nil && t.Strides != nil {
			stridesIn[i] = unsafe.Pointer(&t.Strides[0])
		}
	}

	// numModesOut of -1 asks the library to infer the output modes; 0 forces a
	// full reduction. Both are documented sentinels, not error codes.
	numModesOut := int32(-1)
	var extentsOut, stridesOut *int64
	var modesOut *int32
	if output != nil {
		numModesOut = int32(len(output.Modes))
		if numModesOut > 0 {
			extentsOut = &output.Extents[0]
			modesOut = &output.Modes[0]
			if output.Strides != nil {
				stridesOut = &output.Strides[0]
			}
		}
	}

	desc, err := h.CreateNetworkDescriptor(int32(len(inputs)), &numModesIn[0],
		extentsIn, stridesIn, modesIn, &qualifiersIn[0],
		numModesOut, extentsOut, stridesOut, modesOut, dataType, computeType)
	runtime.KeepAlive(inputs)
	runtime.KeepAlive(output)
	runtime.KeepAlive(numModesIn)
	runtime.KeepAlive(extentsIn)
	runtime.KeepAlive(stridesIn)
	runtime.KeepAlive(modesIn)
	runtime.KeepAlive(qualifiersIn)
	if err != nil {
		return nil, err
	}
	return &Network{h: h, desc: desc, numInputs: len(inputs), dataType: dataType}, nil
}

// Descriptor returns the raw descriptor, for the generated calls that this
// package does not wrap. The descriptor is owned by n and is invalid after
// [Network.Close].
func (n *Network) Descriptor() NetworkDescriptor { return n.desc }

// NumInputs returns the number of input operands the network was built with.
func (n *Network) NumInputs() int { return n.numInputs }

// DataType returns the element type shared by every operand.
func (n *Network) DataType() DataType { return n.dataType }

// Close releases the network descriptor.
//
// It reports [ErrNetworkInUse] and releases nothing while a [Plan] built from n
// is still live, because the library reads the descriptor during contraction.
// Close is idempotent once it has succeeded.
//
// cuTensorNet 2.13 renamed the destructor for this handle from
// cutensornetDestroyNetworkDescriptor to cutensornetDestroyNetwork, deprecating
// the former. Close calls the current name and falls back to the deprecated one
// when the loaded library is old enough not to export it, so the descriptor is
// released either way rather than leaked.
func (n *Network) Close() error {
	if n.closed {
		return nil
	}
	if n.plans > 0 {
		return ErrNetworkInUse
	}
	n.closed = true
	err := n.desc.Destroy()
	var missing *SymbolError
	if errors.As(err, &missing) {
		return n.desc.DestroyDescriptor()
	}
	return err
}

// ElementSize returns the size in bytes of one element of type d, and reports
// whether d is a type whose size this package knows. Sub-byte types (the 4-bit
// and 6-bit formats) report false: their storage is packed and not addressable
// per element.
func ElementSize(d DataType) (int, bool) {
	switch d {
	case R8I, R8U, R8F_E4M3, R8F_E5M2, R8F_UE8M0:
		return 1, true
	case R16F, R16BF, R16I, R16U, C8I, C8U:
		return 2, true
	case R32F, R32I, R32U, C16F, C16BF, C16I, C16U:
		return 4, true
	case R64F, R64I, R64U, C32F, C32I, C32U:
		return 8, true
	case C64F, C64I, C64U:
		return 16, true
	}
	return 0, false
}
