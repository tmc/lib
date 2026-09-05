package cutensornet

import (
	"fmt"
	"unsafe"
)

// The attribute calls in this API take an untyped buffer and a size in bytes.
// The library validates the size against the attribute and rejects a mismatch,
// but a buffer of the right size and the wrong type is accepted and silently
// reinterpreted — reading a double-valued attribute into two int32s, for
// instance. The helpers below carry the value type of every attribute the
// pinned header documents, so a wrong-typed access is refused in Go instead.
//
// Attributes whose value is a struct or a variable-length array are marked
// composite: there is no single Go scalar for them, so they must go through the
// generated GetAttribute and SetAttribute with a buffer the caller sizes.

// attrKind classifies the value behind an attribute buffer.
type attrKind int

const (
	attrComposite attrKind = iota // struct or variable-length array
	attrInt32
	attrInt64
	attrFloat64
	attrUndocumented // the pinned header states no type
)

func (k attrKind) String() string {
	switch k {
	case attrInt32:
		return "int32"
	case attrInt64:
		return "int64"
	case attrFloat64:
		return "float64"
	case attrUndocumented:
		return "an undocumented type"
	}
	return "a composite value"
}

// optimizerConfigKinds gives the documented value type of each contraction
// optimizer configuration attribute. Enum-valued attributes are int32: the C
// enums here have no value outside the int32 range.
var optimizerConfigKinds = map[ContractionOptimizerConfigAttributes]attrKind{
	ContractionOptimizerConfigGraphNumPartitions:      attrInt32,
	ContractionOptimizerConfigGraphCutoffSize:         attrInt32,
	ContractionOptimizerConfigGraphAlgorithm:          attrInt32,
	ContractionOptimizerConfigGraphImbalanceFactor:    attrInt32,
	ContractionOptimizerConfigGraphNumIterations:      attrInt32,
	ContractionOptimizerConfigGraphNumCuts:            attrInt32,
	ContractionOptimizerConfigReconfigNumIterations:   attrInt32,
	ContractionOptimizerConfigReconfigNumLeaves:       attrInt32,
	ContractionOptimizerConfigSlicerDisableSlicing:    attrInt32,
	ContractionOptimizerConfigSlicerMemoryModel:       attrInt32,
	ContractionOptimizerConfigSlicerMemoryFactor:      attrInt32,
	ContractionOptimizerConfigSlicerMinSlices:         attrInt32,
	ContractionOptimizerConfigSlicerSliceFactor:       attrInt32,
	ContractionOptimizerConfigHyperNumSamples:         attrInt32,
	ContractionOptimizerConfigHyperNumThreads:         attrInt32,
	ContractionOptimizerConfigSimplificationDisableDr: attrInt32,
	ContractionOptimizerConfigSeed:                    attrInt32,
	ContractionOptimizerConfigCostFunctionObjective:   attrInt32,
	ContractionOptimizerConfigCacheReuseNruns:         attrInt32,
	ContractionOptimizerConfigSmartOption:             attrInt32,
	ContractionOptimizerConfigGpuArch:                 attrUndocumented,
}

// optimizerInfoKinds gives the documented value type of each contraction
// optimizer info attribute.
var optimizerInfoKinds = map[ContractionOptimizerInfoAttributes]attrKind{
	ContractionOptimizerInfoPath:                 attrComposite,
	ContractionOptimizerInfoNumSlices:            attrInt64,
	ContractionOptimizerInfoNumSlicedModes:       attrInt32,
	ContractionOptimizerInfoSlicedMode:           attrComposite,
	ContractionOptimizerInfoSlicedExtent:         attrComposite,
	ContractionOptimizerInfoSlicingConfig:        attrComposite,
	ContractionOptimizerInfoSlicingOverhead:      attrFloat64,
	ContractionOptimizerInfoPhase1FlopCount:      attrFloat64,
	ContractionOptimizerInfoFlopCount:            attrFloat64,
	ContractionOptimizerInfoEffectiveFlopsEst:    attrFloat64,
	ContractionOptimizerInfoRuntimeEst:           attrFloat64,
	ContractionOptimizerInfoLargestTensor:        attrFloat64,
	ContractionOptimizerInfoIntermediateModes:    attrComposite,
	ContractionOptimizerInfoNumIntermediateModes: attrComposite,
}

// networkKinds gives the documented value type of each network attribute.
var networkKinds = map[NetworkAttributes]attrKind{
	NetworkInputTensorsNumConstant:    attrInt32,
	NetworkInputTensorsConstant:       attrComposite,
	NetworkInputTensorsNumConjugated:  attrInt32,
	NetworkInputTensorsConjugated:     attrComposite,
	NetworkInputTensorsNumRequireGrad: attrInt32,
	NetworkInputTensorsRequireGrad:    attrComposite,
	NetworkComputeType:                attrInt32,
}

// autotuneKinds gives the documented value type of each contraction autotune
// preference attribute.
var autotuneKinds = map[ContractionAutotunePreferenceAttributes]attrKind{
	ContractionAutotuneMaxIterations:     attrInt32,
	ContractionAutotuneIntermediateModes: attrInt32,
}

// networkAutotuneKinds gives the documented value type of each network autotune
// preference attribute.
var networkAutotuneKinds = map[NetworkAutotunePreferenceAttributes]attrKind{
	NetworkAutotuneMaxIterations:     attrInt32,
	NetworkAutotuneIntermediateModes: attrInt32,
}

// checkKind reports an error unless attr is known to hold a value of kind want.
func checkKind(attr fmt.Stringer, got attrKind, known bool, want attrKind) error {
	if !known {
		return fmt.Errorf("cutensornet: unknown attribute %v", attr)
	}
	if got != want {
		return fmt.Errorf("cutensornet: attribute %v holds %v, not %v; use GetAttribute or SetAttribute with a buffer you size", attr, got, want)
	}
	return nil
}

// SetInt32 sets an int32-valued or enum-valued configuration attribute.
func (c ContractionOptimizerConfig) SetInt32(handle Handle, attr ContractionOptimizerConfigAttributes, v int32) error {
	kind, ok := optimizerConfigKinds[attr]
	if err := checkKind(attr, kind, ok, attrInt32); err != nil {
		return err
	}
	return c.SetAttribute(handle, attr, unsafe.Pointer(&v), 4)
}

// Int32 returns an int32-valued or enum-valued configuration attribute.
func (c ContractionOptimizerConfig) Int32(handle Handle, attr ContractionOptimizerConfigAttributes) (int32, error) {
	kind, ok := optimizerConfigKinds[attr]
	if err := checkKind(attr, kind, ok, attrInt32); err != nil {
		return 0, err
	}
	var v int32
	if err := c.GetAttribute(handle, attr, unsafe.Pointer(&v), 4); err != nil {
		return 0, err
	}
	return v, nil
}

// Int32 returns an int32-valued optimizer info attribute.
func (c ContractionOptimizerInfo) Int32(handle Handle, attr ContractionOptimizerInfoAttributes) (int32, error) {
	kind, ok := optimizerInfoKinds[attr]
	if err := checkKind(attr, kind, ok, attrInt32); err != nil {
		return 0, err
	}
	var v int32
	if err := c.GetAttribute(handle, attr, unsafe.Pointer(&v), 4); err != nil {
		return 0, err
	}
	return v, nil
}

// Int64 returns an int64-valued optimizer info attribute, such as the number of
// slices the path was split into.
func (c ContractionOptimizerInfo) Int64(handle Handle, attr ContractionOptimizerInfoAttributes) (int64, error) {
	kind, ok := optimizerInfoKinds[attr]
	if err := checkKind(attr, kind, ok, attrInt64); err != nil {
		return 0, err
	}
	var v int64
	if err := c.GetAttribute(handle, attr, unsafe.Pointer(&v), 8); err != nil {
		return 0, err
	}
	return v, nil
}

// SetInt64 sets an int64-valued optimizer info attribute.
func (c ContractionOptimizerInfo) SetInt64(handle Handle, attr ContractionOptimizerInfoAttributes, v int64) error {
	kind, ok := optimizerInfoKinds[attr]
	if err := checkKind(attr, kind, ok, attrInt64); err != nil {
		return err
	}
	return c.SetAttribute(handle, attr, unsafe.Pointer(&v), 8)
}

// Float64 returns a double-valued optimizer info attribute, such as the
// estimated FLOP count of the chosen path.
//
// The FLOP and runtime attributes are the library's own estimates for the path
// it selected. They are not measurements of anything this package ran.
func (c ContractionOptimizerInfo) Float64(handle Handle, attr ContractionOptimizerInfoAttributes) (float64, error) {
	kind, ok := optimizerInfoKinds[attr]
	if err := checkKind(attr, kind, ok, attrFloat64); err != nil {
		return 0, err
	}
	var v float64
	if err := c.GetAttribute(handle, attr, unsafe.Pointer(&v), 8); err != nil {
		return 0, err
	}
	return v, nil
}

// SetInt32 sets an int32-valued autotune preference attribute.
func (a ContractionAutotunePreference) SetInt32(handle Handle, attr ContractionAutotunePreferenceAttributes, v int32) error {
	kind, ok := autotuneKinds[attr]
	if err := checkKind(attr, kind, ok, attrInt32); err != nil {
		return err
	}
	return a.SetAttribute(handle, attr, unsafe.Pointer(&v), 4)
}

// Int32 returns an int32-valued autotune preference attribute.
func (a ContractionAutotunePreference) Int32(handle Handle, attr ContractionAutotunePreferenceAttributes) (int32, error) {
	kind, ok := autotuneKinds[attr]
	if err := checkKind(attr, kind, ok, attrInt32); err != nil {
		return 0, err
	}
	var v int32
	if err := a.GetAttribute(handle, attr, unsafe.Pointer(&v), 4); err != nil {
		return 0, err
	}
	return v, nil
}

// SetInt32 sets an int32-valued network autotune preference attribute.
func (n NetworkAutotunePreference) SetInt32(handle Handle, attr NetworkAutotunePreferenceAttributes, v int32) error {
	kind, ok := networkAutotuneKinds[attr]
	if err := checkKind(attr, kind, ok, attrInt32); err != nil {
		return err
	}
	return n.SetAttribute(handle, attr, unsafe.Pointer(&v), 4)
}

// Int32 returns an int32-valued network autotune preference attribute.
func (n NetworkAutotunePreference) Int32(handle Handle, attr NetworkAutotunePreferenceAttributes) (int32, error) {
	kind, ok := networkAutotuneKinds[attr]
	if err := checkKind(attr, kind, ok, attrInt32); err != nil {
		return 0, err
	}
	var v int32
	if err := n.GetAttribute(handle, attr, unsafe.Pointer(&v), 4); err != nil {
		return 0, err
	}
	return v, nil
}

// SetInt32 sets an int32-valued or enum-valued network attribute.
func (n *Network) SetInt32(attr NetworkAttributes, v int32) error {
	if n.closed {
		return ErrNetworkClosed
	}
	kind, ok := networkKinds[attr]
	if err := checkKind(attr, kind, ok, attrInt32); err != nil {
		return err
	}
	return n.h.NetworkSetAttribute(n.desc, attr, unsafe.Pointer(&v), 4)
}

// Int32 returns an int32-valued or enum-valued network attribute.
func (n *Network) Int32(attr NetworkAttributes) (int32, error) {
	if n.closed {
		return 0, ErrNetworkClosed
	}
	kind, ok := networkKinds[attr]
	if err := checkKind(attr, kind, ok, attrInt32); err != nil {
		return 0, err
	}
	var v int32
	if err := n.h.NetworkGetAttribute(n.desc, attr, unsafe.Pointer(&v), 4); err != nil {
		return 0, err
	}
	return v, nil
}
