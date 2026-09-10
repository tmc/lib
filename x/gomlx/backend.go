package gomlx

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"slices"
	"strings"
	"sync"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/notimplemented"
	"github.com/gomlx/compute/shapes"
	hlo "github.com/tmc/lib/x/stablehlo/builder"
	"github.com/tmc/lib/x/xla"
)

var unsupported = compute.ErrNotImplemented

// Backend executes GoMLX computations through TMC PJRT.
// Use New to construct a Backend; its zero value is closed.
// Runtime methods are safe for concurrent use and serialize native calls.
// A builder and its functions must be used by one goroutine at a time.
type Backend struct {
	mu sync.Mutex // Guards the engine and all native child resources.
	notimplemented.Backend
	engine      *xla.Engine
	compile     func([]byte, string, *xla.CompileOptions) (nativeExecutable, error)
	closeErr    error
	buffers     map[*buffer]struct{}
	executables map[*executable]struct{}
}

// New loads a PJRT plugin and exposes its first device.
// The plugin must match the current host architecture.
func New(path string) (*Backend, error) {
	e, err := xla.NewEngine(path)
	if err != nil {
		return nil, err
	}
	devices, err := e.Devices()
	if err != nil || len(devices) == 0 {
		return nil, errors.Join(fmt.Errorf("gomlx: no addressable devices"), err, e.Close())
	}
	b := &Backend{engine: e, buffers: make(map[*buffer]struct{}), executables: make(map[*executable]struct{})}
	b.compile = func(code []byte, format string, opts *xla.CompileOptions) (nativeExecutable, error) {
		return e.Compile(code, format, opts)
	}
	return b, nil
}
func (b *Backend) Name() string   { return "tmc-pjrt" }
func (b *Backend) String() string { return b.Name() }
func (b *Backend) DeviceDescription(device compute.DeviceNum) string {
	if device != 0 {
		return "invalid device"
	}
	return "PJRT device 0"
}
func (b *Backend) Description() string {
	return "TMC PJRT backend for static computations"
}

// Finalize implements compute.Backend. Use Close to observe cleanup errors.
func (b *Backend) Finalize() { _ = b.Close() }

// Close releases all live buffers and executables before closing the engine.
// Errors from previously finalized children are retained.
// Repeated calls return the retained cleanup error.
func (b *Backend) Close() error {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.engine != nil {
		for v := range b.buffers {
			_ = v.finalize()
		}
		for e := range b.executables {
			e.finalize()
		}
		b.closeErr = errors.Join(b.closeErr, b.engine.Close())
		b.engine = nil
	}
	return b.closeErr
}
func (b *Backend) IsFinalized() bool { b.mu.Lock(); defer b.mu.Unlock(); return b.engine == nil }
func (b *Backend) Capabilities() compute.Capabilities {
	return compute.Capabilities{Functions: true, Operations: map[compute.OpType]bool{compute.OpTypeFusedDense: true, compute.OpTypeFusedLayerNorm: true, compute.OpTypeFusedGelu: true, compute.OpTypeFusedSoftmax: true, compute.OpTypeFFT: true, compute.OpTypeComplex: true, compute.OpTypeConj: true, compute.OpTypeReal: true, compute.OpTypeImag: true, compute.OpTypeOptimizationBarrier: true, compute.OpTypeSchedulingBarrier: true, compute.OpTypeAtan2: true, compute.OpTypeShiftLeft: true, compute.OpTypeShiftRightArithmetic: true, compute.OpTypeShiftRightLogical: true, compute.OpTypeBitcast: true, compute.OpTypeNotEqual: true, compute.OpTypeEqualTotalOrder: true, compute.OpTypeNotEqualTotalOrder: true, compute.OpTypeLessThanTotalOrder: true, compute.OpTypeLessOrEqualTotalOrder: true, compute.OpTypeGreaterThanTotalOrder: true, compute.OpTypeGreaterOrEqualTotalOrder: true, compute.OpTypeBatchNormForInference: true, compute.OpTypeBatchNormForTraining: true, compute.OpTypeBatchNormGradient: true, compute.OpTypeDynamicSlice: true, compute.OpTypeLogistic: true, compute.OpTypeErf: true, compute.OpTypePad: true, compute.OpTypeReverse: true, compute.OpTypeReduceWindow: true, compute.OpTypeSelectAndScatterMax: true, compute.OpTypeSelectAndScatterMin: true, compute.OpTypeReduceMin: true, compute.OpTypeReduceBitwiseAnd: true, compute.OpTypeReduceBitwiseOr: true, compute.OpTypeReduceBitwiseXor: true, compute.OpTypeReduceLogicalAnd: true, compute.OpTypeReduceLogicalOr: true, compute.OpTypeReduceLogicalXor: true, compute.OpTypeExpm1: true, compute.OpTypeLog1p: true, compute.OpTypeCeil: true, compute.OpTypeFloor: true, compute.OpTypeRound: true, compute.OpTypeTanh: true, compute.OpTypeSign: true, compute.OpTypeIsFinite: true, compute.OpTypeIsNaN: true, compute.OpTypeBitwiseAnd: true, compute.OpTypeBitwiseOr: true, compute.OpTypeBitwiseXor: true, compute.OpTypeLogicalXor: true, compute.OpTypeBitCount: true, compute.OpTypeClz: true, compute.OpTypeArgMinMax: true, compute.OpTypeCumSum: true, compute.OpTypeClamp: true, compute.OpTypeRsqrt: true, compute.OpTypeLogicalNot: true, compute.OpTypeBitwiseNot: true, compute.OpTypeReduceProduct: true, compute.OpTypeSort: true, compute.OpTypeCall: true, compute.OpTypeIf: true, compute.OpTypeWhile: true, compute.OpTypeScatterMin: true, compute.OpTypeScatterMax: true, compute.OpTypeScatterSum: true, compute.OpTypeGather: true, compute.OpTypeRem: true, compute.OpTypeGreaterThan: true, compute.OpTypeGreaterOrEqual: true, compute.OpTypeLogicalOr: true, compute.OpTypeDynamicUpdateSlice: true, compute.OpTypeSlice: true, compute.OpTypeConcatenate: true, compute.OpTypeSin: true, compute.OpTypeSub: true, compute.OpTypeDiv: true, compute.OpTypeConvertDType: true, compute.OpTypeSqrt: true, compute.OpTypePow: true, compute.OpTypeExp: true, compute.OpTypeNeg: true, compute.OpTypeIdentity: true, compute.OpTypeEqual: true, compute.OpTypeLessThan: true, compute.OpTypeLogicalAnd: true, compute.OpTypeWhere: true, compute.OpTypeLessOrEqual: true, compute.OpTypeIota: true, compute.OpTypeRNGBitGenerator: true, compute.OpTypeAbs: true, compute.OpTypeCos: true, compute.OpTypeMin: true, compute.OpTypeMax: true, compute.OpTypeTranspose: true, compute.OpTypeLog: true, compute.OpTypeReduceMax: true, compute.OpTypeMul: true, compute.OpTypeBroadcastInDim: true, compute.OpTypeReshape: true, compute.OpTypeReduceSum: true, compute.OpTypeParameter: true, compute.OpTypeConstant: true, compute.OpTypeAdd: true, compute.OpTypeDotGeneral: true}, DTypes: map[dtypes.DType]bool{dtypes.Complex64: true, dtypes.Complex128: true, dtypes.Float16: true, dtypes.BFloat16: true, dtypes.Float64: true, dtypes.Bool: true, dtypes.Uint8: true, dtypes.Uint16: true, dtypes.Uint32: true, dtypes.Uint64: true, dtypes.Float32: true, dtypes.Int8: true, dtypes.Int16: true, dtypes.Int32: true, dtypes.Int64: true}}
}
func (b *Backend) Builder(name string) compute.Builder {
	v := &builder{backend: b, name: name}
	v.main = &function{owner: v, name: "main", raw: hlo.NewFunc("main"), valueShapes: make(map[string]shapes.Shape)}
	v.functions = []*function{v.main}
	v.main.Function.ErrFn = func(op compute.OpType) error { return fmt.Errorf("%s: %w", op, unsupported) }
	return v
}
func validate(s shapes.Shape) error {
	for _, name := range s.AxisNames {
		if name != "" {
			return unsupported
		}
	}
	info, ok := element(s.DType)
	if !ok {
		return unsupported
	}
	if len(s.TupleShapes) > 0 {
		return unsupported
	}
	zero := false
	for _, d := range s.Dimensions {
		if d < 0 {
			return fmt.Errorf("negative dimension")
		}
		zero = zero || d == 0
	}
	if zero {
		return nil
	}
	n := 1
	for _, d := range s.Dimensions {
		if n > int(^uint(0)>>1)/info.size/d {
			return fmt.Errorf("shape byte count overflows int")
		}
		n *= d
	}
	return nil
}

func validateGraphShape(s shapes.Shape) error {
	if _, ok := element(s.DType); !ok || len(s.TupleShapes) > 0 {
		return unsupported
	}
	for axis, dim := range s.Dimensions {
		name := s.AxisName(axis)
		if dim == shapes.DynamicDim {
			if name == "" {
				return fmt.Errorf("dynamic axis %d has no name", axis)
			}
			continue
		}
		if dim < 0 {
			return fmt.Errorf("negative dimension")
		}
		if name != "" {
			return fmt.Errorf("static axis %d has name %q", axis, name)
		}
	}
	if !s.IsDynamic() {
		return validate(s)
	}
	return nil
}
func (b *Backend) BufferFromFlatData(device compute.DeviceNum, flat any, s shapes.Shape) (compute.Buffer, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.bufferFromFlatData(device, flat, s)
}
func (b *Backend) bufferFromFlatData(device compute.DeviceNum, flat any, s shapes.Shape) (compute.Buffer, error) {
	if b.engine == nil {
		return nil, fmt.Errorf("backend closed")
	}
	if device != 0 {
		return nil, unsupported
	}
	if err := validate(s); err != nil {
		return nil, err
	}
	dtype, n := flatShape(flat)
	if dtype != s.DType || n != s.Size() {
		return nil, fmt.Errorf("flat data mismatch")
	}
	info, _ := element(dtype)
	raw := make([]byte, n*info.size)
	encodeFlat(flat, raw)
	devices, err := b.engine.Devices()
	if err != nil {
		return nil, err
	}
	dims := make([]int64, len(s.Dimensions))
	for i, v := range s.Dimensions {
		dims[i] = int64(v)
	}
	v, err := b.engine.NewBufferFromHost(raw, info.pjrt, dims, devices[0])
	if err != nil {
		return nil, err
	}
	out := &buffer{backend: b, raw: v, shape: s.Clone()}
	b.buffers[out] = struct{}{}
	return out, nil
}

type buffer struct {
	backend *Backend
	raw     *xla.Buffer
	shape   shapes.Shape
}

func (b *buffer) Backend() compute.Backend { return b.backend }
func (b *buffer) Finalize() error {
	b.backend.mu.Lock()
	defer b.backend.mu.Unlock()
	return b.finalize()
}
func (b *buffer) finalize() error {
	if b.raw == nil {
		return nil
	}
	v := b.raw
	b.raw = nil
	delete(b.backend.buffers, b)
	err := v.Close()
	b.backend.closeErr = errors.Join(b.backend.closeErr, err)
	return err
}
func (b *buffer) Shape() (shapes.Shape, error) {
	b.backend.mu.Lock()
	defer b.backend.mu.Unlock()
	if b.raw == nil {
		return shapes.Shape{}, fmt.Errorf("buffer closed")
	}
	return b.shape.Clone(), nil
}
func (b *buffer) DeviceNum() (compute.DeviceNum, error) {
	b.backend.mu.Lock()
	defer b.backend.mu.Unlock()
	if b.raw == nil {
		return 0, fmt.Errorf("buffer closed")
	}
	return 0, nil
}
func (b *buffer) Data() (any, error) { return nil, unsupported }
func (b *buffer) ToFlatData(flat any) error {
	b.backend.mu.Lock()
	defer b.backend.mu.Unlock()
	return b.toFlatData(flat)
}
func (b *buffer) toFlatData(flat any) error {
	dtype, n := flatShape(flat)
	if dtype != b.shape.DType || n != b.shape.Size() {
		return fmt.Errorf("flat data mismatch")
	}
	if b.raw == nil {
		return fmt.Errorf("buffer closed")
	}
	info, _ := element(dtype)
	raw := make([]byte, n*info.size)
	if err := b.raw.ToHost(raw); err != nil {
		return err
	}
	decodeFlat(flat, raw)
	return nil
}
func (b *buffer) CopyToDevice(device compute.DeviceNum) (compute.Buffer, error) {
	b.backend.mu.Lock()
	defer b.backend.mu.Unlock()
	if device != 0 {
		return nil, unsupported
	}
	v := newFlat(b.shape.DType, b.shape.Size())
	if err := b.toFlatData(v); err != nil {
		return nil, err
	}
	return b.backend.bufferFromFlatData(device, v, b.shape)
}

type builder struct {
	notimplemented.Builder
	backend   *Backend
	name      string
	main      *function
	functions []*function
	closed    bool
}

func (b *builder) Name() string           { return b.name }
func (b *builder) Main() compute.Function { return b.main }

func (b *builder) DeviceAssignment(d ...compute.DeviceNum) error {
	if len(d) != 1 || d[0] != 0 {
		return unsupported
	}
	return nil
}
func (b *builder) OpShape(v compute.Value) (shapes.Shape, error) {
	n, ok := v.(*node)
	if !ok || n == nil || n.owner == nil || n.owner.owner != b {
		return shapes.Shape{}, fmt.Errorf("invalid value")
	}
	return n.owner.Shape(v)
}
func (b *builder) Compile() (compute.Executable, error) {
	b.backend.mu.Lock()
	defer b.backend.mu.Unlock()
	if b.closed || !b.main.returned || b.backend.engine == nil {
		return nil, fmt.Errorf("builder unavailable")
	}
	b.closed = true
	if slices.ContainsFunc(b.main.inputs, shapes.Shape.IsDynamic) || slices.ContainsFunc(b.main.dynamicOps, func(op dynamicOp) bool { return strings.HasSuffix(op.kind, "_dynamic") }) {
		if err := b.main.checkDynamicSlice(); err != nil {
			return nil, err
		}
		template := b.main.snapshot()
		out := &executable{owner: b.backend, template: template, names: slices.Clone(b.main.names), inputs: cloneShapes(b.main.inputs), outputs: cloneShapes(b.main.outputs), specializations: make(map[string]*specialization), maxSpecializations: 32}
		b.backend.executables[out] = struct{}{}
		return out, nil
	}
	var text strings.Builder
	text.WriteString("module {\n")
	for _, f := range b.functions {
		if !f.returned {
			return nil, fmt.Errorf("function %q has not returned", f.name)
		}
		var body strings.Builder
		raw := *f.raw
		raw.Args = append(slices.Clone(raw.Args), f.captureArgs...)
		if err := raw.Emit(&body); err != nil {
			return nil, err
		}
		code := body.String()
		if f != b.main {
			code = strings.Replace(code, "func.func @", "func.func private @", 1)
		}
		text.WriteString(code)
	}
	text.WriteString("}\n")
	e, err := b.backend.compileProgram([]byte(text.String()), "mlir", nil)
	if err != nil {
		return nil, err
	}
	out := &executable{owner: b.backend, raw: e, names: b.main.names, inputs: b.main.inputs, outputs: b.main.outputs}
	b.backend.executables[out] = struct{}{}
	return out, nil
}

func (f *function) snapshot() *function {
	copy := *f
	raw := *f.raw
	raw.Args = cloneValues(f.raw.Args)
	raw.Ops = slices.Clone(f.raw.Ops)
	raw.Returns = cloneValues(f.raw.Returns)
	copy.raw = &raw
	copy.names = slices.Clone(f.names)
	copy.inputs = cloneShapes(f.inputs)
	copy.outputs = cloneShapes(f.outputs)
	copy.dynamicOps = slices.Clone(f.dynamicOps)
	for i, op := range copy.dynamicOps {
		copy.dynamicOps[i].operand = cloneValue(op.operand)
		copy.dynamicOps[i].result = cloneValue(op.result)
		copy.dynamicOps[i].operands = cloneValues(op.operands)
		copy.dynamicOps[i].axes = slices.Clone(op.axes)
		copy.dynamicOps[i].starts = slices.Clone(op.starts)
		copy.dynamicOps[i].limits = slices.Clone(op.limits)
		copy.dynamicOps[i].strides = slices.Clone(op.strides)
		copy.dynamicOps[i].dimensions = slices.Clone(op.dimensions)
		copy.dynamicOps[i].pads = slices.Clone(op.pads)
		for j := range op.dimensions {
			copy.dynamicOps[i].dimensions[j].value = cloneValue(op.dimensions[j].value)
		}
		for j, pad := range op.pads {
			copy.dynamicOps[i].pads[j].start.value = cloneValue(pad.start.value)
			copy.dynamicOps[i].pads[j].end.value = cloneValue(pad.end.value)
			copy.dynamicOps[i].pads[j].interior.value = cloneValue(pad.interior.value)
		}
	}
	copy.valueShapes = make(map[string]shapes.Shape, len(f.valueShapes))
	for id, shape := range f.valueShapes {
		copy.valueShapes[id] = shape.Clone()
	}
	return &copy
}

func cloneValue(v hlo.Value) hlo.Value {
	v.Type.Shape = slices.Clone(v.Type.Shape)
	return v
}

func cloneValues(values []hlo.Value) []hlo.Value {
	out := make([]hlo.Value, len(values))
	for i, v := range values {
		out[i] = cloneValue(v)
	}
	return out
}

type node struct {
	owner *function
	value hlo.Value
}
type function struct {
	notimplemented.Function
	owner           *builder
	raw             *hlo.FuncBuilder
	name            string
	names           []string
	inputs, outputs []shapes.Shape
	returned        bool
	parent          *function
	captures        []*node
	captureArgs     []hlo.Value
	dynamicOps      []dynamicOp
	valueShapes     map[string]shapes.Shape
}

func (f *function) Builder() compute.Builder { return f.owner }
func (f *function) Parameter(name string, s shapes.Shape, sharding *compute.ShardingSpec) (compute.Value, error) {
	if f.owner.closed || f.returned {
		return nil, fmt.Errorf("function closed")
	}
	if sharding != nil {
		return nil, unsupported
	}
	if err := validateGraphShape(s); err != nil {
		return nil, err
	}
	for _, n := range f.names {
		if n == name {
			return nil, fmt.Errorf("duplicate parameter")
		}
	}
	dims := make([]int64, len(s.Dimensions))
	for i, v := range s.Dimensions {
		dims[i] = int64(v)
	}
	info, _ := element(s.DType)
	v := hlo.Value{ID: fmt.Sprintf("%%arg%d", len(f.raw.Args)), Type: hlo.Type{Shape: dims, ElementType: info.hlo}}
	f.raw.Args = append(f.raw.Args, v)
	f.names = append(f.names, name)
	f.inputs = append(f.inputs, s.Clone())
	if f.valueShapes == nil {
		f.valueShapes = make(map[string]shapes.Shape)
	}
	f.valueShapes[v.ID] = s.Clone()
	return &node{f, v}, nil
}
func (f *function) value(v compute.Value) (hlo.Value, error) {
	n, ok := v.(*node)
	if !ok || n == nil || n.owner == nil || f.owner.closed {
		return hlo.Value{}, fmt.Errorf("invalid value")
	}
	if n.owner == f {
		return n.value, nil
	}
	visible := false
	for p := f.parent; p != nil; p = p.parent {
		if p == n.owner {
			visible = true
			break
		}
	}
	if !visible {
		return hlo.Value{}, fmt.Errorf("value is outside lexical scope")
	}
	for i, c := range f.captures {
		if c == n {
			return f.captureArgs[i], nil
		}
	}
	if f.returned {
		return hlo.Value{}, fmt.Errorf("function closed")
	}
	// Register outer captures through each intervening lexical scope.
	if _, err := f.parent.value(n); err != nil {
		return hlo.Value{}, err
	}
	captured := hlo.Value{ID: fmt.Sprintf("%%capture%d", len(f.captures)), Type: n.value.Type}
	f.captures = append(f.captures, n)
	f.captureArgs = append(f.captureArgs, captured)
	return captured, nil
}
func (f *function) Shape(v compute.Value) (shapes.Shape, error) {
	n, err := f.value(v)
	if err != nil {
		return shapes.Shape{}, err
	}
	if shape, ok := f.valueShapes[n.ID]; ok {
		return shape.Clone(), nil
	}
	dims := make([]int, len(n.Type.Shape))
	for i, v := range n.Type.Shape {
		dims[i] = int(v)
	}
	return shapes.Shape{DType: elementDType(n.Type.ElementType), Dimensions: dims}, nil
}
func (f *function) Return(values []compute.Value, shardings []*compute.ShardingSpec) error {
	if f.returned || f.owner.closed {
		return fmt.Errorf("function closed")
	}
	if len(shardings) > 0 {
		return unsupported
	}
	raw := []hlo.Value{}
	ss := []shapes.Shape{}
	for _, v := range values {
		n, err := f.value(v)
		if err != nil {
			return err
		}
		s, err := f.Shape(v)
		if err != nil {
			return err
		}
		raw = append(raw, n)
		ss = append(ss, s)
	}
	f.raw.Return(raw...)
	f.outputs = ss
	f.returned = true
	return nil
}
func (f *function) Add(a, b compute.Value) (compute.Value, error) {
	return f.binaryRecord(a, b, f.raw.Add, "add")
}

type executable struct {
	owner              *Backend
	raw                nativeExecutable
	names              []string
	inputs, outputs    []shapes.Shape
	closeErr           error
	template           *function
	specializations    map[string]*specialization
	maxSpecializations int
	clock              uint64
	compileCalls       int
	finalized          bool
}

type nativeExecutable interface {
	Execute(context.Context, []*xla.Buffer) ([]*xla.Buffer, error)
	Close() error
}

type specialization struct {
	raw             nativeExecutable
	inputs, outputs []shapes.Shape
	lastUsed        uint64
}

func (e *executable) Finalize() {
	e.owner.mu.Lock()
	defer e.owner.mu.Unlock()
	e.finalize()
}
func (e *executable) finalize() {
	if e.finalized {
		return
	}
	e.finalized = true
	for key, spec := range e.specializations {
		e.closeErr = errors.Join(e.closeErr, spec.raw.Close())
		delete(e.specializations, key)
	}
	if e.raw != nil {
		e.closeErr = errors.Join(e.closeErr, e.raw.Close())
		e.raw = nil
	}
	e.template = nil
	delete(e.owner.executables, e)
	e.owner.closeErr = errors.Join(e.owner.closeErr, e.closeErr)
}
func cloneShapes(ss []shapes.Shape) []shapes.Shape {
	out := make([]shapes.Shape, len(ss))
	for i, s := range ss {
		out[i] = s.Clone()
	}
	return out
}
func (e *executable) Inputs() ([]string, []shapes.Shape) {
	return slices.Clone(e.names), cloneShapes(e.inputs)
}
func (e *executable) Outputs() []shapes.Shape { return cloneShapes(e.outputs) }
func (e *executable) Execute(inputs []compute.Buffer, donate []bool, device compute.DeviceNum) (outputs []compute.Buffer, resultErr error) {
	e.owner.mu.Lock()
	defer e.owner.mu.Unlock()
	if e.raw == nil && e.template == nil || e.owner.engine == nil {
		return nil, fmt.Errorf("executable closed")
	}
	if device != 0 {
		return nil, unsupported
	}
	if len(inputs) != len(e.inputs) || len(donate) > 0 && len(donate) != len(inputs) {
		return nil, fmt.Errorf("input count mismatch")
	}
	expectedInputs, expectedOutputs, raw := e.inputs, e.outputs, e.raw
	if e.template != nil {
		bindings := make(shapes.AxisBindings)
		if len(inputs) != len(e.inputs) {
			return nil, fmt.Errorf("input count mismatch")
		}
		for i, input := range inputs {
			buf, ok := input.(*buffer)
			if !ok || buf.backend != e.owner || buf.raw == nil {
				return nil, fmt.Errorf("invalid input")
			}
			if buf.shape.DType != e.inputs[i].DType {
				return nil, fmt.Errorf("bind input %q: dtype mismatch", e.names[i])
			}
			if err := bindings.Extract(e.inputs[i], buf.shape); err != nil {
				return nil, fmt.Errorf("bind input %q: %w", e.names[i], err)
			}
		}
		nativeInputs := make([]*xla.Buffer, len(inputs))
		for i, input := range inputs {
			nativeInputs[i] = input.(*buffer).raw
		}
		evaluate := func(raw *hlo.FuncBuilder, value hlo.Value) (int64, error) {
			return e.owner.evaluateDimension(raw, value, nativeInputs)
		}
		program, concreteInputs, concreteOutputs, err := e.template.specializeRuntime(bindings, evaluate)
		if err != nil {
			return nil, err
		}
		key := fmt.Sprintf("%s:%x", bindings.Key(), sha256.Sum256([]byte(program)))
		spec := e.specializations[key]
		if spec == nil {
			e.compileCalls++
			compiled, err := e.owner.compileProgram([]byte(program), "mlir", nil)
			if err != nil {
				return nil, err
			}
			if len(e.specializations) >= e.maxSpecializations {
				var oldestKey string
				var oldest uint64 = ^uint64(0)
				for k, candidate := range e.specializations {
					if candidate.lastUsed < oldest {
						oldestKey, oldest = k, candidate.lastUsed
					}
				}
				old := e.specializations[oldestKey]
				delete(e.specializations, oldestKey)
				if err := old.raw.Close(); err != nil {
					return nil, errors.Join(err, compiled.Close())
				}
			}
			spec = &specialization{raw: compiled, inputs: concreteInputs, outputs: concreteOutputs}
			e.specializations[key] = spec
		}
		e.clock++
		spec.lastUsed = e.clock
		raw, expectedInputs, expectedOutputs = spec.raw, spec.inputs, spec.outputs
	}
	args := make([]*xla.Buffer, len(inputs))
	for i, v := range inputs {
		b, ok := v.(*buffer)
		if !ok || b.backend != e.owner || b.raw == nil || b.shape.DType != expectedInputs[i].DType || !b.shape.EqualDimensions(expectedInputs[i]) {
			return nil, fmt.Errorf("invalid input")
		}
		args[i] = b.raw
	}
	// Donation transfers ownership; native storage reuse is not required.
	defer func() {
		var closeErr error
		for i, d := range donate {
			if d {
				closeErr = errors.Join(closeErr, inputs[i].(*buffer).finalize())
			}
		}
		if closeErr != nil {
			for _, out := range outputs {
				closeErr = errors.Join(closeErr, out.(*buffer).finalize())
			}
			outputs = nil
			resultErr = errors.Join(resultErr, closeErr)
		}
	}()
	rawOutputs, err := raw.Execute(context.Background(), args)
	if err != nil {
		return nil, err
	}
	fail := func(err error) ([]compute.Buffer, error) {
		for _, v := range rawOutputs {
			err = errors.Join(err, v.Close())
		}
		return nil, err
	}
	if len(rawOutputs) != len(expectedOutputs) {
		return fail(fmt.Errorf("output count mismatch"))
	}
	for i, v := range rawOutputs {
		dtype, dims, err := v.Shape()
		if err != nil {
			return fail(err)
		}
		info, _ := element(expectedOutputs[i].DType)
		if dtype != info.pjrt || len(dims) != len(expectedOutputs[i].Dimensions) {
			return fail(fmt.Errorf("output shape mismatch"))
		}
		for j, d := range dims {
			if d != int64(expectedOutputs[i].Dimensions[j]) {
				return fail(fmt.Errorf("output shape mismatch"))
			}
		}
	}
	out := make([]compute.Buffer, len(rawOutputs))
	for i, v := range rawOutputs {
		buf := &buffer{e.owner, v, expectedOutputs[i].Clone()}
		e.owner.buffers[buf] = struct{}{}
		out[i] = buf
	}
	return out, nil
}

func (b *Backend) compileProgram(code []byte, format string, opts *xla.CompileOptions) (nativeExecutable, error) {
	if b.compile != nil {
		return b.compile(code, format, opts)
	}
	return b.engine.Compile(code, format, opts)
}

var _ compute.Backend = (*Backend)(nil)
var _ compute.Buffer = (*buffer)(nil)
var _ compute.Builder = (*builder)(nil)
var _ compute.Function = (*function)(nil)
var _ compute.Executable = (*executable)(nil)

func int64s(v []int) []int64 {
	out := make([]int64, len(v))
	for i, n := range v {
		out[i] = int64(n)
	}
	return out
}
func contractionShape(a, b []int64, ac, ab, bc, bb []int) ([]int64, error) {
	if len(ac) != len(bc) || len(ab) != len(bb) {
		return nil, fmt.Errorf("axis count mismatch")
	}
	used := func(rank int, groups ...[]int) ([]bool, error) {
		seen := make([]bool, rank)
		for _, group := range groups {
			for _, i := range group {
				if i < 0 || i >= rank || seen[i] {
					return nil, fmt.Errorf("invalid or repeated axis %d", i)
				}
				seen[i] = true
			}
		}
		return seen, nil
	}
	au, err := used(len(a), ac, ab)
	if err != nil {
		return nil, err
	}
	bu, err := used(len(b), bc, bb)
	if err != nil {
		return nil, err
	}
	for _, pair := range [][2][]int{{ac, bc}, {ab, bb}} {
		for i, j := range pair[0] {
			if a[j] != b[pair[1][i]] {
				return nil, fmt.Errorf("paired dimensions differ")
			}
		}
	}
	var out []int64
	for _, i := range ab {
		out = append(out, a[i])
	}
	for i, n := range a {
		if !au[i] {
			out = append(out, n)
		}
	}
	for i, n := range b {
		if !bu[i] {
			out = append(out, n)
		}
	}
	return out, nil
}
func (f *function) broadcastPair(a, b hlo.Value) (hlo.Value, hlo.Value, error) {
	if a.Type.ElementType != b.Type.ElementType {
		return hlo.Value{}, hlo.Value{}, fmt.Errorf("operand dtypes mismatch: %s vs %s", a.Type.ElementType, b.Type.ElementType)
	}
	shape, err := broadcastShape(f.valueShape(a), f.valueShape(b))
	if err != nil {
		return hlo.Value{}, hlo.Value{}, err
	}
	dims := int64s(shape.Dimensions)
	rank := len(dims)
	expand := func(v hlo.Value) (hlo.Value, error) {
		if slices.Equal(v.Type.Shape, dims) {
			return v, nil
		}
		axes := make([]int64, len(v.Type.Shape))
		for i := range axes {
			axes[i] = int64(rank - len(axes) + i)
		}
		result, err := f.raw.BroadcastInDim(v, axes, hlo.Type{Shape: dims, ElementType: v.Type.ElementType})
		if err != nil {
			return hlo.Value{}, err
		}
		f.dynamicOps = append(f.dynamicOps, dynamicOp{kind: "broadcast_pair", operand: v, operands: []hlo.Value{a, b}, axes: ints(axes), result: result})
		f.valueShapes[result.ID] = shape.Clone()
		return result, nil
	}
	x, err := expand(a)
	if err != nil {
		return hlo.Value{}, hlo.Value{}, err
	}
	y, err := expand(b)
	return x, y, err
}

func (f *function) Constant(flat any, dims ...int) (compute.Value, error) {
	if f.owner.closed || f.returned {
		return nil, fmt.Errorf("function closed")
	}
	dtype, n := flatShape(flat)
	shape := shapes.Shape{DType: dtype, Dimensions: dims}
	if err := validate(shape); err != nil {
		return nil, err
	}
	if n != shape.Size() {
		return nil, fmt.Errorf("constant data length mismatch")
	}
	info, _ := element(dtype)
	var text strings.Builder
	index := 0
	var emit func([]int)
	emit = func(dims []int) {
		if len(dims) == 0 {
			text.WriteString(constantLiteral(flat, index))
			index++
			return
		}
		text.WriteByte('[')
		for i := range dims[0] {
			if i > 0 {
				text.WriteString(", ")
			}
			emit(dims[1:])
		}
		text.WriteByte(']')
	}
	if n > 0 {
		emit(dims)
	}
	v := hlo.Value{ID: fmt.Sprintf("%%constant%d", len(f.raw.Ops)), Type: hlo.Type{Shape: int64s(dims), ElementType: info.hlo}}
	f.raw.Ops = append(f.raw.Ops, fmt.Sprintf("%s = stablehlo.constant dense<%s> : %s", v.ID, text.String(), v.Type))
	f.dynamicOps = append(f.dynamicOps, dynamicOp{kind: "literal", literal: text.String(), result: v})
	f.valueShapes[v.ID] = shape.Clone()
	return &node{f, v}, nil
}
