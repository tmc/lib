package frontend

import (
	"encoding/binary"
	"fmt"
	"math"
	"slices"

	"github.com/tmc/lib/nvidia/x/egraph/hopper"
)

// Protobuf wire types.
const (
	wireVarint = 0
	wireI64    = 1
	wireLen    = 2
	wireI32    = 5
)

// A wireReader reads protobuf wire format. Errors carry the absolute byte
// offset of the field that failed.
type wireReader struct {
	b    []byte
	off  int
	base int
}

func (r *wireReader) done() bool { return r.off >= len(r.b) }

func (r *wireReader) errorf(format string, args ...any) error {
	return fmt.Errorf("offset %d: %s", r.base+r.off, fmt.Sprintf(format, args...))
}

func (r *wireReader) varint() (uint64, error) {
	v, n := binary.Uvarint(r.b[r.off:])
	if n <= 0 {
		return 0, r.errorf("bad varint")
	}
	r.off += n
	return v, nil
}

// next reads a field tag.
func (r *wireReader) next() (field int, wt int, err error) {
	start := r.off
	tag, err := r.varint()
	if err != nil {
		return 0, 0, err
	}
	field, wt = int(tag>>3), int(tag&7)
	if field == 0 || field > 1<<29-1 || wt == 3 || wt == 4 || wt > 5 {
		r.off = start
		return 0, 0, r.errorf("invalid tag %#x (field %d, wire type %d)", tag, field, wt)
	}
	return field, wt, nil
}

// bytes reads a length-delimited field and returns a reader over it.
func (r *wireReader) bytes() (*wireReader, error) {
	n, err := r.varint()
	if err != nil {
		return nil, err
	}
	if n > uint64(len(r.b)-r.off) {
		return nil, r.errorf("length %d overruns the message", n)
	}
	sub := &wireReader{b: r.b[r.off : r.off+int(n)], base: r.base + r.off}
	r.off += int(n)
	return sub, nil
}

func (r *wireReader) fixed32() (uint32, error) {
	if len(r.b)-r.off < 4 {
		return 0, r.errorf("truncated fixed32")
	}
	v := binary.LittleEndian.Uint32(r.b[r.off:])
	r.off += 4
	return v, nil
}

func (r *wireReader) skip(wt int) error {
	switch wt {
	case wireVarint:
		_, err := r.varint()
		return err
	case wireI64:
		if len(r.b)-r.off < 8 {
			return r.errorf("truncated fixed64")
		}
		r.off += 8
		return nil
	case wireLen:
		_, err := r.bytes()
		return err
	case wireI32:
		_, err := r.fixed32()
		return err
	}
	return r.errorf("cannot skip wire type %d", wt)
}

// int64s reads a repeated int64 field, packed or not.
func (r *wireReader) int64s(wt int, dst []int64) ([]int64, error) {
	if wt == wireVarint {
		v, err := r.varint()
		return append(dst, int64(v)), err
	}
	if wt != wireLen {
		return dst, r.errorf("int64 field with wire type %d", wt)
	}
	sub, err := r.bytes()
	if err != nil {
		return dst, err
	}
	for !sub.done() {
		v, err := sub.varint()
		if err != nil {
			return dst, err
		}
		dst = append(dst, int64(v))
	}
	return dst, nil
}

// float32s reads a repeated float field, packed or not.
func (r *wireReader) float32s(wt int, dst []float32) ([]float32, error) {
	if wt == wireI32 {
		v, err := r.fixed32()
		return append(dst, math.Float32frombits(v)), err
	}
	if wt != wireLen {
		return dst, r.errorf("float field with wire type %d", wt)
	}
	sub, err := r.bytes()
	if err != nil {
		return dst, err
	}
	for !sub.done() {
		v, err := sub.fixed32()
		if err != nil {
			return dst, err
		}
		dst = append(dst, math.Float32frombits(v))
	}
	return dst, nil
}

func (r *wireReader) str(wt int) (string, error) {
	if wt != wireLen {
		return "", r.errorf("string field with wire type %d", wt)
	}
	sub, err := r.bytes()
	if err != nil {
		return "", err
	}
	return string(sub.b), nil
}

func (r *wireReader) uvarint(wt int) (uint64, error) {
	if wt != wireVarint {
		return 0, r.errorf("varint field with wire type %d", wt)
	}
	return r.varint()
}

// The ONNX messages and fields read (onnx/onnx.proto):
//
//	ModelProto      graph=7
//	GraphProto      node=1 initializer=5 input=11 output=12
//	NodeProto       input=1 output=2 name=3 op_type=4 attribute=5 domain=7
//	AttributeProto  name=1 f=2 i=3 s=4 t=5 floats=7 ints=8
//	TensorProto     dims=1 data_type=2 float_data=4 int64_data=7 name=8 raw_data=9
//	ValueInfoProto  name=1 type=2
//	TypeProto       tensor_type=1
//	TypeProto.Tensor elem_type=1 shape=2
//	TensorShapeProto dim=1; Dimension dim_value=1 dim_param=2
type onnxGraph struct {
	nodes        []onnxNode
	initializers []onnxTensor
	inputs       []onnxValueInfo
	outputs      []onnxValueInfo
}

type onnxNode struct {
	off          int
	inputs       []string
	outputs      []string
	name, opType string
	domain       string
	attrs        map[string]onnxAttr
}

type onnxAttr struct {
	f    float32
	i    int64
	s    string
	ints []int64
	t    *onnxTensor
	hasF bool
	hasI bool
}

type onnxTensor struct {
	off      int
	name     string
	dims     []int64
	dataType int
	floats   []float32
	int64s   []int64
	raw      []byte
}

type onnxValueInfo struct {
	off      int
	name     string
	elemType int
	dims     []int64
	symbolic bool
}

func readModel(b []byte) (*onnxGraph, error) {
	r := &wireReader{b: b}
	var g *onnxGraph
	for !r.done() {
		f, wt, err := r.next()
		if err != nil {
			return nil, err
		}
		if f == 7 && wt == wireLen {
			sub, err := r.bytes()
			if err != nil {
				return nil, err
			}
			if g, err = readGraph(sub); err != nil {
				return nil, err
			}
			continue
		}
		if err := r.skip(wt); err != nil {
			return nil, err
		}
	}
	if g == nil {
		return nil, fmt.Errorf("model has no graph")
	}
	return g, nil
}

func readGraph(r *wireReader) (*onnxGraph, error) {
	g := &onnxGraph{}
	for !r.done() {
		f, wt, err := r.next()
		if err != nil {
			return nil, err
		}
		if wt != wireLen || (f != 1 && f != 5 && f != 11 && f != 12) {
			if err := r.skip(wt); err != nil {
				return nil, err
			}
			continue
		}
		sub, err := r.bytes()
		if err != nil {
			return nil, err
		}
		switch f {
		case 1:
			n, err := readNode(sub)
			if err != nil {
				return nil, err
			}
			g.nodes = append(g.nodes, n)
		case 5:
			t, err := readTensor(sub)
			if err != nil {
				return nil, err
			}
			g.initializers = append(g.initializers, t)
		case 11, 12:
			v, err := readValueInfo(sub)
			if err != nil {
				return nil, err
			}
			if f == 11 {
				g.inputs = append(g.inputs, v)
			} else {
				g.outputs = append(g.outputs, v)
			}
		}
	}
	return g, nil
}

func readNode(r *wireReader) (onnxNode, error) {
	n := onnxNode{off: r.base, attrs: make(map[string]onnxAttr)}
	for !r.done() {
		f, wt, err := r.next()
		if err != nil {
			return n, err
		}
		var s string
		switch f {
		case 1, 2, 3, 4, 7:
			if s, err = r.str(wt); err != nil {
				return n, err
			}
		}
		switch f {
		case 1:
			n.inputs = append(n.inputs, s)
		case 2:
			n.outputs = append(n.outputs, s)
		case 3:
			n.name = s
		case 4:
			n.opType = s
		case 7:
			n.domain = s
		case 5:
			if wt != wireLen {
				return n, r.errorf("attribute with wire type %d", wt)
			}
			sub, err := r.bytes()
			if err != nil {
				return n, err
			}
			name, a, err := readAttr(sub)
			if err != nil {
				return n, err
			}
			n.attrs[name] = a
		default:
			if err := r.skip(wt); err != nil {
				return n, err
			}
		}
	}
	return n, nil
}

func readAttr(r *wireReader) (string, onnxAttr, error) {
	var name string
	var a onnxAttr
	for !r.done() {
		f, wt, err := r.next()
		if err != nil {
			return "", a, err
		}
		switch f {
		case 1:
			name, err = r.str(wt)
		case 2:
			if wt != wireI32 {
				return "", a, r.errorf("float attribute with wire type %d", wt)
			}
			var v uint32
			v, err = r.fixed32()
			a.f, a.hasF = math.Float32frombits(v), true
		case 3:
			var v uint64
			v, err = r.uvarint(wt)
			a.i, a.hasI = int64(v), true
		case 4:
			a.s, err = r.str(wt)
		case 5:
			if wt != wireLen {
				return "", a, r.errorf("tensor attribute with wire type %d", wt)
			}
			var sub *wireReader
			if sub, err = r.bytes(); err == nil {
				var t onnxTensor
				t, err = readTensor(sub)
				a.t = &t
			}
		case 8:
			a.ints, err = r.int64s(wt, a.ints)
		default:
			err = r.skip(wt)
		}
		if err != nil {
			return "", a, err
		}
	}
	return name, a, nil
}

func readTensor(r *wireReader) (onnxTensor, error) {
	t := onnxTensor{off: r.base}
	for !r.done() {
		f, wt, err := r.next()
		if err != nil {
			return t, err
		}
		switch f {
		case 1:
			t.dims, err = r.int64s(wt, t.dims)
		case 2:
			var v uint64
			v, err = r.uvarint(wt)
			t.dataType = int(v)
		case 4:
			t.floats, err = r.float32s(wt, t.floats)
		case 7:
			t.int64s, err = r.int64s(wt, t.int64s)
		case 8:
			t.name, err = r.str(wt)
		case 9:
			if wt != wireLen {
				return t, r.errorf("raw_data with wire type %d", wt)
			}
			var sub *wireReader
			if sub, err = r.bytes(); err == nil {
				t.raw = sub.b
			}
		default:
			err = r.skip(wt)
		}
		if err != nil {
			return t, err
		}
	}
	return t, nil
}

func readValueInfo(r *wireReader) (onnxValueInfo, error) {
	v := onnxValueInfo{off: r.base}
	for !r.done() {
		f, wt, err := r.next()
		if err != nil {
			return v, err
		}
		switch {
		case f == 1:
			v.name, err = r.str(wt)
		case f == 2 && wt == wireLen:
			var typ *wireReader
			if typ, err = r.bytes(); err == nil {
				err = readType(typ, &v)
			}
		default:
			err = r.skip(wt)
		}
		if err != nil {
			return v, err
		}
	}
	return v, nil
}

// readType reads TypeProto.tensor_type.
func readType(r *wireReader, v *onnxValueInfo) error {
	return eachLen(r, 1, func(tt *wireReader) error {
		for !tt.done() {
			f, wt, err := tt.next()
			if err != nil {
				return err
			}
			switch {
			case f == 1:
				var e uint64
				e, err = tt.uvarint(wt)
				v.elemType = int(e)
			case f == 2 && wt == wireLen:
				var shape *wireReader
				if shape, err = tt.bytes(); err == nil {
					err = eachLen(shape, 1, func(dim *wireReader) error {
						d := int64(-1)
						for !dim.done() {
							f, wt, err := dim.next()
							if err != nil {
								return err
							}
							switch f {
							case 1:
								var u uint64
								u, err = dim.uvarint(wt)
								d = int64(u)
							case 2:
								v.symbolic = true
								err = dim.skip(wt)
							default:
								err = dim.skip(wt)
							}
							if err != nil {
								return err
							}
						}
						v.dims = append(v.dims, d)
						return nil
					})
				}
			default:
				err = tt.skip(wt)
			}
			if err != nil {
				return err
			}
		}
		return nil
	})
}

// eachLen calls fn for every length-delimited field number field of r.
func eachLen(r *wireReader, field int, fn func(*wireReader) error) error {
	for !r.done() {
		f, wt, err := r.next()
		if err != nil {
			return err
		}
		if f != field || wt != wireLen {
			if err := r.skip(wt); err != nil {
				return err
			}
			continue
		}
		sub, err := r.bytes()
		if err != nil {
			return err
		}
		if err := fn(sub); err != nil {
			return err
		}
	}
	return nil
}

// ONNX TensorProto.DataType values.
const (
	onnxFloat    = 1
	onnxInt64    = 7
	onnxBFloat16 = 16
)

func onnxDType(t int) (hopper.DType, error) {
	switch t {
	case onnxFloat:
		return hopper.F32, nil
	case onnxBFloat16:
		return hopper.BF16, nil
	}
	return 0, fmt.Errorf("unsupported element type %d", t)
}

func shapeOf(dims []int64) ([]int, error) {
	out := make([]int, len(dims))
	for i, d := range dims {
		if d <= 0 || d > 1<<31 {
			return nil, fmt.Errorf("dimension %d is %d", i, d)
		}
		out[i] = int(d)
	}
	return out, nil
}

// ImportONNXWire parses a binary ONNX ModelProto and lowers its graph's
// first output. Initializers are parameters (their data is not read, so
// external data is fine); other graph inputs are activations and must
// have static shapes. Nodes must be in topological order, as ONNX
// requires. Errors carry the byte offset of the message at fault.
func ImportONNXWire(onnxBytes []byte) (ImportResult, error) {
	res, err := importONNX(onnxBytes)
	if err != nil {
		return ImportResult{}, fmt.Errorf("import onnx: %w", err)
	}
	return res, nil
}

func importONNX(b []byte) (ImportResult, error) {
	g, err := readModel(b)
	if err != nil {
		return ImportResult{}, err
	}
	l := newLowerer()
	env := make(map[string]value)
	ints := make(map[string][]int64)
	for _, t := range g.initializers {
		if t.dataType == onnxInt64 {
			ints[t.name] = t.int64s
			if len(t.int64s) == 0 {
				for i := 0; i+8 <= len(t.raw); i += 8 {
					ints[t.name] = append(ints[t.name], int64(binary.LittleEndian.Uint64(t.raw[i:])))
				}
			}
			continue
		}
		d, err := onnxDType(t.dataType)
		if err != nil {
			return ImportResult{}, fmt.Errorf("offset %d: initializer %q: %v", t.off, t.name, err)
		}
		shape, err := shapeOf(t.dims)
		if err != nil {
			return ImportResult{}, fmt.Errorf("offset %d: initializer %q: %v", t.off, t.name, err)
		}
		if size(shape) == 1 {
			k, ok := scalarOf(t)
			if !ok {
				return ImportResult{}, fmt.Errorf("offset %d: scalar initializer %q has no data", t.off, t.name)
			}
			env[t.name] = l.constant(k, d)
			continue
		}
		env[t.name] = l.weight(t.name, d, shape)
	}
	for _, in := range g.inputs {
		if _, ok := env[in.name]; ok {
			continue
		}
		if _, ok := ints[in.name]; ok {
			continue
		}
		d, err := onnxDType(in.elemType)
		if err != nil {
			return ImportResult{}, fmt.Errorf("offset %d: input %q: %v", in.off, in.name, err)
		}
		if in.symbolic || len(in.dims) == 0 {
			return ImportResult{}, fmt.Errorf("offset %d: input %q needs a static shape", in.off, in.name)
		}
		shape, err := shapeOf(in.dims)
		if err != nil {
			return ImportResult{}, fmt.Errorf("offset %d: input %q: %v", in.off, in.name, err)
		}
		env[in.name] = l.input(in.name, d, shape)
	}
	for _, n := range g.nodes {
		outs, err := onnxNodeOp(l, env, ints, n)
		if err != nil {
			return ImportResult{}, fmt.Errorf("offset %d: node %q (%s): %v", n.off, n.name, n.opType, err)
		}
		if len(n.outputs) < len(outs) {
			return ImportResult{}, fmt.Errorf("offset %d: node %q (%s) has %d outputs, want %d", n.off, n.name, n.opType, len(n.outputs), len(outs))
		}
		for i, v := range outs {
			env[n.outputs[i]] = v
		}
	}
	if len(g.outputs) == 0 {
		return ImportResult{}, fmt.Errorf("graph has no outputs")
	}
	out, ok := env[g.outputs[0].name]
	if !ok {
		return ImportResult{}, fmt.Errorf("offset %d: output %q is never computed", g.outputs[0].off, g.outputs[0].name)
	}
	if o := g.outputs[0]; !o.symbolic && len(o.dims) > 0 {
		want, err := shapeOf(o.dims)
		if err != nil || !slices.Equal(want, out.shape) {
			return ImportResult{}, fmt.Errorf("offset %d: output %q declared %v, computed %v", o.off, o.name, o.dims, out.shape)
		}
	}
	return l.finish(out)
}

func scalarOf(t onnxTensor) (float32, bool) {
	switch {
	case len(t.floats) == 1:
		return t.floats[0], true
	case t.dataType == onnxFloat && len(t.raw) == 4:
		return math.Float32frombits(binary.LittleEndian.Uint32(t.raw)), true
	case t.dataType == onnxBFloat16 && len(t.raw) == 2:
		return math.Float32frombits(uint32(binary.LittleEndian.Uint16(t.raw)) << 16), true
	}
	return 0, false
}

var onnxBinary = map[string]string{"Add": "add", "Sub": "sub", "Mul": "mul", "Div": "div", "Max": "max", "Min": "min"}

var onnxUnary = map[string]string{"Neg": "neg", "Abs": "abs", "Exp": "exp", "Log": "log", "Sqrt": "sqrt", "Sigmoid": "sigmoid"}

func onnxNodeOp(l *lowerer, env map[string]value, ints map[string][]int64, n onnxNode) ([]value, error) {
	if n.domain != "" && n.domain != "ai.onnx" && n.domain != "com.microsoft" {
		return nil, fmt.Errorf("unsupported domain %q", n.domain)
	}
	in := func(i int) (value, error) {
		if i >= len(n.inputs) || n.inputs[i] == "" {
			return value{}, fmt.Errorf("missing input %d", i)
		}
		v, ok := env[n.inputs[i]]
		if !ok {
			return value{}, fmt.Errorf("input %q is not defined before use", n.inputs[i])
		}
		return v, nil
	}
	one := func(v value, err error) ([]value, error) { return []value{v}, err }
	switch op := n.opType; {
	case onnxBinary[op] != "":
		x, err := in(0)
		if err != nil {
			return nil, err
		}
		y, err := in(1)
		if err != nil {
			return nil, err
		}
		return one(l.binary(onnxBinary[op], x, y))
	case onnxUnary[op] != "":
		x, err := in(0)
		if err != nil {
			return nil, err
		}
		return one(l.unary(onnxUnary[op], x))
	case op == "Identity":
		x, err := in(0)
		return []value{x}, err
	case op == "Reciprocal":
		x, err := in(0)
		if err != nil {
			return nil, err
		}
		return one(l.binary("div", l.constant(1, x.dtype), x))
	case op == "Relu":
		x, err := in(0)
		if err != nil {
			return nil, err
		}
		return one(l.binary("max", x, l.constant(0, x.dtype)))
	case op == "Pow":
		x, err := in(0)
		if err != nil {
			return nil, err
		}
		e, err := in(1)
		if err != nil {
			return nil, err
		}
		switch {
		case e.konst != nil && *e.konst == 2:
			return one(l.binary("mul", x, x))
		case e.konst != nil && *e.konst == 0.5:
			return one(l.unary("sqrt", x))
		}
		return nil, fmt.Errorf("only constant exponents 2 and 0.5 are supported")
	case op == "Cast":
		x, err := in(0)
		if err != nil {
			return nil, err
		}
		a, ok := n.attrs["to"]
		if !ok || !a.hasI {
			return nil, fmt.Errorf("cast without to")
		}
		d, err := onnxDType(int(a.i))
		if err != nil {
			return nil, err
		}
		return one(l.convert(x, d))
	case op == "MatMul" || op == "Gemm":
		x, err := in(0)
		if err != nil {
			return nil, err
		}
		w, err := in(1)
		if err != nil {
			return nil, err
		}
		transposed := true
		if op == "Gemm" {
			if a := n.attrs["transA"]; a.i != 0 {
				return nil, fmt.Errorf("transA is not supported")
			}
			for _, k := range []string{"alpha", "beta"} {
				if a, ok := n.attrs[k]; ok && a.hasF && a.f != 1 {
					return nil, fmt.Errorf("%s %v is not supported", k, a.f)
				}
			}
			transposed = n.attrs["transB"].i == 0
		}
		y, err := l.matmul(x, w, transposed)
		if err != nil || op == "MatMul" || len(n.inputs) < 3 || n.inputs[2] == "" {
			return []value{y}, err
		}
		c, err := in(2)
		if err != nil {
			return nil, err
		}
		if c, err = l.convert(c, hopper.F32); err != nil {
			return nil, err
		}
		return one(l.binary("add", y, c))
	case op == "ReduceSum" || op == "ReduceMean":
		x, err := in(0)
		if err != nil {
			return nil, err
		}
		axes := n.attrs["axes"].ints
		if len(n.inputs) > 1 && n.inputs[1] != "" {
			var ok bool
			if axes, ok = ints[n.inputs[1]]; !ok {
				return nil, fmt.Errorf("axes %q must be an int64 initializer", n.inputs[1])
			}
		}
		rank := int64(len(x.shape))
		if len(axes) != 1 || (axes[0] != -1 && axes[0] != rank-1) {
			return nil, fmt.Errorf("axes %v: only the last axis is supported", axes)
		}
		if k, ok := n.attrs["keepdims"]; ok && k.i == 0 {
			return nil, fmt.Errorf("keepdims=0 is not supported")
		}
		r, err := l.reduce("add", x)
		if err != nil || op == "ReduceSum" {
			return []value{r}, err
		}
		return one(l.binary("div", r, l.constant(float32(x.shape[rank-1]), x.dtype)))
	case op == "Reshape":
		x, err := in(0)
		if err != nil {
			return nil, err
		}
		if len(n.inputs) < 2 {
			return nil, fmt.Errorf("reshape without shape")
		}
		dims, ok := ints[n.inputs[1]]
		if !ok {
			return nil, fmt.Errorf("shape %q must be an int64 initializer", n.inputs[1])
		}
		shape, err := reshapeDims(x.shape, dims)
		if err != nil {
			return nil, err
		}
		return one(l.reshape(x, shape))
	case op == "SimplifiedLayerNormalization":
		x, err := in(0)
		if err != nil {
			return nil, err
		}
		gamma, err := in(1)
		if err != nil {
			return nil, err
		}
		if a, ok := n.attrs["axis"]; ok && a.i != -1 && a.i != int64(len(x.shape)-1) {
			return nil, fmt.Errorf("axis %d: only the last axis is supported", a.i)
		}
		eps := float32(1e-5)
		if a, ok := n.attrs["epsilon"]; ok && a.hasF {
			eps = a.f
		}
		return one(l.rmsNorm(x, gamma, eps))
	}
	return nil, fmt.Errorf("unsupported operator")
}

// reshapeDims resolves ONNX Reshape's 0 (copy) and -1 (infer) entries.
func reshapeDims(in []int, dims []int64) ([]int, error) {
	out := make([]int, len(dims))
	infer := -1
	known := 1
	for i, d := range dims {
		switch {
		case d == 0 && i < len(in):
			out[i] = in[i]
		case d == -1 && infer < 0:
			infer = i
			continue
		case d > 0:
			out[i] = int(d)
		default:
			return nil, fmt.Errorf("bad reshape dimension %d", d)
		}
		known *= out[i]
	}
	if infer >= 0 {
		if known == 0 || size(in)%known != 0 {
			return nil, fmt.Errorf("cannot infer reshape of %v to %v", in, dims)
		}
		out[infer] = size(in) / known
	}
	return out, nil
}
