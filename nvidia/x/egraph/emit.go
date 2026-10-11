package egraph

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strconv"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// Symbol identifies the source lines occupied by one emitted device function.
// Line numbers are one-based and inclusive.
type Symbol struct {
	Class               Id
	Operator            string
	Origin              string
	FirstLine, LastLine int
}

// Parameter describes a named, contiguous row-major FP32 kernel input.
type Parameter struct {
	Name  string
	Shape []int
}

// Source holds emitted CUDA and the expression from which its reference is
// derived. Editing CUDA does not change Expression or the reference.
type Source struct {
	CUDA       string
	Expression *Expr
	Symbols    map[string]Symbol
	Parameters []Parameter
	CostModel  string
	Cost       float64
	RuleSuites []string
	Stats      ExtractionStats
}

// ClassAt maps an emitted CUDA line to its e-class. Lines outside device
// function ranges, including the outer kernel, have no attribution.
func (s Source) ClassAt(line int) (Id, bool) {
	if line < 1 {
		return 0, false
	}
	var class Id
	found := false
	for _, symbol := range s.Symbols {
		if line >= symbol.FirstLine && line <= symbol.LastLine {
			if found {
				return 0, false
			}
			class, found = symbol.Class, true
		}
	}
	return class, found
}

type wireExpr struct {
	Class                       Id
	Origin, Operator, ValueType string
	Value                       string
	Children                    []*wireExpr
}

type sourceMetadata struct {
	Version                   int
	Expression                *wireExpr
	Fingerprint, SourceSHA256 string
	Symbols                   map[string]Symbol
	Parameters                []Parameter
	CostModel                 string
	Cost                      float64
	RuleSuites                []string
	Stats                     ExtractionStats
}

func encodeExpr(e *Expr, seen map[*Expr]bool, depth int, nodes *int) (*wireExpr, error) {
	if e == nil || depth > 128 || *nodes >= 4096 || seen[e] {
		return nil, fmt.Errorf("emit metadata: nil, cyclic or oversized expression")
	}
	*nodes++
	if len(e.Op) > 256 || len(e.Origin) > 4096 || len(e.Children) > 64 {
		return nil, fmt.Errorf("emit metadata: expression field limit")
	}
	seen[e] = true
	defer delete(seen, e)
	out := &wireExpr{Class: e.Class, Origin: e.Origin, Operator: e.Op}
	switch v := e.Data.(type) {
	case nil:
	case string:
		if len(v) > 16<<10 {
			return nil, fmt.Errorf("emit metadata: string value limit")
		}
		out.ValueType, out.Value = "string", v
	case int:
		out.ValueType, out.Value = "int", strconv.FormatInt(int64(v), 10)
	case int32:
		out.ValueType, out.Value = "int32", strconv.FormatInt(int64(v), 10)
	case int64:
		out.ValueType, out.Value = "int64", strconv.FormatInt(v, 10)
	case float32:
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return nil, fmt.Errorf("emit metadata: nonfinite constant")
		}
		out.ValueType, out.Value = "float32", strconv.FormatFloat(float64(v), 'g', -1, 32)
	case float64:
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return nil, fmt.Errorf("emit metadata: nonfinite constant")
		}
		out.ValueType, out.Value = "float64", strconv.FormatFloat(v, 'g', -1, 64)
	default:
		return nil, fmt.Errorf("emit metadata: unsupported data type %T", v)
	}
	for _, child := range e.Children {
		x, err := encodeExpr(child, seen, depth+1, nodes)
		if err != nil {
			return nil, err
		}
		out.Children = append(out.Children, x)
	}
	return out, nil
}

func decodeExpr(e *wireExpr, depth int, nodes *int) (*Expr, error) {
	if e == nil || depth > 128 || *nodes >= 4096 || len(e.Children) > 64 || len(e.Operator) > 256 || len(e.Origin) > 4096 || len(e.Value) > 16<<10 {
		return nil, fmt.Errorf("emit metadata: invalid expression bounds")
	}
	*nodes++
	out := &Expr{Class: e.Class, Origin: e.Origin, Op: e.Operator}
	var err error
	switch e.ValueType {
	case "":
		if e.Value != "" {
			return nil, fmt.Errorf("emit metadata: untyped value")
		}
	case "string":
		out.Data = e.Value
	case "int", "int32", "int64":
		bits := 64
		if e.ValueType == "int32" {
			bits = 32
		}
		if e.ValueType == "int" {
			bits = strconv.IntSize
		}
		var v int64
		v, err = strconv.ParseInt(e.Value, 10, bits)
		switch e.ValueType {
		case "int":
			out.Data = int(v)
		case "int32":
			out.Data = int32(v)
		case "int64":
			out.Data = v
		}
	case "float32", "float64":
		bits := 64
		if e.ValueType == "float32" {
			bits = 32
		}
		var v float64
		v, err = strconv.ParseFloat(e.Value, bits)
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return nil, fmt.Errorf("emit metadata: nonfinite constant")
		}
		if bits == 32 {
			out.Data = float32(v)
		} else {
			out.Data = v
		}
	default:
		return nil, fmt.Errorf("emit metadata: unknown value type %q", e.ValueType)
	}
	if err != nil {
		return nil, fmt.Errorf("emit metadata: constant: %w", err)
	}
	for _, child := range e.Children {
		x, err := decodeExpr(child, depth+1, nodes)
		if err != nil {
			return nil, err
		}
		out.Children = append(out.Children, x)
	}
	return out, nil
}

func validateSourceMetadata(s Source) error {
	if len(s.CUDA) > 8<<20 || len(s.Symbols) > 4096 || len(s.Parameters) > 256 || len(s.RuleSuites) > 64 || len(s.CostModel) > 4096 || math.IsNaN(s.Cost) || math.IsInf(s.Cost, 0) {
		return fmt.Errorf("emit metadata: source or metadata limit")
	}
	lines := strings.Count(s.CUDA, "\n") + 1
	var ranges []Symbol
	for name, v := range s.Symbols {
		if len(name) > 256 || len(v.Operator) > 256 || len(v.Origin) > 4096 || v.FirstLine < 1 || v.LastLine < v.FirstLine || v.LastLine > lines {
			return fmt.Errorf("emit metadata: invalid symbol range")
		}
		ranges = append(ranges, v)
	}
	sort.Slice(ranges, func(i, j int) bool { return ranges[i].FirstLine < ranges[j].FirstLine })
	for i := 1; i < len(ranges); i++ {
		if ranges[i].FirstLine <= ranges[i-1].LastLine {
			return fmt.Errorf("emit metadata: overlapping symbol ranges")
		}
	}
	names := make(map[string]bool)
	for _, p := range s.Parameters {
		if p.Name == "" || len(p.Name) > 256 || names[p.Name] || len(p.Shape) > 8 {
			return fmt.Errorf("emit metadata: invalid parameter")
		}
		names[p.Name] = true
		size := 1
		for _, d := range p.Shape {
			if d <= 0 || d > 65536/size {
				return fmt.Errorf("emit metadata: parameter shape limit")
			}
			size *= d
		}
	}
	for _, suite := range s.RuleSuites {
		if len(suite) > 4096 {
			return fmt.Errorf("emit metadata: rule suite limit")
		}
	}
	return nil
}

// WriteFile writes path and its .egraph.json sidecar through temporary regular
// files. Both artifacts are validated before writing. A failure may leave one
// successfully renamed artifact; the pair is not a filesystem transaction.
func (s Source) WriteFile(path string) error {
	if filepath.Ext(path) != ".cu" {
		return fmt.Errorf("write emitted source: .cu path required")
	}
	if err := validateSourceMetadata(s); err != nil {
		return err
	}
	nodes := 0
	expr, err := encodeExpr(s.Expression, make(map[*Expr]bool), 1, &nodes)
	if err != nil {
		return err
	}
	meta := sourceMetadata{Version: 1, Expression: expr, Fingerprint: s.Expression.Fingerprint(), SourceSHA256: fmt.Sprintf("%x", sha256.Sum256([]byte(s.CUDA))), Symbols: s.Symbols, Parameters: s.Parameters, CostModel: s.CostModel, Cost: s.Cost, RuleSuites: s.RuleSuites, Stats: s.Stats}
	data, err := json.MarshalIndent(meta, "", "    ")
	if err != nil {
		return err
	}
	if len(data) > 8<<20 {
		return fmt.Errorf("emit metadata: encoded size limit")
	}
	for _, artifact := range []struct {
		path string
		data []byte
	}{{path, []byte(s.CUDA)}, {strings.TrimSuffix(path, ".cu") + ".egraph.json", append(data, '\n')}} {
		f, err := os.CreateTemp(filepath.Dir(artifact.path), ".egraph-source-")
		if err != nil {
			return err
		}
		name := f.Name()
		_, writeErr := f.Write(artifact.data)
		closeErr := f.Close()
		if writeErr != nil || closeErr != nil {
			os.Remove(name)
			if writeErr != nil {
				return writeErr
			}
			return closeErr
		}
		if err := os.Rename(name, artifact.path); err != nil {
			os.Remove(name)
			return err
		}
	}
	return nil
}

// ReadMetadata reads a bounded sidecar and preserves the original expression
// data types. CUDA comes separately from the caller's possibly edited file;
// its bytes are never used to derive or change the reference expression.
func ReadMetadata(r io.Reader, cuda string) (Source, error) {
	if r == nil {
		return Source{}, fmt.Errorf("read emitted metadata: nil reader")
	}
	data, err := io.ReadAll(io.LimitReader(r, (8<<20)+1))
	if err != nil {
		return Source{}, err
	}
	if len(data) > 8<<20 {
		return Source{}, fmt.Errorf("read emitted metadata: size limit")
	}
	// Bound structure before allocating the decoded expression graph.
	tokens := json.NewDecoder(bytes.NewReader(data))
	depth, count := 0, 0
	for {
		t, err := tokens.Token()
		if err == io.EOF {
			break
		}
		if err != nil {
			return Source{}, err
		}
		count++
		if count > 65536 {
			return Source{}, fmt.Errorf("read emitted metadata: token limit")
		}
		if d, ok := t.(json.Delim); ok {
			if d == '{' || d == '[' {
				depth++
				if depth > 256 {
					return Source{}, fmt.Errorf("read emitted metadata: depth limit")
				}
			} else {
				depth--
			}
		}
	}
	var meta sourceMetadata
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&meta); err != nil {
		return Source{}, err
	}
	if decoder.Decode(new(any)) != io.EOF {
		return Source{}, fmt.Errorf("read emitted metadata: trailing value")
	}
	if meta.Version != 1 {
		return Source{}, fmt.Errorf("read emitted metadata: unsupported version")
	}
	nodes := 0
	expr, err := decodeExpr(meta.Expression, 1, &nodes)
	if err != nil {
		return Source{}, err
	}
	if expr.Fingerprint() != meta.Fingerprint {
		return Source{}, fmt.Errorf("read emitted metadata: fingerprint mismatch")
	}
	if _, err := hex.DecodeString(meta.SourceSHA256); err != nil || len(meta.SourceSHA256) != 64 {
		return Source{}, fmt.Errorf("read emitted metadata: invalid source digest")
	}
	if meta.SourceSHA256 != fmt.Sprintf("%x", sha256.Sum256([]byte(cuda))) {
		meta.Symbols = nil
	}
	s := Source{CUDA: cuda, Expression: expr, Symbols: meta.Symbols, Parameters: meta.Parameters, CostModel: meta.CostModel, Cost: meta.Cost, RuleSuites: meta.RuleSuites, Stats: meta.Stats}
	if err := validateSourceMetadata(s); err != nil {
		return Source{}, err
	}
	return s, nil
}

// ErrEmissionUnsupported reports a node outside the structural emission and
// expression-reference contract. The error also identifies its operator/class.
var ErrEmissionUnsupported = cudair.ErrEmissionUnsupported

func expressionConstant(e *Expr) (float32, bool) {
	if e == nil || len(e.Children) != 0 {
		return 0, false
	}
	var literal float64
	isLiteral := e.Op != "const" && e.Op != "constant"
	if isLiteral {
		var err error
		literal, err = strconv.ParseFloat(e.Op, 32)
		if err != nil || math.IsNaN(literal) || math.IsInf(literal, 0) {
			return 0, false
		}
	}
	var value float64
	switch v := e.Data.(type) {
	case float32:
		value = float64(v)
	case float64:
		value = v
	case int:
		value = float64(v)
	case int32:
		value = float64(v)
	case int64:
		value = float64(v)
	case string:
		if v != e.Op {
			return 0, false
		}
		var err error
		value, err = strconv.ParseFloat(v, 32)
		if err != nil {
			return 0, false
		}
	case nil:
		var err error
		value, err = strconv.ParseFloat(e.Op, 32)
		if err != nil {
			return 0, false
		}
	default:
		return 0, false
	}
	result := float32(value)
	if math.IsNaN(float64(result)) || math.IsInf(float64(result), 0) {
		return 0, false
	}
	if isLiteral && result != float32(literal) {
		return 0, false
	}
	return result, true
}

// ReferenceGraph lowers only the sidecar expression and declared input shapes
// to cudair. Evaluating it with cudaref checks edited source independently of
// that source. Supported operators are add, mul, silu, swish (beta=1), exact-erf
// gelu, rmsnorm, gemv and fused_rmsnorm_gemv. GEMV weights are row-major [M,N].
func (s Source) ReferenceGraph() (*cudair.Graph, error) {
	if err := validateSourceMetadata(s); err != nil {
		return nil, err
	}
	count := 0
	if _, err := encodeExpr(s.Expression, make(map[*Expr]bool), 1, &count); err != nil {
		return nil, err
	}
	g := cudair.NewGraph()
	inputs := make(map[string]*cudair.Node)
	total := 0
	for _, p := range s.Parameters {
		size := 1
		for _, n := range p.Shape {
			size *= n
		}
		if size > 65536-total {
			return nil, fmt.Errorf("reference expression: packed input limit")
		}
		total += size
		node, err := g.AddInputChecked(p.Name, p.Shape, cudair.Float32)
		if err != nil {
			return nil, err
		}
		inputs[p.Name] = node
	}
	scalar := func(n *cudair.Node) *cudair.Node { return g.AddReshape(n, []int{1}) }
	norm := func(x, w, eps *cudair.Node) (*cudair.Node, error) {
		if len(x.Shape) != 1 || !reflect.DeepEqual(x.Shape, w.Shape) || eps.NumElements() != 1 {
			return nil, fmt.Errorf("reference rmsnorm: expected x/w[N] and scalar epsilon")
		}
		n := x.Shape[0]
		square := g.AddBinary("mul", x, x)
		sum := g.AddReduce("sum", square, 0)
		divisor := g.AddConstant("norm_width", []int{1}, cudair.Float32, []float32{float32(n)})
		mean := g.AddBinary("div", sum, divisor)
		variance := g.AddBinary("add", mean, scalar(eps))
		denominator := g.AddUnary("sqrt", variance)
		one := g.AddConstant("one", []int{1}, cudair.Float32, []float32{1})
		scale := g.AddBinary("div", one, denominator)
		ones := make([]float32, n)
		for i := range ones {
			ones[i] = 1
		}
		broadcast := g.AddConstant("broadcast", []int{n, 1}, cudair.Float32, ones)
		vectorScale := g.AddReshape(g.AddMatMul(broadcast, g.AddReshape(scale, []int{1, 1})), []int{n})
		return g.AddBinary("mul", g.AddBinary("mul", x, vectorScale), w), g.Err()
	}
	gemv := func(x, w *cudair.Node) (*cudair.Node, error) {
		if len(x.Shape) != 1 || len(w.Shape) != 2 || w.Shape[1] != x.Shape[0] {
			return nil, fmt.Errorf("reference gemv: expected x[N], weights[M,N]")
		}
		return g.AddReshape(g.AddMatMul(w, g.AddReshape(x, []int{x.Shape[0], 1})), []int{w.Shape[0]}), g.Err()
	}
	var lower func(*Expr) (*cudair.Node, error)
	visits := 0
	lower = func(e *Expr) (*cudair.Node, error) {
		visits++
		if visits > 4096 {
			return nil, fmt.Errorf("reference expression: node limit")
		}
		if len(e.Children) == 0 {
			if n := inputs[e.Op]; n != nil {
				if e.Data != nil {
					text, ok := e.Data.(string)
					if !ok || text != e.Op {
						return nil, fmt.Errorf("reference input: inconsistent leaf name")
					}
				}
				return n, nil
			}
			if value, ok := expressionConstant(e); ok {
				return g.AddConstantChecked("constant", []int{1}, cudair.Float32, []float32{value})
			}
			return nil, fmt.Errorf("%w: operator %q e-class %d: missing declared input", ErrEmissionUnsupported, e.Op, e.Class)
		}
		if e.Data != nil {
			return nil, fmt.Errorf("%w: operator %q e-class %d: unsupported payload", ErrEmissionUnsupported, e.Op, e.Class)
		}
		switch e.Op {
		case "add", "mul", "silu", "swish", "gelu", "rmsnorm", "gemv", "fused_rmsnorm_gemv":
		default:
			return nil, fmt.Errorf("%w: operator %q e-class %d", ErrEmissionUnsupported, e.Op, e.Class)
		}
		nodes := make([]*cudair.Node, len(e.Children))
		for i, c := range e.Children {
			n, err := lower(c)
			if err != nil {
				return nil, err
			}
			nodes[i] = n
		}
		switch e.Op {
		case "add", "mul":
			if len(nodes) != 2 || !reflect.DeepEqual(nodes[0].Shape, nodes[1].Shape) {
				return nil, fmt.Errorf("reference %s: two identical shapes required", e.Op)
			}
			return g.AddBinaryChecked(e.Op, nodes[0], nodes[1])
		case "silu", "swish", "gelu":
			if len(nodes) != 1 {
				return nil, fmt.Errorf("reference %s: one operand required", e.Op)
			}
			op := e.Op
			if op == "swish" {
				op = "silu"
			}
			return g.AddUnaryChecked(op, nodes[0])
		case "rmsnorm":
			if len(nodes) != 3 {
				return nil, fmt.Errorf("reference rmsnorm: three operands required")
			}
			return norm(nodes[0], nodes[1], nodes[2])
		case "gemv":
			if len(nodes) != 2 {
				return nil, fmt.Errorf("reference gemv: two operands required")
			}
			return gemv(nodes[0], nodes[1])
		case "fused_rmsnorm_gemv":
			if len(nodes) != 4 {
				return nil, fmt.Errorf("reference fused_rmsnorm_gemv: four operands required")
			}
			normalized, err := norm(nodes[0], nodes[1], nodes[2])
			if err != nil {
				return nil, err
			}
			return gemv(normalized, nodes[3])
		default:
			return nil, fmt.Errorf("%w: operator %q e-class %d", ErrEmissionUnsupported, e.Op, e.Class)
		}
	}
	root, err := lower(s.Expression)
	if err != nil {
		return nil, err
	}
	if err := g.Err(); err != nil {
		return nil, err
	}
	root = g.AddReshape(root, root.Shape)
	if err := g.Err(); err != nil {
		return nil, err
	}
	root.Name = "result"
	if err := g.SetOutputs(root); err != nil {
		return nil, err
	}
	if _, err := g.Snapshot(); err != nil {
		return nil, err
	}
	return g, nil
}

// EmitOptions provides authoritative input shapes and extraction metadata.
// Inputs are contiguous row-major FP32 buffers, ordered lexically by Name.
// No broadcasting or undeclared shape is inferred. The emitted outer kernel
// computes one vector result on thread zero; no throughput claim is implied.
type EmitOptions struct {
	KernelName string
	Parameters []Parameter
	CostModel  string
	Cost       float64
	RuleSuites []string
	Stats      ExtractionStats
}

type expressionEmitter struct {
	expression   *Expr
	key, symbol  string
	inputWidth   int
	shape        []int
	offset       int
	constant     *float32
	dependencies []*expressionEmitter
}

func (n *expressionEmitter) DAGKey() string              { return n.key }
func (n *expressionEmitter) DAGSymbol() string           { return n.symbol }
func (n *expressionEmitter) Name() string                { return n.expression.Op }
func (n *expressionEmitter) InputWidth() int             { return n.inputWidth }
func (n *expressionEmitter) OutputWidth() int            { return shapeElements(n.shape) }
func (n *expressionEmitter) PaddedOutputWidth() int      { return n.OutputWidth() }
func (*expressionEmitter) FwdCtxBytes() (int, error)     { return 0, nil }
func (*expressionEmitter) FwdCtxAlignedPerElement() bool { return false }
func (n *expressionEmitter) Dependencies() []cudair.DAGEmitter {
	out := make([]cudair.DAGEmitter, len(n.dependencies))
	for i, d := range n.dependencies {
		out[i] = d
	}
	return out
}
func (n *expressionEmitter) EmitForward(name string) (string, error) {
	return n.EmitForwardWithDependencies(name, nil)
}
func shapeElements(shape []int) int {
	size := 1
	for _, d := range shape {
		size *= d
	}
	return size
}

func floatLiteral(value float32) string {
	s := strconv.FormatFloat(float64(value), 'g', -1, 32)
	if !strings.ContainsAny(s, ".eE") {
		s += ".0"
	}
	return s + "f"
}

func (n *expressionEmitter) EmitForwardWithDependencies(name string, children []cudair.ChildSymbol) (string, error) {
	if len(children) != len(n.dependencies) {
		return "", fmt.Errorf("emit %s: dependency mismatch", n.expression.Op)
	}
	var body strings.Builder
	for i, c := range children {
		fmt.Fprintf(&body, "    auto arg%d = %s(input, params, %s);\n", i, c.Name, c.FwdCtxArg())
	}
	fmt.Fprintf(&body, "    tvec<float, %d> output;\n", n.OutputWidth())
	if len(children) == 0 {
		if n.constant != nil {
			fmt.Fprintf(&body, "    output[0] = %s;\n", floatLiteral(*n.constant))
		} else {
			fmt.Fprintf(&body, "    for (int i = 0; i < %d; ++i) {\n        output[i] = input[%d + i];\n    }\n", n.OutputWidth(), n.offset)
		}
	} else {
		switch n.expression.Op {
		case "add", "mul":
			op := "+"
			if n.expression.Op == "mul" {
				op = "*"
			}
			fmt.Fprintf(&body, "    for (int i = 0; i < %d; ++i) {\n        output[i] = arg0[i] %s arg1[i];\n    }\n", n.OutputWidth(), op)
		case "silu", "swish":
			fmt.Fprintf(&body, "    for (int i = 0; i < %d; ++i) {\n        output[i] = arg0[i] / (1.0f + expf(-arg0[i]));\n    }\n", n.OutputWidth())
		case "gelu":
			fmt.Fprintf(&body, "    for (int i = 0; i < %d; ++i) {\n        output[i] = 0.5f * arg0[i] * (1.0f + erff(arg0[i] * 0.7071067811865475f));\n    }\n", n.OutputWidth())
		case "rmsnorm", "fused_rmsnorm_gemv":
			width := n.dependencies[0].OutputWidth()
			fmt.Fprintf(&body, "    float sum = 0.0f;\n    for (int i = 0; i < %d; ++i) {\n        sum += arg0[i] * arg0[i];\n    }\n    float scale = 1.0f / sqrtf(sum / %s + arg2[0]);\n", width, floatLiteral(float32(width)))
			if n.expression.Op == "rmsnorm" {
				fmt.Fprintf(&body, "    for (int i = 0; i < %d; ++i) {\n        output[i] = (arg0[i] * scale) * arg1[i];\n    }\n", width)
			} else {
				fmt.Fprintf(&body, "    for (int row = 0; row < %d; ++row) {\n        float dot = 0.0f;\n        for (int i = 0; i < %d; ++i) {\n            float normalized = (arg0[i] * scale) * arg1[i];\n            dot += arg3[row * %d + i] * normalized;\n        }\n        output[row] = dot;\n    }\n", n.OutputWidth(), width, width)
			}
		case "gemv":
			width := n.dependencies[0].OutputWidth()
			fmt.Fprintf(&body, "    for (int row = 0; row < %d; ++row) {\n        float dot = 0.0f;\n        for (int i = 0; i < %d; ++i) {\n            dot += arg1[row * %d + i] * arg0[i];\n        }\n        output[row] = dot;\n    }\n", n.OutputWidth(), width, width)
		default:
			return "", fmt.Errorf("%w: operator %q e-class %d", ErrEmissionUnsupported, n.expression.Op, n.expression.Class)
		}
	}
	body.WriteString("    return output;")
	return cudair.EmitFromBody(n, name, body.String())
}

func cudaName(name string) bool {
	if name == "" || len(name) > 128 || strings.HasPrefix(name, "__") || strings.HasPrefix(name, "eg_") {
		return false
	}
	for i, r := range name {
		if !(r == '_' || r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || i > 0 && r >= '0' && r <= '9') {
			return false
		}
	}
	for _, word := range strings.Fields("alignas alignof and and_eq asm auto bitand bitor bool break case catch char char16_t char32_t class compl const constexpr const_cast continue decltype default delete do double dynamic_cast else enum explicit export extern false float for friend goto if inline int long mutable namespace new noexcept not not_eq nullptr operator or or_eq private protected public register reinterpret_cast return short signed sizeof static static_assert static_cast struct switch template this thread_local throw true try typedef typeid typename union unsigned using virtual void volatile wchar_t while xor xor_eq input output params fwd_ctx tvec uint8_t blockIdx threadIdx blockDim gridDim warpSize expf erff sqrtf i") {
		if name == word {
			return false
		}
	}
	return true
}

// EmitCUDA structurally emits the expression through cudair's opt-in fan-out
// contract. Unsupported nodes refuse by operator and e-class; existing CUDA
// generators are never used as a fallback. All functions and leaf buffers have
// deterministic names, and shared e-class nodes share one function definition.
func EmitCUDA(expr *Expr, opts EmitOptions) (Source, error) {
	if opts.KernelName == "" {
		opts.KernelName = "egraph_kernel"
	}
	if !cudaName(opts.KernelName) {
		return Source{}, fmt.Errorf("emit CUDA: invalid kernel name")
	}
	count := 0
	wire, err := encodeExpr(expr, make(map[*Expr]bool), 1, &count)
	if err != nil {
		return Source{}, err
	}
	copyExpr, err := decodeExpr(wire, 1, new(int))
	if err != nil {
		return Source{}, err
	}
	parameters := append([]Parameter(nil), opts.Parameters...)
	for i := range parameters {
		parameters[i].Shape = append([]int(nil), parameters[i].Shape...)
	}
	sort.Slice(parameters, func(i, j int) bool { return parameters[i].Name < parameters[j].Name })
	source := Source{Expression: copyExpr, Parameters: parameters, CostModel: opts.CostModel, Cost: opts.Cost, RuleSuites: append([]string(nil), opts.RuleSuites...), Stats: opts.Stats}
	// The actual reference lowering validates operator semantics and every shape
	// before constructing device code. It never consumes emitted CUDA.
	if _, err := source.ReferenceGraph(); err != nil {
		return Source{}, err
	}
	offsets := make(map[string]int)
	shapes := make(map[string][]int)
	used := make(map[string]bool)
	inputWidth := 0
	for _, p := range parameters {
		if !cudaName(p.Name) || p.Name == opts.KernelName {
			return Source{}, fmt.Errorf("emit CUDA: invalid input name")
		}
		offsets[p.Name] = inputWidth
		shapes[p.Name] = p.Shape
		inputWidth += shapeElements(p.Shape)
	}
	// A constant-only expression still gets a one-element zero input vector.
	if inputWidth == 0 {
		inputWidth = 1
	}
	records := make(map[string]*expressionEmitter)
	symbols := make(map[string]string)
	visits := 0
	var build func(*Expr) (*expressionEmitter, error)
	build = func(e *Expr) (*expressionEmitter, error) {
		visits++
		if visits > 4096 {
			return nil, fmt.Errorf("emit CUDA: expression bounds or missing provenance")
		}
		n := &expressionEmitter{expression: e, inputWidth: inputWidth}
		op := e.Op
		if len(e.Children) == 0 {
			if shape, ok := shapes[e.Op]; ok {
				n.shape = shape
				n.offset = offsets[e.Op]
				used[e.Op] = true
			} else if value, ok := expressionConstant(e); ok {
				n.shape = []int{1}
				n.constant = &value
				op = "constant"
			} else {
				return nil, fmt.Errorf("%w: operator %q e-class %d", ErrEmissionUnsupported, e.Op, e.Class)
			}
		} else {
			for _, child := range e.Children {
				dep, err := build(child)
				if err != nil {
					return nil, err
				}
				n.dependencies = append(n.dependencies, dep)
			}
			switch e.Op {
			case "add", "mul", "silu", "swish", "gelu", "rmsnorm":
				n.shape = append([]int(nil), n.dependencies[0].shape...)
			case "gemv":
				n.shape = []int{n.dependencies[1].shape[0]}
			case "fused_rmsnorm_gemv":
				n.shape = []int{n.dependencies[3].shape[0]}
			default:
				return nil, fmt.Errorf("%w: operator %q e-class %d", ErrEmissionUnsupported, e.Op, e.Class)
			}
		}
		n.symbol = fmt.Sprintf("eg_c%d_%s", e.Class, op)
		fingerprint := fmt.Sprintf("%d|%s|%s|%v|%d", e.Class, e.Origin, e.Fingerprint(), n.shape, inputWidth)
		n.key = fmt.Sprintf("%x", sha256.Sum256([]byte(fingerprint)))
		if prior, ok := symbols[n.symbol]; ok && prior != n.key {
			return nil, fmt.Errorf("%w: operator %q e-class %d: conflicting class symbol", ErrEmissionUnsupported, e.Op, e.Class)
		}
		symbols[n.symbol] = n.key
		if prior := records[n.key]; prior != nil {
			return prior, nil
		}
		if len(records) >= 256 {
			return nil, fmt.Errorf("emit CUDA: node limit")
		}
		records[n.key] = n
		return n, nil
	}
	root, err := build(copyExpr)
	if err != nil {
		return Source{}, err
	}
	for _, p := range parameters {
		if !used[p.Name] {
			return Source{}, fmt.Errorf("emit CUDA: unused input declaration %q", p.Name)
		}
	}
	unit, err := cudair.EmitTree(root, root.DAGSymbol())
	if err != nil {
		return Source{}, err
	}
	code := unit.Source
	for _, symbol := range unit.Symbols {
		n := records[symbols[symbol]]
		marker := "__device__ auto " + symbol + "("
		comment := fmt.Sprintf("// e-class %d: %q, from %q\n", n.expression.Class, n.expression.Op, n.expression.Origin)
		if !strings.Contains(code, marker) {
			return Source{}, fmt.Errorf("emit CUDA: missing admitted symbol")
		}
		code = strings.Replace(code, marker, comment+marker, 1)
	}
	var outer strings.Builder
	fmt.Fprintf(&outer, "extern \"C\" __global__ void %s(", opts.KernelName)
	for _, p := range parameters {
		fmt.Fprintf(&outer, "const float* %s, ", p.Name)
	}
	outer.WriteString("float* eg_output) {\n    if (blockIdx.x != 0 || blockIdx.y != 0 || blockIdx.z != 0 || threadIdx.x != 0 || threadIdx.y != 0 || threadIdx.z != 0) {\n        return;\n    }\n")
	fmt.Fprintf(&outer, "    tvec<float, %d> eg_input;\n", inputWidth)
	if len(parameters) == 0 {
		outer.WriteString("    eg_input[0] = 0.0f;\n")
	}
	for _, p := range parameters {
		fmt.Fprintf(&outer, "    for (int i = 0; i < %d; ++i) {\n        eg_input[%d + i] = %s[i];\n    }\n", shapeElements(p.Shape), offsets[p.Name], p.Name)
	}
	fmt.Fprintf(&outer, "    auto eg_result = %s(eg_input, nullptr, nullptr);\n    for (int i = 0; i < %d; ++i) {\n        eg_output[i] = eg_result[i];\n    }\n}\n", root.symbol, root.OutputWidth())
	header := fmt.Sprintf("// expression: %q\n// fingerprint: %q\n// cost model: %q, cost: %s\n// rule suites: %q\n// extraction stats: work=%d nodes=%d exhausted=%t\n", copyExpr.String(), copyExpr.Fingerprint(), opts.CostModel, strconv.FormatFloat(opts.Cost, 'g', -1, 64), opts.RuleSuites, opts.Stats.Work, opts.Stats.Nodes, opts.Stats.WorkExhausted)
	template := `{{HEADER}}
typedef unsigned char uint8_t;
template <typename T, int N> struct tvec {
    T values[N];
    __device__ T& operator[](int i) { return values[i]; }
    __device__ const T& operator[](int i) const { return values[i]; }
};

{{FUNCTIONS}}

{{KERNEL}}`
	source.CUDA, err = nvrtcx.SpliceTemplate(template, map[string]string{"HEADER": header, "FUNCTIONS": code, "KERNEL": outer.String()})
	if err != nil {
		return Source{}, err
	}
	source.CUDA = strings.ReplaceAll(source.CUDA, "\t", "    ")
	source.Symbols = make(map[string]Symbol)
	for _, symbol := range unit.Symbols {
		start := strings.Index(source.CUDA, "__device__ auto "+symbol+"(")
		if start < 0 {
			return Source{}, fmt.Errorf("emit CUDA: missing symbol range")
		}
		end := strings.Index(source.CUDA[start:], "\n}")
		if end < 0 {
			return Source{}, fmt.Errorf("emit CUDA: unterminated symbol range")
		}
		n := records[symbols[symbol]]
		source.Symbols[symbol] = Symbol{Class: n.expression.Class, Operator: n.expression.Op, Origin: n.expression.Origin, FirstLine: strings.Count(source.CUDA[:start], "\n") + 1, LastLine: strings.Count(source.CUDA[:start+end+2], "\n") + 1}
	}
	if err := validateSourceMetadata(source); err != nil {
		return Source{}, err
	}
	return source, nil
}
