package frontend

import (
	"fmt"
	"math"
	"slices"
	"strconv"
	"strings"

	"github.com/tmc/lib/nvidia/x/egraph/hopper"
)

// A tensorType is a StableHLO ranked tensor type.
type tensorType struct {
	shape []int
	dtype hopper.DType
}

func (t tensorType) String() string {
	var b strings.Builder
	b.WriteString("tensor<")
	for _, d := range t.shape {
		fmt.Fprintf(&b, "%dx", d)
	}
	b.WriteString(map[hopper.DType]string{hopper.F32: "f32", hopper.BF16: "bf16"}[t.dtype] + ">")
	return b.String()
}

func parseTensorType(s string) (tensorType, error) {
	s = strings.TrimSpace(s)
	body, ok := strings.CutPrefix(s, "tensor<")
	if !ok || !strings.HasSuffix(body, ">") {
		return tensorType{}, fmt.Errorf("bad tensor type %q", s)
	}
	parts := strings.Split(strings.TrimSuffix(body, ">"), "x")
	var t tensorType
	switch parts[len(parts)-1] {
	case "f32":
		t.dtype = hopper.F32
	case "bf16":
		t.dtype = hopper.BF16
	default:
		return t, fmt.Errorf("unsupported element type in %q", s)
	}
	for _, p := range parts[:len(parts)-1] {
		n, err := strconv.Atoi(p)
		if err != nil || n <= 0 {
			return t, fmt.Errorf("bad dimension %q in %q", p, s)
		}
		t.shape = append(t.shape, n)
	}
	return t, nil
}

// splitTop splits s at sep outside of <>, () and [] nesting.
func splitTop(s string, sep byte) []string {
	var out []string
	depth, start := 0, 0
	for i := 0; i < len(s); i++ {
		switch s[i] {
		case '<', '(', '[':
			depth++
		case '>', ')', ']':
			depth--
		case sep:
			if depth == 0 {
				out = append(out, s[start:i])
				start = i + 1
			}
		}
	}
	return append(out, s[start:])
}

// parseSignature parses "(t1, t2) -> t3" or "t".
func parseSignature(s string) (ins []tensorType, out tensorType, err error) {
	s = strings.TrimSpace(s)
	if lhs, rhs, ok := strings.Cut(s, "->"); ok {
		lhs = strings.TrimSpace(lhs)
		if !strings.HasPrefix(lhs, "(") || !strings.HasSuffix(lhs, ")") {
			return nil, out, fmt.Errorf("bad signature %q", s)
		}
		for _, p := range splitTop(lhs[1:len(lhs)-1], ',') {
			t, err := parseTensorType(p)
			if err != nil {
				return nil, out, err
			}
			ins = append(ins, t)
		}
		out, err = parseTensorType(rhs)
		return ins, out, err
	}
	out, err = parseTensorType(s)
	return nil, out, err
}

func parseInts(s string) ([]int, error) {
	s = strings.TrimSpace(s)
	if !strings.HasPrefix(s, "[") || !strings.HasSuffix(s, "]") {
		return nil, fmt.Errorf("bad integer list %q", s)
	}
	var out []int
	for _, p := range strings.Split(s[1:len(s)-1], ",") {
		if p = strings.TrimSpace(p); p == "" {
			continue
		}
		n, err := strconv.Atoi(p)
		if err != nil {
			return nil, fmt.Errorf("bad integer %q", p)
		}
		out = append(out, n)
	}
	return out, nil
}

// parseFloatLiteral parses an MLIR float literal, including hexadecimal
// bit patterns such as 0xFF800000.
func parseFloatLiteral(s string, d hopper.DType) (float32, error) {
	s = strings.TrimSpace(s)
	if h, ok := strings.CutPrefix(strings.ToLower(s), "0x"); ok {
		bits, err := strconv.ParseUint(h, 16, 32)
		if err != nil {
			return 0, fmt.Errorf("bad float literal %q", s)
		}
		if d == hopper.BF16 {
			bits <<= 16
		}
		return math.Float32frombits(uint32(bits)), nil
	}
	f, err := strconv.ParseFloat(s, 32)
	if err != nil {
		return 0, fmt.Errorf("bad float literal %q", s)
	}
	return float32(f), nil
}

var shloBinary = map[string]string{
	"add": "add", "subtract": "sub", "multiply": "mul", "divide": "div",
	"maximum": "max", "minimum": "min",
}

var shloUnary = map[string]string{
	"negate": "neg", "abs": "abs", "exponential": "exp", "log": "log",
	"sqrt": "sqrt", "rsqrt": "rsqrt", "logistic": "sigmoid",
	"cosine": "cos", "sine": "sin",
}

type shloVal struct {
	v   value
	typ tensorType
}

// ImportStableHLO parses the pretty-printed StableHLO form of one
// func.func and lowers it. %arg0 is the activation input; the remaining
// arguments are parameters. Errors carry the line number.
func ImportStableHLO(mlirText string) (ImportResult, error) {
	res, err := importStableHLO(mlirText)
	if err != nil {
		return ImportResult{}, fmt.Errorf("import stablehlo: %w", err)
	}
	return res, nil
}

func importStableHLO(text string) (ImportResult, error) {
	l := newLowerer()
	env := make(map[string]shloVal)
	var root *shloVal
	inFunc := false
	for i, line := range strings.Split(text, "\n") {
		lineNo := i + 1
		line = strings.TrimSpace(line)
		if c := strings.Index(line, "//"); c >= 0 {
			line = strings.TrimSpace(line[:c])
		}
		lerr := func(format string, args ...any) error {
			return fmt.Errorf("line %d: %s", lineNo, fmt.Sprintf(format, args...))
		}
		switch {
		case line == "" || line == "}" || strings.HasPrefix(line, "module"):
			continue
		case strings.HasPrefix(line, "func.func"):
			if inFunc {
				return ImportResult{}, lerr("only one function is supported")
			}
			inFunc = true
			open, close := strings.Index(line, "("), strings.LastIndex(line, ") ->")
			if open < 0 || close < open {
				return ImportResult{}, lerr("bad function signature")
			}
			for j, arg := range splitTop(line[open+1:close], ',') {
				name, typ, ok := strings.Cut(arg, ":")
				name = strings.TrimSpace(name)
				if !ok || !strings.HasPrefix(name, "%") {
					return ImportResult{}, lerr("bad argument %q", arg)
				}
				if a := strings.Index(typ, "{"); a >= 0 {
					typ = typ[:a]
				}
				t, err := parseTensorType(typ)
				if err != nil {
					return ImportResult{}, lerr("%v", err)
				}
				var v value
				if j == 0 {
					v = l.input(name[1:], t.dtype, t.shape)
				} else {
					v = l.weight(name[1:], t.dtype, t.shape)
				}
				env[name] = shloVal{v, t}
			}
			continue
		case !inFunc:
			return ImportResult{}, lerr("statement outside func.func")
		case strings.HasPrefix(line, "return") || strings.HasPrefix(line, "stablehlo.return") || strings.HasPrefix(line, "func.return"):
			_, rest, _ := strings.Cut(line, " ")
			name, _, _ := strings.Cut(rest, ":")
			v, ok := env[strings.TrimSpace(name)]
			if !ok {
				return ImportResult{}, lerr("return of undefined value %q", strings.TrimSpace(name))
			}
			root = &v
			continue
		}
		name, rhs, ok := strings.Cut(line, "=")
		name = strings.TrimSpace(name)
		if !ok || !strings.HasPrefix(name, "%") || strings.Contains(name, ":") {
			return ImportResult{}, lerr("expected %%name = stablehlo.op")
		}
		if _, dup := env[name]; dup {
			return ImportResult{}, lerr("%s redefined", name)
		}
		v, err := shloOp(l, env, strings.TrimSpace(rhs))
		if err != nil {
			return ImportResult{}, lerr("%s: %v", name, err)
		}
		env[name] = v
	}
	if root == nil {
		return ImportResult{}, fmt.Errorf("no return")
	}
	return l.finish(root.v)
}

func shloOp(l *lowerer, env map[string]shloVal, rhs string) (shloVal, error) {
	colon := strings.LastIndex(rhs, " : ")
	if colon < 0 {
		return shloVal{}, fmt.Errorf("missing type")
	}
	body, sig := rhs[:colon], rhs[colon+3:]
	ins, out, err := parseSignature(sig)
	if err != nil {
		return shloVal{}, err
	}
	opTok, rest, _ := strings.Cut(body, " ")
	if i := strings.Index(opTok, "("); i >= 0 {
		opTok, rest = opTok[:i], opTok[i:]+" "+rest
	}
	opName, ok := strings.CutPrefix(opTok, "stablehlo.")
	if !ok {
		return shloVal{}, fmt.Errorf("unsupported operation %q", opTok)
	}
	// Operands are the leading %names; attributes follow.
	var operands []shloVal
	var attrs []string
	for _, f := range splitTop(rest, ',') {
		f = strings.TrimSpace(f)
		if strings.HasPrefix(f, "%") && len(attrs) == 0 {
			v, ok := env[f]
			if !ok {
				return shloVal{}, fmt.Errorf("undefined value %s", f)
			}
			operands = append(operands, v)
			continue
		}
		if f != "" {
			attrs = append(attrs, f)
		}
	}
	// Check declared operand types against what the operands are.
	if ins != nil && opName != "reduce" {
		if len(ins) != len(operands) {
			return shloVal{}, fmt.Errorf("%d operand types for %d operands", len(ins), len(operands))
		}
		for i, t := range ins {
			if !slices.Equal(t.shape, operands[i].typ.shape) || t.dtype != operands[i].typ.dtype {
				return shloVal{}, fmt.Errorf("operand %d is %v, declared %v", i, operands[i].typ, t)
			}
		}
	}
	sameShape := func(n int) error {
		if len(operands) != n {
			return fmt.Errorf("%s takes %d operands, have %d", opName, n, len(operands))
		}
		for _, o := range operands {
			if !slices.Equal(o.typ.shape, out.shape) || o.typ.dtype != out.dtype {
				return fmt.Errorf("%s operand %v does not match result %v", opName, o.typ, out)
			}
		}
		return nil
	}
	var v value
	switch {
	case shloBinary[opName] != "":
		if err := sameShape(2); err != nil {
			return shloVal{}, err
		}
		v, err = l.binary(shloBinary[opName], operands[0].v, operands[1].v)
	case shloUnary[opName] != "":
		if err := sameShape(1); err != nil {
			return shloVal{}, err
		}
		v, err = l.unary(shloUnary[opName], operands[0].v)
	case opName == "constant":
		lit, ok := strings.CutPrefix(strings.TrimSpace(rest), "dense<")
		if !ok || !strings.HasSuffix(lit, ">") || strings.ContainsAny(lit, "[,") {
			return shloVal{}, fmt.Errorf("only splat dense<...> constants are supported")
		}
		var f float32
		if f, err = parseFloatLiteral(strings.TrimSuffix(lit, ">"), out.dtype); err != nil {
			return shloVal{}, err
		}
		v = l.constant(f, out.dtype)
	case opName == "convert":
		if len(operands) != 1 || !slices.Equal(operands[0].typ.shape, out.shape) {
			return shloVal{}, fmt.Errorf("convert must keep the shape")
		}
		v, err = l.convert(operands[0].v, out.dtype)
	case opName == "reshape":
		if len(operands) != 1 || size(operands[0].typ.shape) != size(out.shape) || operands[0].typ.dtype != out.dtype {
			return shloVal{}, fmt.Errorf("reshape must keep the element count and type")
		}
		v, err = l.reshape(operands[0].v, out.shape)
	case opName == "broadcast_in_dim":
		v, err = shloBroadcast(l, operands, attrs, out)
	case opName == "reduce":
		v, err = shloReduce(l, env, rest, ins, out)
	case opName == "dot_general":
		v, err = shloDot(l, operands, attrs, out)
	default:
		return shloVal{}, fmt.Errorf("unsupported operation stablehlo.%s", opName)
	}
	if err != nil {
		return shloVal{}, err
	}
	return shloVal{v, out}, nil
}

func attr(attrs []string, key string) (string, bool) {
	for _, a := range attrs {
		k, v, ok := strings.Cut(a, "=")
		if ok && strings.TrimSpace(k) == key {
			return strings.TrimSpace(v), true
		}
	}
	return "", false
}

// shloBroadcast maps operand dimension i to result dimension dims[i]. The
// operand is viewed with NumPy alignment, reshaping it when its size-1
// dimensions are not already where the result needs them.
func shloBroadcast(l *lowerer, operands []shloVal, attrs []string, out tensorType) (value, error) {
	if len(operands) != 1 {
		return value{}, fmt.Errorf("broadcast_in_dim takes 1 operand")
	}
	o := operands[0]
	ds, ok := attr(attrs, "dims")
	if !ok {
		ds, ok = attr(attrs, "broadcast_dimensions")
	}
	if !ok {
		return value{}, fmt.Errorf("broadcast_in_dim without dims")
	}
	dims, err := parseInts(ds)
	if err != nil {
		return value{}, err
	}
	if len(dims) != len(o.typ.shape) || o.typ.dtype != out.dtype {
		return value{}, fmt.Errorf("broadcast_in_dim of %v with dims %v to %v", o.typ, dims, out)
	}
	view := make([]int, len(out.shape))
	for i := range view {
		view[i] = 1
	}
	for i, d := range dims {
		if d < 0 || d >= len(out.shape) || i > 0 && d <= dims[i-1] {
			return value{}, fmt.Errorf("broadcast_in_dim dims %v must increase within rank %d", dims, len(out.shape))
		}
		if s := o.typ.shape[i]; s != 1 && s != out.shape[d] {
			return value{}, fmt.Errorf("broadcast_in_dim dimension %d of size %d cannot become %d", i, s, out.shape[d])
		}
		view[d] = o.typ.shape[i]
	}
	if o.v.konst != nil {
		return o.v, nil
	}
	trim := func(s []int) []int {
		for len(s) > 0 && s[0] == 1 {
			s = s[1:]
		}
		return s
	}
	if slices.Equal(trim(o.v.shape), trim(view)) {
		return o.v, nil
	}
	return l.reshape(o.v, view)
}

// shloReduce handles
//
//	stablehlo.reduce(%x init: %c) applies stablehlo.add across dimensions = [d]
func shloReduce(l *lowerer, env map[string]shloVal, rest string, ins []tensorType, out tensorType) (value, error) {
	rest = strings.TrimSpace(rest)
	args, tail, ok := strings.Cut(strings.TrimPrefix(rest, "("), ")")
	if !ok || !strings.HasPrefix(rest, "(") {
		return value{}, fmt.Errorf("bad reduce operands")
	}
	xs, cs, ok := strings.Cut(args, "init:")
	if !ok {
		return value{}, fmt.Errorf("reduce without init")
	}
	x, ok1 := env[strings.TrimSpace(xs)]
	c, ok2 := env[strings.TrimSpace(cs)]
	if !ok1 || !ok2 {
		return value{}, fmt.Errorf("reduce of undefined value")
	}
	if len(ins) != 2 || !slices.Equal(ins[0].shape, x.typ.shape) || ins[0].dtype != x.typ.dtype {
		return value{}, fmt.Errorf("reduce operand is %v, declared %v", x.typ, ins)
	}
	f := strings.Fields(tail)
	if len(f) < 6 || f[0] != "applies" || f[2] != "across" || f[3] != "dimensions" || f[4] != "=" {
		return value{}, fmt.Errorf("bad reduce body %q", tail)
	}
	dims, err := parseInts(strings.Join(f[5:], ""))
	if err != nil {
		return value{}, err
	}
	rank := len(x.typ.shape)
	if len(dims) != 1 || dims[0] != rank-1 {
		return value{}, fmt.Errorf("reduce across %v: only the last dimension (%d) is supported", dims, rank-1)
	}
	want := x.typ.shape[:rank-1]
	if !slices.Equal(want, out.shape) || out.dtype != x.typ.dtype {
		return value{}, fmt.Errorf("reduce of %v to %v: rank or shape mismatch", x.typ, out)
	}
	if c.v.konst == nil {
		return value{}, fmt.Errorf("reduce init must be a constant")
	}
	var op string
	switch f[1] {
	case "stablehlo.add":
		op = "add"
		if *c.v.konst != 0 {
			return value{}, fmt.Errorf("reduce add init %v, want 0", *c.v.konst)
		}
	case "stablehlo.maximum":
		op = "max"
		if !math.IsInf(float64(*c.v.konst), -1) {
			return value{}, fmt.Errorf("reduce maximum init %v, want -inf", *c.v.konst)
		}
	default:
		return value{}, fmt.Errorf("unsupported reduce body %s", f[1])
	}
	return l.reduce(op, x.v)
}

func shloDot(l *lowerer, operands []shloVal, attrs []string, out tensorType) (value, error) {
	if len(operands) != 2 {
		return value{}, fmt.Errorf("dot_general takes 2 operands")
	}
	if _, ok := attr(attrs, "batching_dims"); ok {
		return value{}, fmt.Errorf("dot_general batching dimensions are not supported")
	}
	cd, ok := attr(attrs, "contracting_dims")
	if !ok {
		return value{}, fmt.Errorf("dot_general without contracting_dims")
	}
	ls, rs, ok := strings.Cut(cd, " x ")
	if !ok {
		return value{}, fmt.Errorf("bad contracting_dims %q", cd)
	}
	lc, err1 := parseInts(ls)
	rc, err2 := parseInts(rs)
	if err1 != nil || err2 != nil || len(lc) != 1 || len(rc) != 1 || lc[0] != 1 || rc[0] < 0 || rc[0] > 1 {
		return value{}, fmt.Errorf("contracting_dims %q: want [1] x [0] or [1] x [1]", cd)
	}
	x, w := operands[0], operands[1]
	if len(x.typ.shape) != 2 || len(w.typ.shape) != 2 {
		return value{}, fmt.Errorf("dot_general of ranks %d and %d, want 2 and 2", len(x.typ.shape), len(w.typ.shape))
	}
	transposed := rc[0] == 0
	n := w.typ.shape[1-rc[0]]
	if !slices.Equal(out.shape, []int{x.typ.shape[0], n}) {
		return value{}, fmt.Errorf("dot_general result %v, want [%d %d]", out, x.typ.shape[0], n)
	}
	v, err := l.matmul(x.v, w.v, transposed)
	if err != nil {
		return value{}, err
	}
	if out.dtype == hopper.BF16 {
		return l.convert(v, hopper.BF16)
	}
	return v, nil
}
