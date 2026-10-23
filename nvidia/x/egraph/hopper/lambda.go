package hopper

import (
	"fmt"
	"math"
	"strconv"
	"strings"
)

// A Lambda is a scalar FP32 function of operands $0, $1, ..., written as an
// S-expression such as "(mul $0 (sigmoid $1))" or "(add $0 1.5)". It is the
// body of elementwise maps and of GEMM and attention epilogues.
//
// Operators:
//
//	add sub mul div max min        binary
//	neg abs exp log sqrt rsqrt     unary
//	recip sigmoid silu cos sin
//	bf16 e4m3                      round to that storage type and back
//
// Arithmetic is IEEE FP32 in the order written; exp, log, sigmoid, silu, cos
// and sin may differ from CUDA's expf and friends by a few ulp.
type Lambda struct {
	src   string
	root  *lambdaNode
	arity int
}

type lambdaNode struct {
	op    string // operator, "$" for an operand, "#" for a constant
	arg   int
	value float32
	kids  []*lambdaNode
}

var lambdaArity = map[string]int{
	"add": 2, "sub": 2, "mul": 2, "div": 2, "max": 2, "min": 2,
	"neg": 1, "abs": 1, "exp": 1, "log": 1, "sqrt": 1, "rsqrt": 1, "recip": 1,
	"sigmoid": 1, "silu": 1, "cos": 1, "sin": 1, "bf16": 1, "e4m3": 1,
}

// Identity is the lambda $0.
var Identity = MustLambda("$0")

// ParseLambda parses a lambda body.
func ParseLambda(s string) (Lambda, error) {
	toks := strings.Fields(strings.NewReplacer("(", " ( ", ")", " ) ").Replace(s))
	l := Lambda{src: strings.Join(strings.Fields(s), " ")}
	pos := 0
	root, err := l.parse(toks, &pos, 0)
	if err != nil {
		return Lambda{}, fmt.Errorf("parse lambda %q: %w", s, err)
	}
	if pos != len(toks) {
		return Lambda{}, fmt.Errorf("parse lambda %q: trailing %q", s, toks[pos])
	}
	l.root = root
	return l, nil
}

// MustLambda is like ParseLambda but panics on error. It is for package-level
// lambdas whose text is a constant.
func MustLambda(s string) Lambda {
	l, err := ParseLambda(s)
	if err != nil {
		panic(err)
	}
	return l
}

func (l *Lambda) parse(toks []string, pos *int, depth int) (*lambdaNode, error) {
	if depth > 64 {
		return nil, fmt.Errorf("too deep")
	}
	if *pos >= len(toks) {
		return nil, fmt.Errorf("unexpected end")
	}
	tok := toks[*pos]
	*pos++
	switch {
	case tok == "(":
		if *pos >= len(toks) {
			return nil, fmt.Errorf("unexpected end")
		}
		op := toks[*pos]
		*pos++
		n, ok := lambdaArity[op]
		if !ok {
			return nil, fmt.Errorf("unknown operator %q", op)
		}
		node := &lambdaNode{op: op}
		for i := 0; i < n; i++ {
			kid, err := l.parse(toks, pos, depth+1)
			if err != nil {
				return nil, err
			}
			node.kids = append(node.kids, kid)
		}
		if *pos >= len(toks) || toks[*pos] != ")" {
			return nil, fmt.Errorf("%s takes %d operands", op, n)
		}
		*pos++
		return node, nil
	case tok == ")":
		return nil, fmt.Errorf("unexpected )")
	case strings.HasPrefix(tok, "$"):
		i, err := strconv.Atoi(tok[1:])
		if err != nil || i < 0 || i > 31 {
			return nil, fmt.Errorf("bad operand %q", tok)
		}
		l.arity = max(l.arity, i+1)
		return &lambdaNode{op: "$", arg: i}, nil
	default:
		f, err := strconv.ParseFloat(tok, 32)
		if err != nil || math.IsInf(f, 0) || math.IsNaN(f) {
			return nil, fmt.Errorf("bad constant %q", tok)
		}
		return &lambdaNode{op: "#", value: float32(f)}, nil
	}
}

// String returns the lambda's normalized source.
func (l Lambda) String() string { return l.src }

// Arity reports one more than the largest operand index used.
func (l Lambda) Arity() int { return l.arity }

// IsIdentity reports whether l is $0.
func (l Lambda) IsIdentity() bool { return l.root != nil && l.root.op == "$" && l.root.arg == 0 }

// Eval applies l to args in FP32.
func (l Lambda) Eval(args ...float32) float32 { return l.root.eval(args) }

func (n *lambdaNode) eval(args []float32) float32 {
	switch n.op {
	case "$":
		return args[n.arg]
	case "#":
		return n.value
	}
	a := n.kids[0].eval(args)
	var b float32
	if len(n.kids) > 1 {
		b = n.kids[1].eval(args)
	}
	f := func(g func(float64) float64) float32 { return float32(g(float64(a))) }
	switch n.op {
	case "add":
		return a + b
	case "sub":
		return a - b
	case "mul":
		return a * b
	case "div":
		return a / b
	case "max":
		return float32(math.Max(float64(a), float64(b)))
	case "min":
		return float32(math.Min(float64(a), float64(b)))
	case "neg":
		return -a
	case "abs":
		return float32(math.Abs(float64(a)))
	case "exp":
		return f(math.Exp)
	case "log":
		return f(math.Log)
	case "sqrt":
		return f(math.Sqrt)
	case "rsqrt":
		return f(func(x float64) float64 { return 1 / math.Sqrt(x) })
	case "recip":
		return 1 / a
	case "sigmoid":
		return 1 / (1 + f(func(x float64) float64 { return math.Exp(-x) }))
	case "silu":
		return a / (1 + f(func(x float64) float64 { return math.Exp(-x) }))
	case "cos":
		return f(math.Cos)
	case "sin":
		return f(math.Sin)
	case "bf16":
		return BF16.Round(a)
	case "e4m3":
		return E4M3.Round(a)
	}
	panic("unreachable: lambda operator " + n.op)
}

// CUDA returns a C expression computing l from the given operand expressions.
func (l Lambda) CUDA(args ...string) (string, error) {
	if len(args) < l.arity {
		return "", fmt.Errorf("lambda %s needs %d operands, have %d", l, l.arity, len(args))
	}
	return l.root.cuda(args), nil
}

func (n *lambdaNode) cuda(args []string) string {
	switch n.op {
	case "$":
		return "(" + args[n.arg] + ")"
	case "#":
		return floatLiteral(n.value)
	}
	k := make([]string, len(n.kids))
	for i, kid := range n.kids {
		k[i] = kid.cuda(args)
	}
	switch n.op {
	case "add", "sub", "mul", "div":
		sym := map[string]string{"add": "+", "sub": "-", "mul": "*", "div": "/"}[n.op]
		return "(" + k[0] + " " + sym + " " + k[1] + ")"
	case "max":
		return "fmaxf(" + k[0] + ", " + k[1] + ")"
	case "min":
		return "fminf(" + k[0] + ", " + k[1] + ")"
	case "neg":
		return "(-" + k[0] + ")"
	case "abs":
		return "fabsf(" + k[0] + ")"
	case "exp":
		return "expf(" + k[0] + ")"
	case "log":
		return "logf(" + k[0] + ")"
	case "sqrt":
		return "sqrtf(" + k[0] + ")"
	case "rsqrt":
		return "rsqrtf(" + k[0] + ")"
	case "recip":
		return "(1.0f / " + k[0] + ")"
	case "sigmoid":
		return "eg_sigmoid(" + k[0] + ")"
	case "silu":
		return "eg_silu(" + k[0] + ")"
	case "cos":
		return "cosf(" + k[0] + ")"
	case "sin":
		return "sinf(" + k[0] + ")"
	case "bf16":
		return "eg_bf16_to_f32(eg_f32_to_bf16(" + k[0] + "))"
	case "e4m3":
		return "eg_e4m3_to_f32(eg_f32x2_to_e4m3x2(" + k[0] + ", 0.0f) & 0xffu)"
	}
	panic("unreachable: lambda operator " + n.op)
}

// Compose returns l with operand i replaced by inner, whose operands are
// renumbered to start at l.Arity() (so l's other operands keep their
// indices). Compose("(mul $0 $1)", 0, "(exp $0)") is "(mul (exp $2) $1)".
func (l Lambda) Compose(i int, inner Lambda) Lambda {
	shift := l.arity
	var sub func(n *lambdaNode) *lambdaNode
	shifted := func(n *lambdaNode) *lambdaNode {
		var walk func(*lambdaNode) *lambdaNode
		walk = func(n *lambdaNode) *lambdaNode {
			c := *n
			if c.op == "$" {
				c.arg += shift
			}
			c.kids = nil
			for _, k := range n.kids {
				c.kids = append(c.kids, walk(k))
			}
			return &c
		}
		return walk(n)
	}
	sub = func(n *lambdaNode) *lambdaNode {
		if n.op == "$" && n.arg == i {
			return shifted(inner.root)
		}
		c := *n
		c.kids = nil
		for _, k := range n.kids {
			c.kids = append(c.kids, sub(k))
		}
		return &c
	}
	root := sub(l.root)
	out := Lambda{root: root, arity: max(l.arity, shift+inner.arity)}
	out.src = root.format()
	return out
}

func (n *lambdaNode) format() string {
	switch n.op {
	case "$":
		return "$" + strconv.Itoa(n.arg)
	case "#":
		return strconv.FormatFloat(float64(n.value), 'g', -1, 32)
	}
	parts := []string{n.op}
	for _, k := range n.kids {
		parts = append(parts, k.format())
	}
	return "(" + strings.Join(parts, " ") + ")"
}

func floatLiteral(v float32) string {
	s := strconv.FormatFloat(float64(v), 'g', -1, 32)
	if !strings.ContainsAny(s, ".eE") {
		s += ".0"
	}
	return s + "f"
}
