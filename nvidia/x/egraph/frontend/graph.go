package frontend

import (
	"fmt"
	"slices"
	"strconv"
	"strings"

	"github.com/tmc/lib/nvidia/x/egraph"
	"github.com/tmc/lib/nvidia/x/egraph/hopper"
)

// A Weight is a parameter tensor of an imported graph.
type Weight struct {
	// Name is the name in the source (HuggingFace tensor name, StableHLO
	// argument, ONNX initializer or input).
	Name string
	// Leaf is the name of the hopper Tensor leaf that reads it.
	Leaf string
	// DType and Shape are the leaf's. A matmul weight is [N, K].
	DType hopper.DType
	Shape []int
	// Transposed reports that the source stores the weight [K, N] and
	// the loader must transpose it into the leaf.
	Transposed bool
}

// An ImportResult is an imported graph.
type ImportResult struct {
	Graph *egraph.EGraph
	Root  egraph.Id
	// Expr is the term [Optimize] extracted for Root, and Plan its
	// kernels from [hopper.Compile] (named "block", output F32).
	Expr *egraph.Expr
	Plan *hopper.Plan
	// Inputs are the activation leaves, Weights the parameter leaves.
	Inputs  []hopper.Tensor
	Weights []Weight
	// WeightMap maps the source weight names of the imported layer to
	// their shapes, including weights that were not lowered.
	WeightMap map[string][]int
	// Unlowered names parts of the source that were recognized but not
	// lowered to terms.
	Unlowered []string
}

// A value is an imported tensor: a term, a pending leaf, or a splat
// scalar constant.
type value struct {
	id     egraph.Id
	shape  []int
	dtype  hopper.DType
	konst  *float32
	leaf   *pendingLeaf
	weight bool
	// cast holds the term's storage type when it is a BF16 leaf or cast,
	// which a matmul needs; map results are FP32.
	cast hopper.DType
}

type pendingLeaf struct {
	name  string
	ids   map[bool]egraph.Id // by transposed
	shape []int
}

// lowerer builds hopper terms for an importer.
type lowerer struct {
	b       *hopper.Builder
	inputs  []hopper.Tensor
	weights []Weight
	names   map[string]bool
}

func newLowerer() *lowerer {
	return &lowerer{b: hopper.NewBuilder(), names: make(map[string]bool)}
}

// leafName turns a source name into a unique C identifier.
func (l *lowerer) leafName(src string) string {
	var sb strings.Builder
	for i, r := range src {
		switch {
		case r == '_' || r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z':
			sb.WriteRune(r)
		case r >= '0' && r <= '9':
			if i == 0 {
				sb.WriteString("t_")
			}
			sb.WriteRune(r)
		default:
			sb.WriteByte('_')
		}
	}
	name := strings.Trim(sb.String(), "_")
	if name == "" {
		name = "t"
	}
	base := name
	for i := 2; l.names[name]; i++ {
		name = base + "_" + strconv.Itoa(i)
	}
	l.names[name] = true
	return name
}

// input declares an activation leaf.
func (l *lowerer) input(src string, d hopper.DType, shape []int) value {
	name := l.leafName(src)
	l.inputs = append(l.inputs, hopper.Tensor{Name: name, DType: d, Shape: slices.Clone(shape)})
	return value{id: l.b.Input(name, d, shape...), shape: slices.Clone(shape), dtype: d, cast: d}
}

// weight declares a parameter whose leaf is created on first use, so a
// matmul operand stored [K, N] can become an [N, K] leaf.
func (l *lowerer) weight(src string, d hopper.DType, shape []int) value {
	return value{shape: slices.Clone(shape), dtype: d, weight: true, cast: d,
		leaf: &pendingLeaf{name: src, ids: make(map[bool]egraph.Id), shape: slices.Clone(shape)}}
}

func (l *lowerer) constant(f float32, d hopper.DType) value {
	return value{konst: &f, dtype: d}
}

// term returns v's term, creating its leaf untransposed if pending.
func (l *lowerer) term(v value) (egraph.Id, error) {
	if v.konst != nil {
		return 0, fmt.Errorf("constant %v used as a tensor", *v.konst)
	}
	if v.leaf != nil {
		return l.leafTerm(v, false), nil
	}
	return v.id, nil
}

func (l *lowerer) leafTerm(v value, transposed bool) egraph.Id {
	p := v.leaf
	if id, ok := p.ids[transposed]; ok {
		return id
	}
	shape := slices.Clone(p.shape)
	if transposed {
		shape[0], shape[1] = shape[1], shape[0]
	}
	name := l.leafName(p.name)
	l.weights = append(l.weights, Weight{Name: p.name, Leaf: name, DType: v.cast, Shape: shape, Transposed: transposed})
	id := l.b.Input(name, v.cast, shape...)
	p.ids[transposed] = id
	return id
}

// broadcastShape is NumPy broadcasting.
func broadcastShape(a, b []int) ([]int, error) {
	n := max(len(a), len(b))
	out := make([]int, n)
	for i := range n {
		x, y := 1, 1
		if j := len(a) - n + i; j >= 0 {
			x = a[j]
		}
		if j := len(b) - n + i; j >= 0 {
			y = b[j]
		}
		switch {
		case x == y, y == 1:
			out[i] = x
		case x == 1:
			out[i] = y
		default:
			return nil, fmt.Errorf("shapes %v and %v do not broadcast", a, b)
		}
	}
	return out, nil
}

func fmtConst(f float32) string {
	return strconv.FormatFloat(float64(f), 'g', -1, 32)
}

// rounded wraps body in the storage rounding of d.
func rounded(body string, d hopper.DType) string {
	if d == hopper.BF16 {
		return "(bf16 " + body + ")"
	}
	return body
}

// unary applies a lambda operator to x.
func (l *lowerer) unary(op string, x value) (value, error) {
	if x.konst != nil {
		k := hopper.MustLambda(rounded("("+op+" "+fmtConst(*x.konst)+")", x.dtype)).Eval()
		return l.constant(k, x.dtype), nil
	}
	return l.mapN(rounded("("+op+" $0)", x.dtype), x.dtype, x)
}

// binary applies a lambda operator to x and y, folding a constant operand
// into the lambda.
func (l *lowerer) binary(op string, x, y value) (value, error) {
	switch {
	case x.konst != nil && y.konst != nil:
		k := hopper.MustLambda(rounded("("+op+" "+fmtConst(*x.konst)+" "+fmtConst(*y.konst)+")", x.dtype)).Eval()
		return l.constant(k, x.dtype), nil
	case y.konst != nil:
		return l.mapN(rounded("("+op+" $0 "+fmtConst(*y.konst)+")", x.dtype), x.dtype, x)
	case x.konst != nil:
		return l.mapN(rounded("("+op+" "+fmtConst(*x.konst)+" $0)", y.dtype), y.dtype, y)
	}
	if x.dtype != y.dtype {
		return value{}, fmt.Errorf("%s of %v and %v", op, x.dtype, y.dtype)
	}
	return l.mapN(rounded("("+op+" $0 $1)", x.dtype), x.dtype, x, y)
}

func (l *lowerer) mapN(src string, d hopper.DType, args ...value) (value, error) {
	fn, err := hopper.ParseLambda(src)
	if err != nil {
		return value{}, err
	}
	var shape []int
	ids := make([]egraph.Id, len(args))
	for i, a := range args {
		if shape, err = broadcastShape(shape, a.shape); err != nil {
			return value{}, err
		}
		if ids[i], err = l.term(a); err != nil {
			return value{}, err
		}
	}
	return value{id: l.b.Map(fn, ids...), shape: shape, dtype: d}, nil
}

// convert changes x's element type: to BF16 rounds, to F32 is exact.
func (l *lowerer) convert(x value, d hopper.DType) (value, error) {
	if x.dtype == d || d == hopper.F32 {
		x.dtype = d
		return x, nil
	}
	if d != hopper.BF16 {
		return value{}, fmt.Errorf("convert to %v is not supported", d)
	}
	if x.konst != nil {
		return l.constant(hopper.BF16.Round(*x.konst), d), nil
	}
	id, err := l.term(x)
	if err != nil {
		return value{}, err
	}
	return value{id: l.b.Cast(hopper.BF16, id), shape: x.shape, dtype: d, cast: d}, nil
}

// reduce sums (or maxes) x over its last axis, keeping it as size 1.
func (l *lowerer) reduce(op string, x value) (value, error) {
	id, err := l.term(x)
	if err != nil {
		return value{}, err
	}
	if len(x.shape) == 0 {
		return value{}, fmt.Errorf("reduce of a scalar")
	}
	shape := slices.Clone(x.shape)
	shape[len(shape)-1] = 1
	switch op {
	case "add":
		id = l.b.ReduceSum(id)
	case "max":
		id = l.b.ReduceMax(id)
	default:
		return value{}, fmt.Errorf("reduce with %s is not supported", op)
	}
	return value{id: id, shape: shape, dtype: x.dtype}, nil
}

// reshape gives x a new shape with the same element count.
func (l *lowerer) reshape(x value, shape []int) (value, error) {
	if x.konst != nil {
		return x, nil
	}
	if size(x.shape) != size(shape) {
		return value{}, fmt.Errorf("reshape %v to %v changes the element count", x.shape, shape)
	}
	if slices.Equal(x.shape, shape) {
		return x, nil
	}
	id, err := l.term(x)
	if err != nil {
		return value{}, err
	}
	return value{id: l.b.Reshape(id, shape...), shape: slices.Clone(shape), dtype: x.dtype, cast: x.cast}, nil
}

// matmul is x [M, K] times w. If transposed, w is stored [K, N] in the
// source and must be a weight; otherwise it is [N, K].
func (l *lowerer) matmul(x, w value, transposed bool) (value, error) {
	if len(x.shape) != 2 || len(w.shape) != 2 {
		return value{}, fmt.Errorf("matmul of ranks %d and %d, want 2 and 2", len(x.shape), len(w.shape))
	}
	k, n := w.shape[1], w.shape[0]
	if transposed {
		k, n = n, k
	}
	if x.shape[1] != k {
		return value{}, fmt.Errorf("matmul of %v and %v: contracting sizes %d and %d differ", x.shape, w.shape, x.shape[1], k)
	}
	if x.cast != hopper.BF16 || w.cast != hopper.BF16 {
		return value{}, fmt.Errorf("matmul operands must be stored BF16 (convert them)")
	}
	xid, err := l.term(x)
	if err != nil {
		return value{}, err
	}
	var wid egraph.Id
	switch {
	case w.leaf != nil:
		wid = l.leafTerm(w, transposed)
	case transposed:
		return value{}, fmt.Errorf("matmul operand stored [K, N] must be a weight")
	default:
		if wid, err = l.term(w); err != nil {
			return value{}, err
		}
	}
	return value{id: l.b.MatMul(xid, wid), shape: []int{x.shape[0], n}, dtype: hopper.F32}, nil
}

// rmsNorm is the canonical RMSNorm decomposition all importers use:
// x·rsqrt(Σx²/D + eps)·gamma, in FP32 with x read as F32.
func (l *lowerer) rmsNorm(x, gamma value, eps float32) (value, error) {
	d := x.shape[len(x.shape)-1]
	x, err := l.convert(x, hopper.F32)
	if err != nil {
		return value{}, err
	}
	sq, err := l.binary("mul", x, x)
	if err != nil {
		return value{}, err
	}
	ss, err := l.reduce("add", sq)
	if err != nil {
		return value{}, err
	}
	ms, err := l.binary("div", ss, l.constant(float32(d), hopper.F32))
	if err != nil {
		return value{}, err
	}
	v, err := l.binary("add", ms, l.constant(eps, hopper.F32))
	if err != nil {
		return value{}, err
	}
	r, err := l.unary("rsqrt", v)
	if err != nil {
		return value{}, err
	}
	y, err := l.binary("mul", x, r)
	if err != nil {
		return value{}, err
	}
	if gamma, err = l.convert(gamma, hopper.F32); err != nil {
		return value{}, err
	}
	return l.binary("mul", y, gamma)
}

func size(shape []int) int {
	n := 1
	for _, d := range shape {
		n *= d
	}
	return n
}

// finish optimizes and compiles root.
func (l *lowerer) finish(root value) (ImportResult, error) {
	id, err := l.term(root)
	if err != nil {
		return ImportResult{}, err
	}
	res := ImportResult{Graph: l.b.G, Root: id, Inputs: l.inputs, Weights: l.weights}
	if res.Expr, err = Optimize(l.b.G, id); err != nil {
		return res, err
	}
	if res.Plan, err = hopper.Compile(res.Expr, "block", hopper.F32); err != nil {
		return res, err
	}
	return res, nil
}

// Optimize saturates g with [hopper.Rules], [FuseMaps] and
// [hopper.FuseEpilogues] and extracts the cheapest term for root with
// [hopper.Extract].
func Optimize(g *egraph.EGraph, root egraph.Id) (*egraph.Expr, error) {
	for range 8 {
		if _, err := egraph.SaturateChecked(g, hopper.Rules(), egraph.SaturationOptions{MaxIterations: 16}); err != nil {
			return nil, fmt.Errorf("optimize: %w", err)
		}
		n, err := FuseMaps(g)
		if err != nil {
			return nil, fmt.Errorf("optimize: %w", err)
		}
		m, err := hopper.FuseEpilogues(g)
		if err != nil {
			return nil, fmt.Errorf("optimize: %w", err)
		}
		if n+m == 0 {
			break
		}
	}
	expr, err := hopper.Extract(g, root)
	if err != nil {
		return nil, fmt.Errorf("optimize: %w", err)
	}
	return expr, nil
}

// maxFusedLambda bounds the size of a composed lambda's source.
const maxFusedLambda = 512

// FuseMaps adds, for every (map f a...) whose operand a_i is itself
// (map g b...), the equivalent (map f[$i := g] a... b...), and rewrites
// every map to its normal form: duplicate operands merged and operands
// renumbered in order of first use in the lambda. Both are exact in FP32:
// the lambdas evaluate the same operations in the same order. It returns
// how many classes gained a node and rebuilds the graph.
func FuseMaps(g *egraph.EGraph) (int, error) {
	added := 0
	for _, class := range g.Classes() {
		for _, n := range class.Nodes {
			if n.Op != hopper.OpMap || len(n.Children) < 2 {
				continue
			}
			f, ok := lambdaSource(g, n.Children[0])
			if !ok {
				continue
			}
			args := n.Children[1:]
			cands := [][2]any{{f, args}}
			for i, a := range args {
				for _, inner := range g.Class(a).Nodes {
					if inner.Op != hopper.OpMap || len(inner.Children) < 2 {
						continue
					}
					gl, ok := lambdaSource(g, inner.Children[0])
					if !ok {
						continue
					}
					fused, fargs, err := compose(f, args, i, gl, inner.Children[1:])
					if err != nil {
						return added, err
					}
					if len(fused) <= maxFusedLambda {
						cands = append(cands, [2]any{fused, fargs})
					}
				}
			}
			for _, c := range cands {
				src, cargs, err := normalize(g, c[0].(string), c[1].([]egraph.Id))
				if err != nil {
					return added, err
				}
				fn, err := hopper.ParseLambda(src)
				if err != nil {
					return added, fmt.Errorf("fuse maps: %w", err)
				}
				kids := append([]egraph.Id{g.Add(egraph.ENode{Op: "fn:" + fn.String()})}, cargs...)
				if _, merged := g.Union(class.ID, g.Add(egraph.NewNode(hopper.OpMap, kids...))); merged {
					added++
				}
			}
		}
	}
	g.Rebuild()
	return added, nil
}

func lambdaSource(g *egraph.EGraph, id egraph.Id) (string, bool) {
	for _, n := range g.Class(id).Nodes {
		if src, ok := strings.CutPrefix(n.Op, "fn:"); ok && len(n.Children) == 0 {
			return src, true
		}
	}
	return "", false
}

// A lnode is a parsed lambda: an operator with operands, an operand
// reference $arg, or a numeric literal.
type lnode struct {
	op   string
	arg  int
	kids []*lnode
}

func parseLambda(s string) (*lnode, error) {
	toks := strings.Fields(strings.NewReplacer("(", " ( ", ")", " ) ").Replace(s))
	pos := 0
	var parse func(depth int) (*lnode, error)
	parse = func(depth int) (*lnode, error) {
		if depth > 256 || pos >= len(toks) {
			return nil, fmt.Errorf("lambda %q: unexpected end", s)
		}
		t := toks[pos]
		pos++
		switch {
		case t == "(":
			if pos >= len(toks) {
				return nil, fmt.Errorf("lambda %q: unexpected end", s)
			}
			n := &lnode{op: toks[pos]}
			pos++
			for pos < len(toks) && toks[pos] != ")" {
				k, err := parse(depth + 1)
				if err != nil {
					return nil, err
				}
				n.kids = append(n.kids, k)
			}
			if pos >= len(toks) {
				return nil, fmt.Errorf("lambda %q: missing )", s)
			}
			pos++
			return n, nil
		case strings.HasPrefix(t, "$"):
			i, err := strconv.Atoi(t[1:])
			if err != nil || i < 0 {
				return nil, fmt.Errorf("lambda %q: bad operand %s", s, t)
			}
			return &lnode{op: "$", arg: i}, nil
		}
		return &lnode{op: t}, nil
	}
	n, err := parse(0)
	if err == nil && pos != len(toks) {
		err = fmt.Errorf("lambda %q: trailing tokens", s)
	}
	return n, err
}

func (n *lnode) String() string {
	switch {
	case n.op == "$":
		return "$" + strconv.Itoa(n.arg)
	case n.kids == nil:
		return n.op
	}
	parts := []string{n.op}
	for _, k := range n.kids {
		parts = append(parts, k.String())
	}
	return "(" + strings.Join(parts, " ") + ")"
}

func (n *lnode) walk(f func(*lnode)) {
	f(n)
	for _, k := range n.kids {
		k.walk(f)
	}
}

// compose substitutes g (over gargs) for operand i of f (over fargs).
func compose(f string, fargs []egraph.Id, i int, g string, gargs []egraph.Id) (string, []egraph.Id, error) {
	fn, err := parseLambda(f)
	if err != nil {
		return "", nil, err
	}
	gn, err := parseLambda(g)
	if err != nil {
		return "", nil, err
	}
	off := len(fargs)
	var subst func(n *lnode) *lnode
	subst = func(n *lnode) *lnode {
		if n.op == "$" && n.arg == i {
			c := cloneShift(gn, off)
			return c
		}
		c := &lnode{op: n.op, arg: n.arg}
		for _, k := range n.kids {
			c.kids = append(c.kids, subst(k))
		}
		return c
	}
	out := subst(fn)
	return out.String(), append(slices.Clone(fargs), gargs...), nil
}

func cloneShift(n *lnode, off int) *lnode {
	c := &lnode{op: n.op, arg: n.arg}
	if n.op == "$" {
		c.arg += off
	}
	for _, k := range n.kids {
		c.kids = append(c.kids, cloneShift(k, off))
	}
	return c
}

// normalize merges operands in the same e-class and renumbers operands in
// order of first use.
func normalize(g *egraph.EGraph, src string, args []egraph.Id) (string, []egraph.Id, error) {
	n, err := parseLambda(src)
	if err != nil {
		return "", nil, err
	}
	slot := make(map[egraph.Id]int)
	var out []egraph.Id
	var bad error
	n.walk(func(m *lnode) {
		if m.op != "$" {
			return
		}
		if m.arg >= len(args) {
			bad = fmt.Errorf("lambda %q uses operand %d of %d", src, m.arg, len(args))
			return
		}
		c := g.Find(args[m.arg])
		s, ok := slot[c]
		if !ok {
			s = len(out)
			slot[c] = s
			out = append(out, c)
		}
		m.arg = s
	})
	if bad != nil {
		return "", nil, bad
	}
	return n.String(), out, nil
}

// CanonicalFingerprint is e's fingerprint with each Tensor leaf renamed
// by its first appearance in a depth-first walk, so terms that differ only
// in leaf names agree.
func CanonicalFingerprint(e *egraph.Expr) string {
	names := make(map[string]string)
	var walk func(e *egraph.Expr) *egraph.Expr
	walk = func(e *egraph.Expr) *egraph.Expr {
		c := *e
		if t, ok := e.Data.(hopper.Tensor); ok {
			n, ok := names[t.Name]
			if !ok {
				n = "v" + strconv.Itoa(len(names))
				names[t.Name] = n
			}
			t.Name = n
			c.Op = "t:" + n
			c.Data = t
		}
		c.Children = nil
		for _, k := range e.Children {
			c.Children = append(c.Children, walk(k))
		}
		return &c
	}
	return walk(e).Fingerprint()
}
