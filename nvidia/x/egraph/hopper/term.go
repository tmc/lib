package hopper

import (
	"fmt"
	"math"
	"slices"
	"strings"

	"github.com/tmc/lib/nvidia/x/egraph"
)

// Term operators. Every term denotes a row-major tensor; shapes are inferred
// from the Tensor leaves.
//
//	t:NAME                   a Tensor leaf (payload Tensor)
//	fn:LAMBDA                a lambda leaf (the op is "fn:" + its source)
//	(map fn a b ...)         elementwise fn, NumPy broadcasting
//	(reduce_sum x)           sum over the last axis, kept as size 1
//	(reduce_max x)           max over the last axis, kept as size 1
//	(slice x start n)        columns [start, start+n) of the last axis
//	(reshape x dims)         same elements, new shape (dims leaf payload Dims)
//	(matmul x w)             x [M, K] · w [N, K]ᵀ with FP32 accumulation
//	(nvfp4 packed sf gs)     an NVFP4 weight [N, K] (WeightNVFP4)
//	(hilo hi lo)             a BF16 hi + lo weight (WeightBF16HiLo)
//	(cast dtype x)           round to dtype (dtype leaf payload DType)
//	(col x) (sgn x)          column index, and -1/+1 for even/odd columns, shaped like x
//	(pair x)                 x with columns 2i and 2i+1 swapped
//	(gemm fn x w e1 ...)     map(fn, matmul(x, w), e1, ...): an epilogue-fused GEMM
//
// matmul is not lowerable; [Rules] rewrite it to gemm, and [FuseEpilogues]
// folds maps over a gemm into its epilogue.
const (
	OpMap       = "map"
	OpReduceSum = "reduce_sum"
	OpReduceMax = "reduce_max"
	OpSlice     = "slice"
	OpReshape   = "reshape"
	OpMatMul    = "matmul"
	OpNVFP4     = "nvfp4"
	OpHiLo      = "hilo"
	OpCast      = "cast"
	OpCol       = "col"
	OpSgn       = "sgn"
	OpPair      = "pair"
	OpGEMM      = "gemm"
)

// A Tensor is a kernel argument: the payload of a t:NAME leaf.
type Tensor struct {
	Name  string
	DType DType
	Shape []int
}

// Dims is the payload of a reshape's shape leaf.
type Dims struct{ Shape []int }

// Builder adds terms to an e-graph.
type Builder struct {
	G *egraph.EGraph
}

// NewBuilder returns a Builder over a new e-graph.
func NewBuilder() *Builder { return &Builder{G: egraph.New()} }

func (b *Builder) node(op string, kids ...egraph.Id) egraph.Id {
	return b.G.Add(egraph.NewNode(op, kids...))
}

// Input adds a tensor argument.
func (b *Builder) Input(name string, d DType, shape ...int) egraph.Id {
	return b.G.Add(egraph.ENode{Op: "t:" + name, Data: Tensor{Name: name, DType: d, Shape: slices.Clone(shape)}})
}

func (b *Builder) fn(l Lambda) egraph.Id { return b.G.Add(egraph.ENode{Op: "fn:" + l.String()}) }

// Map applies fn elementwise.
func (b *Builder) Map(fn Lambda, args ...egraph.Id) egraph.Id {
	return b.node(OpMap, append([]egraph.Id{b.fn(fn)}, args...)...)
}

// ReduceSum sums the last axis, keeping it as size 1.
func (b *Builder) ReduceSum(x egraph.Id) egraph.Id { return b.node(OpReduceSum, x) }

// ReduceMax takes the max over the last axis, keeping it as size 1.
func (b *Builder) ReduceMax(x egraph.Id) egraph.Id { return b.node(OpReduceMax, x) }

// Slice selects columns [start, start+n) of the last axis.
func (b *Builder) Slice(x egraph.Id, start, n int) egraph.Id {
	return b.node(OpSlice, x, b.G.Add(egraph.NewLeaf(start)), b.G.Add(egraph.NewLeaf(n)))
}

// Reshape reinterprets x with the given shape.
func (b *Builder) Reshape(x egraph.Id, shape ...int) egraph.Id {
	d := Dims{Shape: slices.Clone(shape)}
	return b.node(OpReshape, x, b.G.Add(egraph.ENode{Op: fmt.Sprint(shape), Data: d}))
}

// MatMul is x · wᵀ.
func (b *Builder) MatMul(x, w egraph.Id) egraph.Id { return b.node(OpMatMul, x, w) }

// NVFP4 views packed codes, block scales and a global scale as one weight.
func (b *Builder) NVFP4(packed, sf, gs egraph.Id) egraph.Id { return b.node(OpNVFP4, packed, sf, gs) }

// HiLo views a BF16 hi + lo pair as one weight.
func (b *Builder) HiLo(hi, lo egraph.Id) egraph.Id { return b.node(OpHiLo, hi, lo) }

// Cast rounds x to d.
func (b *Builder) Cast(d DType, x egraph.Id) egraph.Id {
	return b.node(OpCast, b.G.Add(egraph.ENode{Op: "dtype:" + d.String(), Data: d}), x)
}

// Col is the column index of each element of x.
func (b *Builder) Col(x egraph.Id) egraph.Id { return b.node(OpCol, x) }

// Sgn is -1 at even columns of x and +1 at odd ones.
func (b *Builder) Sgn(x egraph.Id) egraph.Id { return b.node(OpSgn, x) }

// Pair swaps columns 2i and 2i+1 of x.
func (b *Builder) Pair(x egraph.Id) egraph.Id { return b.node(OpPair, x) }

// Rules returns the rewrites that make matmuls lowerable and fuse cross-op
// attention and MoE subgraphs:
// (matmul x w) = (gemm fn:$0 x w).
func Rules() []*egraph.Rewrite {
	return append([]*egraph.Rewrite{
		{Name: "matmul-gemm",
			LHS: egraph.Pat(OpMatMul, egraph.Var("x"), egraph.Var("w")),
			RHS: egraph.Pat(OpGEMM, egraph.Pat("fn:$0"), egraph.Var("x"), egraph.Var("w"))},
	}, modelRules()...)
}

func lambdaOf(n egraph.ENode) (Lambda, bool) {
	src, ok := strings.CutPrefix(n.Op, "fn:")
	if !ok || len(n.Children) != 0 {
		return Lambda{}, false
	}
	l, err := ParseLambda(src)
	return l, err == nil
}

// FuseEpilogues adds, for every (map f (gemm g x w e...) a...) in g whose
// operands a... are inputs or slices and reshapes of inputs, the equivalent
// (gemm f∘g x w e... a...), and returns how many it added. A gemm whose
// result has other users is not fused, so W is streamed once. Operands computed
// by other terms are left to a row kernel, since reading them from an
// epilogue would cost the kernels that produce them. It rebuilds the graph.
func FuseEpilogues(g *egraph.EGraph) (int, error) {
	// users[c] counts the classes with a node reading class c. Fusing a
	// map into a gemm whose result has other users would stream W twice.
	users := make(map[egraph.Id]map[egraph.Id]bool)
	for _, class := range g.Classes() {
		for _, n := range class.Nodes {
			for _, k := range n.Children {
				k = g.Find(k)
				if users[k] == nil {
					users[k] = make(map[egraph.Id]bool)
				}
				users[k][class.ID] = true
			}
		}
	}
	added := 0
	for _, class := range g.Classes() {
		for _, n := range class.Nodes {
			if n.Op != OpMap || len(n.Children) < 2 {
				continue
			}
			f, ok := lambdaOf(g.Class(n.Children[0]).Nodes[0])
			if !ok {
				continue
			}
			if !allLeafViews(g, n.Children[2:]) || len(users[g.Find(n.Children[1])]) > 1 {
				continue
			}
			for _, inner := range g.Class(n.Children[1]).Nodes {
				if inner.Op != OpGEMM {
					continue
				}
				gl, ok := lambdaOf(g.Class(inner.Children[0]).Nodes[0])
				if !ok {
					continue
				}
				// f's operand 0 is the gemm; its operands 1.. come after the
				// gemm's own epilogue operands.
				ge := len(inner.Children) - 3 // gemm epilogue operands
				shifted := f.shiftArgs(1, ge)
				composed := shifted.Compose(0, gl)
				// Compose put gl's operands at shifted.Arity()...; renumber
				// them to 0..ge so gemm keeps $0 = accumulator.
				fused := composed.moveArgs(shifted.Arity(), 0, gl.Arity()).compact()
				kids := []egraph.Id{g.Add(egraph.ENode{Op: "fn:" + fused.String()})}
				kids = append(kids, inner.Children[1:]...)
				kids = append(kids, n.Children[2:]...)
				id := g.Add(egraph.NewNode(OpGEMM, kids...))
				if _, merged := g.Union(class.ID, id); merged {
					added++
				}
			}
		}
	}
	g.Rebuild()
	mAdded, err := fuseModelPatterns(g)
	if err != nil {
		return added, err
	}
	return added + mAdded, nil
}

// allLeafViews reports whether every class holds a Tensor leaf or a slice
// or reshape of one: operands an epilogue can read without another kernel.
func allLeafViews(g *egraph.EGraph, ids []egraph.Id) bool {
	var leafView func(id egraph.Id, depth int) bool
	leafView = func(id egraph.Id, depth int) bool {
		if depth > 8 {
			return false
		}
		for _, n := range g.Class(id).Nodes {
			if _, ok := n.Data.(Tensor); ok {
				return true
			}
			if (n.Op == OpSlice || n.Op == OpReshape) && leafView(n.Children[0], depth+1) {
				return true
			}
		}
		return false
	}
	for _, id := range ids {
		if !leafView(id, 0) {
			return false
		}
	}
	return true
}

// shiftArgs renumbers operands i >= from to i+by.
func (l Lambda) shiftArgs(from, by int) Lambda {
	return l.remap(func(i int) int {
		if i >= from {
			return i + by
		}
		return i
	})
}

// moveArgs renumbers operands [from, from+n) to [to, to+n), leaving others.
func (l Lambda) moveArgs(from, to, n int) Lambda {
	return l.remap(func(i int) int {
		if i >= from && i < from+n {
			return i - from + to
		}
		return i
	})
}

// compact is l with its arity recomputed from the operands it uses.
func (l Lambda) compact() Lambda {
	out, err := ParseLambda(l.String())
	if err != nil {
		panic("unreachable: reparse " + l.String())
	}
	return out
}

func (l Lambda) remap(f func(int) int) Lambda {
	var walk func(*lambdaNode) *lambdaNode
	walk = func(n *lambdaNode) *lambdaNode {
		c := *n
		if c.op == "$" {
			c.arg = f(c.arg)
		}
		c.kids = nil
		for _, k := range n.kids {
			c.kids = append(c.kids, walk(k))
		}
		return &c
	}
	root := walk(l.root)
	return MustLambda(root.format())
}

// Cost prices terms for extraction: a launch is 2000 and a byte of weight
// streamed is 1/1000 (about 2 µs and 1 µs/MB at H200's 4.8 TB/s); an unfused
// map, reduction or view over a kernel's result costs a launch; unlowerable
// operators cost 1e12.
type Cost struct{}

// NodeCost implements egraph.CostModel.
func (Cost) NodeCost(n egraph.ENode, children []float64) float64 {
	sum := 0.0
	for _, c := range children {
		sum += c
	}
	switch n.Op {
	case OpMatMul:
		return sum + 1e12
	case OpGroupedMatMul:
		return sum + 2500
	case OpGEMM, OpGroupedGEMM, OpGroupedGEMMGLU, OpGroupedGEMMSumSq:
		return sum + 2000
	case OpPagedAttn, OpPagedAttnGated, OpRMSNormGate:
		return sum + 2000
	case OpRowTopK, OpExpertAlign, OpSwiGLU, OpPostExpertNorm, OpWeightedTopKSum:
		return sum + 2000
	case OpExpertSum:
		return sum + 1500
	case OpMap, OpReduceSum, OpReduceMax:
		return sum + 2000
	}
	return sum
}

// Optimize saturates the term graph with [Rules] and [FuseEpilogues] and
// extracts the cheapest form of root under [Cost].
func Optimize(g *egraph.EGraph, root egraph.Id) (*egraph.Expr, error) {
	for range 8 {
		if _, err := egraph.SaturateChecked(g, Rules(), egraph.SaturationOptions{MaxIterations: 16}); err != nil {
			return nil, fmt.Errorf("optimize: %w", err)
		}
		n, err := FuseEpilogues(g)
		if err != nil {
			return nil, fmt.Errorf("optimize: %w", err)
		}
		if n == 0 {
			break
		}
	}
	expr, _, err := egraph.ExtractBest(g, root, Cost{})
	if err != nil {
		return nil, fmt.Errorf("optimize: %w", err)
	}
	return expr, nil
}

// A Value is a dense row-major tensor of values, for [Eval].
type Value struct {
	Shape []int
	Data  []float64
}

func size(shape []int) int {
	n := 1
	for _, d := range shape {
		n *= d
	}
	return n
}

// Eval evaluates a term on the CPU. env holds each Tensor leaf's values (as
// the numbers its storage encodes; NVFP4 packed bytes and scale bytes as
// their integer codes). Arithmetic follows each operator's definition:
// FP32 lambdas, float64 matmul and reduction accumulation.
func Eval(e *egraph.Expr, env map[string]Value) (Value, error) {
	memo := make(map[*egraph.Expr]Value)
	return eval(e, env, memo)
}

func eval(e *egraph.Expr, env map[string]Value, memo map[*egraph.Expr]Value) (Value, error) {
	if v, ok := memo[e]; ok {
		return v, nil
	}
	v, err := eval1(e, env, memo)
	if err != nil {
		return Value{}, err
	}
	memo[e] = v
	return v, nil
}

func eval1(e *egraph.Expr, env map[string]Value, memo map[*egraph.Expr]Value) (Value, error) {
	kid := func(i int) (Value, error) {
		if i >= len(e.Children) {
			return Value{}, fmt.Errorf("eval %s: missing operand %d", e.Op, i)
		}
		return eval(e.Children[i], env, memo)
	}
	intLeaf := func(i int) (int, error) {
		if i >= len(e.Children) {
			return 0, fmt.Errorf("eval %s: missing operand %d", e.Op, i)
		}
		n, ok := e.Children[i].Data.(int)
		if !ok {
			return 0, fmt.Errorf("eval %s: operand %d is not an int", e.Op, i)
		}
		return n, nil
	}
	if t, ok := e.Data.(Tensor); ok {
		v, ok := env[t.Name]
		if !ok || !slices.Equal(v.Shape, t.Shape) || len(v.Data) != size(t.Shape) {
			return Value{}, fmt.Errorf("eval: tensor %s missing or not shaped %v", t.Name, t.Shape)
		}
		return v, nil
	}
	switch e.Op {
	case OpMap, OpGEMM:
		fnExpr := e.Children[0]
		f, ok := lambdaOf(egraph.ENode{Op: fnExpr.Op})
		if !ok {
			return Value{}, fmt.Errorf("eval %s: bad lambda %q", e.Op, fnExpr.Op)
		}
		var args []Value
		start := 1
		if e.Op == OpGEMM {
			x, err := kid(1)
			if err != nil {
				return Value{}, err
			}
			w, err := evalWeight(e.Children[2], env, memo)
			if err != nil {
				return Value{}, err
			}
			mm, err := matmul(x, w)
			if err != nil {
				return Value{}, err
			}
			args = append(args, mm)
			start = 3
		}
		for i := start; i < len(e.Children); i++ {
			v, err := kid(i)
			if err != nil {
				return Value{}, err
			}
			args = append(args, v)
		}
		if f.Arity() > len(args) {
			return Value{}, fmt.Errorf("eval %s: %s needs %d operands", e.Op, f, f.Arity())
		}
		return broadcastMap(f, args)
	case OpReduceSum, OpReduceMax:
		x, err := kid(0)
		if err != nil {
			return Value{}, err
		}
		d := x.Shape[len(x.Shape)-1]
		out := Value{Shape: append(slices.Clone(x.Shape[:len(x.Shape)-1]), 1)}
		for r := 0; r < len(x.Data)/d; r++ {
			row := x.Data[r*d : (r+1)*d]
			acc := 0.0
			if e.Op == OpReduceMax {
				acc = math.Inf(-1)
			}
			for _, f := range row {
				if e.Op == OpReduceMax {
					acc = math.Max(acc, f)
				} else {
					acc += f
				}
			}
			out.Data = append(out.Data, float64(float32(acc)))
		}
		return out, nil
	case OpSlice:
		x, err := kid(0)
		if err != nil {
			return Value{}, err
		}
		start, err := intLeaf(1)
		if err != nil {
			return Value{}, err
		}
		n, err := intLeaf(2)
		if err != nil {
			return Value{}, err
		}
		d := x.Shape[len(x.Shape)-1]
		if start < 0 || n < 1 || start+n > d {
			return Value{}, fmt.Errorf("eval slice: [%d,+%d) outside %d columns", start, n, d)
		}
		out := Value{Shape: append(slices.Clone(x.Shape[:len(x.Shape)-1]), n)}
		for r := 0; r < len(x.Data)/d; r++ {
			out.Data = append(out.Data, x.Data[r*d+start:r*d+start+n]...)
		}
		return out, nil
	case OpReshape:
		x, err := kid(0)
		if err != nil {
			return Value{}, err
		}
		dims, ok := e.Children[1].Data.(Dims)
		if !ok || size(dims.Shape) != len(x.Data) {
			return Value{}, fmt.Errorf("eval reshape: bad shape for %v", x.Shape)
		}
		return Value{Shape: slices.Clone(dims.Shape), Data: x.Data}, nil
	case OpCast:
		d, ok := e.Children[0].Data.(DType)
		if !ok {
			return Value{}, fmt.Errorf("eval cast: missing dtype")
		}
		x, err := kid(1)
		if err != nil {
			return Value{}, err
		}
		out := Value{Shape: x.Shape, Data: make([]float64, len(x.Data))}
		for i, f := range x.Data {
			out.Data[i] = float64(d.Round(float32(f)))
		}
		return out, nil
	case OpCol, OpSgn, OpPair:
		x, err := kid(0)
		if err != nil {
			return Value{}, err
		}
		d := x.Shape[len(x.Shape)-1]
		out := Value{Shape: x.Shape, Data: make([]float64, len(x.Data))}
		for i := range out.Data {
			c := i % d
			switch e.Op {
			case OpCol:
				out.Data[i] = float64(c)
			case OpSgn:
				out.Data[i] = float64(2*(c&1) - 1)
			case OpPair:
				out.Data[i] = x.Data[i-c+(c^1)]
			}
		}
		return out, nil
	case OpMatMul:
		x, err := kid(0)
		if err != nil {
			return Value{}, err
		}
		w, err := evalWeight(e.Children[1], env, memo)
		if err != nil {
			return Value{}, err
		}
		return matmul(x, w)
	}
	return evalModel(e, env, memo)
}

// evalWeight evaluates a weight operand to [N, K] values.
func evalWeight(e *egraph.Expr, env map[string]Value, memo map[*egraph.Expr]Value) (Value, error) {
	switch e.Op {
	case OpHiLo:
		hi, err := eval(e.Children[0], env, memo)
		if err != nil {
			return Value{}, err
		}
		lo, err := eval(e.Children[1], env, memo)
		if err != nil {
			return Value{}, err
		}
		out := Value{Shape: hi.Shape, Data: make([]float64, len(hi.Data))}
		for i := range out.Data {
			out.Data[i] = hi.Data[i] + lo.Data[i]
		}
		return out, nil
	case OpNVFP4:
		var vs [3]Value
		for i := range vs {
			v, err := eval(e.Children[i], env, memo)
			if err != nil {
				return Value{}, err
			}
			vs[i] = v
		}
		packed, sf := vs[0], vs[1]
		n, k := packed.Shape[0], 2*packed.Shape[1]
		out := Value{Shape: []int{n, k}, Data: make([]float64, n*k)}
		row := make([]float32, k)
		pb, sb := make([]uint8, k/2), make([]uint8, k/16)
		for r := 0; r < n; r++ {
			for i := range pb {
				pb[i] = uint8(packed.Data[r*k/2+i])
			}
			for i := range sb {
				sb[i] = uint8(sf.Data[r*k/16+i])
			}
			if err := DequantNVFP4(row, pb, sb, float32(vs[2].Data[0])); err != nil {
				return Value{}, err
			}
			for i, f := range row {
				out.Data[r*k+i] = float64(f)
			}
		}
		return out, nil
	}
	return eval(e, env, memo)
}

func matmul(x, w Value) (Value, error) {
	if len(x.Shape) != 2 || len(w.Shape) != 2 || x.Shape[1] != w.Shape[1] {
		return Value{}, fmt.Errorf("eval matmul: shapes %v and %v", x.Shape, w.Shape)
	}
	m, n, k := x.Shape[0], w.Shape[0], x.Shape[1]
	out := Value{Shape: []int{m, n}, Data: make([]float64, m*n)}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var s float64
			for c := 0; c < k; c++ {
				s += x.Data[i*k+c] * w.Data[j*k+c]
			}
			out.Data[i*n+j] = float64(float32(s))
		}
	}
	return out, nil
}

// broadcastShape is the NumPy broadcast of shapes.
func broadcastShape(shapes ...[]int) ([]int, error) {
	rank := 0
	for _, s := range shapes {
		rank = max(rank, len(s))
	}
	out := make([]int, rank)
	for i := range out {
		out[i] = 1
	}
	for _, s := range shapes {
		for i, d := range s {
			j := rank - len(s) + i
			switch {
			case out[j] == 1:
				out[j] = d
			case d != 1 && d != out[j]:
				return nil, fmt.Errorf("cannot broadcast %v", shapes)
			}
		}
	}
	return out, nil
}

func broadcastMap(f Lambda, args []Value) (Value, error) {
	shapes := make([][]int, len(args))
	for i, a := range args {
		shapes[i] = a.Shape
	}
	shape, err := broadcastShape(shapes...)
	if err != nil {
		return Value{}, err
	}
	out := Value{Shape: shape, Data: make([]float64, size(shape))}
	idx := make([]int, len(shape))
	ops := make([]float32, len(args))
	for i := range out.Data {
		rem := i
		for d := len(shape) - 1; d >= 0; d-- {
			idx[d] = rem % shape[d]
			rem /= shape[d]
		}
		for a, v := range args {
			off, stride := 0, 1
			for d := len(v.Shape) - 1; d >= 0; d-- {
				j := idx[len(shape)-len(v.Shape)+d]
				if v.Shape[d] == 1 {
					j = 0
				}
				off += j * stride
				stride *= v.Shape[d]
			}
			ops[a] = float32(v.Data[off])
		}
		out.Data[i] = float64(f.Eval(ops...))
	}
	return out, nil
}
