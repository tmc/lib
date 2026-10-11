package metalshape

import (
	"fmt"
	"math"
	"sort"
	"strconv"
)

// Expr is an immutable expression for a non-negative dimension.
// Its zero value denotes the constant zero.
type Expr struct {
	node *dimensionNode
}

// Interval is a closed, non-negative interval for a dimension.
type Interval struct {
	Min int64
	Max int64
}

// Equal reports whether two expressions have the same canonical structure.
func (e Expr) Equal(other Expr) bool {
	return dimensionKey(dimensionNodeOf(e)) == dimensionKey(dimensionNodeOf(other))
}

type dimensionOp uint8

const (
	dimConstant dimensionOp = iota
	dimSymbol
	dimAdd
	dimMul
	dimCeilDiv
	dimFloorDiv
	dimMin
	dimMax
	dimMod
)

type dimensionNode struct {
	op    dimensionOp
	value int64
	name  string
	a, b  *dimensionNode
}

// DimensionConstant constructs a constant dimension.
func DimensionConstant(value int64) (Expr, error) {
	if value < 0 {
		return Expr{}, fmt.Errorf("metalshape: dimension constant must be non-negative")
	}
	return Expr{node: &dimensionNode{op: dimConstant, value: value}}, nil
}

// DimensionSymbol constructs a named symbolic dimension.
func DimensionSymbol(name string) (Expr, error) {
	if name == "" {
		return Expr{}, fmt.Errorf("metalshape: dimension symbol must be non-empty")
	}
	return Expr{node: &dimensionNode{op: dimSymbol, name: name}}, nil
}

// DimensionAdd constructs a checked sum and canonicalizes associative terms.
// Overflow is reported by Eval or Bounds.
func DimensionAdd(a, b Expr) Expr {
	return canonicalDimensionAdd(a, b)
}

// DimensionMul constructs a checked product. Overflow is reported by Eval or Bounds.
func DimensionMul(a, b Expr) Expr {
	return dimensionBinary(dimMul, a, b)
}

// DimensionCeilDiv constructs a ceiling division. Bounds and Eval require a
// strictly positive denominator for every value represented by the expression.
func DimensionCeilDiv(a, b Expr) (Expr, error) {
	if v, ok := dimensionConstantValue(b); ok && v == 0 {
		return Expr{}, fmt.Errorf("metalshape: dimension divisor must be positive")
	}
	if v, ok := dimensionConstantValue(b); ok && v == 1 {
		return a, nil
	}
	if dimensionEqual(a, b) {
		if v, ok := dimensionConstantValue(a); ok && v > 0 {
			return Expr{node: &dimensionNode{op: dimConstant, value: 1}}, nil
		}
	}
	if av, aok := dimensionConstantValue(a); aok {
		if bv, bok := dimensionConstantValue(b); bok {
			q, r := av/bv, av%bv
			if r != 0 {
				q++
			}
			return Expr{node: &dimensionNode{op: dimConstant, value: q}}, nil
		}
	}
	return Expr{node: &dimensionNode{op: dimCeilDiv, a: dimensionNodeOf(a), b: dimensionNodeOf(b)}}, nil
}

// DimensionFloorDiv constructs a floor division. Bounds and Eval require a
// strictly positive denominator for every value represented by the expression.
func DimensionFloorDiv(a, b Expr) (Expr, error) {
	if v, ok := dimensionConstantValue(b); ok && v == 0 {
		return Expr{}, fmt.Errorf("metalshape: dimension divisor must be positive")
	}
	if v, ok := dimensionConstantValue(b); ok && v == 1 {
		return a, nil
	}
	if dimensionEqual(a, b) {
		if v, ok := dimensionConstantValue(a); ok {
			if v == 0 {
				return Expr{}, fmt.Errorf("metalshape: dimension divisor must be positive")
			}
			return Expr{node: &dimensionNode{op: dimConstant, value: 1}}, nil
		}
	}
	if av, aok := dimensionConstantValue(a); aok {
		if bv, bok := dimensionConstantValue(b); bok {
			return Expr{node: &dimensionNode{op: dimConstant, value: av / bv}}, nil
		}
	}
	return Expr{node: &dimensionNode{op: dimFloorDiv, a: dimensionNodeOf(a), b: dimensionNodeOf(b)}}, nil
}

// DimensionMod constructs a non-negative remainder. Bounds and Eval require a
// strictly positive denominator for every value represented by the expression.
func DimensionMod(a, b Expr) (Expr, error) {
	if v, ok := dimensionConstantValue(b); ok && v == 0 {
		return Expr{}, fmt.Errorf("metalshape: dimension divisor must be positive")
	}
	if v, ok := dimensionConstantValue(b); ok && v == 1 {
		return DimensionConstant(0)
	}
	if av, aok := dimensionConstantValue(a); aok {
		if bv, bok := dimensionConstantValue(b); bok {
			return Expr{node: &dimensionNode{op: dimConstant, value: av % bv}}, nil
		}
	}
	return Expr{node: &dimensionNode{op: dimMod, a: dimensionNodeOf(a), b: dimensionNodeOf(b)}}, nil
}

// DimensionMin constructs the minimum of two dimensions.
func DimensionMin(a, b Expr) Expr { return dimensionBinary(dimMin, a, b) }

// DimensionMax constructs the maximum of two dimensions.
func DimensionMax(a, b Expr) Expr { return dimensionBinary(dimMax, a, b) }

// Eval evaluates the expression using exact non-negative symbol values.
func (e Expr) Eval(values map[string]int64) (int64, error) {
	return evalDimension(dimensionNodeOf(e), values)
}

// Bounds returns sound inclusive bounds using the supplied symbol intervals.
// It errors when a symbol is missing, arithmetic may overflow, or a divisor
// interval includes zero.
func (e Expr) Bounds(intervals map[string]Interval) (Interval, error) {
	return boundsDimension(dimensionNodeOf(e), intervals)
}

// SimplifyWithIntervals simplifies e using sound interval assumptions. Every
// referenced symbol must have a valid interval. An expression whose bounds are
// a single value is replaced by that constant.
func (e Expr) SimplifyWithIntervals(intervals map[string]Interval) (Expr, error) {
	if _, err := e.Bounds(intervals); err != nil {
		return Expr{}, err
	}
	out, err := simplifyDimension(dimensionNodeOf(e), intervals)
	if err != nil {
		return Expr{}, err
	}
	if _, err := out.Bounds(intervals); err != nil {
		return Expr{}, err
	}
	return out, nil
}

func simplifyDimension(n *dimensionNode, intervals map[string]Interval) (Expr, error) {
	if n.op == dimConstant {
		return Expr{node: n}, nil
	}
	if n.op == dimSymbol {
		interval := intervals[n.name]
		if interval.Min == interval.Max {
			return DimensionConstant(interval.Min)
		}
		return Expr{node: n}, nil
	}
	a, err := simplifyDimension(n.a, intervals)
	if err != nil {
		return Expr{}, err
	}
	b, err := simplifyDimension(n.b, intervals)
	if err != nil {
		return Expr{}, err
	}
	var out Expr
	switch n.op {
	case dimAdd:
		if term, ok := dimensionMergeDivMod(a.node, b.node); ok {
			out = Expr{node: term}
		} else if folded, ok := dimensionAddFloorDiv(a, b, intervals); ok {
			out = folded
		} else {
			out = DimensionAdd(a, b)
		}
	case dimMul:
		out = DimensionMul(a, b)
	case dimCeilDiv:
		if dimensionEqual(a, b) {
			bound, boundsErr := a.Bounds(intervals)
			if boundsErr != nil {
				return Expr{}, boundsErr
			}
			if bound.Min > 0 {
				out, err = DimensionConstant(1)
			} else {
				out, err = DimensionCeilDiv(a, b)
			}
		} else {
			out, err = DimensionCeilDiv(a, b)
		}
	case dimFloorDiv:
		if divisor, ok := dimensionConstantValue(b); ok && divisor > 0 {
			if numerator, ok := dimensionCancelProductPlusRemainder(a.node, divisor); ok {
				out = Expr{node: numerator}
				break
			}
		}
		// For non-negative dimensions, nested floor divisions by positive
		// constants combine when their product remains representable.
		if a.node.op == dimFloorDiv {
			innerDivisor, innerOK := dimensionConstantValue(Expr{node: a.node.b})
			outerDivisor, outerOK := dimensionConstantValue(b)
			if innerOK && outerOK && innerDivisor > 0 && outerDivisor > 0 && innerDivisor <= math.MaxInt64/outerDivisor {
				combined, constErr := DimensionConstant(innerDivisor * outerDivisor)
				if constErr != nil {
					return Expr{}, constErr
				}
				out, err = DimensionFloorDiv(Expr{node: a.node.a}, combined)
				break
			}
		}
		// A non-negative value below a positive divisor floors to zero.
		if divisor, ok := dimensionConstantValue(b); ok && divisor > 0 {
			bound, boundsErr := a.Bounds(intervals)
			if boundsErr != nil {
				return Expr{}, boundsErr
			}
			if bound.Min >= 0 && bound.Max < divisor {
				out, err = DimensionConstant(0)
				break
			}
		}
		if dimensionEqual(a, b) {
			bound, boundsErr := a.Bounds(intervals)
			if boundsErr != nil {
				return Expr{}, boundsErr
			}
			if bound.Min > 0 {
				out, err = DimensionConstant(1)
			} else {
				out, err = DimensionFloorDiv(a, b)
			}
		} else {
			out, err = DimensionFloorDiv(a, b)
		}
	case dimMod:
		if divisor, ok := dimensionConstantValue(b); ok && divisor > 0 {
			if _, remainder, ok := dimensionProductPlusRemainder(a.node, divisor); ok {
				out, err = DimensionConstant(remainder)
				break
			}
			if a.node.op == dimMod {
				innerDivisor, innerOK := dimensionConstantValue(Expr{node: a.node.b})
				if innerOK && innerDivisor > 0 {
					if innerDivisor <= divisor {
						out = a
						break
					}
					if innerDivisor%divisor == 0 {
						out, err = DimensionMod(Expr{node: a.node.a}, b)
						break
					}
				}
			}
		}
		if a.node.op == dimMul && (dimensionEqual(Expr{node: a.node.a}, b) || dimensionEqual(Expr{node: a.node.b}, b)) {
			out, err = DimensionConstant(0)
			break
		}
		// A non-negative value below a positive divisor is unchanged by mod.
		if divisor, ok := dimensionConstantValue(b); ok && divisor > 0 {
			bound, boundsErr := a.Bounds(intervals)
			if boundsErr != nil {
				return Expr{}, boundsErr
			}
			if bound.Min >= 0 && bound.Max < divisor {
				out = a
				break
			}
		}
		if dimensionEqual(a, b) {
			bound, boundsErr := a.Bounds(intervals)
			if boundsErr != nil {
				return Expr{}, boundsErr
			}
			if bound.Min > 0 {
				out, err = DimensionConstant(0)
			} else {
				out, err = DimensionMod(a, b)
			}
		} else {
			out, err = DimensionMod(a, b)
		}
	case dimMin, dimMax:
		aa, boundsErr := a.Bounds(intervals)
		if boundsErr != nil {
			return Expr{}, boundsErr
		}
		bb, boundsErr := b.Bounds(intervals)
		if boundsErr != nil {
			return Expr{}, boundsErr
		}
		if n.op == dimMin {
			if aa.Max <= bb.Min {
				out = a
			} else if bb.Max <= aa.Min {
				out = b
			} else {
				out = DimensionMin(a, b)
			}
		} else if aa.Min >= bb.Max {
			out = a
		} else if bb.Min >= aa.Max {
			out = b
		} else {
			out = DimensionMax(a, b)
		}
	default:
		return Expr{}, fmt.Errorf("metalshape: invalid dimension expression")
	}
	if err != nil {
		return Expr{}, err
	}
	bound, err := out.Bounds(intervals)
	if err != nil {
		return Expr{}, err
	}
	if bound.Min == bound.Max {
		return DimensionConstant(bound.Min)
	}
	return out, nil
}

func dimensionNodeOf(e Expr) *dimensionNode {
	if e.node == nil {
		return &dimensionNode{op: dimConstant}
	}
	return e.node
}

func dimensionConstantValue(e Expr) (int64, bool) {
	n := dimensionNodeOf(e)
	return n.value, n.op == dimConstant
}

func dimensionEqual(a, b Expr) bool {
	return dimensionKey(dimensionNodeOf(a)) == dimensionKey(dimensionNodeOf(b))
}

func dimensionBinary(op dimensionOp, a, b Expr) Expr {
	na, nb := dimensionNodeOf(a), dimensionNodeOf(b)
	if op == dimAdd || op == dimMul || op == dimMin || op == dimMax {
		if dimensionKey(na) > dimensionKey(nb) {
			na, nb = nb, na
		}
	}
	if va, ok := dimensionConstantNode(na); ok {
		if vb, ok := dimensionConstantNode(nb); ok {
			switch op {
			case dimAdd:
				if va <= math.MaxInt64-vb {
					return Expr{node: &dimensionNode{op: dimConstant, value: va + vb}}
				}
			case dimMul:
				if va == 0 || vb <= math.MaxInt64/va {
					return Expr{node: &dimensionNode{op: dimConstant, value: va * vb}}
				}
			case dimMin:
				if va < vb {
					return Expr{node: &dimensionNode{op: dimConstant, value: va}}
				}
				return Expr{node: &dimensionNode{op: dimConstant, value: vb}}
			case dimMax:
				if va > vb {
					return Expr{node: &dimensionNode{op: dimConstant, value: va}}
				}
				return Expr{node: &dimensionNode{op: dimConstant, value: vb}}
			}
		}
	}
	if dimensionKey(na) == dimensionKey(nb) {
		if op == dimMin || op == dimMax {
			return Expr{node: na}
		}
		if op == dimAdd {
			return DimensionMul(Expr{node: na}, Expr{node: &dimensionNode{op: dimConstant, value: 2}})
		}
	}
	if op == dimAdd {
		if v, ok := dimensionConstantNode(na); ok && v == 0 {
			return Expr{node: nb}
		}
	}
	if op == dimMul {
		if v, ok := dimensionConstantNode(na); ok && v == 0 {
			return Expr{node: na}
		}
		if v, ok := dimensionConstantNode(nb); ok && v == 0 {
			return Expr{node: nb}
		}
		if v, ok := dimensionConstantNode(na); ok && v == 1 {
			return Expr{node: nb}
		}
		if v, ok := dimensionConstantNode(nb); ok && v == 1 {
			return Expr{node: na}
		}
	}
	return Expr{node: &dimensionNode{op: op, a: na, b: nb}}
}

func dimensionConstantNode(n *dimensionNode) (int64, bool) { return n.value, n.op == dimConstant }

func dimensionCancelProductPlusRemainder(n *dimensionNode, divisor int64) (*dimensionNode, bool) {
	factor, _, ok := dimensionProductPlusRemainder(n, divisor)
	return factor, ok
}

func dimensionProductPlusRemainder(n *dimensionNode, divisor int64) (*dimensionNode, int64, bool) {
	if factor, ok := dimensionProductFactor(n, divisor); ok {
		return factor, 0, true
	}
	if n.op != dimAdd {
		return nil, 0, false
	}
	for _, pair := range [][2]*dimensionNode{{n.a, n.b}, {n.b, n.a}} {
		remainder, constant := dimensionConstantNode(pair[1])
		if !constant || remainder < 0 || remainder >= divisor {
			continue
		}
		if factor, ok := dimensionProductFactor(pair[0], divisor); ok {
			return factor, remainder, true
		}
	}
	return nil, 0, false
}

func dimensionProductFactor(n *dimensionNode, factor int64) (*dimensionNode, bool) {
	if n.op != dimMul {
		return nil, false
	}
	if value, ok := dimensionConstantNode(n.a); ok && value == factor {
		return n.b, true
	}
	if value, ok := dimensionConstantNode(n.b); ok && value == factor {
		return n.a, true
	}
	return nil, false
}

func dimensionMergeDivMod(a, b *dimensionNode) (*dimensionNode, bool) {
	for _, pair := range [][2]*dimensionNode{{a, b}, {b, a}} {
		product, remainder := pair[0], pair[1]
		if product.op != dimMul || remainder.op != dimMod {
			continue
		}
		for _, factors := range [][2]*dimensionNode{{product.a, product.b}, {product.b, product.a}} {
			quotient, divisor := factors[0], factors[1]
			if quotient.op == dimFloorDiv &&
				dimensionEqual(Expr{node: quotient.a}, Expr{node: remainder.a}) &&
				dimensionEqual(Expr{node: quotient.b}, Expr{node: remainder.b}) &&
				dimensionEqual(Expr{node: divisor}, Expr{node: remainder.b}) {
				return remainder.a, true
			}
		}
	}
	return nil, false
}

func dimensionAddFloorDiv(a, b Expr, intervals map[string]Interval) (Expr, bool) {
	for _, pair := range [][2]Expr{{a, b}, {b, a}} {
		quotient, addend := dimensionNodeOf(pair[0]), dimensionNodeOf(pair[1])
		if quotient.op != dimFloorDiv {
			continue
		}
		divisor := Expr{node: quotient.b}
		bound, err := divisor.Bounds(intervals)
		if err != nil || bound.Min <= 0 {
			continue
		}
		numerator := canonicalDimensionAdd(
			Expr{node: quotient.a},
			DimensionMul(Expr{node: addend}, divisor),
		)
		if _, err := numerator.Bounds(intervals); err != nil {
			continue
		}
		folded, err := DimensionFloorDiv(numerator, divisor)
		if err != nil {
			continue
		}
		return folded, true
	}
	return Expr{}, false
}

func canonicalDimensionAdd(a, b Expr) Expr {
	type term struct {
		node        *dimensionNode
		coefficient int64
	}
	byNode := make(map[string]term)
	var constants int64
	coefficientOverflow := false
	var collect func(*dimensionNode)
	collect = func(n *dimensionNode) {
		if n.op == dimAdd {
			collect(n.a)
			collect(n.b)
			return
		}
		if value, ok := dimensionConstantNode(n); ok {
			if value > math.MaxInt64-constants {
				coefficientOverflow = true
				return
			}
			constants += value
			return
		}
		atom, coefficient := dimensionTermCoefficient(n)
		key := dimensionKey(atom)
		current := byNode[key]
		if coefficient > math.MaxInt64-current.coefficient {
			coefficientOverflow = true
			return
		}
		byNode[key] = term{node: atom, coefficient: current.coefficient + coefficient}
	}
	collect(dimensionNodeOf(a))
	collect(dimensionNodeOf(b))
	if coefficientOverflow {
		return dimensionBinary(dimAdd, a, b)
	}
	terms := make([]*dimensionNode, 0, len(byNode)+1)
	for _, grouped := range byNode {
		if grouped.coefficient == 1 {
			terms = append(terms, grouped.node)
			continue
		}
		coefficient := &dimensionNode{op: dimConstant, value: grouped.coefficient}
		terms = append(terms, dimensionNodeOf(DimensionMul(Expr{node: grouped.node}, Expr{node: coefficient})))
	}
	if constants > 0 || len(terms) == 0 {
		terms = append(terms, &dimensionNode{op: dimConstant, value: constants})
	}
	sort.Slice(terms, func(i, j int) bool {
		return dimensionKey(terms[i]) < dimensionKey(terms[j])
	})
	out := Expr{node: terms[0]}
	for _, term := range terms[1:] {
		out = dimensionBinary(dimAdd, out, Expr{node: term})
	}
	return out
}

func dimensionTermCoefficient(n *dimensionNode) (*dimensionNode, int64) {
	if n.op != dimMul {
		return n, 1
	}
	if coefficient, ok := dimensionConstantNode(n.a); ok {
		return n.b, coefficient
	}
	if coefficient, ok := dimensionConstantNode(n.b); ok {
		return n.a, coefficient
	}
	return n, 1
}

func evalDimension(n *dimensionNode, values map[string]int64) (int64, error) {
	switch n.op {
	case dimConstant:
		return n.value, nil
	case dimSymbol:
		v, ok := values[n.name]
		if !ok {
			return 0, fmt.Errorf("metalshape: unknown dimension symbol %q", n.name)
		}
		if v < 0 {
			return 0, fmt.Errorf("metalshape: dimension symbol %q is negative", n.name)
		}
		return v, nil
	}
	a, err := evalDimension(n.a, values)
	if err != nil {
		return 0, err
	}
	b, err := evalDimension(n.b, values)
	if err != nil {
		return 0, err
	}
	switch n.op {
	case dimAdd:
		if a > math.MaxInt64-b {
			return 0, fmt.Errorf("metalshape: dimension addition overflows")
		}
		return a + b, nil
	case dimMul:
		if a != 0 && b > math.MaxInt64/a {
			return 0, fmt.Errorf("metalshape: dimension multiplication overflows")
		}
		return a * b, nil
	case dimCeilDiv, dimFloorDiv, dimMod:
		if b == 0 {
			return 0, fmt.Errorf("metalshape: dimension divisor must be positive")
		}
		q, r := a/b, a%b
		if n.op == dimMod {
			return r, nil
		}
		if n.op == dimCeilDiv && r != 0 {
			q++
		}
		return q, nil
	case dimMin:
		if a < b {
			return a, nil
		}
		return b, nil
	case dimMax:
		if a > b {
			return a, nil
		}
		return b, nil
	}
	return 0, fmt.Errorf("metalshape: invalid dimension expression")
}

func boundsDimension(n *dimensionNode, intervals map[string]Interval) (Interval, error) {
	if n.op == dimConstant {
		return Interval{Min: n.value, Max: n.value}, nil
	}
	if n.op == dimSymbol {
		v, ok := intervals[n.name]
		if !ok {
			return Interval{}, fmt.Errorf("metalshape: unknown dimension symbol %q", n.name)
		}
		if v.Min < 0 || v.Min > v.Max {
			return Interval{}, fmt.Errorf("metalshape: invalid interval for dimension symbol %q", n.name)
		}
		return v, nil
	}
	a, err := boundsDimension(n.a, intervals)
	if err != nil {
		return Interval{}, err
	}
	b, err := boundsDimension(n.b, intervals)
	if err != nil {
		return Interval{}, err
	}
	switch n.op {
	case dimAdd:
		if a.Max > math.MaxInt64-b.Max {
			return Interval{}, fmt.Errorf("metalshape: dimension addition bounds overflow")
		}
		return Interval{Min: a.Min + b.Min, Max: a.Max + b.Max}, nil
	case dimMul:
		if a.Max != 0 && b.Max > math.MaxInt64/a.Max {
			return Interval{}, fmt.Errorf("metalshape: dimension multiplication bounds overflow")
		}
		return Interval{Min: a.Min * b.Min, Max: a.Max * b.Max}, nil
	case dimCeilDiv, dimFloorDiv, dimMod:
		if b.Min <= 0 {
			return Interval{}, fmt.Errorf("metalshape: dimension divisor interval must be positive")
		}
		if n.op == dimMod {
			if b.Min > a.Max {
				return a, nil
			}
			return Interval{Min: 0, Max: minInt64(a.Max, b.Max-1)}, nil
		}
		lo := a.Min / b.Max
		hi := a.Max / b.Min
		if n.op == dimCeilDiv {
			if a.Min%b.Max != 0 {
				lo++
			}
			if a.Max%b.Min != 0 {
				hi++
			}
		}
		return Interval{Min: lo, Max: hi}, nil
	case dimMin:
		return Interval{Min: minInt64(a.Min, b.Min), Max: minInt64(a.Max, b.Max)}, nil
	case dimMax:
		return Interval{Min: maxInt64(a.Min, b.Min), Max: maxInt64(a.Max, b.Max)}, nil
	}
	return Interval{}, fmt.Errorf("metalshape: invalid dimension expression")
}

func dimensionKey(n *dimensionNode) string {
	switch n.op {
	case dimConstant:
		return fmt.Sprintf("c%020d", n.value)
	case dimSymbol:
		return "s" + strconv.Quote(n.name)
	}
	return fmt.Sprintf("%d(%s,%s)", n.op, dimensionKey(n.a), dimensionKey(n.b))
}

// DimensionSymbols returns the distinct symbols used by e in sorted order.
func (e Expr) DimensionSymbols() []string {
	set := make(map[string]struct{})
	var walk func(*dimensionNode)
	walk = func(n *dimensionNode) {
		if n.op == dimSymbol {
			set[n.name] = struct{}{}
			return
		}
		if n.op != dimConstant {
			walk(n.a)
			walk(n.b)
		}
	}
	walk(dimensionNodeOf(e))
	out := make([]string, 0, len(set))
	for name := range set {
		out = append(out, name)
	}
	sort.Strings(out)
	return out
}

func minInt64(a, b int64) int64 {
	if a < b {
		return a
	}
	return b
}

func maxInt64(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}
