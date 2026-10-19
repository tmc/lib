package egraph

import (
	"fmt"
	"math"
	"reflect"
	"sort"
	"strconv"

	"github.com/tmc/lib/egraphcore"
)

type graphDiagnostic struct {
	message string
	cause   error
}

// Capture only package-owned sentinel identities and their literal messages.
// An opaque callback error is propagated unchanged without invoking Error.
var graphDiagnosticMessages = [...]struct {
	cause   error
	message string
}{
	{ErrMemoryOperation, "egraph: operation lacks memory accounting"},
	{ErrPendingGraphChanges, "egraph: graph has pending changes"},
	{ErrMatchWorkLimit, "egraph: pattern match work limit exceeded"},
	{ErrMissingTypedMetadata, "egraph: mandatory rewrite requires typed metadata"},
	{ErrIncompatibleDType, "egraph: incompatible data types in rewrite"},
	{ErrInvalidRewriteProof, "egraph: rewrite proof precondition not met"},
	{egraphcore.ErrInvalidPattern, "egraphcore: invalid pattern matcher input"},
	{egraphcore.ErrMatchWorkLimit, "egraphcore: pattern match work limit exceeded"},
	{egraphcore.ErrInvalidJoin, "egraphcore: invalid relation join"},
	{egraphcore.ErrMemorySize, "egraphcore: invalid memory size"},
}

func graphDiagnosticMessage(err error) (string, bool) {
	if diagnostic, ok := err.(*graphDiagnostic); ok {
		return diagnostic.message, true
	}
	if err != nil && reflect.TypeOf(err).Comparable() {
		for _, known := range graphDiagnosticMessages {
			if known.cause == err {
				return known.message, true
			}
		}
	}
	return "", false
}

func (e *graphDiagnostic) Error() string { return e.message }
func (e *graphDiagnostic) Unwrap() error { return e.cause }

// graphErrorf handles only the primitive formats used by checked graph code.
// Refused reservations propagate directly, without formatting their errors.
func graphErrorf(budget *egraphcore.WorkBudget, format string, args ...any) error {
	if budget.MemoryScope() == nil {
		return fmt.Errorf(format, args...)
	}
	if err := budget.MemoryError(); err != nil {
		return err
	}
	b := memoryText{budget: budget}
	arg := 0
	var cause error
	for i := 0; i < len(format); i++ {
		if format[i] != '%' {
			b.write(format[i : i+1])
			continue
		}
		i++
		if i == len(format) {
			return ErrMemoryOperation
		}
		if format[i] == '%' {
			b.write("%")
			continue
		}
		if arg == len(args) {
			return ErrMemoryOperation
		}
		value := args[arg]
		arg++
		switch format[i] {
		case 's', 'q':
			s, ok := value.(string)
			if !ok {
				return ErrMemoryOperation
			}
			if format[i] == 'q' {
				b.quote(s)
			} else {
				b.write(s)
			}
		case 'd':
			v := reflect.ValueOf(value)
			switch v.Kind() {
			case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
				b.integer(v.Int())
			case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
				b.uint(v.Uint(), 10)
			default:
				return ErrMemoryOperation
			}
		case 'w':
			err, ok := value.(error)
			if !ok {
				return ErrMemoryOperation
			}
			message, known := graphDiagnosticMessage(err)
			if !known {
				return err
			}
			cause = err
			b.write(message)
		default:
			return ErrMemoryOperation
		}
		if b.err != nil {
			return b.err
		}
	}
	message, err := b.finish()
	if err != nil {
		return err
	}
	if err := graphSlots[graphDiagnostic](budget, 1); err != nil {
		return err
	}
	return &graphDiagnostic{message: message, cause: cause}
}

// transitionGraph carries the borrowed owner into common graph algorithms.
type transitionGraph struct {
	*EGraph
	budget *egraphcore.WorkBudget
}

func (g transitionGraph) AddChecked(n ENode) (Id, error) {
	if g.budget.MemoryScope() == nil {
		return g.EGraph.AddChecked(n)
	}
	return g.addFromWithBudget(n, "input", g.budget)
}
func (g transitionGraph) UnionChecked(a, b Id) (Id, bool, error) {
	return g.unionCheckedWithBudget(a, b, g.budget)
}
func (g transitionGraph) RebuildChecked() error { _, err := g.rebuildWithBudget(g.budget); return err }

func (g *EGraph) canonicalizeWithBudget(n ENode, budget *egraphcore.WorkBudget) (ENode, error) {
	if budget.MemoryScope() == nil {
		return g.Canonicalize(n), nil
	}
	children, err := graphSlice[Id](budget, len(n.Children), len(n.Children))
	if err != nil {
		return ENode{}, err
	}
	for i, id := range n.Children {
		children[i] = g.Find(id)
	}
	n.Children = children
	return n, nil
}

// memoryText admits backing storage and the detached final string separately.
type memoryText struct {
	data   []byte
	budget *egraphcore.WorkBudget
	err    error
}

func (b *memoryText) grow(n int) {
	if b.err != nil {
		return
	}
	if n < 0 || n > math.MaxInt-len(b.data) {
		b.err = egraphcore.ErrMemorySize
		return
	}
	needed := len(b.data) + n
	if needed <= cap(b.data) {
		return
	}
	capacity := needed
	if cap(b.data) <= math.MaxInt/2 {
		capacity = max(capacity, 2*cap(b.data))
	}
	var data []byte
	data, b.err = graphSlice[byte](b.budget, len(b.data), capacity)
	if b.err == nil {
		copy(data, b.data)
		b.data = data
	}
}
func (b *memoryText) write(s string) {
	b.grow(len(s))
	if b.err == nil {
		b.data = append(b.data, s...)
	}
}
func (b *memoryText) uint(n uint64, base int) {
	b.grow(20)
	if b.err == nil {
		b.data = strconv.AppendUint(b.data, n, base)
	}
}
func (b *memoryText) integer(n int64) {
	b.grow(20)
	if b.err == nil {
		b.data = strconv.AppendInt(b.data, n, 10)
	}
}
func (b *memoryText) quote(s string) {
	if len(s) > (math.MaxInt-2)/6 {
		b.err = egraphcore.ErrMemorySize
		return
	}
	b.grow(6*len(s) + 2)
	if b.err == nil {
		b.data = strconv.AppendQuote(b.data, s)
	}
}
func (b *memoryText) finish() (string, error) {
	if b.err != nil {
		return "", b.err
	}
	if err := graphSlots[byte](b.budget, len(b.data)); err != nil {
		return "", err
	}
	return string(b.data), nil
}
func (b *memoryText) pattern(root *Pattern) {
	if root == nil {
		b.write("<nil>")
		return
	}
	type frame struct {
		p     *Pattern
		child int
	}
	stack, err := graphSlice[frame](b.budget, 1, 1)
	if err != nil {
		b.err = err
		return
	}
	stack[0].p = root
	for len(stack) > 0 && b.err == nil {
		i := len(stack) - 1
		f := &stack[i]
		if f.child == 0 {
			if !f.p.IsVar && len(f.p.Children) > 0 {
				b.write("(")
			}
			b.write(f.p.Op)
		}
		if f.p.IsVar || f.child == len(f.p.Children) {
			if !f.p.IsVar && len(f.p.Children) > 0 {
				b.write(")")
			}
			stack = stack[:i]
			continue
		}
		child := f.p.Children[f.child]
		f.child++
		b.write(" ")
		stack, b.err = graphAppend(b.budget, stack, frame{p: child})
	}
}
func (b *memoryText) value(v egraphcore.Value[Id], goSyntax bool) {
	if goSyntax {
		b.write(reflect.TypeFor[egraphcore.Value[Id]]().String())
		b.write("{Kind:0x")
		b.uint(uint64(v.Kind), 16)
		b.write(", Class:0x")
		b.uint(uint64(v.Class), 16)
		b.write(", Int64:")
		b.integer(v.Int64)
		b.write(", Bool:")
		b.write(strconv.FormatBool(v.Bool))
		b.write(", Float:0x")
		b.uint(v.Float, 16)
		b.write(", String:")
		b.quote(v.String)
		b.write("}")
		return
	}
	b.uint(uint64(v.Kind), 10)
	b.write(":")
	b.uint(uint64(v.Class), 10)
	b.write(":")
	b.integer(v.Int64)
	b.write(":")
	b.write(strconv.FormatBool(v.Bool))
	b.write(":")
	b.uint(v.Float, 10)
	b.write(":")
	b.quote(v.String)
}
func ruleIdentityWithBudget(r *Rewrite, budget *egraphcore.WorkBudget) (string, error) {
	if r == nil {
		return "<nil>", nil
	}
	b := memoryText{budget: budget}
	b.write(r.Name)
	b.write("|")
	b.pattern(r.LHS)
	b.write("=>")
	b.pattern(r.RHS)
	if r.subsumesLHS {
		b.write(" subsume-lhs")
	}
	for i, p := range r.Conditions {
		if i == 0 {
			b.write(" where ")
		} else {
			b.write(" & ")
		}
		b.pattern(p)
	}
	for i, c := range r.RelationConditions {
		if i == 0 {
			b.write(" facts ")
		} else {
			b.write(" & ")
		}
		b.write(c.Name)
		b.write("(")
		for j, t := range c.Terms {
			if j > 0 {
				b.write(",")
			}
			switch t.Kind {
			case egraphcore.WildcardTerm:
				b.write("_")
			case egraphcore.VariableTerm:
				b.write("?")
				b.write(t.Name)
			case egraphcore.ConstantTerm:
				b.value(t.Value, false)
			default:
				b.write("invalid:")
				b.uint(uint64(t.Kind), 10)
			}
		}
		b.write(")")
	}
	for i, a := range r.RelationActions {
		if i == 0 {
			b.write(" do ")
		} else {
			b.write(" & ")
		}
		b.uint(uint64(a.Kind), 10)
		b.write(":")
		b.write(a.Relation)
		b.write("(")
		for j, t := range a.Terms {
			if j > 0 {
				b.write(",")
			}
			switch t.Kind {
			case egraphcore.VariableTerm:
				b.write("?")
				b.write(t.Name)
			case egraphcore.ConstantTerm:
				b.write("=")
				b.value(t.Value, true)
			default:
				b.write("invalid:")
				b.uint(uint64(t.Kind), 10)
			}
		}
		b.write(")")
	}
	return b.finish()
}
func classFingerprintWithBudget(g *EGraph, id Id, budget *egraphcore.WorkBudget) (string, error) {
	if budget.MemoryScope() == nil {
		return classFingerprint(g, id), nil
	}
	c := g.class(id)
	if c == nil {
		return "<missing>", nil
	}
	parts, err := graphSlice[string](budget, len(c.Nodes), len(c.Nodes))
	if err != nil {
		return "", err
	}
	for i, n := range c.Nodes {
		parts[i], err = lookupNodeKey(n, budget)
		if err != nil {
			return "", err
		}
	}
	sort.Strings(parts)
	b := memoryText{budget: budget}
	for i, s := range parts {
		if i > 0 {
			b.write("|")
		}
		b.write(s)
	}
	return b.finish()
}

func copyActionBindings(subst Subst, binding egraphcore.Binding[Id], budget *egraphcore.WorkBudget) (Subst, egraphcore.Binding[Id], error) {
	s, err := graphMap[string, Id](budget, len(subst))
	if err != nil {
		return nil, nil, err
	}
	for name, id := range subst {
		if err := graphPut(budget, s, name, id); err != nil {
			return nil, nil, err
		}
	}
	b, err := graphMap[string, egraphcore.Value[Id]](budget, len(binding))
	if err != nil {
		return nil, nil, err
	}
	for name, value := range binding {
		if err := graphPut(budget, b, name, value); err != nil {
			return nil, nil, err
		}
	}
	return Subst(s), b, nil
}
