package gomlx

import (
	"fmt"
	"slices"
	"strings"

	"github.com/gomlx/compute"
	hlo "github.com/tmc/lib/x/stablehlo/builder"
)

// callRegion emits a call and terminator for an operation's region. Captures
// remain values in the enclosing function; only the call belongs to the region.
func (f *function) callRegion(target compute.Function, inputs []compute.Value) (string, []hlo.Type, error) {
	n := len(f.raw.Ops)
	out, err := f.Call(target, inputs...)
	if err != nil {
		return "", nil, err
	}
	call := f.raw.Ops[n]
	f.raw.Ops = f.raw.Ops[:n]
	ids := make([]string, len(out))
	types := make([]hlo.Type, len(out))
	names := make([]string, len(out))
	for i, o := range out {
		v := o.(*node).value
		ids[i] = v.ID
		types[i] = v.Type
		names[i] = v.Type.String()
	}
	terminator := "stablehlo.return"
	if len(out) > 0 {
		terminator += " " + strings.Join(ids, ", ") + " : " + strings.Join(names, ", ")
	}
	return call + "\n" + terminator, types, nil
}
func sameType(a, b hlo.Type) bool {
	return a.ElementType == b.ElementType && slices.Equal(a.Shape, b.Shape)
}

func (f *function) regionResult(kind, op string, types []hlo.Type) []compute.Value {
	out := make([]compute.Value, len(types))
	ids := make([]string, len(types))
	for i, t := range types {
		v := hlo.Value{ID: fmt.Sprintf("%%%s%d_%d", kind, len(f.raw.Ops), i), Type: t}
		out[i] = &node{f, v}
		ids[i] = v.ID
	}
	prefix := ""
	if len(ids) > 0 {
		prefix = strings.Join(ids, ", ") + " = "
	}
	f.raw.Ops = append(f.raw.Ops, prefix+op)
	return out
}
func typeNames(types []hlo.Type) string {
	names := make([]string, len(types))
	for i, t := range types {
		names[i] = t.String()
	}
	return strings.Join(names, ", ")
}

func (f *function) If(pred compute.Value, yes, no compute.Function) ([]compute.Value, error) {
	if f.returned || f.owner.closed {
		return nil, fmt.Errorf("function closed")
	}
	p, err := f.value(pred)
	if err != nil {
		return nil, err
	}
	if p.Type.ElementType != "i1" || len(p.Type.Shape) != 0 {
		return nil, fmt.Errorf("if: scalar boolean predicate required")
	}
	a, at, err := f.callRegion(yes, nil)
	if err != nil {
		return nil, err
	}
	b, bt, err := f.callRegion(no, nil)
	if err != nil {
		return nil, err
	}
	if len(at) != len(bt) {
		return nil, fmt.Errorf("if: branch output count mismatch")
	}
	for i := range at {
		if !sameType(at[i], bt[i]) {
			return nil, fmt.Errorf("if: branch output %d shape mismatch", i)
		}
	}
	op := fmt.Sprintf("\"stablehlo.if\"(%s) ({\n%s\n}, {\n%s\n}) : (%s) -> (%s)", p.ID, a, b, p.Type, typeNames(at))
	return f.regionResult("if", op, at), nil
}

func (f *function) While(cond, body compute.Function, initial ...compute.Value) ([]compute.Value, error) {
	if f.returned || f.owner.closed {
		return nil, fmt.Errorf("function closed")
	}
	types := make([]hlo.Type, len(initial))
	ids := make([]string, len(initial))
	args := make([]string, len(initial))
	state := make([]compute.Value, len(initial))
	for i, input := range initial {
		v, err := f.value(input)
		if err != nil {
			return nil, err
		}
		types[i], ids[i] = v.Type, v.ID
		arg := hlo.Value{ID: fmt.Sprintf("%%state%d_%d", len(f.raw.Ops), i), Type: v.Type}
		args[i] = arg.ID + ": " + arg.Type.String()
		state[i] = &node{f, arg}
	}
	c, ct, err := f.callRegion(cond, state)
	if err != nil {
		return nil, err
	}
	if len(ct) != 1 || ct[0].ElementType != "i1" || len(ct[0].Shape) != 0 {
		return nil, fmt.Errorf("while: condition must return a scalar boolean")
	}
	b, bt, err := f.callRegion(body, state)
	if err != nil {
		return nil, err
	}
	if len(bt) != len(types) {
		return nil, fmt.Errorf("while: body output count mismatch")
	}
	for i := range bt {
		if !sameType(bt[i], types[i]) {
			return nil, fmt.Errorf("while: body output %d shape mismatch", i)
		}
	}
	block := "^bb0(" + strings.Join(args, ", ") + "):"
	op := fmt.Sprintf("\"stablehlo.while\"(%s) ({\n%s\n%s\n}, {\n%s\n%s\n}) : (%s) -> (%s)", strings.Join(ids, ", "), block, c, block, b, typeNames(types), typeNames(types))
	return f.regionResult("while", op, types), nil
}
