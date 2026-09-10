package gomlx

import (
	"fmt"
	"slices"
	"strings"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/shapes"
	hlo "github.com/tmc/lib/x/stablehlo/builder"
)

func (f *function) Name() string { return f.name }
func (f *function) Parent() compute.Function {
	if f.parent == nil {
		return nil
	}
	return f.parent
}
func (f *function) Closure() (compute.Function, error) {
	if f.owner.closed || f.returned {
		return nil, fmt.Errorf("function closed")
	}
	c := &function{owner: f.owner, parent: f, raw: hlo.NewFunc(fmt.Sprintf("function%d", len(f.owner.functions))), valueShapes: make(map[string]shapes.Shape)}
	c.Function.ErrFn = func(op compute.OpType) error { return fmt.Errorf("%s: %w", op, unsupported) }
	f.owner.functions = append(f.owner.functions, c)
	return c, nil
}

func (b *builder) NewFunction(name string) (compute.Function, error) {
	if b.closed {
		return nil, fmt.Errorf("builder closed")
	}
	if name == "" {
		return nil, fmt.Errorf("function name is empty")
	}
	for _, f := range b.functions {
		if f.name == name {
			return nil, fmt.Errorf("duplicate function name %q", name)
		}
	}
	f := &function{owner: b, name: name, raw: hlo.NewFunc(fmt.Sprintf("function%d", len(b.functions))), valueShapes: make(map[string]shapes.Shape)}
	f.Function.ErrFn = func(op compute.OpType) error { return fmt.Errorf("%s: %w", op, unsupported) }
	b.functions = append(b.functions, f)
	return f, nil
}

func (f *function) Call(target compute.Function, inputs ...compute.Value) ([]compute.Value, error) {
	if f.owner.closed || f.returned {
		return nil, fmt.Errorf("function closed")
	}
	callee, ok := target.(*function)
	if !ok || callee == nil || callee.owner != f.owner || callee == f.owner.main {
		return nil, fmt.Errorf("invalid call target")
	}
	if callee.parent != nil {
		visible := false
		for p := f; p != nil; p = p.parent {
			if p == callee.parent {
				visible = true
				break
			}
		}
		if !visible {
			return nil, fmt.Errorf("call target is outside lexical scope")
		}
	}
	// Requiring a completed target also prevents recursive call graphs.
	if !callee.returned {
		return nil, fmt.Errorf("call target has not returned")
	}
	if len(inputs) != len(callee.inputs) {
		return nil, fmt.Errorf("call input count mismatch")
	}
	ids := make([]string, len(inputs))
	types := make([]string, len(inputs))
	for i, input := range inputs {
		v, err := f.value(input)
		if err != nil {
			return nil, err
		}
		want := callee.raw.Args[i].Type
		if v.Type.ElementType != want.ElementType || !slices.Equal(v.Type.Shape, want.Shape) {
			return nil, fmt.Errorf("call input %d shape mismatch", i)
		}
		ids[i], types[i] = v.ID, v.Type.String()
	}
	for _, capture := range callee.captures {
		v, err := f.value(capture)
		if err != nil {
			return nil, err
		}
		ids = append(ids, v.ID)
		types = append(types, v.Type.String())
	}
	out := make([]compute.Value, len(callee.raw.Returns))
	resultIDs := make([]string, len(out))
	resultTypes := make([]string, len(out))
	for i, r := range callee.raw.Returns {
		v := hlo.Value{ID: fmt.Sprintf("%%call%d_%d", len(f.raw.Ops), i), Type: r.Type}
		out[i] = &node{f, v}
		resultIDs[i] = v.ID
		resultTypes[i] = v.Type.String()
	}
	prefix := ""
	if len(out) > 0 {
		prefix = strings.Join(resultIDs, ", ") + " = "
	}
	f.raw.Ops = append(f.raw.Ops, fmt.Sprintf("%sfunc.call @%s(%s) : (%s) -> (%s)", prefix, callee.raw.Name, strings.Join(ids, ", "), strings.Join(types, ", "), strings.Join(resultTypes, ", ")))
	return out, nil
}
