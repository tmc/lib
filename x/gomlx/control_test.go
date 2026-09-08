package gomlx

import (
	"os"
	"reflect"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/testutil"
)

func TestNativeCapturedControlFlow(t *testing.T) {
	path := os.Getenv("TMC_COMPUTE_PLUGIN")
	if path == "" {
		t.Skip("set TMC_COMPUTE_PLUGIN")
	}
	b, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := b.Close(); err != nil {
			t.Error(err)
		}
	}()
	for _, tc := range []struct {
		name         string
		limit, delta int32
		pred         bool
		want         int32
	}{
		{"zero iterations", 0, 2, true, -2}, {"one iteration", 1, 3, true, 0}, {"many iterations", 7, -2, true, -12}, {"false branch", 3, 2, false, -6},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := testutil.Exec1(b, []any{tc.limit, tc.delta, tc.pred}, func(main compute.Function, v []compute.Value) (compute.Value, error) {
				value := func(v compute.Value, err error) compute.Value {
					if err != nil {
						t.Fatal(err)
					}
					return v
				}
				closure := func(f compute.Function) compute.Function {
					c, err := f.Closure()
					if err != nil {
						t.Fatal(err)
					}
					if c.Parent() != f || c.Name() != "" {
						t.Fatal("closure identity")
					}
					return c
				}
				param := func(f compute.Function, name string) compute.Value {
					return value(f.Parameter(name, shapes.Make(dtypes.Int32), nil))
				}
				ret := func(f compute.Function, v ...compute.Value) {
					if err := f.Return(v, nil); err != nil {
						t.Fatal(err)
					}
				}
				cond := closure(main)
				i := param(cond, "i")
				param(cond, "sum")
				ret(cond, value(cond.LessThan(i, v[0])))
				body := closure(main)
				bi := param(body, "i")
				sum := param(body, "sum")
				one := value(body.Constant([]int32{1}))
				ret(body, value(body.Add(bi, one)), value(body.Add(sum, v[1])))
				zero := value(main.Constant([]int32{0}))
				loop, err := main.While(cond, body, zero, zero)
				if err != nil {
					return nil, err
				}
				yes := closure(main)
				nested := closure(yes)
				// Capture both a main parameter and a preceding loop result through yes.
				result := value(nested.Sub(loop[1], v[1]))
				ret(nested, result)
				called, err := yes.Call(nested)
				if err != nil {
					return nil, err
				}
				ret(yes, called...)
				no := closure(main)
				ret(no, value(no.Neg(loop[1])))
				resultValues, err := main.If(v[2], yes, no)
				if err != nil {
					return nil, err
				}
				return resultValues[0], nil
			})
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("got%v want%v", got, tc.want)
			}
		})
	}
}

func TestControlValidation(t *testing.T) {
	b := (&Backend{}).Builder("control")
	main := b.Main()
	value := func(v compute.Value, err error) compute.Value {
		if err != nil {
			t.Fatal(err)
		}
		return v
	}
	closure := func(f compute.Function) compute.Function {
		c, err := f.Closure()
		if err != nil {
			t.Fatal(err)
		}
		return c
	}
	ret := func(f compute.Function, v ...compute.Value) {
		if err := f.Return(v, nil); err != nil {
			t.Fatal(err)
		}
	}
	pred := value(main.Parameter("pred", shapes.Make(dtypes.Bool), nil))
	x := value(main.Parameter("x", shapes.Make(dtypes.Int32), nil))
	yes := closure(main)
	ret(yes, x)
	no := closure(main)
	ret(no, pred)
	cond := closure(main)
	value(cond.Parameter("x", shapes.Make(dtypes.Int32), nil))
	ret(cond, pred)
	body := closure(main)
	bodyValue := value(body.Parameter("x", shapes.Make(dtypes.Int32), nil))
	ret(body, pred)
	sibling := closure(main)
	foreign := closure(sibling)
	ret(foreign, x)
	for _, tc := range []struct {
		name string
		call func() error
	}{
		{"predicate type", func() error { _, err := main.If(x, yes, yes); return err }},
		{"branch shape", func() error { _, err := main.If(pred, yes, no); return err }},
		{"condition shape", func() error { _, err := main.While(yes, body); return err }},
		{"body shape", func() error { _, err := main.While(cond, body, x); return err }},
		{"closure escape", func() error { _, err := main.Call(foreign); return err }},
		{"sibling value", func() error { _, err := sibling.Add(x, bodyValue); return err }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			before := len(main.(*function).raw.Ops)
			if err := tc.call(); err == nil {
				t.Fatal("accepted invalid control flow")
			}
			if len(main.(*function).raw.Ops) != before {
				t.Fatal("invalid control flow emitted IR")
			}
		})
	}
}

func TestNativeCapturedCall(t *testing.T) {
	path := os.Getenv("TMC_COMPUTE_PLUGIN")
	if path == "" {
		t.Skip("set TMC_COMPUTE_PLUGIN")
	}
	b, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := b.Close(); err != nil {
			t.Error(err)
		}
	}()
	got, err := testutil.Exec1(b, []any{int32(7), int32(2), int32(3)}, func(main compute.Function, v []compute.Value) (compute.Value, error) {
		c, err := main.Closure()
		if err != nil {
			return nil, err
		}
		diff, err := c.Sub(v[0], v[1])
		if err != nil {
			return nil, err
		}
		// Explicit parameters may be declared after captures have been discovered.
		scale, err := c.Parameter("scale", shapes.Make(dtypes.Int32), nil)
		if err != nil {
			return nil, err
		}
		product, err := c.Mul(diff, scale)
		if err != nil {
			return nil, err
		}
		if err := c.Return([]compute.Value{product}, nil); err != nil {
			return nil, err
		}
		out, err := main.Call(c, v[2])
		if err != nil {
			return nil, err
		}
		return out[0], nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if got != int32(15) {
		t.Fatalf("got%v want15", got)
	}
}
