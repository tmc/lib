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

func TestNativeFunctionCall(t *testing.T) {
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
	got, err := testutil.Exec1(b, []any{[]int64{9007199254740993, -7}}, func(main compute.Function, v []compute.Value) (compute.Value, error) {
		sub, err := main.Builder().NewFunction("pair with spaces")
		if err != nil {
			return nil, err
		}
		x, err := sub.Parameter("x", shapes.Make(dtypes.Int64, 2), nil)
		if err != nil {
			return nil, err
		}
		one, err := sub.Constant([]int64{1, 1}, 2)
		if err != nil {
			return nil, err
		}
		plus, err := sub.Add(x, one)
		if err != nil {
			return nil, err
		}
		minus, err := sub.Sub(x, one)
		if err != nil {
			return nil, err
		}
		if err := sub.Return([]compute.Value{plus, minus}, nil); err != nil {
			return nil, err
		}
		wrapper, err := main.Builder().NewFunction("wrapper")
		if err != nil {
			return nil, err
		}
		wx, err := wrapper.Parameter("x", shapes.Make(dtypes.Int64, 2), nil)
		if err != nil {
			return nil, err
		}
		pair, err := wrapper.Call(sub, wx)
		if err != nil {
			return nil, err
		}
		if err := wrapper.Return(pair, nil); err != nil {
			return nil, err
		}
		a, err := main.Call(wrapper, v[0])
		if err != nil {
			return nil, err
		}
		c, err := main.Call(sub, a[0])
		if err != nil {
			return nil, err
		}
		return main.Sub(c[0], a[1])
	})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, []int64{3, 3}) {
		t.Fatalf("got%v", got)
	}
}

func TestFunctionValidation(t *testing.T) {
	b := (&Backend{}).Builder("functions")
	main := b.Main()
	sub, err := b.NewFunction("sub")
	if err != nil {
		t.Fatal(err)
	}
	if sub.Name() != "sub" || sub.Parent() != nil || main.Name() != "main" {
		t.Fatal("function identity mismatch")
	}
	x, err := sub.Parameter("x", shapes.Make(dtypes.Int32), nil)
	if err != nil {
		t.Fatal(err)
	}
	y, err := main.Parameter("x", shapes.Make(dtypes.Int32), nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := sub.Add(x, y); err == nil {
		t.Fatal("named function captured parent")
	}
	if _, err := main.Call(sub, y); err == nil {
		t.Fatal("called unfinished target")
	}
	if err := sub.Return([]compute.Value{x}, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := sub.Add(x, x); err == nil {
		t.Fatal("modified returned function")
	}
	if _, err := main.Add(y, y); err != nil {
		t.Fatal("sub return closed main", err)
	}
	wrong, err := main.Parameter("wrong", shapes.Make(dtypes.Int64), nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := b.OpShape(x); err != nil {
		t.Fatal("builder rejected subfunction shape", err)
	}
	foreign := (&Backend{}).Builder("foreign").Main()
	for _, tc := range []struct {
		name   string
		target compute.Function
		inputs []compute.Value
	}{
		{"foreign", foreign, []compute.Value{y}}, {"main", main, []compute.Value{y}}, {"shape", sub, []compute.Value{wrong}}, {"count", sub, nil}, {"ownership", sub, []compute.Value{x}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			before := len(main.(*function).raw.Ops)
			if _, err := main.Call(tc.target, tc.inputs...); err == nil {
				t.Fatal("accepted invalid call")
			}
			if len(main.(*function).raw.Ops) != before {
				t.Fatal("invalid call emitted IR")
			}
		})
	}
	if _, err := b.NewFunction("sub"); err == nil {
		t.Fatal("accepted duplicate function")
	}
}
