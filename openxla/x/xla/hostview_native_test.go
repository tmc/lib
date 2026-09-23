package xla

import (
	"context"
	"errors"
	"os"
	"slices"
	"testing"
)

func TestNativeHostView(t *testing.T) {
	path := os.Getenv("XLA_PJRT_CPU_PLUGIN")
	if path == "" {
		t.Skip("set XLA_PJRT_CPU_PLUGIN")
	}
	e, err := NewEngine(path)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	p, err := e.Compile([]byte(`module { func.func @main() -> tensor<2x3xf32> { %x = stablehlo.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32> return %x : tensor<2x3xf32> } }`), "mlir", nil)
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	out, err := p.Execute(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	b := out[0]
	defer b.Close()
	if err := b.WithReadOnlyFloats(func([]float32) error { t.Fatal("unqualified callback"); return nil }); !errors.Is(err, ErrHostAccessUnavailable) {
		t.Fatalf("default: %v", err)
	}
	if err := e.EnableCPUHostAccess(); err != nil {
		t.Fatal(err)
	}
	want := make([]float32, 6)
	if err := b.ReadFloats(want); err != nil {
		t.Fatal(err)
	}
	callbackErr := errors.New("callback sentinel")
	for _, fail := range []bool{false, true, false} {
		err := b.WithReadOnlyFloats(func(got []float32) error {
			if !slices.Equal(got, want) {
				t.Fatalf("direct %v copied %v", got, want)
			}
			if err := b.Close(); err == nil {
				t.Fatal("closed active view")
			}
			if fail {
				return callbackErr
			}
			return nil
		})
		if fail {
			if !errors.Is(err, callbackErr) {
				t.Fatal(err)
			}
		} else if err != nil {
			t.Fatal(err)
		}
	}
	if err := b.Close(); err != nil {
		t.Fatal(err)
	}
	if err := b.WithReadOnlyFloats(func([]float32) error { return nil }); err == nil {
		t.Fatal("closed view accepted")
	}
	for _, shape := range []string{"f32", "0xf32"} {
		t.Run(shape, func(t *testing.T) {
			program := `module { func.func @main() -> tensor<` + shape + `> { %x = stablehlo.constant dense<0.0> : tensor<` + shape + `> return %x : tensor<` + shape + `> } }`
			exe, err := e.Compile([]byte(program), "mlir", nil)
			if err != nil {
				t.Fatal(err)
			}
			defer exe.Close()
			outputs, err := exe.Execute(context.Background(), nil)
			if err != nil {
				t.Fatal(err)
			}
			defer outputs[0].Close()
			err = outputs[0].WithReadOnlyFloats(func(v []float32) error {
				n := 1
				if shape == "0xf32" {
					n = 0
				}
				if len(v) != n {
					t.Fatalf("length %d want %d", len(v), n)
				}
				return nil
			})
			if err != nil {
				t.Fatal(err)
			}
		})
	}
}
