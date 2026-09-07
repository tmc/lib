package builder_test

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/x/stablehlo"
	"github.com/tmc/lib/x/stablehlo/builder"
)

// TestNativeCPU is an optional integration gate. STABLEHLO_PYTHON must name a
// Python interpreter with JAX and NumPy installed. It validates generated IR,
// not the Go PJRT transport, and never selects an accelerator backend.
func TestNativeCPU(t *testing.T) {
	python := os.Getenv("STABLEHLO_PYTHON")
	if python == "" {
		t.Skip("set STABLEHLO_PYTHON to an interpreter with JAX and NumPy")
	}
	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatal(err)
	}
	base := filepath.Join(home, "tmp")
	if err := os.MkdirAll(base, 0755); err != nil {
		t.Fatal(err)
	}
	dir, err := os.MkdirTemp(base, "stablehlo-cpu-")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)
	for _, name := range []string{"causal_dynamic", "update", "matmul", "rmsnorm", "tiny", "softmax", "attention", "attention_f16", "attention_bf16", "attention_masked", "attention_masked_f16", "attention_masked_bf16"} {
		var b *builder.FuncBuilder
		var out builder.Value
		var err error
		typ := func(dims ...int64) builder.Type { return builder.Type{Shape: dims, ElementType: "f32"} }
		switch name {
		case "causal_dynamic":
			b = builder.NewFunc("main", builder.Type{ElementType: "i64"})
			out, err = b.CausalMaskAt(2, 3, b.Arg(0))
		case "update":
			b = builder.NewFunc("main", builder.Type{Shape: []int64{4}, ElementType: "i64"}, builder.Type{Shape: []int64{2}, ElementType: "i64"}, builder.Type{ElementType: "i64"})
			out, err = b.DynamicUpdateSlice(b.Arg(0), b.Arg(1), []builder.Value{b.Arg(2)})
		case "matmul":
			b = builder.NewFunc("main", typ(2, 3), typ(3, 4))
			out, err = b.DotGeneral(b.Arg(0), b.Arg(1), nil, nil, []int64{1}, []int64{0}, typ(2, 4))
		case "rmsnorm":
			b = builder.NewFunc("main", typ(2, 3), typ(3))
			out, err = b.RMSNorm(b.Arg(0), b.Arg(1), 1, 1e-8)
		case "tiny":
			b = builder.NewFunc("main")
			out, err = b.Constant(1e-8, typ())
		case "softmax":
			b = builder.NewFunc("main", typ(2, 3))
			out, err = b.Softmax(b.Arg(0), 1)
		case "attention", "attention_f16", "attention_bf16", "attention_masked", "attention_masked_f16", "attention_masked_bf16":
			if strings.HasSuffix(name, "_f16") || strings.HasSuffix(name, "_bf16") {
				element := name[strings.LastIndex(name, "_")+1:]
				typ = func(dims ...int64) builder.Type { return builder.Type{Shape: dims, ElementType: element} }
			}
			b = builder.NewFunc("main", typ(2, 3, 4), typ(2, 5, 4), typ(2, 5, 6))
			if strings.Contains(name, "masked") {
				b = builder.NewFunc("main", typ(2, 3, 4), typ(2, 5, 4), typ(2, 5, 6), builder.Type{Shape: []int64{2, 1, 5}, ElementType: "i1"})
				mask, maskErr := b.CausalMask(3, 5, 0)
				if maskErr != nil {
					t.Fatal(maskErr)
				}
				full := builder.Type{Shape: []int64{2, 3, 5}, ElementType: "i1"}
				mask, maskErr = b.BroadcastInDim(mask, []int64{1, 2}, full)
				if maskErr != nil {
					t.Fatal(maskErr)
				}
				padding, maskErr := b.BroadcastInDim(b.Arg(3), []int64{0, 1, 2}, full)
				if maskErr != nil {
					t.Fatal(maskErr)
				}
				mask, maskErr = b.And(mask, padding)
				if maskErr != nil {
					t.Fatal(maskErr)
				}
				out, err = b.MaskedScaledDotProductAttention(b.Arg(0), b.Arg(1), b.Arg(2), mask)
			} else {
				out, err = b.ScaledDotProductAttention(b.Arg(0), b.Arg(1), b.Arg(2))
			}
		}
		if err != nil {
			t.Fatal(err)
		}
		b.Return(out)
		f, err := os.Create(filepath.Join(dir, name+".mlir"))
		if err != nil {
			t.Fatal(err)
		}
		err = (&stablehlo.Module{Funcs: []stablehlo.Emitter{b}}).Emit(f)
		closeErr := f.Close()
		if err != nil {
			t.Fatal(err)
		}
		if closeErr != nil {
			t.Fatal(closeErr)
		}
	}
	run := func() ([]byte, error) { return exec.Command(python, "testdata/verify_cpu.py", dir).CombinedOutput() }
	output, err := run()
	if err != nil || !strings.Contains(string(output), "verified 12 CPU programs") {
		t.Fatalf("native CPU verification: %v\n%s", err, output)
	}
	t.Log(string(output))
	path := filepath.Join(dir, "attention.mlir")
	text, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	mutant := strings.Replace(string(text), "dense<0.5>", "dense<1.0>", 1)
	if mutant == string(text) {
		t.Fatal("attention-scale mutation did not apply")
	}
	if err := os.WriteFile(path, []byte(mutant), 0600); err != nil {
		t.Fatal(err)
	}
	output, err = run()
	if err == nil || !strings.Contains(string(output), "Not equal to tolerance") {
		t.Fatalf("attention mutation not detected: %v\n%s", err, output)
	}
}
