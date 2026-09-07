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

func TestNativeModelOps(t *testing.T) {
	python := os.Getenv("STABLEHLO_PYTHON")
	if python == "" {
		t.Skip("set STABLEHLO_PYTHON to an interpreter with JAX and NumPy")
	}
	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatal(err)
	}
	dir, err := os.MkdirTemp(filepath.Join(home, "tmp"), "stablehlo-model-ops-")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)
	write := func(name string, b *builder.FuncBuilder) {
		t.Helper()
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
	for _, element := range []string{"f32", "f16", "bf16"} {
		typ := builder.Type{Shape: []int64{2, 4, 3, 6}, ElementType: element}
		table := builder.Type{Shape: []int64{2, 1, 3, 2}, ElementType: "f32"}
		b := builder.NewFunc("main", typ, typ, table, table)
		a, err := b.SiLU(b.Arg(0))
		if err != nil {
			t.Fatal(err)
		}
		g, err := b.SwiGLU(b.Arg(0), b.Arg(1))
		if err != nil {
			t.Fatal(err)
		}
		r, err := b.RotaryEmbedding(b.Arg(0), b.Arg(2), b.Arg(3))
		if err != nil {
			t.Fatal(err)
		}
		b.Return(a, g, r)
		write(element, b)
	}
	for _, name := range []string{"gather", "gather_scalar", "gather_vector"} {
		operand := builder.Type{Shape: []int64{4, 2, 3}, ElementType: "f32"}
		indices := builder.Type{Shape: []int64{2, 3}, ElementType: "i64"}
		if name == "gather_scalar" {
			indices.Shape = nil
		}
		if name == "gather_vector" {
			operand.Shape = []int64{4}
			indices.ElementType = "i32"
		}
		b := builder.NewFunc("main", operand, indices)
		out, err := b.GatherRows(b.Arg(0), b.Arg(1))
		if err != nil {
			t.Fatal(err)
		}
		b.Return(out)
		write(name, b)
	}
	// Full rotation with sequence tables broadcasts across both batch and heads.
	b := builder.NewFunc("main", builder.Type{Shape: []int64{2, 2, 3, 4}, ElementType: "f32"}, builder.Type{Shape: []int64{3, 2}, ElementType: "f32"}, builder.Type{Shape: []int64{3, 2}, ElementType: "f32"})
	out, err := b.RotaryEmbedding(b.Arg(0), b.Arg(1), b.Arg(2))
	if err != nil {
		t.Fatal(err)
	}
	b.Return(out)
	write("rotary_full", b)

	scalar := builder.Type{ElementType: "f32"}
	scalarBuilder := builder.NewFunc("main", scalar)
	scalarOut, err := scalarBuilder.Slice(scalarBuilder.Arg(0), nil, nil, nil, scalar)
	if err != nil {
		t.Fatal(err)
	}
	scalarBuilder.Return(scalarOut)
	write("slice_scalar", scalarBuilder)
	input := builder.Type{Shape: []int64{2, 3, 6}, ElementType: "f32"}
	sliced := builder.Type{Shape: []int64{2, 2, 3}, ElementType: "f32"}
	slicedBuilder := builder.NewFunc("main", input)
	part, err := slicedBuilder.Slice(slicedBuilder.Arg(0), []int64{0, 0, 1}, []int64{2, 3, 6}, []int64{1, 2, 2}, sliced)
	if err != nil {
		t.Fatal(err)
	}
	joined, err := slicedBuilder.Concatenate([]builder.Value{part, part}, 1, builder.Type{Shape: []int64{2, 4, 3}, ElementType: "f32"})
	if err != nil {
		t.Fatal(err)
	}
	slicedBuilder.Return(joined)
	write("slice_concat", slicedBuilder)
	run := func() ([]byte, error) {
		return exec.Command(python, "testdata/verify_model_ops.py", dir).CombinedOutput()
	}
	output, err := run()
	if err != nil || !strings.Contains(string(output), "verified 9 model operation programs") {
		t.Fatalf("model operations: %v\n%s", err, output)
	}
	t.Log(string(output))
	p := filepath.Join(dir, "f32.mlir")
	data, err := os.ReadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	mutant := strings.Replace(string(data), "stablehlo.subtract", "stablehlo.add", 1)
	if mutant == string(data) {
		t.Fatal("rotation mutation did not apply")
	}
	if err := os.WriteFile(p, []byte(mutant), 0600); err != nil {
		t.Fatal(err)
	}
	output, err = run()
	if err == nil || !strings.Contains(string(output), "Not equal to tolerance") {
		t.Fatalf("rotation mutation escaped: %v\n%s", err, output)
	}
	t.Log("rotation-sign mutation compiled and failed numerical reference")
}
