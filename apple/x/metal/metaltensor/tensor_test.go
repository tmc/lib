package metaltensor

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/apple/x/metal/metalir"
)

func TestTensorBuildsSymbolicGraph(t *testing.T) {
	g := metalir.NewGraph()
	x := NewInput(g, "x", []int{2, 3}, metalir.Float32)
	w := NewInput(g, "w", []int{3, 4}, metalir.Float32)
	y := x.MatMul(w).Silu()
	if y.Graph() != g || y.Node() != g.Nodes[len(g.Nodes)-1] {
		t.Fatal("tensor does not retain graph node")
	}
	if got := y.Shape(); len(got) != 2 || got[0] != 2 || got[1] != 4 {
		t.Fatalf("shape = %v", got)
	}
	if y.DataType() != metalir.Float32 || y.Node().Op != "silu" {
		t.Fatalf("dtype=%v op=%q", y.DataType(), y.Node().Op)
	}
	if y.Node().SourceLoc.File == "" || y.Node().SourceLoc.Line == 0 || !strings.Contains(y.Node().SourceLoc.Function, "TestTensorBuildsSymbolicGraph") {
		t.Fatalf("source location = %+v", y.Node().SourceLoc)
	}
}

func TestCompileProducesValidMSL(t *testing.T) {
	metal, err := exec.LookPath("xcrun")
	if err != nil {
		t.Skip("xcrun not installed")
	}
	g := metalir.NewGraph()
	x := NewInput(g, "x", []int{8}, metalir.Float32)
	y := NewInput(g, "y", []int{8}, metalir.Float32)
	msl, err := x.Silu().Mul(y).Compile("syntax_test")
	if err != nil {
		t.Fatal(err)
	}
	src := filepath.Join(t.TempDir(), "kernel.metal")
	if err := os.WriteFile(src, []byte(msl), 0o600); err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(metal, "metal", "-c", src, "-o", filepath.Join(t.TempDir(), "kernel.air"))
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("compile MSL: %v\n%s\n%s", err, out, msl)
	}
}

func TestCompileEmitsReceiverExpression(t *testing.T) {
	g := metalir.NewGraph()
	x := NewInput(g, "x", []int{1, 8}, metalir.Float32)
	y := NewInput(g, "y", []int{1, 8}, metalir.Float32)
	msl, err := x.Silu().Mul(y).Add(x).Compile("characterized_kernel")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(msl, "characterized_kernel") {
		t.Fatalf("MSL does not contain kernel name:\n%s", msl)
	}
	if !strings.Contains(msl, "kernel void") {
		t.Fatalf("MSL does not contain a kernel declaration:\n%s", msl)
	}
	for _, want := range []string{"exp(-v", " * ", " + ", "out[i]"} {
		if !strings.Contains(msl, want) {
			t.Fatalf("MSL missing %q:\n%s", want, msl)
		}
	}
}

func TestCompileRejectsUnsupportedOrInvalid(t *testing.T) {
	tests := []struct {
		name    string
		compile func() error
	}{
		{"kernel name", func() error {
			g := metalir.NewGraph()
			x := NewInput(g, "x", []int{1}, metalir.Float32)
			_, err := x.Compile("bad-name")
			return err
		}},
		{"matmul", func() error {
			g := metalir.NewGraph()
			a := NewInput(g, "a", []int{2, 3}, metalir.Float32)
			b := NewInput(g, "b", []int{3, 2}, metalir.Float32)
			_, err := a.MatMul(b).Compile("k")
			return err
		}},
		{"dtype", func() error {
			g := metalir.NewGraph()
			x := NewInput(g, "x", []int{1}, metalir.Float16)
			_, err := x.Compile("k")
			return err
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.compile(); err == nil {
				t.Fatal("Compile succeeded")
			}
		})
	}
}
