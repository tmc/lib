package main

import (
	"bytes"
	"context"
	"path/filepath"
	"strings"
	"testing"
)

const testdata = "../../cudalineinfo/testdata"

// setFlags points the artifact flags at a fixture, which was compiled from
// the annotated source beside it.
func setFlags(t *testing.T, unit, f string) {
	t.Helper()
	oldSASS, oldPTX, oldFormat := *sassFile, *ptxFile, *format
	*sassFile = filepath.Join(testdata, unit+".sass")
	*ptxFile = filepath.Join(testdata, unit+".ptx")
	*format = f
	t.Cleanup(func() { *sassFile, *ptxFile, *format = oldSASS, oldPTX, oldFormat })
}

func TestFormats(t *testing.T) {
	for _, tt := range []struct{ format, want string }{
		{"table", "kernels.go:14"},
		{"listing", "FMUL"},
		{"rollup", "kernels.go:14"},
		{"digraph", "kernels.go:14 $__internal_0_$__cuda_sm3x_div_rn_noftz_f32_slowpath fused_gate\n"},
		{"dot", `"kernels.go:14" -> "fused_gate";`},
	} {
		setFlags(t, "fused_gate", tt.format)
		var buf bytes.Buffer
		if err := run(context.Background(), filepath.Join(testdata, "fused_gate.cu"), &buf); err != nil {
			t.Fatalf("%s: %v", tt.format, err)
		}
		if !strings.Contains(buf.String(), tt.want) {
			t.Errorf("%s output lacks %q:\n%s", tt.format, tt.want, buf.String())
		}
	}
}

// TestUnitJSON reads a unit exported by the compiler path, whose line map
// carries the function and callers of every line, and checks that the graph
// reaches the kernel through them.
func TestUnitJSON(t *testing.T) {
	setFlags(t, "fused_layered", "digraph")
	var buf bytes.Buffer
	if err := run(context.Background(), filepath.Join(testdata, "fused_layered.json"), &buf); err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"main.main main.run\n",
		"main.attribute main.layered\n",
		"main.layered kernels.go:38 kernels.go:39 main.activate main.normalize\n",
		"main.activate kernels.go:45\n",
	} {
		if !strings.Contains(buf.String(), want) {
			t.Errorf("digraph lacks %q:\n%s", want, buf.String())
		}
	}
}

func TestUnknownFormat(t *testing.T) {
	setFlags(t, "fused_gate", "csv")
	if err := run(context.Background(), filepath.Join(testdata, "fused_gate.cu"), &bytes.Buffer{}); err == nil {
		t.Error("no error for an unknown format")
	}
}

// A CUBIN carries no PTX, so attributing one without -ptx would report every
// line as named by no PTX. The refusal is what keeps the missing tier from
// reading as an empty one.
func TestCubinNeedsPTX(t *testing.T) {
	oldCubin, oldPTX := *cubin, *ptxFile
	*cubin, *ptxFile = filepath.Join(testdata, "fused_gate.cubin"), ""
	t.Cleanup(func() { *cubin, *ptxFile = oldCubin, oldPTX })
	err := run(context.Background(), filepath.Join(testdata, "fused_gate.cu"), &bytes.Buffer{})
	if err == nil || !strings.Contains(err.Error(), "-ptx") {
		t.Errorf("err = %v, want a refusal naming -ptx", err)
	}
}
