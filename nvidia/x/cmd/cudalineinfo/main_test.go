package main

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudalineinfo"
)

const testdata = "../../cudalineinfo/testdata"

// setFlags points the artifact flags at the fused_gate fixture, which was
// compiled from the annotated source beside it.
func setFlags(t *testing.T, f string) {
	t.Helper()
	oldSASS, oldPTX, oldFormat := *sassFile, *ptxFile, *format
	*sassFile = filepath.Join(testdata, "fused_gate.sass")
	*ptxFile = filepath.Join(testdata, "fused_gate.ptx")
	*format = f
	t.Cleanup(func() { *sassFile, *ptxFile, *format = oldSASS, oldPTX, oldFormat })
}

func TestFormats(t *testing.T) {
	for _, tt := range []struct{ format, want string }{
		{"table", "kernels.go:19"},
		{"listing", "FMUL"},
		{"rollup", "kernels.go:19"},
		{"digraph", "kernels.go:19 $__internal_0_$__cuda_sm3x_div_rn_noftz_f32_slowpath fused_gate\n"},
		{"dot", `"kernels.go:19" -> "fused_gate";`},
	} {
		setFlags(t, tt.format)
		var buf bytes.Buffer
		if err := run(context.Background(), filepath.Join(testdata, "fused_gate.cu"), &buf); err != nil {
			t.Fatalf("%s: %v", tt.format, err)
		}
		if !strings.Contains(buf.String(), tt.want) {
			t.Errorf("%s output lacks %q:\n%s", tt.format, tt.want, buf.String())
		}
	}
}

func TestUnitJSON(t *testing.T) {
	src, err := os.ReadFile(filepath.Join(testdata, "fused_gate.cu"))
	if err != nil {
		t.Fatal(err)
	}
	u := cudalineinfo.ParseUnit("fused_gate", string(src))
	for n, loc := range u.LineMap {
		loc.Function = "main.gate"
		loc.Callers = []cudair.Frame{{File: "main.go", Line: 9, Function: "main.main"}}
		u.LineMap[n] = loc
	}
	data, err := json.Marshal(u)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "fused_gate.json")
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
	setFlags(t, "digraph")
	var buf bytes.Buffer
	if err := run(context.Background(), path, &buf); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(buf.String(), "main.main main.gate\n") || !strings.Contains(buf.String(), "main.gate kernels.go:17 kernels.go:18 kernels.go:19\n") {
		t.Errorf("digraph lacks the program frames:\n%s", buf.String())
	}
}

func TestUnknownFormat(t *testing.T) {
	setFlags(t, "csv")
	if err := run(context.Background(), filepath.Join(testdata, "fused_gate.cu"), &bytes.Buffer{}); err == nil {
		t.Error("no error for an unknown format")
	}
}
