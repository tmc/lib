package cudafuse

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudair"
)

func TestSourceWithLineDirectives(t *testing.T) {
	at := func(file string, line int) cudair.SourceLocation {
		return cudair.SourceLocation{File: file, Line: line}
	}
	tests := []struct {
		name string
		fk   FusedKernel
		want string
	}{
		{
			name: "no map leaves the source alone",
			fk:   FusedKernel{Source: "a\nb"},
			want: "a\nb",
		},
		{
			name: "annotated line is attributed and then restored",
			fk: FusedKernel{
				Source:  "a\nb\nc",
				LineMap: map[int]cudair.SourceLocation{2: at("m.go", 7)},
			},
			want: "a\n#line 7 \"m.go\"\nb\n#line 3 \"k.cu\"\nc",
		},
		{
			name: "trailing annotation needs no restore",
			fk: FusedKernel{
				Source:  "a\nb",
				LineMap: map[int]cudair.SourceLocation{2: at("m.go", 7)},
			},
			want: "a\n#line 7 \"m.go\"\nb",
		},
		{
			name: "a location without a file is not a directive",
			fk: FusedKernel{
				Source:  "a\nb",
				LineMap: map[int]cudair.SourceLocation{2: at("", 7)},
			},
			want: "a\nb",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.fk.SourceWithLineDirectives("k.cu")
			if got != tt.want {
				t.Errorf("got:\n%s\nwant:\n%s", got, tt.want)
			}
		})
	}
}

// TestFuseEmitsGoCoordinates checks the directives Fuse produces name the Go
// file the graph was built in, not the generated kernel.
func TestFuseEmitsGoCoordinates(t *testing.T) {
	g := cudair.NewGraph()
	x := g.AddInput("x", []int{8}, cudair.Float32)
	x.SourceLoc = cudair.SourceLocation{File: "model.go", Line: 11}
	y := g.AddUnary("relu", x)
	y.SourceLoc = cudair.SourceLocation{File: "model.go", Line: 12}

	src, err := Fuse(g, "k")
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{`#line 11 "model.go"`, `#line 12 "model.go"`, `"k.cu"`} {
		if !strings.Contains(src, want) {
			t.Errorf("Fuse output missing %s:\n%s", want, src)
		}
	}
}
