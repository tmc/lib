package cudacapture_test

import (
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudacapture"
)

func TestRewriteSASSSources(t *testing.T) {
	m := cudacapture.SourceMap{
		"/tmp/build-2687102363/kernels.cu": "kernels.cu",
		"/a/b.cu":                          "kernels.cu",
		"/home/tmc/kernels.cu":             "kernels.cu",
	}
	tests := []struct {
		name, in, want string
	}{{
		name: "mapped build path is replaced",
		in:   "\t//## File \"/tmp/build-2687102363/kernels.cu\", line 12\n",
		want: "\t//## File \"kernels.cu\", line 12\n",
	}, {
		name: "already bundled name is unchanged",
		in:   "\t//## File \"kernels.cu\", line 3\n",
		want: "\t//## File \"kernels.cu\", line 3\n",
	}, {
		name: "multiple directives across lines",
		in:   "//## File \"/a/b.cu\", line 1\n\tMOV R1, R2 ;\n//## File \"/a/b.cu\", line 2\n",
		want: "//## File \"kernels.cu\", line 1\n\tMOV R1, R2 ;\n//## File \"kernels.cu\", line 2\n",
	}, {
		name: "unterminated quote is left alone",
		in:   "//## File \"/a/b.cu, line 1\n",
		want: "//## File \"/a/b.cu, line 1\n",
	}, {
		name: "instruction lines are untouched",
		in:   "\t/*0000*/ IMAD R2, R3, R4, RZ ;\n",
		want: "\t/*0000*/ IMAD R2, R3, R4, RZ ;\n",
	}, {
		name: "inlined directive rewrites both call site and inlined path",
		in:   "\t//## File \"/home/tmc/kernels.cu\", line 27 inlined at \"/home/tmc/kernels.cu\", line 98\n",
		want: "\t//## File \"kernels.cu\", line 27 inlined at \"kernels.cu\", line 98\n",
	}, {
		name: "ptx pseudo-file is preserved",
		in:   "\t//## File \".nv_debug_ptx_txt\", line 1695\n",
		want: "\t//## File \".nv_debug_ptx_txt\", line 1695\n",
	}, {
		name: "unmapped path is retained",
		in:   "\t//## File \"/usr/include/cuda_fp16.hpp\", line 40\n",
		want: "\t//## File \"/usr/include/cuda_fp16.hpp\", line 40\n",
	}, {
		name: "no trailing newline",
		in:   "//## File \"/a/b.cu\", line 9",
		want: "//## File \"kernels.cu\", line 9",
	}}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := string(cudacapture.RewriteSASSSources([]byte(tt.in), m)); got != tt.want {
				t.Errorf("RewriteSASSSources(%q) = %q, want %q", tt.in, got, tt.want)
			}
		})
	}
}

// Two directories can each hold a kernels.cu. Rewriting both to one artifact
// name merges two files' instructions, which is the defect this map replaces.
func TestRewriteSASSSourcesKeepsSameBaseNamesDistinct(t *testing.T) {
	const sass = "//## File \"/a/kernels.cu\", line 1\n" +
		"//## File \"/b/kernels.cu\", line 1\n"
	m := cudacapture.SourceMap{
		"/a/kernels.cu": "a/kernels.cu",
		"/b/kernels.cu": "b/kernels.cu",
	}
	want := "//## File \"a/kernels.cu\", line 1\n" +
		"//## File \"b/kernels.cu\", line 1\n"
	if got := string(cudacapture.RewriteSASSSources([]byte(sass), m)); got != want {
		t.Errorf("RewriteSASSSources = %q, want %q", got, want)
	}
}

func TestSASSSourcePaths(t *testing.T) {
	const sass = "//## File \"/a/kernels.cu\", line 1 inlined at \"/a/util.cuh\", line 7\n" +
		"\tMOV R1, R2 ;\n" +
		"//## File \"/a/kernels.cu\", line 2\n" +
		"//## File \".nv_debug_ptx_txt\", line 1695\n"
	want := []string{"/a/kernels.cu", "/a/util.cuh", ".nv_debug_ptx_txt"}
	if got := cudacapture.SASSSourcePaths([]byte(sass)); !slices.Equal(got, want) {
		t.Errorf("SASSSourcePaths = %q, want %q", got, want)
	}
}

func TestSourceMapByBase(t *testing.T) {
	const sass = "//## File \"/tmp/build/kernels.cu\", line 1\n" +
		"//## File \"/usr/include/cuda_fp16.hpp\", line 40\n"
	m, err := cudacapture.SourceMapByBase([]byte(sass), "kernels.cu")
	if err != nil {
		t.Fatalf("SourceMapByBase: %v", err)
	}
	if got, want := m["/tmp/build/kernels.cu"], "kernels.cu"; got != want {
		t.Errorf("map[%q] = %q, want %q", "/tmp/build/kernels.cu", got, want)
	}
	if _, ok := m["/usr/include/cuda_fp16.hpp"]; ok {
		t.Error("unbundled header was mapped; it must be retained instead")
	}
}

// An ambiguous base name has no right answer, so the map must not offer one.
func TestSourceMapByBaseRefusesAmbiguousBaseName(t *testing.T) {
	const sass = "//## File \"/a/kernels.cu\", line 1\n" +
		"//## File \"/b/kernels.cu\", line 1\n"
	m, err := cudacapture.SourceMapByBase([]byte(sass), "kernels.cu")
	if err == nil {
		t.Fatalf("SourceMapByBase = %v, want an error for two sources named kernels.cu", m)
	}
	if m != nil {
		t.Errorf("SourceMapByBase returned map %v beside its error", m)
	}
	for _, want := range []string{"/a/kernels.cu", "/b/kernels.cu"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q does not name the conflicting source %q", err, want)
		}
	}
}

// The recorded side and the artifact side are separate ambiguities. One
// recorded kernels.cu with two bundled files of that name is just as
// undecidable as two recorded sources, and indexing the artifacts by base name
// silently kept whichever came last.
func TestSourceMapByBaseRefusesAmbiguousArtifact(t *testing.T) {
	const sass = "//## File \"/build/kernels.cu\", line 1\n"
	m, err := cudacapture.SourceMapByBase([]byte(sass), "a/kernels.cu", "b/kernels.cu")
	if err == nil {
		t.Fatalf("SourceMapByBase = %v, want an error for two artifacts named kernels.cu", m)
	}
	if m != nil {
		t.Errorf("SourceMapByBase returned map %v beside its error", m)
	}
	for _, want := range []string{"a/kernels.cu", "b/kernels.cu"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q does not name the conflicting artifact %q", err, want)
		}
	}
}

// The artifact check must not reject a caller that names the same file twice,
// which is unambiguous.
func TestSourceMapByBaseAcceptsRepeatedArtifact(t *testing.T) {
	const sass = "//## File \"/build/kernels.cu\", line 1\n"
	m, err := cudacapture.SourceMapByBase([]byte(sass), "a/kernels.cu", "a/kernels.cu")
	if err != nil {
		t.Fatalf("SourceMapByBase: %v", err)
	}
	if got := m["/build/kernels.cu"]; got != "a/kernels.cu" {
		t.Errorf("map[/build/kernels.cu] = %q, want a/kernels.cu", got)
	}
}
