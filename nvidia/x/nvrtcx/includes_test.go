package nvrtcx

import (
	"slices"
	"strings"
	"testing"
)

func TestScanIncludes(t *testing.T) {
	for _, tt := range []struct {
		name string
		src  string
		want []string
	}{
		{"empty", "", nil},
		{"directives", " #include \"a.h\"\r\n#include <b.h> // comment\n#include \"a.h\"", []string{"a.h", "b.h", "a.h"}},
		{"conditional", "#if 0\n#include <disabled.h>\n#endif", []string{"disabled.h"}},
		{"malformed", "#include MACRO\n#include <>\n#include \"\n#include <missing", nil},
		{"long line", strings.Repeat("x", 2<<20) + "\n#include <after.h>\n", []string{"after.h"}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if got := scanIncludes(tt.src); !slices.Equal(got, tt.want) {
				t.Fatalf("scanIncludes: got %q, want %q", got, tt.want)
			}
		})
	}
}

func BenchmarkScanIncludes(b *testing.B) {
	src := "#include <math.h>\n" + strings.Repeat("float x = 1.0f;\n", 200) + "#include \"local.h\"\n"
	b.ReportAllocs()
	for b.Loop() {
		if len(scanIncludes(src)) != 2 {
			b.Fatal("missing headers")
		}
	}
}
