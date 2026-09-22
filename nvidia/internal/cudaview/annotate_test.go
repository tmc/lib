package cudaview

import (
	"strings"
	"testing"
)

func TestSourceIdentity(t *testing.T) {
	for _, tt := range []struct{ name, source, want string }{
		{"first", ".file 1 \"first.cu\"", "first.cu"},
		{"second", ".file 1 \"second.cu\"", "second.cu"},
		{"build-host path", ".file 1 \"/home/builder/src/third.cu\"", "third.cu"},
		{"missing", "", ""},
	} {
		t.Run(tt.name, func(t *testing.T) {
			text := ".version 8.0\n.target sm_80\n.address_size 64\n" + tt.source + "\n.visible .entry k() {\n.loc 1 7 0\nret;\n}\n"
			tier := PipelineTier{Language: "ptx"}
			for i, line := range strings.Split(text, "\n") {
				tier.Lines = append(tier.Lines, PipelineLine{Num: i + 1, Text: line})
			}
			Annotate(&tier)
			for _, line := range tier.Lines {
				if line.Text == "ret;" && line.SourceFile != tt.want {
					t.Fatalf("source = %q, want %q", line.SourceFile, tt.want)
				}
			}
		})
	}
}
