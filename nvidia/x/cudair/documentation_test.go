package cudair_test

import (
	"bytes"
	"go/format"
	"os"
	"strings"
	"testing"
)

func TestEmissionDocumentationFormatting(t *testing.T) {
	for _, name := range []string{"doc.go", "devicefunc.go"} {
		t.Run(name, func(t *testing.T) {
			source, err := os.ReadFile(name)
			if err != nil {
				t.Fatal(err)
			}
			formatted, err := format.Source(source)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(source, formatted) {
				t.Fatal("emission documentation is not gofmt clean")
			}
			for i, line := range strings.Split(string(source), "\n") {
				if strings.HasPrefix(strings.TrimSpace(line), "//") && len(line) > 100 {
					t.Errorf("comment line %d has %d columns, want at most 100", i+1, len(line))
				}
			}
		})
	}
}
