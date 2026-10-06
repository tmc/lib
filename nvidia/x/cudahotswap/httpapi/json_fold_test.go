package httpapi

import (
	"strings"
	"testing"
	"unicode"
	"unicode/utf8"
)

func TestJSONNameFolding(t *testing.T) {
	for r := rune(0); r <= unicode.MaxRune; r++ {
		if !utf8.ValidRune(r) {
			continue
		}
		name, alias := string(r), string(unicode.SimpleFold(r))
		if !strings.EqualFold(name, alias) || foldJSONName(name) != foldJSONName(alias) {
			t.Fatalf("folding differs at rune %U", r)
		}
	}
}

func FuzzJSONNameFolding(f *testing.F) {
	for _, pair := range [][2]string{{"name", "NAME"}, {"source", "ſource"}, {"kernel", "Kernel"}, {"i", "ı"}, {"Σ", "ς"}, {"", ""}} {
		f.Add(pair[0], pair[1])
	}
	f.Fuzz(func(t *testing.T, name, alias string) {
		if len(name) > 128 || len(alias) > 128 || !utf8.ValidString(name) || !utf8.ValidString(alias) {
			t.Skip()
		}
		if (foldJSONName(name) == foldJSONName(alias)) != strings.EqualFold(name, alias) {
			t.Fatalf("folding differs for %q and %q", name, alias)
		}
	})
}
