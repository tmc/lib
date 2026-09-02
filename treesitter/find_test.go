package treesitter

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestDetectLanguage(t *testing.T) {
	tests := []struct{ name, want string }{
		{"main.go", "go"},
		{"go.mod", "gomod"},
		{"/src/lib/parser.c", "c"},
		{"api.h", "c"},
		{"Model.swift", "swift"},
		{"Foo.CPP", "cpp"},
		{"Makefile", "make"},
		{"build/Dockerfile", "dockerfile"},
		{"Program.cs", "c_sharp"},
		{"README", ""},
		{"archive.tar.gz", ""},
		{"", ""},
	}
	for _, tt := range tests {
		if got := DetectLanguage(tt.name); got != tt.want {
			t.Errorf("DetectLanguage(%q) = %q, want %q", tt.name, got, tt.want)
		}
	}
}

func TestGrammarFileNames(t *testing.T) {
	names := grammarFileNames("c_sharp")
	want := []string{"c_sharp.so", "libtree-sitter-c_sharp.so", "c-sharp.so", "libtree-sitter-c-sharp.dylib"}
	for _, w := range want {
		found := false
		for _, n := range names {
			if n == w {
				found = true
			}
		}
		if !found {
			t.Errorf("grammarFileNames(c_sharp) lacks %s: %v", w, names)
		}
	}
}

func TestFindGrammar(t *testing.T) {
	dir := t.TempDir()
	for _, name := range []string{"c.so", "libtree-sitter-c-sharp.so"} {
		if err := os.WriteFile(filepath.Join(dir, name), nil, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	t.Setenv(GrammarPathEnv, dir+string(os.PathListSeparator)+filepath.Join(dir, "missing"))

	dirs := GrammarDirs()
	if len(dirs) == 0 || dirs[0] != dir {
		t.Fatalf("GrammarDirs() = %v, want %s first", dirs, dir)
	}
	for _, d := range dirs {
		if strings.HasSuffix(d, "missing") {
			t.Errorf("GrammarDirs() kept nonexistent %s", d)
		}
	}

	if got, err := FindGrammar("c"); err != nil || got != filepath.Join(dir, "c.so") {
		t.Errorf("FindGrammar(c) = %q, %v", got, err)
	}
	if got, err := FindGrammar("c_sharp"); err != nil || got != filepath.Join(dir, "libtree-sitter-c-sharp.so") {
		t.Errorf("FindGrammar(c_sharp) = %q, %v", got, err)
	}
	_, err := FindGrammar("nonexistent_language")
	var nf *GrammarNotFoundError
	if !errors.As(err, &nf) || nf.Name != "nonexistent_language" || len(nf.Dirs) == 0 {
		t.Errorf("FindGrammar(nonexistent) = %v, want GrammarNotFoundError", err)
	}
	if !strings.Contains(err.Error(), GrammarPathEnv) {
		t.Errorf("error %q does not mention %s", err, GrammarPathEnv)
	}
	if _, err := FindGrammar(""); err == nil {
		t.Error("FindGrammar(\"\") succeeded")
	}
}

func TestFindLanguage(t *testing.T) {
	lang := testGrammar(t)
	path, name := testGrammarSpec(t)
	if path == "" {
		t.Skip("grammar path unknown")
	}
	dir := t.TempDir()
	// Link the grammar under the file name FindGrammar expects.
	if err := os.Symlink(path, filepath.Join(dir, name+filepath.Ext(path))); err != nil {
		t.Skip(err)
	}
	t.Setenv(GrammarPathEnv, dir)
	foundLanguages.Delete(name)

	got, err := FindLanguage(name)
	if err != nil {
		t.Fatalf("FindLanguage(%s): %v", name, err)
	}
	if got.SymbolCount() != lang.SymbolCount() {
		t.Errorf("FindLanguage symbol count = %d, want %d", got.SymbolCount(), lang.SymbolCount())
	}
	again, err := FindLanguage(name)
	if err != nil || again != got {
		t.Errorf("second FindLanguage = %p, %v, want cached %p", again, err, got)
	}
	if _, err := FindLanguage("nonexistent_language"); err == nil {
		t.Error("FindLanguage(nonexistent) succeeded")
	}
}
