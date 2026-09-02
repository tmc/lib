package grammars

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"context"
	"encoding/hex"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/treesitter"
)

func TestManifest(t *testing.T) {
	all := All()
	if len(all) == 0 {
		t.Fatal("no grammars are pinned")
	}
	seen := make(map[string]bool)
	for _, g := range all {
		if g.Name == "" || g.Version == "" {
			t.Errorf("%v: missing a name or version", g)
		}
		if seen[g.Name] {
			t.Errorf("%s is pinned twice", g.Name)
		}
		seen[g.Name] = true
		if !strings.HasPrefix(g.URL, "https://") {
			t.Errorf("%s: URL %q is not https", g.Name, g.URL)
		}
		if _, err := hex.DecodeString(g.SHA256); err != nil || len(g.SHA256) != 64 {
			t.Errorf("%s: SHA256 %q is not a SHA-256 digest", g.Name, g.SHA256)
		}
	}
	if !slices.IsSortedFunc(all, func(a, b Grammar) int { return strings.Compare(a.Name, b.Name) }) {
		t.Error("pinned grammars are not sorted by name")
	}
	if _, ok := Lookup("go"); !ok {
		t.Error("no pin for go")
	}
	if _, ok := Lookup("no-such-grammar"); ok {
		t.Error("Lookup invented a grammar")
	}
}

// tarball builds a gzipped tar with the given path/content pairs, wrapped
// in one top-level directory the way a GitHub archive is.
func tarball(t *testing.T, files map[string]string) []byte {
	t.Helper()
	var buf bytes.Buffer
	zw := gzip.NewWriter(&buf)
	tw := tar.NewWriter(zw)
	for name, body := range files {
		h := &tar.Header{Name: name, Mode: 0o644, Size: int64(len(body)), Typeflag: tar.TypeReg}
		if err := tw.WriteHeader(h); err != nil {
			t.Fatal(err)
		}
		if _, err := tw.Write([]byte(body)); err != nil {
			t.Fatal(err)
		}
	}
	if err := tw.Close(); err != nil {
		t.Fatal(err)
	}
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func TestExtract(t *testing.T) {
	dir := t.TempDir()
	archive := tarball(t, map[string]string{
		"tree-sitter-x-1.0/src/parser.c":          "int main;",
		"tree-sitter-x-1.0/src/tree_sitter/x.h":   "#define X",
		"tree-sitter-x-1.0/README.md":             "hi",
		"tree-sitter-x-1.0/typescript/src/only.c": "void f(void);",
	})
	if err := extract(archive, dir); err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"src/parser.c",
		"src/tree_sitter/x.h",
		"README.md",
		"typescript/src/only.c",
	} {
		if _, err := os.Stat(filepath.Join(dir, filepath.FromSlash(want))); err != nil {
			t.Errorf("%s not extracted: %v", want, err)
		}
	}
}

func TestExtractRejectsEscape(t *testing.T) {
	for _, name := range []string{
		"tree-sitter-x-1.0/../../escaped.c",
		"/etc/passwd",
		"../../etc/passwd",
	} {
		dir := t.TempDir()
		err := extract(tarball(t, map[string]string{name: "x"}), dir)
		if err == nil {
			// A name that stays inside after the leading directory is
			// dropped is fine; one that climbs out is not.
			if _, serr := os.Stat(filepath.Join(dir, "..", "escaped.c")); serr == nil {
				t.Errorf("%s escaped the destination", name)
			}
			continue
		}
		if !strings.Contains(err.Error(), "outside the archive") {
			t.Errorf("%s: err = %v, want an escape error", name, err)
		}
	}
}

func TestBuildRejectsIncomplete(t *testing.T) {
	if _, err := (Grammar{Name: "x"}).Build(context.Background(), t.TempDir()); err == nil {
		t.Error("Build accepted a grammar with no URL or checksum")
	}
	if _, err := Install(context.Background(), "no-such-grammar"); err == nil {
		t.Error("Install accepted an unpinned name")
	}
}

// TestInstall builds a real grammar. It downloads and compiles, so it runs
// only when TREE_SITTER_INSTALL_TEST is set; see manual-gates.md.
func TestInstall(t *testing.T) {
	if os.Getenv("TREE_SITTER_INSTALL_TEST") == "" {
		t.Skip("set TREE_SITTER_INSTALL_TEST=1 to download and compile grammars")
	}
	if err := treesitter.Load(""); err != nil {
		t.Skipf("libtree-sitter unavailable: %v", err)
	}
	if _, err := exec.LookPath("cc"); err != nil {
		t.Skipf("no C compiler: %v", err)
	}

	dir := t.TempDir()
	for _, name := range []string{"json", "go", "typescript"} {
		t.Run(name, func(t *testing.T) {
			g, ok := Lookup(name)
			if !ok {
				t.Fatalf("no pin for %s", name)
			}
			path, err := g.Build(context.Background(), dir)
			if err != nil {
				t.Fatal(err)
			}
			lang, err := treesitter.LoadLanguage(path, name)
			if err != nil {
				t.Fatal(err)
			}
			if lang.SymbolCount() == 0 {
				t.Error("grammar has no symbols")
			}

			// A second Build reuses what is there.
			fi, err := os.Stat(path)
			if err != nil {
				t.Fatal(err)
			}
			again, err := g.Build(context.Background(), dir)
			if err != nil {
				t.Fatal(err)
			}
			fi2, err := os.Stat(again)
			if err != nil {
				t.Fatal(err)
			}
			if !fi2.ModTime().Equal(fi.ModTime()) {
				t.Error("Build rebuilt a grammar that was already installed")
			}
		})
	}

	// An installed grammar is found and used by the ordinary API.
	t.Setenv("TREE_SITTER_GRAMMAR_PATH", dir)
	lang, err := treesitter.FindLanguage("json")
	if err != nil {
		t.Fatal(err)
	}
	tree, err := treesitter.Parse(lang, []byte(`{"a": [1, true, null]}`))
	if err != nil {
		t.Fatal(err)
	}
	defer tree.Close()
	if tree.HasError() {
		t.Error("parse error")
	}
	if got := tree.RootNode().NamedChild(0).Type(); got != "object" {
		t.Errorf("root child = %q, want object", got)
	}

	// The checksum is checked before anything is built.
	bad, _ := Lookup("json")
	bad.SHA256 = strings.Repeat("00", 32)
	if _, err := bad.Build(context.Background(), t.TempDir()); err == nil {
		t.Error("Build accepted a mismatched checksum")
	} else if !strings.Contains(err.Error(), "checksum mismatch") {
		t.Errorf("err = %v, want a checksum mismatch", err)
	}
}
