package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestExternalOutput(t *testing.T) {
	root := t.TempDir()
	git := filepath.Join(root, "other-tree")
	if err := os.Mkdir(git, 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(git, ".git"), []byte("gitdir: elsewhere"), 0600); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(root, "link")
	if err := os.Symlink(git, link); err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{filepath.Join(git, "generated"), filepath.Join(link, "generated")} {
		if err := externalOutput(path); err == nil {
			t.Fatalf("accepted worktree output %s", path)
		}
	}
	if err := externalOutput(filepath.Join(root, "external", "generated")); err != nil {
		t.Fatal(err)
	}
}

func TestGenerationRequiresExplicitPaths(t *testing.T) {
	t.Setenv("DS4_ORACLE_SOURCE", "")
	t.Setenv("DS4_EVAL_CORPUS_OUT", "")
	if err := generate(); err == nil {
		t.Fatal("implicit corpus generation accepted")
	}
}

func TestCorpusRemainsExternal(t *testing.T) {
	for _, name := range []string{"hard-smoke.json", "Apache-2.0.txt"} {
		_, err := os.Lstat(filepath.Join("..", "..", "testdata", name))
		if !os.IsNotExist(err) {
			t.Fatalf("third-party corpus resource %s must remain external: %v", name, err)
		}
	}
}
