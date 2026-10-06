//go:build native

package testvec

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

func TestWitnessReceiptDirectory(t *testing.T) {
	dir, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if err := witnessDirectory(dir); err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{"", ".", "relative", filepath.Join(dir, "missing")} {
		if err := witnessDirectory(path); err == nil {
			t.Fatalf("accepted %q", path)
		}
	}
	file := filepath.Join(dir, "ordinary")
	if err := os.WriteFile(file, []byte("body"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := witnessDirectory(file); err == nil {
		t.Fatal("accepted regular file as directory")
	}
	link := filepath.Join(dir, "link")
	if err := os.Symlink(dir, link); err != nil {
		t.Fatal(err)
	}
	if err := witnessDirectory(link); err == nil {
		t.Fatal("accepted linked directory")
	}
}

func TestWitnessReceiptExclusiveAndBounded(t *testing.T) {
	path := filepath.Join(t.TempDir(), "body")
	if err := witnessBody(path, []byte("original")); err != nil {
		t.Fatal(err)
	}
	if err := witnessBody(path, []byte("replacement")); err == nil {
		t.Fatal("overwrote prior receipt")
	}
	got, err := os.ReadFile(path)
	if err != nil || string(got) != "original" {
		t.Fatalf("body %q: %v", got, err)
	}
	for _, body := range [][]byte{nil, make([]byte, 8<<20+1)} {
		path := filepath.Join(t.TempDir(), "refused")
		if err := witnessBody(path, body); err == nil {
			t.Fatal("accepted body outside bound")
		}
		if _, err := os.Stat(path); !os.IsNotExist(err) {
			t.Fatalf("refused body created file: %v", err)
		}
	}
	path = filepath.Join(t.TempDir(), "json")
	if err := witnessWrite(path, strings.Repeat("x", 1<<20)); err == nil {
		t.Fatal("accepted oversized JSON")
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("refused JSON created file: %v", err)
	}
	if err := witnessWrite(path, make(chan int)); err == nil {
		t.Fatal("ignored serialization error")
	}
}

func TestWitnessReceiptSelectedImageRefusesUnavailable(t *testing.T) {
	for _, kind := range []nvrtcx.OutputKind{nvrtcx.OutputCUBIN, nvrtcx.OutputPTX, nvrtcx.OutputAuto, nvrtcx.OutputTileIR} {
		for _, cm := range []*nvrtcx.CompiledModule{nil, {}} {
			image, name, err := witnessImage(cm, kind)
			if err == nil || len(image) != 0 || name != "" {
				t.Fatalf("kind %v: image=%x name=%q error=%v", kind, image, name, err)
			}
		}
	}
}

func TestWitnessReceiptDirectoryLexicalAlias(t *testing.T) {
	root, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"target", "leaf"} {
		if err := os.Mkdir(filepath.Join(root, name), 0700); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.Mkdir(filepath.Join(root, "target", "child"), 0700); err != nil {
		t.Fatal(err)
	}
	alias := filepath.Join(root, "alias")
	if err := os.Symlink(filepath.Join(root, "target"), alias); err != nil {
		t.Fatal(err)
	}
	// Join would erase the attack before the filesystem observes it.
	path := alias + string(os.PathSeparator) + ".." + string(os.PathSeparator) + "leaf"
	if st, err := os.Stat(path); err != nil || !st.IsDir() {
		t.Fatalf("counterexample: %v %v", st, err)
	}
	if err := witnessDirectory(path); err == nil {
		t.Fatal("accepted symlink ancestor concealed by dot-dot")
	}
	if err := witnessDirectory(filepath.Join(root, "leaf")); err != nil {
		t.Fatal(err)
	}
	if err := witnessDirectory(alias + string(os.PathSeparator) + "child"); err == nil {
		t.Fatal("accepted real symlink ancestor")
	}
}
