package metalkv

import (
	"context"
	"errors"
	"io"
	"os"
	"testing"
)

func TestSSDOffloaderCloseRetry(t *testing.T) {
	o, err := NewSSDOffloader(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer o.Close()
	path := o.dir + "/occupied"
	if err := os.Mkdir(path, 0700); err != nil {
		t.Fatal(err)
	}
	child := path + "/child"
	if err := os.WriteFile(child, []byte("occupied"), 0600); err != nil {
		t.Fatal(err)
	}
	o.files[0] = path
	if err := o.Close(); err == nil {
		t.Fatal("Close removed nonempty directory")
	}
	if o.files[0] != path {
		t.Fatal("Close forgot a file it failed to remove")
	}
	if err := os.Remove(child); err != nil {
		t.Fatal(err)
	}
	if err := o.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(path); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("retry did not remove path: %v", err)
	}
}

func TestSSDOffloaderCanceledOperations(t *testing.T) {
	o, err := NewSSDOffloader(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer o.Close()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	for _, operation := range []struct {
		name string
		run  func(context.Context, LayerCache) error
	}{{"evict", o.EvictLayer}, {"restore", o.RestoreLayer}} {
		t.Run(operation.name, func(t *testing.T) {
			if err := operation.run(ctx, LayerCache{}); !errors.Is(err, context.Canceled) {
				t.Fatalf("got %v, want canceled", err)
			}
		})
	}
}

type shortSpillWriter struct{}

func (shortSpillWriter) Write(p []byte) (int, error) { return len(p) - 1, nil }

func TestWriteFloatsShortWrite(t *testing.T) {
	if err := writeFloats(shortSpillWriter{}, []float32{1}); err != io.ErrShortWrite {
		t.Fatalf("got %v, want short write", err)
	}
}

func TestSpillFailedPublication(t *testing.T) {
	o, err := NewSSDOffloader(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer o.Close()
	// A directory cannot be replaced by the temporary regular file.
	old := o.dir + "/old"
	if err := os.Mkdir(old, 0700); err != nil {
		t.Fatal(err)
	}
	o.files[0] = old
	if err := o.spill(context.Background(), 0, []float32{1}, []float32{2}); err == nil {
		t.Fatal("replacement succeeded over a directory")
	}
	if o.files[0] != old {
		t.Fatal("failed publication changed the owned path")
	}
	entries, err := os.ReadDir(o.dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 {
		t.Fatalf("failed publication leaked files: %d", len(entries))
	}
}

func TestSpillOwnership(t *testing.T) {
	dir := t.TempDir()
	a, err := NewSSDOffloader(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	b, err := NewSSDOffloader(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()
	if err := a.spill(context.Background(), 0, []float32{1}, []float32{2}); err != nil {
		t.Fatal(err)
	}
	if err := b.spill(context.Background(), 0, []float32{3}, []float32{4}); err != nil {
		t.Fatal(err)
	}
	if a.files[0] == b.files[0] {
		t.Fatal("independent offloaders share a file")
	}
	if err := b.Close(); err != nil {
		t.Fatal(err)
	}
	before, err := os.ReadFile(a.files[0])
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := a.spill(ctx, 0, []float32{5}, []float32{6}); err != context.Canceled {
		t.Fatalf("canceled spill: %v", err)
	}
	after, err := os.ReadFile(a.files[0])
	if err != nil {
		t.Fatal(err)
	}
	if string(before) != string(after) {
		t.Fatal("canceled replacement changed previous spill")
	}
	if err := a.spill(context.Background(), 0, []float32{7}, []float32{8}); err != nil {
		t.Fatal(err)
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 {
		t.Fatalf("got %d files, want one", len(entries))
	}
}
