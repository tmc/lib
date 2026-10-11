package cudaflightrecorder

import (
	"archive/tar"
	"bytes"
	"errors"
	"fmt"
	"io"
	"testing"
)

func TestReadBundleRefusesUnsafeEntries(t *testing.T) {
	for _, tt := range []struct {
		name   string
		header tar.Header
	}{
		{"traversal", tar.Header{Name: "../outside", Typeflag: tar.TypeReg}},
		{"absolute", tar.Header{Name: "/outside", Typeflag: tar.TypeReg}},
		{"symlink", tar.Header{Name: "link", Typeflag: tar.TypeSymlink, Linkname: "outside"}},
		{"duplicate", tar.Header{Name: "manifest.json", Typeflag: tar.TypeReg, Size: 2}},
		{"oversize", tar.Header{Name: "large", Typeflag: tar.TypeReg, Size: (256 << 20) + 1}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var b bytes.Buffer
			w := tar.NewWriter(&b)
			raw := []byte(`{}`)
			if err := w.WriteHeader(&tar.Header{Name: "manifest.json", Typeflag: tar.TypeReg, Size: int64(len(raw))}); err != nil {
				t.Fatal(err)
			}
			if _, err := w.Write(raw); err != nil {
				t.Fatal(err)
			}
			if err := w.WriteHeader(&tt.header); err != nil {
				t.Fatal(err)
			}
			if tt.name == "duplicate" {
				if _, err := w.Write([]byte(`{}`)); err != nil {
					t.Fatal(err)
				}
			}
			if tt.name != "oversize" {
				if err := w.Close(); err != nil {
					t.Fatal(err)
				}
			}
			if _, err := ReadBundle(bytes.NewReader(b.Bytes())); err == nil {
				t.Fatal("unsafe entry accepted")
			}
		})
	}
}

type footerFailureWriter struct{ remaining int }

func (w *footerFailureWriter) Write(p []byte) (int, error) {
	if len(p) > w.remaining {
		return 0, io.ErrClosedPipe
	}
	w.remaining -= len(p)
	return len(p), nil
}
func TestWriteBundleReturnsFooterFailure(t *testing.T) {
	b := NewBundle(Manifest{})
	var good bytes.Buffer
	if err := WriteBundle(&good, b); err != nil {
		t.Fatal(err)
	}
	// TAR Close adds two 512-byte end blocks, after the manifest and its padding.
	w := &footerFailureWriter{remaining: good.Len() - 1024}
	if err := WriteBundle(w, b); !errors.Is(err, io.ErrClosedPipe) {
		t.Fatalf("footer error lost: %v", err)
	}
}

func TestBundlePreflightAndNilIO(t *testing.T) {
	if _, err := ReadBundle(nil); err == nil {
		t.Fatal("nil reader accepted")
	}
	if err := WriteBundle(nil, NewBundle(Manifest{})); err == nil {
		t.Fatal("nil writer accepted")
	}
	for _, name := range []string{"../bad", "manifest.json"} {
		var w bytes.Buffer
		b := NewBundle(Manifest{})
		b.Files[name] = []byte("x")
		if err := WriteBundle(&w, b); err == nil || w.Len() != 0 {
			t.Fatalf("invalid bundle wrote %d bytes: %v", w.Len(), err)
		}
	}
	var w bytes.Buffer
	b := NewBundle(Manifest{})
	for i := 0; i < 1024; i++ {
		b.Files[fmt.Sprintf("f%d", i)] = nil
	}
	if err := WriteBundle(&w, b); err == nil || w.Len() != 0 {
		t.Fatalf("excessive entry count wrote %d bytes: %v", w.Len(), err)
	}
}
