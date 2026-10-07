package ncclinspector

import (
	"bytes"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func TestDecoderAppend(t *testing.T) {
	b := fixture(t, "collective")
	var input bytes.Buffer
	input.Write(b[:len(b)-1])
	d, err := NewDecoder(&input, Limits{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := d.Next(); err != io.EOF || d.Offset() != 0 {
		t.Fatalf("pending: %v offset %d", err, d.Offset())
	}
	if d.Finish() == nil {
		t.Fatal("accepted partial epoch")
	}
	input.WriteByte('\n')
	r, err := d.Next()
	if err != nil || !bytes.Equal(r.Raw, b) || d.Offset() != int64(len(b)) {
		t.Fatalf("append: %v", err)
	}
	input.Write(p2pFixture(t))
	r, err = d.Next()
	if err != nil || r.PointToPoint == nil || r.Collective != nil {
		t.Fatalf("p2p: %v", err)
	}
	if _, err = d.Next(); err != io.EOF {
		t.Fatal(err)
	}
	if err = d.Finish(); err != nil {
		t.Fatal(err)
	}
}

func TestDecoderLimits(t *testing.T) {
	for _, tc := range []struct {
		name   string
		limits Limits
		body   []byte
	}{
		{"partial_line", Limits{Line: 3}, []byte("1234")},
		{"bytes", Limits{Bytes: 3}, []byte("1234")},
		{"malformed", Limits{}, []byte("{}\n")},
		{"records", Limits{Records: 1}, append(fixture(t, "stats"), fixture(t, "stats")...)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			d, err := NewDecoder(bytes.NewReader(tc.body), tc.limits)
			if err != nil {
				t.Fatal(err)
			}
			if tc.name == "records" {
				if _, err = d.Next(); err != nil {
					t.Fatal(err)
				}
			}
			if _, err = d.Next(); err == nil || err == io.EOF {
				t.Fatalf("accepted limit: %v", err)
			}
			if _, err = d.Next(); err == nil || err == io.EOF {
				t.Fatalf("lost terminal error: %v", err)
			}
		})
	}
}

func TestFileResume(t *testing.T) {
	if !fileSupported() {
		t.Skip("file adapter unsupported")
	}
	path := filepath.Join(t.TempDir(), "capture")
	b := fixture(t, "stats")
	if err := os.WriteFile(path, append(append([]byte{}, b...), b[:10]...), 0600); err != nil {
		t.Fatal(err)
	}
	d, err := OpenFile(path, Limits{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = d.Next(); err != nil {
		t.Fatal(err)
	}
	if _, err = d.Next(); err != io.EOF {
		t.Fatal(err)
	}
	pos, err := d.Position()
	if err != nil {
		t.Fatal(err)
	}
	if err = d.Close(); err != nil {
		t.Fatal(err)
	}
	f, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		t.Fatal(err)
	}
	_, err = f.Write(b[10:])
	if err != nil {
		t.Fatal(err)
	}
	if err = f.Close(); err != nil {
		t.Fatal(err)
	}
	d, err = OpenFile(path, Limits{}, &pos)
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	r, err := d.Next()
	if err != nil || !bytes.Equal(r.Raw, b) || d.decoder.records != 2 {
		t.Fatalf("resume %v", err)
	}
	if err = os.Truncate(path, 1); err != nil {
		t.Fatal(err)
	}
	if _, err = d.Next(); err == nil {
		t.Fatal("accepted truncation")
	}
	if _, err = OpenFile(path, Limits{}, &pos); err == nil {
		t.Fatal("resumed truncated epoch")
	}
}

func TestFileReplacement(t *testing.T) {
	if !fileSupported() {
		t.Skip("file adapter unsupported")
	}
	path := filepath.Join(t.TempDir(), "capture")
	b := fixture(t, "stats")
	if err := os.WriteFile(path, b, 0600); err != nil {
		t.Fatal(err)
	}
	d, err := OpenFile(path, Limits{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	if _, err = d.Next(); err != nil {
		t.Fatal(err)
	}
	pos, err := d.Position()
	if err != nil {
		t.Fatal(err)
	}
	if err = os.Rename(path, path+".old"); err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(path, b, 0600); err != nil {
		t.Fatal(err)
	}
	if _, err = d.Next(); err == nil {
		t.Fatal("accepted replacement")
	}
	if _, err = OpenFile(path, Limits{}, &pos); err == nil {
		t.Fatal("resumed replacement")
	}
}

func fileSupported() bool { return runtime.GOOS == "linux" || runtime.GOOS == "darwin" }
