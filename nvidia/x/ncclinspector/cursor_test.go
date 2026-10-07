package ncclinspector

import (
	"bytes"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestPersistentCursorReopen(t *testing.T) {
	if !fileSupported() {
		t.Skip("file adapter unsupported")
	}
	path := filepath.Join(t.TempDir(), "capture")
	b := fixture(t, "stats")
	if err := os.WriteFile(path, append(append([]byte{}, b...), b[:8]...), 0600); err != nil {
		t.Fatal(err)
	}
	d, err := OpenFile(path, Limits{Records: 2}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = d.Next(); err != nil {
		t.Fatal(err)
	}
	if _, err = d.Next(); err != io.EOF {
		t.Fatal(err)
	}
	position, err := d.Position()
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := position.MarshalText()
	if err != nil {
		t.Fatal(err)
	}
	if err = d.Close(); err != nil {
		t.Fatal(err)
	}
	var restored FilePosition
	if err = restored.UnmarshalText(encoded); err != nil {
		t.Fatal(err)
	}
	if restored.Offset() != int64(len(b)) {
		t.Fatal("lost committed offset")
	}
	f, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = f.Write(b[8:]); err != nil {
		t.Fatal(err)
	}
	if err = f.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err = OpenFile(path, Limits{}, &restored); err == nil {
		t.Fatal("changed epoch limits accepted")
	}
	d, err = OpenFile(path, Limits{Records: 2}, &restored)
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	r, err := d.Next()
	if err != nil || !bytes.Equal(r.Raw, b) {
		t.Fatalf("resume: %v", err)
	}
	if err = os.Rename(path, path+".old"); err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(path, append(append([]byte{}, b...), b...), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err = OpenFile(path, Limits{Records: 2}, &restored); err == nil {
		t.Fatal("replacement accepted")
	}
}

func TestCursorMalformed(t *testing.T) {
	p := FilePosition{device: 1, inode: 2, epoch: strings.Repeat("a", 32), offset: 4, high: 8, records: 1, limits: Limits{Bytes: 100, Line: 10, Records: 3}}
	good, err := p.MarshalText()
	if err != nil {
		t.Fatal(err)
	}
	for _, bad := range []string{
		"", "{}", string(good) + "{}", strings.Repeat("x", 4097),
		strings.Replace(string(good), `"version":1`, `"version":2`, 1),
		strings.Replace(string(good), `"inode":2`, `"inode":0`, 1),
		strings.Replace(string(good), `"offset":4`, `"offset":-1`, 1),
		strings.Replace(string(good), `"committed_bytes":4`, `"committed_bytes":5`, 1),
		strings.Replace(string(good), `"version":1`, `"version":1,"version":1`, 1),
		strings.Replace(string(good), `"device":1`, `"device":null`, 1),
		strings.Replace(string(good), `"high_water":8`, `"high_water":3`, 1),
		strings.Replace(string(good), `"records":1`, `"records":4`, 1),
		strings.Replace(string(good), `"version":1`, `"unknown":1,"version":1`, 1),
	} {
		q := p
		if err = q.UnmarshalText([]byte(bad)); err == nil {
			t.Fatalf("accepted %s", bad)
		}
		if q.offset != p.offset || q.inode != p.inode {
			t.Fatal("changed receiver on error")
		}
	}
}
