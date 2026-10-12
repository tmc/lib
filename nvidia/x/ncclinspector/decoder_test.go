package ncclinspector

import (
	"bytes"
	"errors"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
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
	if _, err = d.Next(); !errors.Is(err, ErrFileTruncated) {
		t.Fatalf("truncation: %v", err)
	}
	if _, err = OpenFile(path, Limits{}, &pos); !errors.Is(err, ErrFileTruncated) {
		t.Fatalf("resumed truncated epoch: %v", err)
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
	if _, err = d.Next(); !errors.Is(err, ErrFileReplaced) {
		t.Fatalf("replacement: %v", err)
	}
	if _, err = OpenFile(path, Limits{}, &pos); !errors.Is(err, ErrFileReplaced) {
		t.Fatalf("resumed replacement: %v", err)
	}
}

// TestFileEpochErrors checks that replacement and truncation are reported as
// themselves, never as the temporary io.EOF a caller would retry.
func TestFileEpochErrors(t *testing.T) {
	if !fileSupported() {
		t.Skip("file adapter unsupported")
	}
	b := fixture(t, "stats")
	appendBytes := func(t *testing.T, path string, p []byte) {
		f, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0600)
		if err != nil {
			t.Fatal(err)
		}
		if _, err = f.Write(p); err != nil {
			t.Fatal(err)
		}
		if err = f.Close(); err != nil {
			t.Fatal(err)
		}
	}
	for _, tc := range []struct {
		name   string
		change func(t *testing.T, path string)
		want   error // nil means the appended record is returned
		resume error // nil means OpenFile refuses the old position for another reason
	}{
		{"append", func(t *testing.T, path string) { appendBytes(t, path, b[10:]) }, nil, nil},
		{"replace", func(t *testing.T, path string) {
			if err := os.Rename(path, path+".old"); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(path, append(append([]byte{}, b...), b...), 0600); err != nil {
				t.Fatal(err)
			}
		}, ErrFileReplaced, ErrFileReplaced},
		{"replace_symlink", func(t *testing.T, path string) {
			if err := os.Rename(path, path+".old"); err != nil {
				t.Fatal(err)
			}
			if err := os.Symlink(path+".old", path); err != nil {
				t.Fatal(err)
			}
		}, ErrFileReplaced, nil}, // O_NOFOLLOW refuses to open the symlink
		{"truncate_pending", func(t *testing.T, path string) {
			if err := os.Truncate(path, int64(len(b))); err != nil {
				t.Fatal(err)
			}
		}, ErrFileTruncated, ErrFileTruncated},
		{"truncate_committed", func(t *testing.T, path string) {
			if err := os.Truncate(path, 1); err != nil {
				t.Fatal(err)
			}
		}, ErrFileTruncated, ErrFileTruncated},
		{"truncate_and_regrow", func(t *testing.T, path string) {
			if err := os.Truncate(path, 1); err != nil {
				t.Fatal(err)
			}
			appendBytes(t, path, b[1:])
		}, ErrFileTruncated, ErrFileTruncated},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "capture")
			if err := os.WriteFile(path, append(append([]byte{}, b...), b[:10]...), 0600); err != nil {
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
			if _, err = d.Next(); err != io.EOF {
				t.Fatalf("pending bytes: %v", err)
			}
			pos, err := d.Position()
			if err != nil {
				t.Fatal(err)
			}
			tc.change(t, path)
			r, err := d.Next()
			if tc.want == nil {
				if err != nil || !bytes.Equal(r.Raw, b) {
					t.Fatalf("append: %v", err)
				}
				return
			}
			if !errors.Is(err, tc.want) || errors.Is(err, io.EOF) {
				t.Fatalf("Next = %v, want %v", err, tc.want)
			}
			if _, err = d.Position(); !errors.Is(err, tc.want) {
				t.Fatalf("Position = %v, want %v", err, tc.want)
			}
			r2, err := OpenFile(path, Limits{}, &pos)
			if err == nil {
				r2.Close()
			}
			if err == nil || tc.resume != nil && !errors.Is(err, tc.resume) {
				t.Fatalf("resume = %v, want %v", err, tc.resume)
			}
		})
	}
}

func TestFilePrefixShortRead(t *testing.T) {
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
	// A file shorter than the committed prefix was truncated, even when the
	// size check raced with the truncation; it is not a temporary io.EOF.
	if _, err = d.prefix(int64(len(b)) + 1); !errors.Is(err, ErrFileTruncated) || errors.Is(err, io.EOF) {
		t.Fatalf("short prefix: %v", err)
	}
}

// rewriteAt overwrites len(p) bytes at off without changing the file size.
func rewriteAt(t *testing.T, path string, off int64, p []byte) {
	t.Helper()
	f, err := os.OpenFile(path, os.O_WRONLY, 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = f.WriteAt(p, off); err != nil {
		t.Fatal(err)
	}
	if err = f.Close(); err != nil {
		t.Fatal(err)
	}
}

// samePID returns rec with its pid changed without changing its length.
func samePID(t *testing.T, rec []byte) []byte {
	t.Helper()
	out := bytes.Replace(rec, []byte(`"pid":1639453`), []byte(`"pid":1639454`), 1)
	if bytes.Equal(out, rec) || len(out) != len(rec) {
		t.Fatal("fixture pid missing")
	}
	return out
}

// TestFilePrefixFreshness checks that a same-size rewrite of committed bytes
// is refused by the next Next, by resume, and by Position after the decoder
// returned bytes that no longer match the file.
func TestFilePrefixFreshness(t *testing.T) {
	if !fileSupported() {
		t.Skip("file adapter unsupported")
	}
	b := fixture(t, "stats")
	changed := func(err error) bool {
		return err != nil && err != io.EOF && strings.Contains(err.Error(), "changed file prefix")
	}
	open := func(t *testing.T, n int) (string, *FileDecoder) {
		path := filepath.Join(t.TempDir(), "capture")
		if err := os.WriteFile(path, bytes.Repeat(b, n), 0600); err != nil {
			t.Fatal(err)
		}
		d, err := OpenFile(path, Limits{}, nil)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { d.Close() })
		if _, err = d.Next(); err != nil {
			t.Fatal(err)
		}
		return path, d
	}
	t.Run("next", func(t *testing.T) {
		path, d := open(t, 3)
		rewriteAt(t, path, 0, samePID(t, b))
		if _, err := d.Next(); !changed(err) {
			t.Fatalf("Next after committed rewrite = %v", err)
		}
	})
	t.Run("resume", func(t *testing.T) {
		path, d := open(t, 2)
		pos, err := d.Position()
		if err != nil {
			t.Fatal(err)
		}
		if err = d.Close(); err != nil {
			t.Fatal(err)
		}
		rewriteAt(t, path, 0, samePID(t, b))
		r, err := OpenFile(path, Limits{}, &pos)
		if err == nil {
			r.Close()
		}
		if !changed(err) {
			t.Fatalf("resume after committed rewrite = %v", err)
		}
	})
	t.Run("parsed_versus_disk", func(t *testing.T) {
		// Both records are buffered by the first Next. Rewriting the second on
		// disk does not change what the decoder parses, so the digest must bind
		// the returned Raw and the next freshness check must see the disk differ.
		path, d := open(t, 2)
		if n := d.decoder.reader.Buffered(); n != len(b) {
			t.Fatalf("buffered %d bytes, want %d", n, len(b))
		}
		rewriteAt(t, path, int64(len(b)), samePID(t, b))
		r, err := d.Next()
		if err != nil || !bytes.Equal(r.Raw, b) {
			t.Fatalf("buffered record: %v", err)
		}
		if _, err = d.Position(); !changed(err) {
			t.Fatalf("Position after parse-versus-disk rewrite = %v", err)
		}
		if _, err = d.Next(); !changed(err) {
			t.Fatalf("Next after parse-versus-disk rewrite = %v", err)
		}
	})
}

func fileSupported() bool { return runtime.GOOS == "linux" || runtime.GOOS == "darwin" }
