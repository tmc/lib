package ncclinspector

import (
	"bytes"
	"context"
	"encoding/base64"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestIndependentFileFinishFreshness(t *testing.T) {
	if !fileSupported() {
		t.Skip("file adapter unsupported")
	}
	for _, mutation := range []string{"append", "rewrite"} {
		t.Run(mutation, func(t *testing.T) {
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
			if _, err = d.Next(); err != io.EOF {
				t.Fatal(err)
			}
			if err = d.Finish(); err != nil {
				t.Fatal(err)
			}
			if mutation == "append" {
				f, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0600)
				if err != nil {
					t.Fatal(err)
				}
				_, err = f.Write(b)
				if err != nil {
					t.Fatal(err)
				}
				if err = f.Close(); err != nil {
					t.Fatal(err)
				}
			} else {
				body := bytes.Replace(b, []byte(`"pid":`), []byte(`"pXd":`), 1)
				if bytes.Equal(body, b) {
					t.Fatal("fixture PID missing")
				}
				if err = os.WriteFile(path, body, 0600); err != nil {
					t.Fatal(err)
				}
			}
			if err = d.Finish(); err == nil {
				t.Fatal("stale completion accepted")
			}
		})
	}
}
func TestIndependentCrossProcessResume(t *testing.T) {
	if path := os.Getenv("INSPECTOR_OWN_RESTART_FILE"); path != "" {
		raw, err := base64.StdEncoding.DecodeString(os.Getenv("INSPECTOR_OWN_CURSOR"))
		if err != nil {
			t.Fatal(err)
		}
		var cursor FilePosition
		if err = cursor.UnmarshalText(raw); err != nil {
			t.Fatal(err)
		}
		d, err := OpenFile(path, Limits{Records: 2}, &cursor)
		if err != nil {
			t.Fatal(err)
		}
		defer d.Close()
		r, err := d.Next()
		if err != nil || !bytes.Equal(r.Raw, fixture(t, "stats")) {
			t.Fatalf("crossprocess resume %v", err)
		}
		if _, err = d.Next(); err != io.EOF {
			t.Fatal(err)
		}
		if err = d.Finish(); err != nil {
			t.Fatal(err)
		}
		return
	}
	if !fileSupported() {
		t.Skip("file adapter unsupported")
	}
	path := filepath.Join(t.TempDir(), "capture")
	b := fixture(t, "stats")
	if err := os.WriteFile(path, append(append([]byte{}, b...), b[:13]...), 0600); err != nil {
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
	if d.readOffset <= d.decoder.Offset() {
		t.Fatal("physical offset not ahead of committed pending prefix")
	}
	p, err := d.Position()
	if err != nil {
		t.Fatal(err)
	}
	raw, err := p.MarshalText()
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
	if _, err = f.Write(b[13:]); err != nil {
		t.Fatal(err)
	}
	if err = f.Close(); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestIndependentCrossProcessResume$", "-test.count=1")
	cmd.Env = append(os.Environ(), "INSPECTOR_OWN_RESTART_FILE="+path, "INSPECTOR_OWN_CURSOR="+base64.StdEncoding.EncodeToString(raw))
	if output, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("child %v %s", err, output)
	}
}
func TestIndependentCumulativeResumeLimits(t *testing.T) {
	if !fileSupported() {
		t.Skip("file adapter unsupported")
	}
	b := fixture(t, "stats")
	path := filepath.Join(t.TempDir(), "capture")
	if err := os.WriteFile(path, append(append([]byte{}, b...), b...), 0600); err != nil {
		t.Fatal(err)
	}
	l := Limits{Bytes: len(b)*2 - 1, Records: 2}
	d, err := OpenFile(path, l, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = d.Next(); err != nil {
		t.Fatal(err)
	}
	p, err := d.Position()
	if err != nil {
		t.Fatal(err)
	}
	if err = d.Close(); err != nil {
		t.Fatal(err)
	}
	d, err = OpenFile(path, l, &p)
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	if _, err = d.Next(); err == nil || err == io.EOF {
		t.Fatal("resume reset cumulative byte budget")
	}
}
func TestIndependentUnknownVersionAndTerminal(t *testing.T) {
	b := fixture(t, "stats")
	bad := bytes.Replace(b, []byte(`"v4.1"`), []byte(`"v99.0"`), 1)
	if bytes.Equal(b, bad) {
		t.Fatal("version fixture missing")
	}
	d, _ := NewDecoder(bytes.NewReader(bad), Limits{})
	if _, err := d.Next(); err == nil || err == io.EOF {
		t.Fatal("unknown version accepted")
	}
	if _, err := d.Next(); err == nil || err == io.EOF {
		t.Fatal("lost terminal refusal")
	}
	if d.Offset() != 0 {
		t.Fatal("invalid advanced offset")
	}
	d, _ = NewDecoder(strings.NewReader("{}\n"), Limits{})
	if _, err := d.Next(); err == nil {
		t.Fatal("malformed accepted")
	}
}
