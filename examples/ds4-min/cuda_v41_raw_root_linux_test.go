//go:build linux

package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

var v41RawRootChild = flag.String("v41-raw-root-child", "", "internal registered-cleanup child evidence path")

type v41RawRootEvidence struct {
	Retained, OwnerAlive, FileLive, Sticky bool
	Pending, Drains, Released              int
}

// This process intentionally exits 1 from the real registered cleanup callback.
// Its older cleanup observes global custody after that callback has finished.
func TestCUDAV41RawRegisteredCleanupChild(t *testing.T) {
	if *v41RawRootChild == "" {
		t.Skip("invoked by registered-cleanup host parent")
	}
	if !filepath.IsAbs(*v41RawRootChild) {
		t.Fatal("absolute child evidence path required")
	}
	f, err := os.OpenFile(*v41RawRootChild+".borrowed", os.O_CREATE|os.O_EXCL|os.O_RDWR, 0600)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.WriteString("borrowed model"); err != nil {
		t.Fatal(err)
	}
	marker := &struct{ file *os.File }{f}
	drains, released := 0, 0
	planted := errors.New("planted registered raw drain failure")
	c := &v41RawCleanup{owner: marker, drain: func() error { drains++; return planted }}
	c.releases = []func() error{func() error { released++; return nil }, f.Close}
	// LIFO: the actual callback registered below runs before this observer.
	t.Cleanup(func() {
		e := v41RawRootEvidence{OwnerAlive: c.owner == marker, Sticky: errors.Is(c.err, planted), Pending: len(c.releases), Drains: drains, Released: released}
		v41RawRetained.Lock()
		for _, held := range v41RawRetained.transactions {
			if held == c {
				e.Retained = true
			}
		}
		v41RawRetained.Unlock()
		_, err := f.Stat()
		e.FileLive = err == nil
		raw, err := json.MarshalIndent(e, "", "  ")
		if err != nil {
			t.Error(err)
			return
		}
		out, err := os.OpenFile(*v41RawRootChild, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
		if err != nil {
			t.Error(err)
			return
		}
		n, we := out.Write(raw)
		ce := out.Close()
		if we != nil || ce != nil || n != len(raw) {
			t.Error("root custody artifact write", we, ce, n)
		}
	})
	registerV41RawCleanup(t, c)
}

func TestCUDAV41RawRegisteredCleanupRetainsRoot(t *testing.T) {
	capture := newV41RawWorkspaceCapture(t)
	path := filepath.Join(capture.root, "registered-root.json")
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestCUDAV41RawRegisteredCleanupChild$", "-test.count=1", "-test.v", "-v41-raw-root-child="+path)
	stdout, err := cmd.CombinedOutput()
	capture.saveBytes(t, "child-output", stdout)
	var exit *exec.ExitError
	if ctx.Err() != nil || !errors.As(err, &exit) || exit.ExitCode() != 1 || !strings.Contains(string(stdout), "retaining CUDA inputs/module/context until process exit: attention test drain: planted registered raw drain failure") {
		t.Fatal("registered cleanup did not report intended failure", err, string(stdout))
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal("registered root artifact missing", err)
	}
	capture.saveBytes(t, "root-custody", raw)
	var e v41RawRootEvidence
	if err := json.Unmarshal(raw, &e); err != nil {
		t.Fatal(err)
	}
	if !e.Retained || !e.OwnerAlive || !e.FileLive || !e.Sticky || e.Pending != 2 || e.Drains != 1 || e.Released != 0 {
		t.Fatalf("registered cleanup lost global custody: %+v", e)
	}
}
