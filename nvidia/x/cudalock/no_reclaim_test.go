package cudalock

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestNoReclaimContended(t *testing.T) {
	for _, kind := range []string{"ownerless", "malformed", "dead", "foreign"} {
		t.Run(kind, func(t *testing.T) {
			dir := filepath.Join(t.TempDir(), "lock")
			if err := os.Mkdir(dir, 0755); err != nil {
				t.Fatal(err)
			}
			ownerPath := filepath.Join(dir, "owner")
			var raw []byte
			switch kind {
			case "malformed":
				raw = []byte("invalid owner\n")
			case "dead":
				raw = []byte(fmt.Sprintf("pid=99999999 host=%s label=retained epoch=1 at=2000-01-01T00:00:00Z\n", shortHostname()))
			case "foreign":
				raw = []byte("pid=99999999 host=foreign-host label=retained epoch=1 at=2000-01-01T00:00:00Z\n")
			}
			if raw != nil {
				if e := os.WriteFile(ownerPath, raw, 0644); e != nil {
					t.Fatal(e)
				}
				if e := os.Chtimes(ownerPath, time.Unix(1, 0), time.Unix(1, 0)); e != nil {
					t.Fatal(e)
				}
			}
			if e := os.Chtimes(dir, time.Unix(1, 0), time.Unix(1, 0)); e != nil {
				t.Fatal(e)
			}
			before, e := os.Stat(dir)
			if e != nil {
				t.Fatal(e)
			}
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
			defer cancel()
			h, e := Acquire(ctx, WithDir(dir), WithNoReclaim(), WithPollInterval(time.Millisecond))
			if h != nil || !errors.Is(e, context.DeadlineExceeded) {
				t.Fatalf("%v %v", h, e)
			}
			after, e := os.Stat(dir)
			if e != nil || !os.SameFile(before, after) {
				t.Fatal("directory replaced", e)
			}
			got, e := os.ReadFile(ownerPath)
			if raw == nil {
				if !errors.Is(e, os.ErrNotExist) {
					t.Fatal("ownerless mutated", e)
				}
			} else if e != nil || string(got) != string(raw) {
				t.Fatal("owner mutated", e)
			}
		})
	}
}

func TestNoReclaimConcurrentCreators(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "lock")
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	results := make(chan *Handle, 2)
	errs := make(chan error, 2)
	for i := 0; i < 2; i++ {
		go func() {
			h, e := Acquire(ctx, WithDir(dir), WithNoReclaim(), WithPollInterval(time.Millisecond))
			results <- h
			errs <- e
		}()
	}
	first := <-results
	if e := <-errs; e != nil {
		t.Fatal(e)
	}
	if first == nil {
		t.Fatal("no handle")
	}
	// Concurrent process-local acquisition may share the verified canonical hold.
	// Both paths must retain one real owner, without replacing the directory.
	before, e := os.Stat(dir)
	if e != nil {
		t.Fatal(e)
	}
	if e = first.Release(); e != nil {
		t.Fatal(e)
	}
	second := <-results
	if e = <-errs; e != nil {
		t.Fatal(e)
	}
	if second == nil {
		t.Fatal("no second handle")
	}
	env := func(k string) string {
		if k == "GPU_LOCK_HELD" {
			return fmt.Sprint(second.Owner().PID)
		}
		if k == "GPU_LOCKDIR" {
			return dir
		}
		return os.Getenv(k)
	}
	if _, e = Inherited(env); e != nil {
		t.Fatal(e)
	}
	after, e := os.Stat(dir)
	if e != nil {
		t.Fatal(e)
	}
	if second.IsNested() && !os.SameFile(before, after) {
		t.Fatal("nested directory replaced")
	}
	if e = second.Release(); e != nil {
		t.Fatal(e)
	}
}
