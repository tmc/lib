package fswatch

import (
	"context"
	"crypto/sha256"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"
)

func content(s string) stamp { return stamp{sha256.Sum256([]byte(s)), true} }

func TestLatestContent(t *testing.T) {
	start := time.Unix(100, 0)
	for _, tt := range []struct {
		name string
		last map[string]stamp
		want []Change
	}{
		{"latest", map[string]stamp{"kernel.cu": content("last")}, []Change{{"kernel.cu", content("last").digest, false}}},
		{"reverted", map[string]stamp{"kernel.cu": content("original")}, nil},
		{"deleted", map[string]stamp{}, []Change{{"kernel.cu", [32]byte{}, true}}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			s := newState(map[string]stamp{"kernel.cu": content("original")})
			s.observe(map[string]stamp{"kernel.cu": content("intermediate")}, start, 100*time.Millisecond)
			s.observe(tt.last, start.Add(50*time.Millisecond), 100*time.Millisecond)
			if got := s.ready(start.Add(100 * time.Millisecond)); len(got) != 0 {
				t.Fatalf("early event: %+v", got)
			}
			if got := s.ready(start.Add(150 * time.Millisecond)); !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("events = %+v, want %+v", got, tt.want)
			}
			if len(s.ready(start.Add(time.Second))) != 0 {
				t.Fatal("duplicate delivery")
			}
		})
	}
}

func TestRenameAndContentHash(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "kernel.cu")
	write := func(path, data string) {
		t.Helper()
		if err := os.WriteFile(path, []byte(data), 0600); err != nil {
			t.Fatal(err)
		}
	}
	read := func() map[string]stamp {
		t.Helper()
		m, err := scan([]string{dir})
		if err != nil {
			t.Fatal(err)
		}
		return m
	}
	write(path, "old")
	s := newState(read())
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	// An mtime-only edit does not rebuild.
	if err := os.Chtimes(path, info.ModTime().Add(time.Hour), info.ModTime().Add(time.Hour)); err != nil {
		t.Fatal(err)
	}
	now := time.Unix(100, 0)
	s.observe(read(), now, time.Millisecond)
	if len(s.ready(now.Add(time.Second))) != 0 {
		t.Fatal("mtime triggered a content change")
	}
	// Rename-save replaces the inode, with its old timestamp and same byte count.
	replacement := filepath.Join(dir, "new")
	write(replacement, "new")
	if err := os.Chtimes(replacement, info.ModTime(), info.ModTime()); err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(replacement, path); err != nil {
		t.Fatal(err)
	}
	s.observe(read(), now, time.Millisecond)
	got := s.ready(now.Add(time.Second))
	if len(got) != 1 || got[0].Path != path || got[0].Digest != content("new").digest {
		t.Fatalf("rename-save lost: %+v", got)
	}
}

func TestCanceledWatch(t *testing.T) {
	for _, poll := range []bool{true, false} {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		err := Watch(ctx, []string{t.TempDir()}, Config{Poll: poll}, func(Change) error {
			t.Fatal("callback after cancellation")
			return nil
		})
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("cancel = %v", err)
		}
	}
}

func TestWaitCancellation(t *testing.T) {
	n, err := openNotify([]string{t.TempDir()}, false)
	if err != nil {
		t.Fatal(err)
	}
	defer n.close()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := n.wait(ctx, time.Hour); !errors.Is(err, context.Canceled) {
		t.Fatalf("long interval ignores cancellation: %v", err)
	}
}

func TestDeletedPathsAreReleased(t *testing.T) {
	s := newState(map[string]stamp{})
	now := time.Unix(100, 0)
	for i := 0; i < 10000; i++ {
		path := filepath.Join("dir", time.Unix(int64(i), 0).UTC().Format(time.RFC3339))
		s.observe(map[string]stamp{path: content("value")}, now, time.Millisecond)
		if len(s.ready(now.Add(time.Second))) != 1 {
			t.Fatal("creation lost")
		}
		s.observe(map[string]stamp{}, now, time.Millisecond)
		if len(s.ready(now.Add(time.Second))) != 1 {
			t.Fatal("deletion lost")
		}
	}
	if len(s.observed)+len(s.delivered)+len(s.pending) != 0 {
		t.Fatalf("deleted paths retained: observed=%d delivered=%d pending=%d", len(s.observed), len(s.delivered), len(s.pending))
	}
	s.observe(map[string]stamp{"readded": content("same")}, now, time.Millisecond)
	if len(s.ready(now.Add(time.Second))) != 1 {
		t.Fatal("readded content lost")
	}
}

func FuzzLatestContent(f *testing.F) {
	f.Add("old", "middle", "last")
	f.Fuzz(func(t *testing.T, old, middle, last string) {
		if len(old)+len(middle)+len(last) > 4096 {
			t.Skip()
		}
		start := time.Unix(100, 0)
		s := newState(map[string]stamp{"kernel.cu": content(old)})
		s.observe(map[string]stamp{"kernel.cu": content(middle)}, start, time.Second)
		s.observe(map[string]stamp{"kernel.cu": content(last)}, start.Add(time.Millisecond), time.Second)
		got := s.ready(start.Add(2 * time.Second))
		if content(last) == content(old) {
			if len(got) != 0 {
				t.Fatalf("unchanged content delivered: %+v", got)
			}
		} else if len(got) != 1 || got[0].Digest != content(last).digest || got[0].Removed {
			t.Fatalf("latest content lost: %+v", got)
		}
	})
}
