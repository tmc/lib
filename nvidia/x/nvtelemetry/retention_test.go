package nvtelemetry

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

func TestRetentionPublicRecoveryAndBounds(t *testing.T) {
	b, err := os.ReadFile("testdata/canonical_v2.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	cfg := RetentionConfig{Owner: "task-111", MaxFiles: 2, MaxStreamBytes: int64(len(b)), MaxBytes: int64(2 * len(b))}
	path := filepath.Join(t.TempDir(), "owned")
	r, err := OpenRetention(path, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := OpenRetention(path, cfg); err == nil {
		t.Fatal("duplicate writer admitted")
	}
	for i := uint64(1); i <= 20; i++ {
		if err := r.Store(i, bytes.NewReader(b)); err != nil {
			t.Fatal(err)
		}
	}
	if err := r.Store(20, bytes.NewReader(b)); err == nil {
		t.Fatal("duplicate admitted")
	}
	if err := r.Store(21, bytes.NewReader(b[:len(b)/2])); err == nil {
		t.Fatal("truncated stream admitted")
	}
	if err := r.Close(); err != nil {
		t.Fatal(err)
	}
	entries, _ := os.ReadDir(path)
	if len(entries) != 3 {
		t.Fatalf("entries %d", len(entries))
	}
	small := cfg
	small.MaxFiles = 1
	if _, err := OpenRetention(path, small); err == nil {
		t.Fatal("invalid reload deleted state")
	}
	if err := os.Link(filepath.Join(path, "owner.json"), filepath.Join(path, "lease")); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(path, "pending.jsonl"), []byte("crash partial"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := RecoverRetention(path, "foreign"); err == nil {
		t.Fatal("foreign takeover")
	}
	if err := RecoverRetention(path, cfg.Owner); err != nil {
		t.Fatal(err)
	}
	if _, err := OpenRetention(path, cfg); err == nil {
		t.Fatal("unpublished artifact silently discarded")
	}
	if pending, err := os.ReadFile(filepath.Join(path, "pending.jsonl")); err != nil || string(pending) != "crash partial" {
		t.Fatalf("pending artifact changed: %q %v", pending, err)
	}
	if err := os.Rename(filepath.Join(path, "pending.jsonl"), filepath.Join(t.TempDir(), "preserved.jsonl")); err != nil {
		t.Fatal(err)
	}
	r, err = OpenRetention(path, cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	if _, err := os.Stat(filepath.Join(path, "pending.jsonl")); !os.IsNotExist(err) {
		t.Fatal("pending stream retained")
	}
	if err := r.Store(21, bytes.NewReader(b)); err != nil {
		t.Fatal(err)
	}
}

func TestRetentionPublicRefusals(t *testing.T) {
	b, _ := os.ReadFile("testdata/canonical_v2.jsonl")
	cfg := RetentionConfig{Owner: "task", MaxFiles: 2, MaxStreamBytes: int64(len(b)), MaxBytes: int64(2 * len(b))}
	for _, kind := range []string{"foreign", "symlink", "marker", "corrupt", "write failure"} {
		t.Run(kind, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "owned")
			r, err := OpenRetention(path, cfg)
			if err != nil {
				t.Fatal(err)
			}
			if kind == "write failure" {
				os.Mkdir(filepath.Join(path, "pending.jsonl"), 0700)
				if err := r.Store(1, bytes.NewReader(b)); err == nil {
					t.Fatal("failed write admitted")
				}
				os.Remove(filepath.Join(path, "pending.jsonl"))
				r.Close()
				return
			}
			r.Close()
			switch kind {
			case "foreign":
				os.WriteFile(filepath.Join(path, "foreign"), []byte("keep"), 0600)
			case "symlink":
				os.Symlink("owner.json", filepath.Join(path, "pending.jsonl"))
			case "marker":
				os.WriteFile(filepath.Join(path, "owner.json"), []byte(`{"version":1,"owner":"other"}`), 0600)
			case "corrupt":
				os.WriteFile(filepath.Join(path, "stream-0000000000001.jsonl"), []byte("bad"), 0600)
			}
			if _, err := OpenRetention(path, cfg); err == nil {
				t.Fatal("invalid retained state admitted")
			}
		})
	}
}

func TestRetentionStatus(t *testing.T) {
	b, err := os.ReadFile("testdata/canonical_v2.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	cfg := RetentionConfig{Owner: "status", MaxFiles: 1, MaxBytes: int64(len(b)), MaxStreamBytes: int64(len(b))}
	path := filepath.Join(t.TempDir(), "owned")
	r, err := OpenRetention(path, cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	s := r.Status()
	if !s.Ready || !s.CountsAvailable || s.Bytes != 0 || s.AtCapacity || s.Closed || s.MaxFiles != 1 {
		t.Fatalf("initial %+v", s)
	}
	for i := uint64(1); i <= 3; i++ {
		if err := r.Store(i, bytes.NewReader(b)); err != nil {
			t.Fatal(err)
		}
	}
	s = r.Status()
	if s.Files != 1 || s.Bytes != int64(len(b)) || s.Stores != 3 || s.PrunedFiles != 2 || s.PrunedBytes != uint64(2*len(b)) || !s.AtCapacity || !s.Ready {
		t.Fatalf("pressure %+v", s)
	}
	if err := r.Store(3, bytes.NewReader(b)); err == nil {
		t.Fatal("duplicate accepted")
	}
	s = r.Status()
	if s.StoreFailures != 1 || s.LastOperationError != "store_failed" || !s.Ready {
		t.Fatalf("failure %+v", s)
	}
	if err := os.WriteFile(filepath.Join(path, "stream-0000000000003.jsonl"), []byte("corrupt"), 0600); err != nil {
		t.Fatal(err)
	}
	s = r.Status()
	if s.Ready || s.CountsAvailable || s.StateError != "state_unavailable" {
		t.Fatalf("corrupt state %+v", s)
	}
	if err := r.Close(); err != nil {
		t.Fatal(err)
	}
	s = r.Status()
	if !s.Closed || s.Ready || s.CountsAvailable || s.Stores != 3 {
		t.Fatalf("closed %+v", s)
	}
}

func TestRetentionConcurrentStatus(t *testing.T) {
	b, err := os.ReadFile("testdata/canonical_v2.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	r, err := OpenRetention(filepath.Join(t.TempDir(), "owned"), RetentionConfig{Owner: "concurrent", MaxFiles: 1, MaxBytes: int64(len(b)), MaxStreamBytes: int64(len(b))})
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	done := make(chan struct{})
	go func() {
		defer close(done)
		for i := 0; i < 10; i++ {
			r.Status()
		}
	}()
	for i := uint64(1); i <= 10; i++ {
		if err := r.Store(i, bytes.NewReader(b)); err != nil {
			t.Fatal(err)
		}
	}
	<-done
	if s := r.Status(); s.Stores != 10 || s.PrunedFiles != 9 {
		t.Fatalf("counters %+v", s)
	}
}
