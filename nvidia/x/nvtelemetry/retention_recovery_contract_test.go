package nvtelemetry

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

func retentionTestConfig(t *testing.T) (RetentionConfig, []byte) {
	t.Helper()
	b, err := os.ReadFile("testdata/canonical_v2.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	return RetentionConfig{Owner: "recovery", MaxFiles: 2, MaxBytes: int64(2 * len(b)), MaxStreamBytes: int64(len(b))}, b
}

func retentionSnapshot(t *testing.T, path string) map[string]string {
	t.Helper()
	result := make(map[string]string)
	err := filepath.WalkDir(path, func(path string, entry os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		info, err := entry.Info()
		if err != nil {
			return err
		}
		value := info.Mode().String()
		if info.Mode().IsRegular() {
			b, err := os.ReadFile(path)
			if err != nil {
				return err
			}
			value += fmt.Sprintf(" %x", sha256.Sum256(b))
		}
		if info.Mode()&os.ModeSymlink != 0 {
			link, err := os.Readlink(path)
			if err != nil {
				return err
			}
			value += " " + link
		}
		result[path] = value
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	return result
}

// The child holds the same admitted owner fd used by Open or Recover. It never
// guesses process death. The crash arm exits without Close to leave a stale link.
func TestRetentionLeaseProcess(t *testing.T) {
	path := os.Getenv("RETENTION_LEASE_CHILD")
	if path == "" {
		return
	}
	cfg, b := retentionTestConfig(t)
	mode := os.Getenv("RETENTION_LEASE_MODE")
	var r *Retention
	var err error
	if mode == "recoverer" {
		r, err = retentionAdmission(path, cfg.Owner, false)
	} else {
		r, err = OpenRetention(path, cfg)
	}
	if err != nil {
		t.Fatal(err)
	}
	if mode == "crash" {
		if err := r.Store(1, bytes.NewReader(b)); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(path, "pending.jsonl"), []byte("original interrupted bytes"), 0600); err != nil {
			t.Fatal(err)
		}
	}
	fmt.Println("lease-held")
	if _, err := io.Copy(io.Discard, os.Stdin); err != nil {
		t.Fatal(err)
	}
	if mode == "crash" {
		os.Exit(0)
	}
	if mode == "recoverer" {
		err = r.release()
	} else {
		err = r.Close()
	}
	if err != nil {
		t.Fatal(err)
	}
}

func retentionChild(t *testing.T, path, mode string) func() {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestRetentionLeaseProcess$")
	cmd.Env = append(os.Environ(), "RETENTION_LEASE_CHILD="+path, "RETENTION_LEASE_MODE="+mode)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		t.Fatal(err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	finished := false
	t.Cleanup(func() {
		if !finished {
			stdin.Close()
			cmd.Wait()
		}
		cancel()
	})
	scanner := bufio.NewScanner(stdout)
	if !scanner.Scan() || scanner.Text() != "lease-held" {
		t.Fatalf("child admission: %q %v %s", scanner.Text(), scanner.Err(), stderr.String())
	}
	return func() {
		t.Helper()
		stdin.Close()
		rest, err := io.ReadAll(stdout)
		if err != nil {
			t.Fatal(err)
		}
		err = cmd.Wait()
		finished = true
		cancel()
		if err != nil {
			t.Fatalf("child: %v %s %s", err, rest, stderr.String())
		}
	}
}

func TestRetentionCrossProcessAdmissionAndCrash(t *testing.T) {
	cfg, b := retentionTestConfig(t)
	for _, mode := range []string{"writer", "recoverer", "crash"} {
		t.Run(mode, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "owned")
			if mode == "recoverer" {
				r, err := OpenRetention(path, cfg)
				if err != nil {
					t.Fatal(err)
				}
				if err := r.Close(); err != nil {
					t.Fatal(err)
				}
			}
			stop := retentionChild(t, path, mode)
			ownerInfo, err := os.Lstat(filepath.Join(path, "owner.json"))
			if err != nil {
				t.Fatal(err)
			}
			before := retentionSnapshot(t, path)
			if err := RecoverRetention(path, cfg.Owner); err == nil {
				t.Fatal("live owner recovery admitted")
			}
			if _, err := OpenRetention(path, cfg); err == nil {
				t.Fatal("live owner writer admitted")
			}
			if after := retentionSnapshot(t, path); !reflect.DeepEqual(before, after) {
				t.Fatal("live refusal mutated state")
			}
			stop()
			if mode == "crash" {
				if _, err := OpenRetention(path, cfg); err == nil {
					t.Fatal("stale lease bypassed")
				}
				if err := RecoverRetention(path, cfg.Owner); err != nil {
					t.Fatal(err)
				}
				pending, err := os.ReadFile(filepath.Join(path, "pending.jsonl"))
				if err != nil || string(pending) != "original interrupted bytes" {
					t.Fatalf("pending lost: %q %v", pending, err)
				}
				if got, err := os.ReadFile(filepath.Join(path, "stream-0000000000001.jsonl")); err != nil || !bytes.Equal(got, b) {
					t.Fatal("retained stream changed")
				}
				if _, err := OpenRetention(path, cfg); err == nil {
					t.Fatal("pending silently discarded")
				}
				if err := os.Rename(filepath.Join(path, "pending.jsonl"), filepath.Join(t.TempDir(), "preserved.jsonl")); err != nil {
					t.Fatal(err)
				}
			} else if err := RecoverRetention(path, cfg.Owner); err != nil {
				t.Fatal(err)
			}
			currentOwner, err := os.Lstat(filepath.Join(path, "owner.json"))
			if err != nil || !os.SameFile(ownerInfo, currentOwner) {
				t.Fatal("permanent lock inode replaced")
			}
			r, err := OpenRetention(path, cfg)
			if err != nil {
				t.Fatal(err)
			}
			if err := r.Store(2, bytes.NewReader(b)); err != nil {
				t.Fatal(err)
			}
			if err := r.Close(); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestRetentionRecoveryRefusesUnsafeStateUnchanged(t *testing.T) {
	cfg, b := retentionTestConfig(t)
	for _, kind := range []string{"foreign", "corrupt-stream", "corrupt-marker", "legacy", "lease-directory", "lease-symlink", "marker-symlink", "wrong-owner"} {
		t.Run(kind, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "owned")
			r, err := OpenRetention(path, cfg)
			if err != nil {
				t.Fatal(err)
			}
			if err := r.Store(1, bytes.NewReader(b)); err != nil {
				t.Fatal(err)
			}
			if err := r.Close(); err != nil {
				t.Fatal(err)
			}
			if err := os.Link(filepath.Join(path, "owner.json"), filepath.Join(path, "lease")); err != nil {
				t.Fatal(err)
			}
			write := func(name, body string) {
				t.Helper()
				if err := os.WriteFile(filepath.Join(path, name), []byte(body), 0600); err != nil {
					t.Fatal(err)
				}
			}
			owner := cfg.Owner
			switch kind {
			case "foreign":
				write("unrelated", "preserve")
			case "corrupt-stream":
				write("stream-0000000000001.jsonl", "bad")
			case "corrupt-marker":
				write("owner.json", "{")
			case "legacy":
				write("owner.json", `{"version":1,"owner":"recovery"}`)
			case "lease-directory":
				os.Remove(filepath.Join(path, "lease"))
				os.Mkdir(filepath.Join(path, "lease"), 0700)
			case "lease-symlink":
				os.Remove(filepath.Join(path, "lease"))
				os.Symlink("owner.json", filepath.Join(path, "lease"))
			case "marker-symlink":
				os.Remove(filepath.Join(path, "owner.json"))
				os.Symlink("lease", filepath.Join(path, "owner.json"))
			case "wrong-owner":
				owner = "other"
			}
			before := retentionSnapshot(t, path)
			if err := RecoverRetention(path, owner); err == nil {
				t.Fatal("unsafe recovery admitted")
			}
			badcfg := cfg
			badcfg.Owner = owner
			if _, err := OpenRetention(path, badcfg); err == nil {
				t.Fatal("unsafe open admitted")
			}
			if after := retentionSnapshot(t, path); !reflect.DeepEqual(before, after) {
				t.Fatal("refusal mutated artifacts")
			}
		})
	}
}

func TestRetentionNamespaceAndFailedClose(t *testing.T) {
	cfg, b := retentionTestConfig(t)
	for _, kind := range []string{"owner", "lease", "directory"} {
		t.Run(kind, func(t *testing.T) {
			parent := t.TempDir()
			path := filepath.Join(parent, "owned")
			r, err := OpenRetention(path, cfg)
			if err != nil {
				t.Fatal(err)
			}
			restore := func() {}
			switch kind {
			case "owner":
				old := filepath.Join(parent, "original-owner")
				os.Rename(filepath.Join(path, "owner.json"), old)
				data, err := os.ReadFile(old)
				if err != nil {
					t.Fatal(err)
				}
				os.WriteFile(filepath.Join(path, "owner.json"), data, 0600)
				restore = func() {
					os.Remove(filepath.Join(path, "owner.json"))
					os.Rename(old, filepath.Join(path, "owner.json"))
				}
			case "lease":
				old := filepath.Join(parent, "original-lease")
				os.Rename(filepath.Join(path, "lease"), old)
				os.WriteFile(filepath.Join(path, "lease"), []byte("foreign"), 0600)
				restore = func() { os.Remove(filepath.Join(path, "lease")); os.Rename(old, filepath.Join(path, "lease")) }
			case "directory":
				old := filepath.Join(parent, "original-directory")
				os.Rename(path, old)
				os.Mkdir(path, 0700)
				os.WriteFile(filepath.Join(path, "unrelated"), []byte("keep"), 0600)
				restore = func() { os.Remove(filepath.Join(path, "unrelated")); os.Remove(path); os.Rename(old, path) }
			}
			before := retentionSnapshot(t, parent)
			if err := r.Close(); err == nil {
				t.Fatal("substituted namespace close admitted")
			}
			if err := r.Store(1, bytes.NewReader(b)); err == nil {
				t.Fatal("substituted namespace write admitted")
			}
			if s := r.Status(); s.Ready || s.Closed {
				t.Fatalf("failed cleanup lost active admission: %+v", s)
			}
			if err := RecoverRetention(path, cfg.Owner); err == nil {
				t.Fatal("substituted namespace recovered")
			}
			if after := retentionSnapshot(t, parent); !reflect.DeepEqual(before, after) {
				t.Fatal("namespace refusal changed artifacts")
			}
			restore()
			// Failed Close must keep the original inode locked until a successful retry.
			if err := RecoverRetention(path, cfg.Owner); err == nil {
				t.Fatal("failed close released live admission")
			}
			if err := r.Close(); err != nil {
				t.Fatal(err)
			}
			next, err := OpenRetention(path, cfg)
			if err != nil {
				t.Fatal(err)
			}
			if err := next.Close(); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestRetentionFailedOpenPreservesAndReleases(t *testing.T) {
	cfg, b := retentionTestConfig(t)
	path := filepath.Join(t.TempDir(), "owned")
	r, err := OpenRetention(path, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := r.Store(1, bytes.NewReader(b)); err != nil {
		t.Fatal(err)
	}
	if err := r.Close(); err != nil {
		t.Fatal(err)
	}
	os.WriteFile(filepath.Join(path, "pending.jsonl"), []byte("preserve"), 0600)
	before := retentionSnapshot(t, path)
	for i := 0; i < 3; i++ {
		if _, err := OpenRetention(path, cfg); err == nil || !strings.Contains(err.Error(), "pending") {
			t.Fatalf("open: %v", err)
		}
	}
	if after := retentionSnapshot(t, path); !reflect.DeepEqual(before, after) {
		t.Fatal("failed open mutated artifacts")
	}
	if err := RecoverRetention(path, cfg.Owner); err != nil {
		t.Fatalf("failed open leaked admission: %v", err)
	}
	os.Remove(filepath.Join(path, "pending.jsonl")) // This control owns and explicitly disposes its fixture.
	r, err = OpenRetention(path, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := r.Close(); err != nil {
		t.Fatal(err)
	}
}
