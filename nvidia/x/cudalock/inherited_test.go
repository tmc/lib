package cudalock

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"testing"
	"time"
)

// TestInheritedFromAcquire checks Inherited against the owner record a real
// Acquire writes, then against the same directory after Release.
func TestInheritedFromAcquire(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "lock")
	h, err := Acquire(context.Background(), WithDir(dir), WithLabel("inherited-test"))
	if err != nil {
		t.Fatal(err)
	}
	env := mapEnv(map[string]string{
		"GPU_LOCK_HELD": strconv.Itoa(os.Getpid()),
		"GPU_LOCKDIR":   dir,
	})
	owner, err := Inherited(env)
	if err != nil {
		t.Fatalf("Inherited while held = %v", err)
	}
	if owner.PID != os.Getpid() || owner.Label != "inherited-test" {
		t.Errorf("Inherited owner = %+v, want pid %d label inherited-test", owner, os.Getpid())
	}
	if err := h.Release(); err != nil {
		t.Fatal(err)
	}
	if _, err := Inherited(env); !errors.Is(err, ErrNotInherited) {
		t.Errorf("Inherited after Release = %v, want %v", err, ErrNotInherited)
	}
}

func TestInheritedRefuses(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "lock")
	if err := os.Mkdir(dir, 0755); err != nil {
		t.Fatal(err)
	}
	self := os.Getpid()
	writeOwner(t, dir, self)

	dead := exec.Command("true")
	if err := dead.Run(); err != nil {
		t.Fatal(err)
	}
	deadDir := filepath.Join(t.TempDir(), "lock")
	if err := os.Mkdir(deadDir, 0755); err != nil {
		t.Fatal(err)
	}
	writeOwner(t, deadDir, dead.Process.Pid)

	home := t.TempDir()
	homeDir := filepath.Join(home, "ml-explore", ".gpu-lock", "lock")
	if err := os.MkdirAll(homeDir, 0755); err != nil {
		t.Fatal(err)
	}
	writeOwner(t, homeDir, self)

	for _, tc := range []struct {
		name string
		env  map[string]string
		ok   bool
	}{
		{"unset", map[string]string{"GPU_LOCKDIR": dir}, false},
		{"empty", map[string]string{"GPU_LOCK_HELD": "", "GPU_LOCKDIR": dir}, false},
		{"one", map[string]string{"GPU_LOCK_HELD": "1", "GPU_LOCKDIR": dir}, false},
		{"zero", map[string]string{"GPU_LOCK_HELD": "0", "GPU_LOCKDIR": dir}, false},
		{"not a pid", map[string]string{"GPU_LOCK_HELD": "yes", "GPU_LOCKDIR": dir}, false},
		{"other pid", map[string]string{"GPU_LOCK_HELD": strconv.Itoa(self + 1), "GPU_LOCKDIR": dir}, false},
		{"no lock dir", map[string]string{"GPU_LOCK_HELD": strconv.Itoa(self), "GPU_LOCKDIR": filepath.Join(dir, "missing")}, false},
		{"dead holder", map[string]string{"GPU_LOCK_HELD": strconv.Itoa(dead.Process.Pid), "GPU_LOCKDIR": deadDir}, false},
		{"held", map[string]string{"GPU_LOCK_HELD": strconv.Itoa(self), "GPU_LOCKDIR": dir}, true},
		{"held, default dir under HOME", map[string]string{"GPU_LOCK_HELD": strconv.Itoa(self), "HOME": home}, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := Inherited(mapEnv(tc.env))
			if tc.ok {
				if err != nil {
					t.Fatalf("Inherited = %v, want nil", err)
				}
				return
			}
			if !errors.Is(err, ErrNotInherited) {
				t.Fatalf("Inherited = %v, want %v", err, ErrNotInherited)
			}
		})
	}
	if _, err := Inherited(nil); !errors.Is(err, ErrNotInherited) {
		t.Errorf("Inherited(nil) = %v, want %v", err, ErrNotInherited)
	}
}

func mapEnv(m map[string]string) func(string) string {
	return func(k string) string { return m[k] }
}

func writeOwner(t *testing.T, dir string, pid int) {
	t.Helper()
	o := Owner{PID: pid, Host: shortHostname(), Label: "test", Epoch: time.Now().Unix(), At: time.Now().Truncate(time.Second)}
	if err := os.WriteFile(filepath.Join(dir, "owner"), []byte(o.String()), 0644); err != nil {
		t.Fatal(err)
	}
}
