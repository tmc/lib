package cudasass_test

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/cudasass"
)

func TestCandidateLoadRequiresGPULock(t *testing.T) {
	// There is no mirror to refuse, so the lock is the only gate left.
	patched := cudasass.PatchedImage{
		Kernel:  "intmix",
		Section: ".text.intmix",
		Patches: []cudasass.AppliedPatch{{Address: 0x10, FileOffset: 0x910}},
	}
	h, env := loadTestLease(t)
	pid := strconv.Itoa(h.Owner().PID)
	dead := exec.Command("true")
	if err := dead.Run(); err != nil {
		t.Fatal(err)
	}
	deadOwner := h.Owner()
	deadOwner.PID = dead.Process.Pid
	deadDir := loadTestOwnerDir(t, deadOwner)
	foreignOwner := h.Owner()
	foreignOwner.Host += "-foreign"
	foreignDir := loadTestOwnerDir(t, foreignOwner)
	for _, tc := range []struct {
		name string
		held string
		dir  string
	}{
		{"empty", "", h.Dir()},
		{"zero", "0", h.Dir()},
		{"not a pid", "yes", h.Dir()},
		{"presence only", "1", h.Dir()},
		{"mismatched pid", strconv.Itoa(h.Owner().PID + 1), h.Dir()},
		{"live pid without owner record", pid, filepath.Join(t.TempDir(), "missing")},
		{"dead holder", strconv.Itoa(dead.Process.Pid), deadDir},
		{"foreign host with live local pid", pid, foreignDir},
	} {
		t.Run(tc.name, func(t *testing.T) {
			values := map[string]string{"GPU_LOCK_HELD": tc.held, "GPU_LOCKDIR": tc.dir}
			err := cudasass.AuthorizeLoad(patched, func(k string) string { return values[k] })
			if !errors.Is(err, cudasass.ErrLockNotHeld) || !errors.Is(err, cudalock.ErrNotInherited) {
				t.Fatalf("AuthorizeLoad = %v, want lock and inheritance refusal", err)
			}
		})
	}
	if err := cudasass.AuthorizeLoad(patched, env); err != nil {
		t.Fatalf("AuthorizeLoad with corroborated owner = %v", err)
	}
	if err := cudasass.AuthorizeLoad(patched, nil); !errors.Is(err, cudasass.ErrLockNotHeld) {
		t.Errorf("AuthorizeLoad with no environment = %v, want %v", err, cudasass.ErrLockNotHeld)
	}
	if err := h.Release(); err != nil {
		t.Fatal(err)
	}
	if err := cudasass.AuthorizeLoad(patched, env); !errors.Is(err, cudasass.ErrLockNotHeld) {
		t.Fatalf("AuthorizeLoad after release = %v, want %v", err, cudasass.ErrLockNotHeld)
	}
}

func loadTestLease(t *testing.T) (*cudalock.Handle, func(string) string) {
	t.Helper()
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()
	h, err := cudalock.Acquire(ctx, cudalock.WithDir(filepath.Join(t.TempDir(), "lock")))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := h.Release(); err != nil {
			t.Errorf("release test lease: %v", err)
		}
	})
	values := map[string]string{"GPU_LOCK_HELD": strconv.Itoa(h.Owner().PID), "GPU_LOCKDIR": h.Dir()}
	return h, func(k string) string { return values[k] }
}

func loadTestOwnerDir(t *testing.T, owner cudalock.Owner) string {
	t.Helper()
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "owner"), []byte(owner.String()), 0600); err != nil {
		t.Fatal(err)
	}
	return dir
}
