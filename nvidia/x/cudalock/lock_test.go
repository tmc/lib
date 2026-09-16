package cudalock

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestOwnerParseAndString(t *testing.T) {
	now := time.Now().Truncate(time.Second)
	orig := Owner{
		PID:   12345,
		Host:  "dgx",
		Label: "my-test",
		Epoch: now.Unix(),
		At:    now,
	}

	raw := orig.String()
	parsed, err := ParseOwner([]byte(raw))
	if err != nil {
		t.Fatalf("ParseOwner failed: %v", err)
	}

	if parsed.PID != orig.PID {
		t.Errorf("PID = %d, want %d", parsed.PID, orig.PID)
	}
	if parsed.Host != orig.Host {
		t.Errorf("Host = %q, want %q", parsed.Host, orig.Host)
	}
	if parsed.Label != orig.Label {
		t.Errorf("Label = %q, want %q", parsed.Label, orig.Label)
	}
	if parsed.Epoch != orig.Epoch {
		t.Errorf("Epoch = %d, want %d", parsed.Epoch, orig.Epoch)
	}
}

func TestAcquireAndRelease(t *testing.T) {
	tmp := t.TempDir()
	lockDir := filepath.Join(tmp, "sub", "lock")

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	h1, err := Acquire(ctx, WithDir(lockDir), WithLabel("test-1"))
	if err != nil {
		t.Fatalf("Acquire failed: %v", err)
	}

	// Verify owner file exists and is valid
	data, err := os.ReadFile(filepath.Join(lockDir, "owner"))
	if err != nil {
		t.Fatalf("ReadFile owner failed: %v", err)
	}
	owner, err := ParseOwner(data)
	if err != nil {
		t.Fatalf("ParseOwner failed: %v", err)
	}
	if owner.Label != "test-1" {
		t.Errorf("owner label = %q, want 'test-1'", owner.Label)
	}

	// Release
	if err := h1.Release(); err != nil {
		t.Fatalf("Release failed: %v", err)
	}

	// Verify directory is removed
	if _, err := os.Stat(lockDir); !os.IsNotExist(err) {
		t.Errorf("lock directory still exists after Release")
	}
}

func TestMutualExclusionAndContention(t *testing.T) {
	tmp := t.TempDir()
	lockDir := filepath.Join(tmp, "lock")

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	// Simulate external process holding the lock by writing a fake owner with our own PID
	if err := os.Mkdir(lockDir, 0755); err != nil {
		t.Fatalf("mkdir failed: %v", err)
	}
	fakeOwner := Owner{
		PID:   os.Getpid(),
		Host:  shortHostname(),
		Label: "external-holder",
		Epoch: time.Now().Unix(),
		At:    time.Now(),
	}
	if err := os.WriteFile(filepath.Join(lockDir, "owner"), []byte(fakeOwner.String()), 0644); err != nil {
		t.Fatalf("WriteFile failed: %v", err)
	}

	// Clear in-process reentrancy tracking so this is treated as a separate process attempt
	processMu.Lock()
	processHoldCount = 0
	processActiveDir = ""
	processMu.Unlock()
	_ = os.Unsetenv("GPU_LOCK_HELD")

	// Attempting acquire with short timeout should time out because PID is alive
	acquireCtx, acquireCancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer acquireCancel()

	_, err := Acquire(acquireCtx, WithDir(lockDir), WithPollInterval(50*time.Millisecond))
	if err == nil {
		t.Fatal("Acquire succeeded while lock was held by living process, expected timeout error")
	}

	// Now release fake lock
	_ = os.Remove(filepath.Join(lockDir, "owner"))
	_ = os.Remove(lockDir)

	// Now acquire should succeed
	h, err := Acquire(ctx, WithDir(lockDir), WithLabel("resumed"))
	if err != nil {
		t.Fatalf("Acquire failed after release: %v", err)
	}
	_ = h.Release()
}

func TestDeadProcessReclaim(t *testing.T) {
	tmp := t.TempDir()
	lockDir := filepath.Join(tmp, "lock")

	if err := os.Mkdir(lockDir, 0755); err != nil {
		t.Fatalf("mkdir failed: %v", err)
	}
	// Use an impossible PID (99999999) that is dead
	deadPID := 99999999
	fakeOwner := Owner{
		PID:   deadPID,
		Host:  shortHostname(),
		Label: "dead-job",
		Epoch: time.Now().Unix() - 100,
		At:    time.Now().Add(-100 * time.Second),
	}
	if err := os.WriteFile(filepath.Join(lockDir, "owner"), []byte(fakeOwner.String()), 0644); err != nil {
		t.Fatalf("WriteFile failed: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	h, err := Acquire(ctx, WithDir(lockDir), WithLabel("reclaimer"))
	if err != nil {
		t.Fatalf("Acquire failed to reclaim dead process lock: %v", err)
	}
	defer h.Release()

	if h.Owner().Label != "reclaimer" {
		t.Errorf("acquired label = %q, want 'reclaimer'", h.Owner().Label)
	}
}

func TestForeignHostProtection(t *testing.T) {
	tmp := t.TempDir()
	lockDir := filepath.Join(tmp, "lock")

	if err := os.Mkdir(lockDir, 0755); err != nil {
		t.Fatalf("mkdir failed: %v", err)
	}
	// Owner is on another host
	foreignOwner := Owner{
		PID:   99999999, // dead PID on our host, but host is foreign!
		Host:  "other-node-xyz",
		Label: "foreign-job",
		Epoch: time.Now().Unix() - 1000,
		At:    time.Now().Add(-1000 * time.Second),
	}
	if err := os.WriteFile(filepath.Join(lockDir, "owner"), []byte(foreignOwner.String()), 0644); err != nil {
		t.Fatalf("WriteFile failed: %v", err)
	}

	// Reclaim should NOT happen for foreign owner
	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()

	_, err := Acquire(ctx, WithDir(lockDir), WithPollInterval(50*time.Millisecond))
	if err == nil {
		t.Fatal("Acquire should not reclaim foreign host lock")
	}
}

func TestReentrancy(t *testing.T) {
	tmp := t.TempDir()
	lockDir := filepath.Join(tmp, "lock")

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	h1, err := Acquire(ctx, WithDir(lockDir), WithLabel("reentrant-root"))
	if err != nil {
		t.Fatalf("root acquire failed: %v", err)
	}

	h2, err := Acquire(ctx, WithDir(lockDir), WithLabel("reentrant-child"))
	if err != nil {
		t.Fatalf("nested acquire failed: %v", err)
	}

	if err := h2.Release(); err != nil {
		t.Fatalf("nested release failed: %v", err)
	}

	// Root lock should still be active
	if _, err := os.Stat(lockDir); err != nil {
		t.Fatalf("lock directory was removed prematurely: %v", err)
	}

	if err := h1.Release(); err != nil {
		t.Fatalf("root release failed: %v", err)
	}

	// Now lock directory should be gone
	if _, err := os.Stat(lockDir); !os.IsNotExist(err) {
		t.Errorf("lock directory not removed after root release")
	}
}

func TestStatusReporting(t *testing.T) {
	tmp := t.TempDir()
	lockDir := filepath.Join(tmp, "lock")

	rep, err := Status(lockDir)
	if err != nil {
		t.Fatalf("Status failed: %v", err)
	}
	if rep.MutexHeld {
		t.Errorf("expected MutexHeld = false, got true")
	}

	// Acquire lock and check status again
	h, err := Acquire(context.Background(), WithDir(lockDir), WithLabel("status-test"))
	if err != nil {
		t.Fatalf("Acquire failed: %v", err)
	}
	defer h.Release()

	rep2, err := Status(lockDir)
	if err != nil {
		t.Fatalf("Status failed: %v", err)
	}
	if !rep2.MutexHeld {
		t.Errorf("expected MutexHeld = true, got false")
	}
	if rep2.Owner == nil || rep2.Owner.Label != "status-test" {
		t.Errorf("unexpected owner in status: %+v", rep2.Owner)
	}

	s := rep2.String()
	if !strings.Contains(s, "FILESYSTEM MUTEX") || !strings.Contains(s, "PHYSICAL DEVICE ACTIVITY") {
		t.Errorf("Status string missing required section headers:\n%s", s)
	}
}
