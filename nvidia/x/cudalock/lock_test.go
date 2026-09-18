package cudalock

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

func TestOwnerParseAndString(t *testing.T) {
	now := time.Now().Truncate(time.Second)
	orig := Owner{
		PID:   12345,
		Host:  "gpu-host",
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

func TestAncestorInheritedLock(t *testing.T) {
	tmp := t.TempDir()
	lockDir := filepath.Join(tmp, "lock")

	// Set up lock directory as if held by parent process
	if err := os.MkdirAll(lockDir, 0755); err != nil {
		t.Fatalf("MkdirAll failed: %v", err)
	}
	parentPID := os.Getppid()
	ancestorOwner := Owner{
		PID:   parentPID,
		Host:  shortHostname(),
		Label: "ancestor-bash",
		Epoch: time.Now().Unix(),
		At:    time.Now(),
	}
	if err := os.WriteFile(filepath.Join(lockDir, "owner"), []byte(ancestorOwner.String()), 0644); err != nil {
		t.Fatalf("write owner failed: %v", err)
	}

	t.Setenv("GPU_LOCKDIR", lockDir)
	t.Setenv("GPU_LOCK_HELD", strconv.Itoa(parentPID))

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	h, err := Acquire(ctx)
	if err != nil {
		t.Fatalf("Acquire failed with ancestor GPU_LOCK_HELD: %v", err)
	}
	if !h.IsNested() {
		t.Errorf("expected handle to be nested, got false")
	}
	if h.Owner().PID != parentPID {
		t.Errorf("expected owner PID %d, got %d", parentPID, h.Owner().PID)
	}
	if h.Owner().Label != "ancestor-bash" {
		t.Errorf("expected owner label 'ancestor-bash', got %q", h.Owner().Label)
	}

	if err := h.Release(); err != nil {
		t.Fatalf("Release failed: %v", err)
	}

	// Lock dir and owner file should remain untouched because ancestor owns it
	if _, err := os.Stat(lockDir); err != nil {
		t.Errorf("ancestor lock directory was removed by descendant release: %v", err)
	}

	// Re-acquisition after release within descendant should also succeed
	h2, err := Acquire(ctx)
	if err != nil {
		t.Fatalf("second Acquire failed: %v", err)
	}
	if !h2.IsNested() {
		t.Errorf("expected second handle to be nested, got false")
	}
	_ = h2.Release()
}

func TestAncestorInheritedStaleDeadPID(t *testing.T) {
	tmp := t.TempDir()
	lockDir := filepath.Join(tmp, "lock")

	deadPID := 99999999
	for isAlive(deadPID) {
		deadPID++
	}

	t.Setenv("GPU_LOCKDIR", lockDir)
	t.Setenv("GPU_LOCK_HELD", strconv.Itoa(deadPID))

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	h, err := Acquire(ctx)
	if err != nil {
		t.Fatalf("Acquire failed: %v", err)
	}
	if h.IsNested() {
		t.Errorf("expected handle to NOT be nested when ancestor PID is dead, got nested")
	}
	if h.Owner().PID != os.Getpid() {
		t.Errorf("expected owner PID %d (our PID), got %d", os.Getpid(), h.Owner().PID)
	}

	if err := h.Release(); err != nil {
		t.Fatalf("Release failed: %v", err)
	}
	if _, err := os.Stat(lockDir); !os.IsNotExist(err) {
		t.Errorf("lock directory not removed after root release")
	}
}

func TestAncestorInheritedStaleDeadPIDContention(t *testing.T) {
	tmp := t.TempDir()
	lockDir := filepath.Join(tmp, "lock")

	// Another live process holds the lock directory
	if err := os.MkdirAll(lockDir, 0755); err != nil {
		t.Fatalf("MkdirAll failed: %v", err)
	}
	otherPID := os.Getppid()
	liveOwner := Owner{
		PID:   otherPID,
		Host:  shortHostname(),
		Label: "live-holder",
		Epoch: time.Now().Unix(),
		At:    time.Now(),
	}
	if err := os.WriteFile(filepath.Join(lockDir, "owner"), []byte(liveOwner.String()), 0644); err != nil {
		t.Fatalf("write owner failed: %v", err)
	}

	deadPID := 99999999
	for isAlive(deadPID) {
		deadPID++
	}

	t.Setenv("GPU_LOCKDIR", lockDir)
	t.Setenv("GPU_LOCK_HELD", strconv.Itoa(deadPID))

	// Attempt acquire with a short timeout. It must NOT treat deadPID as granting access,
	// and must wait and time out against the live holder.
	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()

	_, err := Acquire(ctx, WithPollInterval(20*time.Millisecond))
	if err == nil {
		t.Fatalf("Acquire should have timed out against live holder with stale GPU_LOCK_HELD, but succeeded")
	}
	if !errors.Is(err, context.DeadlineExceeded) && !strings.Contains(err.Error(), "context deadline exceeded") {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestChildProcessAncestorLock(t *testing.T) {
	if os.Getenv("GO_WANT_CUDALOCK_CHILD_TEST") == "1" {
		// Inside child process: Acquire should succeed immediately without deadlocking
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()

		h, err := Acquire(ctx)
		if err != nil {
			fmt.Fprintf(os.Stderr, "child Acquire failed: %v\n", err)
			os.Exit(1)
		}
		if !h.IsNested() {
			fmt.Fprintf(os.Stderr, "child handle is not nested\n")
			os.Exit(2)
		}
		if err := h.Release(); err != nil {
			fmt.Fprintf(os.Stderr, "child Release failed: %v\n", err)
			os.Exit(3)
		}
		os.Exit(0)
	}

	tmp := t.TempDir()
	lockDir := filepath.Join(tmp, "lock")

	// Parent process creates and holds lock
	if err := os.MkdirAll(lockDir, 0755); err != nil {
		t.Fatalf("MkdirAll failed: %v", err)
	}
	myPID := os.Getpid()
	owner := Owner{
		PID:   myPID,
		Host:  shortHostname(),
		Label: "parent-test",
		Epoch: time.Now().Unix(),
		At:    time.Now(),
	}
	if err := os.WriteFile(filepath.Join(lockDir, "owner"), []byte(owner.String()), 0644); err != nil {
		t.Fatalf("write owner failed: %v", err)
	}

	// Spawn child process with GPU_LOCK_HELD set to our PID (which is a different live PID for the child)
	cmd := exec.Command(os.Args[0], "-test.run=TestChildProcessAncestorLock")
	cmd.Env = append(os.Environ(),
		"GO_WANT_CUDALOCK_CHILD_TEST=1",
		"GPU_LOCKDIR="+lockDir,
		fmt.Sprintf("GPU_LOCK_HELD=%d", myPID),
	)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("child process failed: %v\noutput: %s", err, string(out))
	}
}

// TestAncestorInheritedLivePIDThatHoldsNothing covers the fail-open direction
// the dead-PID tests cannot reach. PID 1 is alive on every Unix, so a caller who
// sets GPU_LOCK_HELD=1 by hand -- the invocation "gpu-lock env GPU_LOCK_HELD=1
// cmd", which circulated among agents -- passes the liveness check while no
// ancestor holds anything. Before corroborating against the owner file, Acquire
// returned a nested handle and the caller ran unlocked believing it was covered.
func TestAncestorInheritedLivePIDThatHoldsNothing(t *testing.T) {
	lockDir := filepath.Join(t.TempDir(), "lock")
	t.Setenv("GPU_LOCKDIR", lockDir)
	t.Setenv("GPU_LOCK_HELD", "1")
	if !isAlive(1) {
		t.Skip("PID 1 is not observable here, so the hazard cannot be reproduced")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	h, err := Acquire(ctx)
	if err != nil {
		t.Fatalf("Acquire failed: %v", err)
	}
	defer h.Release()
	if h.IsNested() {
		t.Fatal("a live PID that holds nothing was accepted as an ancestor holder; the caller would run unlocked")
	}
	if h.Owner().PID != os.Getpid() {
		t.Errorf("expected to hold the lock ourselves (PID %d), got %d", os.Getpid(), h.Owner().PID)
	}
	if _, err := os.Stat(lockDir); err != nil {
		t.Errorf("a non-nested acquire must create the lock directory: %v", err)
	}
}

// TestAncestorInheritedLivePIDDisagreesWithOwner is the same hazard with a lock
// that is genuinely held: the environment names a live PID, but the owner file
// names someone else. The environment must not win.
func TestAncestorInheritedLivePIDDisagreesWithOwner(t *testing.T) {
	lockDir := filepath.Join(t.TempDir(), "lock")
	if err := os.MkdirAll(lockDir, 0755); err != nil {
		t.Fatal(err)
	}
	held := Owner{PID: os.Getppid(), Host: shortHostname(), Label: "real-holder", Epoch: time.Now().Unix(), At: time.Now()}
	if err := os.WriteFile(filepath.Join(lockDir, "owner"), []byte(held.String()), 0644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("GPU_LOCKDIR", lockDir)
	t.Setenv("GPU_LOCK_HELD", "1")

	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()
	if _, err := Acquire(ctx, WithPollInterval(20*time.Millisecond)); err == nil {
		t.Fatal("Acquire granted against a live foreign holder because the environment named a different live PID")
	}
}
