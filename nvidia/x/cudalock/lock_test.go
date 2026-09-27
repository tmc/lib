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
	processHolds = make(map[string]*dirHold)
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

// TestTruthReleaseOrderLeak tests Defect 1:
// Releasing the root handle before a nested one must not leak the lock dir.
func TestTruthReleaseOrderLeak(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "lock")
	ctx := context.Background()
	root, err := Acquire(ctx, WithDir(dir))
	if err != nil {
		t.Fatal(err)
	}
	nested, err := Acquire(ctx, WithDir(dir))
	if err != nil {
		t.Fatal(err)
	}
	root.Release()
	nested.Release()
	if _, err := os.Stat(dir); err == nil {
		t.Errorf("lock dir %s still exists after both handles released", dir)
	}
}

// TestTruthTwoDirsSelfDeadlock tests Defect 2:
// Holding a second lock dir b must not overwrite the hold state of dir a.
// Re-acquiring an already-held dir a must succeed immediately.
func TestTruthTwoDirsSelfDeadlock(t *testing.T) {
	a := filepath.Join(t.TempDir(), "a")
	b := filepath.Join(t.TempDir(), "b")
	ha, err := Acquire(context.Background(), WithDir(a))
	if err != nil {
		t.Fatal(err)
	}
	defer ha.Release()
	hb, err := Acquire(context.Background(), WithDir(b))
	if err != nil {
		t.Fatal(err)
	}
	defer hb.Release()
	ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
	defer cancel()
	h, err := Acquire(ctx, WithDir(a), WithPollInterval(10*time.Millisecond))
	if errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("re-acquire of held dir a blocked on own lock: %v", err)
	} else if err != nil {
		t.Errorf("re-acquire of held dir a failed: %v", err)
	} else if h != nil {
		h.Release()
	}
}

// TestTruthRunInheritedExportsWrongPID tests Defect 3:
// Under an inherited lock, Run must export the holder's PID, not its own PID.
func TestTruthRunInheritedExportsWrongPID(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "lock")
	os.Mkdir(dir, 0755)
	holder := os.Getppid()
	os.WriteFile(filepath.Join(dir, "owner"), []byte(Owner{PID: holder, Host: shortHostname(), Label: "x", At: time.Now()}.String()), 0644)
	t.Setenv("GPU_LOCKDIR", dir)
	t.Setenv("GPU_LOCK_HELD", strconv.Itoa(holder))
	out := filepath.Join(t.TempDir(), "out")
	code, err := Run(context.Background(), []string{"sh", "-c", "echo $GPU_LOCK_HELD > " + out})
	if err != nil || code != 0 {
		t.Fatal(code, err)
	}
	b, _ := os.ReadFile(out)
	if got := strings.TrimSpace(string(b)); got != strconv.Itoa(holder) {
		t.Errorf("child GPU_LOCK_HELD=%s, owner record names %d: child's Inherited check fails", got, holder)
	}
}

// TestReproEmptyOwnerNeverReclaimed tests Defect 5:
// An empty or unparsable owner file older than 15s must be reclaimed.
func TestReproEmptyOwnerNeverReclaimed(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "gpu.lock")
	if err := os.Mkdir(dir, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "owner"), nil, 0644); err != nil {
		t.Fatal(err)
	}
	old := time.Now().Add(-time.Hour)
	os.Chtimes(filepath.Join(dir, "owner"), old, old)
	os.Chtimes(dir, old, old)

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	h, err := Acquire(ctx, WithDir(dir), WithPollInterval(10*time.Millisecond))
	if err != nil {
		t.Fatalf("Acquire with hour-old empty owner file: %v (want reclaim)", err)
	}
	h.Release()
}

// TestFreshEmptyOwnerNotReclaimed tests that an empty owner file created recently
// (within the 15s grace window) is NOT prematurely reclaimed.
func TestFreshEmptyOwnerNotReclaimed(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "gpu.lock")
	if err := os.Mkdir(dir, 0755); err != nil {
		t.Fatal(err)
	}
	ownerPath := filepath.Join(dir, "owner")
	if err := os.WriteFile(ownerPath, nil, 0644); err != nil {
		t.Fatal(err)
	}
	now := time.Now()
	os.Chtimes(ownerPath, now, now)

	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	h, err := Acquire(ctx, WithDir(dir), WithPollInterval(10*time.Millisecond))
	if err == nil {
		h.Release()
		t.Fatal("fresh empty owner file was prematurely reclaimed: expected timeout waiting for grace period")
	}
}

// TestReclaimTOCTOURaceClosed tests Defect 6:
// Reclaim must be atomic. If another process displaces or acquires a fresh lock
// during reclaim, the fresh lock must be restored and not deleted.
func TestReclaimTOCTOURaceClosed(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "gpu.lock")
	if err := os.Mkdir(dir, 0755); err != nil {
		t.Fatal(err)
	}

	deadPID := 99999999
	for isAlive(deadPID) {
		deadPID++
	}
	deadOwner := Owner{
		PID:   deadPID,
		Host:  shortHostname(),
		Label: "dead-initial",
		Epoch: time.Now().Unix() - 100,
		At:    time.Now().Add(-100 * time.Second),
	}
	deadOwnerBytes := []byte(deadOwner.String())
	if err := os.WriteFile(filepath.Join(dir, "owner"), deadOwnerBytes, 0644); err != nil {
		t.Fatal(err)
	}

	livePID := os.Getppid()
	liveOwner := Owner{
		PID:   livePID,
		Host:  shortHostname(),
		Label: "live-concurrent",
		Epoch: time.Now().Unix(),
		At:    time.Now(),
	}

	// Hook triggers inside reclaim() after dead owner read and before aside move.
	// Hook simulates process A winning the race by removing dead lock and acquiring fresh lock.
	reclaimHook = func() {
		reclaimHook = nil // Run hook only once
		_ = os.Remove(filepath.Join(dir, "owner"))
		_ = os.Remove(dir)
		_ = os.Mkdir(dir, 0755)
		_ = os.WriteFile(filepath.Join(dir, "owner"), []byte(liveOwner.String()), 0644)
	}
	defer func() { reclaimHook = nil }()

	// Reclaimer process B tries to reclaim based on deadOwnerBytes
	reclaimed, recErr := reclaim(dir, deadOwnerBytes)
	if recErr != nil {
		t.Fatalf("reclaim returned error: %v", recErr)
	}
	if reclaimed {
		t.Fatal("live lock was destroyed by concurrent stale reclaim: expected reclaim to fail")
	}

	// Verify that live lock directory was restored intact
	data, err := os.ReadFile(filepath.Join(dir, "owner"))
	if err != nil {
		t.Fatalf("live lock was lost: %v", err)
	}
	restored, err := ParseOwner(data)
	if err != nil {
		t.Fatalf("failed to parse restored owner: %v", err)
	}
	if restored.PID != livePID || restored.Label != "live-concurrent" {
		t.Errorf("restored owner = %+v, want live PID %d", restored, livePID)
	}
}

// TestTwoDirsIndependentLocks tests Item 1:
// Holding dir a does not cause Acquire(b) to return a nested handle without creating b.
func TestTwoDirsIndependentLocks(t *testing.T) {
	a := filepath.Join(t.TempDir(), "a")
	b := filepath.Join(t.TempDir(), "b")
	ctx := context.Background()

	ha, err := Acquire(ctx, WithDir(a))
	if err != nil {
		t.Fatal(err)
	}
	defer ha.Release()

	hb, err := Acquire(ctx, WithDir(b))
	if err != nil {
		t.Fatal(err)
	}
	defer hb.Release()

	if hb.IsNested() {
		t.Errorf("expected Acquire(b) to be non-nested root handle, got nested")
	}

	data, err := os.ReadFile(filepath.Join(b, "owner"))
	if err != nil {
		t.Fatalf("b/owner unreadable: %v", err)
	}
	ownerB, err := ParseOwner(data)
	if err != nil {
		t.Fatalf("b/owner invalid: %v", err)
	}
	if ownerB.PID != os.Getpid() {
		t.Errorf("b/owner PID = %d, want %d", ownerB.PID, os.Getpid())
	}

	if err := hb.Release(); err != nil {
		t.Fatalf("hb.Release failed: %v", err)
	}

	if _, err := os.Stat(b); !os.IsNotExist(err) {
		t.Errorf("b still exists after releasing hb")
	}
	if _, err := os.Stat(a); err != nil {
		t.Errorf("a missing after releasing hb: %v", err)
	}
}

// TestAcquireAtomicVisibility tests Item 2:
// The lock directory is never visible without a parsable owner file.
func TestAcquireAtomicVisibility(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "atomic-lock")
	ctx := context.Background()

	stop := make(chan struct{})
	errCh := make(chan error, 1)

	go func() {
		for {
			select {
			case <-stop:
				return
			default:
				if fi, err := os.Stat(dir); err == nil && fi.IsDir() {
					ownerPath := filepath.Join(dir, "owner")
					data, err := os.ReadFile(ownerPath)
					if err != nil {
						if _, statErr := os.Stat(dir); os.IsNotExist(statErr) {
							// dir was removed concurrently (e.g. by Release) between Stat and ReadFile.
							continue
						}
						select {
						case errCh <- fmt.Errorf("lock dir visible without owner file: %w", err):
						default:
						}
						return
					}
					if _, err := ParseOwner(data); err != nil {
						select {
						case errCh <- fmt.Errorf("lock dir visible with unparsable owner: %w", err):
						default:
						}
						return
					}
				}
			}
		}
	}()

	for i := 0; i < 30; i++ {
		h, err := Acquire(ctx, WithDir(dir))
		if err != nil {
			t.Fatal(err)
		}
		if err := h.Release(); err != nil {
			t.Fatal(err)
		}
	}
	close(stop)

	select {
	case err := <-errCh:
		t.Fatalf("atomic acquire/release visibility violation: %v", err)
	default:
	}
}

// TestReleaseRefusesDisplacedOwner tests Item 3:
// Release deletes the lock only if the owner still names this holder.
func TestReleaseRefusesDisplacedOwner(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "lock")
	ctx := context.Background()

	h, err := Acquire(ctx, WithDir(dir), WithLabel("original-holder"))
	if err != nil {
		t.Fatal(err)
	}

	// Tamper with owner file while held
	displacedOwner := Owner{
		PID:   os.Getpid() + 1000,
		Host:  shortHostname(),
		Label: "foreign-or-displaced",
		Epoch: time.Now().Unix(),
		At:    time.Now(),
	}
	ownerPath := filepath.Join(dir, "owner")
	if err := os.WriteFile(ownerPath, []byte(displacedOwner.String()), 0644); err != nil {
		t.Fatal(err)
	}

	// Release must return an error because owner changed
	relErr := h.Release()
	if relErr == nil {
		t.Fatal("Release succeeded despite displaced owner; want error")
	}

	// Verify that displaced owner and directory remain untouched
	data, err := os.ReadFile(ownerPath)
	if err != nil {
		t.Fatalf("owner file missing after rejected Release: %v", err)
	}
	parsed, err := ParseOwner(data)
	if err != nil {
		t.Fatalf("owner file unparsable: %v", err)
	}
	if parsed.PID != displacedOwner.PID || parsed.Label != displacedOwner.Label {
		t.Errorf("displaced owner modified: got %+v, want %+v", parsed, displacedOwner)
	}
	_ = os.Remove(ownerPath)
	_ = os.Remove(dir)
}

// TestReclaimRestoreFailureAbortsAcquire tests Item 4:
// If moving a displaced live lock back during reclaim fails, Acquire aborts
// and returns an error naming the aside path.
func TestReclaimRestoreFailureAbortsAcquire(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "gpu.lock")
	if err := os.Mkdir(dir, 0755); err != nil {
		t.Fatal(err)
	}

	deadPID := 99999999
	for isAlive(deadPID) {
		deadPID++
	}
	deadOwner := Owner{
		PID:   deadPID,
		Host:  shortHostname(),
		Label: "dead-to-reclaim",
		Epoch: time.Now().Unix() - 100,
		At:    time.Now().Add(-100 * time.Second),
	}
	if err := os.WriteFile(filepath.Join(dir, "owner"), []byte(deadOwner.String()), 0644); err != nil {
		t.Fatal(err)
	}
	livePID := os.Getppid()
	liveOwner := Owner{
		PID:   livePID,
		Host:  shortHostname(),
		Label: "live-concurrent",
		Epoch: time.Now().Unix(),
		At:    time.Now(),
	}

	reclaimHook = func() {
		reclaimHook = nil
		// Before aside move: replace owner with liveOwner so matches == false
		_ = os.WriteFile(filepath.Join(dir, "owner"), []byte(liveOwner.String()), 0644)
	}
	restoreHook = func(aside string) {
		restoreHook = nil
		// Before restore: occupy dir so os.Rename(aside, dir) fails
		_ = os.Mkdir(dir, 0755)
		_ = os.WriteFile(filepath.Join(dir, "occupier"), []byte("blocker"), 0644)
	}
	defer func() {
		reclaimHook = nil
		restoreHook = nil
	}()

	ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
	defer cancel()

	_, err := Acquire(ctx, WithDir(dir), WithPollInterval(10*time.Millisecond))
	if err == nil {
		t.Fatal("Acquire succeeded despite failed lock restore; want error")
	}
	if !strings.Contains(err.Error(), "could not restore live lock displaced during reclaim") ||
		!strings.Contains(err.Error(), ".gpu-lock-stale.") {
		t.Fatalf("Acquire error = %v, want error naming displaced aside path (.gpu-lock-stale.)", err)
	}
}

// TestInteropReferenceScript tests mutual exclusion and dead-lock reclaim
// interoperability between cudalock and the reference gpu-lock bash script.
//
// The reference script is resolved from the GPU_LOCK_SCRIPT environment variable,
// or by searching for "gpu-lock" on PATH. If neither is found, the test skips
// unless CUDALOCK_INTEROP_STRICT=1 is set, in which case it fails.
func TestInteropReferenceScript(t *testing.T) {
	scriptPath := os.Getenv("GPU_LOCK_SCRIPT")
	if scriptPath == "" {
		if p, err := exec.LookPath("gpu-lock"); err == nil {
			scriptPath = p
		}
	}
	if scriptPath == "" {
		if os.Getenv("CUDALOCK_INTEROP_STRICT") == "1" {
			t.Fatal("gpu-lock script not found (GPU_LOCK_SCRIPT unset, gpu-lock not on PATH)")
		}
		t.Skip("gpu-lock script not found (GPU_LOCK_SCRIPT unset, gpu-lock not on PATH)")
	}
	if _, err := os.Stat(scriptPath); err != nil {
		if os.Getenv("CUDALOCK_INTEROP_STRICT") == "1" {
			t.Fatalf("reference script missing at %s: %v", scriptPath, err)
		}
		t.Skipf("reference script not found at %s: skipping interop test", scriptPath)
	}
	bashPath, err := exec.LookPath("bash")
	if err != nil {
		if os.Getenv("CUDALOCK_INTEROP_STRICT") == "1" {
			t.Fatalf("bash not found on PATH: %v", err)
		}
		t.Skipf("bash not found: skipping interop test")
	}

	tmp := t.TempDir()
	lockDir := filepath.Join(tmp, "lock")

	// Subtest 1: Reference script blocks when cudalock holds the lock
	t.Run("ReferenceScriptBlocksOnCudalock", func(t *testing.T) {
		h, err := Acquire(context.Background(), WithDir(lockDir), WithLabel("go-holder"))
		if err != nil {
			t.Fatalf("Acquire failed: %v", err)
		}
		defer h.Release()

		// Run gpu-lock with MAXWAIT=1s
		cmd := exec.Command(bashPath, scriptPath, "true")
		cmd.Env = append(os.Environ(),
			"GPU_LOCKDIR="+lockDir,
			"GPU_LOCK_MAXWAIT=1",
			"GPU_LOCK_POLL=1",
		)
		out, err := cmd.CombinedOutput()
		if err == nil {
			t.Fatalf("reference script acquired while cudalock held lock; output: %s", string(out))
		}
		var exitErr *exec.ExitError
		if errors.As(err, &exitErr) && exitErr.ExitCode() != 75 {
			t.Errorf("reference script exit code = %d, want 75 (timeout); output: %s", exitErr.ExitCode(), string(out))
		}
	})

	// Subtest 2: cudalock blocks when reference script holds the lock
	t.Run("CudalockBlocksOnReferenceScript", func(t *testing.T) {
		scriptLockDir := filepath.Join(tmp, "script-lock")
		cmd := exec.Command(bashPath, scriptPath, "sleep", "0.5")
		cmd.Env = append(os.Environ(),
			"GPU_LOCKDIR="+scriptLockDir,
			"GPU_LOCK_POLL=1",
		)
		if err := cmd.Start(); err != nil {
			t.Fatalf("failed to start reference script: %v", err)
		}
		defer cmd.Wait()

		// Wait briefly for script to acquire
		var scriptAcquired bool
		for i := 0; i < 50; i++ {
			if _, err := os.Stat(filepath.Join(scriptLockDir, "owner")); err == nil {
				scriptAcquired = true
				break
			}
			time.Sleep(10 * time.Millisecond)
		}
		if !scriptAcquired {
			t.Fatal("reference script did not acquire in time")
		}

		ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
		defer cancel()
		_, err := Acquire(ctx, WithDir(scriptLockDir), WithPollInterval(20*time.Millisecond))
		if err == nil {
			t.Fatal("cudalock acquired while reference script was holding lock")
		}
		if !errors.Is(err, context.DeadlineExceeded) && !strings.Contains(err.Error(), "context deadline exceeded") {
			t.Errorf("unexpected error: %v", err)
		}

		// Wait for script to finish
		_ = cmd.Wait()

		// Now cudalock should acquire cleanly
		h, err := Acquire(context.Background(), WithDir(scriptLockDir), WithLabel("go-after-script"))
		if err != nil {
			t.Fatalf("cudalock failed to acquire after script release: %v", err)
		}
		_ = h.Release()
	})

	// Subtest 3: cudalock reclaims dead lock left by reference script
	t.Run("CudalockReclaimsReferenceDeadLock", func(t *testing.T) {
		deadDir := filepath.Join(tmp, "dead-script-lock")
		if err := os.MkdirAll(deadDir, 0755); err != nil {
			t.Fatal(err)
		}
		deadPID := 99999999
		for isAlive(deadPID) {
			deadPID++
		}
		deadScriptOwner := fmt.Sprintf("pid=%d host=%s label=dead-bash token=xyz epoch=%d at=%s lane=batch\n",
			deadPID, shortHostname(), time.Now().Unix()-100, time.Now().Add(-100*time.Second).Format(time.RFC3339))
		if err := os.WriteFile(filepath.Join(deadDir, "owner"), []byte(deadScriptOwner), 0644); err != nil {
			t.Fatal(err)
		}

		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		h, err := Acquire(ctx, WithDir(deadDir), WithLabel("cudalock-reclaimer"))
		if err != nil {
			t.Fatalf("cudalock failed to reclaim reference script dead lock: %v", err)
		}
		_ = h.Release()
	})

	// Subtest 4: reference script reclaims dead lock left by cudalock
	t.Run("ReferenceScriptReclaimsCudalockDeadLock", func(t *testing.T) {
		deadDir := filepath.Join(tmp, "dead-cudalock-lock")
		if err := os.MkdirAll(deadDir, 0755); err != nil {
			t.Fatal(err)
		}
		deadPID := 99999999
		for isAlive(deadPID) {
			deadPID++
		}
		deadOwner := Owner{
			PID:   deadPID,
			Host:  shortHostname(),
			Label: "dead-cudalock",
			Epoch: time.Now().Unix() - 100,
			At:    time.Now().Add(-100 * time.Second),
		}
		if err := os.WriteFile(filepath.Join(deadDir, "owner"), []byte(deadOwner.String()), 0644); err != nil {
			t.Fatal(err)
		}

		cmd := exec.Command(bashPath, scriptPath, "true")
		cmd.Env = append(os.Environ(),
			"GPU_LOCKDIR="+deadDir,
			"GPU_LOCK_MAXWAIT=3",
			"GPU_LOCK_POLL=1",
		)
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("reference script failed to reclaim cudalock dead lock: %v; output: %s", err, string(out))
		}
	})
}
