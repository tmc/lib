//go:build darwin || linux

package cudalock

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestSuccessorInheritedChild(t *testing.T) {
	dir, mode := os.Getenv("GPU_LOCKDIR"), os.Getenv("SUCCESSOR_MODE")
	if !filepath.IsAbs(dir) || !strings.Contains(dir, "TestSuccessor") {
		t.Fatal("explicit task-private inherited fixture required")
	}
	if mode == "verify-blocked" {
		owner, err := Inherited(os.Getenv)
		if owner.PID != 0 || !errors.Is(err, ErrNotInherited) || !guardBusy(err) {
			t.Fatalf("busy public Inherited admitted or misclassified: %+v %v", owner, err)
		}
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 70*time.Millisecond)
	defer cancel()
	h, err := Acquire(ctx, WithDir(dir), WithPollInterval(time.Millisecond))
	if mode == "acquire-blocked" {
		if h != nil {
			h.Release()
			t.Fatal("inherited public Acquire bypassed namespace guard")
		}
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("inherited refusal: %v", err)
		}
		return
	}
	if mode != "admit" || err != nil || h == nil || !h.IsNested() || strconv.Itoa(h.Owner().PID) != os.Getenv("GPU_LOCK_HELD") {
		t.Fatalf("fresh inherited admission: %v %v mode=%s", h, err, mode)
	}
	owner, err := Inherited(os.Getenv)
	if err != nil || owner.PID != h.Owner().PID {
		t.Fatalf("fresh public Inherited: %+v %v", owner, err)
	}
	if err := h.Release(); err != nil {
		t.Fatal(err)
	}
}

func successorInheritedChild(t *testing.T, dir, mode string) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestSuccessorInheritedChild$", "-test.v")
	for _, value := range os.Environ() {
		if !strings.HasPrefix(value, "GPU_LOCK_") && !strings.HasPrefix(value, "SUCCESSOR_MODE=") {
			cmd.Env = append(cmd.Env, value)
		}
	}
	cmd.Env = append(cmd.Env, "GPU_LOCKDIR="+dir, "GPU_LOCK_HELD="+strconv.Itoa(os.Getpid()), "SUCCESSOR_MODE="+mode)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("owned inherited child (%s): %v %s", mode, err, out)
	}
}

func TestSuccessorInheritedAdmissionTransaction(t *testing.T) {
	dir := privateGuardDir(t)
	h, err := Acquire(context.Background(), WithDir(dir))
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := h.Release(); err != nil {
			t.Error(err)
		}
	}()
	g, err := openNamespace(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer g.file.Close()
	if err := g.lock(context.Background(), time.Millisecond); err != nil {
		t.Fatal(err)
	}
	successorInheritedChild(t, dir, "acquire-blocked")
	successorInheritedChild(t, dir, "verify-blocked")
	if err := guardUnlock(g.file); err != nil {
		t.Fatal(err)
	}
	successorInheritedChild(t, dir, "admit")
	data, err := os.ReadFile(filepath.Join(dir, "owner"))
	if err != nil || string(data) != h.Owner().String() {
		t.Fatal("inherited reference release changed ancestor owner")
	}
}

func TestSuccessorRealPermissionRetirementRestoresOwner(t *testing.T) {
	dir := privateGuardDir(t)
	h, err := Acquire(context.Background(), WithDir(dir))
	if err != nil {
		t.Fatal(err)
	}
	blocked := filepath.Join(dir, "blocked")
	if err := os.Mkdir(blocked, 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(blocked, "retained"), []byte("cannot remove from nonwritable directory"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(blocked, 0500); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		os.Chmod(blocked, 0700)
		if err := h.Release(); err != nil {
			t.Error("owned permission fixture cleanup:", err)
		}
	})
	if err := h.Release(); err == nil || !errors.Is(err, os.ErrPermission) {
		t.Fatalf("real permission fixture did not fail retirement: %v", err)
	}
	data, err := os.ReadFile(filepath.Join(dir, "owner"))
	if err != nil || string(data) != h.Owner().String() || h.hold.retired != "" || h.released {
		t.Fatalf("failed retirement did not retain exact canonical authority: %q %v", data, err)
	}
	child := startGuardChild(t, dir, "try")
	child.send(t, "go")
	child.want(t, "blocked")
	child.wait(t, false)
	nested, err := Acquire(context.Background(), WithDir(dir))
	if err != nil || !nested.IsNested() || nested.Owner().PID != h.Owner().PID {
		t.Fatalf("restored local reference unverified: %v %v", nested, err)
	}
	if err := nested.Release(); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(blocked, 0700); err != nil {
		t.Fatal(err)
	}
	if err := h.Release(); err != nil {
		t.Fatal("explicit retirement retry:", err)
	}
}

func TestSuccessorPendingRetirementRefusesAndPreservesNewOwner(t *testing.T) {
	dir := privateGuardDir(t)
	h, err := Acquire(context.Background(), WithDir(dir))
	if err != nil {
		t.Fatal(err)
	}
	g, err := openNamespace(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer g.file.Close()
	if err := g.lock(context.Background(), time.Millisecond); err != nil {
		t.Fatal(err)
	}
	// Execute the same private root-release helper with a per-call fault only
	// at final empty-directory removal. All owner/namespace operations are real;
	// no global process state or public acquisition result is manufactured.
	finalRemove := errors.New("final retired-directory removal failed")
	h.mu.Lock()
	processMu.Lock()
	err = h.releaseRoot(g, h.hold, func(string) error { return finalRemove })
	processMu.Unlock()
	h.mu.Unlock()
	if !errors.Is(err, finalRemove) || h.hold.retired == "" || !h.hold.ownerRemoved || h.released {
		t.Fatalf("partial retirement custody lost: %v hold=%+v", err, h.hold)
	}
	if err := guardUnlock(g.file); err != nil {
		t.Fatal(err)
	}
	child := startGuardChild(t, dir, "hold")
	child.send(t, "go")
	child.want(t, "acquired")
	before, err := os.ReadFile(filepath.Join(dir, "owner"))
	if err != nil {
		t.Fatal(err)
	}
	owner, err := ParseOwner(before)
	if err != nil || owner.PID != child.cmd.Process.Pid || owner.PID == h.Owner().PID {
		t.Fatal("independent public root not established")
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if other, err := Acquire(ctx, WithDir(dir)); other != nil || err == nil || !strings.Contains(err.Error(), "retirement cleanup pending") {
		t.Fatalf("pending retirement authorized nested work: %v %v", other, err)
	}
	env := mapEnv(map[string]string{"GPU_LOCKDIR": dir, "GPU_LOCK_HELD": strconv.Itoa(owner.PID)})
	if _, err := Inherited(env); !errors.Is(err, ErrNotInherited) {
		t.Fatalf("public verification bypassed known pending retirement: %v", err)
	}
	// A replacement at the old aside name is not ours to clean. The retry
	// must refuse it while leaving both that snapshot and C's public root.
	aside := h.hold.retired
	if err := os.Rename(aside, aside+".saved"); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(aside, 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(aside, "owner"), []byte("different aside owner\n"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := h.Release(); err == nil {
		t.Fatal("retry adopted a replacement aside inode")
	}
	different, err := os.ReadFile(filepath.Join(aside, "owner"))
	if err != nil || string(different) != "different aside owner\n" {
		t.Fatal("retry removed a replacement aside owner")
	}
	if err := os.Rename(aside, aside+".other"); err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(aside+".saved", aside); err != nil {
		t.Fatal(err)
	}
	if err := h.Release(); err != nil {
		t.Fatal("pending cleanup retry:", err)
	}
	after, err := os.ReadFile(filepath.Join(dir, "owner"))
	if err != nil || string(after) != string(before) {
		t.Fatal("old retirement retry changed independent canonical owner")
	}
	child.send(t, "release")
	child.want(t, "released")
	child.wait(t, false)
}

func TestSuccessorRestoreRequiresExactSnapshotAndVacancy(t *testing.T) {
	for _, change := range []string{"none", "owner", "inode", "occupied", "removed"} {
		t.Run(change, func(t *testing.T) {
			dir := privateGuardDir(t)
			g, err := openNamespace(dir)
			if err != nil {
				t.Fatal(err)
			}
			defer g.file.Close()
			if err := g.lock(context.Background(), time.Millisecond); err != nil {
				t.Fatal(err)
			}
			defer guardUnlock(g.file)
			aside := filepath.Join(filepath.Dir(dir), "owned-aside")
			if err := os.Mkdir(aside, 0700); err != nil {
				t.Fatal(err)
			}
			const original = "exact original owner bytes\n"
			if err := os.WriteFile(filepath.Join(aside, "owner"), []byte(original), 0600); err != nil {
				t.Fatal(err)
			}
			info, err := os.Lstat(aside)
			if err != nil {
				t.Fatal(err)
			}
			hold := &dirHold{ownerData: original, retired: aside, retiredInfo: info}
			switch change {
			case "owner":
				err = os.WriteFile(filepath.Join(aside, "owner"), []byte("newer owner\n"), 0600)
			case "inode":
				err = os.Rename(aside, aside+".original")
				if err == nil {
					err = os.Mkdir(aside, 0700)
				}
			case "occupied":
				err = os.Mkdir(dir, 0700)
				if err == nil {
					err = os.WriteFile(filepath.Join(dir, "owner"), []byte("new canonical owner\n"), 0600)
				}
			case "removed":
				err = os.Remove(filepath.Join(aside, "owner"))
				hold.ownerRemoved = true
			}
			if err != nil {
				t.Fatal(err)
			}
			err = restoreRetired(g, hold)
			if (err == nil) != (change == "none") {
				t.Fatalf("snapshot change %s: %v", change, err)
			}
			if change == "occupied" {
				data, err := os.ReadFile(filepath.Join(dir, "owner"))
				if err != nil || string(data) != "new canonical owner\n" {
					t.Fatal("restore overwrote another canonical owner")
				}
			}
		})
	}
}

func TestSuccessorShellDeadlineHelper(t *testing.T) {
	data, err := os.ReadFile("reference/gpu-lock.sh")
	if err != nil {
		t.Fatal(err)
	}
	_, rest, ok := strings.Cut(string(data), "check_wait() {")
	if !ok {
		t.Fatal("actual shell deadline helper missing")
	}
	body, _, ok := strings.Cut(rest, "\n}")
	if !ok {
		t.Fatal("actual shell deadline helper incomplete")
	}
	for _, expired := range []bool{false, true} {
		t.Run(fmt.Sprint(expired), func(t *testing.T) {
			// Run only the actual pure Bash deadline helper. This does not
			// execute or spoof the Linux script, flock or a CUDA workload.
			start := "WAIT_STARTED=$SECONDS"
			if expired {
				start = "WAIT_STARTED=$((SECONDS - 2))"
			}
			cmd := exec.Command("/bin/bash", "-c", "check_wait() {"+body+"\n}\n"+start+"\ncheck_wait 1\nprintf admitted")
			out, err := cmd.CombinedOutput()
			if expired {
				var exit *exec.ExitError
				if !errors.As(err, &exit) || exit.ExitCode() != 75 || strings.Contains(string(out), "admitted") {
					t.Fatalf("expired admission: %v %s", err, out)
				}
			} else if err != nil || string(out) != "admitted" {
				t.Fatalf("fresh admission: %v %s", err, out)
			}
		})
	}
}

// Err reports the real context value. The third check is acquireGuarded's
// entry check after namespace-guard admission, before processMu is taken.
type deadlineAdmissionContext struct {
	context.Context
	checks atomic.Int32
	ready  chan struct{}
}

func (c *deadlineAdmissionContext) Err() error {
	err := c.Context.Err()
	if c.checks.Add(1) == 3 {
		close(c.ready)
	}
	return err
}

func TestSuccessorDeadlineAfterProcessWait(t *testing.T) {
	for _, mode := range []string{"nested", "inherited", "new root", "new no-reclaim"} {
		t.Run(mode, func(t *testing.T) {
			dir := privateGuardDir(t)
			var root *Handle
			var ownerData []byte
			if mode == "nested" {
				var err error
				root, err = Acquire(t.Context(), WithDir(dir))
				if err != nil {
					t.Fatal(err)
				}
				defer func() {
					if err := root.Release(); err != nil {
						t.Error(err)
					}
				}()
			}
			if mode == "inherited" {
				t.Setenv("GPU_LOCKDIR", dir)
				t.Setenv("GPU_LOCK_HELD", strconv.Itoa(os.Getpid()))
				if err := os.Mkdir(dir, 0755); err != nil {
					t.Fatal(err)
				}
				owner := Owner{PID: os.Getpid(), Host: shortHostname(), Label: "owned-inherited-fixture", Epoch: time.Now().Unix(), At: time.Now()}
				if err := os.WriteFile(filepath.Join(dir, "owner"), []byte(owner.String()), 0644); err != nil {
					t.Fatal(err)
				}
			}
			if mode == "nested" || mode == "inherited" {
				var err error
				ownerData, err = os.ReadFile(filepath.Join(dir, "owner"))
				if err != nil {
					t.Fatal(err)
				}
			}
			raw, cancel := context.WithTimeout(t.Context(), 150*time.Millisecond)
			defer cancel()
			ctx := &deadlineAdmissionContext{Context: raw, ready: make(chan struct{})}
			type result struct {
				h   *Handle
				err error
			}
			done := make(chan result, 1)
			opts := []Option{WithDir(dir), WithPollInterval(time.Millisecond)}
			if mode == "new no-reclaim" {
				opts = append(opts, WithNoReclaim())
			}
			processMu.Lock()
			go func() {
				h, err := Acquire(ctx, opts...)
				done <- result{h, err}
			}()
			reached := false
			select {
			case <-ctx.ready:
				reached = true
			case <-time.After(time.Second):
			}
			<-raw.Done()
			processMu.Unlock()
			var got result
			select {
			case got = <-done:
			case <-time.After(time.Second):
				t.Fatal("acquisition did not finish after process mutex was released")
			}
			if got.h != nil {
				if err := got.h.Release(); err != nil {
					t.Error(err)
				}
			}
			if !reached {
				t.Fatal("acquisition did not reach the guarded entry check")
			}
			if got.h != nil || !errors.Is(got.err, context.DeadlineExceeded) {
				t.Fatalf("expired admission returned handle=%v error=%v", got.h, got.err)
			}
			processMu.Lock()
			hold := processHolds[dir]
			valid := hold == nil
			if root != nil {
				valid = hold == root.hold && hold.count == 1
			}
			processMu.Unlock()
			if !valid {
				t.Fatal("expired admission changed process ownership")
			}
			if ownerData != nil {
				data, err := os.ReadFile(filepath.Join(dir, "owner"))
				if err != nil || string(data) != string(ownerData) {
					t.Fatalf("expired admission changed owner: %v", err)
				}
			} else if _, err := os.Lstat(dir); !errors.Is(err, os.ErrNotExist) {
				t.Fatalf("expired admission published a root: %v", err)
			}
		})
	}
}

// Observe the actual published owner and cancel the real context at that point.
// prepare only creates task-owned disk contents for the permission-failure arm.
type publicationCancelContext struct {
	context.Context
	cancel  context.CancelFunc
	dir     string
	once    sync.Once
	prepare func() error
	err     error
}

func (c *publicationCancelContext) Err() error {
	if _, err := os.Lstat(filepath.Join(c.dir, "owner")); err == nil {
		c.once.Do(func() {
			if c.prepare != nil {
				c.err = c.prepare()
			}
			c.cancel()
		})
	}
	return c.Context.Err()
}

func TestSuccessorCanceledPublication(t *testing.T) {
	for _, noReclaim := range []bool{false, true} {
		for _, failCleanup := range []bool{false, true} {
			t.Run(fmt.Sprintf("no-reclaim=%v/cleanup-error=%v", noReclaim, failCleanup), func(t *testing.T) {
				dir := privateGuardDir(t)
				t.Setenv("GPU_LOCKDIR", dir)
				t.Setenv("GPU_LOCK_HELD", "")
				raw, cancel := context.WithCancel(t.Context())
				defer cancel()
				ctx := &publicationCancelContext{Context: raw, cancel: cancel, dir: dir}
				blocked := filepath.Join(dir, "blocked")
				if failCleanup {
					ctx.prepare = func() error {
						if err := os.Mkdir(blocked, 0755); err != nil {
							return err
						}
						if err := os.WriteFile(filepath.Join(blocked, "retained"), []byte("owned permission fixture"), 0600); err != nil {
							return err
						}
						return os.Chmod(blocked, 0555)
					}
					defer os.Chmod(blocked, 0755)
				}
				opts := []Option{WithDir(dir), WithPollInterval(time.Millisecond)}
				if noReclaim {
					opts = append(opts, WithNoReclaim())
				}
				h, err := Acquire(ctx, opts...)
				if ctx.err != nil {
					t.Fatal(ctx.err)
				}
				if h != nil || !errors.Is(err, context.Canceled) {
					t.Fatalf("postpublication cancellation admitted work: %v %v", h, err)
				}
				var cleanup io.Closer
				if failCleanup {
					if !errors.Is(err, os.ErrPermission) || !errors.As(err, &cleanup) {
						t.Fatalf("cleanup error lost cause or retriable owner: %v", err)
					}
					defer func() {
						_ = os.Chmod(blocked, 0755)
						_ = cleanup.Close()
					}()
					data, readErr := os.ReadFile(filepath.Join(dir, "owner"))
					if readErr != nil {
						t.Fatal("failed cleanup did not restore its original owner:", readErr)
					}
					if other, err := Acquire(t.Context(), WithDir(dir)); other != nil || err == nil {
						t.Fatalf("cleanup-only owner admitted work: %v %v", other, err)
					}
					if _, err := Inherited(os.Getenv); !errors.Is(err, ErrNotInherited) {
						t.Fatalf("cleanup-only owner verified for work: %v", err)
					}
					after, readErr := os.ReadFile(filepath.Join(dir, "owner"))
					if readErr != nil || string(after) != string(data) {
						t.Fatal("cleanup refusal changed owner")
					}
					// Pass the actual environment left by public Acquire unchanged.
					cmdCtx, stop := context.WithTimeout(t.Context(), 3*time.Second)
					cmd := exec.CommandContext(cmdCtx, os.Args[0], "-test.run=^TestSuccessorCanceledAuthorityChild$", "-test.v")
					cmd.Env = os.Environ()
					out, childErr := cmd.CombinedOutput()
					stop()
					if childErr != nil {
						t.Errorf("canceled publication authorized descendant: %v %s", childErr, out)
					}
					if err := os.Chmod(blocked, 0755); err != nil {
						t.Fatal(err)
					}
					if err := cleanup.Close(); err != nil {
						t.Fatal("explicit canceled-publication cleanup retry:", err)
					}
					if err := cleanup.Close(); err != nil {
						t.Fatal("cleanup retry was not idempotent:", err)
					}
				} else if errors.As(err, &cleanup) {
					t.Fatal("successful automatic cleanup retained an unnecessary owner")
				}
				if _, err := os.Lstat(dir); !errors.Is(err, os.ErrNotExist) {
					t.Fatalf("canceled publication remained after successful cleanup: %v", err)
				}
				processMu.Lock()
				hold := processHolds[dir]
				processMu.Unlock()
				if hold != nil || os.Getenv("GPU_LOCK_HELD") != "" {
					t.Fatal("successful canceled-publication cleanup retained authority")
				}
			})
		}
	}
}

func TestSuccessorCanceledCleanupPreservesNewRoot(t *testing.T) {
	dir := privateGuardDir(t)
	h, err := Acquire(t.Context(), WithDir(dir))
	if err != nil {
		t.Fatal(err)
	}
	g, err := openNamespace(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer g.file.Close()
	if err := g.lock(t.Context(), time.Millisecond); err != nil {
		t.Fatal(err)
	}
	// Reach the same cancellation helper used by Acquire with a per-call fault
	// at final removal, after the actual owner file has already been removed.
	finalRemove := errors.New("canceled publication final removal failed")
	err = cancelAdmission(h, g, context.Canceled, func(string) error { return finalRemove })
	if unlockErr := guardUnlock(g.file); unlockErr != nil {
		t.Fatal(unlockErr)
	}
	var cleanup io.Closer
	if !errors.Is(err, context.Canceled) || !errors.Is(err, finalRemove) || !errors.As(err, &cleanup) {
		t.Fatalf("pending cancellation lost cleanup owner or error identities: %v", err)
	}
	defer cleanup.Close()
	if h.hold.retired == "" || !h.hold.ownerRemoved || !h.hold.cleanupOnly || h.released {
		t.Fatal("canceled cleanup lost exact pending custody")
	}
	child := startGuardChild(t, dir, "hold")
	child.send(t, "go")
	child.want(t, "acquired")
	before, err := os.ReadFile(filepath.Join(dir, "owner"))
	if err != nil {
		t.Fatal(err)
	}
	if other, err := Acquire(t.Context(), WithDir(dir)); other != nil || err == nil {
		t.Fatalf("old canceled owner admitted while child holds root: %v %v", other, err)
	}
	if err := cleanup.Close(); err != nil {
		t.Fatal("pending cancellation cleanup retry:", err)
	}
	if err := cleanup.Close(); err != nil {
		t.Fatal("duplicate cancellation cleanup:", err)
	}
	after, err := os.ReadFile(filepath.Join(dir, "owner"))
	if err != nil || string(after) != string(before) {
		t.Fatal("canceled cleanup retry changed newer canonical owner")
	}
	child.send(t, "release")
	child.want(t, "released")
	child.wait(t, false)
}

func TestSuccessorCanceledAuthorityChild(t *testing.T) {
	dir := os.Getenv("GPU_LOCKDIR")
	if !filepath.IsAbs(dir) || !strings.Contains(dir, "TestSuccessorCanceledPublication") {
		t.Fatal("explicit private canceled-publication fixture required")
	}
	owner, verifyErr := Inherited(os.Getenv)
	ctx, cancel := context.WithTimeout(t.Context(), 70*time.Millisecond)
	defer cancel()
	h, err := Acquire(ctx, WithDir(dir), WithPollInterval(time.Millisecond))
	if h != nil {
		_ = h.Release()
	}
	if owner.PID != 0 || !errors.Is(verifyErr, ErrNotInherited) || h != nil || !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("cleanup-only descendant admission: owner=%+v verification=%v handle=%v acquire=%v", owner, verifyErr, h, err)
	}
}

func TestSuccessorCanceledPublicationPreservesOtherAuthority(t *testing.T) {
	for _, mode := range []string{"preexisting", "newer"} {
		t.Run(mode, func(t *testing.T) {
			otherDir := privateGuardDir(t)
			t.Setenv("GPU_LOCKDIR", otherDir)
			t.Setenv("GPU_LOCK_HELD", "")
			var want string
			var change func() error
			if mode == "preexisting" {
				other, err := Acquire(t.Context(), WithDir(otherDir))
				if err != nil {
					t.Fatal(err)
				}
				defer func() {
					if err := other.Release(); err != nil {
						t.Error(err)
					}
				}()
				want = os.Getenv("GPU_LOCK_HELD")
				if want != strconv.Itoa(other.Owner().PID) {
					t.Fatal("successful public root did not export actual authority")
				}
			} else {
				child := startGuardChild(t, otherDir, "hold")
				child.send(t, "go")
				child.want(t, "acquired")
				defer func() {
					child.send(t, "release")
					child.want(t, "released")
					child.wait(t, false)
				}()
				data, err := os.ReadFile(filepath.Join(otherDir, "owner"))
				if err != nil {
					t.Fatal(err)
				}
				owner, err := ParseOwner(data)
				if err != nil || owner.PID != child.cmd.Process.Pid {
					t.Fatal("independent public root not established")
				}
				want = strconv.Itoa(owner.PID)
				change = func() error { return os.Setenv("GPU_LOCK_HELD", want) }
			}
			dir := privateGuardDir(t)
			t.Setenv("GPU_LOCKDIR", dir)
			raw, cancel := context.WithCancel(t.Context())
			defer cancel()
			ctx := &publicationCancelContext{Context: raw, cancel: cancel, dir: dir, prepare: change}
			h, err := Acquire(ctx, WithDir(dir))
			if h != nil || !errors.Is(err, context.Canceled) || ctx.err != nil {
				t.Fatalf("canceled publication: %v %v prepare=%v", h, err, ctx.err)
			}
			if got := os.Getenv("GPU_LOCK_HELD"); got != want {
				t.Fatalf("canceled publication changed other real authority: got %q want %q", got, want)
			}
			if _, err := os.Lstat(dir); !errors.Is(err, os.ErrNotExist) {
				t.Fatalf("canceled publication not cleaned: %v", err)
			}
		})
	}
}
