//go:build darwin || linux

package cudalock

import (
	"context"
	"errors"
	"go/ast"
	"go/parser"
	"go/token"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// Every admission explicitly names a private directory, including tests of the
// default-directory environment export and inherited-style admission.
func acquireCleanupDir(t *testing.T) string {
	t.Helper()
	dir := filepath.Join(t.TempDir(), "lock")
	t.Setenv("GPU_LOCKDIR", dir)
	t.Setenv("GPU_LOCK_HELD", "")
	return dir
}

type acquireCleanupFaults struct {
	t           *testing.T
	unlockErr   error
	closeErr    error
	unlockFails int
	closeFails  int
	unlocks     int
	closes      map[*os.File]int
	beforeClose func()
}

func (f *acquireCleanupFaults) ops() acquireGuardOps {
	f.closes = make(map[*os.File]int)
	return acquireGuardOps{
		unlock: func(file *os.File) error {
			f.unlocks++
			err := guardUnlock(file)
			if f.unlocks <= f.unlockFails {
				err = errors.Join(err, f.unlockErr)
			}
			return err
		},
		close: func(file *os.File) error {
			if f.beforeClose != nil {
				f.beforeClose()
			}
			f.closes[file]++
			if f.closes[file] != 1 {
				f.t.Error("a possibly consumed guard descriptor was closed twice")
			}
			err := file.Close()
			if len(f.closes) <= f.closeFails {
				err = errors.Join(err, f.closeErr)
			}
			return err
		},
	}
}

func acquireCleanupOwner(t *testing.T, h *Handle, err error) io.Closer {
	t.Helper()
	var cleanup io.Closer
	if h != nil || err == nil || !errors.As(err, &cleanup) {
		t.Fatalf("failed admission lost cleanup custody: handle=%v error=%v", h, err)
	}
	return cleanup
}

func acquireCleanupHold(t *testing.T, dir string, count int, cleanupOnly bool) *dirHold {
	t.Helper()
	processMu.Lock()
	defer processMu.Unlock()
	hold := processHolds[dir]
	if hold == nil || hold.count != count || hold.cleanupOnly != cleanupOnly {
		t.Fatalf("hold=%+v, want count=%d cleanupOnly=%v", hold, count, cleanupOnly)
	}
	return hold
}

func TestAcquireGuardCleanupSuccess(t *testing.T) {
	for _, noReclaim := range []bool{false, true} {
		name := "ordinary"
		if noReclaim {
			name = "no reclaim"
		}
		t.Run(name, func(t *testing.T) {
			dir := acquireCleanupDir(t)
			faults := &acquireCleanupFaults{t: t}
			faults.beforeClose = func() {
				if os.Getenv("GPU_LOCK_HELD") != "" {
					t.Error("successful root exported before guard Close")
				}
			}
			opts := []Option{WithDir(dir)}
			if noReclaim {
				opts = append(opts, WithNoReclaim())
			}
			h, err := acquire(t.Context(), faults.ops(), opts...)
			if err != nil || h == nil || h.IsNested() {
				t.Fatalf("successful admission: %v %v", h, err)
			}
			defer h.Release()
			acquireCleanupHold(t, dir, 1, false)
			if faults.unlocks != 1 || len(faults.closes) != 1 || os.Getenv("GPU_LOCK_HELD") != strconv.Itoa(h.owner.PID) {
				t.Fatal("successful admission did not finish guard cleanup before exporting")
			}
		})
	}
}

func TestAcquireGuardCleanupRoot(t *testing.T) {
	for _, mode := range []string{"unlock", "close", "both"} {
		t.Run(mode, func(t *testing.T) {
			dir := acquireCleanupDir(t)
			t.Setenv("GPU_LOCK_HELD", "unrelated authority")
			unlockErr := errors.New("injected admission unlock")
			closeErr := errors.New("injected admission close")
			faults := &acquireCleanupFaults{t: t, unlockErr: unlockErr, closeErr: closeErr}
			if mode != "close" {
				faults.unlockFails = 1
			}
			if mode != "unlock" {
				faults.closeFails = 1
			}
			faults.beforeClose = func() {
				if os.Getenv("GPU_LOCK_HELD") != "unrelated authority" {
					t.Error("root authority exported before guard Close succeeded")
				}
			}
			h, err := acquire(t.Context(), faults.ops(), WithDir(dir), WithNoReclaim())
			cleanup := acquireCleanupOwner(t, h, err)
			defer cleanup.Close()
			if errors.Is(err, unlockErr) != (mode != "close") || errors.Is(err, closeErr) != (mode != "unlock") {
				t.Fatalf("guard cleanup lost error identities: %v", err)
			}
			acquireCleanupHold(t, dir, 1, true)
			before, readErr := os.ReadFile(filepath.Join(dir, "owner"))
			if readErr != nil {
				t.Fatal(readErr)
			}
			other, otherErr := Acquire(t.Context(), WithDir(dir))
			var invented io.Closer
			if other != nil || otherErr == nil || errors.As(otherErr, &invented) {
				t.Fatalf("cleanup-only root admitted or invented custody: %v %v", other, otherErr)
			}
			after, readErr := os.ReadFile(filepath.Join(dir, "owner"))
			if readErr != nil || string(after) != string(before) {
				t.Fatal("refused admission changed retained publication")
			}
			if _, err := Inherited(func(name string) string {
				if name == "GPU_LOCK_HELD" {
					return strconv.Itoa(os.Getpid())
				}
				return os.Getenv(name)
			}); !errors.Is(err, ErrNotInherited) {
				t.Fatalf("cleanup-only root authorized work: %v", err)
			}
			if err := cleanup.Close(); err != nil {
				t.Fatal(err)
			}
			closes, unlocks := len(faults.closes), faults.unlocks
			if err := cleanup.Close(); err != nil || len(faults.closes) != closes || faults.unlocks != unlocks {
				t.Fatalf("completed cleanup repeated guard operations: %v", err)
			}
			if _, err := os.Lstat(dir); !errors.Is(err, os.ErrNotExist) {
				t.Fatalf("failed root publication remained: %v", err)
			}
			processMu.Lock()
			hold := processHolds[dir]
			processMu.Unlock()
			if hold != nil || os.Getenv("GPU_LOCK_HELD") != "unrelated authority" {
				t.Fatal("cleanup retained a hold or changed preexisting authority")
			}
		})
	}
}

// The third Err check is acquireGuarded's entry, after namespace admission
// and before it waits for processMu. Signal once without blocking that check.
type acquireCleanupWaitingContext struct {
	context.Context
	checks  atomic.Int32
	entered chan struct{}
}

func (c *acquireCleanupWaitingContext) Err() error {
	err := c.Context.Err()
	if c.checks.Add(1) == 3 {
		close(c.entered)
	}
	return err
}

func TestAcquireGuardCleanupConcurrentFinalization(t *testing.T) {
	for _, mode := range []string{"success", "unlock", "close", "both"} {
		t.Run(mode, func(t *testing.T) {
			dir := acquireCleanupDir(t)
			ctx, cancel := context.WithTimeout(t.Context(), 3*time.Second)
			defer cancel()
			type result struct {
				h   *Handle
				err error
			}
			rootDone, nestedDone := make(chan result, 1), make(chan result, 1)
			closing, resumeClose := make(chan struct{}), make(chan struct{})
			resume := func() {
				select {
				case <-resumeClose:
				default:
					close(resumeClose)
				}
			}
			var root, nested result
			rootRead, nestedRead, nestedStarted := false, false, false
			defer func() {
				resume()
				if !rootRead {
					select {
					case root = <-rootDone:
					case <-time.After(3 * time.Second):
						t.Error("root did not finish after Close resumed")
					}
				}
				if nestedStarted && !nestedRead {
					select {
					case nested = <-nestedDone:
					case <-time.After(3 * time.Second):
						t.Error("concurrent admission did not finish")
					}
				}
				if nested.h != nil {
					if err := nested.h.Release(); err != nil {
						t.Error(err)
					}
				}
				if root.h != nil {
					if err := root.h.Release(); err != nil {
						t.Error(err)
					}
				} else {
					var cleanup io.Closer
					if errors.As(root.err, &cleanup) {
						if err := cleanup.Close(); err != nil {
							t.Error(err)
						}
					}
				}
			}()
			faults := &acquireCleanupFaults{t: t, unlockErr: errors.New("concurrent root unlock"), closeErr: errors.New("concurrent root close")}
			if mode == "unlock" || mode == "both" {
				faults.unlockFails = 1
			}
			if mode == "close" || mode == "both" {
				faults.closeFails = 1
			}
			var paused atomic.Bool
			faults.beforeClose = func() {
				if paused.CompareAndSwap(false, true) {
					close(closing)
					<-resumeClose
				}
			}
			ops := faults.ops()
			go func() {
				h, err := acquire(ctx, ops, WithDir(dir), WithNoReclaim())
				rootDone <- result{h, err}
			}()
			select {
			case <-closing:
			case <-ctx.Done():
				t.Fatal("root did not reach original guard Close")
			}
			if processMu.TryLock() {
				processMu.Unlock()
				t.Error("root released process admission custody before original Close")
			}
			waiting := &acquireCleanupWaitingContext{Context: ctx, entered: make(chan struct{})}
			nestedStarted = true
			go func() {
				h, err := Acquire(waiting, WithDir(dir), WithNoReclaim())
				nestedDone <- result{h, err}
			}()
			select {
			case <-waiting.entered:
			case <-ctx.Done():
				t.Fatal("concurrent admission did not reach process state check")
			}
			if processMu.TryLock() {
				processMu.Unlock()
				t.Error("root finalization did not retain process admission custody")
			}
			select {
			case nested = <-nestedDone:
				nestedRead = true
				t.Error("concurrent admission escaped before root guard cleanup")
			default:
			}
			resume()
			select {
			case root = <-rootDone:
				rootRead = true
			case <-ctx.Done():
				t.Fatal("root did not complete finalization")
			}
			if !nestedRead {
				select {
				case nested = <-nestedDone:
					nestedRead = true
				case <-ctx.Done():
					t.Fatal("concurrent admission did not observe root outcome")
				}
			}
			if mode == "success" {
				if root.h == nil || root.err != nil || nested.h == nil || nested.err != nil || !nested.h.IsNested() {
					t.Fatalf("healthy concurrent admission did not wait and succeed: root=%+v nested=%+v", root, nested)
				}
				acquireCleanupHold(t, dir, 2, false)
				if owner, err := Inherited(os.Getenv); err != nil || owner.PID != nested.h.owner.PID {
					t.Fatalf("successful nested admission lost verified authority: %+v %v", owner, err)
				}
			} else {
				acquireCleanupOwner(t, root.h, root.err)
				if errors.Is(root.err, faults.unlockErr) != (mode != "close") || errors.Is(root.err, faults.closeErr) != (mode != "unlock") {
					t.Fatalf("concurrent root failure lost guard identities: %v", root.err)
				}
				var invented io.Closer
				if nested.h != nil || nested.err == nil || errors.As(nested.err, &invented) {
					t.Fatalf("failed root authorized concurrent work or invented custody: %+v", nested)
				}
				acquireCleanupHold(t, dir, 1, true)
			}
		})
	}
}

func TestAcquireGuardCleanupNested(t *testing.T) {
	for _, parentFirst := range []bool{false, true} {
		name := "parent retained"
		if parentFirst {
			name = "parent released first"
		}
		t.Run(name, func(t *testing.T) {
			dir := acquireCleanupDir(t)
			root, err := Acquire(t.Context(), WithDir(dir))
			if err != nil {
				t.Fatal(err)
			}
			defer root.Release()
			before, err := os.ReadFile(filepath.Join(dir, "owner"))
			if err != nil {
				t.Fatal(err)
			}
			faults := &acquireCleanupFaults{t: t, unlockErr: errors.New("nested unlock"), unlockFails: 2, closeErr: errors.New("nested close"), closeFails: 2}
			h, err := acquire(t.Context(), faults.ops(), WithDir(dir))
			cleanup := acquireCleanupOwner(t, h, err)
			defer cleanup.Close()
			acquireCleanupHold(t, dir, 2, false)
			if parentFirst {
				if err := root.Release(); err != nil {
					t.Fatal(err)
				}
				acquireCleanupHold(t, dir, 1, false)
			} else {
				// A failed child cannot revoke its independently held parent.
				if owner, err := Inherited(os.Getenv); err != nil || owner.PID != root.owner.PID {
					t.Fatalf("parent authority was revoked: %+v %v", owner, err)
				}
			}
			if err := cleanup.Close(); !errors.Is(err, faults.unlockErr) || !errors.Is(err, faults.closeErr) {
				t.Fatalf("nested retry did not retain guard failures: %v", err)
			}
			if !parentFirst {
				acquireCleanupHold(t, dir, 1, false)
			}
			if err := cleanup.Close(); err != nil {
				t.Fatal(err)
			}
			if err := cleanup.Close(); err != nil {
				t.Fatal(err)
			}
			if parentFirst {
				if _, err := os.Lstat(dir); !errors.Is(err, os.ErrNotExist) {
					t.Fatalf("last nested cleanup orphaned publication: %v", err)
				}
				if os.Getenv("GPU_LOCK_HELD") != "" {
					t.Fatal("last nested cleanup retained retired parent authority")
				}
			} else {
				acquireCleanupHold(t, dir, 1, false)
				after, err := os.ReadFile(filepath.Join(dir, "owner"))
				if err != nil || string(after) != string(before) {
					t.Fatal("failed nested cleanup retired or changed parent")
				}
				if os.Getenv("GPU_LOCK_HELD") != strconv.Itoa(root.owner.PID) {
					t.Fatal("failed nested cleanup changed parent environment")
				}
			}
		})
	}
}

func TestAcquireGuardCleanupNestedRetirementAuthority(t *testing.T) {
	dir := acquireCleanupDir(t)
	root, err := Acquire(t.Context(), WithDir(dir))
	if err != nil {
		t.Fatal(err)
	}
	defer root.Release()
	before, err := os.ReadFile(filepath.Join(dir, "owner"))
	if err != nil {
		t.Fatal(err)
	}
	faults := &acquireCleanupFaults{t: t, unlockErr: errors.New("nested retirement unlock"), unlockFails: 3, closeErr: errors.New("nested retirement close"), closeFails: 3}
	h, err := acquire(t.Context(), faults.ops(), WithDir(dir))
	cleanup := acquireCleanupOwner(t, h, err)
	defer cleanup.Close()
	if err := root.Release(); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "owner"), []byte("displaced owner"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := cleanup.Close(); err == nil || !strings.Contains(err.Error(), "lock owner displaced") {
		t.Fatalf("expected deterministic final-reference retirement refusal: %v", err)
	}
	acquireCleanupHold(t, dir, 1, true)
	if os.Getenv("GPU_LOCK_HELD") != strconv.Itoa(root.owner.PID) {
		t.Fatal("failed retirement cleared retained parent authority")
	}
	if err := os.WriteFile(filepath.Join(dir, "owner"), before, 0644); err != nil {
		t.Fatal(err)
	}
	if err := cleanup.Close(); !errors.Is(err, faults.unlockErr) || !errors.Is(err, faults.closeErr) {
		t.Fatalf("retirement completion lost guard-only retry: %v", err)
	}
	if _, err := os.Lstat(dir); !errors.Is(err, os.ErrNotExist) || os.Getenv("GPU_LOCK_HELD") != "" {
		t.Fatalf("successful retry did not retire parent publication and authority: %v", err)
	}
	next, err := Acquire(t.Context(), WithDir(dir))
	if err != nil {
		t.Fatal(err)
	}
	defer next.Release()
	nextData, err := os.ReadFile(filepath.Join(dir, "owner"))
	if err != nil {
		t.Fatal(err)
	}
	if err := cleanup.Close(); err != nil {
		t.Fatal(err)
	}
	if err := cleanup.Close(); err != nil {
		t.Fatal(err)
	}
	after, err := os.ReadFile(filepath.Join(dir, "owner"))
	if err != nil || string(after) != string(nextData) || os.Getenv("GPU_LOCK_HELD") != strconv.Itoa(next.owner.PID) {
		t.Fatal("guard-only retry repeated retirement or cleared successor authority")
	}
	acquireCleanupHold(t, dir, 1, false)
}

func TestAcquireGuardCleanupCanceledFinalNested(t *testing.T) {
	dir := acquireCleanupDir(t)
	root, err := Acquire(t.Context(), WithDir(dir))
	if err != nil {
		t.Fatal(err)
	}
	defer root.Release()
	nested, err := Acquire(t.Context(), WithDir(dir))
	if err != nil {
		t.Fatal(err)
	}
	if err := root.Release(); err != nil {
		t.Fatal(err)
	}
	g, err := openNamespace(dir)
	if err != nil {
		t.Fatal(err)
	}
	if err := g.tryLock(); err != nil {
		g.file.Close()
		t.Fatal(err)
	}
	finalRemove := errors.New("canceled final nested removal")
	err = cancelAdmission(nested, g, context.Canceled, func(string) error { return finalRemove })
	guardErr := errors.Join(guardUnlock(g.file), g.file.Close())
	cleanup := acquireCleanupOwner(t, nil, err)
	defer cleanup.Close()
	if guardErr != nil || !errors.Is(err, context.Canceled) || !errors.Is(err, finalRemove) {
		t.Fatalf("canceled final nested retirement lost causes: %v %v", err, guardErr)
	}
	var owner *acquireCleanupError
	if !errors.As(err, &owner) || !owner.clearAuthority {
		t.Fatal("cancelAdmission did not retain prior-authority intent")
	}
	acquireCleanupHold(t, dir, 1, true)
	if os.Getenv("GPU_LOCK_HELD") != strconv.Itoa(nested.owner.PID) {
		t.Fatal("failed cancellation retirement cleared parent authority")
	}
	if err := cleanup.Close(); err != nil {
		t.Fatal(err)
	}
	if err := cleanup.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Lstat(dir); !errors.Is(err, os.ErrNotExist) || os.Getenv("GPU_LOCK_HELD") != "" {
		t.Fatalf("canceled final nested retry orphaned publication or authority: %v", err)
	}
}

func TestAcquireGuardCleanupInherited(t *testing.T) {
	dir := acquireCleanupDir(t)
	if err := os.Mkdir(dir, 0755); err != nil {
		t.Fatal(err)
	}
	owner := Owner{PID: os.Getpid(), Host: shortHostname(), Label: "external parent", At: time.Now()}
	before := owner.String()
	if err := os.WriteFile(filepath.Join(dir, "owner"), []byte(before), 0644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("GPU_LOCK_HELD", strconv.Itoa(owner.PID))
	faults := &acquireCleanupFaults{t: t, closeErr: errors.New("inherited close"), closeFails: 1}
	h, err := acquire(t.Context(), faults.ops(), WithDir(dir))
	cleanup := acquireCleanupOwner(t, h, err)
	defer cleanup.Close()
	hold := acquireCleanupHold(t, dir, 1, false)
	if !hold.inherited {
		t.Fatal("fixture did not enter inherited admission")
	}
	if err := cleanup.Close(); err != nil {
		t.Fatal(err)
	}
	if err := cleanup.Close(); err != nil {
		t.Fatal(err)
	}
	after, err := os.ReadFile(filepath.Join(dir, "owner"))
	if err != nil || string(after) != before || os.Getenv("GPU_LOCK_HELD") != strconv.Itoa(owner.PID) {
		t.Fatal("inherited cleanup retired or changed external parent")
	}
	processMu.Lock()
	remaining := processHolds[dir]
	processMu.Unlock()
	if remaining != nil {
		t.Fatal("inherited cleanup leaked its local reference")
	}
}

func TestAcquireGuardCleanupRetryRefusal(t *testing.T) {
	for _, mode := range []string{"busy", "displaced"} {
		t.Run(mode, func(t *testing.T) {
			dir := acquireCleanupDir(t)
			faults := &acquireCleanupFaults{t: t, closeErr: errors.New("initial close"), closeFails: 1}
			h, err := acquire(t.Context(), faults.ops(), WithDir(dir))
			cleanup := acquireCleanupOwner(t, h, err)
			defer cleanup.Close()
			before, err := os.ReadFile(filepath.Join(dir, "owner"))
			if err != nil {
				t.Fatal(err)
			}
			var restore func()
			if mode == "busy" {
				blocker, err := openNamespace(dir)
				if err != nil {
					t.Fatal(err)
				}
				if err := blocker.tryLock(); err != nil {
					blocker.file.Close()
					t.Fatal(err)
				}
				restore = func() {
					if err := errors.Join(guardUnlock(blocker.file), blocker.file.Close()); err != nil {
						t.Error(err)
					}
				}
			} else {
				old := dir + ".guard.saved"
				if err := os.Rename(dir+".guard", old); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(dir+".guard", nil, 0600); err != nil {
					t.Fatal(err)
				}
				restore = func() {
					if err := errors.Join(os.Remove(dir+".guard"), os.Rename(old, dir+".guard")); err != nil {
						t.Error(err)
					}
				}
			}
			start := time.Now()
			err = cleanup.Close()
			elapsed := time.Since(start)
			after, readErr := os.ReadFile(filepath.Join(dir, "owner"))
			acquireCleanupHold(t, dir, 1, true)
			restore()
			if err == nil || elapsed > time.Second || readErr != nil || string(after) != string(before) {
				t.Fatalf("cleanup did not promptly refuse unchanged %s namespace: %v elapsed=%v", mode, err, elapsed)
			}
			if len(faults.closes) != 2 {
				t.Fatal("refused fresh descriptor was not closed exactly once")
			}
			if err := cleanup.Close(); err != nil {
				t.Fatal("same-namespace retry after refusal:", err)
			}
		})
	}
}

func TestAcquireGuardCleanupDoesNotRepeatPublication(t *testing.T) {
	dir := acquireCleanupDir(t)
	faults := &acquireCleanupFaults{
		t:         t,
		unlockErr: errors.New("initial and retry unlock"), unlockFails: 2,
		closeErr: errors.New("initial and retry close"), closeFails: 2,
	}
	h, err := acquire(t.Context(), faults.ops(), WithDir(dir))
	cleanup := acquireCleanupOwner(t, h, err)
	defer cleanup.Close()
	if err := cleanup.Close(); !errors.Is(err, faults.unlockErr) || !errors.Is(err, faults.closeErr) {
		t.Fatalf("retirement guard failure lost causes: %v", err)
	}
	if _, err := os.Lstat(dir); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("first explicit cleanup did not finish publication cleanup: %v", err)
	}
	// Publish a successor in the same namespace after prior content cleanup.
	// The next explicit retry must clean only the remaining guard obligation.
	next, err := Acquire(t.Context(), WithDir(dir))
	if err != nil {
		t.Fatal(err)
	}
	defer next.Release()
	before, err := os.ReadFile(filepath.Join(dir, "owner"))
	if err != nil {
		t.Fatal(err)
	}
	if err := cleanup.Close(); err != nil {
		t.Fatal(err)
	}
	if err := cleanup.Close(); err != nil {
		t.Fatal(err)
	}
	after, err := os.ReadFile(filepath.Join(dir, "owner"))
	if err != nil || string(after) != string(before) {
		t.Fatal("later guard cleanup repeated completed publication removal")
	}
	acquireCleanupHold(t, dir, 1, false)
}

// Cancel at the final context check after root publication. Displacing only the
// owner contents gives a deterministic retirement failure without permissions,
// processes, global hooks, or a change to ordinary release helpers.
type acquireCleanupCancelContext struct {
	context.Context
	dir      string
	cancel   context.CancelFunc
	original []byte
	err      error
}

func (c *acquireCleanupCancelContext) Err() error {
	if c.original == nil {
		if data, err := os.ReadFile(filepath.Join(c.dir, "owner")); err == nil {
			c.original = data
			c.err = os.WriteFile(filepath.Join(c.dir, "owner"), []byte("displaced for cancellation"), 0644)
			c.cancel()
		}
	}
	return c.Context.Err()
}

func TestAcquireGuardCleanupCancellationOwner(t *testing.T) {
	for _, noReclaim := range []bool{false, true} {
		name := "ordinary"
		if noReclaim {
			name = "no reclaim"
		}
		t.Run(name, func(t *testing.T) {
			dir := acquireCleanupDir(t)
			base, cancel := context.WithCancel(t.Context())
			defer cancel()
			ctx := &acquireCleanupCancelContext{Context: base, dir: dir, cancel: cancel}
			faults := &acquireCleanupFaults{t: t, unlockErr: errors.New("cancel unlock"), closeErr: errors.New("cancel close"), unlockFails: 1, closeFails: 1}
			opts := []Option{WithDir(dir)}
			if noReclaim {
				opts = append(opts, WithNoReclaim())
			}
			h, err := acquire(ctx, faults.ops(), opts...)
			cleanup := acquireCleanupOwner(t, h, err)
			defer cleanup.Close()
			if ctx.err != nil || len(ctx.original) == 0 {
				t.Fatal("cancellation fixture did not displace published owner:", ctx.err)
			}
			if !errors.Is(err, context.Canceled) || !errors.Is(err, faults.unlockErr) || !errors.Is(err, faults.closeErr) || !strings.Contains(err.Error(), "lock owner displaced") {
				t.Fatalf("combined admission and guard failures lost causes: %v", err)
			}
			if n := acquireCleanupOwners(err); n != 1 {
				t.Fatalf("error contains %d competing cleanup owners", n)
			}
			acquireCleanupHold(t, dir, 1, true)
			if os.Getenv("GPU_LOCK_HELD") != "" {
				t.Fatal("canceled failed admission exported authority")
			}
			if err := os.WriteFile(filepath.Join(dir, "owner"), ctx.original, 0644); err != nil {
				t.Fatal(err)
			}
			if err := cleanup.Close(); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func acquireCleanupOwners(err error) int {
	n := 0
	if _, ok := err.(*acquireCleanupError); ok {
		n++
	}
	switch err := err.(type) {
	case interface{ Unwrap() []error }:
		for _, child := range err.Unwrap() {
			n += acquireCleanupOwners(child)
		}
	case interface{ Unwrap() error }:
		n += acquireCleanupOwners(err.Unwrap())
	}
	return n
}

func TestAcquireGuardCleanupNoPublication(t *testing.T) {
	dir := acquireCleanupDir(t)
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	faults := &acquireCleanupFaults{t: t, closeErr: errors.New("prepublication close"), closeFails: 1}
	h, err := acquire(ctx, faults.ops(), WithDir(dir))
	var cleanup io.Closer
	if h != nil || !errors.Is(err, context.Canceled) || !errors.Is(err, faults.closeErr) || errors.As(err, &cleanup) {
		t.Fatalf("prepublication failure lost causes or invented custody: %v %v", h, err)
	}
	if len(faults.closes) != 1 || faults.unlocks != 0 {
		t.Fatal("prepublication guard cleanup did not close once")
	}
}

func TestAcquireGuardCleanupPublicBinding(t *testing.T) {
	file, err := parser.ParseFile(token.NewFileSet(), "lock.go", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	var public, admission, cleanup *ast.FuncDecl
	for _, decl := range file.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok {
			continue
		}
		switch fn.Name.Name {
		case "Acquire":
			public = fn
		case "acquire":
			admission = fn
		case "Close":
			if fn.Recv != nil {
				cleanup = fn
			}
		}
	}
	if public == nil || len(public.Body.List) != 1 || admission == nil || cleanup == nil {
		t.Fatal("missing public admission binding or cleanup implementation")
	}
	ret, ok := public.Body.List[0].(*ast.ReturnStmt)
	if !ok || len(ret.Results) != 1 {
		t.Fatal("public Acquire does not return private admission")
	}
	call, ok := ret.Results[0].(*ast.CallExpr)
	if !ok || len(call.Args) != 3 || call.Ellipsis == token.NoPos {
		t.Fatal("public Acquire does not forward per-call options")
	}
	name, ok := call.Fun.(*ast.Ident)
	if !ok || name.Name != "acquire" {
		t.Fatal("public Acquire bypasses private admission")
	}
	ops, ok := call.Args[1].(*ast.CompositeLit)
	if !ok || len(ops.Elts) != 2 {
		t.Fatal("public Acquire has no concrete guard operations")
	}
	unlock, ok := ops.Elts[0].(*ast.Ident)
	if !ok || unlock.Name != "guardUnlock" {
		t.Fatal("public Acquire does not use actual guardUnlock")
	}
	close, ok := ops.Elts[1].(*ast.SelectorExpr)
	if !ok || close.Sel.Name != "Close" {
		t.Fatal("public Acquire does not bind file Close")
	}
	paren, ok := close.X.(*ast.ParenExpr)
	if !ok {
		t.Fatal("public Acquire Close is not a method expression")
	}
	star, ok := paren.X.(*ast.StarExpr)
	if !ok {
		t.Fatal("public Acquire Close has no pointer receiver")
	}
	fileType, ok := star.X.(*ast.SelectorExpr)
	if !ok || fileType.Sel.Name != "File" {
		t.Fatal("public Acquire Close has no os.File receiver")
	}
	pkg, ok := fileType.X.(*ast.Ident)
	if !ok || pkg.Name != "os" {
		t.Fatal("public Acquire Close has no os.File receiver")
	}
	fresh, bounded := false, false
	ast.Inspect(cleanup.Body, func(node ast.Node) bool {
		if call, ok := node.(*ast.CallExpr); ok {
			if name, ok := call.Fun.(*ast.Ident); ok && name.Name == "openNamespace" {
				fresh = true
			}
			if method, ok := call.Fun.(*ast.SelectorExpr); ok {
				if method.Sel.Name == "Release" || method.Sel.Name == "lock" {
					t.Error("cleanup uses a potentially blocking Release path")
				}
				bounded = bounded || method.Sel.Name == "tryLock"
			}
		}
		return true
	})
	if !fresh || !bounded {
		t.Fatal("cleanup does not open and try a fresh guard")
	}
	// The environment export must live in the guard-closing finalizer, after
	// that Close call, rather than in the acquisition loop's success return.
	closePos, exportPos := token.NoPos, token.NoPos
	ast.Inspect(admission.Body, func(node ast.Node) bool {
		deferStmt, ok := node.(*ast.DeferStmt)
		if !ok {
			return true
		}
		ast.Inspect(deferStmt, func(node ast.Node) bool {
			call, ok := node.(*ast.CallExpr)
			if !ok {
				return true
			}
			method, ok := call.Fun.(*ast.SelectorExpr)
			if !ok {
				return true
			}
			if method.Sel.Name == "close" {
				closePos = call.Pos()
			}
			if method.Sel.Name == "Setenv" {
				exportPos = call.Pos()
			}
			return true
		})
		return false
	})
	if closePos == token.NoPos || exportPos <= closePos {
		t.Fatal("authority export does not follow guard Close in finalizer")
	}
}
