//go:build darwin || linux

package cudalock

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

// This helper is selected only by a parent that supplies an explicit private
// path. It never consults an installed gpu-lock command or canonical GPU path.
func TestGuardChild(t *testing.T) {
	dir, mode := os.Getenv("CUDALOCK_GUARD_TEST_DIR"), os.Getenv("CUDALOCK_GUARD_TEST_MODE")
	if !filepath.IsAbs(dir) || mode == "" {
		t.Fatal("explicit private child configuration required")
	}
	in := bufio.NewReader(os.Stdin)
	fmt.Println("started")
	if line, _ := in.ReadString('\n'); line != "go\n" {
		t.Fatal("missing start handshake")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	if mode == "guard" {
		g, err := openNamespace(dir)
		if err != nil {
			t.Fatal(err)
		}
		defer g.file.Close()
		if err := g.lock(ctx, time.Millisecond); err != nil {
			t.Fatal(err)
		}
		fmt.Println("guarded")
		if line, _ := in.ReadString('\n'); line != "release\n" {
			t.Fatal("missing guard release handshake")
		}
		if err := guardUnlock(g.file); err != nil {
			t.Fatal(err)
		}
		return
	}
	if mode == "try" {
		var stop context.CancelFunc
		ctx, stop = context.WithTimeout(ctx, 120*time.Millisecond)
		defer stop()
	}
	h, err := Acquire(ctx, WithDir(dir), WithLabel("guard-child"), WithPollInterval(time.Millisecond))
	if mode == "try" {
		if h != nil {
			h.Release()
			t.Fatal("public contender acquired while exclusion was required")
		}
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("contender error %v, want deadline", err)
		}
		fmt.Println("blocked")
		return
	}
	if err != nil {
		t.Fatal(err)
	}
	if h.IsNested() || h.Owner().PID != os.Getpid() {
		t.Fatal("child must acquire its own public root handle")
	}
	for {
		_, err := Inherited(os.Getenv)
		if err == nil {
			break
		}
		if !errors.Is(err, ErrNotInherited) || ctx.Err() != nil {
			t.Fatal(err)
		}
		time.Sleep(time.Millisecond)
	}
	marker := filepath.Join(filepath.Dir(dir), "active-work")
	if err := os.Mkdir(marker, 0700); err != nil {
		t.Fatal("overlapping public holders:", err)
	}
	fmt.Println("acquired")
	if line, _ := in.ReadString('\n'); line != "release\n" {
		t.Fatal("missing root release handshake")
	}
	if err := os.Remove(marker); err != nil {
		t.Fatal(err)
	}
	if err := h.Release(); err != nil {
		t.Fatal(err)
	}
	fmt.Println("released")
}

type guardChild struct {
	cmd   *exec.Cmd
	in    io.WriteCloser
	lines chan string
	err   bytes.Buffer
	done  bool
}

func startGuardChild(t *testing.T, dir, mode string) *guardChild {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 8*time.Second)
	c := &guardChild{cmd: exec.CommandContext(ctx, os.Args[0], "-test.run=^TestGuardChild$", "-test.v"), lines: make(chan string, 32)}
	for _, value := range os.Environ() {
		if !strings.HasPrefix(value, "GPU_LOCK_") && !strings.HasPrefix(value, "CUDALOCK_GUARD_TEST_") {
			c.cmd.Env = append(c.cmd.Env, value)
		}
	}
	c.cmd.Env = append(c.cmd.Env, "GPU_LOCKDIR="+dir, "GPU_LOCK_HELD=", "CUDALOCK_GUARD_TEST_DIR="+dir, "CUDALOCK_GUARD_TEST_MODE="+mode)
	var err error
	c.in, err = c.cmd.StdinPipe()
	if err != nil {
		cancel()
		t.Fatal(err)
	}
	out, err := c.cmd.StdoutPipe()
	if err != nil {
		cancel()
		c.in.Close()
		t.Fatal(err)
	}
	c.cmd.Stderr = &c.err
	if err := c.cmd.Start(); err != nil {
		cancel()
		c.in.Close()
		t.Fatal(err)
	}
	go func() {
		defer close(c.lines)
		scan := bufio.NewScanner(out)
		for scan.Scan() {
			c.lines <- scan.Text()
		}
	}()
	t.Cleanup(func() {
		if !c.done {
			c.in.Close()
			c.cmd.Process.Kill()
			c.cmd.Wait()
		}
		cancel()
	})
	c.want(t, "started")
	return c
}

func (c *guardChild) send(t *testing.T, s string) {
	t.Helper()
	if _, err := fmt.Fprintln(c.in, s); err != nil {
		t.Fatal(err)
	}
}

func (c *guardChild) want(t *testing.T, want string) {
	t.Helper()
	timer := time.NewTimer(4 * time.Second)
	defer timer.Stop()
	var seen []string
	for {
		select {
		case line, ok := <-c.lines:
			if !ok {
				t.Fatalf("child ended before %q: %s", want, strings.Join(seen, "\n"))
			}
			seen = append(seen, line)
			if line == want {
				return
			}
		case <-timer.C:
			t.Fatalf("child did not report %q", want)
		}
	}
}

func (c *guardChild) wait(t *testing.T, killed bool) {
	t.Helper()
	c.in.Close()
	for range c.lines {
	}
	err := c.cmd.Wait()
	c.done = true
	if (err != nil) != killed {
		t.Fatalf("child terminal error %v, killed=%v; stderr %s", err, killed, c.err.String())
	}
}

func privateGuardDir(t *testing.T) string {
	t.Helper()
	return filepath.Join(t.TempDir(), "lock")
}

func TestGuardRecoveryBlocksThirdContender(t *testing.T) {
	dir := privateGuardDir(t)
	if err := os.Mkdir(dir, 0700); err != nil {
		t.Fatal(err)
	}
	const deadPID = 2147483647
	if isAlive(deadPID) {
		t.Fatal("synthetic stale PID unexpectedly exists")
	}
	stale := Owner{PID: deadPID, Host: shortHostname(), Label: "stale", At: time.Unix(1, 0)}
	if err := os.WriteFile(filepath.Join(dir, "owner"), []byte(stale.String()), 0600); err != nil {
		t.Fatal(err)
	}
	child := startGuardChild(t, dir, "try")
	called := false
	reclaimHook = func() {
		reclaimHook = nil
		called = true
		// The stale snapshot has been read, but the permanent guard is still
		// held. The actual public third contender must not pass this phase.
		child.send(t, "go")
		child.want(t, "blocked")
		child.wait(t, false)
	}
	t.Cleanup(func() { reclaimHook = nil })
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	h, err := Acquire(ctx, WithDir(dir), WithPollInterval(time.Millisecond))
	if err != nil || !called {
		t.Fatalf("stale recovery: handle=%v err=%v barrier=%v", h, err, called)
	}
	if h.IsNested() || h.Owner().PID != os.Getpid() {
		t.Fatal("stale recovery did not return a real root handle")
	}
	if err := h.Release(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(dir + ".guard"); err != nil {
		t.Fatal("release removed permanent guard:", err)
	}
}

func TestGuardOriginalWitnessAndCanceledPublication(t *testing.T) {
	for _, phase := range []string{"live-replacement", "canceled-publication"} {
		t.Run(phase, func(t *testing.T) {
			dir := privateGuardDir(t)
			if err := os.Mkdir(dir, 0700); err != nil {
				t.Fatal(err)
			}
			const deadPID = 2147483647
			if isAlive(deadPID) {
				t.Fatal("synthetic stale PID unexpectedly exists")
			}
			stale := Owner{PID: deadPID, Host: shortHostname(), Label: "stale", At: time.Unix(1, 0)}
			if err := os.WriteFile(filepath.Join(dir, "owner"), []byte(stale.String()), 0600); err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithTimeout(context.Background(), time.Second)
			defer cancel()
			called := false
			reclaimHook = func() {
				reclaimHook = nil
				called = true
				if phase == "canceled-publication" {
					cancel()
					return
				}
				// This is the original witness's A at exactly the original hook:
				// real public acquisition after B read stale S, before B renames.
				// A cannot return a live replacement while B retains the guard.
				firstCtx, stop := context.WithTimeout(context.Background(), 50*time.Millisecond)
				defer stop()
				first, err := Acquire(firstCtx, WithDir(dir), WithPollInterval(time.Millisecond))
				if first != nil {
					first.Release()
					t.Fatal("original witness obtained the live replacement under stale reclaimer's guard")
				}
				if !errors.Is(err, context.DeadlineExceeded) {
					t.Fatalf("replacement refusal: %v", err)
				}
			}
			t.Cleanup(func() { reclaimHook = nil })
			h, err := Acquire(ctx, WithDir(dir), WithPollInterval(time.Millisecond))
			if !called {
				t.Fatal("required source phase was not reached")
			}
			if phase == "canceled-publication" {
				if h != nil || !errors.Is(err, context.Canceled) {
					t.Fatalf("published after cancellation: %v %v", h, err)
				}
				if _, err := os.Stat(dir); !errors.Is(err, os.ErrNotExist) {
					t.Fatal("canceled stale replacement was published")
				}
				fresh, stop := context.WithTimeout(context.Background(), time.Second)
				defer stop()
				h, err = Acquire(fresh, WithDir(dir), WithPollInterval(time.Millisecond))
			}
			if err != nil || h == nil || h.IsNested() {
				t.Fatalf("serialized root acquisition: %v %v", h, err)
			}
			if err := h.Release(); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestGuardConcurrentPublicAcquireRelease(t *testing.T) {
	dir := privateGuardDir(t)
	children := []*guardChild{startGuardChild(t, dir, "hold"), startGuardChild(t, dir, "hold"), startGuardChild(t, dir, "hold")}
	for _, c := range children {
		c.send(t, "go")
	}
	for remaining := len(children); remaining > 0; remaining-- {
		var winner *guardChild
		deadline := time.Now().Add(4 * time.Second)
		for winner == nil && time.Now().Before(deadline) {
			for _, c := range children {
				if c.done {
					continue
				}
				select {
				case line, ok := <-c.lines:
					if !ok {
						t.Fatal("contender exited without acquisition")
					}
					if line == "acquired" {
						if winner != nil {
							t.Fatal("two successful public holders coexist")
						}
						winner = c
					}
				default:
				}
			}
			if winner == nil {
				time.Sleep(time.Millisecond)
			}
		}
		if winner == nil {
			t.Fatal("no contender acquired after previous release")
		}
		winner.send(t, "release")
		winner.want(t, "released")
		winner.wait(t, false)
	}
	if _, err := os.Stat(dir); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("last public release left lock: %v", err)
	}
}

func TestGuardCrashAndCancellation(t *testing.T) {
	dir := privateGuardDir(t)
	child := startGuardChild(t, dir, "guard")
	child.send(t, "go")
	child.want(t, "guarded")
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Millisecond)
	defer cancel()
	if h, err := Acquire(ctx, WithDir(dir), WithPollInterval(time.Millisecond)); h != nil || !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("held guard admitted contender: %v %v", h, err)
	}
	// This is the test's own child holding only a private advisory guard.
	// No real stale owner, GPU job or unrelated process is terminated.
	if err := child.cmd.Process.Kill(); err != nil {
		t.Fatal(err)
	}
	child.wait(t, true)
	ctx, stop := context.WithTimeout(context.Background(), time.Second)
	defer stop()
	h, err := Acquire(ctx, WithDir(dir), WithPollInterval(time.Millisecond))
	if err != nil {
		t.Fatal("OS did not release crashed child's guard:", err)
	}
	if err := h.Release(); err != nil {
		t.Fatal(err)
	}
}

func TestGuardOldInodeAndNamespaceRefusal(t *testing.T) {
	for _, change := range []string{"guard", "parent"} {
		t.Run(change, func(t *testing.T) {
			dir := privateGuardDir(t)
			g, err := openNamespace(dir)
			if err != nil {
				t.Fatal(err)
			}
			defer g.file.Close()
			if change == "guard" {
				if err := os.Rename(g.path, g.path+".old"); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(g.path, nil, 0600); err != nil {
					t.Fatal(err)
				}
			} else {
				if err := os.Rename(g.parent, g.parent+".old"); err != nil {
					t.Fatal(err)
				}
				t.Cleanup(func() { os.RemoveAll(g.parent + ".old") })
				if err := os.Mkdir(g.parent, 0700); err != nil {
					t.Fatal(err)
				}
			}
			if err := g.lock(context.Background(), time.Millisecond); err == nil {
				guardUnlock(g.file)
				t.Fatal("old inode admitted namespace transaction")
			}
		})
	}
}

func TestGuardRefusesAliasesAndUnsafeObjects(t *testing.T) {
	for _, kind := range []string{"symlink", "hardlink", "directory", "lock-symlink"} {
		t.Run(kind, func(t *testing.T) {
			dir := privateGuardDir(t)
			other := filepath.Join(filepath.Dir(dir), "other")
			if err := os.WriteFile(other, nil, 0600); err != nil {
				t.Fatal(err)
			}
			var err error
			switch kind {
			case "symlink":
				err = os.Symlink(other, dir+".guard")
			case "hardlink":
				err = os.Link(other, dir+".guard")
			case "directory":
				err = os.Mkdir(dir+".guard", 0700)
			case "lock-symlink":
				err = os.Symlink(filepath.Dir(dir), dir)
			}
			if err != nil {
				t.Fatal(err)
			}
			if h, err := Acquire(context.Background(), WithDir(dir)); h != nil || err == nil {
				t.Fatalf("unsafe namespace admitted: %v %v", h, err)
			}
		})
	}
}

func TestGuardReleaseRechecksIdentity(t *testing.T) {
	dir := privateGuardDir(t)
	h, err := Acquire(context.Background(), WithDir(dir))
	if err != nil {
		t.Fatal(err)
	}
	path := dir + ".guard"
	if err := os.Rename(path, path+".old"); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, nil, 0600); err != nil {
		t.Fatal(err)
	}
	if err := h.Release(); err == nil {
		t.Fatal("root retired under replacement guard inode")
	}
	data, err := os.ReadFile(filepath.Join(dir, "owner"))
	if err != nil || string(data) != h.Owner().String() {
		t.Fatal("failed retirement changed live owner")
	}
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(path+".old", path); err != nil {
		t.Fatal(err)
	}
	if err := h.Release(); err != nil {
		t.Fatal("failed retirement lost retriable reference:", err)
	}
}

func TestGuardPhysicalParentAlias(t *testing.T) {
	parent := t.TempDir()
	real := filepath.Join(parent, "real")
	alias := filepath.Join(parent, "alias")
	if err := os.Mkdir(real, 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(real, alias); err != nil {
		t.Fatal(err)
	}
	a, err := Acquire(context.Background(), WithDir(filepath.Join(real, "lock")))
	if err != nil {
		t.Fatal(err)
	}
	b, err := Acquire(context.Background(), WithDir(filepath.Join(alias, "lock")))
	if err != nil || !b.IsNested() {
		t.Fatalf("physical alias did not share hold: %v %v", b, err)
	}
	if err := a.Release(); err != nil {
		t.Fatal(err)
	}
	if err := b.Release(); err != nil {
		t.Fatal(err)
	}
}

func TestGuardReferenceScriptContract(t *testing.T) {
	const path = "reference/gpu-lock.sh"
	if out, err := exec.Command("/bin/bash", "-n", path).CombinedOutput(); err != nil {
		t.Fatalf("reference syntax: %v %s", err, out)
	}
	if runtime.GOOS == "darwin" {
		dir := privateGuardDir(t)
		marker := filepath.Join(filepath.Dir(dir), "must-not-run")
		cmd := exec.Command("/bin/bash", path, "/usr/bin/touch", marker)
		for _, value := range os.Environ() {
			if !strings.HasPrefix(value, "GPU_LOCK_") {
				cmd.Env = append(cmd.Env, value)
			}
		}
		cmd.Env = append(cmd.Env, "GPU_LOCKDIR="+dir, "GPU_LOCK_HELD=")
		out, err := cmd.CombinedOutput()
		if err == nil || !strings.Contains(string(out), "requires Linux") {
			t.Fatalf("unsupported reference platform: %v %s", err, out)
		}
		if _, err := os.Stat(marker); !errors.Is(err, os.ErrNotExist) {
			t.Fatal("unsupported reference platform ran work")
		}
	}
}
