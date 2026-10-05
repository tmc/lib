package cudahotswap

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudalock"
)

func TestKernelRollbackCustodyHolderHelper(t *testing.T) {
	if os.Getenv("KERNEL_ROLLBACK_HOLDER_HELPER") != "1" {
		return
	}
	h, err := cudalock.Acquire(context.Background())
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	fmt.Println("ready")
	_, _ = io.Copy(io.Discard, os.Stdin)
	if err := h.Release(); err != nil {
		os.Exit(3)
	}
	os.Exit(0)
}

func TestKernelRollbackStandaloneCustody(t *testing.T) {
	t.Setenv("GPU_LOCKDIR", filepath.Join(t.TempDir(), "lock"))
	t.Setenv("GPU_LOCK_HELD", "")
	h, err := cudalock.Acquire(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	c, err := captureRollbackCustody(requestLease{h})
	if err != nil || c == nil || c.inherited {
		t.Fatalf("custody=%+v err=%v", c, err)
	}
	if err := h.Release(); err != nil {
		t.Fatal(err)
	}
	h, err = cudalock.Acquire(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	defer h.Release()
	if err := c.verifyLease(requestLease{h}); err != nil {
		t.Fatal(err)
	}
	exerciseKernelRollbackActualLockHostSimulation(t)
	t.Setenv("GPU_LOCKDIR", filepath.Join(t.TempDir(), "different-lock"))
	if err := c.verifyInherited(); err == nil {
		t.Fatal("accepted incompatible directory before acquisition")
	}
}

func TestKernelRollbackInheritedLiveChangedAndDeadCustody(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "lock")
	t.Setenv("GPU_LOCKDIR", dir)
	t.Setenv("GPU_LOCK_HELD", "")
	cmd := exec.Command(os.Args[0], "-test.run=^TestKernelRollbackCustodyHolderHelper$")
	for _, e := range os.Environ() {
		if !strings.HasPrefix(e, "GPU_LOCK_HELD=") && !strings.HasPrefix(e, "KERNEL_ROLLBACK_HOLDER_HELPER=") {
			cmd.Env = append(cmd.Env, e)
		}
	}
	cmd.Env = append(cmd.Env, "GPU_LOCK_HELD=", "KERNEL_ROLLBACK_HOLDER_HELPER=1")
	out, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	in, err := cmd.StdinPipe()
	if err != nil {
		t.Fatal(err)
	}
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	waited := false
	defer func() {
		in.Close()
		if !waited {
			_ = cmd.Process.Kill()
			_ = cmd.Wait()
		}
	}()
	line, err := bufio.NewReader(out).ReadString('\n')
	if err != nil || line != "ready\n" {
		t.Fatalf("holder=%q err=%v", line, err)
	}
	t.Setenv("GPU_LOCK_HELD", strconv.Itoa(cmd.Process.Pid))
	h, err := cudalock.Acquire(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	c, err := captureRollbackCustody(requestLease{h})
	if err != nil || c == nil || !c.inherited {
		t.Fatalf("custody=%+v err=%v", c, err)
	}
	if err := h.Release(); err != nil {
		t.Fatal(err)
	}
	h, err = cudalock.Acquire(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if err := c.verifyLease(requestLease{h}); err != nil {
		t.Fatal(err)
	}
	if err := h.Release(); err != nil {
		t.Fatal(err)
	}
	exerciseKernelRollbackActualLockHostSimulation(t)
	t.Setenv("GPU_LOCK_HELD", "1")
	if err := c.verifyInherited(); err == nil {
		t.Fatal("forged inheritance admitted")
	}
	t.Setenv("GPU_LOCK_HELD", strconv.Itoa(cmd.Process.Pid))
	if err := cmd.Process.Kill(); err != nil {
		t.Fatal(err)
	}
	_ = cmd.Wait()
	waited = true
	if err := c.verifyInherited(); err == nil {
		t.Fatal("dead inherited owner admitted")
	}
	// No acquire callback may run after the custody check refuses this holder.
	s, _ := testService(t, &serviceLoader{kernel: &serviceKernel{raw: 1}})
	registerService(t, s)
	prior, _, err := s.table.pin("kernel")
	if err != nil {
		t.Fatal(err)
	}
	token, err := s.table.reserveRollback("kernel", prior.gen)
	if err != nil {
		t.Fatal(err)
	}
	checking := &pushCheck{prior: prior, rollback: token, custody: c}
	s.mu.Lock()
	s.checking = map[string]*pushCheck{"kernel": checking}
	s.mu.Unlock()
	defer s.releasePushCheck("kernel", checking)
	calls := 0
	s.acquire = func(context.Context, string) (io.Closer, error) { calls++; return new(serviceCloser), nil }
	_, err = s.rollbackCheckedKernel(t.Context(), "kernel", 1, "rollback", checking)
	if err == nil || !strings.Contains(err.Error(), "rollback custody") || calls != 0 {
		t.Fatalf("dead holder fell back to fresh root acquisition: calls=%d err=%v", calls, err)
	}
}

// Real cudalock acquisition with simulated compiler/kernel operations. This
// validates custody and cancellation control flow, not CUDA execution.
func exerciseKernelRollbackActualLockHostSimulation(t *testing.T) {
	t.Helper()
	loader := &serviceLoader{kernel: &serviceKernel{raw: 1}}
	s, err := NewService(loader)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.Register(t.Context(), "kernel", Source{Code: "source", Entry: "entry"}, "original"); err != nil {
		t.Fatal(err)
	}
	loader.kernel = &serviceKernel{raw: 2}
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	if _, err := s.SetCheck("kernel", CheckConfig{Checker: func(context.Context, State) error { cancel(); return context.Canceled }}, "cancel"); err != nil {
		t.Fatal(err)
	}
	r, err := s.Push(ctx, "kernel", Source{Code: "candidate", Entry: "entry"}, "unchecked")
	if !errors.Is(err, context.Canceled) || r == nil || !r.Published || !r.RolledBack || r.Selected.Version != 1 {
		t.Fatalf("actual lock host simulation r=%+v err=%v", r, err)
	}
	if err := s.Close(t.Context()); err != nil {
		t.Fatal(err)
	}
}
