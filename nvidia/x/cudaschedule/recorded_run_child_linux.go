//go:build linux

package cudaschedule

import (
	"fmt"
	"os"
	"syscall"
)

// recordedChild owns its pipes and wait status. Poll performs finite work;
// there is no abandoned Wait or output-reader goroutine on cancellation.
type recordedChild struct {
	process            *os.Process
	identity           recordedProcess
	stdout, stderr     *os.File
	stdoutFD, stderrFD int
	out, errOut        *recordedOutput
	status             syscall.WaitStatus
	reaped             bool
	outEOF, errEOF     bool
}

func recordedStartChild(executable string, args, env []string, dir string, outputLimit int) (*recordedChild, error) {
	return recordedStartProcess(executable, args, env, dir, outputLimit, 1)
}

// recordedStartKeeper starts independent custody outside an inherited lease.
func recordedStartKeeper(executable string, args, env []string, dir string, outputLimit int) (*recordedChild, error) {
	if err := recordedOutsideLease(); err != nil {
		return nil, err
	}
	return recordedStartProcess(executable, args, env, dir, outputLimit, 2)
}

// recordedStartWorker retains a truthful PID in the admitted supervisor group.
func recordedStartWorker(executable string, args, env []string, dir string, outputLimit int) (*recordedChild, error) {
	return recordedStartProcess(executable, args, env, dir, outputLimit, 0)
}

func recordedStartProcess(executable string, args, env []string, dir string, outputLimit int, groupMode int) (*recordedChild, error) {
	out, err := newRecordedOutput(outputLimit)
	if err != nil {
		return nil, err
	}
	errOut, err := newRecordedOutput(outputLimit)
	if err != nil {
		return nil, err
	}
	stdin, err := os.Open(os.DevNull)
	if err != nil {
		return nil, err
	}
	defer stdin.Close()
	stdout, outWrite, err := os.Pipe()
	if err != nil {
		return nil, err
	}
	stderr, errWrite, err := os.Pipe()
	if err != nil {
		stdout.Close()
		outWrite.Close()
		return nil, err
	}
	closeAll := func() {
		stdout.Close()
		outWrite.Close()
		stderr.Close()
		errWrite.Close()
	}
	stdoutFD, stderrFD := int(stdout.Fd()), int(stderr.Fd())
	if err := syscall.SetNonblock(stdoutFD, true); err != nil {
		closeAll()
		return nil, err
	}
	if err := syscall.SetNonblock(stderrFD, true); err != nil {
		closeAll()
		return nil, err
	}
	proc, err := os.StartProcess(executable, append([]string{executable}, args...), &os.ProcAttr{
		Dir: dir, Env: env, Files: []*os.File{stdin, outWrite, errWrite},
		Sys: &syscall.SysProcAttr{Setpgid: groupMode == 1, Setsid: groupMode == 2},
	})
	outWrite.Close()
	errWrite.Close()
	if err != nil {
		stdout.Close()
		stderr.Close()
		return nil, fmt.Errorf("recorded helper: start: %w", err)
	}
	// Do not require a live executable here: a child may already have exited.
	// The authenticated admission stage separately pins the worker image.
	group := syscall.Getpgrp()
	if groupMode != 0 {
		group = proc.Pid
	}
	child := &recordedChild{process: proc, identity: recordedProcess{PID: proc.Pid, Group: group, Parent: os.Getpid()}, stdout: stdout, stderr: stderr, stdoutFD: stdoutFD, stderrFD: stderrFD, out: out, errOut: errOut}
	identity, _, err := recordedProcStat(proc.Pid)
	if err != nil || identity.Group != group || identity.Parent != os.Getpid() {
		// A returned owner must be retired even when identification failed.
		// Never block in Wait on a startup-error path or discard its pipes.
		return child, fmt.Errorf("recorded helper: failed to retain new child identity")
	}
	child.identity = identity
	if image, err := recordedIdentify(proc.Pid); err == nil && image.Start == identity.Start && image.Group == identity.Group && image.Parent == identity.Parent {
		child.identity = image
	}
	return child, nil
}

func (c *recordedChild) signal(signal syscall.Signal) error {
	if c.reaped {
		return nil
	}
	return recordedSignalProcess(c.identity, signal)
}

func recordedDrain(fd int, output *recordedOutput) (bool, error) {
	var buffer [32 << 10]byte
	// A flood cannot starve the controller lifeline or a runtime deadline.
	for range 8 {
		n, err := syscall.Read(fd, buffer[:])
		if n > 0 {
			output.Write(buffer[:n])
		}
		if err == syscall.EAGAIN || err == syscall.EWOULDBLOCK {
			return false, nil
		}
		if err == syscall.EINTR {
			continue
		}
		if err != nil {
			return false, fmt.Errorf("recorded helper: drain: %w", err)
		}
		if n == 0 {
			return true, nil
		}
	}
	return false, nil
}

func (c *recordedChild) poll() error {
	var err error
	if !c.outEOF {
		c.outEOF, err = recordedDrain(c.stdoutFD, c.out)
	}
	if err != nil {
		return err
	}
	if !c.errEOF {
		c.errEOF, err = recordedDrain(c.stderrFD, c.errOut)
	}
	if err != nil {
		return err
	}
	if !c.reaped {
		if _, err := recordedSignalTarget(c.identity); err != nil {
			return err
		}
		pid, err := syscall.Wait4(c.identity.PID, &c.status, syscall.WNOHANG, nil)
		if err == syscall.EINTR {
			return nil
		}
		if err != nil {
			return fmt.Errorf("recorded helper: wait: %w", err)
		}
		if pid == c.identity.PID {
			c.reaped = true
			if err := c.process.Release(); err != nil {
				return fmt.Errorf("recorded helper: release: %w", err)
			}
		}
	}
	return nil
}

// closePipes follows group retirement and final bounded draining. Closing a
// pipe or reaping a leader alone never establishes complete group cleanup.
func (c *recordedChild) closePipes() error {
	err := c.stdout.Close()
	if other := c.stderr.Close(); err == nil {
		err = other
	}
	return err
}
