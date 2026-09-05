//go:build unix

package cudasanitizer

import (
	"os/exec"
	"syscall"
)

// setGroup puts the run in its own process group so that a timeout can kill
// compute-sanitizer and the application it launched together.
func setGroup(cmd *exec.Cmd) {
	if cmd.SysProcAttr == nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{}
	}
	cmd.SysProcAttr.Setpgid = true
	cmd.SysProcAttr.Pgid = 0
}

// killGroup kills the whole process group. Killing only the direct child can
// leave the instrumented application running, still holding the output pipes,
// which would turn a timeout into a hang.
func killGroup(cmd *exec.Cmd) error {
	if cmd.Process == nil {
		return nil
	}
	pid := cmd.Process.Pid
	attrs := cmd.SysProcAttr
	if pid <= 1 || attrs == nil || !attrs.Setpgid || attrs.Pgid != 0 {
		// Only kill a whole group when setGroup created it for this command.
		return cmd.Process.Kill()
	}
	// Setpgid with Pgid zero makes the child's PID its group ID. The leader
	// can already be reaped while descendants remain in this group, so do
	// not require Getpgid(pid) to succeed before signaling it.
	err := syscall.Kill(-pid, syscall.SIGKILL)
	if err == syscall.ESRCH {
		return nil
	}
	return err
}
