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
}

// killGroup kills the whole process group. Killing only the direct child can
// leave the instrumented application running, still holding the output pipes,
// which would turn a timeout into a hang.
func killGroup(cmd *exec.Cmd) error {
	if cmd.Process == nil {
		return nil
	}
	pgid, err := syscall.Getpgid(cmd.Process.Pid)
	if err != nil || pgid <= 1 || pgid != cmd.Process.Pid {
		// The group is not ours to kill wholesale; fall back to the child.
		return cmd.Process.Kill()
	}
	return syscall.Kill(-pgid, syscall.SIGKILL)
}
