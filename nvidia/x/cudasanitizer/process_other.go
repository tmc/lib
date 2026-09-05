//go:build !unix

package cudasanitizer

import "os/exec"

// setGroup is a no-op on systems without process groups.
func setGroup(cmd *exec.Cmd) {}

// killGroup kills the direct child. Descendants are not reaped here, so a
// timeout on such a system can leave the instrumented application running.
func killGroup(cmd *exec.Cmd) error {
	if cmd.Process == nil {
		return nil
	}
	return cmd.Process.Kill()
}
