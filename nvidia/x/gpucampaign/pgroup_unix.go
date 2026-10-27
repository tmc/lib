//go:build unix

package gpucampaign

import (
	"os"
	"os/exec"
	"syscall"
)

func setProcessGroup(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
}

func killProcessGroup(p *os.Process) error {
	return syscall.Kill(-p.Pid, syscall.SIGKILL)
}
