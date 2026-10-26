//go:build !unix

package gpucampaign

import (
	"os"
	"os/exec"
)

func setProcessGroup(*exec.Cmd) {}

func killProcessGroup(p *os.Process) error { return p.Kill() }
