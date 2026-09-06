//go:build !unix && !windows

package cudasanitizer

import (
	"errors"
	"os"
	"os/exec"
)

// Other platforms have only direct-child cleanup.
type processGroup struct{ cmd *exec.Cmd }

func startGroup(cmd *exec.Cmd) (*processGroup, error) {
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	return &processGroup{cmd: cmd}, nil
}
func (g *processGroup) kill() error {
	err := g.cmd.Process.Kill()
	if errors.Is(err, os.ErrProcessDone) {
		return nil
	}
	return err
}
func (g *processGroup) close() error { return g.kill() }
