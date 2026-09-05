//go:build !unix && !windows

package cudasanitizer

import "os/exec"

// Other platforms have only direct-child cleanup.
type processGroup struct{ cmd *exec.Cmd }

func startGroup(cmd *exec.Cmd) (*processGroup, error) {
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	return &processGroup{cmd: cmd}, nil
}
func (g *processGroup) kill() error { return g.cmd.Process.Kill() }
func (g *processGroup) close()      { _ = g.kill() }
