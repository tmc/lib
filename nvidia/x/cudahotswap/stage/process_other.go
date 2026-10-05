//go:build !linux && !darwin

package stage

import (
	"fmt"
	"os/exec"
)

func configureCompiler(cmd *exec.Cmd) {}

func finishCompiler(cmd *exec.Cmd) error { return nil }

func readCompilerImage(string) ([]byte, error) {
	return nil, fmt.Errorf("stage libraries require Linux or Darwin")
}
