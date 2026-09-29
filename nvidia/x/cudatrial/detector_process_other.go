//go:build !unix

package cudatrial

import (
	"os/exec"
	"time"
)

// Other targets use exec.CommandContext's direct-child cancellation. WaitDelay
// still bounds waiting for pipes held open by descendants.
func configureDetectorCommand(cmd *exec.Cmd) {
	cmd.WaitDelay = time.Second
}
