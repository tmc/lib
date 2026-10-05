package cudahotswap

import (
	"context"
	"os/exec"
	"time"
)

// Execute the prepared harness directly. Its process inherits the admitted
// native group, so the outer supervisor also owns it if this test process dies.
// Building is a separate preparation step, outside process startup timing.
func publicColdCommand(ctx context.Context, binary, file string) *exec.Cmd {
	command := exec.CommandContext(ctx, binary, "-task", "axpby", "-kernel", file, "-json")
	command.WaitDelay = 2 * time.Second
	return command
}
