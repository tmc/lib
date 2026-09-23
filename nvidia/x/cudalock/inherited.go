package cudalock

import (
	"errors"
	"fmt"
	"strconv"
	"strings"
)

// ErrNotInherited reports that no ancestor process is known to hold the GPU
// lock on this process's behalf.
var ErrNotInherited = errors.New("cudalock: lock not inherited")

// Inherited reports the owner of the GPU lock an ancestor process holds and
// exported to this one, as the host's gpu-lock script and [Run] do.
//
// env is the environment lookup, normally os.Getenv. GPU_LOCK_HELD must be
// the holder's pid, the lock directory (GPU_LOCKDIR, or the default under
// HOME) must hold an owner record naming that pid, and that process must be
// alive. Anything else is ErrNotInherited. In particular GPU_LOCK_HELD=1
// names no holder and is refused: the variable carries identity, not
// presence, and "1" set by hand says nothing about whether a lock is held.
func Inherited(env func(string) string) (Owner, error) {
	if env == nil {
		return Owner{}, fmt.Errorf("%w: no environment", ErrNotInherited)
	}
	held := env("GPU_LOCK_HELD")
	pid, err := strconv.Atoi(strings.TrimSpace(held))
	if err != nil || pid <= 0 {
		return Owner{}, fmt.Errorf("%w: GPU_LOCK_HELD is %q, want the holder's pid", ErrNotInherited, held)
	}
	dir := defaultDir(env)
	owner, ok := ownerHeldBy(dir, pid)
	if !ok {
		return Owner{}, fmt.Errorf("%w: GPU_LOCK_HELD is %d but the owner record in %s does not name it", ErrNotInherited, pid, dir)
	}
	if !isAlive(pid) {
		return Owner{}, fmt.Errorf("%w: holder %d is not alive", ErrNotInherited, pid)
	}
	return owner, nil
}
