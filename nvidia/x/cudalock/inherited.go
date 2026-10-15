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
// alive. Verification does not wait for the namespace guard or process state.
// A busy or unavailable check returns ErrNotInherited because ownership is
// unverified; it does not establish that no owner exists. The caller may retry.
// Anything else is ErrNotInherited. In particular GPU_LOCK_HELD=1
// names no holder and is refused: the variable carries identity, not
// presence, and "1" set by hand says nothing about whether a lock is held.
func Inherited(env func(string) string) (owner Owner, retErr error) {
	if env == nil {
		return Owner{}, fmt.Errorf("%w: no environment", ErrNotInherited)
	}
	held := env("GPU_LOCK_HELD")
	pid, err := strconv.Atoi(strings.TrimSpace(held))
	if err != nil || pid <= 0 {
		return Owner{}, fmt.Errorf("%w: GPU_LOCK_HELD is %q, want the holder's pid", ErrNotInherited, held)
	}
	dir := defaultDir(env)
	g, err := openNamespace(dir)
	if err != nil {
		return Owner{}, fmt.Errorf("%w: %w", ErrNotInherited, err)
	}
	defer func() {
		if err := g.file.Close(); err != nil {
			owner = Owner{}
			retErr = errors.Join(retErr, fmt.Errorf("%w: close guard: %w", ErrNotInherited, err))
		}
	}()
	if err := g.tryLock(); err != nil {
		return Owner{}, fmt.Errorf("%w: %w", ErrNotInherited, err)
	}
	defer func() {
		if err := guardUnlock(g.file); err != nil {
			owner = Owner{}
			retErr = errors.Join(retErr, fmt.Errorf("%w: unlock guard: %w", ErrNotInherited, err))
		}
	}()
	var data string
	owner, data, err = ownerHeldBy(g.dir, pid)
	if err != nil {
		return Owner{}, fmt.Errorf("%w: %w", ErrNotInherited, err)
	}
	if !processMu.TryLock() {
		return Owner{}, fmt.Errorf("%w: process ownership state is busy", ErrNotInherited)
	}
	hold := processHolds[g.dir]
	verified := hold == nil || hold.count <= 0 || (!hold.cleanupOnly && hold.retired == "" && g.matches(hold.namespace) && hold.ownerData == data)
	processMu.Unlock()
	if !verified {
		return Owner{}, fmt.Errorf("%w: retained process ownership is unverified", ErrNotInherited)
	}
	if !isAlive(pid) {
		return Owner{}, fmt.Errorf("%w: holder %d is not alive", ErrNotInherited, pid)
	}
	return owner, nil
}
