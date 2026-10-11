package cudalock

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	// DefaultMaxHold is the fallback threshold for evaluating dead lock holders.
	DefaultMaxHold = 3600 * time.Second
	// DefaultPollInterval is the delay between acquisition attempts.
	DefaultPollInterval = 250 * time.Millisecond

	// emptyOwnerGrace is the threshold before reclaiming empty or unparsable owner files.
	emptyOwnerGrace = 15 * time.Second
)

var (
	// ErrTimeout is returned when lock acquisition exceeds maximum wait duration.
	ErrTimeout = errors.New("cudalock: timed out waiting for lock")
	// ErrParentUnwritable is returned when the parent directory of the lock is missing or unwritable.
	ErrParentUnwritable = errors.New("cudalock: lock parent directory missing or unwritable")
)

// Process-wide reentrancy tracking per directory.
type dirHold struct {
	count     int
	holderPID int
	inherited bool
	owner     Owner
}

var (
	processMu    sync.Mutex
	processHolds = make(map[string]*dirHold)
	reclaimHook  func()             // testing hook for deterministic TOCTOU race simulation
	restoreHook  func(aside string) // testing hook for deterministic restore failure simulation
)

// DefaultDir returns the default lock directory.
func DefaultDir() string {
	return defaultDir(os.Getenv)
}

func defaultDir(env func(string) string) string {
	if dir := env("GPU_LOCKDIR"); dir != "" {
		return dir
	}
	home := env("HOME")
	if home == "" {
		home = "/tmp"
	}
	return filepath.Join(home, "ml-explore", ".gpu-lock", "lock")
}

// Option configures lock acquisition.
type Option func(*config)

type config struct {
	dir          string
	label        string
	pollInterval time.Duration
	maxHold      time.Duration
	noReclaim    bool
}

// WithDir sets the target lock directory.
func WithDir(dir string) Option {
	return func(c *config) {
		c.dir = dir
	}
}

// WithLabel sets a descriptive human label for the lock holder.
func WithLabel(label string) Option {
	return func(c *config) {
		c.label = label
	}
}

// WithPollInterval sets the interval between acquisition retries.
func WithPollInterval(d time.Duration) Option {
	return func(c *config) {
		c.pollInterval = d
	}
}

// WithMaxHold sets the grace threshold for evaluating stale dead holders.
func WithMaxHold(d time.Duration) Option {
	return func(c *config) {
		c.maxHold = d
	}
}

// WithNoReclaim refuses to remove or replace a contended lock, including dead,
// malformed and ownerless locks. Acquisition waits until the caller's context
// expires or the lock is released. The owner is published after exclusive
// directory creation; no work should start before [Inherited] verifies it.
// Publication failure leaves the owned directory in place for inspection.
func WithNoReclaim() Option { return func(c *config) { c.noReclaim = true } }

// A Handle represents an acquired GPU lock.
type Handle struct {
	mu       sync.Mutex
	dir      string
	owner    Owner
	released bool
	isNested bool
}

// Dir returns the directory path of the acquired lock.
func (h *Handle) Dir() string {
	return h.dir
}

// Owner returns the owner information recorded for this lock hold.
func (h *Handle) Owner() Owner {
	return h.owner
}

// IsNested returns true if this handle was acquired re-entrantly within the
// same process or inherited from an ancestor process holding GPU_LOCK_HELD.
func (h *Handle) IsNested() bool {
	return h.isNested
}

// Release releases the lock. Release must fail toward stuck: if removing the
// lock directory fails, the directory remains and an error is returned.
func (h *Handle) Release() error {
	h.mu.Lock()
	defer h.mu.Unlock()

	if h.released {
		return nil
	}

	cleanDir := filepath.Clean(h.dir)

	processMu.Lock()
	defer processMu.Unlock()

	hold, ok := processHolds[cleanDir]
	if !ok || hold.count <= 0 {
		h.released = true
		return nil
	}

	hold.count--
	h.released = true
	if hold.count > 0 {
		return nil
	}

	inherited := hold.inherited
	delete(processHolds, cleanDir)

	if inherited {
		return nil
	}

	// Root release for this directory.
	// The reference script's release() deletes the lock only if the owner still
	// matches our own identity. If another process reclaimed, moved, or displaced
	// our lock, we must not delete whatever is currently in place.
	ownerFile := filepath.Join(h.dir, "owner")
	data, err := os.ReadFile(ownerFile)
	if err != nil {
		return fmt.Errorf("cudalock: release aborted: owner file missing or unreadable: %w", err)
	}
	if string(data) != h.owner.String() {
		return fmt.Errorf("cudalock: release aborted: lock owner displaced (found %q, expected %q)", strings.TrimSpace(string(data)), strings.TrimSpace(h.owner.String()))
	}

	// Rename aside to avoid races with concurrent re-acquirers while deleting contents
	parent := filepath.Dir(h.dir)
	aside := filepath.Join(parent, fmt.Sprintf(".gpu-lock-released.%d.%d", os.Getpid(), time.Now().UnixNano()))
	if err := os.Rename(h.dir, aside); err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil
		}
		return fmt.Errorf("cudalock: release failed to move lock aside: %w", err)
	}
	_ = os.RemoveAll(aside)

	if cleanDir == filepath.Clean(DefaultDir()) {
		_ = os.Unsetenv("GPU_LOCK_HELD")
	}
	return nil
}

// Acquire acquires exclusive GPU access via the atomic directory mutex.
// It blocks until the lock is acquired, the context is canceled, or an unrecoverable
// preflight error occurs.
func Acquire(ctx context.Context, opts ...Option) (*Handle, error) {
	cfg := config{
		dir:          DefaultDir(),
		label:        os.Getenv("GPU_LOCK_LABEL"),
		pollInterval: DefaultPollInterval,
		maxHold:      DefaultMaxHold,
	}
	for _, opt := range opts {
		opt(&cfg)
	}
	if cfg.label == "" {
		cfg.label = strconv.Itoa(os.Getpid())
	}
	if cfg.pollInterval <= 0 {
		cfg.pollInterval = DefaultPollInterval
	}

	cleanDir := filepath.Clean(cfg.dir)

	// Check reentrancy within this process or inherited from parent gpu-lock.
	processMu.Lock()
	if hold, ok := processHolds[cleanDir]; ok && hold.count > 0 {
		hold.count++
		processMu.Unlock()
		return &Handle{
			dir:      cfg.dir,
			isNested: true,
			owner:    hold.owner,
		}, nil
	}

	envHeld := os.Getenv("GPU_LOCK_HELD")
	if envHeld != "" && cleanDir == filepath.Clean(DefaultDir()) {
		if heldPID, err := strconv.Atoi(strings.TrimSpace(envHeld)); err == nil && heldPID > 0 {
			// A live PID in the environment is not evidence that the lock is
			// held. PID 1 is alive on every Unix, and the invocation
			// "gpu-lock env GPU_LOCK_HELD=1 cmd" circulated among agents, which
			// would take this branch and run unlocked believing an ancestor
			// covered it -- the fail-open direction. Corroborate against the
			// lock itself: every holder that exports this variable also wrote
			// its own PID to the owner file (this file's acquire path below,
			// Run, and the host's gpu-lock alike), so the two must agree. A
			// missing directory, a disagreeing owner, or a foreign host falls
			// through to a normal acquire, which blocks on a real holder.
			if owner, err := ownerHeldBy(cfg.dir, heldPID); err == nil && isAlive(heldPID) {
				// Inherited from an ancestor process that ran us under gpu-lock
				if owner.Label == "" {
					owner.Label = cfg.label
				}
				processHolds[cleanDir] = &dirHold{
					count:     1,
					holderPID: heldPID,
					inherited: true,
					owner:     owner,
				}
				processMu.Unlock()

				return &Handle{
					dir:      cfg.dir,
					isNested: true,
					owner:    owner,
				}, nil
			}
		}
	}
	processMu.Unlock()

	// Preflight parent directory
	parent := filepath.Dir(cfg.dir)
	if err := os.MkdirAll(parent, 0755); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrParentUnwritable, err)
	}
	// Verify write permission by creating a temporary probe file
	probe := filepath.Join(parent, fmt.Sprintf(".probe-%d", os.Getpid()))
	if err := os.WriteFile(probe, []byte("ok"), 0600); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrParentUnwritable, err)
	}
	_ = os.Remove(probe)

	myHost := shortHostname()
	myPID := os.Getpid()

	ticker := time.NewTicker(cfg.pollInterval)
	defer ticker.Stop()

	for {
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("cudalock: %w", err)
		}

		if cfg.noReclaim {
			if mkdirErr := os.Mkdir(cfg.dir, 0755); mkdirErr == nil {
				owner := Owner{PID: myPID, Host: myHost, Label: cfg.label, Epoch: time.Now().Unix(), At: time.Now()}
				f, err := os.OpenFile(filepath.Join(cfg.dir, "owner"), os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0644)
				if err != nil {
					return nil, fmt.Errorf("cudalock: owner publication failed, directory retained: %w", err)
				}
				_, writeErr := f.WriteString(owner.String())
				syncErr := f.Sync()
				closeErr := f.Close()
				if err := errors.Join(writeErr, syncErr, closeErr); err != nil {
					return nil, fmt.Errorf("cudalock: owner publication failed, directory retained: %w", err)
				}
				if cleanDir == filepath.Clean(DefaultDir()) {
					_ = os.Setenv("GPU_LOCK_HELD", strconv.Itoa(myPID))
				}
				processMu.Lock()
				processHolds[cleanDir] = &dirHold{count: 1, holderPID: myPID, owner: owner}
				processMu.Unlock()
				return &Handle{dir: cfg.dir, owner: owner}, nil
			} else if !errors.Is(mkdirErr, os.ErrExist) {
				return nil, fmt.Errorf("cudalock: create lock directory: %w", mkdirErr)
			}
		} else {
			// Attempt atomic acquisition via staging directory rename
			staging, stagingErr := os.MkdirTemp(parent, ".gpu-lock-acquire.*")
			if stagingErr == nil {
				owner := Owner{
					PID:   myPID,
					Host:  myHost,
					Label: cfg.label,
					Epoch: time.Now().Unix(),
					At:    time.Now(),
				}
				ownerFile := filepath.Join(staging, "owner")
				if writeErr := os.WriteFile(ownerFile, []byte(owner.String()), 0644); writeErr == nil {
					if renameErr := os.Rename(staging, cfg.dir); renameErr == nil {
						// Successfully acquired directory mutex
						if cleanDir == filepath.Clean(DefaultDir()) {
							_ = os.Setenv("GPU_LOCK_HELD", strconv.Itoa(myPID))
						}
						processMu.Lock()
						processHolds[cleanDir] = &dirHold{
							count:     1,
							holderPID: myPID,
							inherited: false,
							owner:     owner,
						}
						processMu.Unlock()

						return &Handle{
							dir:   cfg.dir,
							owner: owner,
						}, nil
					}
				}
				_ = os.RemoveAll(staging)
			}

			// Mutex directory already exists: check owner record
			ownerFile := filepath.Join(cfg.dir, "owner")
			data, readErr := os.ReadFile(ownerFile)
			if readErr == nil {
				existingOwner, parseErr := ParseOwner(data)
				if parseErr == nil {
					// Safety check 1: Foreign host locks are NEVER reclaimed.
					if existingOwner.Host != "" && existingOwner.Host != myHost {
						goto wait
					}

					// Safety check 2: Process liveness check.
					// If process is alive, never reclaim regardless of age!
					if isAlive(existingOwner.PID) {
						goto wait
					}

					// Safety check 3: Process is confirmed dead on our host.
					// Reclaim the abandoned lock directory atomically.
					reclaimed, reclaimErr := reclaim(cfg.dir, data)
					if reclaimErr != nil {
						return nil, reclaimErr
					}
					if reclaimed {
						continue
					}
				} else {
					// Owner file exists but is empty or unparsable.
					// Treat an empty or unparsable owner file like a missing one:
					// reclaim it once it has been older than the grace period,
					// measured from the owner file's mtime.
					if fi, statErr := os.Stat(ownerFile); statErr == nil {
						if time.Since(fi.ModTime()) > emptyOwnerGrace {
							reclaimed, reclaimErr := reclaim(cfg.dir, data)
							if reclaimErr != nil {
								return nil, reclaimErr
							}
							if reclaimed {
								continue
							}
						}
					}
				}
			} else if errors.Is(readErr, os.ErrNotExist) {
				// Lock directory exists but has no owner file.
				// Could be left by a crashed older creator.
				if fi, statErr := os.Stat(cfg.dir); statErr == nil {
					if time.Since(fi.ModTime()) > emptyOwnerGrace {
						// Directory abandoned for > grace period with no owner file: reclaim.
						reclaimed, reclaimErr := reclaim(cfg.dir, nil)
						if reclaimErr != nil {
							return nil, reclaimErr
						}
						if reclaimed {
							continue
						}
					}
				}
			}

		}

	wait:
		select {
		case <-ctx.Done():
			return nil, fmt.Errorf("cudalock: %w", ctx.Err())
		case <-ticker.C:
		}
	}
}

// reclaim atomically moves staleDir aside, verifies that its owner file still
// matches expectedData, and removes it. If the owner changed or another reclaimer
// intervened, the displaced directory is moved back intact.
func reclaim(dir string, expectedData []byte) (bool, error) {
	if reclaimHook != nil {
		reclaimHook()
	}
	parent := filepath.Dir(dir)
	aside, err := os.MkdirTemp(parent, ".gpu-lock-stale.*")
	if err != nil {
		return false, nil
	}
	_ = os.Remove(aside)

	// Atomically move dir aside
	if err := os.Rename(dir, aside); err != nil {
		return false, nil
	}

	ownerFile := filepath.Join(aside, "owner")
	newData, readErr := os.ReadFile(ownerFile)

	matches := false
	if expectedData == nil {
		if errors.Is(readErr, os.ErrNotExist) {
			matches = true
		}
	} else {
		if readErr == nil && bytes.Equal(newData, expectedData) {
			matches = true
		}
	}

	if matches {
		_ = os.RemoveAll(aside)
		return true, nil
	}

	// Not the lock we diagnosed: a concurrent reclaimer or new holder won.
	// Put it back exactly as we found it.
	if restoreHook != nil {
		restoreHook(aside)
	}
	if err := os.Rename(aside, dir); err != nil {
		return false, fmt.Errorf("cudalock: could not restore live lock displaced during reclaim: %s: %w", aside, err)
	}
	return false, nil
}

// WithLock runs fn while holding exclusive GPU access.
func WithLock(ctx context.Context, fn func() error, opts ...Option) error {
	handle, err := Acquire(ctx, opts...)
	if err != nil {
		return err
	}
	defer handle.Release()
	return fn()
}

// ownerHeldBy reports whether dir is a lock directory whose owner file names
// pid and belongs to the current host.
func ownerHeldBy(dir string, pid int) (Owner, error) {
	data, err := os.ReadFile(filepath.Join(dir, "owner"))
	if err != nil {
		return Owner{}, fmt.Errorf("reading owner in %s: %w", dir, err)
	}
	owner, err := ParseOwner(data)
	if err != nil {
		return Owner{}, fmt.Errorf("parsing owner in %s: %w", dir, err)
	}
	if owner.PID != pid {
		return Owner{}, fmt.Errorf("owner record in %s names pid %d, want %d", dir, owner.PID, pid)
	}
	if owner.Host != "" && owner.Host != shortHostname() {
		return Owner{}, fmt.Errorf("owner host %q differs from current host %q", owner.Host, shortHostname())
	}
	return owner, nil
}

// Run executes command args under the GPU lock, exporting GPU_LOCK_HELD.
func Run(ctx context.Context, args []string, opts ...Option) (int, error) {
	if len(args) == 0 {
		return 2, errors.New("cudalock: no command given")
	}

	handle, err := Acquire(ctx, opts...)
	if err != nil {
		return 1, err
	}
	defer handle.Release()

	cmd := exec.CommandContext(ctx, args[0], args[1:]...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Env = append(os.Environ(), fmt.Sprintf("GPU_LOCK_HELD=%d", handle.Owner().PID))
	if handle.dir != DefaultDir() {
		cmd.Env = append(cmd.Env, fmt.Sprintf("GPU_LOCKDIR=%s", handle.dir))
	}

	if err := cmd.Run(); err != nil {
		var exitErr *exec.ExitError
		if errors.As(err, &exitErr) {
			return exitErr.ExitCode(), nil
		}
		return 1, err
	}
	return 0, nil
}
