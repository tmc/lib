package cudalock

import (
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
)

var (
	// ErrTimeout is returned when lock acquisition exceeds maximum wait duration.
	ErrTimeout = errors.New("cudalock: timed out waiting for lock")
	// ErrParentUnwritable is returned when the parent directory of the lock is missing or unwritable.
	ErrParentUnwritable = errors.New("cudalock: lock parent directory missing or unwritable")
	// ErrForeignOwner is returned when attempting to reclaim a lock owned by a different host.
	ErrForeignOwner = errors.New("cudalock: lock is held by a foreign host")
)

// Process-wide reentrancy tracking.
var (
	processMu        sync.Mutex
	processHoldCount int
	processActiveDir string
	processHolderPID int
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

	processMu.Lock()
	defer processMu.Unlock()

	if h.isNested {
		processHoldCount--
		if processHoldCount == 0 {
			processActiveDir = ""
			processHolderPID = 0
		}
		h.released = true
		return nil
	}

	processHoldCount--
	if processHoldCount > 0 {
		h.released = true
		return nil
	}

	// Root release
	ownerFile := filepath.Join(h.dir, "owner")
	_ = os.Remove(ownerFile)
	if err := os.Remove(h.dir); err != nil && !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("cudalock: release failed: %w", err)
	}

	_ = os.Unsetenv("GPU_LOCK_HELD")
	processActiveDir = ""
	processHolderPID = 0
	h.released = true
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

	// Check reentrancy within this process or inherited from parent gpu-lock.
	processMu.Lock()
	envHeld := os.Getenv("GPU_LOCK_HELD")
	if processHoldCount > 0 && processActiveDir == cfg.dir {
		processHoldCount++
		processMu.Unlock()
		return &Handle{
			dir:      cfg.dir,
			isNested: true,
			owner: Owner{
				PID:   processHolderPID,
				Host:  shortHostname(),
				Label: cfg.label,
			},
		}, nil
	}
	if envHeld != "" && processActiveDir == "" && cfg.dir == DefaultDir() {
		if heldPID, err := strconv.Atoi(strings.TrimSpace(envHeld)); err == nil && heldPID > 0 {
			// A live PID in the environment is not evidence that the lock is
			// held. PID 1 is alive on every Unix, and the invocation
			// "gpu-lock env GPU_LOCK_HELD=1 cmd" circulated among agents, which
			// would take this branch and run unlocked believing an ancestor
			// covered it -- the fail-open direction. Corroborate against the
			// lock itself: every holder that exports this variable also wrote
			// its own PID to the owner file (this file's acquire path below,
			// Run, and the host's gpu-lock alike), so the two must agree. A
			// missing directory or a disagreeing owner falls through to a
			// normal acquire, which blocks on a real holder.
			if owner, ok := ownerHeldBy(cfg.dir, heldPID); ok && isAlive(heldPID) {
				// Inherited from an ancestor process that ran us under gpu-lock
				processHoldCount++
				processActiveDir = cfg.dir
				processHolderPID = heldPID
				processMu.Unlock()

				if owner.Label == "" {
					owner.Label = cfg.label
				}
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

		// Attempt atomic acquisition
		if err := os.Mkdir(cfg.dir, 0755); err == nil {
			// Successfully created directory
			owner := Owner{
				PID:   myPID,
				Host:  myHost,
				Label: cfg.label,
				Epoch: time.Now().Unix(),
				At:    time.Now(),
			}
			ownerFile := filepath.Join(cfg.dir, "owner")
			if writeErr := os.WriteFile(ownerFile, []byte(owner.String()), 0644); writeErr != nil {
				// Revert on write failure
				_ = os.Remove(cfg.dir)
				return nil, fmt.Errorf("cudalock: failed writing owner: %w", writeErr)
			}

			_ = os.Setenv("GPU_LOCK_HELD", strconv.Itoa(myPID))
			processMu.Lock()
			processHoldCount = 1
			processActiveDir = cfg.dir
			processHolderPID = myPID
			processMu.Unlock()

			return &Handle{
				dir:   cfg.dir,
				owner: owner,
			}, nil
		}

		// Mutex directory already exists: check owner record for dead process
		ownerFile := filepath.Join(cfg.dir, "owner")
		data, readErr := os.ReadFile(ownerFile)
		if readErr == nil {
			if existingOwner, parseErr := ParseOwner(data); parseErr == nil {
				// Safety check 1: Foreign host locks are NEVER reclaimed.
				if existingOwner.Host != "" && existingOwner.Host != myHost {
					// Owned by another host; must wait for that host to release.
					goto wait
				}

				// Safety check 2: Process liveness check.
				// If process is alive, never reclaim regardless of age!
				if isAlive(existingOwner.PID) {
					goto wait
				}

				// Safety check 3: Process is confirmed dead on our host.
				// Reclaim the abandoned lock directory.
				_ = os.Remove(ownerFile)
				if removeErr := os.Remove(cfg.dir); removeErr == nil {
					// Reclaimed cleanly; loop immediately to acquire
					continue
				}
			}
		} else if errors.Is(readErr, os.ErrNotExist) {
			// Lock directory exists but has no owner file.
			// Could be mid-acquire or left by a crashed creator.
			if fi, statErr := os.Stat(cfg.dir); statErr == nil {
				if time.Since(fi.ModTime()) > 15*time.Second {
					// Directory abandoned for > 15s with no owner file: reclaim.
					_ = os.Remove(cfg.dir)
					continue
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
// pid. It is the corroboration for an inherited GPU_LOCK_HELD: the environment
// says who holds the lock, and this says whether the lock agrees.
func ownerHeldBy(dir string, pid int) (Owner, bool) {
	data, err := os.ReadFile(filepath.Join(dir, "owner"))
	if err != nil {
		return Owner{}, false
	}
	owner, err := ParseOwner(data)
	if err != nil || owner.PID != pid {
		return Owner{}, false
	}
	return owner, true
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
	cmd.Env = append(os.Environ(), fmt.Sprintf("GPU_LOCK_HELD=%d", os.Getpid()))
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
