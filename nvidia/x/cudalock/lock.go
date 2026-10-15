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
	count        int
	holderPID    int
	inherited    bool
	owner        Owner
	namespace    *namespace
	ownerData    string
	retired      string
	retiredInfo  os.FileInfo
	ownerRemoved bool
	cleanupOnly  bool
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
	hold     *dirHold
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

// Release releases the lock. A failed content removal restores the original
// canonical owner when its exact snapshot remains intact. If restoration is
// impossible, the handle retains only pending retirement cleanup and cannot
// authorize another acquisition. Release may be retried; it never removes a
// newer canonical owner while cleaning an earlier retirement.
func (h *Handle) Release() (retErr error) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.released {
		return nil
	}
	processMu.Lock()
	hold := processHolds[h.dir]
	if hold == nil || hold != h.hold || hold.count <= 0 {
		h.released = true
		processMu.Unlock()
		return nil
	}
	if hold.retired == "" && (hold.count > 1 || hold.inherited) {
		hold.count--
		h.released = true
		if hold.count == 0 {
			delete(processHolds, h.dir)
		}
		processMu.Unlock()
		return nil
	}
	processMu.Unlock()

	g, err := openNamespace(h.dir)
	if err != nil {
		return err
	}
	defer func() { retErr = errors.Join(retErr, g.file.Close()) }()
	if err := g.lock(context.Background(), DefaultPollInterval); err != nil {
		return err
	}
	defer func() { retErr = errors.Join(retErr, guardUnlock(g.file)) }()
	processMu.Lock()
	defer processMu.Unlock()
	hold = processHolds[h.dir]
	if hold == nil || hold != h.hold || hold.count <= 0 {
		h.released = true
		return nil
	}
	if !g.matches(hold.namespace) {
		return errors.New("cudalock: root retirement guard identity changed")
	}
	if hold.retired == "" && hold.count > 1 {
		hold.count--
		h.released = true
		return nil
	}
	return h.releaseRoot(g, hold, os.Remove)
}

// releaseRoot runs under h.mu, the namespace guard and processMu. removeDir is
// the final, empty-directory removal; the ordinary path uses os.Remove.
func (h *Handle) releaseRoot(g *namespaceGuard, hold *dirHold, removeDir func(string) error) error {
	if hold.retired == "" {
		data, err := os.ReadFile(filepath.Join(h.dir, "owner"))
		if err != nil {
			return fmt.Errorf("cudalock: release aborted: owner unreadable: %w", err)
		}
		if string(data) != hold.ownerData {
			return errors.New("cudalock: release aborted: lock owner displaced")
		}
		info, err := os.Lstat(h.dir)
		if err != nil {
			return fmt.Errorf("cudalock: inspect retiring lock: %w", err)
		}
		if err := g.check(); err != nil {
			return err
		}
		aside, err := os.MkdirTemp(g.parent, ".gpu-lock-released.*")
		if err != nil {
			return fmt.Errorf("cudalock: reserve retirement: %w", err)
		}
		if err := os.Remove(aside); err != nil {
			return fmt.Errorf("cudalock: prepare retirement: %w", err)
		}
		if err := os.Rename(h.dir, aside); err != nil {
			return fmt.Errorf("cudalock: move retiring lock: %w", err)
		}
		hold.retired, hold.retiredInfo = aside, info
	}
	if err := removeRetired(hold, removeDir); err != nil {
		primary := fmt.Errorf("cudalock: remove retired lock: %w", err)
		if restoreErr := restoreRetired(g, hold); restoreErr != nil {
			return errors.Join(primary, fmt.Errorf("cudalock: no canonical ownership; cleanup pending at %s: %w", hold.retired, restoreErr))
		}
		return primary
	}
	delete(processHolds, h.dir)
	hold.count = 0
	h.released = true
	// A canceled admission never exported authority. Its cleanup must leave
	// any preexisting or subsequently changed environment value intact.
	if !hold.cleanupOnly && isDefaultDir(h.dir) && os.Getenv("GPU_LOCK_HELD") == strconv.Itoa(h.owner.PID) {
		_ = os.Unsetenv("GPU_LOCK_HELD")
	}
	return nil
}

// Preserve owner until all other contents have been removed, so an ordinary
// content-removal error can restore its exact original owner. A later failure
// after owner removal has explicit pending-retirement state instead.
func removeRetired(hold *dirHold, removeDir func(string) error) error {
	info, err := os.Lstat(hold.retired)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil || !os.SameFile(info, hold.retiredInfo) {
		return errors.New("retired directory identity changed")
	}
	ownerPath := filepath.Join(hold.retired, "owner")
	data, ownerErr := os.ReadFile(ownerPath)
	if hold.ownerRemoved {
		if !errors.Is(ownerErr, os.ErrNotExist) {
			return errors.New("owner appeared after retirement")
		}
	} else {
		if ownerErr != nil || string(data) != hold.ownerData {
			return errors.New("retired owner identity changed")
		}
		entries, err := os.ReadDir(hold.retired)
		if err != nil {
			return err
		}
		for _, entry := range entries {
			if entry.Name() != "owner" {
				if err := os.RemoveAll(filepath.Join(hold.retired, entry.Name())); err != nil {
					return err
				}
			}
		}
		if err := os.Remove(ownerPath); err != nil {
			return err
		}
		hold.ownerRemoved = true
	}
	return removeDir(hold.retired)
}

func restoreRetired(g *namespaceGuard, hold *dirHold) error {
	if hold.ownerRemoved {
		return errors.New("original owner was already removed")
	}
	if err := g.check(); err != nil {
		return err
	}
	info, err := os.Lstat(hold.retired)
	if err != nil || !os.SameFile(info, hold.retiredInfo) {
		return errors.New("cannot restore changed retired directory")
	}
	data, err := os.ReadFile(filepath.Join(hold.retired, "owner"))
	if err != nil || string(data) != hold.ownerData {
		return errors.New("cannot restore changed retired owner")
	}
	if _, err := os.Lstat(g.dir); !errors.Is(err, os.ErrNotExist) {
		return errors.New("cannot restore over occupied canonical path")
	}
	if err := os.Rename(hold.retired, g.dir); err != nil {
		return err
	}
	hold.retired, hold.retiredInfo = "", nil
	return nil
}

// Acquire acquires exclusive GPU access via the atomic directory mutex.
// It blocks until the lock is acquired, the context is canceled, or an unrecoverable
// preflight error occurs. Cancellation after root publication retires that exact
// publication before returning. If retirement fails, the returned error wraps
// the context and cleanup errors and implements io.Closer for retrying cleanup;
// it does not authorize work. A caller can obtain it with errors.As.
func Acquire(ctx context.Context, opts ...Option) (handle *Handle, retErr error) {
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

	if err := os.MkdirAll(filepath.Dir(cfg.dir), 0755); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrParentUnwritable, err)
	}
	g, err := openNamespace(cfg.dir)
	if err != nil {
		return nil, err
	}
	defer func() {
		if err := g.file.Close(); err != nil {
			forgetHandle(handle)
			handle = nil
			retErr = errors.Join(retErr, fmt.Errorf("cudalock: close guard: %w", err))
		}
	}()
	cfg.dir = g.dir

	ticker := time.NewTicker(cfg.pollInterval)
	defer ticker.Stop()
	for {
		if err := g.lock(ctx, cfg.pollInterval); err != nil {
			return nil, err
		}
		h, acquireErr := acquireGuarded(ctx, cfg, g)
		if h != nil {
			if err := ctx.Err(); err != nil {
				acquireErr = errors.Join(acquireErr, cancelAdmission(h, g, err, os.Remove))
				h = nil
			}
		}
		unlockErr := guardUnlock(g.file)
		if err := errors.Join(acquireErr, unlockErr); err != nil {
			forgetHandle(h)
			return nil, err
		}
		if h != nil {
			if !h.isNested && isDefaultDir(h.dir) {
				_ = os.Setenv("GPU_LOCK_HELD", strconv.Itoa(h.owner.PID))
			}
			return h, nil
		}
		select {
		case <-ctx.Done():
			return nil, fmt.Errorf("cudalock: %w", ctx.Err())
		case <-ticker.C:
		}
	}
}

func acquireGuarded(ctx context.Context, cfg config, g *namespaceGuard) (*Handle, error) {
	cleanDir := cfg.dir
	parent := filepath.Dir(cfg.dir)
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("cudalock: %w", err)
	}
	// Every admission, including process-local and inherited references, reads
	// the current owner while holding the same namespace guard.
	processMu.Lock()
	if err := ctx.Err(); err != nil {
		processMu.Unlock()
		return nil, fmt.Errorf("cudalock: %w", err)
	}
	if hold := processHolds[cleanDir]; hold != nil && hold.count > 0 {
		if hold.retired != "" || hold.cleanupOnly {
			processMu.Unlock()
			return nil, errors.New("cudalock: prior retirement cleanup pending; no work admission")
		}
		if !g.matches(hold.namespace) {
			processMu.Unlock()
			return nil, errors.New("cudalock: held guard identity changed")
		}
		_, data, err := ownerHeldBy(cfg.dir, hold.owner.PID)
		if err != nil || data != hold.ownerData || !isAlive(hold.owner.PID) {
			processMu.Unlock()
			return nil, errors.New("cudalock: retained owner is no longer verified")
		}
		if err := ctx.Err(); err != nil {
			processMu.Unlock()
			return nil, fmt.Errorf("cudalock: %w", err)
		}
		hold.count++
		processMu.Unlock()
		return &Handle{dir: cfg.dir, owner: hold.owner, isNested: true, hold: hold}, nil
	}
	if envHeld := os.Getenv("GPU_LOCK_HELD"); envHeld != "" && isDefaultDir(cleanDir) {
		if pid, err := strconv.Atoi(strings.TrimSpace(envHeld)); err == nil && pid > 0 {
			if owner, data, err := ownerHeldBy(cfg.dir, pid); err == nil && isAlive(pid) {
				if owner.Label == "" {
					owner.Label = cfg.label
				}
				if err := ctx.Err(); err != nil {
					processMu.Unlock()
					return nil, fmt.Errorf("cudalock: %w", err)
				}
				hold := &dirHold{count: 1, holderPID: pid, inherited: true, owner: owner, namespace: &g.namespace, ownerData: data}
				processHolds[cleanDir] = hold
				processMu.Unlock()
				return &Handle{dir: cfg.dir, owner: owner, isNested: true, hold: hold}, nil
			}
		}
	}
	processMu.Unlock()
	myHost := shortHostname()
	myPID := os.Getpid()

	for {
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("cudalock: %w", err)
		}
		if err := g.check(); err != nil {
			return nil, err
		}
		if cfg.noReclaim {
			if err := ctx.Err(); err != nil {
				return nil, fmt.Errorf("cudalock: %w", err)
			}
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
				processMu.Lock()
				hold := &dirHold{count: 1, holderPID: myPID, owner: owner, namespace: &g.namespace, ownerData: owner.String()}
				processHolds[cleanDir] = hold
				processMu.Unlock()
				return &Handle{dir: cfg.dir, owner: owner, hold: hold}, nil
			} else if !errors.Is(mkdirErr, os.ErrExist) {
				return nil, fmt.Errorf("cudalock: create lock directory: %w", mkdirErr)
			}
		} else {
			if _, err := os.Lstat(cfg.dir); errors.Is(err, os.ErrNotExist) {
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
						if err := g.check(); err != nil {
							_ = os.RemoveAll(staging)
							return nil, err
						}
						if err := ctx.Err(); err != nil {
							_ = os.RemoveAll(staging)
							return nil, fmt.Errorf("cudalock: %w", err)
						}
						if renameErr := os.Rename(staging, cfg.dir); renameErr == nil {
							// Successfully acquired directory mutex
							processMu.Lock()
							hold := &dirHold{
								count:     1,
								holderPID: myPID,
								inherited: false,
								owner:     owner,
								namespace: &g.namespace,
								ownerData: owner.String(),
							}
							processHolds[cleanDir] = hold
							processMu.Unlock()

							return &Handle{
								dir:   cfg.dir,
								owner: owner,
								hold:  hold,
							}, nil
						}
					}
					_ = os.RemoveAll(staging)
				}
			} else if err != nil {
				return nil, fmt.Errorf("cudalock: inspect lock: %w", err)
			}

			// Mutex directory already exists: check owner record
			ownerFile := filepath.Join(cfg.dir, "owner")
			data, readErr := os.ReadFile(ownerFile)
			if readErr == nil {
				existingOwner, parseErr := ParseOwner(data)
				if parseErr == nil {
					// Safety check 1: Foreign host locks are NEVER reclaimed.
					if existingOwner.Host != "" && existingOwner.Host != myHost {
						return nil, nil
					}

					// Safety check 2: Process liveness check.
					// If process is alive, never reclaim regardless of age!
					if isAlive(existingOwner.PID) {
						return nil, nil
					}

					// Safety check 3: Process is confirmed dead on our host.
					// Reclaim the abandoned lock directory atomically.
					reclaimed, reclaimErr := reclaimChecked(cfg.dir, data, g.check)
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
							reclaimed, reclaimErr := reclaimChecked(cfg.dir, data, g.check)
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
						reclaimed, reclaimErr := reclaimChecked(cfg.dir, nil, g.check)
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

		return nil, nil
	}
}

// cancelAdmission runs with the namespace guard held, before h escapes Acquire.
// A root reference is cleanup custody until the final context check admits it.
func cancelAdmission(h *Handle, g *namespaceGuard, cause error, removeDir func(string) error) error {
	h.mu.Lock()
	defer h.mu.Unlock()
	processMu.Lock()
	defer processMu.Unlock()
	primary := fmt.Errorf("cudalock: %w", cause)
	hold := processHolds[h.dir]
	if hold == nil || hold != h.hold || hold.count <= 0 {
		return &acquireCleanupError{errors.Join(primary, errors.New("cudalock: canceled admission ownership changed")), h}
	}
	if h.isNested {
		hold.count--
		if hold.count == 0 {
			delete(processHolds, h.dir)
		}
		h.released = true
		return primary
	}
	hold.cleanupOnly = true
	if err := h.releaseRoot(g, hold, removeDir); err != nil {
		return &acquireCleanupError{errors.Join(primary, err), h}
	}
	return primary
}

type acquireCleanupError struct {
	err    error
	handle *Handle
}

func (e *acquireCleanupError) Error() string { return e.err.Error() }
func (e *acquireCleanupError) Unwrap() error { return e.err }
func (e *acquireCleanupError) Close() error  { return e.handle.Release() }

func forgetHandle(h *Handle) {
	if h == nil {
		return
	}
	processMu.Lock()
	defer processMu.Unlock()
	if hold := processHolds[h.dir]; hold != nil && hold == h.hold {
		hold.count--
		if hold.count == 0 {
			delete(processHolds, h.dir)
		}
	}
}

func isDefaultDir(dir string) bool {
	abs, err := filepath.Abs(DefaultDir())
	if err != nil {
		return false
	}
	parent, err := filepath.EvalSymlinks(filepath.Dir(abs))
	return err == nil && dir == filepath.Join(parent, filepath.Base(abs))
}

// reclaim atomically moves staleDir aside, verifies that its owner file still
// matches expectedData, and removes it. The caller must hold the permanent
// namespace guard from its original owner inspection through root publication.
// If an uncoordinated client changed the owner or another reclaimer
// intervened, the displaced directory is moved back intact.
func reclaim(dir string, expectedData []byte) (bool, error) {
	return reclaimChecked(dir, expectedData, nil)
}

func reclaimChecked(dir string, expectedData []byte, check func() error) (bool, error) {
	if reclaimHook != nil {
		reclaimHook()
	}
	parent := filepath.Dir(dir)
	aside, err := os.MkdirTemp(parent, ".gpu-lock-stale.*")
	if err != nil {
		return false, nil
	}
	_ = os.Remove(aside)

	if check != nil {
		if err := check(); err != nil {
			return false, err
		}
	}

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
func ownerHeldBy(dir string, pid int) (Owner, string, error) {
	data, err := os.ReadFile(filepath.Join(dir, "owner"))
	if err != nil {
		return Owner{}, "", fmt.Errorf("reading owner in %s: %w", dir, err)
	}
	owner, err := ParseOwner(data)
	if err != nil {
		return Owner{}, "", fmt.Errorf("parsing owner in %s: %w", dir, err)
	}
	if owner.PID != pid {
		return Owner{}, "", fmt.Errorf("owner record in %s names pid %d, want %d", dir, owner.PID, pid)
	}
	if owner.Host != "" && owner.Host != shortHostname() {
		return Owner{}, "", fmt.Errorf("owner host %q differs from current host %q", owner.Host, shortHostname())
	}
	return owner, string(data), nil
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
	if !isDefaultDir(handle.dir) {
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
