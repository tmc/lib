package cudalock

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

// A namespace is the physical parent and permanent guard inode for one lock.
// Neither may be replaced while any protocol participant is running.
type namespace struct {
	dir, parent, path string
	parentInfo        os.FileInfo
	guardInfo         os.FileInfo
}

type namespaceGuard struct {
	namespace
	file *os.File
}

func openNamespace(dir string) (*namespaceGuard, error) {
	if dir == "" || filepath.Clean(dir) == "." {
		return nil, errors.New("cudalock: lock path must name a directory leaf")
	}
	abs, err := filepath.Abs(dir)
	if err != nil {
		return nil, fmt.Errorf("cudalock: absolute lock path: %w", err)
	}
	if abs == filepath.Dir(abs) {
		return nil, errors.New("cudalock: lock cannot be a filesystem root")
	}
	parent, err := filepath.EvalSymlinks(filepath.Dir(abs))
	if err != nil {
		return nil, fmt.Errorf("cudalock: resolve lock parent: %w", err)
	}
	parentInfo, err := os.Lstat(parent)
	if err != nil {
		return nil, fmt.Errorf("cudalock: inspect lock parent: %w", err)
	}
	if !parentInfo.IsDir() {
		return nil, errors.New("cudalock: lock parent is not a directory")
	}
	dir = filepath.Join(parent, filepath.Base(abs))
	path := dir + ".guard"
	if info, err := os.Lstat(path); err == nil && (!info.Mode().IsRegular() || !guardSingleLink(info)) {
		return nil, errors.New("cudalock: guard is not a single-link regular file")
	} else if err != nil && !errors.Is(err, os.ErrNotExist) {
		return nil, fmt.Errorf("cudalock: inspect guard path: %w", err)
	}
	f, err := openGuardFile(path)
	if err != nil {
		return nil, fmt.Errorf("cudalock: open permanent guard: %w", err)
	}
	info, err := f.Stat()
	if err != nil {
		return nil, errors.Join(fmt.Errorf("cudalock: inspect guard: %w", err), f.Close())
	}
	g := &namespaceGuard{namespace{dir, parent, path, parentInfo, info}, f}
	if err := g.check(); err != nil {
		return nil, errors.Join(err, f.Close())
	}
	return g, nil
}

// check refuses an old descriptor or replaced parent rather than reopening a
// new inode. Reopening would split participants into independent lock domains.
func (g *namespaceGuard) check() error {
	parent, err := os.Lstat(g.parent)
	if err != nil || !os.SameFile(g.parentInfo, parent) || !parent.IsDir() {
		return errors.New("cudalock: lock parent identity changed")
	}
	info, err := os.Lstat(g.path)
	if err != nil || !info.Mode().IsRegular() || !guardSingleLink(info) || !os.SameFile(g.guardInfo, info) {
		return errors.New("cudalock: permanent guard identity changed or is not a single-link regular file")
	}
	if info, err := os.Lstat(g.dir); err == nil && !info.IsDir() {
		return errors.New("cudalock: lock path is not a physical directory")
	} else if err != nil && !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("cudalock: inspect lock directory: %w", err)
	}
	return nil
}

func (g *namespaceGuard) lock(ctx context.Context, poll time.Duration) error {
	for {
		if err := ctx.Err(); err != nil {
			return fmt.Errorf("cudalock: wait for guard: %w", err)
		}
		if err := guardLock(g.file); err == nil {
			if err := errors.Join(ctx.Err(), g.check()); err != nil {
				return errors.Join(err, guardUnlock(g.file))
			}
			return nil
		} else if !guardBusy(err) {
			return fmt.Errorf("cudalock: lock permanent guard: %w", err)
		}
		timer := time.NewTimer(poll)
		select {
		case <-ctx.Done():
			timer.Stop()
			return fmt.Errorf("cudalock: wait for guard: %w", ctx.Err())
		case <-timer.C:
		}
	}
}

// tryLock supports admission checks that have no context argument. Contention
// means unverified ownership, not evidence that an owner is absent.
func (g *namespaceGuard) tryLock() error {
	if err := guardLock(g.file); err != nil {
		return fmt.Errorf("cudalock: cannot verify busy or unavailable guard: %w", err)
	}
	if err := g.check(); err != nil {
		return errors.Join(err, guardUnlock(g.file))
	}
	return nil
}

func (g *namespaceGuard) matches(n *namespace) bool {
	return n != nil && g.dir == n.dir && os.SameFile(g.guardInfo, n.guardInfo) && os.SameFile(g.parentInfo, n.parentInfo)
}
