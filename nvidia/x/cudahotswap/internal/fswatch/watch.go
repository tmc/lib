// Package fswatch watches directory contents, including rename saves.
package fswatch

import (
	"context"
	"crypto/sha256"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"time"
)

// Config sets the debounce and polling intervals. Zero values select defaults.
type Config struct {
	Poll     bool
	Interval time.Duration
	Debounce time.Duration
	// Ready runs after the initial snapshot and notification owner are ready,
	// before waiting for changes. Its error stops Watch and closes that owner.
	Ready func() error
}

// Change reports the latest contents of a file after the debounce interval.
// Removed is true when the file no longer exists.
type Change struct {
	Path    string
	Digest  [32]byte
	Removed bool
}

type stamp struct {
	digest [32]byte
	exists bool
}

type pending struct {
	stamp stamp
	due   time.Time
}

type state struct {
	observed  map[string]stamp
	delivered map[string]stamp
	pending   map[string]pending
}

func newState(initial map[string]stamp) *state {
	delivered := make(map[string]stamp, len(initial))
	for path, s := range initial {
		delivered[path] = s
	}
	return &state{initial, delivered, make(map[string]pending)}
}

func (s *state) observe(next map[string]stamp, now time.Time, debounce time.Duration) {
	for path, value := range next {
		if value != s.observed[path] {
			s.pending[path] = pending{value, now.Add(debounce)}
		}
	}
	for path := range s.observed {
		if _, ok := next[path]; !ok {
			s.pending[path] = pending{stamp{}, now.Add(debounce)}
		}
	}
	s.observed = next
}

func (s *state) ready(now time.Time) []Change {
	var changes []Change
	for path, p := range s.pending {
		if now.Before(p.due) {
			continue
		}
		delete(s.pending, path)
		if s.delivered[path] == p.stamp {
			continue
		}
		if p.stamp.exists {
			s.delivered[path] = p.stamp
		} else {
			delete(s.delivered, path)
		}
		changes = append(changes, Change{path, p.stamp.digest, !p.stamp.exists})
	}
	sort.Slice(changes, func(i, j int) bool { return changes[i].Path < changes[j].Path })
	return changes
}

func scan(dirs []string) (map[string]stamp, error) {
	result := make(map[string]stamp)
	var entries int
	var bytesRead int64
	for _, dir := range dirs {
		err := filepath.WalkDir(dir, func(path string, entry fs.DirEntry, err error) error {
			if os.IsNotExist(err) {
				return nil // An editor can rename a file while a scan is in progress.
			}
			if err != nil {
				return err
			}
			entries++
			if entries > 4096 {
				return fmt.Errorf("scan: more than 4096 entries")
			}
			if entry.Type()&os.ModeSymlink != 0 {
				return nil
			}
			if entry.IsDir() {
				return nil
			}
			info, err := os.Stat(path)
			if os.IsNotExist(err) {
				return nil
			}
			if err != nil {
				return err
			}
			if !info.Mode().IsRegular() {
				return nil
			}
			if info.Size() > 8<<20 {
				return fmt.Errorf("scan: file exceeds 8 MiB limit")
			}
			limit := int64(8 << 20)
			if remaining := (64 << 20) - bytesRead; remaining < limit {
				limit = remaining
			}
			data, err := ReadFile(path, limit)
			if os.IsNotExist(err) {
				return nil
			}
			if err != nil {
				return err
			}
			bytesRead += int64(len(data))
			if bytesRead > 64<<20 {
				return fmt.Errorf("scan: aggregate exceeds 64 MiB limit")
			}
			digest := sha256.Sum256(data)
			result[path] = stamp{digest, true}
			return nil
		})
		if err != nil {
			return nil, fmt.Errorf("scan %q: %w", dir, err)
		}
	}
	return result, nil
}

func waitPoll(ctx context.Context, interval time.Duration) error {
	timer := time.NewTimer(interval)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

// Watch calls notify for content changes in dirs until ctx ends or an error
// occurs. Initial contents are not reported. notify runs serially and its
// error stops Watch. Symlinks and nonregular entries are skipped. Scans refuse
// more than 4096 entries, files over 8 MiB or aggregate content over 64 MiB.
func Watch(ctx context.Context, dirs []string, cfg Config, notify func(Change) error) error {
	if ctx == nil || notify == nil || len(dirs) == 0 {
		return fmt.Errorf("watch: directories and callback are required")
	}
	if cfg.Interval < 0 || cfg.Debounce < 0 {
		return fmt.Errorf("watch: negative interval")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if cfg.Interval == 0 {
		cfg.Interval = 50 * time.Millisecond
	}
	if cfg.Debounce == 0 {
		cfg.Debounce = 100 * time.Millisecond
	}
	roots := make([]string, 0, len(dirs))
	for _, dir := range dirs {
		path, err := filepath.Abs(dir)
		if err != nil {
			return err
		}
		info, err := os.Lstat(path)
		if err != nil {
			return err
		}
		if !info.IsDir() {
			return fmt.Errorf("watch: %q is not a directory", path)
		}
		roots = append(roots, path)
	}
	initial, err := scan(roots)
	if err != nil {
		return err
	}
	watch, err := openNotify(roots, cfg.Poll)
	if err != nil {
		return fmt.Errorf("watch directories: %w", err)
	}
	defer watch.close()
	s := newState(initial)
	if cfg.Ready != nil {
		if err := cfg.Ready(); err != nil {
			return err
		}
	}
	for {
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := watch.wait(ctx, cfg.Interval); err != nil {
			return err
		}
		// A complete scan also catches new subdirectories and network mounts
		// that do not deliver notifications. Hashes, not mtimes, decide changes.
		next, err := scan(roots)
		if err != nil {
			return err
		}
		now := time.Now()
		s.observe(next, now, cfg.Debounce)
		for _, change := range s.ready(now) {
			if err := ctx.Err(); err != nil {
				return err
			}
			if err := notify(change); err != nil {
				return err
			}
		}
	}
}
