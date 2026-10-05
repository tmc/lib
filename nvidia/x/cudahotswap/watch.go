package cudahotswap

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"time"

	"github.com/tmc/lib/nvidia/x/cudahotswap/internal/fswatch"
	"github.com/tmc/lib/nvidia/x/cudakernel"
)

// WatchPaths identifies a kernel source file and additional directories to
// watch. The source's directory and tracked compiler-header directories are
// included automatically. Paths are copied and made absolute by SetWatchPaths.
type WatchPaths struct {
	SourceFile  string
	Directories []string
}

// WatchConfig selects polling for network mounts and scan/debounce intervals.
// Zero intervals use 50ms scans and 100ms debounce. Linux uses inotify unless
// Poll is true; other platforms poll. Changes are decided by content hashes.
type WatchConfig struct {
	Poll     bool
	Interval time.Duration
	Debounce time.Duration
}

type watchInput struct {
	source Source
	deps   []cudakernel.Dependency
	err    error
}

type watchState struct {
	inputs  map[string]watchInput
	changed chan struct{}
	running bool
}

// These helpers require s.mu. Neither takes a GPU lease.
func (s *Service) watchChangedLocked() {
	if s.watch.changed != nil {
		close(s.watch.changed)
	}
	s.watch.changed = make(chan struct{})
}

func (s *Service) rememberWatchInputLocked(name string, source Source, c *Compilation) {
	if s.watch.inputs == nil {
		s.watch.inputs = make(map[string]watchInput)
	}
	input := boundedWatchInput(source, c.Dependencies)
	prior, ok := s.watch.inputs[name]
	s.watch.inputs[name] = input
	if !ok || !reflect.DeepEqual(prior.deps, input.deps) || (prior.err == nil) != (input.err == nil) {
		s.watchChangedLocked()
	}
}

func boundedWatchInput(source Source, deps []cudakernel.Dependency) watchInput {
	invalid := watchInput{err: errors.New("watch compiler inputs exceed source, flags or dependency limits")}
	if len(source.Code) > 8<<20 || len(source.Filename) > 16<<10 || len(source.Entry) > 256 || len(source.Flags) > 64 || len(deps) > 4096 {
		return invalid
	}
	bytes := 0
	for _, flag := range source.Flags {
		if len(flag) > (16<<10)-bytes {
			return invalid
		}
		bytes += len(flag)
	}
	for _, dep := range deps {
		if len(dep.Path) > 16<<10 || len(dep.Digest) > 64 {
			return invalid
		}
	}
	source.Flags = append([]string(nil), source.Flags...)
	input := watchInput{source: source, deps: make([]cudakernel.Dependency, len(deps))}
	for i, dep := range deps {
		input.deps[i] = cudakernel.Dependency{Path: dep.Path, Digest: dep.Digest, Tracked: dep.Tracked}
	}
	return input
}

// SetWatchPaths sets or disables watching for name. A nil paths disables it.
// This advances ConfigRevision without compiling, loading or selecting code.
// Tracked Dependencies from the actual compiler define header-to-kernel fanout;
// watching refuses unresolved dependencies rather than ignoring them.
func (s *Service) SetWatchPaths(name string, paths *WatchPaths, label string) (State, error) {
	if s == nil || s.loader == nil {
		return State{}, errors.New("set watch paths: nil service or loader")
	}
	var copy *WatchPaths
	if paths != nil {
		if paths.SourceFile == "" || len(paths.Directories) > 64 {
			return State{}, errors.New("set watch paths: source file and at most 64 directories are required")
		}
		file, err := filepath.Abs(paths.SourceFile)
		if err != nil {
			return State{}, err
		}
		if _, err := readWatchFile(file); err != nil {
			return State{}, fmt.Errorf("set watch source: %w", err)
		}
		copy = &WatchPaths{SourceFile: file}
		for _, dir := range paths.Directories {
			path, err := watchDirectory(dir)
			if err != nil {
				return State{}, err
			}
			copy.Directories = append(copy.Directories, path)
		}
	}
	s.mutations.Lock()
	defer s.mutations.Unlock()
	if copy != nil {
		s.mu.Lock()
		input, ok := s.watch.inputs[name]
		s.mu.Unlock()
		if !ok {
			return State{}, errors.New("set watch paths: compiler inputs are unavailable")
		}
		if _, err := watchDependencies(input); err != nil {
			return State{}, err
		}
	}
	state, err := s.setSettingsLocked(name, label, func(c *kernelSettings) { c.watch = copy })
	if err == nil {
		s.mu.Lock()
		s.watchChangedLocked()
		s.mu.Unlock()
	}
	return state, err
}

// WatchPaths returns detached absolute paths. A nil result means watching is
// disabled. It takes no GPU lease and does not change settings.
func (s *Service) WatchPaths(name string) (*WatchPaths, error) {
	if s == nil {
		return nil, errors.New("watch paths: nil service")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.states[name]; !ok {
		return nil, errors.New("watch paths: kernel is not registered")
	}
	if paths := s.settings[name].watch; paths != nil {
		copy := *paths
		copy.Directories = append([]string(nil), paths.Directories...)
		return &copy, nil
	}
	return nil, nil
}

func readWatchFile(path string) ([]byte, error) {
	return fswatch.ReadFile(path, 8<<20)
}

func watchDirectory(path string) (string, error) {
	path, err := filepath.Abs(path)
	if err != nil {
		return "", err
	}
	info, err := os.Lstat(path)
	if err != nil {
		return "", err
	}
	if !info.IsDir() {
		return "", fmt.Errorf("watch directory %q is not a directory", path)
	}
	return path, nil
}

func watchDependencies(input watchInput) ([]cudakernel.Dependency, error) {
	if input.err != nil {
		return nil, input.err
	}
	if len(input.deps) > 4096 {
		return nil, errors.New("watch dependencies: more than 4096 headers")
	}
	deps := append([]cudakernel.Dependency(nil), input.deps...)
	for i := range deps {
		if !deps[i].Tracked || deps[i].Path == "" {
			return nil, errors.New("watch dependencies: unresolved compiler header")
		}
		if _, err := hex.DecodeString(deps[i].Digest); err != nil || len(deps[i].Digest) != 64 {
			return nil, errors.New("watch dependencies: invalid compiler header digest")
		}
		path, err := filepath.Abs(deps[i].Path)
		if err != nil {
			return nil, err
		}
		deps[i].Path = path
	}
	sort.Slice(deps, func(i, j int) bool { return deps[i].Path < deps[j].Path })
	return deps, nil
}

type watchedKernel struct {
	name  string
	paths WatchPaths
	input watchInput
	deps  []cudakernel.Dependency
}

func (s *Service) watchSnapshot() ([]watchedKernel, []string, <-chan struct{}, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil, nil, nil, io.ErrClosedPipe
	}
	if s.poison != nil {
		return nil, nil, nil, fmt.Errorf("watch: %w: %w", ErrPoisoned, s.poison)
	}
	if s.watch.changed == nil {
		s.watch.changed = make(chan struct{})
	}
	var kernels []watchedKernel
	dirs := make(map[string]bool)
	for name, settings := range s.settings {
		if settings.watch == nil {
			continue
		}
		if len(kernels) >= 128 {
			return nil, nil, nil, errors.New("watch: more than 128 configured kernels")
		}
		input := s.watch.inputs[name]
		deps, err := watchDependencies(input)
		if err != nil {
			return nil, nil, nil, err
		}
		paths := *settings.watch
		paths.Directories = append([]string(nil), paths.Directories...)
		kernels = append(kernels, watchedKernel{name, paths, input, deps})
		dirs[filepath.Dir(paths.SourceFile)] = true
		for _, dir := range paths.Directories {
			dirs[dir] = true
		}
		for _, dep := range deps {
			dirs[filepath.Dir(dep.Path)] = true
		}
		if len(dirs) > 1024 {
			return nil, nil, nil, errors.New("watch: more than 1024 distinct directories")
		}
	}
	sort.Slice(kernels, func(i, j int) bool { return kernels[i].name < kernels[j].name })
	var directories []string
	for dir := range dirs {
		directories = append(directories, dir)
	}
	sort.Strings(directories)
	return kernels, directories, s.watch.changed, nil
}

func watchFingerprint(code []byte, deps []cudakernel.Dependency) [32]byte {
	h := sha256.New()
	fmt.Fprintf(h, "%x\n", sha256.Sum256(code))
	for _, dep := range deps {
		fmt.Fprintf(h, "%q %s\n", dep.Path, dep.Digest)
	}
	var digest [32]byte
	copy(digest[:], h.Sum(nil))
	return digest
}

// Watch pushes changed source/header contents through the actual checked Push
// path. Only one Watch may run per service. SetWatchPaths changes take effect
// while it runs; Close stops it. Watching itself holds no GPU lease.
//
// notify, when nonnil, runs serially after each attempted Push. Ordinary compile
// or check failures are reported and watching continues; a notify error stops
// Watch. Poisoning stops Watch after publishing the terminal result. A removed
// source reports an error and can be restored by a later rename save. Unchanged
// bytes, including mtime-only edits, never trigger another Push.
func (s *Service) Watch(ctx context.Context, config WatchConfig, notify func(string, *PushResult, error) error) error {
	if ctx == nil || s == nil || s.loader == nil {
		return errors.New("watch: service and context are required")
	}
	if config.Interval < 0 || config.Debounce < 0 {
		return errors.New("watch: negative interval")
	}
	s.mu.Lock()
	if s.watch.running {
		s.mu.Unlock()
		return errors.New("watch: another watcher is active")
	}
	s.watch.running = true
	s.mu.Unlock()
	defer func() { s.mu.Lock(); s.watch.running = false; s.mu.Unlock() }()
	last := make(map[string][32]byte)
	for {
		if err := ctx.Err(); err != nil {
			return err
		}
		kernels, dirs, changed, err := s.watchSnapshot()
		if err != nil {
			return err
		}
		active := make(map[string]bool)
		for _, kernel := range kernels {
			active[kernel.name] = true
			if _, ok := last[kernel.name]; !ok {
				last[kernel.name] = watchFingerprint([]byte(kernel.input.source.Code), kernel.deps)
			}
		}
		for name := range last {
			if !active[name] {
				delete(last, name)
			}
		}
		if len(dirs) == 0 {
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-changed:
				continue
			}
		}
		workCtx, cancel := context.WithCancel(ctx)
		done := make(chan error, 1)
		go func() {
			push := func(path string) error {
				for _, kernel := range kernels {
					affected := path == "" || path == kernel.paths.SourceFile
					for _, dep := range kernel.deps {
						affected = affected || path == dep.Path
					}
					if affected {
						// Reconfiguration stops scans, but never cancels an in-flight
						// request: only the caller's context does that.
						if err := s.pushWatched(ctx, kernel, last, notify); err != nil {
							return err
						}
					}
				}
				return nil
			}
			done <- fswatch.Watch(workCtx, dirs, fswatch.Config{Poll: config.Poll, Interval: config.Interval, Debounce: config.Debounce, Ready: func() error { return push("") }}, func(change fswatch.Change) error { return push(change.Path) })
		}()
		select {
		case err := <-done:
			cancel()
			return err
		case <-changed:
			cancel()
			err := <-done
			if err != nil && !errors.Is(err, context.Canceled) {
				return err
			}
		case <-ctx.Done():
			cancel()
			<-done
			return ctx.Err()
		}
	}
}

func (s *Service) pushWatched(ctx context.Context, kernel watchedKernel, last map[string][32]byte, notify func(string, *PushResult, error) error) error {
	s.mu.Lock()
	paths := s.settings[kernel.name].watch
	current := paths != nil && reflect.DeepEqual(*paths, kernel.paths) && !s.closed && s.poison == nil
	s.mu.Unlock()
	if !current {
		return nil
	}
	code, readErr := readWatchFile(kernel.paths.SourceFile)
	deps := append([]cudakernel.Dependency(nil), kernel.deps...)
	for i := range deps {
		data, err := readWatchFile(deps[i].Path)
		if err != nil {
			deps[i].Digest = "missing: " + err.Error()
		} else {
			deps[i].Digest = fmt.Sprintf("%x", sha256.Sum256(data))
		}
	}
	if readErr != nil {
		code = []byte("missing: " + readErr.Error())
	}
	digest := watchFingerprint(code, deps)
	if digest == last[kernel.name] {
		return nil
	}
	last[kernel.name] = digest
	var result *PushResult
	var err error
	if readErr != nil {
		state, _ := s.Status(kernel.name)
		result = &PushResult{State: state, Selected: state, Report: Report{Task: kernel.name, Candidate: "watch", Outcome: OutcomeHarnessError}}
		err = s.finishPush(result, fmt.Errorf("watch source: %w", readErr), false)
	} else {
		source := kernel.input.source
		source.Code, source.Filename, source.Flags = string(code), kernel.paths.SourceFile, nil
		result, err = s.Push(ctx, kernel.name, source, "watch: "+kernel.paths.SourceFile)
		// A successful compilation may discover new headers. Use its complete
		// input identity before restarting scans, avoiding a duplicate Push.
		s.mu.Lock()
		input := s.watch.inputs[kernel.name]
		s.mu.Unlock()
		if input.source.Code == source.Code {
			if deps, depErr := watchDependencies(input); depErr == nil {
				last[kernel.name] = watchFingerprint(code, deps)
			}
		}
	}
	if notify != nil {
		if notifyErr := notify(kernel.name, result, err); notifyErr != nil {
			return notifyErr
		}
	}
	return s.available()
}
