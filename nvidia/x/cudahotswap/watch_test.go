package cudahotswap

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudakernel"
)

type watchTestLoader struct {
	compiles  atomic.Int64
	loads     atomic.Int64
	headers   func(Source) []string
	signature func(Source) string
}

func (l *watchTestLoader) Compile(_ context.Context, source Source) (*Compilation, error) {
	l.compiles.Add(1)
	c := &Compilation{Entry: "entry", PTX: ".version 7.0\n.target sm_70\n.address_size 64\n.visible .entry entry() { ret; }"}
	if l.signature != nil {
		c.PTX = l.signature(source)
	}
	identity := source.Code
	if l.headers != nil {
		for _, path := range l.headers(source) {
			data, err := os.ReadFile(path)
			if err != nil {
				return nil, err
			}
			digest := fmt.Sprintf("%x", sha256.Sum256(data))
			c.Dependencies = append(c.Dependencies, cudakernel.Dependency{Path: path, Digest: digest, Tracked: true})
			identity += path + digest
		}
	}
	c.Key = fmt.Sprintf("%x", sha256.Sum256([]byte(identity)))
	return c, nil
}

func (l *watchTestLoader) Load(context.Context, *Compilation) (Kernel, error) {
	return &serviceKernel{raw: driver.Function(l.loads.Add(1))}, nil
}

type watchTestLease struct {
	active *atomic.Int64
	closed atomic.Bool
}

func (l *watchTestLease) Close() error {
	if !l.closed.Swap(true) {
		l.active.Add(-1)
	}
	return nil
}

func watchTestService(t *testing.T, l *watchTestLoader) (*Service, *atomic.Int64) {
	t.Helper()
	s, err := NewService(l)
	if err != nil {
		t.Fatal(err)
	}
	active := new(atomic.Int64)
	s.acquire = func(context.Context, string) (io.Closer, error) {
		if active.Add(1) != 1 {
			active.Add(-1)
			return nil, errors.New("another exclusive lease is active")
		}
		return &watchTestLease{active: active}, nil
	}
	t.Cleanup(func() {
		if err := s.Close(context.Background()); err != nil {
			t.Error(err)
		}
	})
	return s, active
}

func watchWrite(t *testing.T, path, code string) {
	t.Helper()
	// A rename-save presents complete candidate bytes. In-place truncation can
	// legitimately be observed by the startup probe before its write finishes.
	name := path + ".save"
	if err := os.WriteFile(name, []byte(code), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(name, path); err != nil {
		t.Fatal(err)
	}
}

func watchRegister(t *testing.T, s *Service, name, path, code string, checker Checker) {
	t.Helper()
	watchWrite(t, path, code)
	if _, err := s.Register(context.Background(), name, Source{Code: code, Filename: path, Entry: "entry"}, "baseline"); err != nil {
		t.Fatal(err)
	}
	if checker == nil {
		checker = func(context.Context, State) error { return nil }
	}
	if _, err := s.SetCheck(name, CheckConfig{Checker: checker}, "check"); err != nil {
		t.Fatal(err)
	}
	if _, err := s.SetWatchPaths(name, &WatchPaths{SourceFile: path}, "watch"); err != nil {
		t.Fatal(err)
	}
}

func watchWait[T any](t *testing.T, ch <-chan T) T {
	t.Helper()
	select {
	case value := <-ch:
		return value
	case <-time.After(3 * time.Second):
		t.Fatal("watch did not deliver the expected boundary")
		var zero T
		return zero
	}
}

func TestWatchPublicPushRenameDigestAndIdleLease(t *testing.T) {
	for _, poll := range []bool{false, true} {
		t.Run(fmt.Sprintf("poll_%t", poll), func(t *testing.T) {
			l := new(watchTestLoader)
			s, active := watchTestService(t, l)
			dir := t.TempDir()
			path := filepath.Join(dir, "kernel.cu")
			watchRegister(t, s, "kernel", path, "old", nil)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			events := make(chan *PushResult, 8)
			done := make(chan error, 1)
			go func() {
				done <- s.Watch(ctx, WatchConfig{Poll: poll, Interval: 2 * time.Millisecond, Debounce: 100 * time.Millisecond}, func(_ string, result *PushResult, err error) error {
					if err != nil {
						return err
					}
					events <- result
					return nil
				})
			}()
			// This edit can precede or follow the initial directory snapshot.
			watchWrite(t, path, "startup")
			first := watchWait(t, events)
			if !first.Published || first.Report.Outcome != OutcomeOK || active.Load() != 0 {
				t.Fatal("watch bypassed checked Push or retained an idle GPU lease")
			}
			before, err := os.Stat(path)
			if err != nil {
				t.Fatal(err)
			}
			if err := os.Chtimes(path, before.ModTime().Add(time.Hour), before.ModTime().Add(time.Hour)); err != nil {
				t.Fatal(err)
			}
			time.Sleep(150 * time.Millisecond)
			if l.compiles.Load() != 2 || active.Load() != 0 {
				t.Fatal("mtime-only edit triggered a compile or idle lease")
			}
			replacement := filepath.Join(dir, "replacement")
			watchWrite(t, replacement, "latest")
			if err := os.Chtimes(replacement, before.ModTime(), before.ModTime()); err != nil {
				t.Fatal(err)
			}
			// Burst coalescing uses explicit timestamps in fswatch.TestLatestContent.
			// A scheduler pause between two writes can exceed the debounce window.
			if err := os.Rename(replacement, path); err != nil {
				t.Fatal(err)
			}
			last := watchWait(t, events)
			if last.State.SourceDigest != fmt.Sprintf("%x", sha256.Sum256([]byte("latest"))) || l.compiles.Load() != 3 {
				t.Fatalf("rename: digest=%s compiles=%d, want latest once", last.State.SourceDigest, l.compiles.Load())
			}
			cancel()
			if err := watchWait(t, done); !errors.Is(err, context.Canceled) || active.Load() != 0 {
				t.Fatalf("watch cancellation or request lease: %v", err)
			}
		})
	}
}

func TestWatchCompilerHeaderFanout(t *testing.T) {
	header := filepath.Join(t.TempDir(), "shared.h")
	watchWrite(t, header, "header A")
	l := &watchTestLoader{headers: func(Source) []string { return []string{header} }}
	s, active := watchTestService(t, l)
	files := make(map[string]string)
	for _, name := range []string{"first", "second"} {
		files[name] = filepath.Join(t.TempDir(), name+".cu")
		watchRegister(t, s, name, files[name], "same source", nil)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	done := make(chan error, 1)
	type observed struct {
		name   string
		result *PushResult
	}
	events := make(chan observed, 8)
	go func() {
		done <- s.Watch(ctx, WatchConfig{Poll: true, Interval: time.Millisecond, Debounce: time.Millisecond}, func(name string, result *PushResult, err error) error {
			if err != nil || !result.Published {
				return fmt.Errorf("header Push failed: %v", err)
			}
			events <- observed{name, result}
			return nil
		})
	}()
	// First establish the actual Ready/callback boundary. The header edit must
	// be delivered by the running directory watcher, not its initial probe.
	watchWrite(t, files["first"], "same source bootstrap")
	if first := watchWait(t, events); first.name != "first" {
		t.Fatal("wrong startup source consumer")
	}
	watchWrite(t, header, "header B")
	first, second := watchWait(t, events), watchWait(t, events)
	if first.name == second.name || l.compiles.Load() != 5 || active.Load() != 0 {
		t.Fatalf("actual compiler Dependencies fanout: names=%v/%v compiles=%d active=%d", first.name, second.name, l.compiles.Load(), active.Load())
	}
	for _, event := range []observed{first, second} {
		code := "same source"
		if event.name == "first" {
			code += " bootstrap"
		}
		if event.result.Report.Compile.SourceSHA256 != fmt.Sprintf("%x", sha256.Sum256([]byte(code))) {
			t.Fatal("header rebuild changed source bytes")
		}
	}
	cancel()
	if err := watchWait(t, done); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
}

func TestWatchReportsWrongCandidateAndContinues(t *testing.T) {
	l := new(watchTestLoader)
	s, _ := watchTestService(t, l)
	path := filepath.Join(t.TempDir(), "kernel.cu")
	wrong := fmt.Sprintf("%x", sha256.Sum256([]byte("wrong")))
	watchRegister(t, s, "kernel", path, "baseline", func(_ context.Context, state State) error {
		if state.SourceDigest == wrong {
			return errors.New("wrong application output")
		}
		return nil
	})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	events := make(chan *PushResult, 4)
	done := make(chan error, 1)
	go func() {
		done <- s.Watch(ctx, WatchConfig{Poll: true, Interval: time.Millisecond, Debounce: time.Millisecond}, func(_ string, result *PushResult, _ error) error { events <- result; return nil })
	}()
	watchWrite(t, path, "wrong")
	failed := watchWait(t, events)
	if !failed.RolledBack || failed.Report.Outcome != OutcomeWrongAnswer || failed.Selected.Version != 1 {
		t.Fatalf("watch admitted wrong application candidate: %+v", failed)
	}
	watchWrite(t, path, "correct")
	passed := watchWait(t, events)
	if !passed.Published || passed.RolledBack || passed.Report.Outcome != OutcomeOK {
		t.Fatal("watch did not continue through a later passing Push")
	}
	cancel()
	if err := watchWait(t, done); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
}

func TestWatchDiscoversNewHeaderDirectoryWithoutDuplicatePush(t *testing.T) {
	header := filepath.Join(t.TempDir(), "extra.h")
	watchWrite(t, header, "one")
	l := &watchTestLoader{headers: func(source Source) []string {
		if strings.Contains(source.Code, "extra") {
			return []string{header}
		}
		return nil
	}}
	s, _ := watchTestService(t, l)
	path := filepath.Join(t.TempDir(), "kernel.cu")
	watchRegister(t, s, "kernel", path, "baseline", nil)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	events := make(chan *PushResult, 8)
	done := make(chan error, 1)
	go func() {
		done <- s.Watch(ctx, WatchConfig{Poll: true, Interval: time.Millisecond, Debounce: time.Millisecond}, func(_ string, result *PushResult, err error) error {
			if err != nil {
				return err
			}
			events <- result
			return nil
		})
	}()
	watchWrite(t, path, "extra")
	watchWait(t, events)
	time.Sleep(20 * time.Millisecond)
	if l.compiles.Load() != 2 {
		t.Fatal("new compiler-header topology caused a duplicate Push")
	}
	watchWrite(t, header, "two")
	watchWait(t, events)
	if l.compiles.Load() != 3 {
		t.Fatal("new header directory was not watched through actual Push")
	}
	cancel()
	if err := watchWait(t, done); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
}

func TestWatchPathsSettingsAndShutdown(t *testing.T) {
	l := new(watchTestLoader)
	s, active := watchTestService(t, l)
	path := filepath.Join(t.TempDir(), "kernel.cu")
	watchRegister(t, s, "kernel", path, "baseline", nil)
	before, _ := s.Status("kernel")
	paths, err := s.WatchPaths("kernel")
	if err != nil || paths.SourceFile != path {
		t.Fatal(err)
	}
	paths.SourceFile = "changed detached result"
	again, _ := s.WatchPaths("kernel")
	if again.SourceFile != path {
		t.Fatal("watch paths alias service settings")
	}
	state, err := s.SetWatchPaths("kernel", nil, "disable")
	if err != nil || state.Generation != before.Generation || state.ConfigRevision != before.ConfigRevision+1 || active.Load() != 0 {
		t.Fatal("watch settings changed code or acquired GPU lease")
	}
	done := make(chan error, 1)
	go func() { done <- s.Watch(context.Background(), WatchConfig{}, nil) }()
	deadline := time.Now().Add(time.Second)
	for {
		s.mu.Lock()
		running := s.watch.running
		s.mu.Unlock()
		if running {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("watch did not start")
		}
		time.Sleep(time.Millisecond)
	}
	if err := s.Watch(context.Background(), WatchConfig{}, nil); err == nil {
		t.Fatal("duplicate concurrent watcher admitted")
	}
	if err := s.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := watchWait(t, done); !errors.Is(err, io.ErrClosedPipe) || active.Load() != 0 {
		t.Fatalf("Close did not wake idle watcher: %v", err)
	}
}

func TestWatchHeaderRepairsABIRefusedCompilation(t *testing.T) {
	header := filepath.Join(t.TempDir(), "signature.h")
	watchWrite(t, header, "bad")
	l := &watchTestLoader{headers: func(source Source) []string {
		if strings.Contains(source.Code, "needs-header") {
			return []string{header}
		}
		return nil
	}}
	l.signature = func(source Source) string {
		params := ""
		if strings.Contains(source.Code, "needs-header") {
			data, _ := os.ReadFile(header)
			if string(data) == "bad" {
				params = ".param .u32 incompatible"
			}
		}
		return ".version 7.0\n.target sm_70\n.address_size 64\n.visible .entry entry(" + params + ") { ret; }"
	}
	s, _ := watchTestService(t, l)
	path := filepath.Join(t.TempDir(), "kernel.cu")
	watchRegister(t, s, "kernel", path, "baseline", nil)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	events := make(chan *PushResult, 8)
	done := make(chan error, 1)
	go func() {
		done <- s.Watch(ctx, WatchConfig{Poll: true, Interval: time.Millisecond, Debounce: time.Millisecond}, func(_ string, result *PushResult, _ error) error { events <- result; return nil })
	}()
	watchWrite(t, path, "needs-header")
	failed := watchWait(t, events)
	if failed.Published || failed.Report.Outcome != OutcomeABIMismatch || l.loads.Load() != 1 {
		t.Fatal("ABI mismatch was loaded or misreported")
	}
	watchWrite(t, header, "good")
	passed := watchWait(t, events)
	if !passed.Published || passed.Report.Outcome != OutcomeOK || l.compiles.Load() != 3 {
		t.Fatal("header repair of refused candidate did not reach actual Push")
	}
	cancel()
	if err := watchWait(t, done); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
}

func TestWatchCompilerMetadataBoundedBeforeRetention(t *testing.T) {
	for _, input := range []watchInput{
		boundedWatchInput(Source{Code: strings.Repeat("x", (8<<20)+1)}, nil),
		boundedWatchInput(Source{}, make([]cudakernel.Dependency, 4097)),
		boundedWatchInput(Source{}, []cudakernel.Dependency{{Path: strings.Repeat("x", (16<<10)+1)}}),
	} {
		if input.err == nil || input.source.Code != "" || len(input.source.Flags) != 0 || len(input.deps) != 0 {
			t.Fatal("oversized compiler input retained before refusal")
		}
		if _, err := watchDependencies(input); err == nil {
			t.Fatal("invalid compiler metadata admitted to watching")
		}
	}
}
