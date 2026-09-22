package cudaplan

import (
	"errors"
	"sync"
	"testing"

	"github.com/tmc/lib/nvidia/x/cublasltx"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudagraph"
	"github.com/tmc/lib/nvidia/x/cudakernel"
)

func cleanupFixture(close func(string) error) (*ExecutablePlan, planCloseOps) {
	p := &ExecutablePlan{
		graphExec: &cudagraph.Exec{}, graphObj: &cudagraph.Graph{},
		cublasEngine: &cublasltx.Engine{}, workspace: &WorkspaceManager{},
		modules: map[*cudakernel.Artifact]*cuda.Module{new(cudakernel.Artifact): {}},
	}
	return p, planCloseOps{
		graphExec: func(*cudagraph.Exec) error { return close("exec") },
		graph:     func(*cudagraph.Graph) error { return close("graph") },
		engine:    func(*cublasltx.Engine) error { return close("engine") },
		workspace: func(*WorkspaceManager) error { return close("workspace") },
		module:    func(*cuda.Module) error { return close("module") },
	}
}

func TestResourceCloseRetainsDependencies(t *testing.T) {
	order := []string{"exec", "graph", "engine", "workspace", "module"}
	failure := errors.New("destroy failed")
	for index, stage := range order {
		t.Run(stage, func(t *testing.T) {
			calls := map[string]int{}
			fail := true
			p, ops := cleanupFixture(func(name string) error {
				calls[name]++
				if fail && name == stage {
					return failure
				}
				return nil
			})
			if err := p.closeWith(ops); !errors.Is(err, failure) {
				t.Fatal(err)
			}
			if !p.closed {
				t.Fatal("failed teardown admits execution")
			}
			if _, err := p.Execute(nil, nil); !errors.Is(err, ErrClosed) {
				t.Fatalf("execute after failed teardown: %v", err)
			}
			owned := []bool{p.graphExec != nil, p.graphObj != nil, p.cublasEngine != nil, p.workspace != nil, len(p.modules) != 0}
			for j, name := range order {
				if owned[j] != (j >= index) {
					t.Fatalf("%s ownership=%v after %s failure", name, owned[j], stage)
				}
				want := 0
				if j <= index {
					want = 1
				}
				if calls[name] != want {
					t.Fatalf("%s close calls=%d want %d", name, calls[name], want)
				}
			}
			fail = false
			if err := p.closeWith(ops); err != nil {
				t.Fatal(err)
			}
			if err := p.closeWith(ops); err != nil {
				t.Fatal(err)
			}
			for _, name := range order {
				want := 1
				if name == stage {
					want = 2
				}
				if calls[name] != want {
					t.Fatalf("%s close calls=%d want %d", name, calls[name], want)
				}
			}
		})
	}
}

func TestCompileFailureRetainsCleanupOwner(t *testing.T) {
	cause, cleanup := errors.New("compile failed"), errors.New("cleanup failed")
	fail := true
	p, ops := cleanupFixture(func(string) error {
		if fail {
			return cleanup
		}
		return nil
	})
	owner, err := failedCompilation(p, cause, func() error { return p.closeWith(ops) })
	if owner != p || !errors.Is(err, cause) || !errors.Is(err, cleanup) || !p.closed {
		t.Fatalf("owner=%p plan=%p error=%v", owner, p, err)
	}
	fail = false
	if err := owner.closeWith(ops); err != nil {
		t.Fatal(err)
	}
	if err := owner.Close(); err != nil {
		t.Fatal(err)
	}
	p, ops = cleanupFixture(func(string) error { return nil })
	owner, err = failedCompilation(p, cause, func() error { return p.closeWith(ops) })
	if owner != nil || !errors.Is(err, cause) {
		t.Fatalf("successful cleanup owner=%p error=%v", owner, err)
	}
}

func TestWorkspaceCloseRetry(t *testing.T) {
	wm := &WorkspaceManager{ownedBuffer: true, totalBytes: 256}
	failure := errors.New("free failed")
	calls := 0
	free := func() error {
		calls++
		if calls == 1 {
			return failure
		}
		return nil
	}
	if err := wm.closeWith(free); !errors.Is(err, failure) {
		t.Fatal(err)
	}
	if !wm.ownedBuffer || wm.totalBytes != 256 {
		t.Fatal("failed free discarded workspace ownership")
	}
	if err := wm.EnsureSize(256); !errors.Is(err, ErrClosed) {
		t.Fatal("failed close allowed resize")
	}
	if _, err := wm.SliceAt(0, 1); !errors.Is(err, ErrClosed) {
		t.Fatal("failed close allowed slice")
	}
	if err := wm.closeWith(free); err != nil {
		t.Fatal(err)
	}
	if wm.ownedBuffer || wm.totalBytes != 0 {
		t.Fatal("successful retry retained storage")
	}
	if err := wm.closeWith(free); err != nil || calls != 2 {
		t.Fatal("workspace double free")
	}
}

func TestConcurrentResourceClose(t *testing.T) {
	calls := map[string]int{}
	p, ops := cleanupFixture(func(name string) error { calls[name]++; return nil })
	var wg sync.WaitGroup
	for range 16 {
		wg.Go(func() {
			if err := p.closeWith(ops); err != nil {
				t.Error(err)
			}
		})
	}
	wg.Wait()
	for name, count := range calls {
		if count != 1 {
			t.Fatalf("%s closed %d times", name, count)
		}
	}
}
