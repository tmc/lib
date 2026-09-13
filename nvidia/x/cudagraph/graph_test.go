package cudagraph_test

import (
	"errors"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudagraph"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudatest"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

const incrementKernel = `
extern "C" __global__ void increment(unsigned int *p, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) p[i]++;
}
`

func TestGraphCapture(t *testing.T) {
	dev := cudatest.Require(t)
	stream, err := dev.NewStream()
	if err != nil {
		t.Fatalf("NewStream failed: %v", err)
	}
	defer stream.Destroy()

	g, err := cudagraph.Capture(stream, func() {
		// Captured operations on stream
	})
	if err != nil {
		t.Fatalf("Capture failed: %v", err)
	}
	defer g.Destroy()

	exec, err := g.Instantiate()
	if err != nil {
		t.Fatalf("Instantiate failed: %v", err)
	}
	defer exec.Destroy()

	if err := exec.Launch(stream); err != nil {
		t.Fatalf("Launch failed: %v", err)
	}

	if err := stream.Synchronize(nil); err != nil {
		t.Fatalf("Synchronize failed: %v", err)
	}
}

func TestGraphReplayOutputAndLifetime(t *testing.T) {
	dev := cudatest.Require(t)
	compiled, err := nvrtcx.Compile(incrementKernel, nvrtcx.WithDeviceArch(dev), nvrtcx.WithNoCache())
	if err != nil {
		t.Fatal(err)
	}
	mod, err := compiled.Load(dev)
	if err != nil {
		t.Fatal(err)
	}
	defer dev.WithContext(mod.Unload)
	fn, err := mod.GetFunction("increment")
	if err != nil {
		t.Fatal(err)
	}
	stream, err := dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer dev.WithContext(stream.Destroy)

	const n = 257
	direct, err := cudamem.Alloc[uint32](dev, n)
	if err != nil {
		t.Fatal(err)
	}
	defer dev.WithContext(direct.Free)
	replayed, err := cudamem.Alloc[uint32](dev, n)
	if err != nil {
		t.Fatal(err)
	}
	defer dev.WithContext(replayed.Free)
	zero := make([]uint32, n)
	if err := direct.CopyFromHost(zero); err != nil {
		t.Fatal(err)
	}
	if err := replayed.CopyFromHost(zero); err != nil {
		t.Fatal(err)
	}

	launch := func(dst cudamem.Slice[uint32]) error {
		return cudalaunch.Kernel(fn).Grid((n+127)/128).Block(128).On(stream).Launch(dst, int32(n))
	}
	var captureErr error
	graph, err := cudagraph.Capture(stream, func() { captureErr = launch(replayed) })
	if err != nil {
		t.Fatal(err)
	}
	if captureErr != nil {
		graph.Destroy()
		t.Fatalf("captured launch: %v", captureErr)
	}
	exec, err := graph.Instantiate()
	if err != nil {
		graph.Destroy()
		t.Fatal(err)
	}
	// An instantiated graph owns its executable topology independently of the
	// source graph. Destroy it before replay to exercise that lifetime.
	if err := graph.Destroy(); err != nil {
		exec.Destroy()
		t.Fatal(err)
	}
	defer exec.Destroy()

	for range 3 {
		if err := launch(direct); err != nil {
			t.Fatal(err)
		}
		if err := exec.Launch(stream); err != nil {
			t.Fatal(err)
		}
	}
	if err := stream.Synchronize(t.Context()); err != nil {
		t.Fatal(err)
	}
	want := make([]uint32, n)
	for i := range want {
		want[i] = 3
	}
	gotDirect := make([]uint32, n)
	gotReplay := make([]uint32, n)
	if err := dev.WithContext(func() error {
		if err := direct.CopyToHost(gotDirect); err != nil {
			return err
		}
		return replayed.CopyToHost(gotReplay)
	}); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(gotDirect, want) || !reflect.DeepEqual(gotReplay, gotDirect) {
		t.Fatalf("output mismatch: direct=%v replay=%v want=%v", gotDirect[:8], gotReplay[:8], want[:8])
	}
}

func TestBucketManager(t *testing.T) {
	bm := cudagraph.NewBucketManager([]int{128, 256, 512, 1024}, func(s *cuda.Stream, dim int) (*cudagraph.Graph, error) {
		return cudagraph.New()
	})
	defer bm.Close()

	if bm == nil {
		t.Fatal("expected non-nil bucket manager")
	}
}

func TestBucketManagerRejectsInvalidConfiguration(t *testing.T) {
	tests := []struct {
		name    string
		manager *cudagraph.BucketManager
		dim     int
		want    string
	}{
		{"nil manager", nil, 1, "nil bucket manager"},
		{"no buckets", cudagraph.NewBucketManager(nil, func(*cuda.Stream, int) (*cudagraph.Graph, error) { return nil, nil }), 1, "no buckets configured"},
		{"nil builder", cudagraph.NewBucketManager([]int{1}, nil), 1, "nil bucket builder"},
		{"zero bucket", cudagraph.NewBucketManager([]int{0}, func(*cuda.Stream, int) (*cudagraph.Graph, error) { return nil, nil }), 1, "invalid bucket 0"},
		{"duplicate bucket", cudagraph.NewBucketManager([]int{2, 1, 2}, func(*cuda.Stream, int) (*cudagraph.Graph, error) { return nil, nil }), 1, "duplicate bucket 2"},
		{"zero dimension", cudagraph.NewBucketManager([]int{1}, func(*cuda.Stream, int) (*cudagraph.Graph, error) { return nil, nil }), 0, "invalid dimension 0"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, _, err := test.manager.GetExec(nil, test.dim)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("GetExec error = %v, want text %q", err, test.want)
			}
			if err := test.manager.Close(); err != nil {
				t.Fatalf("Close: %v", err)
			}
		})
	}
}

func TestClosedHandlesRejectOperations(t *testing.T) {
	var graph *cudagraph.Graph
	if graph.Raw() != 0 {
		t.Fatal("nil graph has nonzero raw handle")
	}
	if err := graph.Destroy(); err != nil {
		t.Fatalf("nil Graph.Destroy: %v", err)
	}
	if _, err := graph.Instantiate(); err == nil || !strings.Contains(err.Error(), "graph is closed") {
		t.Fatalf("nil Graph.Instantiate error = %v", err)
	}

	var exec *cudagraph.Exec
	if err := exec.Destroy(); err != nil {
		t.Fatalf("nil Exec.Destroy: %v", err)
	}
	if err := exec.Launch(nil); err == nil || !strings.Contains(err.Error(), "executable graph is closed") {
		t.Fatalf("nil Exec.Launch error = %v", err)
	}
	if err := exec.Update(nil); err == nil || !strings.Contains(err.Error(), "executable graph is closed") {
		t.Fatalf("nil Exec.Update error = %v", err)
	}
}

func TestBucketManagerBuildDoesNotHoldManagerLock(t *testing.T) {
	started := make(chan struct{})
	release := make(chan struct{})
	bm := cudagraph.NewBucketManager([]int{1}, func(*cuda.Stream, int) (*cudagraph.Graph, error) {
		close(started)
		<-release
		return nil, errors.New("stopped")
	})
	result := make(chan error, 1)
	go func() {
		_, _, err := bm.GetExec(nil, 1)
		result <- err
	}()
	<-started
	closed := make(chan error, 1)
	go func() { closed <- bm.Close() }()
	select {
	case <-closed:
		t.Fatal("Close returned before the active build stopped")
	case <-time.After(20 * time.Millisecond):
	}
	close(release)
	if err := <-closed; err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := <-result; err == nil || !strings.Contains(err.Error(), "stopped") {
		t.Fatalf("GetExec error = %v, want stopped", err)
	}
}

func TestBucketManagerDeduplicatesConcurrentBuild(t *testing.T) {
	started := make(chan struct{})
	release := make(chan struct{})
	var calls atomic.Int32
	bm := cudagraph.NewBucketManager([]int{1}, func(*cuda.Stream, int) (*cudagraph.Graph, error) {
		if calls.Add(1) == 1 {
			close(started)
		}
		<-release
		return nil, errors.New("build failed")
	})
	const callers = 8
	errs := make(chan error, callers)
	var ready sync.WaitGroup
	ready.Add(callers)
	start := make(chan struct{})
	for range callers {
		go func() {
			ready.Done()
			<-start
			_, _, err := bm.GetExec(nil, 1)
			errs <- err
		}()
	}
	ready.Wait()
	close(start)
	<-started
	time.Sleep(20 * time.Millisecond)
	close(release)
	for range callers {
		if err := <-errs; err == nil || !strings.Contains(err.Error(), "build failed") {
			t.Fatalf("GetExec error = %v, want build failed", err)
		}
	}
	if got := calls.Load(); got != 1 {
		t.Fatalf("builder calls = %d, want 1", got)
	}
	if err := bm.Close(); err != nil {
		t.Fatal(err)
	}
}
