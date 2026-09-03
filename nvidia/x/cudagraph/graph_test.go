package cudagraph_test

import (
	"reflect"
	"testing"

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
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		t.Fatal(err)
	}
	compiled, err := nvrtcx.Compile(incrementKernel, nvrtcx.WithArch(major, minor), nvrtcx.WithNoCache())
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
