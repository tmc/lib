package cudagraph_test

import (
	"context"
	"runtime"
	"testing"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudagraph"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudatest"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// scaleKernelSource is a second module defining the same symbol with different
// arithmetic, so a launch says which module the executable is running. Testing
// argument values alone would not: the native graph may copy them.
const scaleKernelSource = `
extern "C" __global__ void addOffset(unsigned int *p, int n, unsigned int offset) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		p[i] *= offset;
	}
}
`

// loadKernel compiles src as its own module and returns its addOffset.
func loadKernel(t *testing.T, dev *cuda.Device, src string) *cuda.Kernel {
	t.Helper()
	compiled, err := nvrtcx.Compile(src, nvrtcx.WithDeviceArch(dev), nvrtcx.WithNoCache())
	if err != nil {
		t.Fatal(err)
	}
	raw, err := compiled.Load(dev)
	if err != nil {
		t.Fatal(err)
	}
	mod, err := cuda.OwnModule(dev, raw)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { mod.Close() })
	kern, err := mod.Function("addOffset")
	if err != nil {
		t.Fatal(err)
	}
	return kern
}

// TestExecOutlivesTheGraphItWasUpdatedFrom is the native gate for the
// ownership transfer in Exec.Update. The updated executable runs the second
// module's kernel, whose parameters point into the second graph's argument
// storage. That graph is then destroyed and dropped, so only the Exec's own
// reference keeps the storage alive; the driver reads it through an untraced
// uintptr and the collector will not find it otherwise.
//
// The two modules define the same symbol with different arithmetic, so the
// output identifies which module ran rather than merely which values were
// passed.
func TestExecOutlivesTheGraphItWasUpdatedFrom(t *testing.T) {
	ctx := context.Background()
	dev := cudatest.Require(t)

	add := loadKernel(t, dev, updateKernelSource)
	scale := loadKernel(t, dev, scaleKernelSource)

	stream, err := dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer dev.WithContext(stream.Destroy)

	const (
		n      = 256
		block  = 128
		factor = uint32(3)
	)
	seed := make([]uint32, n)
	for i := range seed {
		seed[i] = uint32(i)
	}
	buf, err := cudamem.Alloc[uint32](dev, n)
	if err != nil {
		t.Fatal(err)
	}
	defer dev.WithContext(buf.Free)
	if err := buf.CopyFromHost(seed); err != nil {
		t.Fatal(err)
	}

	// build returns a one-node graph launching kern over buf.
	build := func(kern *cuda.Kernel) *cudagraph.Graph {
		t.Helper()
		g, err := cudagraph.New(dev)
		if err != nil {
			t.Fatal(err)
		}
		l := cudalaunch.OwnedKernel(kern).Grid(n / block).Block(block)
		if _, err := g.AddLauncher(l, []any{buf, int32(n), factor}); err != nil {
			t.Fatal(err)
		}
		return g
	}

	first := build(add)
	exec, err := first.Instantiate()
	if err != nil {
		t.Fatal(err)
	}
	defer dev.WithContext(exec.Destroy)
	if err := first.Destroy(); err != nil {
		t.Fatal(err)
	}

	second := build(scale)
	if err := exec.Update(second); err != nil {
		t.Fatal(err)
	}
	// Destroy and drop the graph the executable now runs. If Update did not
	// transfer ownership, nothing reaches its argument storage from here.
	if err := second.Destroy(); err != nil {
		t.Fatal(err)
	}
	second = nil
	runtime.GC()
	runtime.GC()

	if err := exec.Launch(stream); err != nil {
		t.Fatal(err)
	}
	if err := stream.Synchronize(ctx); err != nil {
		t.Fatal(err)
	}
	got := make([]uint32, n)
	if err := buf.CopyToHost(got); err != nil {
		t.Fatal(err)
	}
	for i := range got {
		if want := seed[i] * factor; got[i] != want {
			t.Fatalf("element %d = %d, want %d; the executable did not run the updated module", i, got[i], want)
		}
	}
}
