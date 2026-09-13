package cudagraph_test

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudagraph"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudatest"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// swapKernels holds two kernels that agree on nothing but their first argument.
// increment takes two parameters and no shared memory; scale takes three and
// reads a dynamic shared-memory block.
const swapKernels = `
extern "C" __global__ void swapIncrement(unsigned int *p, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) p[i]++;
}

extern "C" __global__ void swapScale(unsigned int *p, int n, unsigned int factor) {
	extern __shared__ unsigned int scratch[];
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	scratch[threadIdx.x] = factor;
	__syncthreads();
	if (i < n) p[i] *= scratch[threadIdx.x];
}
`

// TestUpdateDoesNotValidateKernelIdentity pins the behavior [Exec.Update]
// documents: CUDA checks that the two graphs have the same topology and nothing
// about what the kernel nodes inside them do.
//
// The assertion direction is deliberate and is the opposite of what the name
// "update" suggests. The test asserts the swap SUCCEEDS. A test written to
// assert that Update rejects a different kernel would fail against the driver
// and invite a fix that papers over the behavior a caller has to defend
// against. The guard belongs above the driver; this documents permission.
//
// Returning nil from Update is the weaker half of the claim, since a driver
// that silently ignored the new graph would also return nil. So the graph is
// replayed afterwards and the output is read back: the buffer holds what
// swapScale computes, which is how we know the substitution was applied and
// not merely tolerated.
func TestUpdateDoesNotValidateKernelIdentity(t *testing.T) {
	dev := cudatest.Require(t)
	compiled, err := nvrtcx.Compile(swapKernels, nvrtcx.WithDeviceArch(dev), nvrtcx.WithNoCache())
	if err != nil {
		t.Fatal(err)
	}
	mod, err := compiled.Load(dev)
	if err != nil {
		t.Fatal(err)
	}
	defer dev.WithContext(mod.Unload)

	incFn, err := mod.GetFunction("swapIncrement")
	if err != nil {
		t.Fatal(err)
	}
	scaleFn, err := mod.GetFunction("swapScale")
	if err != nil {
		t.Fatal(err)
	}

	stream, err := dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer dev.WithContext(stream.Destroy)

	const (
		n      = 256
		block  = 128
		factor = 3
	)
	buf, err := cudamem.Alloc[uint32](dev, n)
	if err != nil {
		t.Fatal(err)
	}
	defer dev.WithContext(buf.Free)

	capture := func(launch func() error) *cudagraph.Graph {
		t.Helper()
		var launchErr error
		g, err := cudagraph.Capture(stream, func() { launchErr = launch() })
		if err != nil {
			t.Fatal(err)
		}
		if launchErr != nil {
			g.Destroy()
			t.Fatalf("captured launch: %v", launchErr)
		}
		return g
	}

	// One kernel node holding swapIncrement: two parameters, no shared memory.
	incGraph := capture(func() error {
		return cudalaunch.Kernel(incFn).Grid(n/block).Block(block).On(stream).
			Launch(buf, int32(n))
	})
	defer incGraph.Destroy()

	exec, err := incGraph.Instantiate()
	if err != nil {
		t.Fatal(err)
	}
	defer exec.Destroy()

	// One kernel node holding swapScale: a different function, three parameters
	// instead of two, and a dynamic shared-memory block the first node had none
	// of. The only thing it shares with the graph above is the node count.
	scaleGraph := capture(func() error {
		return cudalaunch.Kernel(scaleFn).Grid(n/block).Block(block).
			SharedMem(block*4).On(stream).
			Launch(buf, int32(n), uint32(factor))
	})
	defer scaleGraph.Destroy()

	if err := exec.Update(scaleGraph); err != nil {
		t.Fatalf("Update with a different kernel, parameter list and shared-memory size = %v, want nil: "+
			"the driver is documented to check topology only, so this swap is expected to be permitted", err)
	}

	// Replay the updated executable over a known input. swapIncrement would
	// leave 2; swapScale leaves 1*factor.
	ones := make([]uint32, n)
	for i := range ones {
		ones[i] = 1
	}
	if err := buf.CopyFromHost(ones); err != nil {
		t.Fatal(err)
	}
	if err := exec.Launch(stream); err != nil {
		t.Fatal(err)
	}
	if err := stream.Synchronize(nil); err != nil {
		t.Fatal(err)
	}
	got := make([]uint32, n)
	if err := buf.CopyToHost(got); err != nil {
		t.Fatal(err)
	}
	for i, v := range got {
		if v == factor {
			continue
		}
		// 2 is what swapIncrement leaves, and means Update returned nil
		// without replacing the node: permitted but not applied.
		if v == 2 {
			t.Fatalf("after Update, buf[%d] = 2: the graph still ran swapIncrement, "+
				"so Update reported success without applying the swap", i)
		}
		t.Fatalf("after Update, buf[%d] = %d, want %d", i, v, factor)
	}
}

// TestUpdateErrorNamesTheNode checks the two shapes a failed update takes: one
// the driver blames on a node and one it does not.
//
// Both arms are needed. A formatter that always named a node would pass the
// second and fail the first; one that never named a node would do the reverse.
// Asserting only the case that carries a handle would not notice a zero handle
// being printed as a node for every topology change.
func TestUpdateErrorNamesTheNode(t *testing.T) {
	dev := cudatest.Require(t)
	compiled, err := nvrtcx.Compile(swapKernels, nvrtcx.WithDeviceArch(dev), nvrtcx.WithNoCache())
	if err != nil {
		t.Fatal(err)
	}
	mod, err := compiled.Load(dev)
	if err != nil {
		t.Fatal(err)
	}
	defer dev.WithContext(mod.Unload)
	incFn, err := mod.GetFunction("swapIncrement")
	if err != nil {
		t.Fatal(err)
	}
	stream, err := dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer dev.WithContext(stream.Destroy)

	const n, block = 256, 128
	buf, err := cudamem.Alloc[uint32](dev, n)
	if err != nil {
		t.Fatal(err)
	}
	defer dev.WithContext(buf.Free)

	launch := func() error {
		return cudalaunch.Kernel(incFn).Grid(n/block).Block(block).On(stream).Launch(buf, int32(n))
	}
	capture := func(fn func() error) *cudagraph.Graph {
		t.Helper()
		var launchErr error
		g, err := cudagraph.Capture(stream, func() { launchErr = fn() })
		if err != nil {
			t.Fatal(err)
		}
		if launchErr != nil {
			g.Destroy()
			t.Fatalf("captured operation: %v", launchErr)
		}
		return g
	}

	one := capture(launch)
	defer one.Destroy()
	exec, err := one.Instantiate()
	if err != nil {
		t.Fatal(err)
	}
	defer exec.Destroy()

	// A topology change is not a node's fault, and the driver leaves both
	// blame handles zero. Naming a node here would be inventing one.
	two := capture(func() error {
		if err := launch(); err != nil {
			return err
		}
		return launch()
	})
	defer two.Destroy()

	err = exec.Update(two)
	if err == nil {
		t.Fatal("Update with a second node = nil, want a topology-change failure")
	}
	if !strings.Contains(err.Error(), "TopologyChanged") {
		t.Errorf("Update error = %v, want it to report a topology change", err)
	}
	if strings.Contains(err.Error(), "node=") {
		t.Errorf("Update error = %v, want no node named: the driver blames no node for a topology change", err)
	}

	// Same node count, different node type: a memcpy where the executable
	// holds a kernel. This one the driver does blame on a node.
	host := make([]uint32, n)
	copyGraph := capture(func() error { return buf.CopyFromHostAsync(host, stream) })
	defer copyGraph.Destroy()

	err = exec.Update(copyGraph)
	if err == nil {
		t.Fatal("Update with a memcpy node = nil, want a node-type-change failure")
	}
	if !strings.Contains(err.Error(), "NodeTypeChanged") {
		t.Errorf("Update error = %v, want it to report a node type change", err)
	}
	if !strings.Contains(err.Error(), "node=0x") {
		t.Errorf("Update error = %v, want the rejected node named", err)
	}
	if strings.Contains(err.Error(), "node=0x0") {
		t.Errorf("Update error = %v, want a real node handle, not a zero one", err)
	}
}
