package cudagraph_test

import (
	"context"
	"testing"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudagraph"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudatest"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// TestAddLaunchMatchesDirectLaunch runs one kernel two ways — launched directly
// on a stream, and as a graph node built from the same launcher's Description —
// and requires identical device output. It is the acceptance check for the
// conversion that replaced reflection over Launcher's unexported fields.
func TestAddLaunchMatchesDirectLaunch(t *testing.T) {
	ctx := context.Background()
	dev := cudatest.Require(t)

	compiled, err := nvrtcx.Compile(updateKernelSource, nvrtcx.WithDeviceArch(dev), nvrtcx.WithNoCache())
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
	defer mod.Close()

	kern, err := mod.Function("addOffset")
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
		offset = uint32(5)
	)
	seed := make([]uint32, n)
	for i := range seed {
		seed[i] = uint32(i)
	}

	// run applies addOffset once to a freshly seeded buffer and returns it.
	run := func(apply func(l *cudalaunch.Launcher, buf cudamem.Slice[uint32]) error) []uint32 {
		t.Helper()
		buf, err := cudamem.Alloc[uint32](dev, n)
		if err != nil {
			t.Fatal(err)
		}
		defer dev.WithContext(buf.Free)
		if err := buf.CopyFromHost(seed); err != nil {
			t.Fatal(err)
		}
		l := cudalaunch.OwnedKernel(kern).Grid(n / block).Block(block).On(stream)
		if err := apply(l, buf); err != nil {
			t.Fatal(err)
		}
		if err := stream.Synchronize(ctx); err != nil {
			t.Fatal(err)
		}
		got := make([]uint32, n)
		if err := buf.CopyToHost(got); err != nil {
			t.Fatal(err)
		}
		return got
	}

	direct := run(func(l *cudalaunch.Launcher, buf cudamem.Slice[uint32]) error {
		return l.Launch(buf, int32(n), offset)
	})

	graphed := run(func(l *cudalaunch.Launcher, buf cudamem.Slice[uint32]) error {
		desc, err := l.Describe()
		if err != nil {
			return err
		}
		if desc.Kernel() == nil {
			t.Error("Description.Kernel() = nil for an owned kernel; the module is not retained")
		}
		g, err := cudagraph.New(dev)
		if err != nil {
			return err
		}
		defer g.Destroy()
		if _, err := g.AddLaunch(desc, []any{buf, int32(n), offset}); err != nil {
			return err
		}
		exec, err := g.Instantiate()
		if err != nil {
			return err
		}
		defer exec.Destroy()
		// Destroy the graph before launching: the Exec must still own the
		// argument storage the node's parameters point into.
		if err := g.Destroy(); err != nil {
			return err
		}
		return exec.Launch(stream)
	})

	for i := range direct {
		if direct[i] != graphed[i] {
			t.Fatalf("element %d: direct = %d, graph = %d", i, direct[i], graphed[i])
		}
		if want := seed[i] + offset; direct[i] != want {
			t.Fatalf("element %d: got %d, want %d", i, direct[i], want)
		}
	}
}

// TestAddLaunchArgumentsCopied checks that the node executes the arguments as
// they were at AddLaunch time. A caller mutating its own variables afterwards
// must not silently change what an instantiated graph runs.
func TestAddLaunchArgumentsCopied(t *testing.T) {
	ctx := context.Background()
	dev := cudatest.Require(t)

	compiled, err := nvrtcx.Compile(updateKernelSource, nvrtcx.WithDeviceArch(dev), nvrtcx.WithNoCache())
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
	defer mod.Close()

	kern, err := mod.Function("addOffset")
	if err != nil {
		t.Fatal(err)
	}
	stream, err := dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer dev.WithContext(stream.Destroy)

	const (
		n     = 256
		block = 128
	)
	buf, err := cudamem.Alloc[uint32](dev, n)
	if err != nil {
		t.Fatal(err)
	}
	defer dev.WithContext(buf.Free)
	if err := buf.CopyFromHost(make([]uint32, n)); err != nil {
		t.Fatal(err)
	}

	desc, err := cudalaunch.OwnedKernel(kern).Grid(n / block).Block(block).Describe()
	if err != nil {
		t.Fatal(err)
	}

	g, err := cudagraph.New(dev)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Destroy()

	offset := uint32(7)
	if _, err := g.AddLaunch(desc, []any{buf, int32(n), offset}); err != nil {
		t.Fatal(err)
	}
	offset = 99 // must not reach the node

	exec, err := g.Instantiate()
	if err != nil {
		t.Fatal(err)
	}
	defer exec.Destroy()
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
	for i, v := range got {
		if v != 7 {
			t.Fatalf("element %d = %d, want 7 (the value at AddLaunch time)", i, v)
		}
	}
}
