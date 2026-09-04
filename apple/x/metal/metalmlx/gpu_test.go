package metalmlx

import (
	"testing"
	"unsafe"

	"github.com/tmc/apple/metal"
)

// testKernels is the source every GPU test compiles.
const testKernels = `
#include <metal_stdlib>
using namespace metal;

kernel void arange(device float* out [[buffer(0)]],
                   constant uint& n [[buffer(1)]],
                   uint i [[thread_position_in_grid]]) {
  if (i < n) out[i] = float(i);
}

kernel void add_one(device float* inout [[buffer(0)]],
                    constant uint& n [[buffer(1)]],
                    uint i [[thread_position_in_grid]]) {
  if (i < n) inout[i] = inout[i] + 1.0f;
}
`

func testDevice(t *testing.T) *Device {
	t.Helper()
	d, err := GetDevice()
	if err != nil {
		t.Skipf("no Metal device: %v", err)
	}
	return d
}

func testLibrary(t *testing.T, d *Device) metal.MTLLibrary {
	t.Helper()
	l, err := d.LibraryFromSource("metalmlx_test", CompileOptions{},
		func() (string, error) { return testKernels, nil })
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	return l
}

func TestAllocatorReuse(t *testing.T) {
	d := testDevice(t)
	a := d.Allocator()

	small, err := a.Malloc(64)
	if err != nil {
		t.Fatal(err)
	}
	if got := small.Length(); got != 64 {
		t.Errorf("small buffer length = %d, want 64", got)
	}
	a.Free(small)

	// Sizes above a page are rounded up to a page multiple.
	page := uint64(vmPageSize())
	buf, err := a.Malloc(page + 1)
	if err != nil {
		t.Fatal(err)
	}
	if got := uint64(buf.Length()); got != 2*page {
		t.Errorf("buffer length = %d, want %d", got, 2*page)
	}
	active := a.ActiveMemory()
	if active < 2*page {
		t.Errorf("ActiveMemory = %d, want at least %d", active, 2*page)
	}
	a.Free(buf)
	if a.ActiveMemory() != active-2*page {
		t.Errorf("ActiveMemory after free = %d, want %d", a.ActiveMemory(), active-2*page)
	}

	// The freed buffer is cached, so the same request comes back from it.
	again, err := a.Malloc(page + 1)
	if err != nil {
		t.Fatal(err)
	}
	if again.GetID() != buf.GetID() {
		t.Error("second Malloc did not reuse the cached buffer")
	}
	a.Free(again)
	a.ClearCache()
	if got := a.CacheMemory(); got != 0 {
		t.Errorf("CacheMemory after ClearCache = %d, want 0", got)
	}
}

func TestEncoderDispatchChain(t *testing.T) {
	d := testDevice(t)
	lib := testLibrary(t, d)
	arange, err := d.Kernel(lib, "arange")
	if err != nil {
		t.Fatal(err)
	}
	addOne, err := d.Kernel(lib, "add_one")
	if err != nil {
		t.Fatal(err)
	}
	// The pipeline cache returns the same object for the same name.
	if again, _ := d.Kernel(lib, "arange"); again.GetID() != arange.GetID() {
		t.Error("kernel cache returned a different pipeline")
	}

	e, err := NewCommandEncoder(d)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()

	const n = 1024
	a := d.Allocator()
	out, err := a.Malloc(n * 4)
	if err != nil {
		t.Fatal(err)
	}
	defer a.Free(out)

	count := uint32(n)
	e.SetComputePipelineState(arange)
	e.SetOutputBuffer(out, 0, 0)
	e.SetBytes(mustValueBytes(&count), 1)
	e.DispatchThreads(Size{Width: n}, Size{Width: 256})

	// The second dispatch reads what the first wrote, so the encoder has to
	// put a barrier between them.
	e.SetComputePipelineState(addOne)
	e.SetOutputBuffer(out, 0, 0)
	e.SetBytes(mustValueBytes(&count), 1)
	if !e.needsBarrier {
		t.Error("dependent dispatch did not ask for a barrier")
	}
	e.DispatchThreads(Size{Width: n}, Size{Width: 256})

	done := make(chan struct{})
	e.EndEncoding()
	e.Commit(func() { close(done) })
	if err := e.Synchronize(); err != nil {
		t.Fatal(err)
	}
	<-done

	got := unsafe.Slice((*float32)(out.Contents()), n)
	for i := range got {
		if want := float32(i) + 1; got[i] != want {
			t.Fatalf("out[%d] = %v, want %v", i, got[i], want)
		}
	}
}

func TestEncoderCrossEncoderFence(t *testing.T) {
	d := testDevice(t)
	lib := testLibrary(t, d)
	arange, _ := d.Kernel(lib, "arange")
	addOne, _ := d.Kernel(lib, "add_one")

	e, err := NewCommandEncoder(d)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()

	const n = 256
	a := d.Allocator()
	out, err := a.Malloc(n * 4)
	if err != nil {
		t.Fatal(err)
	}
	defer a.Free(out)

	count := uint32(n)
	// Two compute encoders in a row: the second waits on the first's fence.
	for i, kernel := range []metal.MTLComputePipelineState{arange, addOne} {
		e.SetComputePipelineState(kernel)
		e.SetOutputBuffer(out, 0, 0)
		e.SetBytes(mustValueBytes(&count), 1)
		e.DispatchThreads(Size{Width: n}, Size{Width: 64})
		e.EndEncoding()
		if i == 0 && len(e.prevCEOutputs) != 1 {
			t.Fatalf("prevCEOutputs = %d entries, want 1", len(e.prevCEOutputs))
		}
	}
	if err := e.Synchronize(); err != nil {
		t.Fatal(err)
	}
	got := unsafe.Slice((*float32)(out.Contents()), n)
	if got[10] != 11 {
		t.Errorf("out[10] = %v, want 11", got[10])
	}
}

func TestEventSignalAndWait(t *testing.T) {
	d := testDevice(t)
	ev, err := NewEvent(d)
	if err != nil {
		t.Fatal(err)
	}
	defer ev.Destroy()
	signaled, err := ev.IsSignaled(1)
	if err != nil {
		t.Fatal(err)
	}
	if signaled {
		t.Error("new event reports signaled")
	}

	e, err := NewCommandEncoder(d)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()

	if err := e.SignalEvent(ev, 3); err != nil {
		t.Fatal(err)
	}
	e.Commit(nil)
	if err := ev.Wait(3); err != nil {
		t.Fatal(err)
	}
	if got, err := ev.Value(); err != nil || got != 3 {
		t.Errorf("event value = %d, want 3", got)
	}
	signaled, err = ev.IsSignaled(3)
	if err != nil {
		t.Fatal(err)
	}
	if !signaled {
		t.Error("event should be signaled")
	}
}

func TestFenceEventPath(t *testing.T) {
	d := testDevice(t)
	f, err := NewFence(d)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Destroy()
	if f.fast {
		t.Fatal("fence should use the event path by default")
	}

	e, err := NewCommandEncoder(d)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()

	a := d.Allocator()
	buf, err := a.Malloc(64)
	if err != nil {
		t.Fatal(err)
	}
	defer a.Free(buf)

	if err := f.Update(e, buf, false); err != nil {
		t.Fatal(err)
	}
	e.Commit(nil)
	if err := f.WaitCPU(); err != nil {
		t.Fatal(err)
	}
	if got, err := f.Count(); err != nil || got != 1 {
		t.Errorf("count = %d, want 1 (err %v)", got, err)
	}
}

func TestFenceFastPath(t *testing.T) {
	t.Setenv("MLX_METAL_FAST_SYNCH", "1")
	d := testDevice(t)
	f, err := NewFence(d)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Destroy()
	if !f.fast {
		t.Skip("device does not support the fast fence path")
	}

	e, err := NewCommandEncoder(d)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()

	a := d.Allocator()
	buf, err := a.Malloc(64)
	if err != nil {
		t.Fatal(err)
	}
	defer a.Free(buf)

	if err := f.Update(e, buf, true); err != nil {
		t.Fatal(err)
	}
	e.EndEncoding()
	e.Commit(nil)
	if err := f.WaitCPU(); err != nil {
		t.Fatal(err)
	}
	if got, err := f.Count(); err != nil || got != 1 {
		t.Errorf("count = %d, want 1", got)
	}
}

func TestResidencySetsWiring(t *testing.T) {
	d := testDevice(t)
	r := d.ResidencySets()
	if !r.Enabled() {
		t.Skip("residency sets not supported")
	}
	a := d.Allocator()

	prev, err := a.SetWiredLimit(4 << 20)
	if err != nil {
		t.Fatal(err)
	}
	defer a.SetWiredLimit(prev)

	// Raising the limit wires what is already tracked, such as the small
	// allocation heap, so the new buffer is measured against that baseline.
	base := r.WiredSize()
	buf, err := a.Malloc(1 << 20)
	if err != nil {
		t.Fatal(err)
	}
	if got := r.WiredSize(); got <= base {
		t.Errorf("WiredSize = %d after allocating under the wired limit, want more than %d", got, base)
	}
	a.Release(buf)
	if got := r.WiredSize(); got != base {
		t.Errorf("WiredSize = %d after release, want %d", got, base)
	}
	if _, err := a.SetWiredLimit(1 << 62); err == nil {
		t.Error("SetWiredLimit past the working set size should fail")
	}
}
