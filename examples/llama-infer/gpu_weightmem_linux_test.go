//go:build linux && !tpu

package main

import (
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// TestCUDAHalfWeightsHalveDeviceMemory measures the device memory a weight
// upload consumes, in f32 and in f16, and requires the second to be half the
// first.
//
// This is the only assertion in the half-weight change that demonstrates a
// memory saving rather than restating one. WeightSlice.Bytes() cannot do it:
// it is defined as elems*dtype.Bytes(), so an assertion against
// elems*dtype.Bytes() holds whatever the device allocation turns out to be.
// Two independent measurements are used instead:
//
//   - cudamem.Tracker, which accounts the byte counts actually passed to
//     cuMemAlloc.
//   - cuMemGetAddressRange on the resulting device pointer, which is the
//     driver's own record of how large the allocation it made is. It knows
//     nothing about this package's bookkeeping, so it is what distinguishes a
//     real saving from a relabelled one.
//
// Both are exact, and both ratios must be exactly two. No tolerance is involved.
//
// cuMemGetInfo is deliberately not used for the ratio. This device is a GB10,
// whose memory is unified with the host's: nvidia-smi reports no memory.total
// and cuMemGetInfo's free figure moves with host allocations as well as device
// ones. uploadWeights builds a []uint16 host buffer on the half path and none
// on the f32 path, so differencing free memory across the two charges the half
// path for its host packing buffer and reports a ratio near one. It is still
// read here, and required to fall by at least the allocation, because that
// direction is sound on any device; the ratio it implies is only logged.
func TestCUDAHalfWeightsHalveDeviceMemory(t *testing.T) {
	dev := requireCUDAHardware(t)

	// 8M weights: 32 MiB in f32, 16 MiB in f16, both far above any plausible
	// allocation granularity.
	const elems = 8 << 20
	host := make([]float32, elems)
	for i := range host {
		host[i] = float32(i%1000) * 1e-3
	}

	measure := func(d WeightDType) (tracked, allocation uint64, freeDrop int64) {
		var tracker cudamem.Tracker
		arena := cudamem.NewTrackedArena(dev, &tracker)
		defer arena.Free()

		freeBefore := deviceFreeBytes(t, dev)
		dw, err := uploadWeights(arena, d, host)
		if err != nil {
			t.Fatal(err)
		}
		freeAfter := deviceFreeBytes(t, dev)
		if dw.Len() != elems {
			t.Fatalf("%s: uploaded %d weights, want %d", d, dw.Len(), elems)
		}
		cur, _ := tracker.Snapshot()
		return cur, driverAllocationBytes(t, dev, dw.Ptr()), freeBefore - freeAfter
	}

	f32Tracked, f32Alloc, f32Free := measure(WeightF32)
	f16Tracked, f16Alloc, f16Free := measure(WeightF16)

	t.Logf("f32: tracker %d bytes, driver allocation %d bytes, free-memory drop %d bytes", f32Tracked, f32Alloc, f32Free)
	t.Logf("f16: tracker %d bytes, driver allocation %d bytes, free-memory drop %d bytes", f16Tracked, f16Alloc, f16Free)

	for _, c := range []struct {
		name string
		got  uint64
		want uint64
	}{
		{"f32 tracker", f32Tracked, uint64(elems) * 4},
		{"f32 driver allocation", f32Alloc, uint64(elems) * 4},
		{"f16 tracker", f16Tracked, uint64(elems) * 2},
		{"f16 driver allocation", f16Alloc, uint64(elems) * 2},
	} {
		if c.got != c.want {
			t.Errorf("%s is %d bytes, want %d", c.name, c.got, c.want)
		}
	}
	if f16Tracked*2 != f32Tracked {
		t.Errorf("f16 allocated %d bytes against f32's %d; want exactly half", f16Tracked, f32Tracked)
	}
	if f16Alloc*2 != f32Alloc {
		t.Errorf("the driver sized the f16 allocation at %d bytes against f32's %d; want exactly half", f16Alloc, f32Alloc)
	}

	// Free memory has to fall by at least what was allocated, on any device.
	// It may fall by more here, since this one shares its memory with the host
	// and the half path packs into a host buffer; the implied ratio is logged
	// rather than asserted for that reason.
	if f32Free < int64(f32Tracked) || f16Free < int64(f16Tracked) {
		t.Errorf("free memory fell by %d and %d bytes for allocations of %d and %d; the allocations are not reaching the device",
			f32Free, f16Free, f32Tracked, f16Tracked)
	}
	t.Logf("free-memory ratio %.3f (host-shared memory, not a device-only figure)",
		float64(f32Free)/float64(f16Free))
}

// deviceFreeBytes returns the device's free memory as the driver reports it.
func deviceFreeBytes(t *testing.T, dev *cuda.Device) int64 {
	t.Helper()
	var free uint
	err := dev.WithContext(func() error {
		f, _, err := driver.MemGetInfo()
		free = f
		return err
	})
	if err != nil {
		t.Fatalf("MemGetInfo: %v", err)
	}
	return int64(free)
}

// driverAllocationBytes returns the size the driver records for the allocation
// containing p.
//
// This is the independent oracle: it is the driver's own answer, derived from
// nothing this package computed.
func driverAllocationBytes(t *testing.T, dev *cuda.Device, p driver.DevicePtr) uint64 {
	t.Helper()
	var size uint
	err := dev.WithContext(func() error {
		_, err := p.MemGetAddressRange(&size)
		return err
	})
	if err != nil {
		t.Fatalf("MemGetAddressRange: %v", err)
	}
	return uint64(size)
}

// TestCUDAStoreKVCacheOffsetActivationBase launches StoreKVCache with a source
// activation pointer that is a row of a larger buffer and a destination row
// offset by an odd kv_dim.
//
// The alignment work on this branch covered weight pointers only. Activation
// float4 accesses were left unguarded, and the native tests could not see it:
// every activation buffer they allocate is its own 256-byte-aligned
// allocation, so the odd-dimension cases exercised the weight fallback while
// proving nothing about an offset activation base.
//
// An audit of the production launch sites found no live fault. Only
// gpu_batch_cuda.go:225-228 passes row-offset activation pointers, and of the
// kernels they reach only StoreKVCache casts to float4; the offset is
// row*kvDim, and kvDim is a multiple of four for every shipping model. But
// nothing in the loader validates that, and StoreKVCache also computes its own
// row offset internally, k_cache + pos*kv_dim, on both the single-token and the
// batch path. So the guard was added and this test exercises the case
// regardless of whether any current configuration reaches it.
//
// kv_dim is 6 here: even, so RoPE's head_dim/2 still works, but not a multiple
// of four, so both the source row and the destination row are 8 bytes past a
// 16-byte boundary. Without the guard this is a misaligned-address fault, not a
// wrong answer.
func TestCUDAStoreKVCacheOffsetActivationBase(t *testing.T) {
	dev := requireCUDAHardware(t)
	stream, err := dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Destroy()

	modules, err := CompileGPUModulesAs(dev, WeightF32)
	if err != nil {
		t.Fatal(err)
	}
	defer modules.Close()

	const (
		numKVHeads = 3
		headDim    = 2
		kvDim      = numKVHeads * headDim // 6: even, not a multiple of four
		rows       = 4
		pos        = 1
		srcRow     = 1
	)
	k := deterministicFloats(rows*kvDim, 11)
	v := deterministicFloats(rows*kvDim, 12)

	arena := cudamem.NewArena(dev)
	defer arena.Free()
	dk := cudamem.AllocIn[float32](arena, "k", len(k))
	dv := cudamem.AllocIn[float32](arena, "v", len(v))
	kCache := cudamem.AllocIn[float32](arena, "kcache", rows*kvDim)
	vCache := cudamem.AllocIn[float32](arena, "vcache", rows*kvDim)
	if err := arena.Err(); err != nil {
		t.Fatal(err)
	}
	if err := dk.CopyFromHost(k); err != nil {
		t.Fatal(err)
	}
	if err := dv.CopyFromHost(v); err != nil {
		t.Fatal(err)
	}
	zero := make([]float32, rows*kvDim)
	if err := kCache.CopyFromHost(zero); err != nil {
		t.Fatal(err)
	}
	if err := vCache.CopyFromHost(zero); err != nil {
		t.Fatal(err)
	}

	// The source is a row of a larger buffer, exactly as gpu_batch_cuda.go
	// slices sp.K and sp.V per token. srcRow*kvDim is 6 elements, so the base
	// is 24 bytes in: 16-byte misaligned.
	ksrc := dk.Slice(srcRow*kvDim, (srcRow+1)*kvDim)
	vsrc := dv.Slice(srcRow*kvDim, (srcRow+1)*kvDim)

	if err := cudalaunch.OwnedKernel(modules.StoreKV).Grid(2).Block(64).On(stream).
		Launch(ksrc, vsrc, kCache, vCache, pos, numKVHeads, headDim); err != nil {
		t.Fatalf("launch: %v", err)
	}
	if err := stream.Synchronize(t.Context()); err != nil {
		t.Fatalf("sync: %v", err)
	}

	gotK := make([]float32, rows*kvDim)
	gotV := make([]float32, rows*kvDim)
	if err := kCache.CopyToHost(gotK); err != nil {
		t.Fatal(err)
	}
	if err := vCache.CopyToHost(gotV); err != nil {
		t.Fatal(err)
	}

	// The copy is bit-exact: the kernel moves floats and computes nothing.
	for i := 0; i < rows*kvDim; i++ {
		wantK, wantV := float32(0), float32(0)
		if i >= pos*kvDim && i < (pos+1)*kvDim {
			j := srcRow*kvDim + (i - pos*kvDim)
			wantK, wantV = k[j], v[j]
		}
		if gotK[i] != wantK {
			t.Errorf("k_cache[%d] = %v, want %v", i, gotK[i], wantK)
		}
		if gotV[i] != wantV {
			t.Errorf("v_cache[%d] = %v, want %v", i, gotV[i], wantV)
		}
	}
}
