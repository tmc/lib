package nvcomp_test

import (
	"bytes"
	"os"
	"runtime"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvcomp"
)

// The tests in this file are the only ones here that could show the bindings
// work, as opposed to showing that their shapes agree with the header.
//
// None of them has ever been run. This working tree has no NVIDIA GPU, no CUDA
// driver and no libnvcomp, and the session that wrote them was not permitted
// to reach a machine that has them. Treat them as an unexecuted specification
// of what would count as evidence, not as passing tests.
//
// They are opt-in so that absent hardware cannot masquerade as success. Once
// the environment variable is set, every missing prerequisite is a failure and
// nothing below skips.

// TestBatchedLZ4RoundTrip compresses a batch, decompresses it and compares the
// bytes, then corrupts a compressed chunk and checks that the failure shows up
// in the per-chunk statuses.
//
// The second half is the part worth having. A status array that the library
// never wrote reads as all zeros, and zero is nvcompSuccess, so a test that
// only ever compresses data that works cannot tell a batch that succeeded from
// a batch whose statuses were never filled in. Corrupting one chunk and
// requiring a non-success status for exactly that chunk is what distinguishes
// them.
//
// Prerequisites, all of them:
//   - libnvcomp.so.5 on the loader path, or NVCOMP_LIB_PATH pointing at it.
//     The nvCOMP 5.3.0 redist ships it; nothing in this repository does.
//   - libcuda.so.1 from an installed NVIDIA driver, and a visible GPU. The
//     batched API runs CUDA kernels; there is no host fallback in libnvcomp.
//   - a GPU whose architecture the installed libnvcomp was built for. nvCOMP
//     ships fatbinaries, so this is normally satisfied, but a device newer
//     than the library still fails at launch rather than at load.
//
// The decompression here names the CUDA backend explicitly. On the hardware
// decompression engine a corrupt input is documented as undefined behavior
// with the statuses set to success regardless, so the negative control below
// would be meaningless there.
func TestBatchedLZ4RoundTrip(t *testing.T) {
	if os.Getenv("NVCOMP_PHYSICAL_TEST") != "1" {
		t.Skip("set NVCOMP_PHYSICAL_TEST=1 to run against a real libnvcomp and NVIDIA GPU")
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	check := func(err error) {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
	}

	check(driver.Init(0))
	dev, err := driver.NewDeviceGet(0)
	check(err)
	name, err := dev.GetName()
	check(err)
	ctx, err := dev.PrimaryCtxRetain()
	check(err)
	t.Cleanup(func() { dev.PrimaryCtxRelease() })
	check(ctx.SetCurrent())

	props, err := nvcomp.GetProperties()
	check(err)
	t.Logf("%v from %s on %s", props, nvcomp.Path(), name)

	const (
		numChunks  = 4
		chunkBytes = 4096
	)
	// Data with long runs, so that LZ4 has something to find. What matters
	// here is that the round trip is lossless, not how small the output is.
	uncompressed := make([]byte, numChunks*chunkBytes)
	for i := range uncompressed {
		uncompressed[i] = byte(i / 64)
	}

	opts := nvcomp.CompressOpts{}
	align, err := nvcomp.CompressGetRequiredAlignments(opts)
	check(err)
	if chunkBytes%align.Input != 0 {
		t.Fatalf("chunk stride %d does not satisfy the required input alignment %d", chunkBytes, align.Input)
	}
	maxOut, err := nvcomp.CompressGetMaxOutputChunkSize(chunkBytes, opts)
	check(err)
	if maxOut%align.Output != 0 {
		t.Fatalf("max output chunk size %d is not a multiple of the required output alignment %d", maxOut, align.Output)
	}
	tempBytes, err := nvcomp.CompressGetTempSizeAsync(numChunks, chunkBytes, opts, numChunks*chunkBytes)
	check(err)

	// Every allocation below has to outlive the stream. Nothing here uses a
	// non-default stream, and the context synchronization after each launch is
	// what makes that safe.
	stream := nvcomp.Stream(0)
	inData := deviceAllocT(t, uint(len(uncompressed)))
	inPtrs := deviceAllocT(t, numChunks*nvcomp.SizeofDevicePointer)
	inBytes := deviceAllocT(t, numChunks*nvcomp.SizeofDeviceSize)
	outData := deviceAllocT(t, numChunks*maxOut)
	outPtrs := deviceAllocT(t, numChunks*nvcomp.SizeofDevicePointer)
	outBytes := deviceAllocT(t, numChunks*nvcomp.SizeofDeviceSize)
	statuses := deviceAllocT(t, numChunks*nvcomp.SizeofDeviceStatus)
	temp := deviceAllocT(t, tempBytes)

	check(inData.MemcpyHtoD(unsafe.Pointer(&uncompressed[0]), uint(len(uncompressed))))
	writeUintptrs(t, inPtrs, chunkAddrs(inData, numChunks, chunkBytes))
	writeUints(t, inBytes, repeat(chunkBytes, numChunks))
	writeUintptrs(t, outPtrs, chunkAddrs(outData, numChunks, maxOut))

	check(nvcomp.CompressAsync(
		nvcomp.DevicePointerArray(inPtrs), nvcomp.DeviceSizeArray(inBytes),
		chunkBytes, numChunks,
		nvcomp.DeviceBuffer(temp), tempBytes,
		nvcomp.DevicePointerArray(outPtrs), nvcomp.DeviceSizeArray(outBytes),
		opts, nvcomp.DeviceStatusArray(statuses), stream,
	))
	check(ctx.SynchronizeV2())
	check(nvcomp.CheckStatuses(readStatuses(t, statuses, numChunks)))

	compressedBytes := readUints(t, outBytes, numChunks)
	for i, n := range compressedBytes {
		if n == 0 || n > maxOut {
			t.Fatalf("chunk %d compressed to %d bytes, want 1..%d", i, n, maxOut)
		}
	}

	// Decompress into fresh buffers and compare.
	decompressOpts := nvcomp.DecompressOpts{Backend: nvcomp.DecompressBackendCuda}
	dAlign, err := nvcomp.DecompressGetRequiredAlignments(decompressOpts)
	check(err)
	if chunkBytes%dAlign.Output != 0 {
		t.Fatalf("chunk stride %d does not satisfy the required output alignment %d", chunkBytes, dAlign.Output)
	}
	dTempBytes, err := nvcomp.DecompressGetTempSizeAsync(numChunks, chunkBytes, decompressOpts, numChunks*chunkBytes)
	check(err)

	roundTrip := deviceAllocT(t, uint(len(uncompressed)))
	roundTripPtrs := deviceAllocT(t, numChunks*nvcomp.SizeofDevicePointer)
	bufferBytes := deviceAllocT(t, numChunks*nvcomp.SizeofDeviceSize)
	actualBytes := deviceAllocT(t, numChunks*nvcomp.SizeofDeviceSize)
	sizeQuery := deviceAllocT(t, numChunks*nvcomp.SizeofDeviceSize)
	dTemp := deviceAllocT(t, dTempBytes)
	writeUintptrs(t, roundTripPtrs, chunkAddrs(roundTrip, numChunks, chunkBytes))
	writeUints(t, bufferBytes, repeat(chunkBytes, numChunks))

	// The uncompressed sizes are not recorded anywhere in the compressed data
	// the batched API produces; this call is how a reader that did not keep
	// them recovers them.
	check(nvcomp.GetDecompressSizeAsync(
		nvcomp.DevicePointerArray(outPtrs), nvcomp.DeviceSizeArray(outBytes),
		nvcomp.DeviceSizeArray(sizeQuery), numChunks, stream,
	))
	check(ctx.SynchronizeV2())
	for i, n := range readUints(t, sizeQuery, numChunks) {
		if n != chunkBytes {
			t.Errorf("GetDecompressSizeAsync reported %d bytes for chunk %d, want %d", n, i, chunkBytes)
		}
	}

	check(nvcomp.DecompressAsync(
		nvcomp.DevicePointerArray(outPtrs), nvcomp.DeviceSizeArray(outBytes),
		nvcomp.DeviceSizeArray(bufferBytes), nvcomp.DeviceSizeArray(actualBytes),
		numChunks, nvcomp.DeviceBuffer(dTemp), dTempBytes,
		nvcomp.DevicePointerArray(roundTripPtrs), decompressOpts,
		nvcomp.DeviceStatusArray(statuses), stream,
	))
	check(ctx.SynchronizeV2())
	check(nvcomp.CheckStatuses(readStatuses(t, statuses, numChunks)))

	got := make([]byte, len(uncompressed))
	check(roundTrip.MemcpyDtoH(unsafe.Pointer(&got[0]), uint(len(got))))
	if !bytes.Equal(got, uncompressed) {
		t.Fatal("round trip did not reproduce the input")
	}

	// Negative control. Overwrite the first compressed chunk with bytes that
	// are not an LZ4 stream and require that chunk, and only that chunk, to
	// report a failure. If this passes with every status still success, the
	// statuses are not being written and every check of them above was
	// vacuous.
	garbage := make([]byte, compressedBytes[0])
	for i := range garbage {
		garbage[i] = 0xa5
	}
	check(outData.MemcpyHtoD(unsafe.Pointer(&garbage[0]), uint(len(garbage))))
	err = nvcomp.DecompressAsync(
		nvcomp.DevicePointerArray(outPtrs), nvcomp.DeviceSizeArray(outBytes),
		nvcomp.DeviceSizeArray(bufferBytes), nvcomp.DeviceSizeArray(actualBytes),
		numChunks, nvcomp.DeviceBuffer(dTemp), dTempBytes,
		nvcomp.DevicePointerArray(roundTripPtrs), decompressOpts,
		nvcomp.DeviceStatusArray(statuses), stream,
	)
	check(err)
	check(ctx.SynchronizeV2())
	corrupt := readStatuses(t, statuses, numChunks)
	if corrupt[0] == nvcomp.Success {
		t.Error("decompressing a corrupted chunk reported success; the per-chunk statuses carry no information")
	}
	for i, s := range corrupt[1:] {
		if s != nvcomp.Success {
			t.Errorf("chunk %d reported %v, but only chunk 0 was corrupted", i+1, s)
		}
	}
}

// TestPropertiesMatchHeaders reports whether the installed library is the
// release these bindings were generated from. It needs libnvcomp but no GPU:
// nvcompGetProperties reads compiled-in constants.
//
// A mismatch is not a failure. The bindings are generated from 5.3.0 headers
// and the loader binds by symbol name, so an older or newer libnvcomp.so.5
// with the same symbols will work; this exists so that a surprising result
// elsewhere can be read against the version that produced it.
func TestPropertiesMatchHeaders(t *testing.T) {
	if os.Getenv("NVCOMP_LIBRARY_TEST") != "1" {
		t.Skip("set NVCOMP_LIBRARY_TEST=1 to run against a real libnvcomp (no GPU required)")
	}
	props, err := nvcomp.GetProperties()
	if err != nil {
		t.Fatal(err)
	}
	major, minor, patch := props.SemanticVersion()
	t.Logf("loaded %v from %s", props, nvcomp.Path())
	if major != nvcomp.HeaderVersionMajor || minor != nvcomp.HeaderVersionMinor || patch != nvcomp.HeaderVersionPatch {
		t.Logf("library is %d.%d.%d, bindings were generated from %d.%d.%d",
			major, minor, patch,
			nvcomp.HeaderVersionMajor, nvcomp.HeaderVersionMinor, nvcomp.HeaderVersionPatch)
	}
	for _, sym := range []string{
		"nvcompBatchedLZ4CompressAsync",
		"nvcompBatchedLZ4CompressGetTempSizeAsync",
		"nvcompBatchedLZ4CompressGetTempSizeSync",
		"nvcompBatchedLZ4DecompressAsync",
		"nvcompBatchedLZ4GetDecompressSizeAsync",
	} {
		if !nvcomp.Has(sym) {
			t.Errorf("%s is absent from %s", sym, nvcomp.Path())
		}
	}
	if nvcomp.Has("nvcompBatchedLZ4CompressGetTempSize") {
		t.Errorf("%s exports the pre-5.3 nvcompBatchedLZ4CompressGetTempSize; these bindings target the "+
			"Async and Sync variants and do not bind it", nvcomp.Path())
	}
}

// deviceAllocT allocates device memory for the duration of the test.
func deviceAllocT(t *testing.T, bytes uint) driver.DevicePtr {
	t.Helper()
	p, err := driver.MemAlloc(bytes)
	if err != nil {
		t.Fatalf("allocating %d bytes on the device: %v", bytes, err)
	}
	t.Cleanup(func() { p.MemFree() })
	return p
}

// chunkAddrs returns the device address of each chunk in one allocation.
func chunkAddrs(base driver.DevicePtr, chunks int, stride uint) []uintptr {
	out := make([]uintptr, chunks)
	for i := range out {
		out[i] = uintptr(base) + uintptr(i)*uintptr(stride)
	}
	return out
}

// repeat returns n copies of v.
func repeat(v uint, n int) []uint {
	out := make([]uint, n)
	for i := range out {
		out[i] = v
	}
	return out
}

// writeUintptrs copies an array of device addresses into device memory. This
// is the step that makes a DevicePointerArray what it is: the pointers
// themselves have to be on the device, not just what they point at.
func writeUintptrs(t *testing.T, dst driver.DevicePtr, values []uintptr) {
	t.Helper()
	if err := dst.MemcpyHtoD(unsafe.Pointer(&values[0]), uint(len(values))*nvcomp.SizeofDevicePointer); err != nil {
		t.Fatal(err)
	}
}

// writeUints copies an array of C size_t values into device memory.
func writeUints(t *testing.T, dst driver.DevicePtr, values []uint) {
	t.Helper()
	if err := dst.MemcpyHtoD(unsafe.Pointer(&values[0]), uint(len(values))*nvcomp.SizeofDeviceSize); err != nil {
		t.Fatal(err)
	}
}

// readUints copies an array of C size_t values back from device memory.
func readUints(t *testing.T, src driver.DevicePtr, n int) []uint {
	t.Helper()
	out := make([]uint, n)
	if err := src.MemcpyDtoH(unsafe.Pointer(&out[0]), uint(n)*nvcomp.SizeofDeviceSize); err != nil {
		t.Fatal(err)
	}
	return out
}

// readStatuses copies a per-chunk status array back from device memory. A
// failure to copy is fatal rather than ignored: an unread array is all zeros,
// which is indistinguishable from every chunk having succeeded.
func readStatuses(t *testing.T, src driver.DevicePtr, n int) []nvcomp.Result {
	t.Helper()
	out := make([]nvcomp.Result, n)
	if err := src.MemcpyDtoH(unsafe.Pointer(&out[0]), uint(n)*nvcomp.SizeofDeviceStatus); err != nil {
		t.Fatal(err)
	}
	return out
}
