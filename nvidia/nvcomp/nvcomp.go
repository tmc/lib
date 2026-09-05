package nvcomp

import (
	"fmt"
	"unsafe"
)

// Element sizes, in bytes, of the arrays the batched API reads and writes in
// device memory. A caller sizing one of those allocations needs the size of
// the C element, which is not the size of any Go value it holds: the arrays
// live on the device and Go never indexes them.
//
// They are the LP64 sizes the library ships for. [SizeofDeviceStatus] is the
// size of a C enum, which is 4 on every ABI nvCOMP is built for; it is not
// derived from [Result], whose Go size happens to match.
const (
	// SizeofDevicePointer is the size of one element of a [DevicePointerArray].
	SizeofDevicePointer = 8
	// SizeofDeviceSize is the size of one element of a [DeviceSizeArray],
	// which is a C size_t.
	SizeofDeviceSize = 8
	// SizeofDeviceStatus is the size of one element of a [DeviceStatusArray],
	// which is a C nvcompStatus_t.
	SizeofDeviceStatus = 4
)

// SemanticVersion splits the version [GetProperties] reports into its parts.
//
// nvCOMP encodes it as major*1000 + minor*100 + patch, which is what
// nvcomp/version.h's NVCOMP_*_FROM_SEMVER macros decode. Compare the result
// against [HeaderVersionMajor] and its siblings to see whether the loaded
// library matches the headers these bindings were generated from.
func (p Properties) SemanticVersion() (major, minor, patch int) {
	v := int(p.Version)
	return v / 1000, (v % 1000) / 100, (v % 1000) % 100
}

// String describes the library version and the CUDA Runtime it was built
// against.
func (p Properties) String() string {
	major, minor, patch := p.SemanticVersion()
	return fmt.Sprintf("nvcomp %d.%d.%d (CUDA Runtime %d)", major, minor, patch, p.CudartVersion)
}

// ChunkError reports that a chunk in a batch failed.
//
// The batched API reports a per-chunk outcome only through the status array it
// writes in device memory. A nil error from an asynchronous call means the
// launch was accepted, so a batch can fail with every host-side call having
// returned nil. This is the error [CheckStatuses] returns for such a batch.
type ChunkError struct {
	// Chunk is the index of the first chunk that failed.
	Chunk int
	// Result is that chunk's status.
	Result Result
	// Failed is how many chunks in the batch failed.
	Failed int
	// Chunks is how many chunks were inspected.
	Chunks int
}

// Error describes how many chunks failed and what the first failure was.
func (e *ChunkError) Error() string {
	return fmt.Sprintf("nvcomp: %d of %d chunks failed; chunk %d: %v", e.Failed, e.Chunks, e.Chunk, e.Result)
}

// Unwrap returns the status of the first failing chunk, so that a caller can
// test for a particular one with errors.Is.
func (e *ChunkError) Unwrap() error {
	return e.Result
}

// CheckStatuses reports the first failing chunk in a batch's per-chunk status
// array, or nil if every status is Success. It does not verify output bytes
// or capacities and cannot detect an incorrect success status from the library.
//
// Pass the statuses after copying them back from the [DeviceStatusArray] the
// operation wrote and after the stream has completed. Both are the caller's
// responsibility and neither can be checked here: an array read before the
// stream completes, or one whose copy back from the device silently failed,
// is all zeros, and zero is [Success]. Checking statuses that were never
// filled in is indistinguishable from a batch that worked.
//
// An empty slice returns nil, since a batch of no chunks has nothing to fail.
// Passing zero for the status array in the call that produced the batch
// disables per-chunk reporting entirely; there is then nothing to check and no
// way to learn that a chunk failed.
func CheckStatuses(statuses []Result) error {
	first := -1
	failed := 0
	for i, s := range statuses {
		if s == Success {
			continue
		}
		if first < 0 {
			first = i
		}
		failed++
	}
	if first < 0 {
		return nil
	}
	return &ChunkError{Chunk: first, Result: statuses[first], Failed: failed, Chunks: len(statuses)}
}

// Compile-time checks that the element sizes above match the Go types the
// package would use for the same values. They are separate facts -- the
// constants describe C memory on the device -- but on every platform these
// bindings can reach a CUDA library they agree, and a platform where they do
// not is one where the sizes are wrong rather than one to build on.
const (
	_ = uint(SizeofDevicePointer - unsafe.Sizeof(uintptr(0)))
	_ = uint(unsafe.Sizeof(uintptr(0)) - SizeofDevicePointer)
	_ = uint(SizeofDeviceSize - unsafe.Sizeof(uint(0)))
	_ = uint(unsafe.Sizeof(uint(0)) - SizeofDeviceSize)
	_ = uint(SizeofDeviceStatus - unsafe.Sizeof(Result(0)))
	_ = uint(unsafe.Sizeof(Result(0)) - SizeofDeviceStatus)
)
