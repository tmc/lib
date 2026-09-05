package nvjpeg2k_test

import (
	"errors"
	"fmt"
	"log"
	"os"

	"github.com/tmc/lib/nvidia/nvjpeg2k"
)

// Example decodes one JPEG 2000 file into caller-allocated device memory.
//
// It is written out in full because the order matters: the per-component
// metadata is read before anything is allocated, and the allocation sizes come
// from that metadata rather than from the image dimensions.
//
// cudaMalloc and cudaFree stand in for whichever CUDA binding the caller uses;
// this package never allocates device memory.
func Example() {
	data, err := os.ReadFile("image.jp2")
	if err != nil {
		log.Fatal(err)
	}

	handle, err := nvjpeg2k.NewHandle()
	if err != nil {
		log.Fatal(err)
	}
	defer handle.Destroy()

	state, err := nvjpeg2k.NewDecodeState(handle)
	if err != nil {
		log.Fatal(err)
	}
	defer state.Destroy()

	bitstream, err := nvjpeg2k.NewBitstream()
	if err != nil {
		log.Fatal(err)
	}
	defer bitstream.Destroy()

	if err := handle.Parse(data, bitstream); err != nil {
		log.Fatal(err)
	}

	// The metadata step. Every allocation below is sized from what this
	// returns, never from Layout.Info: a subsampled chroma plane is smaller
	// than the image.
	layout, err := nvjpeg2k.LayoutOf(bitstream, nvjpeg2k.Uint8, nvjpeg2k.FormatPlanar)
	if err != nil {
		log.Fatal(err)
	}

	var buffers []nvjpeg2k.DeviceBuffer
	for _, req := range layout.Buffers() {
		ptr, err := cudaMalloc(req.MinSizeInBytes)
		if err != nil {
			log.Fatal(err)
		}
		defer cudaFree(ptr)
		buffers = append(buffers, nvjpeg2k.DeviceBuffer{
			Ptr:          ptr,
			PitchInBytes: req.MinPitchInBytes,
			SizeInBytes:  req.MinSizeInBytes,
		})
	}

	destination, err := layout.Bind(buffers...)
	if err != nil {
		log.Fatal(err)
	}

	// Asynchronous: the buffers must stay allocated until the stream has
	// completed, and the decode state must not be reused before then.
	if err := handle.DecodeInto(state, bitstream, nil, destination, 0); err != nil {
		log.Fatal(err)
	}
	if err := cudaStreamSynchronize(0); err != nil {
		log.Fatal(err)
	}
}

// ExampleLayoutOf_subsampled shows why the destination sizes come from the
// per-component metadata. For a 4:2:0 image the three planes have three
// different sizes, and the image dimensions describe only the first.
func ExampleLayoutOf_subsampled() {
	// Values a 64x32 4:2:0 8-bit image would produce.
	requirements := []nvjpeg2k.BufferRequirement{
		{MinPitchInBytes: 64, Rows: 32, MinSizeInBytes: 2048},
		{MinPitchInBytes: 32, Rows: 16, MinSizeInBytes: 512},
		{MinPitchInBytes: 32, Rows: 16, MinSizeInBytes: 512},
	}
	for i, req := range requirements {
		fmt.Printf("component %d: %d bytes\n", i, req.MinSizeInBytes)
	}
	// Output:
	// component 0: 2048 bytes
	// component 1: 512 bytes
	// component 2: 512 bytes
}

func ExampleImageType_BytesPerSample() {
	for _, t := range []nvjpeg2k.ImageType{nvjpeg2k.Uint8, nvjpeg2k.Uint16, nvjpeg2k.FP32} {
		fmt.Println(t, t.BytesPerSample())
	}
	// Output:
	// Uint8 1
	// Uint16 2
	// FP32 4
}

// ExampleHas shows probing before calling. The library is opened on the first
// call, not at import, so this is also how to find out whether it is installed
// without provoking an error.
func ExampleHas() {
	if !nvjpeg2k.Has("nvjpeg2kStreamGetComponentNLTInfo") {
		fmt.Println("no NLT metadata: floating-point destinations are unavailable")
	}
}

// ExampleLibraryError shows the shape of a load failure. Nothing in this
// package panics when the library is missing.
func ExampleLibraryError() {
	_, err := nvjpeg2k.NewHandle()
	var libErr *nvjpeg2k.LibraryError
	if errors.As(err, &libErr) {
		fmt.Println("libnvjpeg2k not loadable; names tried:")
		for _, attempt := range libErr.Attempts {
			fmt.Println(" ", attempt.Name)
		}
	}
}

func cudaMalloc(size int) (uintptr, error)  { panic("stand-in for a CUDA allocation") }
func cudaFree(ptr uintptr) error            { panic("stand-in for a CUDA free") }
func cudaStreamSynchronize(s uintptr) error { panic("stand-in for cudaStreamSynchronize") }
