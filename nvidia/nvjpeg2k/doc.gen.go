// Package nvjpeg2k provides Go bindings for NVIDIA's nvJPEG2000 decoder, libnvjpeg2k.
//
// This package loads libnvjpeg2k on its own. It shares no handle, no loader,
// and no symbol table with nvJPEG: they are separate libraries.
//
// # Decode order
//
// A decode is four steps, and the metadata step is not optional. Parse the
// bitstream into a [Bitstream], read [Bitstream.ImageInfo] and, for every
// component, [Bitstream.ComponentInfo], allocate one device buffer per
// component from those per-component dimensions and bit depths, then decode
// into them. Component width, height, precision and sign vary per component:
// a subsampled chroma plane is smaller than the luma plane, and the image
// dimensions in [ImageInfo] do not describe it. Sizing every plane from the
// image dimensions overruns nothing detectable on the host and corrupts the
// decode.
//
// [LayoutOf] performs the metadata step and returns a [Layout]; [Layout.Bind]
// turns caller-owned device buffers into a [Destination] only if they match
// that layout, and [Handle.DecodeInto] accepts nothing else. The raw
// [Handle.Decode], [Handle.DecodeImage] and [Handle.DecodeTile] wrappers take
// an [Image] you filled in yourself and check nothing.
//
// A tile decode has its own extents. Use [LayoutOfTile] and
// [Handle.DecodeTileInto] for one; a whole-image layout is refused there,
// and a tile layout is refused by [Handle.DecodeInto].
//
// # Ownership
//
// [Handle], [DecodeState], [Bitstream], and [DecodeParams] are library-owned
// resources reached through opaque handles. Each has a Destroy method and
// none is finalized automatically; a dropped handle leaks. Destroy the
// decode state and the bitstream before the library handle that created
// them.
//
// Decode output buffers are caller-owned device memory. This package does
// not allocate them, does not free them, and cannot check their size. The
// pixel-data and pitch arrays in [Image] are host arrays that the library
// reads during the call; the device addresses they hold must stay valid
// until the CUDA stream has completed, because the decode functions are
// asynchronous with respect to the host.
//
// The input buffer passed to [Handle.Parse] is read during that call and
// is not retained; [Handle.Parse] keeps it alive across the call. The
// parsed metadata lives in the [Bitstream] until it is reparsed or
// destroyed, and a [Layout] read from a [Bitstream] describes the
// bitstream as it was parsed at that moment.
//
// The header documents a [Handle] as thread safe and usable from several
// host threads at once. It documents a [DecodeState] as reusable only if
// the caller makes a stream or device synchronize CUDA call between two
// decodes that share it. This package does not enforce either rule and
// has not verified them.
//
// # Availability
//
// Importing this package does not open the library. The first call loads it,
// or returns *LibraryError describing every name tried. A symbol missing
// from an older libnvjpeg2k returns *SymbolError naming it. Neither panics.
// Use [Has] or the generated Has* predicates to probe before calling.
//
// Encode is not bound. See the [nvJPEG2000 documentation] for the encode API.
//
// # Library lifecycle
//
//   - [NewHandleWithAllocators]: creates a library handle with caller-supplied device and pinned allocators.
//   - [NewHandle]: creates a library handle with the default backend and allocators.
//   - [NewHandleWithAllocatorsV2]: creates a library handle with stream-ordered caller-supplied allocators.
//   - [Handle.Destroy]: releases a library handle and the resources it owns.
//   - [Handle.SetDeviceMemoryPadding]: rounds later device allocations up to a multiple of padding bytes.
//   - [Handle.SetPinnedMemoryPadding]: rounds later pinned host allocations up to a multiple of padding bytes.
//
// # Decode
//
//   - [Handle.Decode]: decodes a whole image into caller-allocated device buffers.
//   - [Handle.DecodeImage]: decodes a whole image into caller-allocated device buffers, under decode parameters.
//   - [Handle.DecodeTile]: decodes one tile into caller-allocated device buffers.
//
// # Decode parameters
//
//   - [NewDecodeParams]: creates a decode parameters handle holding output options.
//   - [DecodeParams.Destroy]: releases a decode parameters handle.
//   - [DecodeParams.SetDecodeArea]: restricts decoding to a rectangle within the tile of interest.
//   - [DecodeParams.SetOutputFormat]: selects planar or interleaved output.
//   - [DecodeParams.SetRGBOutput]: requests RGB output for chroma-subsampled sYCC streams.
//
// # Decode state
//
//   - [NewDecodeState]: creates a decode state holding the intermediate buffers a decode needs.
//   - [DecodeState.Destroy]: releases a decode state.
//
// # Library version
//
//   - [GetCudartProperty]: returns one component of the CUDA Toolkit version the library was built against.
//   - [Handle.GetDeviceMemoryPadding]: returns the current device allocation padding in bytes.
//   - [Handle.GetPinnedMemoryPadding]: returns the current pinned host allocation padding in bytes.
//   - [GetProperty]: returns one component of the loaded library's version.
//
// # Bitstream parsing and metadata
//
//   - [NewBitstream]: creates an empty bitstream handle to parse into.
//   - [Bitstream.Destroy]: releases a bitstream handle and the metadata parsed into it.
//   - [Bitstream.ColorSpace]: returns the color space recorded in a JP2 container.
//   - [Bitstream.ComponentNLTInfo]: returns one component's nonlinear-transformation parameters.
//   - [Bitstream.ComponentInfo]: returns one component's dimensions, bit depth, and signedness.
//   - [Bitstream.ImageInfo]: returns the image dimensions, tile grid, and component count.
//   - [Bitstream.ResolutionComponentDim]: returns one tile component's width and height at a resolution level.
//   - [Bitstream.ResolutionsInTile]: returns the number of resolution levels present in a tile.
//   - [Bitstream.TileComponentDim]: returns one tile component's width and height.
//   - [Handle.ParseFile]: parses a JPEG 2000 file from disk.
//
// [nvJPEG2000 documentation]: https://docs.nvidia.com/cuda/nvjpeg2000/
package nvjpeg2k

// Code generated by gen-nvjpeg2k from nvjpeg2k.h. DO NOT EDIT.
