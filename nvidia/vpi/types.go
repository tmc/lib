//go:build linux

package vpi

import "fmt"

// ImageFormat identifies an image's memory layout, color model, and channel
// packing. It is the C type VPIImageFormat, a 64-bit word whose bit fields the
// header builds with preprocessor macros rather than an enum, so the constants
// below are the expanded values from the pinned VPI 3.2.4 headers.
// TestImageFormatConstants in the generator checks them against those headers.
//
// The zero value is [ImageFormatInvalid].
type ImageFormat uint64

// Image formats. This is a subset of the formats VPI 3.2.4 defines: the
// non-color and pitch-linear color formats that [SubmitRescale] documents for
// the CPU and CUDA backends, plus the block-linear-free YUV formats. Formats
// with a _BL or _BL16 suffix are omitted because they are VIC-only, and this
// package cannot be exercised against a VIC.
//
// A format constant says nothing about whether a backend accepts it. Consult
// the VPI reference for the per-backend format table of each operation.
const (
	ImageFormatInvalid ImageFormat = 0x0000000000000000

	ImageFormatU8   ImageFormat = 0x20000027ffff1001 // single unsigned 8-bit channel
	ImageFormatU16  ImageFormat = 0x20000107ffff1001 // single unsigned 16-bit channel
	ImageFormatU32  ImageFormat = 0x20000307ffff1001 // single unsigned 32-bit channel
	ImageFormatS8   ImageFormat = 0x40000027ffff1001 // single signed 8-bit channel
	ImageFormatS16  ImageFormat = 0x40000107ffff1001 // single signed 16-bit channel
	ImageFormatS32  ImageFormat = 0x40000307ffff1001 // single signed 32-bit channel
	ImageFormatF32  ImageFormat = 0x60000307ffff1001 // single 32-bit float channel
	ImageFormatF64  ImageFormat = 0x60000507ffff1001 // single 64-bit float channel
	ImageFormat2S16 ImageFormat = 0x40000347ffff1011 // two interleaved signed 16-bit channels
	ImageFormat2F32 ImageFormat = 0x60000547ffff1011 // two interleaved 32-bit float channels

	ImageFormatY8    ImageFormat = 0x200000240a201001 // 8-bit luma, limited range
	ImageFormatY8ER  ImageFormat = 0x200000200a201001 // 8-bit luma, full range
	ImageFormatY16   ImageFormat = 0x200001040a201001 // 16-bit luma, limited range
	ImageFormatY16ER ImageFormat = 0x200001000a201001 // 16-bit luma, full range

	ImageFormatRGB8   ImageFormat = 0x20000283c0211ad1 // interleaved 8-bit RGB
	ImageFormatBGR8   ImageFormat = 0x20000283c0211a53 // interleaved 8-bit BGR
	ImageFormatRGBA8  ImageFormat = 0x200003c3c02118d1 // interleaved 8-bit RGBA
	ImageFormatBGRA8  ImageFormat = 0x200003c3c0211853 // interleaved 8-bit BGRA
	ImageFormatRGB8p  ImageFormat = 0x20204023c0211ad1 // planar 8-bit RGB
	ImageFormatBGR8p  ImageFormat = 0x20204023c0211a53 // planar 8-bit BGR
	ImageFormatRGBA8p ImageFormat = 0x3c204023c02118d1 // planar 8-bit RGBA
	ImageFormatBGRA8p ImageFormat = 0x3c204023c0211853 // planar 8-bit BGRA

	ImageFormatNV12   ImageFormat = 0x200180240a2a10d1 // 4:2:0 semi-planar YUV, limited range
	ImageFormatNV12ER ImageFormat = 0x200180200a2a10d1 // 4:2:0 semi-planar YUV, full range
	ImageFormatNV24   ImageFormat = 0x200180240a2010d1 // 4:4:4 semi-planar YUV, limited range
	ImageFormatNV24ER ImageFormat = 0x200180200a2010d1 // 4:4:4 semi-planar YUV, full range
	ImageFormatUYVY   ImageFormat = 0x200001f40a221ad1 // 4:2:2 interleaved YUV, limited range
	ImageFormatUYVYER ImageFormat = 0x200001f00a221ad1 // 4:2:2 interleaved YUV, full range
	ImageFormatYUYV   ImageFormat = 0x200001ec0a221ad1 // 4:2:2 interleaved YUV, limited range
	ImageFormatYUYVER ImageFormat = 0x200001e80a221ad1 // 4:2:2 interleaved YUV, full range
)

var imageFormatNames = map[ImageFormat]string{
	ImageFormatInvalid: "Invalid",
	ImageFormatU8:      "U8", ImageFormatU16: "U16", ImageFormatU32: "U32",
	ImageFormatS8: "S8", ImageFormatS16: "S16", ImageFormatS32: "S32",
	ImageFormatF32: "F32", ImageFormatF64: "F64",
	ImageFormat2S16: "2S16", ImageFormat2F32: "2F32",
	ImageFormatY8: "Y8", ImageFormatY8ER: "Y8_ER",
	ImageFormatY16: "Y16", ImageFormatY16ER: "Y16_ER",
	ImageFormatRGB8: "RGB8", ImageFormatBGR8: "BGR8",
	ImageFormatRGBA8: "RGBA8", ImageFormatBGRA8: "BGRA8",
	ImageFormatRGB8p: "RGB8p", ImageFormatBGR8p: "BGR8p",
	ImageFormatRGBA8p: "RGBA8p", ImageFormatBGRA8p: "BGRA8p",
	ImageFormatNV12: "NV12", ImageFormatNV12ER: "NV12_ER",
	ImageFormatNV24: "NV24", ImageFormatNV24ER: "NV24_ER",
	ImageFormatUYVY: "UYVY", ImageFormatUYVYER: "UYVY_ER",
	ImageFormatYUYV: "YUYV", ImageFormatYUYVER: "YUYV_ER",
}

// String returns the VPI spelling of a known format, or its hexadecimal value.
// A format this package does not name is still a valid VPI format; the numeric
// form is not an error.
func (f ImageFormat) String() string {
	if name, ok := imageFormatNames[f]; ok {
		return name
	}
	return fmt.Sprintf("ImageFormat(%#016x)", uint64(f))
}

// PixelType identifies the type of a single pixel within one image plane. It is
// the C type VPIPixelType.
//
// Only [PixelTypeInvalid] is named here, because it is the only value this
// package's API needs: a plane whose PixelType is [PixelTypeInvalid] takes its
// pixel type from the enclosing buffer's [ImageFormat], which is what a caller
// filling an [ImagePlanePitchLinear] should do.
type PixelType uint64

// PixelTypeInvalid tells VPI to infer the plane's pixel type from the buffer
// format. It is VPI_PIXEL_TYPE_INVALID.
const PixelTypeInvalid PixelType = 0

// ColorSpec describes color primaries, encoding, transfer function, range, and
// chroma siting. It is the C type VPIColorSpec, an enumeration whose values the
// header composes from bit fields.
//
// Only [ColorSpecDefault] is named: it is the value [ImageWrapperParams] needs
// to mean "do not override the color spec implied by the image format", and it
// is the only one this package can state without also binding the color-spec
// query functions, which are outside the generated surface.
type ColorSpec int32

// ColorSpecDefault leaves the color spec implied by the image format in place.
// It is VPI_COLOR_SPEC_DEFAULT from the pinned headers.
const ColorSpecDefault ColorSpec = 15367

// Object creation flags. These combine with [Backend] flags in the flags word
// passed to [ImageCreate], [StreamCreate], [ContextCreate], and
// [ImageCreateWrapper].
const (
	// ExclusiveStreamAccess declares that only one stream at a time will
	// access the object, letting VPI allocate fewer resources. Using the
	// object from two streams concurrently after setting this is undefined.
	ExclusiveStreamAccess uint64 = 1 << 16

	// RestrictMemUsage avoids an internal pitch-linear staging buffer for
	// block-linear formats. Image locking then fails for those formats.
	RestrictMemUsage uint64 = 1 << 17

	// RequireBackends turns the requested backends from a preference into a
	// requirement: creation fails rather than silently dropping a backend
	// that is unavailable or incompatible. Pass it whenever the caller
	// intends to run on a specific backend, so the failure surfaces at
	// creation rather than at submission.
	RequireBackends uint64 = 1 << 18

	// StreamGreedy flushes queued operations to the backend at every
	// submission. It is only valid for [StreamCreate].
	StreamGreedy uint64 = 1 << 63
)
