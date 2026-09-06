//go:build linux

package vpi

import (
	"errors"
	"unsafe"
)

// The generated ImageBuffer is opaque storage sized and aligned like the C
// union VPIImageBuffer. Every accessor below reinterprets that storage, so the
// storage must be at least as large as, and at least as strictly aligned as,
// the largest member. layout_test.go checks both against the pinned headers;
// these assertions catch a regenerated ImageBuffer that no longer fits.
const (
	_ = uint(unsafe.Sizeof(ImageBuffer{}) - unsafe.Sizeof(ImageBufferPitchLinear{}))
	_ = uint(unsafe.Alignof(ImageBuffer{}) - unsafe.Alignof(ImageBufferPitchLinear{}))
)

// ErrBufferTypeMismatch reports an attempt to read a member of an [ImageData]
// union that its discriminant does not select.
//
// Reading the wrong member of VPIImageBuffer is not a detectable mistake at the
// C level: every member starts at the same address, so a wrong read returns
// plausible-looking garbage. The accessors on [ImageData] check BufferType
// first and return this error instead.
type ErrBufferTypeMismatch struct {
	Have ImageBufferType // the discriminant actually stored
	Want string          // the member the caller asked for
}

// Error reports the mismatched image buffer type.
func (e *ErrBufferTypeMismatch) Error() string {
	return "vpi: image buffer is " + e.Have.String() + ", not " + e.Want
}

// PitchLinear returns a pointer to the pitch-linear member of d's buffer union.
//
// The pointer aliases d; it is valid only while d is, and writing through it
// writes d. It is an error unless d.BufferType is
// [ImageBufferHostPitchLinear] or [ImageBufferCudaPitchLinear], the two
// discriminants that select this member.
//
// Whether the plane pointers inside address host memory or device memory is
// decided by the discriminant, not by the member: a
// [ImageBufferCudaPitchLinear] buffer's plane data is a device address and must
// not be dereferenced by Go code. See [ImageData.PlaneBytes].
func (d *ImageData) PitchLinear() (*ImageBufferPitchLinear, error) {
	switch d.BufferType {
	case ImageBufferHostPitchLinear, ImageBufferCudaPitchLinear:
		return (*ImageBufferPitchLinear)(unsafe.Pointer(&d.Buffer)), nil
	}
	return nil, &ErrBufferTypeMismatch{Have: d.BufferType, Want: "pitch-linear"}
}

// SetPitchLinear stores buf in d and sets d's discriminant to bufType.
// bufType must be [ImageBufferHostPitchLinear] or [ImageBufferCudaPitchLinear].
//
// It is the only supported way to build an [ImageData] for
// [ImageCreateWrapper]: setting the discriminant and the member together is
// what keeps them consistent.
func (d *ImageData) SetPitchLinear(bufType ImageBufferType, buf ImageBufferPitchLinear) error {
	switch bufType {
	case ImageBufferHostPitchLinear, ImageBufferCudaPitchLinear:
	default:
		return &ErrBufferTypeMismatch{Have: bufType, Want: "pitch-linear"}
	}
	*d = ImageData{BufferType: bufType}
	*(*ImageBufferPitchLinear)(unsafe.Pointer(&d.Buffer)) = buf
	return nil
}

// CUDAArray returns the cudaArray_t stored in d's buffer union. It is an error
// unless d.BufferType is [ImageBufferCudaArray].
//
// The value is an opaque CUDA handle. This package does not create or destroy
// cudaArray_t objects; one obtained here is owned by whoever allocated it.
func (d *ImageData) CUDAArray() (uintptr, error) {
	if d.BufferType != ImageBufferCudaArray {
		return 0, &ErrBufferTypeMismatch{Have: d.BufferType, Want: "cudaArray_t"}
	}
	return *(*uintptr)(unsafe.Pointer(&d.Buffer)), nil
}

// EGLImage returns the EGLImageKHR stored in d's buffer union. It is an error
// unless d.BufferType is [ImageBufferEglimage].
//
// EGLImage wrapping is documented as Tegra-only. On a non-Tegra platform
// [ImageCreateWrapper] rejects this buffer type.
func (d *ImageData) EGLImage() (uintptr, error) {
	if d.BufferType != ImageBufferEglimage {
		return 0, &ErrBufferTypeMismatch{Have: d.BufferType, Want: "EGLImageKHR"}
	}
	return *(*uintptr)(unsafe.Pointer(&d.Buffer)), nil
}

// NvBufferFD returns the NvBuffer file descriptor stored in d's buffer union.
// It is an error unless d.BufferType is [ImageBufferNvbuffer].
//
// The C member is an int, so only the low 32 bits of the union are read.
// NvBuffer wrapping is documented as Tegra-only.
func (d *ImageData) NvBufferFD() (int32, error) {
	if d.BufferType != ImageBufferNvbuffer {
		return 0, &ErrBufferTypeMismatch{Have: d.BufferType, Want: "NvBuffer fd"}
	}
	return *(*int32)(unsafe.Pointer(&d.Buffer)), nil
}

// ErrNotHostMemory reports an attempt to address image plane bytes that do not
// live in host memory.
var ErrNotHostMemory = errors.New("vpi: image plane is not host memory")

// PlaneBytes returns a slice aliasing plane i of a host pitch-linear buffer.
//
// The slice covers Height*PitchBytes bytes starting at the plane's data
// pointer. It aliases memory this package does not own and does not keep alive:
//
//   - For a buffer filled by [ImageLockData], the memory belongs to the library
//     and is valid only until the matching [ImageUnlock]. Using the slice after
//     unlocking is a use-after-free.
//   - For a buffer the caller built, the memory belongs to the caller, and the
//     rules in [ImageCreateWrapper] apply.
//
// PlaneBytes refuses any buffer type other than [ImageBufferHostPitchLinear],
// because dereferencing a CUDA device address from Go crashes the process at
// best.
func (d *ImageData) PlaneBytes(i int) ([]byte, error) {
	if d.BufferType != ImageBufferHostPitchLinear {
		return nil, ErrNotHostMemory
	}
	buf, err := d.PitchLinear()
	if err != nil {
		return nil, err
	}
	if i < 0 || i >= int(buf.NumPlanes) || i >= len(buf.Planes) {
		return nil, errors.New("vpi: plane index out of range")
	}
	p := &buf.Planes[i]
	if p.Data == 0 {
		return nil, errors.New("vpi: image plane has no data pointer")
	}
	if p.Height <= 0 || p.PitchBytes <= 0 {
		return nil, errors.New("vpi: image plane has no extent")
	}
	n := int(p.Height) * int(p.PitchBytes)
	// Read the C pointer stored in the union through a pointer-typed view.
	// The union's integer storage does not keep caller-owned Go memory alive;
	// the caller must retain it for the lifetime of the returned slice.
	data := *(*unsafe.Pointer)(unsafe.Pointer(&p.Data))
	return unsafe.Slice((*byte)(data), n), nil
}

// InitImageWrapperParams returns [ImageWrapperParams] filled with the library's
// defaults.
//
// Do not build an [ImageWrapperParams] with a struct literal. The C API
// requires the defaults to come from vpiInitImageWrapperParams so that fields
// added in a later VPI release get a usable value; a zero value is not that.
// The zero value of ColorSpec is not [ColorSpecDefault].
//
// Wraps vpiInitImageWrapperParams.
func InitImageWrapperParams() (ImageWrapperParams, error) {
	var params ImageWrapperParams
	if err := checkLibSym("vpiInitImageWrapperParams"); err != nil {
		return params, err
	}
	return params, toError(fnInitImageWrapperParams(&params))
}

// ImageCreateWrapper creates an image that wraps memory the caller owns.
//
// data describes the memory and must have been filled through
// [ImageData.SetPitchLinear] (or, on Tegra, carry an EGLImage or NvBuffer
// discriminant). params may be nil to accept the library defaults; otherwise it
// must come from [InitImageWrapperParams]. flags combines [Backend] bits with
// the object flags in types.go; VPI adds [BackendCPU] for a host pitch-linear
// buffer and [BackendCUDA] for a CUDA pitch-linear one.
//
// # Ownership
//
// VPI does not copy the memory and does not free it. The caller owns it for the
// entire lifetime of the returned [Image], not merely for one submission, and
// must keep it allocated and at a fixed address until all of the following have
// happened:
//
//   - every stream that was given this image has returned from [StreamSync];
//   - every [ImageLock] or [ImageLockData] has been matched by [ImageUnlock];
//   - [ImageDestroy] has been called on the returned image.
//
// Destroying the wrapper does not free the memory; that remains the caller's
// job, after the wrapper is gone.
//
// # Go memory is not eligible
//
// data's plane pointers are plain addresses. The Go runtime may move a Go heap
// object and does not know the library retains its address, so a Go allocation
// does not satisfy the contract above and this function cannot make it satisfy
// it. Wrap memory whose address is stable and whose lifetime the caller
// controls: an mmap region, memory from a C allocator, or a CUDA device
// allocation. A uintptr does not keep anything alive.
//
// Wraps vpiImageCreateWrapper.
func ImageCreateWrapper(data *ImageData, params *ImageWrapperParams, flags uint64) (Image, error) {
	if data == nil {
		return 0, errors.New("vpi: ImageCreateWrapper: data is nil")
	}
	if err := checkLibSym("vpiImageCreateWrapper"); err != nil {
		return 0, err
	}
	var img Image
	if err := toError(fnImageCreateWrapper(data, params, flags, &img)); err != nil {
		return 0, err
	}
	return img, nil
}

// ImageSetWrapper redefines the memory an existing wrapper points at.
//
// img must have come from [ImageCreateWrapper], must not be locked, and the new
// data must have the same dimensions, format, and buffer type as the memory it
// replaces. The ownership rules of [ImageCreateWrapper] apply to the new memory
// from this call onward, and to the old memory until this call returns.
//
// Wraps vpiImageSetWrapper.
func ImageSetWrapper(img Image, data *ImageData) error {
	if data == nil {
		return errors.New("vpi: ImageSetWrapper: data is nil")
	}
	if err := checkLibSym("vpiImageSetWrapper"); err != nil {
		return err
	}
	return toError(fnImageSetWrapper(img, data))
}

// ImageLockData locks img and returns a description of its contents.
//
// bufType selects how the contents are to be addressed and must be
// [ImageBufferHostPitchLinear] or [ImageBufferCudaPitchLinear]; it also decides
// which member of the returned [ImageData] union is valid, so read the result
// through [ImageData.PitchLinear] or [ImageData.PlaneBytes] rather than casting.
// The image must have the matching backend enabled: CPU for a host buffer, CUDA
// for a CUDA one.
//
// # Ownership
//
// The pointers in the returned [ImageData] are owned by the library and are
// valid only until the matching [ImageUnlock]. Locks nest: each successful lock
// must be matched by exactly one unlock, and the memory is released to VPI when
// the count reaches zero. The returned [ImageData] is a value copy, so it
// outlives the lock as a struct while its pointers do not.
//
// Locking is not synchronized with the stream. VPI does not wait for queued
// work: the caller must have called [StreamSync] on every stream that was given
// this image before locking, or the contents are indeterminate and VPI may
// report [ErrBufferLocked] asynchronously on later submissions.
//
// Wraps vpiImageLockData.
func ImageLockData(img Image, mode LockMode, bufType ImageBufferType) (ImageData, error) {
	var data ImageData
	switch bufType {
	case ImageBufferHostPitchLinear, ImageBufferCudaPitchLinear:
	default:
		return data, &ErrBufferTypeMismatch{Have: bufType, Want: "a lockable buffer type"}
	}
	if err := checkLibSym("vpiImageLockData"); err != nil {
		return data, err
	}
	if err := toError(fnImageLockData(img, mode, bufType, &data)); err != nil {
		return ImageData{}, err
	}
	return data, nil
}

// LockData locks img and returns a description of its contents.
//
// It is the method form of [ImageLockData]; see that function for the rules.
func (i Image) LockData(mode LockMode, bufType ImageBufferType) (ImageData, error) {
	return ImageLockData(i, mode, bufType)
}

// SetWrapper redefines the memory an existing wrapper points at.
//
// It is the method form of [ImageSetWrapper]; see that function for the
// ownership rules that apply to the new memory.
func (i Image) SetWrapper(data *ImageData) error {
	return ImageSetWrapper(i, data)
}
