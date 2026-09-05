//go:build linux

package vpi

import (
	"errors"
	"runtime"
	"strings"
	"testing"
	"unsafe"
)

// The generated layout_test.go checks the struct sizes, alignments and field
// offsets against the pinned headers. The tests here check the things that are
// not a struct layout: the widths of the values VPI passes by value, the union
// discriminant contract, and the constants this package states in Go rather
// than deriving from the headers.
//
// None of these need a GPU or the library. They fail on a bad regeneration and
// on a bad hand edit, which is all a machine without VPI installed can check.

// TestByValueWidths checks the sizes of the types that cross the FFI boundary
// as arguments. purego passes a Go value in the register slot its width names,
// so a handle that is not 8 bytes or an enum that is not 4 puts every following
// argument in the wrong place.
//
// The C widths are asserted against the pinned headers by
// TestConstantsMatchHeaders in cmd/gen-vpi.
func TestByValueWidths(t *testing.T) {
	tests := []struct {
		name string
		got  uintptr
		want uintptr
	}{
		{"Context", unsafe.Sizeof(Context(0)), 8},
		{"Stream", unsafe.Sizeof(Stream(0)), 8},
		{"Image", unsafe.Sizeof(Image(0)), 8},
		{"Event", unsafe.Sizeof(Event(0)), 8},
		{"Payload", unsafe.Sizeof(Payload(0)), 8},
		{"Array", unsafe.Sizeof(Array(0)), 8},
		{"Pyramid", unsafe.Sizeof(Pyramid(0)), 8},
		{"ImageFormat", unsafe.Sizeof(ImageFormat(0)), 8},
		{"PixelType", unsafe.Sizeof(PixelType(0)), 8},
		{"ColorSpec", unsafe.Sizeof(ColorSpec(0)), 4},
		{"Result", unsafe.Sizeof(Result(0)), 4},
		{"LockMode", unsafe.Sizeof(LockMode(0)), 4},
		{"ImageBufferType", unsafe.Sizeof(ImageBufferType(0)), 4},
		{"InterpolationType", unsafe.Sizeof(InterpolationType(0)), 4},
		{"BorderExtension", unsafe.Sizeof(BorderExtension(0)), 4},
	}
	for _, tt := range tests {
		if tt.got != tt.want {
			t.Errorf("sizeof(%s) = %d, want %d", tt.name, tt.got, tt.want)
		}
	}
	// Backend is 8 bytes although the C enum VPIBackend is 4. That is correct
	// only because no bound function takes a VPIBackend by value: every place
	// a backend crosses the boundary it is one field of a uint64_t flags word,
	// including vpiSubmitRescale's backend argument. TestGeneration in
	// cmd/gen-vpi pins the signature that this depends on.
	if got := unsafe.Sizeof(Backend(0)); got != 8 {
		t.Errorf("sizeof(Backend) = %d, want 8, the width of a VPI flags word", got)
	}
}

// TestImageBufferUnionFits checks that the opaque union storage is big enough
// and strictly enough aligned for every member the accessors reinterpret it as.
//
// This is the hazard the package is most exposed to: the members of the C union
// VPIImageBuffer all start at the same address, so an undersized or
// underaligned Go storage type produces reads that look plausible and are
// wrong, or writes that scribble past the struct.
func TestImageBufferUnionFits(t *testing.T) {
	const (
		bufSize  = unsafe.Sizeof(ImageBuffer{})
		bufAlign = unsafe.Alignof(ImageBuffer{})
	)
	members := []struct {
		name  string
		size  uintptr
		align uintptr
	}{
		{"VPIImageBufferPitchLinear", unsafe.Sizeof(ImageBufferPitchLinear{}), unsafe.Alignof(ImageBufferPitchLinear{})},
		{"cudaArray_t", unsafe.Sizeof(uintptr(0)), unsafe.Alignof(uintptr(0))},
		{"EGLImageKHR", unsafe.Sizeof(uintptr(0)), unsafe.Alignof(uintptr(0))},
		{"NvBuffer fd", unsafe.Sizeof(int32(0)), unsafe.Alignof(int32(0))},
	}
	for _, m := range members {
		if m.size > bufSize {
			t.Errorf("union storage is %d bytes, smaller than member %s at %d", bufSize, m.name, m.size)
		}
		if m.align > bufAlign {
			t.Errorf("union storage is aligned %d, less than member %s at %d", bufAlign, m.name, m.align)
		}
	}
	// The discriminant must precede the union, and the union must be the rest
	// of the struct. A gap here means the Go struct is not the C struct.
	if off := unsafe.Offsetof(ImageData{}.Buffer); off+bufSize != unsafe.Sizeof(ImageData{}) {
		t.Errorf("ImageData is %d bytes with the union at %d+%d; the union must end the struct",
			unsafe.Sizeof(ImageData{}), off, bufSize)
	}
	if unsafe.Offsetof(ImageData{}.BufferType) != 0 {
		t.Error("the discriminant must be the first field of ImageData")
	}
}

// TestPlaneCount checks the plane array against VPI_MAX_PLANE_COUNT. The
// generator does not emit that constant, so the array length is the only place
// it appears; cmd/gen-vpi asserts the header value is 6.
func TestPlaneCount(t *testing.T) {
	var buf ImageBufferPitchLinear
	if got := len(buf.Planes); got != 6 {
		t.Errorf("ImageBufferPitchLinear.Planes has %d entries, want VPI_MAX_PLANE_COUNT = 6", got)
	}
}

// TestSetPitchLinearRoundTrip checks that a value written into the union
// storage reads back through the union accessor, byte for byte. It is what
// establishes that the reinterpretation in PitchLinear and SetPitchLinear
// addresses the same bytes.
func TestSetPitchLinearRoundTrip(t *testing.T) {
	want := ImageBufferPitchLinear{
		Format:    ImageFormatU8,
		NumPlanes: 1,
	}
	want.Planes[0] = ImagePlanePitchLinear{
		PixelType:  PixelTypeInvalid,
		Width:      64,
		Height:     48,
		PitchBytes: 64,
		Data:       0xdeadbeef,
	}
	var data ImageData
	if err := data.SetPitchLinear(ImageBufferHostPitchLinear, want); err != nil {
		t.Fatal(err)
	}
	if data.BufferType != ImageBufferHostPitchLinear {
		t.Errorf("BufferType = %v, want ImageBufferHostPitchLinear", data.BufferType)
	}
	got, err := data.PitchLinear()
	if err != nil {
		t.Fatal(err)
	}
	if *got != want {
		t.Errorf("round trip changed the buffer:\n got %+v\nwant %+v", *got, want)
	}
	// The accessor must alias the storage, not copy it: the wrappers hand
	// &data to the library, so a write through the accessor has to land in
	// data itself.
	got.Planes[0].Width = 32
	again, err := data.PitchLinear()
	if err != nil {
		t.Fatal(err)
	}
	if again.Planes[0].Width != 32 {
		t.Error("PitchLinear returned a copy; it must alias the union storage")
	}
	if uintptr(unsafe.Pointer(got)) != uintptr(unsafe.Pointer(&data.Buffer)) {
		t.Error("PitchLinear does not point at the union storage")
	}
}

// TestUnionAccessorsCheckDiscriminant checks that every accessor refuses a
// discriminant that does not select its member. Without this, a wrong read is
// silent.
func TestUnionAccessorsCheckDiscriminant(t *testing.T) {
	all := []ImageBufferType{
		ImageBufferInvalid,
		ImageBufferHostPitchLinear,
		ImageBufferCudaPitchLinear,
		ImageBufferCudaArray,
		ImageBufferEglimage,
		ImageBufferNvbuffer,
	}
	accessors := []struct {
		name  string
		call  func(*ImageData) error
		valid map[ImageBufferType]bool
	}{
		{"PitchLinear", func(d *ImageData) error { _, err := d.PitchLinear(); return err },
			map[ImageBufferType]bool{ImageBufferHostPitchLinear: true, ImageBufferCudaPitchLinear: true}},
		{"CUDAArray", func(d *ImageData) error { _, err := d.CUDAArray(); return err },
			map[ImageBufferType]bool{ImageBufferCudaArray: true}},
		{"EGLImage", func(d *ImageData) error { _, err := d.EGLImage(); return err },
			map[ImageBufferType]bool{ImageBufferEglimage: true}},
		{"NvBufferFD", func(d *ImageData) error { _, err := d.NvBufferFD(); return err },
			map[ImageBufferType]bool{ImageBufferNvbuffer: true}},
	}
	for _, a := range accessors {
		for _, bt := range all {
			data := ImageData{BufferType: bt}
			err := a.call(&data)
			if a.valid[bt] {
				if err != nil {
					t.Errorf("%s with %v = %v, want nil", a.name, bt, err)
				}
				continue
			}
			var mismatch *ErrBufferTypeMismatch
			if !errors.As(err, &mismatch) {
				t.Errorf("%s with %v = %v (%T), want *ErrBufferTypeMismatch", a.name, bt, err, err)
				continue
			}
			if mismatch.Have != bt {
				t.Errorf("%s with %v reported Have = %v", a.name, bt, mismatch.Have)
			}
		}
	}
	var data ImageData
	if err := data.SetPitchLinear(ImageBufferCudaArray, ImageBufferPitchLinear{}); err == nil {
		t.Error("SetPitchLinear accepted a non-pitch-linear discriminant")
	}
}

// TestUnionMemberWidths checks that reading a narrow member does not read the
// bytes of a wider one. The NvBuffer member is a C int; the union is 208 bytes.
func TestUnionMemberWidths(t *testing.T) {
	var data ImageData
	// Fill the whole union with a recognizable pattern, then set the members
	// the accessors are supposed to read.
	buf := (*[unsafe.Sizeof(ImageBuffer{})]byte)(unsafe.Pointer(&data.Buffer))
	for i := range buf {
		buf[i] = 0xa5
	}
	data.BufferType = ImageBufferNvbuffer
	*(*int32)(unsafe.Pointer(&data.Buffer)) = 7
	fd, err := data.NvBufferFD()
	if err != nil {
		t.Fatal(err)
	}
	if fd != 7 {
		t.Errorf("NvBufferFD = %#x, want 7; a wider read picked up neighbouring union bytes", fd)
	}

	data.BufferType = ImageBufferCudaArray
	*(*uintptr)(unsafe.Pointer(&data.Buffer)) = 0x1234
	arr, err := data.CUDAArray()
	if err != nil {
		t.Fatal(err)
	}
	if arr != 0x1234 {
		t.Errorf("CUDAArray = %#x, want 0x1234", arr)
	}
}

// TestPlaneBytes checks the host-memory guard and the extent of the returned
// slice. The memory here is a Go allocation kept alive for the duration of the
// test, which is legal to read through but is not legal to hand to
// ImageCreateWrapper; see the ownership rules there.
func TestPlaneBytes(t *testing.T) {
	const (
		width  = 8
		height = 4
		pitch  = 16
	)
	backing := make([]byte, pitch*height)
	for i := range backing {
		backing[i] = byte(i)
	}
	buf := ImageBufferPitchLinear{Format: ImageFormatU8, NumPlanes: 1}
	buf.Planes[0] = ImagePlanePitchLinear{
		Width:      width,
		Height:     height,
		PitchBytes: pitch,
		Data:       uintptr(unsafe.Pointer(&backing[0])),
	}
	var data ImageData
	if err := data.SetPitchLinear(ImageBufferHostPitchLinear, buf); err != nil {
		t.Fatal(err)
	}
	got, err := data.PlaneBytes(0)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != pitch*height {
		t.Fatalf("PlaneBytes returned %d bytes, want Height*PitchBytes = %d", len(got), pitch*height)
	}
	for i := range got {
		if got[i] != backing[i] {
			t.Fatalf("PlaneBytes[%d] = %d, want %d", i, got[i], backing[i])
		}
	}
	if _, err := data.PlaneBytes(1); err == nil {
		t.Error("PlaneBytes accepted a plane index past NumPlanes")
	}
	if _, err := data.PlaneBytes(-1); err == nil {
		t.Error("PlaneBytes accepted a negative plane index")
	}

	// A CUDA pitch-linear buffer holds device addresses. Slicing one would
	// hand Go code a pointer it must not dereference.
	data.BufferType = ImageBufferCudaPitchLinear
	if _, err := data.PlaneBytes(0); !errors.Is(err, ErrNotHostMemory) {
		t.Errorf("PlaneBytes on a CUDA buffer = %v, want ErrNotHostMemory", err)
	}
	// Data holds the address as a uintptr, which keeps nothing alive; the
	// backing array must stay reachable until the last read through it.
	runtime.KeepAlive(backing)
}

// TestBackendBits checks the relationships the Backend constants assert. Their
// numeric values are checked against the pinned headers by
// TestConstantsMatchHeaders in cmd/gen-vpi; what cannot be checked there is
// that this package's derived sets agree with them.
func TestBackendBits(t *testing.T) {
	single := []Backend{BackendCPU, BackendCUDA, BackendPVA, BackendVIC, BackendNVENC, BackendOFA, BackendInvalid}
	seen := map[Backend]bool{}
	for _, b := range single {
		if b == 0 || b&(b-1) != 0 {
			t.Errorf("%v = %#x is not a single bit", b, uint64(b))
		}
		if seen[b] {
			t.Errorf("%#x is used by two backends", uint64(b))
		}
		seen[b] = true
	}
	if BackendTegra&^BackendAll != 0 {
		t.Error("BackendTegra has a bit outside BackendAll")
	}
	if BackendAll&BackendInvalid != 0 {
		t.Error("BackendAll includes VPI_BACKEND_INVALID")
	}
	if BackendAll&^BackendMask != 0 {
		t.Error("BackendMask does not cover every backend bit")
	}
	// The object flags share the flags word with the backend bits and must not
	// collide with them.
	for name, flag := range map[string]uint64{
		"ExclusiveStreamAccess": ExclusiveStreamAccess,
		"RestrictMemUsage":      RestrictMemUsage,
		"RequireBackends":       RequireBackends,
		"StreamGreedy":          StreamGreedy,
	} {
		if flag&uint64(BackendMask) != 0 {
			t.Errorf("%s overlaps VPI_BACKEND_MASK", name)
		}
	}
	if got := (BackendCPU | BackendCUDA).String(); got != "CPU|CUDA" {
		t.Errorf("String = %q, want %q", got, "CPU|CUDA")
	}
	if got := Backend(0).String(); got != "none" {
		t.Errorf("Backend(0).String() = %q, want %q", got, "none")
	}
	if got := (BackendCPU | 1<<40).String(); !strings.Contains(got, "CPU|0x") {
		t.Errorf("String of an unknown bit = %q, want it reported numerically", got)
	}
	if !BackendAll.Has(BackendCUDA | BackendCPU) {
		t.Error("Has must require every bit in want")
	}
	if (BackendCPU).Has(BackendCPU | BackendCUDA) {
		t.Error("Has reported a superset match")
	}
}

// TestImageFormatConstantsDistinct checks that no two named formats collide,
// which a transcription error would produce and which String would then hide.
func TestImageFormatConstantsDistinct(t *testing.T) {
	if len(imageFormatNames) < 30 {
		t.Fatalf("imageFormatNames has %d entries; the constant block has more", len(imageFormatNames))
	}
	names := map[string]ImageFormat{}
	for f, name := range imageFormatNames {
		if prev, ok := names[name]; ok {
			t.Errorf("format name %q is used by %#x and %#x", name, uint64(prev), uint64(f))
		}
		names[name] = f
		if f != ImageFormatInvalid && f.String() != name {
			t.Errorf("String(%#x) = %q, want %q", uint64(f), f.String(), name)
		}
	}
	if got := ImageFormat(1).String(); !strings.Contains(got, "0x") {
		t.Errorf("an unnamed format prints as %q, want its hexadecimal value", got)
	}
	if ImageFormat(0) != ImageFormatInvalid {
		t.Error("the zero ImageFormat must be ImageFormatInvalid")
	}
}

// TestResultErrors checks the error surface that does not need the library:
// Success is not an error, and every named code has a distinct message.
func TestResultErrors(t *testing.T) {
	if err := toError(Success); err != nil {
		t.Errorf("toError(Success) = %v, want nil", err)
	}
	if Loaded() {
		t.Skip("library loaded; Error() then reports the library's own message")
	}
	seen := map[string]Result{}
	for r := ErrNotImplemented; r <= ErrInternal; r++ {
		msg := r.Error()
		if !strings.HasPrefix(msg, "nvvpi: ") {
			t.Errorf("Result(%d).Error() = %q, want a package-qualified message", int32(r), msg)
		}
		if strings.Contains(msg, "unknown error") {
			t.Errorf("Result(%d) has no name", int32(r))
		}
		if prev, ok := seen[msg]; ok {
			t.Errorf("Result(%d) and Result(%d) share the message %q", int32(prev), int32(r), msg)
		}
		seen[msg] = r
	}
	if !strings.Contains(Result(9999).Error(), "unknown error") {
		t.Error("an unnamed result must say so rather than inventing a name")
	}
}

// TestCallsWithoutLibraryReturnErrors checks requirement (5) of this package's
// contract: with no library present, every entry point returns a structured
// error and none panics on a nil function pointer.
//
// On a machine that does have libnvvpi installed the calls succeed or fail for
// their own reasons, and the test only checks that nothing panics.
func TestCallsWithoutLibraryReturnErrors(t *testing.T) {
	available := checkLib() == nil
	var data ImageData
	if err := data.SetPitchLinear(ImageBufferHostPitchLinear, ImageBufferPitchLinear{}); err != nil {
		t.Fatal(err)
	}
	calls := []struct {
		name string
		call func() error
	}{
		{"ContextCreate", func() error { _, err := ContextCreate(0); return err }},
		{"StreamCreate", func() error { _, err := StreamCreate(0); return err }},
		{"ImageCreate", func() error { _, err := ImageCreate(4, 4, ImageFormatU8, 0); return err }},
		{"InitImageWrapperParams", func() error { _, err := InitImageWrapperParams(); return err }},
		{"ImageCreateWrapper", func() error { _, err := ImageCreateWrapper(&data, nil, 0); return err }},
		{"ImageLockData", func() error {
			_, err := ImageLockData(0, LockRead, ImageBufferHostPitchLinear)
			return err
		}},
		{"ImageSetWrapper", func() error { return ImageSetWrapper(0, &data) }},
		{"StreamSync", func() error { return StreamSync(0) }},
		{"SubmitRescale", func() error {
			return SubmitRescale(0, BackendCPU, 0, 0, InterpLinear, BorderZero, 0)
		}},
		{"Backends", func() error { _, err := Backends(); return err }},
	}
	for _, c := range calls {
		err := c.call()
		if available {
			continue // the library answered; its verdict is its own
		}
		if err == nil {
			t.Errorf("%s returned nil with no library loaded", c.name)
			continue
		}
		var libErr *LibraryError
		var symErr *SymbolError
		if !errors.As(err, &libErr) && !errors.As(err, &symErr) {
			t.Errorf("%s = %v (%T), want *LibraryError or *SymbolError", c.name, err, err)
		}
	}
	// Nil arguments are rejected before the library is consulted, so this
	// holds whether or not VPI is installed.
	if _, err := ImageCreateWrapper(nil, nil, 0); err == nil {
		t.Error("ImageCreateWrapper accepted nil data")
	}
	if err := ImageSetWrapper(0, nil); err == nil {
		t.Error("ImageSetWrapper accepted nil data")
	}
}
