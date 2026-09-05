package custatevec

import (
	"errors"
	"strings"
	"testing"
	"unsafe"
)

// These tests need no GPU, no CUDA driver and no cuQuantum installation. They
// check the facts a caller depends on before any call is made: that the Go
// element types have the size and stride of the C ones, that the alignment gap
// this package documents is the only gap, and that reaching the library when it
// is absent produces a structured error rather than a panic.
//
// The size and alignment values below are not copied from the header; they are
// what the C compiler reports for cuComplex and cuDoubleComplex, and
// gen-custatevec's TestNativeFixtures re-derives them from the pinned header on
// every run. This file states them again so that the property is checked in
// the package a caller imports, without a C toolchain.

func TestComplexLayout(t *testing.T) {
	var c64 Complex64
	if got, want := unsafe.Sizeof(c64), uintptr(8); got != want {
		t.Errorf("sizeof(Complex64) = %d, want %d (C cuComplex is a float2)", got, want)
	}
	// CUDA declares float2 __align__(8). Two float32 fields alone would give
	// this struct an alignment of 4, so the type carries a zero-length uint64
	// array to raise it; if that field is ever dropped this fails.
	if got, want := unsafe.Alignof(c64), uintptr(8); got != want {
		t.Errorf("alignof(Complex64) = %d, want %d (C cuComplex is __align__(8))", got, want)
	}
	if got, want := unsafe.Offsetof(c64.X), uintptr(0); got != want {
		t.Errorf("offsetof(Complex64.X) = %d, want %d", got, want)
	}
	if got, want := unsafe.Offsetof(c64.Y), uintptr(4); got != want {
		t.Errorf("offsetof(Complex64.Y) = %d, want %d", got, want)
	}

	var c128 Complex128
	if got, want := unsafe.Sizeof(c128), uintptr(16); got != want {
		t.Errorf("sizeof(Complex128) = %d, want %d (C cuDoubleComplex is a double2)", got, want)
	}
	if got, want := unsafe.Offsetof(c128.X), uintptr(0); got != want {
		t.Errorf("offsetof(Complex128.X) = %d, want %d", got, want)
	}
	if got, want := unsafe.Offsetof(c128.Y), uintptr(8); got != want {
		t.Errorf("offsetof(Complex128.Y) = %d, want %d", got, want)
	}
	// The documented gap: C says __align__(16), Go can only give 8. This is
	// asserted rather than wished away, so that a future Go release that can
	// express the alignment is noticed here.
	if got, want := unsafe.Alignof(c128), uintptr(8); got != want {
		t.Errorf("alignof(Complex128) = %d, want %d; the package documentation "+
			"describes an 8-byte alignment against C's 16", got, want)
	}
}

// TestComplexIsNotGoComplex checks the assumption the package documentation
// warns about. The sizes happen to agree on every platform Go supports today,
// which is exactly why a caller might assume the layouts are interchangeable;
// they are interchangeable only as long as this holds, and the package never
// relies on it.
func TestComplexIsNotGoComplex(t *testing.T) {
	if unsafe.Sizeof(complex128(0)) != unsafe.Sizeof(Complex128{}) {
		t.Errorf("complex128 is %d bytes, Complex128 is %d; a caller must convert, not cast",
			unsafe.Sizeof(complex128(0)), unsafe.Sizeof(Complex128{}))
	}
	if unsafe.Sizeof(complex64(0)) != unsafe.Sizeof(Complex64{}) {
		t.Errorf("complex64 is %d bytes, Complex64 is %d", unsafe.Sizeof(complex64(0)), unsafe.Sizeof(Complex64{}))
	}
	c := MakeComplex128(3 - 4i)
	if c.X != 3 || c.Y != -4 || c.Complex() != 3-4i {
		t.Errorf("MakeComplex128(3-4i) = %+v, round trip %v", c, c.Complex())
	}
	c32 := MakeComplex64(1 + 2i)
	if c32.X != 1 || c32.Y != 2 || c32.Complex() != 1+2i {
		t.Errorf("MakeComplex64(1+2i) = %+v, round trip %v", c32, c32.Complex())
	}
}

// TestArrayStride checks the property an array of amplitudes actually needs:
// consecutive elements are 16 bytes apart, so a Go slice is laid out the way a
// C array of cuDoubleComplex is.
func TestArrayStride(t *testing.T) {
	buf := make([]Complex128, 4)
	stride := uintptr(unsafe.Pointer(&buf[1])) - uintptr(unsafe.Pointer(&buf[0]))
	if stride != 16 {
		t.Errorf("[]Complex128 stride = %d, want 16", stride)
	}
	buf32 := make([]Complex64, 4)
	stride32 := uintptr(unsafe.Pointer(&buf32[1])) - uintptr(unsafe.Pointer(&buf32[0]))
	if stride32 != 8 {
		t.Errorf("[]Complex64 stride = %d, want 8", stride32)
	}
}

func TestAlignedComplex128(t *testing.T) {
	if got := AlignedComplex128(0); got != nil {
		t.Errorf("AlignedComplex128(0) = %v, want nil", got)
	}
	if got := AlignedComplex128(-1); got != nil {
		t.Errorf("AlignedComplex128(-1) = %v, want nil", got)
	}
	// Repeat: the allocator's natural alignment varies with size class, so a
	// single lucky allocation would not show a broken implementation.
	for _, n := range []int{1, 2, 3, 4, 5, 8, 13, 64, 1024} {
		for i := 0; i < 8; i++ {
			buf := AlignedComplex128(n)
			if len(buf) != n {
				t.Fatalf("AlignedComplex128(%d) has length %d", n, len(buf))
			}
			if addr := uintptr(unsafe.Pointer(&buf[0])); addr%16 != 0 {
				t.Fatalf("AlignedComplex128(%d)[0] at %#x is not 16-byte aligned", n, addr)
			}
			// The slice must remain writable Go memory of the right stride.
			buf[n-1] = MakeComplex128(1i)
			if buf[n-1].Complex() != 1i {
				t.Fatalf("AlignedComplex128(%d) last element does not round trip", n)
			}
		}
	}
}

// TestBitOrderingIsRecorded checks that the convention a caller can get
// backwards is a value in the package, not only prose in a doc comment.
func TestBitOrderingIsRecorded(t *testing.T) {
	if !strings.Contains(BitOrdering, "least significant") {
		t.Errorf("BitOrdering = %q, want a statement of which end bit 0 is", BitOrdering)
	}
}

// TestIndexBitConvention states the arithmetic the bit ordering implies. It is
// a check on this package's documentation rather than on the library: if the
// two ever disagree, a caller reading the doc comment computes the wrong index.
func TestIndexBitConvention(t *testing.T) {
	// With three index bits, the basis state where qubit 0 is set is index 1,
	// and the state where qubit 2 is set is index 4.
	for _, tc := range []struct {
		qubit uint
		index int
	}{{0, 1}, {1, 2}, {2, 4}} {
		if got := 1 << tc.qubit; got != tc.index {
			t.Errorf("qubit %d selects index %d, want %d", tc.qubit, got, tc.index)
		}
	}
}

// TestEnumWidths checks that every enum crossing the ABI is a 32-bit value, as
// the C enums are.
func TestEnumWidths(t *testing.T) {
	for _, tc := range []struct {
		name string
		size uintptr
	}{
		{"Result", unsafe.Sizeof(Result(0))},
		{"MatrixLayout", unsafe.Sizeof(MatrixLayout(0))},
		{"ComputeType", unsafe.Sizeof(ComputeType(0))},
		{"CollapseOp", unsafe.Sizeof(CollapseOp(0))},
		{"StateVectorType", unsafe.Sizeof(StateVectorType(0))},
		{"DataType", unsafe.Sizeof(DataType(0))},
		{"PropertyType", unsafe.Sizeof(PropertyType(0))},
	} {
		if tc.size != 4 {
			t.Errorf("sizeof(%s) = %d, want 4", tc.name, tc.size)
		}
	}
	// Handles and streams are pointer-width opaque values.
	if unsafe.Sizeof(Handle(0)) != unsafe.Sizeof(uintptr(0)) {
		t.Errorf("sizeof(Handle) = %d, want pointer width", unsafe.Sizeof(Handle(0)))
	}
	if unsafe.Sizeof(Stream(0)) != unsafe.Sizeof(uintptr(0)) {
		t.Errorf("sizeof(Stream) = %d, want pointer width", unsafe.Sizeof(Stream(0)))
	}
	// size_t parameters are bound as Go uint, which must be 64-bit on the
	// platforms this library ships for.
	if unsafe.Sizeof(uint(0)) != 8 {
		t.Errorf("Go uint is %d bytes; size_t parameters assume 8", unsafe.Sizeof(uint(0)))
	}
}

// TestEnumValues pins the constant values against the pinned header. A
// cuQuantum release that renumbers one of these changes the meaning of a
// caller's compiled code silently, so it must break the build's tests first.
func TestEnumValues(t *testing.T) {
	for _, tc := range []struct {
		name string
		got  int32
		want int32
	}{
		{"Success", int32(Success), 0},
		{"ErrNotSupported", int32(ErrNotSupported), 7},
		{"ErrInsufficientWorkspace", int32(ErrInsufficientWorkspace), 8},
		{"MatrixLayoutCol", int32(MatrixLayoutCol), 0},
		{"MatrixLayoutRow", int32(MatrixLayoutRow), 1},
		{"ComputeDefault", int32(ComputeDefault), 0},
		{"Compute32F", int32(Compute32F), 1 << 2},
		{"Compute64F", int32(Compute64F), 1 << 4},
		{"ComputeTF32", int32(ComputeTF32), 1 << 12},
		{"CollapseNone", int32(CollapseNone), 0},
		{"CollapseNormalizeAndZero", int32(CollapseNormalizeAndZero), 1},
		{"StateVectorTypeZero", int32(StateVectorTypeZero), 0},
		{"StateVectorTypeUniform", int32(StateVectorTypeUniform), 1},
		{"StateVectorTypeGhz", int32(StateVectorTypeGhz), 2},
		{"StateVectorTypeW", int32(StateVectorTypeW), 3},
		// cudaDataType_t, re-derived by the generator from library_types.h.
		{"C32F", int32(C32F), 4},
		{"C64F", int32(C64F), 5},
		{"R64F", int32(R64F), 1},
		{"MajorVersion", int32(MajorVersion), 0},
		{"MinorVersion", int32(MinorVersion), 1},
		{"PatchLevel", int32(PatchLevel), 2},
	} {
		if tc.got != tc.want {
			t.Errorf("%s = %d, want %d", tc.name, tc.got, tc.want)
		}
	}
}

// TestUnavailableLibraryIsAnError checks the behavior on a machine without
// cuQuantum, which is where this package is most often imported: every entry
// point reports a structured error and none of them panics through a nil
// function pointer.
func TestUnavailableLibraryIsAnError(t *testing.T) {
	if Loaded() {
		t.Skip("cuStateVec is installed here; this test covers the absent case")
	}
	_, err := Create()
	if err == nil {
		t.Fatal("Create succeeded without a loaded library")
	}
	var libErr *LibraryError
	if !errors.As(err, &libErr) {
		t.Fatalf("Create error = %v (%T), want *LibraryError", err, err)
	}
	if len(libErr.Attempts) == 0 {
		t.Error("LibraryError does not record which library names were tried")
	}
	if !strings.Contains(err.Error(), "custatevec") {
		t.Errorf("error %q does not name the library", err)
	}

	// A call on a zero handle must fail the same way rather than dereference
	// an unregistered function pointer.
	var h Handle
	if err := h.Destroy(); err == nil {
		t.Error("Destroy on an unloaded library returned nil")
	}
	if err := h.ApplyMatrix(0, C64F, 1, 0, C64F, MatrixLayoutRow, 0, nil, 1, nil, nil, 0, Compute64F, 0, 0); err == nil {
		t.Error("ApplyMatrix on an unloaded library returned nil")
	}

	// GetVersion returns a size_t rather than a status, so it cannot report an
	// error; it must report zero rather than call through a nil pointer.
	if v := GetVersion(); v != 0 {
		t.Errorf("GetVersion() = %d with no library loaded, want 0", v)
	}
}

// TestResultErrorWithoutLibrary checks that a status value formats itself
// without the library's string functions, which are themselves in the library.
func TestResultErrorWithoutLibrary(t *testing.T) {
	if Loaded() {
		t.Skip("cuStateVec is installed here; this test covers the absent case")
	}
	msg := ErrInsufficientWorkspace.Error()
	if msg == "" || !strings.Contains(msg, "8") && !strings.Contains(msg, "InsufficientWorkspace") {
		t.Errorf("Result.Error() = %q, want the constant name or code", msg)
	}
	if got := ErrNotSupported.String(); got != "ErrNotSupported" {
		t.Errorf("Result.String() = %q, want ErrNotSupported", got)
	}
	if got := MatrixLayoutRow.String(); got != "MatrixLayoutRow" {
		t.Errorf("MatrixLayout.String() = %q", got)
	}
	if got := ComputeType(999).String(); !strings.Contains(got, "999") {
		t.Errorf("unknown ComputeType formatted as %q, want the numeric value", got)
	}
}
