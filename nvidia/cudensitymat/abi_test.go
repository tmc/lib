package cudensitymat

import (
	"reflect"
	"testing"
	"unsafe"
)

// The checks here are hand-written from cudensitymat.h and need no GPU and no
// libcudensitymat. They are the part of this package that can be verified on
// the machine that wrote it. They do not show that a call works; they show
// that if the library is present, the bytes this package hands it are the
// bytes the header describes.
//
// layout_test.go covers the four wrapped-callback structs against a clang
// record dump. This file covers what the generator does not emit: the scalar
// types, the enumerator values, and the ABI classification inputs for the
// arguments this package passes by value.

// TestComplex128Layout checks Complex128 against cuComplex.h's cuDoubleComplex,
// which is double2: two doubles, x then y, 16 bytes, 8-byte aligned.
//
// Complex128 is passed by value as the coefficient argument of
// OperatorAppendTerm and OperatorTerm.AppendElementaryProduct. Getting the
// field order wrong would silently conjugate every coefficient in the operator.
func TestComplex128Layout(t *testing.T) {
	var c Complex128
	if got := unsafe.Sizeof(c); got != 16 {
		t.Errorf("sizeof(Complex128) = %d, want 16", got)
	}
	if got := unsafe.Alignof(c); got != 8 {
		t.Errorf("alignof(Complex128) = %d, want 8", got)
	}
	if got := unsafe.Offsetof(c.X); got != 0 {
		t.Errorf("offsetof(Complex128.X) = %d, want 0", got)
	}
	if got := unsafe.Offsetof(c.Y); got != 8 {
		t.Errorf("offsetof(Complex128.Y) = %d, want 8", got)
	}
	if unsafe.Sizeof(c) != unsafe.Sizeof(complex128(0)) {
		t.Fatalf("sizeof(Complex128) = %d, sizeof(complex128) = %d", unsafe.Sizeof(c), unsafe.Sizeof(complex128(0)))
	}
	// Go's complex128 is stored as real then imaginary, the same order as
	// cuDoubleComplex, so a caller may reinterpret one as the other. Assert it
	// rather than assume it: this is what makes the density-matrix buffers a
	// []complex128 in the physical test.
	z := complex(1.5, -2.25)
	got := *(*Complex128)(unsafe.Pointer(&z))
	if got.X != 1.5 || got.Y != -2.25 {
		t.Errorf("complex128 reinterpreted as Complex128 = %+v, want {X:1.5 Y:-2.25}", got)
	}
}

// TestHandleWidth checks that every opaque handle is pointer-sized, since each
// is a C pointer-to-incomplete-struct passed by value.
func TestHandleWidth(t *testing.T) {
	want := unsafe.Sizeof(uintptr(0))
	for _, tc := range []struct {
		name string
		size uintptr
	}{
		{"Handle", unsafe.Sizeof(Handle(0))},
		{"State", unsafe.Sizeof(State(0))},
		{"ElementaryOperator", unsafe.Sizeof(ElementaryOperator(0))},
		{"OperatorTerm", unsafe.Sizeof(OperatorTerm(0))},
		{"Operator", unsafe.Sizeof(Operator(0))},
		{"Expectation", unsafe.Sizeof(Expectation(0))},
		{"WorkspaceDescriptor", unsafe.Sizeof(WorkspaceDescriptor(0))},
		{"Stream", unsafe.Sizeof(Stream(0))},
	} {
		if tc.size != want {
			t.Errorf("sizeof(%s) = %d, want %d", tc.name, tc.size, want)
		}
	}
}

// TestSizeTWidth checks the assumption behind mapping size_t to Go uint. The
// workspace sizes and the component storage sizes cross the boundary as
// size_t, and a 32-bit Go uint against a 64-bit size_t would truncate the
// workspace requirement rather than fail.
func TestSizeTWidth(t *testing.T) {
	if got := unsafe.Sizeof(uint(0)); got != 8 {
		t.Fatalf("sizeof(uint) = %d, want 8; size_t is 8 bytes on every platform this package builds for", got)
	}
}

// TestEnumWidth checks that the generated enum types are 32 bits. C enums with
// only small non-negative enumerators are int-sized here, and the wrapped
// callback structs embed two of them at fixed offsets, so a wider Go type
// would move the fields the layout test pins.
func TestEnumWidth(t *testing.T) {
	for _, tc := range []struct {
		name string
		size uintptr
	}{
		{"Result", unsafe.Sizeof(Result(0))},
		{"ComputeType", unsafe.Sizeof(ComputeType(0))},
		{"DistributedProvider", unsafe.Sizeof(DistributedProvider(0))},
		{"CallbackDevice", unsafe.Sizeof(CallbackDevice(0))},
		{"DifferentiationDir", unsafe.Sizeof(DifferentiationDir(0))},
		{"StatePurity", unsafe.Sizeof(StatePurity(0))},
		{"ElementaryOperatorSparsity", unsafe.Sizeof(ElementaryOperatorSparsity(0))},
		{"Memspace", unsafe.Sizeof(Memspace(0))},
		{"WorkspaceKind", unsafe.Sizeof(WorkspaceKind(0))},
		{"DataType", unsafe.Sizeof(DataType(0))},
	} {
		if tc.size != 4 {
			t.Errorf("sizeof(%s) = %d, want 4", tc.name, tc.size)
		}
	}
}

// TestEnumValues checks every enumerator this package can pass to the library
// against the value written in cudensitymat.h. The generator derives these
// from the header, so a mismatch means the vendored header moved under the
// generated code; the literals below are the pin.
func TestEnumValues(t *testing.T) {
	for _, tc := range []struct {
		name string
		got  int32
		want int32
	}{
		// cudensitymatDistributedProvider_t from the pinned header.
		{"DistributedProviderNone", int32(DistributedProviderNone), 0},
		{"DistributedProviderMpi", int32(DistributedProviderMpi), 1},
		{"DistributedProviderNccl", int32(DistributedProviderNccl), 2},

		// cudensitymatComputeType_t, defined as bit flags.
		{"Compute32F", int32(Compute32F), 1 << 2},
		{"Compute64F", int32(Compute64F), 1 << 4},

		// cudensitymatCallbackDevice_t.
		{"CallbackDeviceCpu", int32(CallbackDeviceCpu), 0},
		{"CallbackDeviceGpu", int32(CallbackDeviceGpu), 1},

		// cudensitymatDifferentiationDir_t.
		{"DifferentiationDirBackward", int32(DifferentiationDirBackward), 1},

		// cudensitymatStatePurity_t.
		{"StatePurityPure", int32(StatePurityPure), 0},
		{"StatePurityMixed", int32(StatePurityMixed), 1},

		// cudensitymatElementaryOperatorSparsity_t.
		{"OperatorSparsityNone", int32(OperatorSparsityNone), 0},
		{"OperatorSparsityMultidiagonal", int32(OperatorSparsityMultidiagonal), 1},

		// cudensitymatMemspace_t.
		{"MemspaceDevice", int32(MemspaceDevice), 0},
		{"MemspaceHost", int32(MemspaceHost), 1},

		// cudensitymatWorkspaceKind_t.
		{"WorkspaceScratch", int32(WorkspaceScratch), 0},

		// cudensitymatStatus_t. The full set, because Result values reach
		// callers as errors and a shifted table would misreport every failure.
		{"Success", int32(Success), 0},
		{"ErrNotInitialized", int32(ErrNotInitialized), 1},
		{"ErrAllocFailed", int32(ErrAllocFailed), 3},
		{"ErrInvalidValue", int32(ErrInvalidValue), 7},
		{"ErrArchMismatch", int32(ErrArchMismatch), 8},
		{"ErrExecutionFailed", int32(ErrExecutionFailed), 13},
		{"ErrInternalError", int32(ErrInternalError), 14},
		{"ErrNotSupported", int32(ErrNotSupported), 15},
		{"ErrCallbackError", int32(ErrCallbackError), 16},
		{"ErrCublasError", int32(ErrCublasError), 17},
		{"ErrCudaError", int32(ErrCudaError), 18},
		{"ErrInsufficientWorkspace", int32(ErrInsufficientWorkspace), 19},
		{"ErrInsufficientDriver", int32(ErrInsufficientDriver), 20},
		{"ErrIoError", int32(ErrIoError), 21},
		{"ErrCutensorVersionMismatch", int32(ErrCutensorVersionMismatch), 22},
		{"ErrNoDeviceAllocator", int32(ErrNoDeviceAllocator), 23},
		{"ErrCutensorError", int32(ErrCutensorError), 24},
		{"ErrCusolverError", int32(ErrCusolverError), 25},
		{"ErrDeviceAllocatorError", int32(ErrDeviceAllocatorError), 26},
		{"ErrDistributedFailure", int32(ErrDistributedFailure), 27},
		{"ErrInterrupted", int32(ErrInterrupted), 28},
		{"ErrCutensornetError", int32(ErrCutensornetError), 29},

		// cudaDataType_t, from the CUDA toolkit's library_types.h rather than
		// from cudensitymat.h. Only the types a quantum state may use are
		// pinned here; the rest are generated for completeness.
		{"R32F", int32(R32F), 0},
		{"R64F", int32(R64F), 1},
		{"C32F", int32(C32F), 4},
		{"C64F", int32(C64F), 5},
	} {
		if tc.got != tc.want {
			t.Errorf("%s = %d, want %d", tc.name, tc.got, tc.want)
		}
	}
}

// TestCallbackNoneIsZero checks that the None callback values are all-zero, so
// they match the header's
//
//	const cudensitymatWrappedScalarCallback_t cudensitymatScalarCallbackNone = {NULL};
//
// A partially initialized wrapper would put a garbage function pointer or a
// nonzero device tag into an argument slot the library dereferences.
func TestCallbackNoneIsZero(t *testing.T) {
	for _, tc := range []struct {
		name string
		v    any
	}{
		{"ScalarCallbackNone", ScalarCallbackNone},
		{"TensorCallbackNone", TensorCallbackNone},
		{"ScalarGradientCallbackNone", ScalarGradientCallbackNone},
		{"TensorGradientCallbackNone", TensorGradientCallbackNone},
	} {
		v := reflect.ValueOf(tc.v)
		if !v.IsZero() {
			t.Errorf("%s = %+v, want the zero value", tc.name, tc.v)
		}
	}
}

// TestCallbackFieldsAreNotFuncs checks the documented decision not to bind Go
// callbacks. The Callback and Wrapper fields must stay uintptr: making either
// a Go func type would let a caller hand the library a Go function whose
// invocation contract -- calling thread, current CUDA context, retention past
// operator destruction -- the header does not state.
func TestCallbackFieldsAreNotFuncs(t *testing.T) {
	for _, typ := range []reflect.Type{
		reflect.TypeOf(WrappedScalarCallback{}),
		reflect.TypeOf(WrappedTensorCallback{}),
		reflect.TypeOf(WrappedScalarGradientCallback{}),
		reflect.TypeOf(WrappedTensorGradientCallback{}),
	} {
		for _, name := range []string{"Callback", "Wrapper"} {
			f, ok := typ.FieldByName(name)
			if !ok {
				t.Errorf("%s has no %s field", typ.Name(), name)
				continue
			}
			if f.Type.Kind() != reflect.Uintptr {
				t.Errorf("%s.%s is %s, want uintptr", typ.Name(), name, f.Type)
			}
		}
	}
}

// TestByValueArgumentSizes pins the sizes that decide how the by-value
// arguments are classified by the System V AMD64 ABI, which is where a purego
// binding of this API is most likely to be wrong.
//
// Complex128 is 16 bytes of nothing but doubles, so it is SSE-class and
// travels in two xmm registers. The four wrapped-callback structs exceed the
// 16-byte register-allocation limit, so they are MEMORY-class and travel on
// the stack. This test does not verify that purego implements that
// classification -- only a call into a real library can -- but it does catch a
// change to these types that would move them between classes without anyone
// noticing.
func TestByValueArgumentSizes(t *testing.T) {
	for _, tc := range []struct {
		name string
		size uintptr
		want uintptr
	}{
		{"Complex128", unsafe.Sizeof(Complex128{}), 16},
		{"WrappedScalarCallback", unsafe.Sizeof(WrappedScalarCallback{}), 24},
		{"WrappedTensorCallback", unsafe.Sizeof(WrappedTensorCallback{}), 24},
		{"WrappedScalarGradientCallback", unsafe.Sizeof(WrappedScalarGradientCallback{}), 32},
		{"WrappedTensorGradientCallback", unsafe.Sizeof(WrappedTensorGradientCallback{}), 32},
	} {
		if tc.size != tc.want {
			t.Errorf("sizeof(%s) = %d, want %d", tc.name, tc.size, tc.want)
		}
	}
	if unsafe.Sizeof(Complex128{}) > 16 {
		t.Error("Complex128 no longer fits the register-allocation limit")
	}
	if unsafe.Sizeof(WrappedScalarCallback{}) <= 16 {
		t.Error("WrappedScalarCallback now fits in registers; the MEMORY-class assumption in the package documentation is stale")
	}
}

// TestResultErrorText checks that an unmapped code still produces an error
// string rather than an empty one, so an error from a newer library version
// is reportable.
func TestResultErrorText(t *testing.T) {
	if got := Success.Error(); got == "" {
		t.Error("Success.Error() is empty")
	}
	unknown := Result(9999)
	if got := unknown.Error(); got == "" {
		t.Error("Result(9999).Error() is empty")
	}
	if got := unknown.String(); got != "Result(9999)" {
		t.Errorf("Result(9999).String() = %q, want %q", got, "Result(9999)")
	}
	if got := ErrInvalidValue.String(); got != "ErrInvalidValue" {
		t.Errorf("ErrInvalidValue.String() = %q", got)
	}
	if toError(Success) != nil {
		t.Error("toError(Success) is non-nil")
	}
	if toError(ErrInvalidValue) == nil {
		t.Error("toError(ErrInvalidValue) is nil")
	}
}
