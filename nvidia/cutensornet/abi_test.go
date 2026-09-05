package cutensornet

import (
	"os"
	"regexp"
	"sort"
	"testing"
	"unsafe"
)

// These tests check the Go side of the cuTensorNet ABI against the pinned
// cuQuantum 2.13.0 headers. Every value below was transcribed by hand from
// cutensornet.h and cutensornet/types.h, independently of the generator, so a
// generator change that silently renames a symbol or renumbers a constant fails
// here.
//
// None of this needs a GPU or the shared library. None of it is evidence that
// the library behaves as documented: it only pins what this package will ask
// for. In particular the struct-layout assertions describe the host that runs
// the test, and are meaningful for the linux/amd64 and linux/arm64 builds of
// libcutensornet only insofar as those ABIs agree with the host's on these
// types, which for arrays of int32 and int64 with pointer members they do.

// wantSymbols is the exact set of C symbols this package binds. Each name was
// checked against a declaration in the pinned headers.
var wantSymbols = []string{
	"cutensornetContractSlices",
	"cutensornetContractionAutotune",
	"cutensornetContractionAutotunePreferenceGetAttribute",
	"cutensornetContractionAutotunePreferenceSetAttribute",
	"cutensornetContractionOptimize",
	"cutensornetContractionOptimizerConfigGetAttribute",
	"cutensornetContractionOptimizerConfigSetAttribute",
	"cutensornetContractionOptimizerInfoGetAttribute",
	"cutensornetContractionOptimizerInfoGetPackedSize",
	"cutensornetContractionOptimizerInfoPackData",
	"cutensornetContractionOptimizerInfoSetAttribute",
	"cutensornetCreate",
	"cutensornetCreateContractionAutotunePreference",
	"cutensornetCreateContractionOptimizerConfig",
	"cutensornetCreateContractionOptimizerInfo",
	"cutensornetCreateContractionOptimizerInfoFromPackedData",
	"cutensornetCreateContractionPlan",
	"cutensornetCreateNetwork",
	"cutensornetCreateNetworkAutotunePreference",
	"cutensornetCreateNetworkDescriptor",
	"cutensornetCreateSliceGroupFromIDRange",
	"cutensornetCreateSliceGroupFromIDs",
	"cutensornetCreateTensorDescriptor",
	"cutensornetCreateWorkspaceDescriptor",
	"cutensornetDestroy",
	"cutensornetDestroyContractionAutotunePreference",
	"cutensornetDestroyContractionOptimizerConfig",
	"cutensornetDestroyContractionOptimizerInfo",
	"cutensornetDestroyContractionPlan",
	"cutensornetDestroyNetwork",
	"cutensornetDestroyNetworkAutotunePreference",
	"cutensornetDestroyNetworkDescriptor",
	"cutensornetDestroySliceGroup",
	"cutensornetDestroyTensorDescriptor",
	"cutensornetDestroyWorkspaceDescriptor",
	"cutensornetGetCudartVersion",
	"cutensornetGetErrorString",
	"cutensornetGetOutputTensorDescriptor",
	"cutensornetGetOutputTensorDetails",
	"cutensornetGetTensorDetails",
	"cutensornetGetVersion",
	"cutensornetLoggerForceDisable",
	"cutensornetLoggerOpenFile",
	"cutensornetLoggerSetLevel",
	"cutensornetLoggerSetMask",
	"cutensornetNetworkAppendTensor",
	"cutensornetNetworkAutotuneContraction",
	"cutensornetNetworkAutotunePreferenceGetAttribute",
	"cutensornetNetworkAutotunePreferenceSetAttribute",
	"cutensornetNetworkContract",
	"cutensornetNetworkGetAttribute",
	"cutensornetNetworkPrepareContraction",
	"cutensornetNetworkSetAttribute",
	"cutensornetNetworkSetInputTensorMemory",
	"cutensornetNetworkSetOptimizerInfo",
	"cutensornetNetworkSetOutputTensor",
	"cutensornetNetworkSetOutputTensorMemory",
	"cutensornetUpdateContractionOptimizerInfoFromPackedData",
	"cutensornetWorkspaceComputeContractionSizes",
	"cutensornetWorkspaceGetMemory",
	"cutensornetWorkspaceGetMemorySize",
	"cutensornetWorkspacePurgeCache",
	"cutensornetWorkspaceSetMemory",
}

var registerRE = regexp.MustCompile(`register\(&\w+, "(\w+)"\)`)

// TestBoundSymbols checks that the generated loader binds exactly the symbols
// this package claims to bind, no more and no fewer. It reads the generated
// source because the symbol names are otherwise only visible after a successful
// dlopen, which needs the library installed.
func TestBoundSymbols(t *testing.T) {
	src, err := os.ReadFile("init.gen.go")
	if err != nil {
		t.Fatalf("read generated loader: %v", err)
	}
	var got []string
	for _, m := range registerRE.FindAllStringSubmatch(string(src), -1) {
		got = append(got, m[1])
	}
	if len(got) == 0 {
		t.Fatal("no register calls found in init.gen.go; the loader shape changed")
	}
	sort.Strings(got)
	want := append([]string(nil), wantSymbols...)
	sort.Strings(want)

	inWant := map[string]bool{}
	for _, s := range want {
		inWant[s] = true
	}
	inGot := map[string]bool{}
	for _, s := range got {
		inGot[s] = true
	}
	for _, s := range got {
		if !inWant[s] {
			t.Errorf("loader binds %s, which is not in the declared surface", s)
		}
	}
	for _, s := range want {
		if !inGot[s] {
			t.Errorf("declared surface names %s, which the loader does not bind", s)
		}
	}
}

// TestSymbolNamesArePrefixed checks that every bound symbol carries the
// library's C prefix. A symbol without it would be a stray binding into some
// other library that happens to be loaded into the process.
func TestSymbolNamesArePrefixed(t *testing.T) {
	for _, s := range wantSymbols {
		if len(s) < len("cutensornet") || s[:len("cutensornet")] != "cutensornet" {
			t.Errorf("symbol %q is not a cutensornet symbol", s)
		}
	}
}

// TestResultCodes pins the cutensornetStatus_t values transcribed from
// cutensornet/types.h. The library reports failures through these; a shift
// would turn one error into another.
func TestResultCodes(t *testing.T) {
	for _, tt := range []struct {
		got  Result
		want int32
		name string
	}{
		{Success, 0, "CUTENSORNET_STATUS_SUCCESS"},
		{ErrNotInitialized, 1, "CUTENSORNET_STATUS_NOT_INITIALIZED"},
		{ErrAllocFailed, 3, "CUTENSORNET_STATUS_ALLOC_FAILED"},
		{ErrInvalidValue, 7, "CUTENSORNET_STATUS_INVALID_VALUE"},
		{ErrArchMismatch, 8, "CUTENSORNET_STATUS_ARCH_MISMATCH"},
		{ErrMappingError, 11, "CUTENSORNET_STATUS_MAPPING_ERROR"},
		{ErrExecutionFailed, 13, "CUTENSORNET_STATUS_EXECUTION_FAILED"},
		{ErrInternalError, 14, "CUTENSORNET_STATUS_INTERNAL_ERROR"},
		{ErrNotSupported, 15, "CUTENSORNET_STATUS_NOT_SUPPORTED"},
		{ErrInsufficientWorkspace, 19, "CUTENSORNET_STATUS_INSUFFICIENT_WORKSPACE"},
		{ErrInsufficientDriver, 20, "CUTENSORNET_STATUS_INSUFFICIENT_DRIVER"},
		{ErrCutensorVersionMismatch, 22, "CUTENSORNET_STATUS_CUTENSOR_VERSION_MISMATCH"},
		{ErrNoDeviceAllocator, 23, "CUTENSORNET_STATUS_NO_DEVICE_ALLOCATOR"},
	} {
		if int32(tt.got) != tt.want {
			t.Errorf("%s = %d, want %d", tt.name, int32(tt.got), tt.want)
		}
	}
}

// TestWorkspaceEnumValues pins the workspace enumerations. These select which
// buffer the library reads and writes; a wrong value silently hands it the
// wrong memory space or the wrong kind.
func TestWorkspaceEnumValues(t *testing.T) {
	if MemspaceDevice != 0 || MemspaceHost != 1 {
		t.Errorf("Memspace = {%d, %d}, want {0, 1}", MemspaceDevice, MemspaceHost)
	}
	if WorkspaceScratch != 0 || WorkspaceCache != 1 {
		t.Errorf("WorkspaceKind = {%d, %d}, want {0, 1}", WorkspaceScratch, WorkspaceCache)
	}
}

// TestWorksizePrefValues pins the workspace preference values, including
// WorksizePrefMax, which the generator drops as a _MAX sentinel and enums.go
// restores by hand.
func TestWorksizePrefValues(t *testing.T) {
	if WorksizePrefMin != 0 {
		t.Errorf("WorksizePrefMin = %d, want 0", WorksizePrefMin)
	}
	if WorksizePrefRecommended != 1 {
		t.Errorf("WorksizePrefRecommended = %d, want 1", WorksizePrefRecommended)
	}
	if WorksizePrefMax != 2 {
		t.Errorf("WorksizePrefMax = %d, want 2", WorksizePrefMax)
	}
}

// TestComputeTypeValues pins the compute-type bit flags. They are a bitmask in
// the header, not a dense enum, so an off-by-one would be a different type
// rather than an invalid one.
func TestComputeTypeValues(t *testing.T) {
	for _, tt := range []struct {
		got  ComputeType
		want int32
		name string
	}{
		{Compute16F, 1 << 0, "CUTENSORNET_COMPUTE_16F"},
		{Compute32F, 1 << 2, "CUTENSORNET_COMPUTE_32F"},
		{Compute64F, 1 << 4, "CUTENSORNET_COMPUTE_64F"},
		{Compute8U, 1 << 6, "CUTENSORNET_COMPUTE_8U"},
		{Compute32U, 1 << 7, "CUTENSORNET_COMPUTE_32U"},
		{Compute8I, 1 << 8, "CUTENSORNET_COMPUTE_8I"},
		{Compute32I, 1 << 9, "CUTENSORNET_COMPUTE_32I"},
		{Compute16BF, 1 << 10, "CUTENSORNET_COMPUTE_16BF"},
		{ComputeTF32, 1 << 12, "CUTENSORNET_COMPUTE_TF32"},
		{Compute3xtf32, 1 << 13, "CUTENSORNET_COMPUTE_3XTF32"},
	} {
		if int32(tt.got) != tt.want {
			t.Errorf("%s = %d, want %d", tt.name, int32(tt.got), tt.want)
		}
	}
}

// TestDataTypeValues pins the cudaDataType_t values this package declares
// itself, since library_types.h is not vendored with the generated package.
func TestDataTypeValues(t *testing.T) {
	for _, tt := range []struct {
		got  DataType
		want int32
		name string
	}{
		{R16F, 2, "CUDA_R_16F"},
		{R32F, 0, "CUDA_R_32F"},
		{R64F, 1, "CUDA_R_64F"},
		{C32F, 4, "CUDA_C_32F"},
		{C64F, 5, "CUDA_C_64F"},
		{R16BF, 14, "CUDA_R_16BF"},
	} {
		if int32(tt.got) != tt.want {
			t.Errorf("%s = %d, want %d", tt.name, int32(tt.got), tt.want)
		}
	}
}

// TestEnumWidth checks that the Go enum types are the four bytes a C enum of
// these ranges occupies. The attribute calls take a byte count, so a wider Go
// type would ask the library to read past the value.
func TestEnumWidth(t *testing.T) {
	for name, size := range map[string]uintptr{
		"Result":                               unsafe.Sizeof(Result(0)),
		"ComputeType":                          unsafe.Sizeof(ComputeType(0)),
		"DataType":                             unsafe.Sizeof(DataType(0)),
		"Memspace":                             unsafe.Sizeof(Memspace(0)),
		"WorkspaceKind":                        unsafe.Sizeof(WorkspaceKind(0)),
		"WorksizePref":                         unsafe.Sizeof(WorksizePref(0)),
		"NetworkAttributes":                    unsafe.Sizeof(NetworkAttributes(0)),
		"ContractionOptimizerConfigAttributes": unsafe.Sizeof(ContractionOptimizerConfigAttributes(0)),
		"ContractionOptimizerInfoAttributes":   unsafe.Sizeof(ContractionOptimizerInfoAttributes(0)),
	} {
		if size != 4 {
			t.Errorf("sizeof(%s) = %d, want 4", name, size)
		}
	}
}

// TestHandleWidth checks that every opaque handle is pointer-width. The C
// typedefs are all void*, and they are passed by value.
func TestHandleWidth(t *testing.T) {
	want := unsafe.Sizeof(uintptr(0))
	for name, size := range map[string]uintptr{
		"Handle":                        unsafe.Sizeof(Handle(0)),
		"NetworkDescriptor":             unsafe.Sizeof(NetworkDescriptor(0)),
		"TensorDescriptor":              unsafe.Sizeof(TensorDescriptor(0)),
		"ContractionOptimizerConfig":    unsafe.Sizeof(ContractionOptimizerConfig(0)),
		"ContractionOptimizerInfo":      unsafe.Sizeof(ContractionOptimizerInfo(0)),
		"ContractionPlan":               unsafe.Sizeof(ContractionPlan(0)),
		"WorkspaceDescriptor":           unsafe.Sizeof(WorkspaceDescriptor(0)),
		"ContractionAutotunePreference": unsafe.Sizeof(ContractionAutotunePreference(0)),
		"NetworkAutotunePreference":     unsafe.Sizeof(NetworkAutotunePreference(0)),
		"SliceGroup":                    unsafe.Sizeof(SliceGroup(0)),
		"Stream":                        unsafe.Sizeof(Stream(0)),
	} {
		if size != want {
			t.Errorf("sizeof(%s) = %d, want %d", name, size, want)
		}
	}
}

// TestTensorQualifiersArray checks that TensorQualifiers is laid out and packed
// so that a Go slice of them is a valid C array. NewNetwork passes the address
// of element zero as cutensornetTensorQualifiers_t qualifiersIn[], so the
// element stride has to be the C sizeof, not a Go-padded one.
func TestTensorQualifiersArray(t *testing.T) {
	var q TensorQualifiers
	if got := unsafe.Sizeof(q); got != 12 {
		t.Errorf("sizeof(TensorQualifiers) = %d, want 12", got)
	}
	if got := unsafe.Offsetof(q.IsConjugate); got != 0 {
		t.Errorf("offsetof(isConjugate) = %d, want 0", got)
	}
	if got := unsafe.Offsetof(q.IsConstant); got != 4 {
		t.Errorf("offsetof(isConstant) = %d, want 4", got)
	}
	if got := unsafe.Offsetof(q.RequiresGradient); got != 8 {
		t.Errorf("offsetof(requiresGradient) = %d, want 8", got)
	}
	a := make([]TensorQualifiers, 3)
	stride := uintptr(unsafe.Pointer(&a[1])) - uintptr(unsafe.Pointer(&a[0]))
	if stride != 12 {
		t.Errorf("[]TensorQualifiers element stride = %d, want 12", stride)
	}
}

// TestPointerCarryingStructLayout checks the records that mix a 32-bit count
// with a pointer. The C compiler pads the count out to the pointer's alignment;
// Go must do the same or the library reads the pointer from the wrong offset.
func TestPointerCarryingStructLayout(t *testing.T) {
	ptrSize := unsafe.Sizeof(uintptr(0))

	var ids TensorIDList
	if got, want := unsafe.Offsetof(ids.Data), ptrSize; got != want {
		t.Errorf("offsetof(cutensornetTensorIDList_t.data) = %d, want %d", got, want)
	}
	if got, want := unsafe.Sizeof(ids), 2*ptrSize; got != want {
		t.Errorf("sizeof(cutensornetTensorIDList_t) = %d, want %d", got, want)
	}

	var path ContractionPath
	if got, want := unsafe.Offsetof(path.Data), ptrSize; got != want {
		t.Errorf("offsetof(cutensornetContractionPath_t.data) = %d, want %d", got, want)
	}

	var cfg SlicingConfig
	if got, want := unsafe.Offsetof(cfg.Data), ptrSize; got != want {
		t.Errorf("offsetof(cutensornetSlicingConfig_t.data) = %d, want %d", got, want)
	}
}

// TestSliceInfoPairLayout checks the int32/int64 pair, where the C compiler
// inserts four bytes of padding before the 64-bit member.
func TestSliceInfoPairLayout(t *testing.T) {
	var p SliceInfoPair
	if got := unsafe.Sizeof(p); got != 16 {
		t.Errorf("sizeof(cutensornetSliceInfoPair_t) = %d, want 16", got)
	}
	if got := unsafe.Offsetof(p.SlicedMode); got != 0 {
		t.Errorf("offsetof(slicedMode) = %d, want 0", got)
	}
	if got := unsafe.Offsetof(p.SlicedExtent); got != 8 {
		t.Errorf("offsetof(slicedExtent) = %d, want 8", got)
	}
}

// TestNodePairLayout checks the pair of int32s used in a contraction path.
func TestNodePairLayout(t *testing.T) {
	var n NodePair
	if got := unsafe.Sizeof(n); got != 8 {
		t.Errorf("sizeof(cutensornetNodePair_t) = %d, want 8", got)
	}
	if got := unsafe.Offsetof(n.Second); got != 4 {
		t.Errorf("offsetof(second) = %d, want 4", got)
	}
}

// TestElementSize pins the element widths used to size device buffers. A wrong
// answer here under-allocates, which the library cannot detect.
func TestElementSize(t *testing.T) {
	for _, tt := range []struct {
		d    DataType
		want int
		ok   bool
		name string
	}{
		{R16F, 2, true, "CUDA_R_16F"},
		{R32F, 4, true, "CUDA_R_32F"},
		{R64F, 8, true, "CUDA_R_64F"},
		{C32F, 8, true, "CUDA_C_32F"},
		{C64F, 16, true, "CUDA_C_64F"},
		{R16BF, 2, true, "CUDA_R_16BF"},
		{R8I, 1, true, "CUDA_R_8I"},
		{C8I, 2, true, "CUDA_C_8I"},
		{R4I, 0, false, "CUDA_R_4I (sub-byte, not addressable)"},
		{R4F_E2M1, 0, false, "CUDA_R_4F_E2M1 (sub-byte, not addressable)"},
	} {
		got, ok := ElementSize(tt.d)
		if got != tt.want || ok != tt.ok {
			t.Errorf("ElementSize(%s) = %d, %v; want %d, %v", tt.name, got, ok, tt.want, tt.ok)
		}
	}
}

// TestAttributeKindsRejectWrongType checks that the typed attribute helpers
// refuse a mismatched access before calling the library, so a caller cannot
// reinterpret a double as two int32s by handing over a same-sized buffer.
func TestAttributeKindsRejectWrongType(t *testing.T) {
	var info ContractionOptimizerInfo
	if _, err := info.Int64(0, ContractionOptimizerInfoFlopCount); err == nil {
		t.Error("Int64 accepted a float64-valued attribute")
	}
	if _, err := info.Float64(0, ContractionOptimizerInfoNumSlices); err == nil {
		t.Error("Float64 accepted an int64-valued attribute")
	}
	if _, err := info.Int32(0, ContractionOptimizerInfoPath); err == nil {
		t.Error("Int32 accepted a composite attribute")
	}
	var cfg ContractionOptimizerConfig
	if err := cfg.SetInt32(0, ContractionOptimizerConfigGpuArch, 90); err == nil {
		t.Error("SetInt32 accepted an attribute whose type the header does not state")
	}
	if err := cfg.SetInt32(0, ContractionOptimizerConfigAttributes(9999), 1); err == nil {
		t.Error("SetInt32 accepted an attribute that is not in the pinned header")
	}
}

// TestAttributeKindTablesComplete checks that every attribute constant the
// generator emitted has an entry in the typed-access tables. A header revision
// that adds an attribute fails here rather than silently falling through to
// "unknown attribute" at run time.
func TestAttributeKindTablesComplete(t *testing.T) {
	// Transcribed from cutensornet/types.h: the count of enumerators in each
	// attribute enum.
	for _, tt := range []struct {
		name string
		got  int
		want int
	}{
		{"cutensornetContractionOptimizerConfigAttributes_t", len(optimizerConfigKinds), 21},
		{"cutensornetContractionOptimizerInfoAttributes_t", len(optimizerInfoKinds), 14},
		{"cutensornetNetworkAttributes_t", len(networkKinds), 7},
		{"cutensornetContractionAutotunePreferenceAttributes_t", len(autotuneKinds), 2},
		{"cutensornetNetworkAutotunePreferenceAttributes_t", len(networkAutotuneKinds), 2},
	} {
		if tt.got != tt.want {
			t.Errorf("%s: typed-access table has %d entries, header declares %d", tt.name, tt.got, tt.want)
		}
	}
}
