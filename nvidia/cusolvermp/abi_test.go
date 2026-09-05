package cusolvermp

import (
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/cusolver"
)

// boundSymbols is the C surface this package binds, and the order the
// generator declares it in. A symbol added to or removed from the generated
// package without a matching change here fails TestSymbolPredicates.
var boundSymbols = []string{
	"cusolverMpCreate",
	"cusolverMpDestroy",
	"cusolverMpSetStream",
	"cusolverMpGetStream",
	"cusolverMpGetVersion",
	"cusolverMpCreateDeviceGrid",
	"cusolverMpDestroyGrid",
	"cusolverMpCreateMatrixDesc",
	"cusolverMpDestroyMatrixDesc",
	"cusolverMpBufferRegister",
	"cusolverMpBufferDeregister",
	"cusolverMpMalloc",
	"cusolverMpFree",
	"cusolverMpNUMROC",
	"cusolverMpMatrixScatterH2D",
	"cusolverMpMatrixGatherD2H",
	"cusolverMpGetrf_bufferSize",
	"cusolverMpGetrf",
	"cusolverMpGetrs_bufferSize",
	"cusolverMpGetrs",
	"cusolverMpLoggerOpenFile",
	"cusolverMpLoggerSetLevel",
	"cusolverMpLoggerSetMask",
	"cusolverMpLoggerForceDisable",
	"cusolverMpSetMathMode",
	"cusolverMpGetMathMode",
}

// TestTypeWidths checks the sizes and alignments of every type that crosses
// the FFI boundary by value. cusolverMp.h declares no structs, so there are no
// field offsets, initializers or version fields to check: the whole ABI is
// opaque pointers, enums and integers, and TestNoStructsInSurface guards that
// assumption against a header bump.
func TestTypeWidths(t *testing.T) {
	ptr := unsafe.Sizeof(uintptr(0))
	for _, tc := range []struct {
		name  string
		size  uintptr
		align uintptr
		want  uintptr
	}{
		{"Handle", unsafe.Sizeof(Handle(0)), unsafe.Alignof(Handle(0)), ptr},
		{"Grid", unsafe.Sizeof(Grid(0)), unsafe.Alignof(Grid(0)), ptr},
		{"MatrixDescriptor", unsafe.Sizeof(MatrixDescriptor(0)), unsafe.Alignof(MatrixDescriptor(0)), ptr},
		{"DevicePtr", unsafe.Sizeof(DevicePtr(0)), unsafe.Alignof(DevicePtr(0)), ptr},
		{"HostPtr", unsafe.Sizeof(HostPtr(0)), unsafe.Alignof(HostPtr(0)), ptr},
		{"GridMapping", unsafe.Sizeof(GridMapping(0)), unsafe.Alignof(GridMapping(0)), 4},
		{"Result", unsafe.Sizeof(Result(0)), unsafe.Alignof(Result(0)), 4},
	} {
		if tc.size != tc.want {
			t.Errorf("sizeof(%s) = %d, want %d", tc.name, tc.size, tc.want)
		}
		if tc.align != tc.want {
			t.Errorf("alignof(%s) = %d, want %d", tc.name, tc.align, tc.want)
		}
	}
	// size_t reaches C as Go's uint, and cuSOLVERMp is 64-bit only.
	if got := unsafe.Sizeof(uint(0)); got != 8 {
		t.Errorf("sizeof(uint) = %d; the workspace sizes assume a 64-bit size_t", got)
	}
	// DevicePtr and HostPtr must stay distinct named types: the whole point
	// is that the compiler refuses to pass one where the other belongs.
	if reflect.TypeOf(DevicePtr(0)) == reflect.TypeOf(HostPtr(0)) {
		t.Error("DevicePtr and HostPtr collapsed into one type")
	}
	for _, typ := range []reflect.Type{reflect.TypeOf(DevicePtr(0)), reflect.TypeOf(HostPtr(0))} {
		if typ.Kind() != reflect.Uintptr {
			t.Errorf("%s is %s; an address the collector must not trace has to be uintptr", typ, typ.Kind())
		}
	}
}

// TestResultIsShared checks that this package reports failures with the same
// error type as cuSOLVER, so a caller does not learn a second vocabulary.
func TestResultIsShared(t *testing.T) {
	var r Result = cusolver.ErrZeroPivot
	var target cusolver.Result
	if !errors.As(error(r), &target) || target != cusolver.ErrZeroPivot {
		t.Errorf("Result does not carry cusolver.Result")
	}
	if toError(cusolver.Success) != nil {
		t.Error("success mapped to a non-nil error")
	}
	if toError(cusolver.ErrInvalidValue) == nil {
		t.Error("failure mapped to a nil error")
	}
}

// TestEnumValues pins the only enum in the bound surface to the values in
// cusolverMp.h. Row major is 1 and column major is 0, which is the reverse of
// what the names suggest, so a hand-written constant would very likely be
// wrong.
func TestEnumValues(t *testing.T) {
	if GridMappingRowMajor != 1 {
		t.Errorf("GridMappingRowMajor = %d, want 1", GridMappingRowMajor)
	}
	if GridMappingColMajor != 0 {
		t.Errorf("GridMappingColMajor = %d, want 0", GridMappingColMajor)
	}
	if got := GridMappingRowMajor.String(); got != "GridMappingRowMajor" {
		t.Errorf("String() = %q", got)
	}
}

// TestDataTypeValues pins the cudaDataType and cublasOperation_t values this
// package names. They are CUDA-wide constants, not cuSOLVERMp's own.
func TestDataTypeValues(t *testing.T) {
	for name, pair := range map[string][2]int32{
		"R32F": {int32(R32F), 0},
		"R64F": {int32(R64F), 1},
		"C32F": {int32(C32F), 4},
		"C64F": {int32(C64F), 5},
		"OpN":  {int32(OpN), 0},
		"OpT":  {int32(OpT), 1},
		"OpC":  {int32(OpC), 2},
	} {
		if pair[0] != pair[1] {
			t.Errorf("%s = %d, want %d", name, pair[0], pair[1])
		}
	}
}

// TestSymbolPredicates checks that every bound symbol has an exported
// predicate reporting the same thing as Has, and that neither panics when no
// library is loaded.
func TestSymbolPredicates(t *testing.T) {
	predicates := map[string]func() bool{
		"cusolverMpCreate":             HasCreate,
		"cusolverMpDestroy":            HasDestroy,
		"cusolverMpSetStream":          HasSetStream,
		"cusolverMpGetStream":          HasGetStream,
		"cusolverMpGetVersion":         HasGetVersion,
		"cusolverMpCreateDeviceGrid":   HasCreateDeviceGrid,
		"cusolverMpDestroyGrid":        HasDestroyGrid,
		"cusolverMpCreateMatrixDesc":   HasCreateMatrixDesc,
		"cusolverMpDestroyMatrixDesc":  HasDestroyMatrixDesc,
		"cusolverMpBufferRegister":     HasBufferRegister,
		"cusolverMpBufferDeregister":   HasBufferDeregister,
		"cusolverMpMalloc":             HasMalloc,
		"cusolverMpFree":               HasFree,
		"cusolverMpNUMROC":             HasNUMROC,
		"cusolverMpMatrixScatterH2D":   HasMatrixScatterH2D,
		"cusolverMpMatrixGatherD2H":    HasMatrixGatherD2H,
		"cusolverMpGetrf_bufferSize":   HasGetrfBufferSize,
		"cusolverMpGetrf":              HasGetrf,
		"cusolverMpGetrs_bufferSize":   HasGetrsBufferSize,
		"cusolverMpGetrs":              HasGetrs,
		"cusolverMpLoggerOpenFile":     HasLoggerOpenFile,
		"cusolverMpLoggerSetLevel":     HasLoggerSetLevel,
		"cusolverMpLoggerSetMask":      HasLoggerSetMask,
		"cusolverMpLoggerForceDisable": HasLoggerForceDisable,
		"cusolverMpSetMathMode":        HasSetMathMode,
		"cusolverMpGetMathMode":        HasGetMathMode,
	}
	if len(predicates) != len(boundSymbols) {
		t.Fatalf("%d predicates for %d bound symbols", len(predicates), len(boundSymbols))
	}
	for _, symbol := range boundSymbols {
		predicate, ok := predicates[symbol]
		if !ok {
			t.Errorf("%s has no exported predicate", symbol)
			continue
		}
		if predicate() != Has(symbol) {
			t.Errorf("Has%s disagrees with Has(%q)", symbol, symbol)
		}
	}
}

// TestNoStructsInSurface guards the claim that this package has no struct
// layouts to check. If a header bump introduces one, this fails and layout
// tests have to be written for it.
func TestNoStructsInSurface(t *testing.T) {
	names, err := filepath.Glob("*.gen.go")
	if err != nil {
		t.Fatal(err)
	}
	if len(names) == 0 {
		t.Fatal("no generated files found")
	}
	for _, name := range names {
		// init.gen.go holds the loader's own error types, which model no C
		// declaration and have no ABI.
		if name == "init.gen.go" {
			continue
		}
		src, err := os.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		if strings.Contains(string(src), " struct {") {
			t.Errorf("%s declares a struct; cusolverMp.h declared none, so its layout is now untested", name)
		}
	}
}

// TestUnavailableLibraryIsAnError calls every exported entry point with zero
// arguments on a machine where the library cannot be loaded, and checks that
// each returns an error rather than panicking on a nil function pointer. This
// is the behavior requirement for a package that must stay importable, and
// buildable, where cuSOLVERMp is not installed.
func TestUnavailableLibraryIsAnError(t *testing.T) {
	if err := Load(""); err == nil {
		t.Skipf("libcusolverMp is loadable from %q; this test covers the unavailable-library path", Path())
	}
	calls := map[string]func() error{
		"Create":            func() error { _, err := Create(0, 0); return err },
		"Destroy":           func() error { return Destroy(0) },
		"SetStream":         func() error { return SetStream(0, 0) },
		"GetStream":         func() error { _, err := GetStream(0); return err },
		"GetVersion":        func() error { _, err := GetVersion(0); return err },
		"CreateDeviceGrid":  func() error { _, err := CreateDeviceGrid(0, 0, 1, 1, GridMappingRowMajor); return err },
		"DestroyGrid":       func() error { return DestroyGrid(0) },
		"CreateMatrixDesc":  func() error { _, err := CreateMatrixDesc(0, R64F, 1, 1, 1, 1, 0, 0, 1); return err },
		"DestroyMatrixDesc": func() error { return DestroyMatrixDesc(0) },
		"BufferRegister":    func() error { return BufferRegister(0, 0, 0) },
		"BufferDeregister":  func() error { return BufferDeregister(0, 0) },
		"Malloc":            func() error { _, err := Malloc(0, 0); return err },
		"Free":              func() error { return Free(0, 0) },
		"NUMROC":            func() error { _, err := NUMROC(1, 1, 0, 0, 1); return err },
		"MatrixScatterH2D":  func() error { return MatrixScatterH2D(0, 1, 1, 0, 1, 1, 0, 0, 0, 1) },
		"MatrixGatherD2H":   func() error { return MatrixGatherD2H(0, 1, 1, 0, 1, 1, 0, 0, 0, 1) },
		"GetrfBufferSize":   func() error { _, _, err := GetrfBufferSize(0, 1, 1, 0, 1, 1, 0, 0, R64F); return err },
		"Getrf":             func() error { return Getrf(0, 1, 1, 0, 1, 1, 0, 0, R64F, 0, 0, 0, 0, 0) },
		"GetrsBufferSize": func() error {
			_, _, err := GetrsBufferSize(0, OpN, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, R64F)
			return err
		},
		"Getrs":              func() error { return Getrs(0, OpN, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, R64F, 0, 0, 0, 0, 0) },
		"LoggerOpenFile":     func() error { return LoggerOpenFile("") },
		"LoggerSetLevel":     func() error { return LoggerSetLevel(0) },
		"LoggerSetMask":      func() error { return LoggerSetMask(0) },
		"LoggerForceDisable": func() error { return LoggerForceDisable() },
		"SetMathMode":        func() error { return SetMathMode(0, cusolver.DefaultMath) },
		"GetMathMode":        func() error { _, err := GetMathMode(0); return err },
	}
	if len(calls) != len(boundSymbols) {
		t.Errorf("%d entry points exercised for %d bound symbols", len(calls), len(boundSymbols))
	}
	for name, call := range calls {
		func() {
			defer func() {
				if r := recover(); r != nil {
					t.Errorf("%s panicked with the library unavailable: %v", name, r)
				}
			}()
			if err := call(); err == nil {
				t.Errorf("%s returned nil with the library unavailable", name)
			}
		}()
	}
}
