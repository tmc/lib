package nvfatbin

import (
	"bufio"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"testing"
	"unsafe"
)

// These tests check the binding's agreement with the pinned C ABI. None of
// them loads the library and none needs a GPU: every fact they assert was read
// out of nvFatbin.h and out of the pinned libnvfatbin.so.13.3.29 at generation
// time, and is restated here independently of the generator so that a
// regeneration that changes one of them fails rather than passes quietly.
//
// What they cannot check is that libnvfatbin actually behaves this way. They
// pin the shapes the FFI call sites assume; only running against the real
// library shows whether those assumptions hold.

const pinnedSymbols = "testdata/symbols-linux-x86_64-13.3.29.txt"

// TestBoundSymbolsMatchPinnedLibrary checks that the set of symbols the
// generated loader binds is exactly the set the pinned shared object exports.
//
// Both directions matter. A symbol bound but not exported is a call that fails
// at run time on the version the bindings were generated from. A symbol
// exported but not bound is API surface the package silently does not offer,
// which looks exactly like a header that never declared it.
func TestBoundSymbolsMatchPinnedLibrary(t *testing.T) {
	want, err := readSymbolList(pinnedSymbols)
	if err != nil {
		t.Fatal(err)
	}
	if len(want) == 0 {
		t.Fatalf("%s lists no symbols", pinnedSymbols)
	}
	got, err := registeredSymbols("init.gen.go")
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("bound symbols =\n\t%s\npinned library exports =\n\t%s",
			strings.Join(got, "\n\t"), strings.Join(want, "\n\t"))
	}
}

// readSymbolList reads one symbol per line, ignoring blank lines and comments.
func readSymbolList(path string) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var out []string
	scan := bufio.NewScanner(f)
	for scan.Scan() {
		line := strings.TrimSpace(scan.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		out = append(out, line)
	}
	if err := scan.Err(); err != nil {
		return nil, err
	}
	sort.Strings(out)
	return out, nil
}

// registeredSymbols returns the symbol names passed to register in path.
//
// It reads the generated source rather than calling Has, because Has needs the
// library to be present and this test must run on a machine that has never had
// CUDA installed.
func registeredSymbols(path string) ([]string, error) {
	file, err := parser.ParseFile(token.NewFileSet(), path, nil, parser.SkipObjectResolution)
	if err != nil {
		return nil, err
	}
	var out []string
	ast.Inspect(file, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		ident, ok := call.Fun.(*ast.Ident)
		if !ok || ident.Name != "register" || len(call.Args) != 2 {
			return true
		}
		lit, ok := call.Args[1].(*ast.BasicLit)
		if !ok || lit.Kind != token.STRING {
			return true
		}
		sym, err := strconv.Unquote(lit.Value)
		if err != nil {
			return true
		}
		out = append(out, sym)
		return true
	})
	sort.Strings(out)
	return out, nil
}

// TestFunctionPointerSignatures pins the Go type of every bound function
// pointer against the prototype in the pinned header.
//
// purego marshals by the Go signature alone: it has no view of the C
// declaration, so a wrong argument width or an argument passed by pointer that
// C takes by value is not a compile error and not a load error. It is a
// corrupted call. The C column is the check on this table -- an entry that
// does not match its prototype is a bug in the table, not in the header.
func TestFunctionPointerSignatures(t *testing.T) {
	for _, tt := range []struct {
		symbol string
		c      string
		fn     any
		want   string
	}{{
		symbol: "nvFatbinCreate",
		c:      "nvFatbinResult nvFatbinCreate(nvFatbinHandle *handle_indirect, const char **options, size_t optionsCount)",
		fn:     fnCreate,
		want:   "func(*nvfatbin.Handle, **uint8, uint) nvfatbin.Result",
	}, {
		symbol: "nvFatbinDestroy",
		c:      "nvFatbinResult nvFatbinDestroy(nvFatbinHandle *handle_indirect)",
		fn:     fnDestroy,
		want:   "func(*nvfatbin.Handle) nvfatbin.Result",
	}, {
		symbol: "nvFatbinAddPTX",
		c:      "nvFatbinResult nvFatbinAddPTX(nvFatbinHandle handle, const char *code, size_t size, const char *arch, const char *identifier, const char *optionsCmdLine)",
		fn:     fnaddPTX,
		want:   "func(nvfatbin.Handle, unsafe.Pointer, uint, *uint8, *uint8, *uint8) nvfatbin.Result",
	}, {
		symbol: "nvFatbinAddCubin",
		c:      "nvFatbinResult nvFatbinAddCubin(nvFatbinHandle handle, const void *code, size_t size, const char *arch, const char *identifier)",
		fn:     fnaddCubin,
		want:   "func(nvfatbin.Handle, unsafe.Pointer, uint, *uint8, *uint8) nvfatbin.Result",
	}, {
		symbol: "nvFatbinAddLTOIR",
		c:      "nvFatbinResult nvFatbinAddLTOIR(nvFatbinHandle handle, const void *code, size_t size, const char *arch, const char *identifier, const char *optionsCmdLine)",
		fn:     fnaddLTOIR,
		want:   "func(nvfatbin.Handle, unsafe.Pointer, uint, *uint8, *uint8, *uint8) nvfatbin.Result",
	}, {
		symbol: "nvFatbinAddIndex",
		c:      "nvFatbinResult nvFatbinAddIndex(nvFatbinHandle handle, const void *code, size_t size, const char *identifier)",
		fn:     fnaddIndex,
		want:   "func(nvfatbin.Handle, unsafe.Pointer, uint, *uint8) nvfatbin.Result",
	}, {
		symbol: "nvFatbinAddReloc",
		c:      "nvFatbinResult nvFatbinAddReloc(nvFatbinHandle handle, const void *code, size_t size)",
		fn:     fnaddReloc,
		want:   "func(nvfatbin.Handle, unsafe.Pointer, uint) nvfatbin.Result",
	}, {
		symbol: "nvFatbinAddTileIR",
		c:      "nvFatbinResult nvFatbinAddTileIR(nvFatbinHandle handle, const void *code, size_t size, const char *identifier, const char *optionsCmdLine)",
		fn:     fnaddTileIR,
		want:   "func(nvfatbin.Handle, unsafe.Pointer, uint, *uint8, *uint8) nvfatbin.Result",
	}, {
		symbol: "nvFatbinSize",
		c:      "nvFatbinResult nvFatbinSize(nvFatbinHandle handle, size_t *size)",
		fn:     fnsize,
		want:   "func(nvfatbin.Handle, *uint) nvfatbin.Result",
	}, {
		symbol: "nvFatbinGet",
		c:      "nvFatbinResult nvFatbinGet(nvFatbinHandle handle, void *buffer)",
		fn:     fnget,
		want:   "func(nvfatbin.Handle, unsafe.Pointer) nvfatbin.Result",
	}, {
		symbol: "nvFatbinVersion",
		c:      "nvFatbinResult nvFatbinVersion(unsigned int *major, unsigned int *minor)",
		fn:     fnVersion,
		want:   "func(*uint32, *uint32) nvfatbin.Result",
	}, {
		symbol: "nvFatbinGetErrorString",
		c:      "const char *nvFatbinGetErrorString(nvFatbinResult result)",
		fn:     fnGetErrorString,
		want:   "func(nvfatbin.Result) *uint8",
	}} {
		t.Run(tt.symbol, func(t *testing.T) {
			if got := reflect.TypeOf(tt.fn).String(); got != tt.want {
				t.Errorf("%s\n  Go   %s\n  want %s\n  C    %s", tt.symbol, got, tt.want, tt.c)
			}
		})
	}
}

// TestSignatureTableIsComplete checks that every bound symbol appears in
// TestFunctionPointerSignatures' table, so that a function added to the header
// cannot be bound with no signature check at all.
func TestSignatureTableIsComplete(t *testing.T) {
	bound, err := registeredSymbols("init.gen.go")
	if err != nil {
		t.Fatal(err)
	}
	src, err := os.ReadFile("abi_test.go")
	if err != nil {
		t.Fatal(err)
	}
	for _, sym := range bound {
		if !strings.Contains(string(src), "symbol: "+strconv.Quote(sym)) {
			t.Errorf("%s is bound but has no signature entry", sym)
		}
	}
}

// TestABIWidths pins the sizes and alignments the FFI call sites depend on.
//
// The library ships for LP64 targets: size_t is 8 bytes, unsigned int is 4,
// and a pointer is 8. Go's uint is 8 on every platform this package can reach
// a CUDA library on, and that coincidence is what lets size_t be spelled uint.
// On a 32-bit platform it would not hold, so the test says so rather than
// passing on a platform where the binding is wrong.
func TestABIWidths(t *testing.T) {
	if unsafe.Sizeof(uintptr(0)) != 8 {
		t.Skipf("nvFatbin's ABI is LP64; this platform has %d-byte pointers", unsafe.Sizeof(uintptr(0)))
	}
	for _, tt := range []struct {
		name             string
		c                string
		size, align      uintptr
		wantSize, wantAl uintptr
	}{
		{"Handle", "nvFatbinHandle, a struct _nvFatbinHandle *", unsafe.Sizeof(Handle(0)), unsafe.Alignof(Handle(0)), 8, 8},
		{"Result", "nvFatbinResult, an enum with values 0..17", unsafe.Sizeof(Result(0)), unsafe.Alignof(Result(0)), 4, 4},
		{"uint", "size_t", unsafe.Sizeof(uint(0)), unsafe.Alignof(uint(0)), 8, 8},
		{"uint32", "unsigned int", unsafe.Sizeof(uint32(0)), unsafe.Alignof(uint32(0)), 4, 4},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if tt.size != tt.wantSize || tt.align != tt.wantAl {
				t.Errorf("%s (%s): size %d align %d, want size %d align %d",
					tt.name, tt.c, tt.size, tt.align, tt.wantSize, tt.wantAl)
			}
		})
	}
}

// TestResultCodes restates nvFatbinResult from the pinned header. The enum is
// unnumbered past its first value, so a value inserted upstream renumbers
// every code after it; a caller comparing against a stale constant would then
// mistake one error for another. Transcribing the order here is what makes
// that a test failure at regeneration time.
func TestResultCodes(t *testing.T) {
	want := []Result{
		Success,                   // NVFATBIN_SUCCESS = 0
		ErrInternal,               // NVFATBIN_ERROR_INTERNAL
		ErrElfArchMismatch,        // NVFATBIN_ERROR_ELF_ARCH_MISMATCH
		ErrElfSizeMismatch,        // NVFATBIN_ERROR_ELF_SIZE_MISMATCH
		ErrMissingPtxVersion,      // NVFATBIN_ERROR_MISSING_PTX_VERSION
		ErrNullPointer,            // NVFATBIN_ERROR_NULL_POINTER
		ErrCompressionFailed,      // NVFATBIN_ERROR_COMPRESSION_FAILED
		ErrCompressedSizeExceeded, // NVFATBIN_ERROR_COMPRESSED_SIZE_EXCEEDED
		ErrUnrecognizedOption,     // NVFATBIN_ERROR_UNRECOGNIZED_OPTION
		ErrInvalidArch,            // NVFATBIN_ERROR_INVALID_ARCH
		ErrInvalidNvvm,            // NVFATBIN_ERROR_INVALID_NVVM
		ErrEmptyInput,             // NVFATBIN_ERROR_EMPTY_INPUT
		ErrMissingPtxArch,         // NVFATBIN_ERROR_MISSING_PTX_ARCH
		ErrPtxArchMismatch,        // NVFATBIN_ERROR_PTX_ARCH_MISMATCH
		ErrMissingFatbin,          // NVFATBIN_ERROR_MISSING_FATBIN
		ErrInvalidIndex,           // NVFATBIN_ERROR_INVALID_INDEX
		ErrIdentifierReuse,        // NVFATBIN_ERROR_IDENTIFIER_REUSE
		ErrInternalPtxOption,      // NVFATBIN_ERROR_INTERNAL_PTX_OPTION
	}
	for i, got := range want {
		if int(got) != i {
			t.Errorf("%s = %d, want %d", got, int(got), i)
		}
	}
	if len(resultNames) != len(want) {
		t.Errorf("resultNames has %d entries, header declares %d", len(resultNames), len(want))
	}
}

// TestErrorStringWithoutLibrary checks that Result.Error falls back to the Go
// constant name when nvFatbinGetErrorString is not bound, rather than
// dereferencing a nil function pointer.
func TestErrorStringWithoutLibrary(t *testing.T) {
	if fnGetErrorString != nil {
		t.Skip("library loaded; the C error string is in use")
	}
	if got := ErrInvalidArch.Error(); !strings.Contains(got, "ErrInvalidArch") {
		t.Errorf("Error() = %q, want the Go constant name", got)
	}
	if got := Result(9999).Error(); !strings.Contains(got, "9999") {
		t.Errorf("Error() for an unknown code = %q, want the numeric code", got)
	}
	if got := Success.Error(); got == "" {
		t.Error("Success.Error() is empty")
	}
}
