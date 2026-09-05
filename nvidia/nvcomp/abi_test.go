package nvcomp

import (
	"bufio"
	"errors"
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

	"github.com/ebitengine/purego"
)

// These tests check the binding's agreement with the pinned C ABI. None of
// them loads the library and none needs a GPU: every fact they assert was read
// out of nvcomp/lz4.h and out of the pinned libnvcomp.so.5.3.0.16 at
// generation time, and is restated here independently of the generator, so
// that a regeneration that changes one of them fails rather than passes
// quietly.
//
// What they cannot check is that libnvcomp behaves this way, and in particular
// they cannot check the two things most likely to corrupt memory: that purego
// lays a 64-byte by-value struct argument out the way the System V and AAPCS
// ABIs do, and that the device pointer arrays a caller builds really are in
// device memory. Only running against the real library on a GPU shows either.

const pinnedSymbols = "testdata/symbols-linux-x86_64-5.3.0.16.txt"

// TestBoundSymbolsMatchPinnedLibrary checks that the set of symbols the
// generated loader binds is exactly the set this package claims as its
// surface, as read from the pinned shared object.
//
// Both directions matter. A symbol bound but not exported is a call that fails
// at run time on the version the bindings were generated from. A symbol
// exported but not bound is API surface the package silently does not offer,
// which looks exactly like a header that never declared it -- and that is not
// hypothetical here: 5.3.0 renamed the batched LZ4 temp-size query.
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

// TestNoGetTempSizeSymbol guards the one surface change that renaming would
// hide. nvCOMP before 5.3 exported nvcompBatchedLZ4CompressGetTempSize; this
// release does not, and a binding that reintroduced the old name would bind a
// symbol the pinned library does not define and fail only at run time.
func TestNoGetTempSizeSymbol(t *testing.T) {
	bound, err := registeredSymbols("init.gen.go")
	if err != nil {
		t.Fatal(err)
	}
	for _, sym := range bound {
		if sym == "nvcompBatchedLZ4CompressGetTempSize" || sym == "nvcompBatchedLZ4DecompressGetTempSize" {
			t.Errorf("%s is bound, but nvCOMP 5.3.0 exports only the Async and Sync variants", sym)
		}
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

// signatures pins the Go type of every bound function pointer against the
// prototype in the pinned header.
//
// purego marshals by the Go signature alone: it has no view of the C
// declaration, so a wrong argument width, an argument passed by pointer that C
// takes by value, or an argument order that drifted is not a compile error and
// not a load error. It is a corrupted call. The C column is the check on this
// table -- an entry that does not match its prototype is a bug in the table,
// not in the header.
//
// Two shapes here are the ones worth reading closely. Every device_ parameter
// is a distinct uintptr type rather than a Go pointer, because those addresses
// belong to the device and dereferencing one on the host is a crash at best.
// And compress_opts and decompress_opts cross by value as 64-byte structs.
var signatures = []struct {
	symbol string
	c      string
	fn     any
	want   string
}{{
	symbol: "nvcompBatchedLZ4CompressAsync",
	c: "nvcompStatus_t nvcompBatchedLZ4CompressAsync(const void *const *device_uncompressed_chunk_ptrs, " +
		"const size_t *device_uncompressed_chunk_bytes, size_t max_uncompressed_chunk_bytes, size_t num_chunks, " +
		"void *device_temp_ptr, size_t temp_bytes, void *const *device_compressed_chunk_ptrs, " +
		"size_t *device_compressed_chunk_bytes, nvcompBatchedLZ4CompressOpts_t compress_opts, " +
		"nvcompStatus_t *device_statuses, cudaStream_t stream)",
	fn: fnCompressAsync,
	want: "func(nvcomp.DevicePointerArray, nvcomp.DeviceSizeArray, uint, uint, nvcomp.DeviceBuffer, uint, " +
		"nvcomp.DevicePointerArray, nvcomp.DeviceSizeArray, nvcomp.CompressOpts, nvcomp.DeviceStatusArray, " +
		"nvcomp.Stream) nvcomp.Result",
}, {
	symbol: "nvcompBatchedLZ4CompressGetMaxOutputChunkSize",
	c: "nvcompStatus_t nvcompBatchedLZ4CompressGetMaxOutputChunkSize(size_t max_uncompressed_chunk_bytes, " +
		"nvcompBatchedLZ4CompressOpts_t compress_opts, size_t *max_compressed_chunk_bytes)",
	fn:   fnCompressGetMaxOutputChunkSize,
	want: "func(uint, nvcomp.CompressOpts, *uint) nvcomp.Result",
}, {
	symbol: "nvcompBatchedLZ4CompressGetRequiredAlignments",
	c: "nvcompStatus_t nvcompBatchedLZ4CompressGetRequiredAlignments(nvcompBatchedLZ4CompressOpts_t compress_opts, " +
		"nvcompAlignmentRequirements_t *alignment_requirements)",
	fn:   fnCompressGetRequiredAlignments,
	want: "func(nvcomp.CompressOpts, *nvcomp.AlignmentRequirements) nvcomp.Result",
}, {
	symbol: "nvcompBatchedLZ4CompressGetTempSizeAsync",
	c: "nvcompStatus_t nvcompBatchedLZ4CompressGetTempSizeAsync(size_t num_chunks, " +
		"size_t max_uncompressed_chunk_bytes, nvcompBatchedLZ4CompressOpts_t compress_opts, size_t *temp_bytes, " +
		"size_t max_total_uncompressed_bytes)",
	fn:   fnCompressGetTempSizeAsync,
	want: "func(uint, uint, nvcomp.CompressOpts, *uint, uint) nvcomp.Result",
}, {
	symbol: "nvcompBatchedLZ4CompressGetTempSizeSync",
	c: "nvcompStatus_t nvcompBatchedLZ4CompressGetTempSizeSync(const void *const *const device_uncompressed_chunk_ptrs, " +
		"const size_t *const device_uncompressed_chunk_bytes, size_t num_chunks, size_t max_uncompressed_chunk_bytes, " +
		"nvcompBatchedLZ4CompressOpts_t compress_opts, size_t *temp_bytes, size_t max_total_uncompressed_bytes, " +
		"cudaStream_t stream)",
	fn: fnCompressGetTempSizeSync,
	want: "func(nvcomp.DevicePointerArray, nvcomp.DeviceSizeArray, uint, uint, nvcomp.CompressOpts, *uint, uint, " +
		"nvcomp.Stream) nvcomp.Result",
}, {
	symbol: "nvcompBatchedLZ4DecompressAsync",
	c: "nvcompStatus_t nvcompBatchedLZ4DecompressAsync(const void *const *device_compressed_chunk_ptrs, " +
		"const size_t *device_compressed_chunk_bytes, const size_t *device_uncompressed_buffer_bytes, " +
		"size_t *device_uncompressed_chunk_bytes, size_t num_chunks, void *const device_temp_ptr, size_t temp_bytes, " +
		"void *const *device_uncompressed_chunk_ptrs, nvcompBatchedLZ4DecompressOpts_t decompress_opts, " +
		"nvcompStatus_t *device_statuses, cudaStream_t stream)",
	fn: fnDecompressAsync,
	want: "func(nvcomp.DevicePointerArray, nvcomp.DeviceSizeArray, nvcomp.DeviceSizeArray, nvcomp.DeviceSizeArray, " +
		"uint, nvcomp.DeviceBuffer, uint, nvcomp.DevicePointerArray, nvcomp.DecompressOpts, " +
		"nvcomp.DeviceStatusArray, nvcomp.Stream) nvcomp.Result",
}, {
	symbol: "nvcompBatchedLZ4DecompressGetRequiredAlignments",
	c: "nvcompStatus_t nvcompBatchedLZ4DecompressGetRequiredAlignments(nvcompBatchedLZ4DecompressOpts_t decompress_opts, " +
		"nvcompAlignmentRequirements_t *alignment_requirements)",
	fn:   fnDecompressGetRequiredAlignments,
	want: "func(nvcomp.DecompressOpts, *nvcomp.AlignmentRequirements) nvcomp.Result",
}, {
	symbol: "nvcompBatchedLZ4DecompressGetTempSizeAsync",
	c: "nvcompStatus_t nvcompBatchedLZ4DecompressGetTempSizeAsync(size_t num_chunks, " +
		"size_t max_uncompressed_chunk_bytes, nvcompBatchedLZ4DecompressOpts_t decompress_opts, size_t *temp_bytes, " +
		"size_t max_total_uncompressed_bytes)",
	fn:   fnDecompressGetTempSizeAsync,
	want: "func(uint, uint, nvcomp.DecompressOpts, *uint, uint) nvcomp.Result",
}, {
	symbol: "nvcompBatchedLZ4DecompressGetTempSizeSync",
	c: "nvcompStatus_t nvcompBatchedLZ4DecompressGetTempSizeSync(const void *const *const device_compressed_chunk_ptrs, " +
		"const size_t *const device_compressed_chunk_bytes, size_t num_chunks, size_t max_uncompressed_chunk_bytes, " +
		"size_t *temp_bytes, size_t max_total_uncompressed_bytes, nvcompBatchedLZ4DecompressOpts_t decompress_opts, " +
		"nvcompStatus_t *device_statuses, cudaStream_t stream)",
	fn: fnDecompressGetTempSizeSync,
	want: "func(nvcomp.DevicePointerArray, nvcomp.DeviceSizeArray, uint, uint, *uint, uint, nvcomp.DecompressOpts, " +
		"nvcomp.DeviceStatusArray, nvcomp.Stream) nvcomp.Result",
}, {
	symbol: "nvcompBatchedLZ4GetDecompressSizeAsync",
	c: "nvcompStatus_t nvcompBatchedLZ4GetDecompressSizeAsync(const void *const *device_compressed_chunk_ptrs, " +
		"const size_t *device_compressed_chunk_bytes, size_t *device_uncompressed_chunk_bytes, size_t num_chunks, " +
		"cudaStream_t stream)",
	fn: fnGetDecompressSizeAsync,
	want: "func(nvcomp.DevicePointerArray, nvcomp.DeviceSizeArray, nvcomp.DeviceSizeArray, uint, " +
		"nvcomp.Stream) nvcomp.Result",
}, {
	symbol: "nvcompGetProperties",
	c:      "nvcompStatus_t nvcompGetProperties(nvcompProperties_t *properties)",
	fn:     fnGetProperties,
	want:   "func(*nvcomp.Properties) nvcomp.Result",
}, {
	symbol: "nvcompGetStatusString",
	c:      "const char *nvcompGetStatusString(nvcompStatus_t status)",
	fn:     fnGetStatusString,
	want:   "func(nvcomp.Result) *uint8",
}}

func TestFunctionPointerSignatures(t *testing.T) {
	for _, tt := range signatures {
		t.Run(tt.symbol, func(t *testing.T) {
			if got := reflect.TypeOf(tt.fn).String(); got != tt.want {
				t.Errorf("%s\n  Go   %s\n  want %s\n  C    %s", tt.symbol, got, tt.want, tt.c)
			}
		})
	}
}

// TestSignatureTableIsComplete checks that every bound symbol appears in the
// signature table, so that a function added to the header cannot be bound with
// no signature check at all.
func TestSignatureTableIsComplete(t *testing.T) {
	bound, err := registeredSymbols("init.gen.go")
	if err != nil {
		t.Fatal(err)
	}
	listed := make(map[string]bool, len(signatures))
	for _, s := range signatures {
		listed[s.symbol] = true
	}
	for _, sym := range bound {
		if !listed[sym] {
			t.Errorf("%s is bound but has no signature entry", sym)
		}
	}
	if len(listed) != len(bound) {
		t.Errorf("signature table has %d entries, %d symbols are bound", len(listed), len(bound))
	}
}

// TestPuregoAcceptsSignatures checks that purego will register every bound
// signature on this platform.
//
// It is worth a test of its own because of the two option structs. purego
// supports struct arguments only on some architectures, only for some field
// kinds, and only within its own argument-slot budget; when it does not, it
// panics at registration rather than returning an error. Registering here,
// into throwaway variables and with a C address that is never called, turns
// that panic into a test failure naming the call, instead of a panic inside
// whatever program first touched the binding.
//
// It does not check that purego's layout matches the C ABI. Nothing that runs
// without the library can.
func TestPuregoAcceptsSignatures(t *testing.T) {
	// Any non-zero address will do: purego rejects a nil one, and these
	// function values are discarded without ever being called.
	const notCalled = uintptr(1)
	for _, tt := range signatures {
		t.Run(tt.symbol, func(t *testing.T) {
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("purego rejects %s (%s): %v", tt.symbol, tt.want, r)
				}
			}()
			fn := reflect.New(reflect.TypeOf(tt.fn))
			purego.RegisterFunc(fn.Interface(), notCalled)
		})
	}
}

// TestABIWidths pins the sizes and alignments the FFI call sites depend on.
//
// The library ships for LP64 targets: size_t is 8 bytes, a pointer is 8, an
// enum is 4. Go's uint is 8 on every platform this package can reach a CUDA
// library on, and that coincidence is what lets size_t be spelled uint. On a
// 32-bit platform it would not hold, so the test says so rather than passing
// on a platform where the binding is wrong.
func TestABIWidths(t *testing.T) {
	if unsafe.Sizeof(uintptr(0)) != 8 {
		t.Skipf("nvCOMP's ABI is LP64; this platform has %d-byte pointers", unsafe.Sizeof(uintptr(0)))
	}
	for _, tt := range []struct {
		name             string
		c                string
		size, align      uintptr
		wantSize, wantAl uintptr
	}{
		{"Result", "nvcompStatus_t, an enum", unsafe.Sizeof(Result(0)), unsafe.Alignof(Result(0)), 4, 4},
		{"DataType", "nvcompType_t, an enum", unsafe.Sizeof(DataType(0)), unsafe.Alignof(DataType(0)), 4, 4},
		{"DecompressBackend", "nvcompDecompressBackend_t, an enum", unsafe.Sizeof(DecompressBackend(0)), unsafe.Alignof(DecompressBackend(0)), 4, 4},
		{"BitshuffleMode", "nvcompBitshuffleMode_t, an enum", unsafe.Sizeof(BitshuffleMode(0)), unsafe.Alignof(BitshuffleMode(0)), 4, 4},
		{"DeviceBuffer", "void *", unsafe.Sizeof(DeviceBuffer(0)), unsafe.Alignof(DeviceBuffer(0)), 8, 8},
		{"DevicePointerArray", "void *const *", unsafe.Sizeof(DevicePointerArray(0)), unsafe.Alignof(DevicePointerArray(0)), 8, 8},
		{"DeviceSizeArray", "size_t *", unsafe.Sizeof(DeviceSizeArray(0)), unsafe.Alignof(DeviceSizeArray(0)), 8, 8},
		{"DeviceStatusArray", "nvcompStatus_t *", unsafe.Sizeof(DeviceStatusArray(0)), unsafe.Alignof(DeviceStatusArray(0)), 8, 8},
		{"Stream", "cudaStream_t, a struct CUstream_st *", unsafe.Sizeof(Stream(0)), unsafe.Alignof(Stream(0)), 8, 8},
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

// TestByValueOptionLayout restates the layout of the two structs the C API
// takes by value.
//
// layout_test.go asserts the same numbers from the clang record dump taken at
// generation time. They are repeated here from a reading of the header because
// this is the failure that cannot be caught any other way without a GPU: a
// by-value argument whose size or field offsets are wrong is passed as
// whatever bytes happen to be there, and the library reads them as options.
func TestByValueOptionLayout(t *testing.T) {
	for _, tt := range []struct {
		name             string
		c                string
		size, align      uintptr
		wantSize, wantAl uintptr
	}{{
		name: "CompressOpts",
		c: "struct { nvcompType_t data_type; nvcompBitshuffleMode_t bitshuffle_mode; char reserved[56]; }" +
			" -- 4 + 4 + 56, no tail padding at align 4",
		size: unsafe.Sizeof(CompressOpts{}), align: unsafe.Alignof(CompressOpts{}), wantSize: 64, wantAl: 4,
	}, {
		name: "DecompressOpts",
		c: "struct { nvcompDecompressBackend_t backend; int sort_before_hw_decompress; nvcompType_t data_type;" +
			" nvcompBitshuffleMode_t bitshuffle_mode; char reserved[48]; } -- 4 + 4 + 4 + 4 + 48",
		size: unsafe.Sizeof(DecompressOpts{}), align: unsafe.Alignof(DecompressOpts{}), wantSize: 64, wantAl: 4,
	}} {
		t.Run(tt.name, func(t *testing.T) {
			if tt.size != tt.wantSize || tt.align != tt.wantAl {
				t.Errorf("%s: size %d align %d, want size %d align %d\n  C %s",
					tt.name, tt.size, tt.align, tt.wantSize, tt.wantAl, tt.c)
			}
		})
	}
	if got := unsafe.Offsetof(CompressOpts{}.Reserved); got != 8 {
		t.Errorf("offsetof(CompressOpts.Reserved) = %d, want 8", got)
	}
	if got := len(CompressOpts{}.Reserved); got != 56 {
		t.Errorf("len(CompressOpts.Reserved) = %d, want 56", got)
	}
	if got := unsafe.Offsetof(DecompressOpts{}.Reserved); got != 16 {
		t.Errorf("offsetof(DecompressOpts.Reserved) = %d, want 16", got)
	}
	if got := len(DecompressOpts{}.Reserved); got != 48 {
		t.Errorf("len(DecompressOpts.Reserved) = %d, want 48", got)
	}
}

// TestZeroValueIsVendorDefault checks the claim the package documentation
// makes: that a zero [CompressOpts] or [DecompressOpts] is the same value as
// the header's own default object.
//
//	static const nvcompBatchedLZ4CompressOpts_t nvcompBatchedLZ4CompressDefaultOpts =
//	  {NVCOMP_TYPE_CHAR, NVCOMP_BITSHUFFLE_NONE, {0}};
//	static const nvcompBatchedLZ4DecompressOpts_t nvcompBatchedLZ4DecompressDefaultOpts =
//	  {NVCOMP_DECOMPRESS_BACKEND_DEFAULT, 0, NVCOMP_TYPE_CHAR, NVCOMP_BITSHUFFLE_NONE, {0}};
//
// The claim holds only because every constant named there is zero. If upstream
// renumbers one of those enums, or changes a default to a non-zero value, a
// caller passing a zero value would silently be asking for something else.
// gen-nvcomp fails when either initializer changes; this fails when a value in
// it does.
func TestZeroValueIsVendorDefault(t *testing.T) {
	if DataTypeChar != 0 {
		t.Errorf("DataTypeChar = %d, want 0: the default compress opts name it as the zero value", DataTypeChar)
	}
	if BitshuffleNone != 0 {
		t.Errorf("BitshuffleNone = %d, want 0", BitshuffleNone)
	}
	if DecompressBackendDefault != 0 {
		t.Errorf("DecompressBackendDefault = %d, want 0", DecompressBackendDefault)
	}
	if got := (CompressOpts{}); got.DataType != DataTypeChar || got.BitshuffleMode != BitshuffleNone {
		t.Errorf("CompressOpts{} = %+v, want the vendor default", got)
	}
	if got := (DecompressOpts{}); got.Backend != DecompressBackendDefault || got.SortBeforeHwDecompress != 0 ||
		got.DataType != DataTypeChar || got.BitshuffleMode != BitshuffleNone {
		t.Errorf("DecompressOpts{} = %+v, want the vendor default", got)
	}
	for i, b := range (CompressOpts{}).Reserved {
		if b != 0 {
			t.Errorf("CompressOpts{}.Reserved[%d] = %d, want 0", i, b)
		}
	}
	for i, b := range (DecompressOpts{}).Reserved {
		if b != 0 {
			t.Errorf("DecompressOpts{}.Reserved[%d] = %d, want 0", i, b)
		}
	}
}

// TestEnumValues restates the option enums from the pinned header. They are
// written into a by-value struct, so a constant whose value drifted from the
// library's would select a different algorithm rather than fail.
func TestEnumValues(t *testing.T) {
	for _, tt := range []struct {
		c    string
		got  int32
		want int32
	}{
		{"NVCOMP_TYPE_CHAR", int32(DataTypeChar), 0},
		{"NVCOMP_TYPE_UCHAR", int32(DataTypeUchar), 1},
		{"NVCOMP_TYPE_SHORT", int32(DataTypeShort), 2},
		{"NVCOMP_TYPE_USHORT", int32(DataTypeUshort), 3},
		{"NVCOMP_TYPE_INT", int32(DataTypeInt), 4},
		{"NVCOMP_TYPE_UINT", int32(DataTypeUint), 5},
		{"NVCOMP_TYPE_LONGLONG", int32(DataTypeLonglong), 6},
		{"NVCOMP_TYPE_ULONGLONG", int32(DataTypeUlonglong), 7},
		{"NVCOMP_TYPE_FLOAT16", int32(DataTypeFloat16), 9},
		{"NVCOMP_TYPE_FLOAT8_E4M3", int32(DataTypeFloat8E4M3), 10},
		{"NVCOMP_TYPE_BITS", int32(DataTypeBits), 0xff},
		{"NVCOMP_DECOMPRESS_BACKEND_DEFAULT", int32(DecompressBackendDefault), 0},
		{"NVCOMP_DECOMPRESS_BACKEND_HARDWARE", int32(DecompressBackendHardware), 1},
		{"NVCOMP_DECOMPRESS_BACKEND_CUDA", int32(DecompressBackendCuda), 2},
		{"NVCOMP_BITSHUFFLE_NONE", int32(BitshuffleNone), 0},
		{"NVCOMP_BITSHUFFLE_MSB_FIRST", int32(BitshuffleMsbFirst), 1},
		{"NVCOMP_BITSHUFFLE_LSB_FIRST", int32(BitshuffleLsbFirst), 2},
	} {
		t.Run(tt.c, func(t *testing.T) {
			if tt.got != tt.want {
				t.Errorf("%s = %d, want %d", tt.c, tt.got, tt.want)
			}
		})
	}
}

// TestResultCodes restates nvcompStatus_t from the pinned header. The values
// are explicit and deliberately sparse -- 0, then 10 through 24, then 1000 and
// 10000 -- so unlike a densely numbered enum they cannot be checked by
// position. A constant transcribed wrong here would turn one failure into
// another silently.
func TestResultCodes(t *testing.T) {
	for _, tt := range []struct {
		c    string
		got  Result
		want Result
	}{
		{"nvcompSuccess", Success, 0},
		{"nvcompErrorInvalidValue", ErrInvalidValue, 10},
		{"nvcompErrorNotSupported", ErrNotSupported, 11},
		{"nvcompErrorCannotDecompress", ErrCannotDecompress, 12},
		{"nvcompErrorBadChecksum", ErrBadChecksum, 13},
		{"nvcompErrorCannotVerifyChecksums", ErrCannotVerifyChecksums, 14},
		{"nvcompErrorOutputBufferTooSmall", ErrOutputBufferTooSmall, 15},
		{"nvcompErrorWrongHeaderLength", ErrWrongHeaderLength, 16},
		{"nvcompErrorAlignment", ErrAlignment, 17},
		{"nvcompErrorChunkSizeTooLarge", ErrChunkSizeTooLarge, 18},
		{"nvcompErrorCannotCompress", ErrCannotCompress, 19},
		{"nvcompErrorWrongInputLength", ErrWrongInputLength, 20},
		{"nvcompErrorBatchSizeTooLarge", ErrBatchSizeTooLarge, 21},
		{"nvcompErrorSubChunkCountTooLarge", ErrSubChunkCountTooLarge, 22},
		{"nvcompErrorSubChunkCountTooSmall", ErrSubChunkCountTooSmall, 23},
		{"nvcompErrorOutputBufferAlignmentTooSmall", ErrOutputBufferAlignmentTooSmall, 24},
		{"nvcompErrorCudaError", ErrCudaError, 1000},
		{"nvcompErrorInternal", ErrInternal, 10000},
	} {
		t.Run(tt.c, func(t *testing.T) {
			if tt.got != tt.want {
				t.Errorf("%s = %d, want %d", tt.c, int(tt.got), int(tt.want))
			}
		})
	}
}

// TestHeaderVersion checks that the generated version constants are the
// release the pinned symbol list was read from, so that regenerating against a
// different nvCOMP without updating testdata fails here.
func TestHeaderVersion(t *testing.T) {
	if HeaderVersionMajor != 5 || HeaderVersionMinor != 3 || HeaderVersionPatch != 0 || HeaderVersionBuild != 16 {
		t.Errorf("header version %d.%d.%d.%d, want 5.3.0.16, which is what %s was read from",
			HeaderVersionMajor, HeaderVersionMinor, HeaderVersionPatch, HeaderVersionBuild, pinnedSymbols)
	}
}

// TestSemanticVersionDecoding checks the decoding against the macros in
// nvcomp/version.h:
//
//	#define MAKE_SEMANTIC_VERSION(major, minor, patch) ((major * 1000) + (minor * 100) + patch)
func TestSemanticVersionDecoding(t *testing.T) {
	for _, tt := range []struct {
		version                 uint32
		major, minor, patch     int
		wantMaj, wantMin, wantP int
	}{
		{version: 5300, wantMaj: 5, wantMin: 3, wantP: 0},
		{version: 5316, wantMaj: 5, wantMin: 3, wantP: 16},
		{version: 4099, wantMaj: 4, wantMin: 0, wantP: 99},
	} {
		p := Properties{Version: tt.version}
		major, minor, patch := p.SemanticVersion()
		if major != tt.wantMaj || minor != tt.wantMin || patch != tt.wantP {
			t.Errorf("Properties{Version: %d}.SemanticVersion() = %d, %d, %d, want %d, %d, %d",
				tt.version, major, minor, patch, tt.wantMaj, tt.wantMin, tt.wantP)
		}
	}
}

// TestLimits restates the four file-scope constants gen-nvcomp transcribes
// from the header. They are not exported symbols, so nothing at run time can
// catch a wrong one.
func TestLimits(t *testing.T) {
	if CompressionMaxAllowedChunkSize != 16777216 {
		t.Errorf("CompressionMaxAllowedChunkSize = %d, want 1 << 24", CompressionMaxAllowedChunkSize)
	}
	if DecompressionMaxAllowedChunkSize != 4294967295 {
		t.Errorf("DecompressionMaxAllowedChunkSize = %d, want (1ull << 32) - 1", DecompressionMaxAllowedChunkSize)
	}
	if RequiredCompressionAlignment != 4 {
		t.Errorf("RequiredCompressionAlignment = %d, want 4", RequiredCompressionAlignment)
	}
	if RequiredDecompressionAlignment != 1 {
		t.Errorf("RequiredDecompressionAlignment = %d, want 1", RequiredDecompressionAlignment)
	}
}

// TestCheckStatuses covers the helper that turns a copied-back status array
// into an error, including the case the C API makes easy to get wrong: an
// array of zeros reads as a batch that succeeded, whether or not it ran.
func TestCheckStatuses(t *testing.T) {
	for _, tt := range []struct {
		name     string
		statuses []Result
		wantErr  bool
		chunk    int
		result   Result
		failed   int
	}{
		{name: "empty", statuses: nil},
		{name: "all success", statuses: []Result{Success, Success, Success}},
		{name: "one failure", statuses: []Result{Success, ErrCannotDecompress, Success},
			wantErr: true, chunk: 1, result: ErrCannotDecompress, failed: 1},
		{name: "first of several", statuses: []Result{ErrAlignment, Success, ErrCannotCompress},
			wantErr: true, chunk: 0, result: ErrAlignment, failed: 2},
	} {
		t.Run(tt.name, func(t *testing.T) {
			err := CheckStatuses(tt.statuses)
			if !tt.wantErr {
				if err != nil {
					t.Fatalf("CheckStatuses(%v) = %v, want nil", tt.statuses, err)
				}
				return
			}
			var chunkErr *ChunkError
			if !errors.As(err, &chunkErr) {
				t.Fatalf("CheckStatuses(%v) = %v (%T), want *ChunkError", tt.statuses, err, err)
			}
			if chunkErr.Chunk != tt.chunk || chunkErr.Result != tt.result || chunkErr.Failed != tt.failed {
				t.Errorf("ChunkError = %+v, want chunk %d, result %v, failed %d",
					chunkErr, tt.chunk, tt.result, tt.failed)
			}
			if chunkErr.Chunks != len(tt.statuses) {
				t.Errorf("ChunkError.Chunks = %d, want %d", chunkErr.Chunks, len(tt.statuses))
			}
			if !errors.Is(err, tt.result) {
				t.Errorf("errors.Is(err, %v) = false, want true", tt.result)
			}
		})
	}
}

// TestCallsWithoutLibraryReportErrors checks that every exported call reports a
// structured error rather than panicking on a nil function pointer when the
// library is not present. On a machine that does have libnvcomp the calls
// resolve, and the test only checks that nothing panicked.
func TestCallsWithoutLibraryReportErrors(t *testing.T) {
	if _, err := GetProperties(); err != nil {
		var symErr *SymbolError
		var libErr *LibraryError
		if !errors.As(err, &symErr) && !errors.As(err, &libErr) {
			t.Errorf("GetProperties error = %v (%T), want *SymbolError or *LibraryError", err, err)
		}
	}
	// The asynchronous calls take only device addresses, so calling one with
	// zero arguments cannot dereference anything on the host. It reaches the
	// library only if the library is present, where a null pointer is the
	// library's own error to report.
	if !Loaded() {
		if err := CompressAsync(0, 0, 0, 0, 0, 0, 0, 0, CompressOpts{}, 0, 0); err == nil {
			t.Error("CompressAsync with no library returned nil, want a load error")
		}
	}
}

// TestErrorStrings checks that a Result describes itself without the library.
// Error() must not open libnvcomp: an error value that needs the library to be
// loaded is useless in the case that matters, which is the library failing to
// load.
func TestErrorStrings(t *testing.T) {
	if got := ErrChunkSizeTooLarge.Error(); !strings.Contains(got, "ErrChunkSizeTooLarge") {
		t.Errorf("ErrChunkSizeTooLarge.Error() = %q, want the Go constant name", got)
	}
	if got := Result(9999).Error(); !strings.Contains(got, "9999") {
		t.Errorf("Result(9999).Error() = %q, want the numeric code", got)
	}
	if got := Success.Error(); got == "" {
		t.Error("Success.Error() is empty")
	}
}
