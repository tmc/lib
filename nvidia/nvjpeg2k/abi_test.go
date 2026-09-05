package nvjpeg2k

import (
	"bufio"
	"os"
	"reflect"
	"regexp"
	"strings"
	"testing"
	"unsafe"
)

// These tests are the ABI evidence that can be gathered without a GPU or the
// shared library. They check the shapes this package hands to the C ABI --
// enumeration values, struct field types, opaque handle widths, and the set of
// symbols bound -- against the pinned header and the pinned export list.
//
// They cannot check anything that only a real call reveals: that the loaded
// library's calling convention matches, that an out-parameter is written, or
// that a decode produces correct pixels. See physical_test.go.

// TestResultValues pins the status codes against nvjpeg2k.h. A shifted status
// enumeration would turn every error into the wrong error, and Success into a
// silent failure.
func TestResultValues(t *testing.T) {
	want := map[Result]int32{
		Success:                       0,
		ErrNotInitialized:             1,
		ErrInvalidParameter:           2,
		ErrBadJpeg:                    3,
		ErrJpegNotSupported:           4,
		ErrAllocatorFailure:           5,
		ErrExecutionFailed:            6,
		ErrArchMismatch:               7,
		ErrInternalError:              8,
		ErrImplementationNotSupported: 9,
	}
	for r, code := range want {
		if got := r.Code(); got != code {
			t.Errorf("%v.Code() = %d, want %d", r, got, code)
		}
	}
	if toError(Success) != nil {
		t.Error("toError(Success) is not nil")
	}
	if toError(ErrBadJpeg) == nil {
		t.Error("toError(ErrBadJpeg) is nil")
	}
}

// TestEnumValues pins the enumerations a caller passes by value into the
// library. A wrong value here is accepted by the C ABI and misinterpreted.
func TestEnumValues(t *testing.T) {
	cases := []struct {
		name string
		got  int32
		want int32
	}{
		{"NVJPEG2K_BACKEND_DEFAULT", int32(BackendDefault), 0},

		{"NVJPEG2K_COLORSPACE_NOT_SUPPORTED", int32(ColorspaceNotSupported), -1},
		{"NVJPEG2K_COLORSPACE_UNKNOWN", int32(ColorspaceUnknown), 0},

		{"NVJPEG2K_NLT_NONE", int32(NltNone), 0},
		{"NVJPEG2K_NLT_TYPE3", int32(NltType3), 3},

		{"NVJPEG2K_UINT8", int32(Uint8), 0},
		{"NVJPEG2K_UINT16", int32(Uint16), 1},
		{"NVJPEG2K_INT16", int32(Int16), 2},
		{"NVJPEG2K_FP16", int32(FP16), 3},
		{"NVJPEG2K_FP32", int32(FP32), 4},

		{"NVJPEG2K_FORMAT_PLANAR", int32(FormatPlanar), 0},
		{"NVJPEG2K_FORMAT_INTERLEAVED", int32(FormatInterleaved), 1},

		{"MAJOR_VERSION", int32(MajorVersion), 0},
		{"MINOR_VERSION", int32(MinorVersion), 1},
		{"PATCH_LEVEL", int32(PatchLevel), 2},
	}
	for _, c := range cases {
		if c.got != c.want {
			t.Errorf("%s = %d, want %d", c.name, c.got, c.want)
		}
	}
	// Every enumeration crosses the ABI as a C enum, which clang lays out as
	// a 4-byte int for these headers.
	for _, v := range []any{BackendDefault, ColorspaceUnknown, NltNone, Uint8, FormatPlanar, MajorVersion, Success} {
		if got := reflect.TypeOf(v).Size(); got != 4 {
			t.Errorf("sizeof(%T) = %d, want 4", v, got)
		}
	}
}

// TestHandleWidth checks that every opaque handle is pointer-sized. They are
// passed by value, so a narrower Go type would truncate.
func TestHandleWidth(t *testing.T) {
	ptr := unsafe.Sizeof(uintptr(0))
	for _, v := range []any{Handle(0), DecodeState(0), Bitstream(0), DecodeParams(0), Stream(0)} {
		if got := reflect.TypeOf(v).Size(); got != ptr {
			t.Errorf("sizeof(%T) = %d, want %d", v, got, ptr)
		}
	}
}

// TestImageFieldTypes checks the field types of nvjpeg2kImage_t, the one
// struct this package fills in and hands to the library by pointer. The
// offsets and total size are checked by the generated layout test; this checks
// that the two array fields are still pointers to the element types
// [Layout.Bind] writes.
func TestImageFieldTypes(t *testing.T) {
	typ := reflect.TypeOf(Image{})
	want := map[string]string{
		"PixelData":     "*uintptr",
		"PitchInBytes":  "*uint",
		"PixelType":     "nvjpeg2k.ImageType",
		"NumComponents": "uint32",
	}
	if typ.NumField() != len(want) {
		t.Fatalf("Image has %d fields, want %d", typ.NumField(), len(want))
	}
	for i := 0; i < typ.NumField(); i++ {
		f := typ.Field(i)
		if got := f.Type.String(); got != want[f.Name] {
			t.Errorf("Image.%s is %s, want %s", f.Name, got, want[f.Name])
		}
	}
	// size_t is what the header uses for the pitch array; on every platform
	// this package supports that is pointer-sized, which is what uint is.
	if unsafe.Sizeof(uint(0)) != unsafe.Sizeof(uintptr(0)) {
		t.Errorf("uint is %d bytes and uintptr is %d; the pitch array does not match size_t",
			unsafe.Sizeof(uint(0)), unsafe.Sizeof(uintptr(0)))
	}
}

// TestMetadataStructFieldTypes checks the structs the library writes metadata
// into. These are the values a caller sizes device allocations from, so a
// widened or narrowed field silently changes what a decode is asked to write.
func TestMetadataStructFieldTypes(t *testing.T) {
	cases := []struct {
		typ  reflect.Type
		want map[string]string
	}{
		{reflect.TypeOf(ImageInfo{}), map[string]string{
			"ImageWidth": "uint32", "ImageHeight": "uint32",
			"TileWidth": "uint32", "TileHeight": "uint32",
			"NumTilesX": "uint32", "NumTilesY": "uint32",
			"NumComponents": "uint32",
		}},
		{reflect.TypeOf(ImageComponentInfo{}), map[string]string{
			"ComponentWidth": "uint32", "ComponentHeight": "uint32",
			"Precision": "uint8", "Sgn": "uint8",
		}},
		{reflect.TypeOf(NLTInfo{}), map[string]string{
			"Type": "nvjpeg2k.NLTType", "BitDepth": "uint8",
		}},
	}
	for _, c := range cases {
		if c.typ.NumField() != len(c.want) {
			t.Errorf("%s has %d fields, want %d", c.typ, c.typ.NumField(), len(c.want))
			continue
		}
		for i := 0; i < c.typ.NumField(); i++ {
			f := c.typ.Field(i)
			if got := f.Type.String(); got != c.want[f.Name] {
				t.Errorf("%s.%s is %s, want %s", c.typ, f.Name, got, c.want[f.Name])
			}
		}
	}
}

// TestHeaderVersion pins the header the bindings were generated from. The
// soname the loader probes is derived from the major version, so a header bump
// that changed it would change which library is opened.
func TestHeaderVersion(t *testing.T) {
	if HeaderVersionMajor != 0 || HeaderVersionMinor != 11 || HeaderVersionPatch != 0 || HeaderVersionBuild != 51 {
		t.Errorf("header version %d.%d.%d.%d, want 0.11.0.51",
			HeaderVersionMajor, HeaderVersionMinor, HeaderVersionPatch, HeaderVersionBuild)
	}
	if MaxResolutions != 33 {
		t.Errorf("MaxResolutions = %d, want 33 (NVJPEG2K_MAXRES)", MaxResolutions)
	}
}

// TestLibraryNames checks that the linux names the loader tries include the
// soname recorded in the pinned library, libnvjpeg2k.so.0. Probing the wrong
// soname is indistinguishable from the library not being installed.
func TestLibraryNames(t *testing.T) {
	got := linuxLibraryNames(t)
	if len(got) == 0 || got[0] != "libnvjpeg2k.so.0" {
		t.Errorf("linux library names = %v, want the pinned soname libnvjpeg2k.so.0 first", got)
	}
}

var reRegister = regexp.MustCompile(`register\(&fn[A-Za-z0-9_]+, "([A-Za-z0-9_]+)"\)`)

// boundSymbols returns the symbols the generated loader registers, read from
// the generated source rather than from a loaded library, so the test runs
// with no library present.
func boundSymbols(t *testing.T) map[string]bool {
	t.Helper()
	src, err := os.ReadFile("init.gen.go")
	if err != nil {
		t.Fatalf("read init.gen.go: %v", err)
	}
	got := map[string]bool{}
	for _, m := range reRegister.FindAllStringSubmatch(string(src), -1) {
		got[m[1]] = true
	}
	if len(got) == 0 {
		t.Fatal("no register() calls found in init.gen.go; the loader shape changed")
	}
	return got
}

func linuxLibraryNames(t *testing.T) []string {
	t.Helper()
	src, err := os.ReadFile("init.gen.go")
	if err != nil {
		t.Fatalf("read init.gen.go: %v", err)
	}
	re := regexp.MustCompile(`case "linux":\s*\n\s*return \[\]string\{([^}]*)\}`)
	m := re.FindStringSubmatch(string(src))
	if m == nil {
		t.Fatal("no linux case found in libraryNames; the loader shape changed")
	}
	var names []string
	for _, f := range strings.Split(m[1], ",") {
		f = strings.TrimSpace(f)
		if f == "" {
			continue
		}
		names = append(names, strings.Trim(f, `"`))
	}
	return names
}

// pinnedExports reads testdata/exported_symbols.txt, the dynamic symbol table
// of the pinned libnvjpeg2k. The second field marks encode-only symbols.
func pinnedExports(t *testing.T) (all map[string]bool, encode map[string]bool) {
	t.Helper()
	f, err := os.Open("testdata/exported_symbols.txt")
	if err != nil {
		t.Fatalf("read pinned exports: %v", err)
	}
	defer f.Close()
	all, encode = map[string]bool{}, map[string]bool{}
	s := bufio.NewScanner(f)
	for s.Scan() {
		line := strings.TrimSpace(s.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		fields := strings.Fields(line)
		all[fields[0]] = true
		if len(fields) > 1 && fields[1] == "encode" {
			encode[fields[0]] = true
		}
	}
	if err := s.Err(); err != nil {
		t.Fatalf("scan pinned exports: %v", err)
	}
	if len(all) == 0 {
		t.Fatal("pinned export list is empty")
	}
	return all, encode
}

// TestBoundSymbolsAreExported checks that every symbol this package looks up
// is actually exported by the pinned libnvjpeg2k. A typo, or a symbol that
// only ever existed in the header, would otherwise show up as a *SymbolError
// at the first call on a correctly installed library.
func TestBoundSymbolsAreExported(t *testing.T) {
	bound := boundSymbols(t)
	exported, _ := pinnedExports(t)
	for sym := range bound {
		if !exported[sym] {
			t.Errorf("package binds %s, which the pinned libnvjpeg2k does not export", sym)
		}
	}
}

// TestUnboundSymbolsAreEncodeOnly checks that the only exported symbols this
// package leaves unbound are the encode entry points. It is the coverage claim
// the package documentation makes, checked rather than asserted: any new
// decode-side export shows up here as a failure.
func TestUnboundSymbolsAreEncodeOnly(t *testing.T) {
	bound := boundSymbols(t)
	exported, encode := pinnedExports(t)
	for sym := range exported {
		if bound[sym] || encode[sym] {
			continue
		}
		t.Errorf("pinned libnvjpeg2k exports %s, which is neither bound nor marked encode-only", sym)
	}
	for sym := range encode {
		if bound[sym] {
			t.Errorf("%s is marked encode-only but the package binds it", sym)
		}
	}
}
