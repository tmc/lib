// Package libclang is a minimal binding of libclang, clang's C interface,
// loaded at run time with purego. It exposes what code generators need to
// read declarations: cursors, types, record layouts, comments and
// diagnostics.
//
// A parse sees what the clang driver would with the same arguments, but the
// headers it finds are those of the libclang it loaded: clang's resource
// directory beside the library, for stddef.h and, in CUDA mode, the CUDA
// wrapper headers. Function bodies are skipped.
package libclang

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/ebitengine/purego"
)

type cxCursor struct {
	kind  CursorKind
	xdata int32
	data  [3]uintptr
}

type cxType struct {
	kind TypeKind
	_    int32
	data [2]uintptr
}

type cxString struct {
	data  uintptr
	flags uint32
	_     uint32
}

type cxSourceLocation struct {
	data [2]uintptr
	int  uint32
	_    uint32
}

type cxUnsavedFile struct {
	name     *byte
	contents *byte
	length   uint64
}

// A Library is a loaded libclang.
type Library struct {
	path string

	getClangVersion                  func() cxString
	getCString                       func(cxString) string
	disposeString                    func(cxString)
	createIndex                      func(int32, int32) uintptr
	disposeIndex                     func(uintptr)
	parseTranslationUnit2            func(uintptr, *byte, **byte, int32, *cxUnsavedFile, uint32, uint32, *uintptr) int32
	disposeTranslationUnit           func(uintptr)
	getTranslationUnitCursor         func(uintptr) cxCursor
	getInclusions                    func(uintptr, uintptr, uintptr)
	getNumDiagnostics                func(uintptr) uint32
	getDiagnostic                    func(uintptr, uint32) uintptr
	getDiagnosticSeverity            func(uintptr) Severity
	formatDiagnostic                 func(uintptr, uint32) cxString
	disposeDiagnostic                func(uintptr)
	visitChildren                    func(cxCursor, uintptr, uintptr) uint32
	getCursorKind                    func(cxCursor) CursorKind
	getCursorSpelling                func(cxCursor) cxString
	getCursorType                    func(cxCursor) cxType
	getCursorResultType              func(cxCursor) cxType
	getCursorDefinition              func(cxCursor) cxCursor
	getCursorLocation                func(cxCursor) cxSourceLocation
	getSpellingLocation              func(cxSourceLocation, *uintptr, *uint32, *uint32, *uint32)
	getFileName                      func(uintptr) cxString
	cursorGetRawCommentText          func(cxCursor) cxString
	cursorGetMangling                func(cxCursor) cxString
	cursorGetNumArguments            func(cxCursor) int32
	cursorGetArgument                func(cxCursor, uint32) cxCursor
	cursorIsBitField                 func(cxCursor) uint32
	cursorGetOffsetOfField           func(cxCursor) int64
	getEnumConstantDeclValue         func(cxCursor) int64
	getEnumDeclIntegerType           func(cxCursor) cxType
	getEnumConstantDeclUnsignedValue func(cxCursor) uint64
	enumDeclIsScoped                 func(cxCursor) uint32
	cursorIsMacroFunctionLike        func(cxCursor) uint32
	cursorEvaluate                   func(cxCursor) uintptr
	evalResultGetKind                func(uintptr) int32
	evalResultIsUnsignedInt          func(uintptr) uint32
	evalResultGetAsLongLong          func(uintptr) int64
	evalResultGetAsUnsigned          func(uintptr) uint64
	evalResultGetAsDouble            func(uintptr) float64
	evalResultDispose                func(uintptr)
	getTypedefDeclUnderlyingType     func(cxCursor) cxType
	getTypeSpelling                  func(cxType) cxString
	getCanonicalType                 func(cxType) cxType
	getPointeeType                   func(cxType) cxType
	getArrayElementType              func(cxType) cxType
	getArraySize                     func(cxType) int64
	getTypeDeclaration               func(cxType) cxCursor
	typeGetSizeOf                    func(cxType) int64
	typeGetAlignOf                   func(cxType) int64
}

// Open loads the libclang at path. An empty path searches the libraries of
// llvm-config and of clang on $PATH, then the LLVM installations in
// /usr/lib, newest first, and prefers one with a resource directory, since
// without one clang cannot find even stddef.h.
func Open(path string) (*Library, error) {
	if path != "" {
		return open(path)
	}
	var libs []*Library
	var errs []error
	for _, p := range candidates() {
		l, err := open(p)
		if err != nil {
			errs = append(errs, err)
			continue
		}
		if l.ResourceDir() != "" {
			return l, nil
		}
		libs = append(libs, l)
	}
	if len(libs) > 0 {
		return libs[0], nil
	}
	if len(errs) == 0 {
		return nil, errors.New("libclang: not found")
	}
	return nil, fmt.Errorf("libclang: not found: %w", errors.Join(errs...))
}

func candidates() []string {
	var paths []string
	if out, err := exec.Command("llvm-config", "--libdir").Output(); err == nil {
		paths = append(paths, filepath.Join(strings.TrimSpace(string(out)), "libclang.so"))
	}
	if out, err := exec.Command("clang", "-print-file-name=libclang.so").Output(); err == nil {
		paths = append(paths, strings.TrimSpace(string(out)))
	}
	llvm, _ := filepath.Glob("/usr/lib/llvm-*/lib/libclang.so*")
	slices.SortFunc(llvm, func(a, b string) int {
		return llvmVersion(b) - llvmVersion(a)
	})
	paths = append(paths, llvm...)
	paths = append(paths, "/usr/local/lib/libclang.so")
	var found []string
	for _, p := range paths {
		if _, err := os.Stat(p); err == nil && filepath.IsAbs(p) && !slices.Contains(found, p) {
			found = append(found, p)
		}
	}
	return found
}

// llvmVersion returns the N of a path in /usr/lib/llvm-N.
func llvmVersion(path string) int {
	s, _ := strings.CutPrefix(path, "/usr/lib/llvm-")
	s, _, _ = strings.Cut(s, "/")
	n, _ := strconv.Atoi(s)
	return n
}

func open(path string) (l *Library, err error) {
	h, err := purego.Dlopen(path, purego.RTLD_NOW|purego.RTLD_LOCAL)
	if err != nil {
		return nil, fmt.Errorf("libclang: %w", err)
	}
	defer func() {
		if r := recover(); r != nil {
			purego.Dlclose(h)
			l, err = nil, fmt.Errorf("libclang: %s: %v", path, r)
		}
	}()
	l = &Library{path: path}
	for _, f := range []struct {
		fn   any
		name string
	}{
		{&l.getClangVersion, "clang_getClangVersion"},
		{&l.getCString, "clang_getCString"},
		{&l.disposeString, "clang_disposeString"},
		{&l.createIndex, "clang_createIndex"},
		{&l.disposeIndex, "clang_disposeIndex"},
		{&l.parseTranslationUnit2, "clang_parseTranslationUnit2"},
		{&l.disposeTranslationUnit, "clang_disposeTranslationUnit"},
		{&l.getTranslationUnitCursor, "clang_getTranslationUnitCursor"},
		{&l.getInclusions, "clang_getInclusions"},
		{&l.getNumDiagnostics, "clang_getNumDiagnostics"},
		{&l.getDiagnostic, "clang_getDiagnostic"},
		{&l.getDiagnosticSeverity, "clang_getDiagnosticSeverity"},
		{&l.formatDiagnostic, "clang_formatDiagnostic"},
		{&l.disposeDiagnostic, "clang_disposeDiagnostic"},
		{&l.visitChildren, "clang_visitChildren"},
		{&l.getCursorKind, "clang_getCursorKind"},
		{&l.getCursorSpelling, "clang_getCursorSpelling"},
		{&l.getCursorType, "clang_getCursorType"},
		{&l.getCursorResultType, "clang_getCursorResultType"},
		{&l.getCursorDefinition, "clang_getCursorDefinition"},
		{&l.getCursorLocation, "clang_getCursorLocation"},
		{&l.getSpellingLocation, "clang_getSpellingLocation"},
		{&l.getFileName, "clang_getFileName"},
		{&l.cursorGetRawCommentText, "clang_Cursor_getRawCommentText"},
		{&l.cursorGetMangling, "clang_Cursor_getMangling"},
		{&l.cursorGetNumArguments, "clang_Cursor_getNumArguments"},
		{&l.cursorGetArgument, "clang_Cursor_getArgument"},
		{&l.cursorIsBitField, "clang_Cursor_isBitField"},
		{&l.cursorGetOffsetOfField, "clang_Cursor_getOffsetOfField"},
		{&l.getEnumConstantDeclValue, "clang_getEnumConstantDeclValue"},
		{&l.getEnumDeclIntegerType, "clang_getEnumDeclIntegerType"},
		{&l.getEnumConstantDeclUnsignedValue, "clang_getEnumConstantDeclUnsignedValue"},
		{&l.enumDeclIsScoped, "clang_EnumDecl_isScoped"},
		{&l.cursorIsMacroFunctionLike, "clang_Cursor_isMacroFunctionLike"},
		{&l.cursorEvaluate, "clang_Cursor_Evaluate"},
		{&l.evalResultGetKind, "clang_EvalResult_getKind"},
		{&l.evalResultIsUnsignedInt, "clang_EvalResult_isUnsignedInt"},
		{&l.evalResultGetAsLongLong, "clang_EvalResult_getAsLongLong"},
		{&l.evalResultGetAsUnsigned, "clang_EvalResult_getAsUnsigned"},
		{&l.evalResultGetAsDouble, "clang_EvalResult_getAsDouble"},
		{&l.evalResultDispose, "clang_EvalResult_dispose"},
		{&l.getTypedefDeclUnderlyingType, "clang_getTypedefDeclUnderlyingType"},
		{&l.getTypeSpelling, "clang_getTypeSpelling"},
		{&l.getCanonicalType, "clang_getCanonicalType"},
		{&l.getPointeeType, "clang_getPointeeType"},
		{&l.getArrayElementType, "clang_getArrayElementType"},
		{&l.getArraySize, "clang_getArraySize"},
		{&l.getTypeDeclaration, "clang_getTypeDeclaration"},
		{&l.typeGetSizeOf, "clang_Type_getSizeOf"},
		{&l.typeGetAlignOf, "clang_Type_getAlignOf"},
	} {
		purego.RegisterLibFunc(f.fn, h, f.name)
	}
	return l, nil
}

// Path returns the path the library was loaded from.
func (l *Library) Path() string {
	return l.path
}

// Version returns clang's version string, such as "clang version 21.0.0".
func (l *Library) Version() string {
	return l.string(l.getClangVersion())
}

func (l *Library) string(s cxString) string {
	defer l.disposeString(s)
	return l.getCString(s)
}

// An UnsavedFile is a file that parsing reads from memory rather than disk.
type UnsavedFile struct {
	Name     string
	Contents string
}

// A TranslationUnit is a parsed source file.
type TranslationUnit struct {
	lib   *Library
	index uintptr
	tu    uintptr
}

const (
	parseDetailedPreprocessingRecord = 0x01 // for MacroDefinition cursors
	parseSkipFunctionBodies          = 0x40
	parseKeepGoing                   = 0x200
)

// Parse parses file with the clang command-line arguments args. The
// unsaved files shadow or add to the files on disk. Parse fails only if
// clang produces no translation unit; a parse with errors succeeds and
// reports them in [TranslationUnit.Diagnostics].
func (l *Library) Parse(file string, args []string, unsaved ...UnsavedFile) (*TranslationUnit, error) {
	var keep [][]byte
	cstr := func(s string) *byte {
		b := append([]byte(s), 0)
		keep = append(keep, b)
		return &b[0]
	}
	argv := make([]*byte, len(args)+1)
	for i, a := range args {
		argv[i] = cstr(a)
	}
	files := make([]cxUnsavedFile, len(unsaved)+1)
	for i, u := range unsaved {
		files[i] = cxUnsavedFile{cstr(u.Name), cstr(u.Contents), uint64(len(u.Contents))}
	}
	index := l.createIndex(0, 0)
	var tu uintptr
	code := l.parseTranslationUnit2(index, cstr(file), &argv[0], int32(len(args)), &files[0], uint32(len(unsaved)),
		parseDetailedPreprocessingRecord|parseSkipFunctionBodies|parseKeepGoing, &tu)
	runtime.KeepAlive(keep)
	runtime.KeepAlive(argv)
	runtime.KeepAlive(files)
	if code != 0 || tu == 0 {
		l.disposeIndex(index)
		return nil, fmt.Errorf("libclang: parse %s: error %d", file, code)
	}
	return &TranslationUnit{l, index, tu}, nil
}

// Close frees the translation unit. Its cursors and types are invalid
// afterward.
func (t *TranslationUnit) Close() {
	t.lib.disposeTranslationUnit(t.tu)
	t.lib.disposeIndex(t.index)
}

// Cursor returns the cursor of the translation unit itself.
func (t *TranslationUnit) Cursor() Cursor {
	return Cursor{t.lib, t.lib.getTranslationUnitCursor(t.tu)}
}

// Files returns the names of the files the translation unit read: the
// main file and every file it included, directly or not.
func (t *TranslationUnit) Files() []string {
	id, v := newVisit()
	defer visits.Delete(id)
	t.lib.getInclusions(t.tu, includer(), id)
	names := make([]string, len(v.files))
	for i, f := range v.files {
		names[i] = t.lib.string(t.lib.getFileName(f))
	}
	return names
}

// Options of clang_formatDiagnostic.
const (
	displaySourceLocation = 0x1
	displayColumn         = 0x2
	displaySourceRanges   = 0x4
)

// Severity is the severity of a diagnostic.
type Severity int32

const (
	Ignored Severity = iota
	Note
	Warning
	Error
	Fatal
)

// A Diagnostic is a message clang produced while parsing.
type Diagnostic struct {
	Severity Severity
	Text     string // formatted as clang prints it, with its location
}

// Diagnostics returns the diagnostics of the parse.
func (t *TranslationUnit) Diagnostics() []Diagnostic {
	l := t.lib
	n := l.getNumDiagnostics(t.tu)
	var ds []Diagnostic
	for i := range n {
		d := l.getDiagnostic(t.tu, i)
		ds = append(ds, Diagnostic{l.getDiagnosticSeverity(d), l.string(l.formatDiagnostic(d, displaySourceLocation|displayColumn|displaySourceRanges))})
		l.disposeDiagnostic(d)
	}
	return ds
}

// A Cursor is a node of the syntax tree.
type Cursor struct {
	lib *Library
	c   cxCursor
}

// Kind returns the kind of the cursor.
func (c Cursor) Kind() CursorKind {
	return c.lib.getCursorKind(c.c)
}

// Spelling returns the name of the declaration or reference.
func (c Cursor) Spelling() string {
	return c.lib.string(c.lib.getCursorSpelling(c.c))
}

// Type returns the type of the declaration.
func (c Cursor) Type() Type {
	return Type{c.lib, c.lib.getCursorType(c.c)}
}

// ResultType returns the result type of a function declaration.
func (c Cursor) ResultType() Type {
	return Type{c.lib, c.lib.getCursorResultType(c.c)}
}

// TypedefUnderlyingType returns the type a typedef names.
func (c Cursor) TypedefUnderlyingType() Type {
	return Type{c.lib, c.lib.getTypedefDeclUnderlyingType(c.c)}
}

// Definition returns the definition of the declared entity, or a null
// cursor if the translation unit has none.
func (c Cursor) Definition() Cursor {
	return Cursor{c.lib, c.lib.getCursorDefinition(c.c)}
}

// IsNull reports whether c is the null cursor.
func (c Cursor) IsNull() bool {
	return c.lib == nil || c.Kind() == cursorInvalidFile
}

// Location returns the file, line and column where c is spelled.
func (c Cursor) Location() (file string, line, col int) {
	var f uintptr
	var ln, cl, off uint32
	c.lib.getSpellingLocation(c.lib.getCursorLocation(c.c), &f, &ln, &cl, &off)
	if f != 0 {
		file = c.lib.string(c.lib.getFileName(f))
	}
	return file, int(ln), int(cl)
}

// RawComment returns the documentation comment attached to the
// declaration, with its comment markers.
func (c Cursor) RawComment() string {
	return c.lib.string(c.lib.cursorGetRawCommentText(c.c))
}

// Mangling returns the symbol name of a declaration with linkage.
func (c Cursor) Mangling() string {
	return c.lib.string(c.lib.cursorGetMangling(c.c))
}

// Arguments returns the parameters of a function declaration.
func (c Cursor) Arguments() []Cursor {
	n := c.lib.cursorGetNumArguments(c.c)
	args := make([]Cursor, max(n, 0))
	for i := range args {
		args[i] = Cursor{c.lib, c.lib.cursorGetArgument(c.c, uint32(i))}
	}
	return args
}

// IsBitField reports whether c is a bit-field.
func (c Cursor) IsBitField() bool {
	return c.lib.cursorIsBitField(c.c) != 0
}

// OffsetOfField returns the offset in bits of a field in its record, or a
// negative value if it has none.
func (c Cursor) OffsetOfField() int64 {
	return c.lib.cursorGetOffsetOfField(c.c)
}

// EnumValue returns the value of an enum constant.
func (c Cursor) EnumValue() int64 {
	return c.lib.getEnumConstantDeclValue(c.c)
}

// EnumUnsignedValue returns an unsigned enum constant's value.
func (c Cursor) EnumUnsignedValue() uint64 {
	return c.lib.getEnumConstantDeclUnsignedValue(c.c)
}

// EnumIntegerType returns the integer type underlying an enum declaration.
func (c Cursor) EnumIntegerType() Type {
	return Type{c.lib, c.lib.getEnumDeclIntegerType(c.c)}
}

// IsScoped reports whether an enum is a scoped enum, an enum class.
func (c Cursor) IsScoped() bool {
	return c.lib.enumDeclIsScoped(c.c) != 0
}

// IsMacroFunctionLike reports whether a macro definition takes arguments.
func (c Cursor) IsMacroFunctionLike() bool {
	return c.lib.cursorIsMacroFunctionLike(c.c) != 0
}

// Evaluate returns the value of c, an expression or a variable with an
// initializer, if it is a constant: an int64, a uint64 if it has an
// unsigned type, or a float64.
func (c Cursor) Evaluate() (any, bool) {
	r := c.lib.cursorEvaluate(c.c)
	if r == 0 {
		return nil, false
	}
	defer c.lib.evalResultDispose(r)
	switch c.lib.evalResultGetKind(r) {
	case evalInt:
		if c.lib.evalResultIsUnsignedInt(r) != 0 {
			return c.lib.evalResultGetAsUnsigned(r), true
		}
		return c.lib.evalResultGetAsLongLong(r), true
	case evalFloat:
		return c.lib.evalResultGetAsDouble(r), true
	}
	return nil, false
}

// Kinds of CXEvalResult.
const (
	evalInt   = 1
	evalFloat = 2
)

// Children returns the children of c.
func (c Cursor) Children() []Cursor {
	id, v := newVisit()
	defer visits.Delete(id)
	c.lib.visitChildren(c.c, visitor(), id)
	for i := range v.cursors {
		v.cursors[i].lib = c.lib
	}
	return v.cursors
}

// A visit collects what clang_visitChildren or clang_getInclusions passes
// its visitor. Each visitor is a single callback, since purego allots a
// fixed number, and the client data selects the visit.
type visit struct {
	cursors []Cursor
	files   []uintptr
}

var (
	visits    sync.Map // uintptr → *visit
	visitNext atomic.Uintptr

	visitor = sync.OnceValue(func() uintptr {
		return purego.NewCallback(func(c, _ cxCursor, id uintptr) uint32 {
			v, _ := visits.Load(id)
			v.(*visit).cursors = append(v.(*visit).cursors, Cursor{c: c})
			return 1 // CXChildVisit_Continue
		})
	})

	includer = sync.OnceValue(func() uintptr {
		return purego.NewCallback(func(file, _ uintptr, _ uint32, id uintptr) {
			v, _ := visits.Load(id)
			v.(*visit).files = append(v.(*visit).files, file)
		})
	})
)

func newVisit() (uintptr, *visit) {
	id, v := visitNext.Add(1), new(visit)
	visits.Store(id, v)
	return id, v
}

// A Type is a C or C++ type.
type Type struct {
	lib *Library
	t   cxType
}

// Kind returns the kind of the type.
func (t Type) Kind() TypeKind {
	return t.t.kind
}

// Spelling returns the type as clang prints it.
func (t Type) Spelling() string {
	return t.lib.string(t.lib.getTypeSpelling(t.t))
}

// Canonical returns the type with typedefs and elaboration removed.
func (t Type) Canonical() Type {
	return Type{t.lib, t.lib.getCanonicalType(t.t)}
}

// Pointee returns the type a pointer or reference points to.
func (t Type) Pointee() Type {
	return Type{t.lib, t.lib.getPointeeType(t.t)}
}

// Elem returns the element type of an array.
func (t Type) Elem() Type {
	return Type{t.lib, t.lib.getArrayElementType(t.t)}
}

// Len returns the length of a constant array, or -1.
func (t Type) Len() int64 {
	return t.lib.getArraySize(t.t)
}

// Declaration returns the declaration of a record, enum or typedef type.
func (t Type) Declaration() Cursor {
	return Cursor{t.lib, t.lib.getTypeDeclaration(t.t)}
}

// Size returns the size of the type in bytes, or a negative
// libclang error code for a type without one.
func (t Type) Size() int64 {
	return t.lib.typeGetSizeOf(t.t)
}

// Align returns the alignment of the type in bytes, or a negative
// libclang error code.
func (t Type) Align() int64 {
	return t.lib.typeGetAlignOf(t.t)
}

// ResourceDir returns clang's resource directory beside the library, which
// holds the headers clang supplies, or "" if there is none. Parsing needs
// it passed as -resource-dir.
func (l *Library) ResourceDir() string {
	// Distribution packages can put the library in a multiarch directory
	// and a symlink beside its resource headers. Try that installation
	// directory before following the symlink.
	paths := []string{l.path}
	if real, err := filepath.EvalSymlinks(l.path); err == nil && real != l.path {
		paths = append(paths, real)
	}
	for _, path := range paths {
		dirs, _ := filepath.Glob(filepath.Join(filepath.Dir(path), "clang", "*", "include", "stddef.h"))
		if len(dirs) > 0 {
			return filepath.Dir(filepath.Dir(dirs[len(dirs)-1]))
		}
	}
	return ""
}
