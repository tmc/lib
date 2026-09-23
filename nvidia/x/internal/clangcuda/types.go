package clangcuda

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"strings"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

// File is an unsaved source file visible to libclang during a parse.
type File struct {
	Name     string
	Contents string
}

// Options controls one translation-unit parse.
type Options struct {
	// Name is the main source filename. It defaults to kernel.cu.
	Name string
	// Args are clang arguments excluding the main source filename.
	Args []string
	// Files are additional unsaved files, commonly .cuh headers.
	Files []File
	// Target, when non-empty, is passed as -target.
	Target string
	// Sysroot, when non-empty, is passed as -isysroot.
	Sysroot string
	// Diagnostics requests diagnostics even when parsing succeeds.
	Diagnostics bool
}

// Declaration is a declaration cursor reported by libclang. Kind is the raw
// CXCursorKind value; keeping it raw avoids coupling callers to a partial
// enum mirror while still allowing CUDA-specific rules to classify it.
type Declaration struct {
	Kind         uint32
	Name         string
	Type         string
	File         string
	Line         uint32
	Column       uint32
	IsDefinition bool
}

// Diagnostic is one diagnostic emitted while parsing.
type Diagnostic struct {
	Severity uint32
	Message  string
}

// Unit is the parsed translation unit and its static declarations.
type Unit struct {
	Declarations []Declaration
	Diagnostics  []Diagnostic
}

type cxCursor struct {
	kind  int32
	xdata int32
	data  [3]uintptr
}

type cxType struct {
	kind int32
	pad  int32
	data [2]uintptr
}

type cxString struct {
	data  uintptr
	flags uint32
	pad   uint32
}

type cxSourceLocation struct {
	data [2]uintptr
	int  uint32
	pad  uint32
}

type unsavedFile struct {
	filename *byte
	contents *byte
	length   uint64
}

type clangAPI struct {
	createIndex              func(int32, int32) uintptr
	disposeIndex             func(uintptr)
	parseTranslationUnit2    func(uintptr, *byte, **byte, int32, *unsavedFile, uint32, uint32, *uintptr) uint32
	disposeTranslationUnit   func(uintptr)
	getTranslationUnitCursor func(uintptr) cxCursor
	visitChildren            func(cxCursor, uintptr, unsafe.Pointer) uint32
	isDeclaration            func(uint32) uint32
	getCursorKind            func(cxCursor) uint32
	getCursorSpelling        func(cxCursor) cxString
	getCursorType            func(cxCursor) cxType
	getTypeSpelling          func(cxType) cxString
	cursorIsDefinition       func(cxCursor) uint32
	getCursorLocation        func(cxCursor) cxSourceLocation
	getSpellingLocation      func(cxSourceLocation, *uintptr, *uint32, *uint32, *uint32)
	getFileName              func(uintptr) cxString
	getCString               func(cxString) *byte
	disposeString            func(cxString)
	getNumDiagnostics        func(uintptr) uint32
	getDiagnostic            func(uintptr, uint32) uintptr
	getDiagnosticSeverity    func(uintptr) uint32
	formatDiagnostic         func(uintptr, uint32) cxString
	disposeDiagnostic        func(uintptr)
}

func (a *clangAPI) bind(lib uintptr) error {
	var missing []string
	bind := func(dst any, name string) {
		defer func() {
			if recover() != nil {
				missing = append(missing, name)
			}
		}()
		purego.RegisterLibFunc(dst, lib, name)
	}
	bind(&a.createIndex, "clang_createIndex")
	bind(&a.disposeIndex, "clang_disposeIndex")
	bind(&a.parseTranslationUnit2, "clang_parseTranslationUnit2")
	bind(&a.disposeTranslationUnit, "clang_disposeTranslationUnit")
	bind(&a.getTranslationUnitCursor, "clang_getTranslationUnitCursor")
	bind(&a.visitChildren, "clang_visitChildren")
	bind(&a.isDeclaration, "clang_isDeclaration")
	bind(&a.getCursorKind, "clang_getCursorKind")
	bind(&a.getCursorSpelling, "clang_getCursorSpelling")
	bind(&a.getCursorType, "clang_getCursorType")
	bind(&a.getTypeSpelling, "clang_getTypeSpelling")
	bind(&a.cursorIsDefinition, "clang_isCursorDefinition")
	bind(&a.getCursorLocation, "clang_getCursorLocation")
	bind(&a.getSpellingLocation, "clang_getSpellingLocation")
	bind(&a.getFileName, "clang_getFileName")
	bind(&a.getCString, "clang_getCString")
	bind(&a.disposeString, "clang_disposeString")
	bind(&a.getNumDiagnostics, "clang_getNumDiagnostics")
	bind(&a.getDiagnostic, "clang_getDiagnostic")
	bind(&a.getDiagnosticSeverity, "clang_getDiagnosticSeverity")
	bind(&a.formatDiagnostic, "clang_formatDiagnostic")
	bind(&a.disposeDiagnostic, "clang_disposeDiagnostic")
	if len(missing) != 0 {
		return fmt.Errorf("libclang missing required symbols: %s", strings.Join(missing, ", "))
	}
	return nil
}

// Index owns a libclang index. Parsing on one Index is serialized; separate
// Index values may parse concurrently.
type Index struct {
	lib   uintptr
	raw   uintptr
	api   clangAPI
	mu    sync.Mutex
	close sync.Once
}

// Open loads libclang using platform-specific discovery. An explicit path can
// be supplied to OpenPath when discovery is insufficient.
func Open() (*Index, error) {
	path, err := findLibclang()
	if err != nil {
		return nil, err
	}
	return OpenPath(path)
}

// OpenPath loads libclang from path.
func OpenPath(path string) (*Index, error) {
	if path == "" {
		return nil, errors.New("clangcuda: empty libclang path")
	}
	lib, err := loadLibrary(path)
	if err != nil {
		return nil, fmt.Errorf("clangcuda: open libclang %q: %w", path, err)
	}
	var api clangAPI
	if err := api.bind(lib); err != nil {
		_ = closeLibrary(lib)
		return nil, fmt.Errorf("clangcuda: bind libclang: %w", err)
	}
	raw := api.createIndex(0, 0)
	if raw == 0 {
		_ = closeLibrary(lib)
		return nil, errors.New("clangcuda: clang_createIndex returned nil")
	}
	return &Index{lib: lib, raw: raw, api: api}, nil
}

// Close releases the index and its dynamic library. It is safe to call more
// than once.
func (i *Index) Close() {
	if i == nil {
		return
	}
	i.close.Do(func() {
		i.mu.Lock()
		defer i.mu.Unlock()
		if i.raw != 0 {
			i.api.disposeIndex(i.raw)
			i.raw = 0
		}
		if i.lib != 0 {
			_ = closeLibrary(i.lib)
			i.lib = 0
		}
	})
}

// Parse parses one CUDA translation unit, including unsaved .cuh files.
func (i *Index) Parse(ctx context.Context, source string, options Options) (*Unit, error) {
	if i == nil {
		return nil, errors.New("clangcuda: nil index")
	}
	if ctx == nil {
		return nil, errors.New("clangcuda: nil context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	name := options.Name
	if name == "" {
		name = "kernel.cu"
	}
	files, names, contents, keep, err := makeUnsavedFiles(name, source, options.Files)
	if err != nil {
		return nil, err
	}
	args := append([]string{"-x", "cuda"}, options.Args...)
	if options.Target != "" {
		args = append(args, "-target", options.Target)
	}
	if options.Sysroot != "" {
		args = append(args, "-isysroot", options.Sysroot)
	}
	cargs := make([]*byte, len(args))
	argBytes := make([][]byte, len(args))
	for n := range args {
		argBytes[n] = append([]byte(args[n]), 0)
		cargs[n] = &argBytes[n][0]
	}
	mainName := append([]byte(name), 0)
	var tu uintptr
	i.mu.Lock()
	defer i.mu.Unlock()
	if i.raw == 0 {
		return nil, errors.New("clangcuda: closed index")
	}
	code := i.api.parseTranslationUnit2(i.raw, &mainName[0], &cargs[0], int32(len(cargs)), &files[0], uint32(len(files)), 0, &tu)
	runtime.KeepAlive(names)
	runtime.KeepAlive(contents)
	runtime.KeepAlive(keep)
	runtime.KeepAlive(argBytes)
	if code != 0 || tu == 0 {
		message := i.diagnosticText(tu)
		if tu != 0 {
			i.api.disposeTranslationUnit(tu)
		}
		if message == "" {
			message = fmt.Sprintf("error code %d", code)
		}
		return nil, fmt.Errorf("clangcuda: parse translation unit failed: %s", message)
	}
	defer i.api.disposeTranslationUnit(tu)
	unit := &Unit{Diagnostics: i.diagnostics(tu)}
	if hasErrors(unit.Diagnostics) {
		return unit, fmt.Errorf("clangcuda: parse diagnostics: %s", formatDiagnostics(unit.Diagnostics))
	}
	if options.Diagnostics {
		// Diagnostics are already retained in Unit.Diagnostics.
	}
	state := &visitState{index: i, unit: unit, ctx: ctx}
	token := visitors.add(state)
	defer visitors.delete(token)
	i.api.visitChildren(i.api.getTranslationUnitCursor(tu), visitorCallback, unsafe.Pointer(&token))
	runtime.KeepAlive(token)
	if err := ctx.Err(); err != nil {
		return unit, err
	}
	return unit, nil
}

type keptString struct{ bytes []byte }

func makeUnsavedFiles(mainName, source string, extra []File) ([]unsavedFile, []string, []string, []keptString, error) {
	all := make([]File, 0, len(extra)+1)
	all = append(all, File{Name: mainName, Contents: source})
	all = append(all, extra...)
	files := make([]unsavedFile, 0, len(all))
	names := make([]string, 0, len(all))
	contents := make([]string, 0, len(all))
	keep := make([]keptString, 0, len(all)*2)
	seen := make(map[string]bool, len(all))
	for _, file := range all {
		if file.Name == "" || strings.ContainsRune(file.Name, 0) || strings.ContainsRune(file.Contents, 0) {
			return nil, nil, nil, nil, fmt.Errorf("clangcuda: invalid unsaved file %q", file.Name)
		}
		if seen[file.Name] {
			return nil, nil, nil, nil, fmt.Errorf("clangcuda: duplicate unsaved file %q", file.Name)
		}
		seen[file.Name] = true
		name := append([]byte(file.Name), 0)
		body := append([]byte(file.Contents), 0)
		keep = append(keep, keptString{name}, keptString{body})
		names = append(names, file.Name)
		contents = append(contents, file.Contents)
		files = append(files, unsavedFile{filename: &keep[len(keep)-2].bytes[0], contents: &keep[len(keep)-1].bytes[0], length: uint64(len(body) - 1)})
	}
	return files, names, contents, keep, nil
}

func hasErrors(diagnostics []Diagnostic) bool {
	for _, diagnostic := range diagnostics {
		if diagnostic.Severity >= 3 {
			return true
		}
	}
	return false
}

func formatDiagnostics(diagnostics []Diagnostic) string {
	var messages []string
	for _, diagnostic := range diagnostics {
		messages = append(messages, diagnostic.Message)
	}
	return strings.Join(messages, "; ")
}

func (i *Index) diagnosticText(tu uintptr) string {
	return formatDiagnostics(i.diagnostics(tu))
}

func (i *Index) diagnostics(tu uintptr) []Diagnostic {
	if tu == 0 {
		return nil
	}
	count := i.api.getNumDiagnostics(tu)
	diagnostics := make([]Diagnostic, 0, count)
	for n := uint32(0); n < count; n++ {
		diagnostic := i.api.getDiagnostic(tu, n)
		message := i.stringValue(i.api.formatDiagnostic(diagnostic, 0))
		diagnostics = append(diagnostics, Diagnostic{Severity: i.api.getDiagnosticSeverity(diagnostic), Message: message})
		i.api.disposeDiagnostic(diagnostic)
	}
	return diagnostics
}

func (i *Index) stringValue(value cxString) string {
	defer i.api.disposeString(value)
	p := i.api.getCString(value)
	if p == nil {
		return ""
	}
	return string(unsafe.Slice(p, lenUntilNUL(p)))
}

func lenUntilNUL(p *byte) int {
	for n := 0; ; n++ {
		if *(*byte)(unsafe.Add(unsafe.Pointer(p), n)) == 0 {
			return n
		}
	}
}

type visitState struct {
	index *Index
	unit  *Unit
	ctx   context.Context
}

var visitors = newVisitorRegistry()

type visitorRegistry struct {
	mu     sync.Mutex
	next   uintptr
	values map[uintptr]*visitState
}

func newVisitorRegistry() *visitorRegistry {
	return &visitorRegistry{values: make(map[uintptr]*visitState)}
}

func (r *visitorRegistry) add(value *visitState) uintptr {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.next++
	if r.next == 0 {
		r.next++
	}
	r.values[r.next] = value
	return r.next
}

func (r *visitorRegistry) get(token uintptr) *visitState {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.values[token]
}

func (r *visitorRegistry) delete(token uintptr) {
	r.mu.Lock()
	delete(r.values, token)
	r.mu.Unlock()
}

var visitorCallback = purego.NewCallback(func(cursor cxCursor, _ cxCursor, data unsafe.Pointer) uint32 {
	state := visitors.get(*(*uintptr)(data))
	if state == nil || state.ctx.Err() != nil {
		return 0
	}
	if state.index.api.isDeclaration(state.index.api.getCursorKind(cursor)) == 0 {
		return 1
	}
	kind := state.index.api.getCursorKind(cursor)
	spelling := state.index.stringValue(state.index.api.getCursorSpelling(cursor))
	typeName := state.index.stringValue(state.index.api.getTypeSpelling(state.index.api.getCursorType(cursor)))
	location := state.index.api.getCursorLocation(cursor)
	var file uintptr
	var line, column, offset uint32
	state.index.api.getSpellingLocation(location, &file, &line, &column, &offset)
	fileName := state.index.stringValue(state.index.api.getFileName(file))
	state.unit.Declarations = append(state.unit.Declarations, Declaration{
		Kind: kind, Name: spelling, Type: typeName, File: fileName,
		Line: line, Column: column, IsDefinition: state.index.api.cursorIsDefinition(cursor) != 0,
	})
	return 1
})
