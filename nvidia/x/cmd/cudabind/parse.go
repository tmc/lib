package main

import (
	"errors"
	"fmt"
	"go/token"
	"go/types"
	"math"
	"os"
	"path"
	"path/filepath"
	"slices"
	"strconv"
	"strings"

	"github.com/tmc/lib/nvidia/x/internal/libclang"
)

// A module is what cudabind binds: the kernels and variables of a
// translation unit and the C types they use.
type module struct {
	files   []string // files to embed, relative to the package directory
	kernels []*kernel
	globals []*global
	consts  []*constant
	records []*record // Go structs, each after the records it contains
	mapped  []mapped  // toolkit types mapped by shape
}

type kernel struct {
	name   string // C name, qualified for C++ linkage
	method string
	cxx    bool // has C++ linkage, so is found by name expression
	decl   string
	doc    string
	params []param
	shared bool
}

type param struct {
	name  string
	typ   string
	slice bool // a pointer, passed as a cudamem.Slice
}

type global struct {
	name   string
	getter string
	setter string
	decl   string
	doc    string
	typ    goType
}

// A constant is an enum constant or a macro whose value is a number.
type constant struct {
	cName  string // qualified
	goName string
	value  string // a Go literal
	float  bool
	check  bool // the generated test can check it
}

// A record is a Go type generated for a C struct.
type record struct {
	cName  string // as C source spells it
	goName string
	size   int64
	opaque bool // an array of size bytes
	fields []field
	decl   string // where C declares it
}

type field struct {
	cName  string // "" for padding
	goName string
	typ    string
	off    int64
}

// A goType is a Go type with the size and alignment Go gives it.
type goType struct {
	expr        string
	size, align int64
}

// A mapped type is a toolkit type mapped by its shape.
type mapped struct {
	cName string
	typ   goType
}

// parse parses cfg.file with libclang.
func parse(cfg *config) (*module, error) {
	lib, err := libclang.Open(cfg.libclang)
	if err != nil {
		return nil, err
	}
	res := lib.ResourceDir()
	if _, err := os.Stat(filepath.Join(res, "include", "__clang_cuda_runtime_wrapper.h")); res == "" || err != nil {
		return nil, fmt.Errorf("%s (%s) has no CUDA headers in its resource directory; "+
			"install clang's (such as libclang-common-N-dev) or set -libclang or $CUDABIND_LIBCLANG to a libclang that has them",
			lib.Path(), lib.Version())
	}
	args := []string{
		"-x", "cuda", "--cuda-device-only", "--cuda-gpu-arch=" + cfg.arch, "-nocudalib",
		"-fparse-all-comments", "-Wno-unknown-cuda-version", "-resource-dir", res,
	}
	if p := os.Getenv("CUDA_PATH"); p != "" {
		args = append(args, "--cuda-path="+p)
	}
	if cfg.std != "" {
		args = append(args, "-std="+cfg.std)
	}
	for _, d := range cfg.defines {
		args = append(args, "-D"+d)
	}
	for _, dir := range cfg.includes {
		args = append(args, "-idirafter", filepath.Join(cfg.dir, dir))
	}
	tu, err := lib.Parse(filepath.Join(cfg.dir, cfg.file), args)
	if err != nil {
		return nil, err
	}
	defer tu.Close()
	var errs []string
	for _, d := range tu.Diagnostics() {
		if d.Severity >= libclang.Error {
			errs = append(errs, d.Text)
		}
	}
	if len(errs) > 0 {
		return nil, fmt.Errorf("%s with %s:\n%s", cfg.file, lib.Version(), strings.Join(errs, "\n"))
	}

	b := &binder{
		cfg:      cfg,
		m:        new(module),
		typedefs: make(map[string]string),
		types:    make(map[string]goType),
		names:    make(map[string]string),
		consts:   make(map[string]bool),
	}
	var decls []decl
	b.walk(tu.Cursor(), "", &decls)
	b.errs = append(b.errs, b.bind(decls), b.evalMacros(lib, args))
	if err := errors.Join(b.errs...); err != nil {
		return nil, err
	}
	if b.m.files, err = b.files(tu.Files()); err != nil {
		return nil, err
	}
	return b.m, nil
}

type binder struct {
	cfg      *config
	m        *module
	typedefs map[string]string // location of an unnamed record → its typedef name
	types    map[string]goType // location of a record → its Go type
	names    map[string]string // Go name → the C name it came from
	consts   map[string]bool   // enum constants and macros bound
	macros   []string          // macros to evaluate
	errs     []error           // from walk
}

// A decl is a kernel or variable declared in the package directory.
type decl struct {
	c     libclang.Cursor
	name  string // qualified
	isVar bool
}

// walk collects the kernels and variables under c, which is in namespace
// ns, and the typedef names of unnamed records.
func (b *binder) walk(c libclang.Cursor, ns string, decls *[]decl) {
	for _, k := range c.Children() {
		switch k.Kind() {
		case libclang.Namespace:
			b.walk(k, ns+k.Spelling()+"::", decls)
		case libclang.LinkageSpec:
			b.walk(k, ns, decls)
		case libclang.TypedefDecl, libclang.TypeAliasDecl:
			d := k.TypedefUnderlyingType().Canonical().Declaration()
			if !d.IsNull() && unnamed(d.Spelling()) {
				b.typedefs[location(d)] = ns + k.Spelling()
			}
		case libclang.EnumDecl:
			if b.user(k) {
				scope := ns
				if k.IsScoped() {
					scope += k.Spelling() + "::"
				}
				b.enum(k, scope)
			}
		case libclang.MacroDefinition:
			name := k.Spelling()
			if b.user(k) && !k.IsMacroFunctionLike() && !b.consts[name] &&
				(len(b.cfg.macros) == 0 || slices.Contains(b.cfg.macros, name)) {
				b.consts[name] = true
				b.macros = append(b.macros, name)
			}
		case libclang.FunctionTemplate:
			if b.user(k) && hasAttr(k, libclang.CUDAGlobalAttr) {
				fmt.Fprintf(os.Stderr, "cudabind: %s: template kernel %s is not bound; call it from an extern \"C\" kernel\n", location(k), ns+k.Spelling())
			}
		case libclang.FunctionDecl:
			if b.user(k) && hasAttr(k, libclang.CUDAGlobalAttr) {
				*decls = append(*decls, decl{k, ns + k.Spelling(), false})
			}
		case libclang.VarDecl:
			if b.user(k) && !hasAttr(k, libclang.CUDASharedAttr) &&
				(hasAttr(k, libclang.CUDAConstantAttr) || hasAttr(k, libclang.CUDADeviceAttr)) {
				*decls = append(*decls, decl{k, ns + k.Spelling(), true})
			}
		}
	}
}

// enum binds the constants of the enum c, whose constants are in scope.
func (b *binder) enum(c libclang.Cursor, scope string) {
	for _, k := range c.Children() {
		if k.Kind() != libclang.EnumConstantDecl {
			continue
		}
		name := scope + k.Spelling()
		if b.consts[name] {
			continue
		}
		b.consts[name] = true
		goName := b.typeName(name)
		if err := b.claim(goName, name); err != nil {
			b.errs = append(b.errs, err)
			continue
		}
		b.m.consts = append(b.m.consts, &constant{cName: name, goName: goName, value: fmt.Sprint(k.EnumValue()), check: true})
	}
}

// evalMacros binds the macros that are numbers. Clang evaluates each as the
// initializer of a variable declared at the end of cfg.file, so it has the
// value the source has there.
func (b *binder) evalMacros(lib *libclang.Library, args []string) error {
	if len(b.macros) == 0 {
		return nil
	}
	var src strings.Builder
	fmt.Fprintf(&src, "#include %q\n", filepath.Join(b.cfg.dir, b.cfg.file))
	for i, name := range b.macros {
		fmt.Fprintf(&src, "constexpr auto cudabind_macro_%d = (%s);\n", i, name)
	}
	file := filepath.Join(b.cfg.dir, "cudabind_macros.cu")
	tu, err := lib.Parse(file, args, libclang.UnsavedFile{Name: file, Contents: src.String()})
	if err != nil {
		return err
	}
	defer tu.Close()
	values := make([]any, len(b.macros))
	for _, c := range tu.Cursor().Children() {
		// Only declarations in our unsaved helper file belong to this table.
		// The included source may itself declare names with the same prefix.
		declFile, _, _ := c.Location()
		if n, ok := macroValueIndex(c.Spelling(), declFile, file, len(values)); ok && c.Kind() == libclang.VarDecl {
			values[n], _ = c.Evaluate()
		}
	}
	var errs []error
	for i, name := range b.macros {
		v := values[i]
		if v == nil {
			if len(b.cfg.macros) > 0 {
				errs = append(errs, fmt.Errorf("macro %s is not a number", name))
			}
			continue
		}
		c := &constant{cName: name, goName: b.typeName(name), check: true}
		switch v := v.(type) {
		case int64:
			c.value = fmt.Sprint(v)
		case uint64:
			c.value = fmt.Sprint(v)
			c.check = v <= math.MaxInt64
		case float64:
			c.value = strconv.FormatFloat(v, 'g', -1, 64)
			c.float = true
			c.check = !math.IsInf(v, 0) && !math.IsNaN(v)
		}
		if err := b.claim(c.goName, name); err != nil {
			errs = append(errs, err)
			continue
		}
		b.m.consts = append(b.m.consts, c)
	}
	for _, name := range b.cfg.macros {
		if !slices.Contains(b.macros, name) {
			errs = append(errs, fmt.Errorf("no macro %s", name))
		}
	}
	return errors.Join(errs...)
}

func macroValueIndex(name, declFile, helperFile string, count int) (int, bool) {
	if filepath.Clean(declFile) != filepath.Clean(helperFile) {
		return 0, false
	}
	index, ok := strings.CutPrefix(name, "cudabind_macro_")
	if !ok {
		return 0, false
	}
	n, err := strconv.Atoi(index)
	if err != nil || n < 0 || n >= count || strconv.Itoa(n) != index {
		return 0, false
	}
	return n, true
}

// bind binds the kernels and variables the flags select.
func (b *binder) bind(decls []decl) error {
	seen := make(map[string]string) // name → mangled name
	kernels, globals := make(map[string]bool), make(map[string]bool)
	var errs []error
	for _, d := range decls {
		// With function bodies skipped, clang knows no definition of a
		// kernel, only of a variable.
		def := d.c.Definition()
		if def.IsNull() {
			def = d.c
		}
		if m, ok := seen[d.name]; ok {
			if m != d.c.Mangling() {
				errs = append(errs, fmt.Errorf("kernel %s is overloaded, which cudabind does not support", d.name))
			}
			continue
		}
		seen[d.name] = d.c.Mangling()
		if d.isVar {
			if len(b.cfg.globals) > 0 && !slices.Contains(b.cfg.globals, d.name) {
				continue
			}
			globals[d.name] = true
			errs = append(errs, b.global(def, d.name))
		} else {
			if len(b.cfg.kernels) > 0 && !slices.Contains(b.cfg.kernels, d.name) {
				continue
			}
			kernels[d.name] = true
			errs = append(errs, b.kernel(def, d.name))
		}
	}
	for _, name := range b.cfg.kernels {
		if !kernels[name] {
			errs = append(errs, fmt.Errorf("no kernel %s", name))
		}
	}
	for _, name := range b.cfg.globals {
		if !globals[name] {
			errs = append(errs, fmt.Errorf("no __device__ or __constant__ variable %s", name))
		}
	}
	for _, name := range b.cfg.shared {
		if !kernels[name] {
			errs = append(errs, fmt.Errorf("-shared: no kernel %s", name))
		}
	}
	return errors.Join(errs...)
}

func (b *binder) kernel(c libclang.Cursor, name string) error {
	k := &kernel{
		name:   name,
		method: b.funcName(c.Spelling()),
		cxx:    c.Mangling() != c.Spelling(),
		doc:    docComment(c.RawComment(), c.Spelling()),
		shared: slices.Contains(b.cfg.shared, name),
	}
	if err := b.claim(k.method, name); err != nil {
		return err
	}
	used := make(map[string]bool)
	for _, name := range []string{"m", "ptrs", "stream", "grid", "block", "sharedBytes",
		"cuda", "cudamem", "driver", "unsafe", "fmt", "nvrtcx"} {
		used[name] = true
	}
	var cparams []string
	for _, a := range c.Arguments() {
		t := a.Type()
		cparams = append(cparams, strings.TrimSpace(t.Spelling()+" "+a.Spelling()))
		var p param
		var err error
		if p.typ, p.slice, err = b.paramType(t); err != nil {
			return fmt.Errorf("kernel %s: parameter %s: %w", name, a.Spelling(), err)
		}
		k.params = append(k.params, p)
	}
	// Name the parameters once their types are known, so that no name
	// hides a Go type the signature uses.
	for i, a := range c.Arguments() {
		p := &k.params[i]
		p.name = lowerFirst(camel(a.Spelling()))
		if p.name == "" {
			p.name = fmt.Sprintf("arg%d", i)
		}
		for used[p.name] || token.IsKeyword(p.name) || types.Universe.Lookup(p.name) != nil || b.names[p.name] != "" {
			p.name += "_"
		}
		used[p.name] = true
	}
	k.decl = fmt.Sprintf("__global__ void %s(%s)", c.Spelling(), strings.Join(cparams, ", "))
	b.m.kernels = append(b.m.kernels, k)
	return nil
}

func (b *binder) global(c libclang.Cursor, name string) error {
	if c.Mangling() != c.Spelling() {
		return fmt.Errorf("variable %s: C++ linkage is not supported", name)
	}
	t, err := b.valueType(c.Type())
	if err != nil {
		return fmt.Errorf("variable %s: %w", name, err)
	}
	space := "__device__"
	if hasAttr(c, libclang.CUDAConstantAttr) {
		space = "__constant__"
	}
	getter := b.funcName(c.Spelling())
	setter := b.funcName("set") + camel(c.Spelling())
	for _, n := range []string{getter, setter} {
		if err := b.claim(n, name); err != nil {
			return err
		}
	}
	b.m.globals = append(b.m.globals, &global{
		name:   name,
		getter: getter,
		setter: setter,
		decl:   fmt.Sprintf("%s %s %s", space, c.Type().Spelling(), c.Spelling()),
		doc:    docComment(c.RawComment(), c.Spelling()),
		typ:    t,
	})
	return nil
}

// paramType returns the Go type of a kernel parameter of type t.
func (b *binder) paramType(t libclang.Type) (typ string, slice bool, err error) {
	c := t.Canonical()
	if c.Kind() != libclang.Pointer {
		g, err := b.valueType(t)
		return g.expr, false, err
	}
	p := c.Pointee()
	switch pc := p.Canonical(); {
	case pc.Kind() == libclang.Void, pc.Kind() == libclang.Record && pc.Size() < 0:
		return "cudamem.Slice[byte]", true, nil
	case pc.Kind() == libclang.Pointer:
		return "cudamem.Slice[driver.DevicePtr]", true, nil
	case pc.Kind() == libclang.FunctionProto:
		return "", false, fmt.Errorf("function pointer %s", t.Spelling())
	}
	g, err := b.valueType(p)
	return "cudamem.Slice[" + g.expr + "]", true, err
}

var scalars = map[libclang.TypeKind]string{
	libclang.Bool:      "bool",
	libclang.CharS:     "int8",
	libclang.SChar:     "int8",
	libclang.CharU:     "uint8",
	libclang.UChar:     "uint8",
	libclang.Short:     "int16",
	libclang.UShort:    "uint16",
	libclang.Int:       "int32",
	libclang.UInt:      "uint32",
	libclang.Long:      "int",
	libclang.ULong:     "uint",
	libclang.LongLong:  "int",
	libclang.ULongLong: "uint",
	libclang.Float:     "float32",
	libclang.Double:    "float64",
	libclang.Half:      "uint16",
	libclang.Float16:   "uint16",
	libclang.BFloat16:  "uint16",
	libclang.Enum:      "int",
}

// valueType returns the Go type of a value of C type t.
func (b *binder) valueType(t libclang.Type) (goType, error) {
	c := t.Canonical()
	size, align := c.Size(), c.Align()
	switch k := c.Kind(); k {
	case libclang.Pointer:
		return goType{"driver.DevicePtr", 8, 8}, nil
	case libclang.ConstantArray:
		e, err := b.valueType(c.Elem())
		if err != nil {
			return goType{}, err
		}
		return goType{fmt.Sprintf("[%d]%s", c.Len(), e.expr), c.Len() * e.size, e.align}, nil
	case libclang.Record:
		return b.recordType(c)
	default:
		name, ok := scalars[k]
		if !ok || size <= 0 {
			return goType{}, fmt.Errorf("unsupported type %s", t.Spelling())
		}
		switch name {
		case "int", "uint":
			// long is 8 bytes on the 64-bit platforms CUDA runs on, but
			// an enum's size depends on its values.
			name = fmt.Sprintf("%s%d", name, 8*size)
		}
		return goType{name, size, align}, nil
	}
}

// recordType returns the Go type of the C struct or union t.
func (b *binder) recordType(t libclang.Type) (goType, error) {
	def := t.Declaration().Definition()
	if def.IsNull() || t.Size() < 0 {
		return goType{}, fmt.Errorf("incomplete type %s", t.Spelling())
	}
	loc := location(def)
	if g, ok := b.types[loc]; ok {
		return g, nil
	}
	var g goType
	var err error
	if b.user(def) {
		g, err = b.mirror(t, def)
	} else {
		g, err = b.shape(t, def)
	}
	if err != nil {
		return goType{}, err
	}
	b.types[loc] = g
	return g, nil
}

// shape maps a toolkit struct to Go by its shape: a struct of one field to
// the type of that field, of n fields of one type to an array of them, and
// anything else to an array of bytes.
func (b *binder) shape(t libclang.Type, def libclang.Cursor) (goType, error) {
	size := t.Size()
	g := goType{fmt.Sprintf("[%d]byte", size), size, 1}
	fields := fieldsOf(def)
	if def.Kind() == libclang.StructDecl || def.Kind() == libclang.ClassDecl {
		var elem goType
		ok := len(fields) > 0
		for i, f := range fields {
			e, err := b.valueType(f.Type())
			if err != nil || f.IsBitField() || i > 0 && e != elem || f.OffsetOfField() != int64(i)*e.size*8 {
				ok = false
				break
			}
			elem = e
		}
		if ok && int64(len(fields))*elem.size == size {
			g = elem
			if len(fields) > 1 {
				g = goType{fmt.Sprintf("[%d]%s", len(fields), elem.expr), size, elem.align}
			}
		}
	}
	b.m.mapped = append(b.m.mapped, mapped{t.Spelling(), g})
	return g, nil
}

// mirror generates a Go struct with the layout of a struct declared in the
// package directory, or an array of its bytes for a union or a struct with
// bit-fields.
func (b *binder) mirror(t libclang.Type, def libclang.Cursor) (goType, error) {
	cName := t.Spelling()
	if name, ok := b.typedefs[location(def)]; ok {
		cName = name
	} else if unnamed(def.Spelling()) {
		return goType{}, fmt.Errorf("unnamed type %s", cName)
	}
	short := cName[strings.LastIndex(cName, ":")+1:]
	r := &record{cName: cName, goName: b.typeName(short), size: t.Size()}
	file, line, _ := def.Location()
	if rel, err := relative(b.cfg.dir, file); err == nil {
		file = rel
	}
	r.decl = fmt.Sprintf("%s:%d", file, line)
	if err := b.claim(r.goName, cName); err != nil {
		return goType{}, err
	}
	fields := fieldsOf(def)
	r.opaque = def.Kind() == libclang.UnionDecl || slices.ContainsFunc(fields, func(f libclang.Cursor) bool {
		return f.IsBitField() || f.Spelling() == ""
	})
	var off, align int64 = 0, 1
	if !r.opaque {
		names := make(map[string]bool)
		for _, f := range fields {
			ft, err := b.valueType(f.Type())
			if err != nil {
				return goType{}, fmt.Errorf("%s.%s: %w", cName, f.Spelling(), err)
			}
			coff := f.OffsetOfField() / 8
			if roundUp(off, ft.align) != coff {
				if coff < off {
					return goType{}, fmt.Errorf("%s.%s is at offset %d, before Go can place it at %d", cName, f.Spelling(), coff, roundUp(off, ft.align))
				}
				r.fields = append(r.fields, field{typ: fmt.Sprintf("[%d]byte", coff-off), off: off})
			}
			name := camel(f.Spelling())
			if names[name] {
				return goType{}, fmt.Errorf("%s: two fields named %s in Go", cName, name)
			}
			names[name] = true
			r.fields = append(r.fields, field{f.Spelling(), name, ft.expr, coff})
			off, align = coff+ft.size, max(align, ft.align)
		}
		if off < r.size {
			r.fields = append(r.fields, field{typ: fmt.Sprintf("[%d]byte", r.size-off), off: off})
			off = r.size
		}
		if roundUp(off, align) != r.size {
			return goType{}, fmt.Errorf("%s is %d bytes, but %d in Go", cName, r.size, roundUp(off, align))
		}
	}
	b.m.records = append(b.m.records, r)
	return goType{r.goName, r.size, align}, nil
}

// files returns the files to embed: those of names in the package
// directory, and every file in the include directories, which NVRTC may
// need where clang used the toolkit's.
func (b *binder) files(names []string) ([]string, error) {
	var files []string
	for _, name := range names {
		if rel, err := relative(b.cfg.dir, name); err == nil {
			files = append(files, rel)
		}
	}
	for _, dir := range b.cfg.includes {
		entries, err := os.ReadDir(filepath.Join(b.cfg.dir, dir))
		if err != nil {
			return nil, err
		}
		for _, e := range entries {
			if e.Type().IsRegular() && !strings.HasSuffix(e.Name(), ".go") {
				files = append(files, path.Join(dir, e.Name()))
			}
		}
	}
	slices.Sort(files)
	return slices.Compact(files), nil
}

// user reports whether c is declared in the package directory.
func (b *binder) user(c libclang.Cursor) bool {
	file, _, _ := c.Location()
	if file == "" {
		return false
	}
	_, err := relative(b.cfg.dir, file)
	return err == nil
}

// claim records that the generated Go name comes from the C name, or fails
// if another C name produced it.
func (b *binder) claim(goName, cName string) error {
	if prev, ok := b.names[goName]; ok && prev != cName {
		return fmt.Errorf("%s and %s are both %s in Go", prev, cName, goName)
	}
	b.names[goName] = cName
	return nil
}

func (b *binder) exported() bool {
	return isUpper(b.cfg.typeName)
}

// typeName returns the Go name of the C type name.
func (b *binder) typeName(name string) string {
	if b.cfg.prefix != "" {
		return b.cfg.prefix + camel(name)
	}
	return b.funcName(name)
}

// funcName returns the Go name of a kernel or variable.
func (b *binder) funcName(name string) string {
	if b.exported() {
		return camel(name)
	}
	return lowerFirst(camel(name))
}

func fieldsOf(c libclang.Cursor) []libclang.Cursor {
	var fields []libclang.Cursor
	for _, k := range c.Children() {
		if k.Kind() == libclang.FieldDecl {
			fields = append(fields, k)
		}
	}
	return fields
}

func hasAttr(c libclang.Cursor, kind libclang.CursorKind) bool {
	return slices.ContainsFunc(c.Children(), func(k libclang.Cursor) bool { return k.Kind() == kind })
}

func location(c libclang.Cursor) string {
	file, line, col := c.Location()
	return fmt.Sprintf("%s:%d:%d", file, line, col)
}

func unnamed(spelling string) bool {
	return spelling == "" || strings.Contains(spelling, "(unnamed") || strings.Contains(spelling, "(anonymous")
}

func roundUp(n, align int64) int64 {
	return (n + align - 1) / align * align
}

// camel returns the possibly qualified name in CamelCase. The words of a
// name in upper case, like BREAKOUT_STATE_GROUPS, are capitalized.
func camel(name string) string {
	var sb strings.Builder
	for name := range strings.SplitSeq(name, "::") {
		upper := strings.ToUpper(name) == name
		for part := range strings.SplitSeq(name, "_") {
			if part == "" {
				continue
			}
			if upper {
				part = strings.ToLower(part)
			}
			sb.WriteString(strings.ToUpper(part[:1]) + part[1:])
		}
	}
	return sb.String()
}

func lowerFirst(s string) string {
	if s == "" {
		return s
	}
	return strings.ToLower(s[:1]) + s[1:]
}

func upperFirst(s string) string {
	if s == "" {
		return s
	}
	return strings.ToUpper(s[:1]) + s[1:]
}

func isUpper(s string) bool {
	return s != "" && 'A' <= s[0] && s[0] <= 'Z'
}

// docComment returns the text of a C comment if it documents name, as Go
// doc comments do, by starting with it.
func docComment(raw, name string) string {
	var lines []string
	for line := range strings.Lines(raw) {
		line = strings.TrimSuffix(strings.TrimSpace(line), "*/")
		for _, p := range []string{"///", "//!", "//", "/**", "/*!", "/*", "*"} {
			if s, ok := strings.CutPrefix(line, p); ok {
				line = s
				break
			}
		}
		lines = append(lines, strings.TrimRight(strings.TrimPrefix(line, " "), " \t"))
	}
	text := strings.TrimSpace(strings.Join(lines, "\n"))
	if w, _, _ := strings.Cut(text, " "); w != name {
		return ""
	}
	return text
}
