package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"golang.org/x/tools/imports"
)

const imps = `import (
	"crypto/sha256"
	"embed"
	"fmt"
	"io/fs"
	"math"
	"os"
	"sync/atomic"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/cudatest"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)
`

// A gen accumulates Go source.
type gen struct {
	bytes.Buffer
}

func (g *gen) p(format string, args ...any) {
	fmt.Fprintf(g, format, args...)
	g.WriteByte('\n')
}

// doc writes text as a doc comment.
func (g *gen) doc(text string) {
	for line := range strings.Lines(text) {
		g.p("//%s", strings.TrimRight(" "+line, " \n"))
	}
}

// generate returns the Go source of the binding of m and of its test.
func generate(cfg *config, m *module) (src, test []byte, err error) {
	if err := validateWitnessSchemas(m); err != nil {
		return nil, nil, err
	}
	sum, err := sumFiles(cfg.dir, m.files)
	if err != nil {
		return nil, nil, err
	}
	t := cfg.typeName
	T := upperFirst(t) // for the names of functions about t
	loadName, compileName, newName := "load"+T, "compile"+T, "new"+T
	if isUpper(t) {
		loadName = "Load" + T
	}

	var g gen
	g.header(cfg)
	g.p("//go:embed %s", strings.Join(m.files, " "))
	g.p("var %sFiles embed.FS", t)
	g.p("")
	g.p("// %sSum is the SHA-256 of the files %s was generated from.", t, cfg.output)
	g.p("const %sSum = %q", t, sum)
	g.p("")
	g.p("// %s is %s, compiled with NVRTC and loaded on a device.", t, cfg.file)
	g.p("type %s struct {", t)
	g.p("dev *cuda.Device")
	g.p("mod *cuda.Module")
	g.p("raw driver.Module")
	g.p("unchecked atomic.Bool")
	g.p("kernels [%d]*cuda.Kernel", len(m.kernels))
	g.p("}")
	g.p("")
	if len(m.consts) > 0 {
		g.p("// The enum constants and numeric macros of %s.", cfg.file)
		g.p("const (")
		for _, c := range m.consts {
			g.p("%s = %s // %s", c.goName, c.value, c.cName)
		}
		g.p(")")
		g.p("")
	}
	g.p("// %sKernels are the names of the kernels of %s; those with C++ linkage are NVRTC name expressions.", t, t)
	g.p("var %sKernels = [...]struct {", t)
	g.p("name string")
	g.p("cxx bool")
	g.p("}{")
	for _, k := range m.kernels {
		g.p("{%q, %v},", k.name, k.cxx)
	}
	g.p("}")
	g.p("")

	g.p("// %s compiles %s for dev with the options cudabind was run with, then opts.", compileName, cfg.file)
	g.p("// It appends extra to the source.")
	g.p("func %s(dev *cuda.Device, extra string, opts ...nvrtcx.Option) (*nvrtcx.CompiledModule, error) {", compileName)
	g.p("h := sha256.New()")
	g.p("for _, name := range [...]string{%s} {", quoteList(m.files))
	g.p("b, err := %sFiles.ReadFile(name)", t)
	g.p("if err != nil { return nil, err }")
	g.p(`fmt.Fprintf(h, "%%s\x00%%d\x00", name, len(b))`)
	g.p("h.Write(b)")
	g.p("}")
	g.p("if fmt.Sprintf(\"%%x\", h.Sum(nil)) != %sSum {", t)
	g.p("return nil, fmt.Errorf(\"%s or a file it includes has changed since %s was generated; run go generate\")", cfg.file, cfg.output)
	g.p("}")
	g.p("src, err := %sFiles.ReadFile(%q)", t, cfg.file)
	g.p("if err != nil { return nil, err }")
	g.p("var exprs []string")
	g.p("for _, k := range %sKernels {", t)
	g.p("if k.cxx { exprs = append(exprs, k.name) }")
	g.p("}")
	g.p("o := []nvrtcx.Option{")
	g.p("nvrtcx.WithName(%q),", cfg.file)
	g.p("nvrtcx.WithHeadersFS(%sFiles),", t)
	g.p("nvrtcx.WithNameExpressions(exprs...),")
	g.p("nvrtcx.WithDeviceArch(dev),")
	if len(cfg.defines) > 0 {
		g.p("nvrtcx.WithDefines(%s),", quoteList(cfg.defines))
	}
	if cfg.std != "" {
		g.p("nvrtcx.WithStandard(%q),", cfg.std)
	}
	if cfg.defaultDevice {
		g.p("nvrtcx.WithDefaultDevice(),")
	}
	g.p("}")
	for i, dir := range cfg.includes {
		if dir == "." {
			continue
		}
		g.p("dir%d, err := fs.Sub(%sFiles, %q)", i, t, dir)
		g.p("if err != nil { return nil, err }")
		g.p("o = append(o, nvrtcx.WithHeadersFS(dir%d))", i)
	}
	g.p("return nvrtcx.Compile(string(src)+extra, append(o, opts...)...)")
	g.p("}")
	g.p("")

	g.p("// %s compiles %s for dev and loads it. The options follow those", loadName, cfg.file)
	g.p("// cudabind was run with.")
	g.p("func %s(dev *cuda.Device, opts ...nvrtcx.Option) (*%s, error) {", loadName, t)
	g.p("cm, err := %s(dev, \"\", opts...)", compileName)
	g.p("if err != nil { return nil, fmt.Errorf(\"%s: %%w\", err) }", cfg.file)
	g.p("return %s(dev, cm)", newName)
	g.p("}")
	g.p("")
	g.p("func %s(dev *cuda.Device, cm *nvrtcx.CompiledModule) (*%s, error) {", newName, t)
	g.p("raw, err := cm.Load(dev)")
	g.p("if err != nil { return nil, fmt.Errorf(\"%s: %%w\", err) }", cfg.file)
	g.p("mod, err := cuda.OwnModule(dev, raw)")
	g.p("if err != nil {")
	g.p("dev.WithContext(raw.Close)")
	g.p("return nil, fmt.Errorf(\"%s: %%w\", err)", cfg.file)
	g.p("}")
	g.p("m := &%s{dev: dev, mod: mod, raw: raw}", t)
	g.p("for i, k := range %sKernels {", t)
	g.p("name := k.name")
	g.p("if k.cxx { name, _ = cm.LoweredName(k.name) }")
	g.p("if m.kernels[i], err = mod.Function(name); err != nil {")
	g.p("mod.Close()")
	g.p("return nil, fmt.Errorf(\"%s: %%w\", err)", cfg.file)
	g.p("}")
	g.p("}")
	g.p("return m, nil")
	g.p("}")
	g.p("")
	g.p("// Close unloads the module. Work that uses it must be complete.")
	g.p("func (m *%s) Close() error {", t)
	g.p("if m == nil { return nil }")
	g.p("return m.mod.Close()")
	g.p("}")
	g.p("")
	g.p("func (m *%s) launch(k int, stream *cuda.Stream, grid, block [3]uint32, sharedBytes uint32, args ...unsafe.Pointer) error {", t)
	g.p("if m == nil || m.kernels[k] == nil || stream == nil { return fmt.Errorf(\"closed module or nil stream\") }")
	g.p("err := stream.WithHandle(func(h driver.Stream) error {")
	g.p("return m.kernels[k].Launch(grid, block, sharedBytes, h, args)")
	g.p("})")
	g.p("if err != nil { return fmt.Errorf(\"%%s: %%w\", %sKernels[k].name, err) }", t)
	g.p("return nil")
	g.p("}")
	g.launchHelpers(t)

	for i, k := range m.kernels {
		g.p("")
		g.p("// %s launches %s:", k.method, k.name)
		g.p("//")
		g.p("//\t%s", k.decl)
		if k.doc != "" {
			g.p("//")
			g.doc(k.doc)
		}
		var params, args, ptrs []string
		for _, p := range k.params {
			params = append(params, p.name+" "+p.typ)
			if p.slice {
				args = append(args, fmt.Sprintf("unsafe.Pointer(&ptrs[%d])", len(ptrs)))
				ptrs = append(ptrs, p.name+".Ptr()")
			} else {
				args = append(args, "unsafe.Pointer(&"+p.name+")")
			}
		}
		shared := "0"
		if k.shared {
			params = append([]string{"sharedBytes uint32"}, params...)
			shared = "sharedBytes"
		}
		g.p("func (m *%s) %s(stream *cuda.Stream, grid, block [3]uint32, %s) error {", t, k.method, strings.Join(params, ", "))
		g.kernelChecks(t, k, i, shared)
		if len(ptrs) > 0 {
			g.p("ptrs := [...]driver.DevicePtr{%s}", strings.Join(ptrs, ", "))
		}
		g.p("return m.launch(%d, stream, grid, block, %s, %s)", i, shared, strings.Join(args, ", "))
		g.p("}")
	}

	if len(m.globals) > 0 {
		g.p("")
		g.p("func (m *%s) copyGlobal(name string, p unsafe.Pointer, size uintptr, toDevice bool) error {", t)
		g.p("return m.dev.WithContext(func() error {")
		g.p("d, n, err := m.raw.Global(name)")
		g.p("if err != nil { return fmt.Errorf(\"%%s: %%w\", name, err) }")
		g.p("if n != uint(size) { return fmt.Errorf(\"%%s is %%d bytes, want %%d\", name, n, size) }")
		g.p("if toDevice { return d.CopyFromHostPtr(p, n) }")
		g.p("return d.CopyToHostPtr(p, n)")
		g.p("})")
		g.p("}")
	}
	for _, gl := range m.globals {
		g.p("")
		g.p("// %s returns the value of %s:", gl.getter, gl.name)
		g.p("//")
		g.p("//\t%s", gl.decl)
		if gl.doc != "" {
			g.p("//")
			g.doc(gl.doc)
		}
		g.p("func (m *%s) %s() (%s, error) {", t, gl.getter, gl.typ.expr)
		g.p("var v %s", gl.typ.expr)
		g.p("err := m.copyGlobal(%q, unsafe.Pointer(&v), unsafe.Sizeof(v), false)", gl.name)
		g.p("return v, err")
		g.p("}")
		g.p("")
		g.p("// %s sets %s to v.", gl.setter, gl.name)
		g.p("func (m *%s) %s(v %s) error {", t, gl.setter, gl.typ.expr)
		g.p("return m.copyGlobal(%q, unsafe.Pointer(&v), unsafe.Sizeof(v), true)", gl.name)
		g.p("}")
	}

	for _, r := range m.records {
		g.p("")
		g.p("// %s is the C type %s, declared at %s.", r.goName, r.cName, r.decl)
		if r.opaque {
			g.p("type %s [%d]byte", r.goName, r.size)
			continue
		}
		g.p("type %s struct {", r.goName)
		for _, f := range r.fields {
			name := f.goName
			if name == "" {
				name = "_"
			}
			g.p("%s %s", name, f.typ)
		}
		g.p("}")
	}

	// Assert at compile time that each Go type has the size and field
	// offsets clang gave its C type: a difference is a constant overflow
	// or an index out of range.
	if len(m.records) > 0 {
		g.p("")
		g.p("// The sizes and offsets of the C types, as clang lays them out.")
		for _, r := range m.records {
			g.p("var _ = [1]struct{}{}[unsafe.Sizeof(%s{})-%d]", r.goName, r.size)
			for _, f := range r.fields {
				if f.cName != "" {
					g.p("var _ = [1]struct{}{}[unsafe.Offsetof(%s{}.%s)-%d]", r.goName, f.goName, f.off)
				}
			}
		}
	}

	if src, err = format(cfg.output, g.Bytes()); err != nil {
		return nil, nil, err
	}
	test, err = generateTest(cfg, m, compileName, newName)
	return src, test, err
}

// generateTest returns a test that checks the layouts against NVRTC's.
func generateTest(cfg *config, m *module, compileName, newName string) ([]byte, error) {
	T := upperFirst(cfg.typeName)
	// A check compares the value of the C expression cExpr, described by
	// name, with that of goExpr.
	type check struct{ name, cExpr, goExpr string }
	var checks []check
	c := func(cExpr, goExpr string) check { return check{cExpr, cExpr, goExpr} }
	for _, r := range m.records {
		for _, f := range r.fields {
			if f.cName != "" {
				checks = append(checks, check{
					fmt.Sprintf("offsetof(%s, %s)", r.cName, f.cName),
					fmt.Sprintf("__builtin_offsetof(%s, %s)", r.cName, f.cName),
					fmt.Sprintf("int64(unsafe.Offsetof(%s{}.%s))", r.goName, f.goName),
				})
				checks = append(checks, check{fmt.Sprintf("sizeof(%s.%s)", r.cName, f.cName), fmt.Sprintf("sizeof(((%s*)0)->%s)", r.cName, f.cName), fmt.Sprintf("int64(unsafe.Sizeof(%s{}.%s))", r.goName, f.goName)})
			}
		}
		checks = append(checks, c("sizeof("+r.cName+")", fmt.Sprintf("int64(unsafe.Sizeof(%s{}))", r.goName)))
		checks = append(checks, c("alignof("+r.cName+")", fmt.Sprintf("int64(abiAlignment%d.Params[0].Alignment)", len(checks))))
	}
	for _, mt := range m.mapped {
		checks = append(checks, c("sizeof("+mt.cName+")", fmt.Sprintf("int64(unsafe.Sizeof(*new(%s)))", mt.typ.expr)))
		checks = append(checks, c("alignof("+mt.cName+")", fmt.Sprintf("int64(abiAlignment%d.Params[0].Alignment)", len(checks))))
	}
	for _, k := range m.kernels {
		for _, p := range k.params {
			typ := p.typ
			if p.slice {
				typ = "driver.DevicePtr"
			}
			checks = append(checks, check{fmt.Sprintf("sizeof(%s argument %s)", k.name, p.cName), "sizeof(" + p.cType + ")", fmt.Sprintf("int64(unsafe.Sizeof(*new(%s)))", typ)})
			checks = append(checks, check{fmt.Sprintf("alignof(%s argument %s)", k.name, p.cName), "alignof(" + p.cType + ")", fmt.Sprintf("int64(abiAlignment%d.Params[0].Alignment)", len(checks))})
		}
	}
	for _, k := range m.consts {
		switch {
		case !k.check:
		case k.float:
			checks = append(checks, c(fmt.Sprintf("__double_as_longlong((double)(%s))", k.cName), fmt.Sprintf("int64(math.Float64bits(%s))", k.goName)))
		default:
			checks = append(checks, c(k.cName, k.goName))
		}
	}

	var g gen
	g.header(cfg)
	if len(checks) == 0 {
		if err := g.parameterWitness(cfg, m, compileName, newName); err != nil {
			return nil, err
		}
		return format(testName(cfg.output), g.Bytes())
	}
	g.p("type %sLayoutValue struct { c string; v int64 }", cfg.typeName)
	g.p("func %sLayoutExpected() ([]%sLayoutValue, error) {", cfg.typeName, cfg.typeName)
	for i, c := range checks {
		if !strings.HasPrefix(c.cExpr, "alignof(") {
			continue
		}
		var align int64
		var width int64
		var typeName string
		for _, r := range m.records {
			if c.cExpr == "alignof("+r.cName+")" {
				align = r.align
				width, typeName = r.size, r.cName
			}
		}
		for _, mt := range m.mapped {
			if c.cExpr == "alignof("+mt.cName+")" {
				align = mt.align
				width, typeName = mt.typ.size, mt.cName
			}
		}
		for _, k := range m.kernels {
			for _, p := range k.params {
				if c.name == "alignof("+k.name+" argument "+p.cName+")" {
					align = p.abiAlignment
					width = p.width
					typeName = p.cType
				}
			}
		}
		g.p("abiAlignment%d, err := cudakernel.NewSchema(cudakernel.Param{Name:%q,Type:%q,ByteWidth:%d,Alignment:%d}); if err != nil { return nil, err }", i, c.name, typeName, width, align)
	}
	g.p("want := []%sLayoutValue{", cfg.typeName)
	for _, c := range checks {
		g.p("{%q, %s},", c.name, c.goExpr)
	}
	g.p("}")
	g.p("return want, nil")
	g.p("}")
	g.p("// %sCheckLayout compares device readback, not the generator's own layout constants.", cfg.typeName)
	g.p("func %sCheckLayout(got []int64) error {", cfg.typeName)
	g.p("want, err := %sLayoutExpected(); if err != nil { return err }", cfg.typeName)
	g.p("if len(got) != len(want) { return fmt.Errorf(\"ABI witness has %%d values; requires %%d\", len(got), len(want)) }")
	g.p("for i, w := range want { if got[i] != w.v { return fmt.Errorf(\"%%s = %%d; Go schema has %%d\", w.c, got[i], w.v) } }; return nil")
	g.p("}")
	g.p("func %sLayoutSource() string { return `", cfg.typeName)
	g.p(`extern "C" __global__ void cudabind_layout(long long* out) {`)
	for i, c := range checks {
		g.p("out[%d] = (long long)(%s);", i, c.cExpr)
	}
	g.p("}")
	g.p("` }")
	g.p("func %sSentinelPointer(t *testing.T,dev *cuda.Device) driver.DevicePtr {", cfg.typeName)
	g.p("t.Helper();if _,err:=cudalock.Inherited(os.Getenv);err!=nil{t.Fatal(err)}")
	g.p("buffer,err:=cudamem.Alloc[byte](dev,1);if err!=nil{t.Fatal(err)};t.Cleanup(func(){if err:=buffer.Free();err!=nil{t.Error(err)}});return buffer.Ptr()")
	g.p("}")
	g.p("func %sReadLayout(t *testing.T, dev *cuda.Device, m *%s) []int64 {", cfg.typeName, cfg.typeName)
	g.p("t.Helper(); if _,err:=cudalock.Inherited(os.Getenv); err!=nil {t.Fatal(err)}")
	g.p("want,err:=%sLayoutExpected();if err!=nil{t.Fatal(err)}", cfg.typeName)
	g.p("kernel,err:=m.mod.Function(\"cudabind_layout\");if err!=nil{t.Fatal(err)}")
	g.p("out,err:=cudamem.Alloc[int64](dev,len(want));if err!=nil{t.Fatal(err)};defer func(){if err:=out.Free();err!=nil{t.Error(err)}}()")
	g.p("pointer:=out.Ptr();stream:=dev.DefaultStream()")
	g.p("err=stream.WithHandle(func(h driver.Stream)error{return kernel.Launch([3]uint32{1,1,1},[3]uint32{1,1,1},0,h,[]unsafe.Pointer{unsafe.Pointer(&pointer)})});if err!=nil{t.Fatal(err)}")
	g.p("if err:=stream.Synchronize(t.Context());err!=nil{t.Fatal(err)}")
	g.p("got:=make([]int64,len(want));if err:=out.CopyToHost(got);err!=nil{t.Fatal(err)};return got")
	g.p("}")
	g.p("// Test%sLayout checks that the Go types and constants of %s have the", T, cfg.output)
	g.p("// sizes, field offsets and values NVRTC gives the C ones.")
	g.p("func Test%sLayout(t *testing.T) {", T)
	g.nativeGuard()
	g.p("dev := cudatest.Require(t)")
	g.p("defer func() { if err := dev.Close(); err != nil { t.Error(err) } }()")
	g.p("want, err := %sLayoutExpected(); if err != nil { t.Fatal(err) }", cfg.typeName)
	g.p("cm, err := %s(dev, %sLayoutSource())", compileName, cfg.typeName)
	g.p("if err != nil { t.Fatal(err) }")
	g.p("m, err := %s(dev, cm)", newName)
	g.p("if err != nil { t.Fatal(err) }")
	g.p("defer func() { if err := m.Close(); err != nil { t.Error(err) } }()")
	g.p("k, err := m.mod.Function(\"cudabind_layout\")")
	g.p("if err != nil { t.Fatal(err) }")
	g.p("out, err := cudamem.Alloc[int64](dev, len(want))")
	g.p("if err != nil { t.Fatal(err) }")
	g.p("defer func() { if err := out.Free(); err != nil { t.Error(err) } }()")
	g.p("p := out.Ptr()")
	g.p("stream := dev.DefaultStream()")
	g.p("err = stream.WithHandle(func(h driver.Stream) error {")
	g.p("return k.Launch([3]uint32{1, 1, 1}, [3]uint32{1, 1, 1}, 0, h, []unsafe.Pointer{unsafe.Pointer(&p)})")
	g.p("})")
	g.p("if err != nil { t.Fatal(err) }")
	g.p("if err := stream.Synchronize(t.Context()); err != nil { t.Fatal(err) }")
	g.p("got := make([]int64, len(want))")
	g.p("if err := out.CopyToHost(got); err != nil { t.Fatal(err) }")
	g.p("if err := %sCheckLayout(got); err != nil { t.Fatal(err) }", cfg.typeName)
	g.p("}")
	if err := g.parameterWitness(cfg, m, compileName, newName); err != nil {
		return nil, err
	}
	return format(testName(cfg.output), g.Bytes())
}

func (g *gen) header(cfg *config) {
	g.p("// Code generated by cudabind %s; DO NOT EDIT.", strings.Join(cfg.args, " "))
	g.p("")
	g.p("package %s", cfg.pkg)
	g.p("")
	g.WriteString(imps)
	g.p("")
}

func (g *gen) nativeGuard() {
	g.p("if os.Getenv(\"CUDABIND_NATIVE\") != \"1\" { t.Skip(\"set CUDABIND_NATIVE=1 under gpu-lock for ABI witness execution\") }")
	g.p("if _,err := cudalock.Inherited(os.Getenv); err != nil { t.Fatal(err) }")
}

// format removes the unused imports from src and formats it.
func format(name string, src []byte) ([]byte, error) {
	out, err := imports.Process(name, src, &imports.Options{Comments: true, TabIndent: true, TabWidth: 8})
	if err != nil {
		return nil, fmt.Errorf("format %s: %w\n%s", name, err, src)
	}
	return out, nil
}

// sumFiles returns the hash the generated code checks the embedded files
// against.
func sumFiles(dir string, files []string) (string, error) {
	h := sha256.New()
	for _, name := range files {
		b, err := os.ReadFile(filepath.Join(dir, filepath.FromSlash(name)))
		if err != nil {
			return "", err
		}
		fmt.Fprintf(h, "%s\x00%d\x00", name, len(b))
		h.Write(b)
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func quoteList(s []string) string {
	q := make([]string, len(s))
	for i, v := range s {
		q[i] = fmt.Sprintf("%q", v)
	}
	return strings.Join(q, ", ")
}
