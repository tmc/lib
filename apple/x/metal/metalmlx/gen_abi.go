//go:build ignore

// Gen_abi writes Go mirrors of the parameter structs MLX's Metal kernels
// take, measured in MLX's own headers rather than transcribed from them.
//
// The structs are passed to a kernel by value with setBytes, so a Go host has
// to agree with the shading language on every offset. clang is the oracle:
// the generator compiles a probe that names each record and reads back
// -fdump-record-layouts, which prints the fields, their offsets, and the size
// and alignment of the record. Those numbers are written into abi.gen.go as
// Go types and into abi_layout_gen_test.go as assertions, so a Go type that
// stops matching the header is a test failure rather than a wrong result.
//
// MLX's Metal parameter headers are C++ (namespaces and templates), which the
// appledocs clangdump layout tool used by the CUDA port cannot parse: it
// names its translation unit for Objective-C and has no flag to say otherwise.
//
// Usage:
//
//	MLX_ROOT=/path/to/mlx go run gen_abi.go
package main

import (
	"bytes"
	"fmt"
	"go/format"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
)

// record is one struct to mirror. Name is the Go type, CPP is the record as
// clang prints it, and Header is included by the probe.
type record struct {
	Name   string
	CPP    string
	Header string
	Doc    string
}

var records = []record{
	{
		Name:   "GEMMParams",
		CPP:    "mlx::steel::GEMMParams",
		Header: "mlx/backend/metal/kernels/steel/gemm/params.h",
		Doc:    "GEMMParams is the parameter block of MLX's steel GEMM kernels.",
	},
	{
		Name:   "GEMMSpiltKParams",
		CPP:    "mlx::steel::GEMMSpiltKParams",
		Header: "mlx/backend/metal/kernels/steel/gemm/params.h",
		Doc:    "GEMMSpiltKParams is the parameter block of the split-k GEMM kernels.\nThe name is spelled as MLX spells it.",
	},
	{
		Name:   "GEMMAddMMParams",
		CPP:    "mlx::steel::GEMMAddMMParams",
		Header: "mlx/backend/metal/kernels/steel/gemm/params.h",
		Doc:    "GEMMAddMMParams is the extra parameter block of the addmm kernels.",
	},
	{
		Name:   "ConvParams2D",
		CPP:    "MLXConvParams<2>",
		Header: "mlx/backend/metal/kernels/steel/conv/params.h",
		Doc:    "ConvParams2D is the parameter block of the 2D convolution kernels.",
	},
}

// cppTypes maps a C++ type as clang prints it to its Go equivalent.
var cppTypes = map[string]string{
	"int":                "int32",
	"unsigned int":       "uint32",
	"short":              "int16",
	"unsigned short":     "uint16",
	"char":               "int8",
	"unsigned char":      "uint8",
	"long long":          "int64",
	"unsigned long long": "uint64",
	"int32_t":            "int32",
	"uint32_t":           "uint32",
	"int64_t":            "int64",
	"uint64_t":           "uint64",
	"size_t":             "uint64",
	"float":              "float32",
	"double":             "float64",
	"_Bool":              "bool",
	"bool":               "bool",
}

type field struct {
	Name   string
	GoName string
	GoType string
	Offset int
}

type layout struct {
	record
	Fields []field
	Size   int
	Align  int
}

func main() {
	root := os.Getenv("MLX_ROOT")
	if root == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			fatal(err)
		}
		root = filepath.Join(home, "ml-explore", "mlx")
	}
	dump, err := dumpLayouts(root)
	if err != nil {
		fatal(err)
	}
	layouts, err := parseLayouts(dump)
	if err != nil {
		fatal(err)
	}
	revision, err := sourceRevision(root)
	if err != nil {
		fatal(err)
	}
	if err := write("abi.gen.go", emitTypes(layouts, revision)); err != nil {
		fatal(err)
	}
	if err := write("abi_layout_gen_test.go", emitTest(layouts, revision)); err != nil {
		fatal(err)
	}
}

func sourceRevision(root string) (string, error) {
	args := []string{"diff", "--quiet", "--"}
	for _, r := range records {
		args = append(args, r.Header)
	}
	cmd := exec.Command("git", args...)
	cmd.Dir = root
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("MLX parameter headers have local changes: %w", err)
	}
	cmd = exec.Command("git", "rev-parse", "HEAD")
	cmd.Dir = root
	out, err := cmd.Output()
	if err != nil {
		return "", fmt.Errorf("read MLX revision: %w", err)
	}
	return strings.TrimSpace(string(out)), nil
}

// dumpLayouts compiles a probe naming every record and returns clang's record
// layout dump.
func dumpLayouts(root string) (string, error) {
	var probe bytes.Buffer
	// The headers are written for the shading language, where the fixed width
	// integer types are always in scope.
	probe.WriteString("#include <cstdint>\nusing std::int64_t;\nusing std::uint64_t;\nusing std::int32_t;\nusing std::uint32_t;\n")
	seen := map[string]bool{}
	for _, r := range records {
		if !seen[r.Header] {
			fmt.Fprintf(&probe, "#include %q\n", r.Header)
			seen[r.Header] = true
		}
	}
	// clang lays a record out when something needs its size.
	for _, r := range records {
		fmt.Fprintf(&probe, "static_assert(sizeof(%s) > 0);\n", r.CPP)
	}

	dir, err := os.MkdirTemp("", "metalmlx-abi")
	if err != nil {
		return "", err
	}
	defer os.RemoveAll(dir)
	src := filepath.Join(dir, "probe.cc")
	if err := os.WriteFile(src, probe.Bytes(), 0o644); err != nil {
		return "", err
	}

	cmd := exec.Command("xcrun", "clang++", "-std=c++17", "-I", root,
		"-fsyntax-only", "-Xclang", "-fdump-record-layouts", src)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("%s: %v\n%s", strings.Join(cmd.Args, " "), err, out)
	}
	return string(out), nil
}

var (
	headerLine = regexp.MustCompile(`^\s*0 \| (?:struct|class) (\S.*)$`)
	fieldLine  = regexp.MustCompile(`^\s*(\d+) \|\s{3,}(.+)$`)
	sizeLine   = regexp.MustCompile(`\[sizeof=(\d+), dsize=\d+, align=(\d+)`)
)

// parseLayouts reads clang's dump, keeping the records the manifest asks for.
func parseLayouts(dump string) ([]layout, error) {
	byCPP := map[string]record{}
	for _, r := range records {
		byCPP[r.CPP] = r
	}
	found := map[string]*layout{}

	var cur *layout
	for _, line := range strings.Split(dump, "\n") {
		if m := headerLine.FindStringSubmatch(line); m != nil {
			cur = nil
			if r, ok := byCPP[strings.TrimSpace(m[1])]; ok {
				cur = &layout{record: r}
				found[r.CPP] = cur
			}
			continue
		}
		if cur == nil {
			continue
		}
		if m := sizeLine.FindStringSubmatch(line); m != nil {
			cur.Size, _ = strconv.Atoi(m[1])
			cur.Align, _ = strconv.Atoi(m[2])
			cur = nil
			continue
		}
		m := fieldLine.FindStringSubmatch(line)
		if m == nil {
			continue
		}
		offset, _ := strconv.Atoi(m[1])
		f, err := parseField(strings.TrimSpace(m[2]), offset)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", cur.CPP, err)
		}
		cur.Fields = append(cur.Fields, f)
	}

	var layouts []layout
	for _, r := range records {
		l, ok := found[r.CPP]
		if !ok {
			return nil, fmt.Errorf("%s: not in the layout dump", r.CPP)
		}
		if l.Size == 0 || len(l.Fields) == 0 {
			return nil, fmt.Errorf("%s: measured as empty", r.CPP)
		}
		layouts = append(layouts, *l)
	}
	return layouts, nil
}

// parseField reads one dumped field, which clang prints as its type followed
// by its name, with any array bounds on the type: "const int64_t[4] strides".
func parseField(decl string, offset int) (field, error) {
	i := strings.LastIndex(decl, " ")
	if i < 0 {
		return field{}, fmt.Errorf("field %q has no name", decl)
	}
	name := decl[i+1:]
	typ := strings.TrimPrefix(strings.TrimSpace(decl[:i]), "const ")

	bound := ""
	if j := strings.Index(typ, "["); j >= 0 {
		bound, typ = typ[j:], strings.TrimSpace(typ[:j])
	}
	goType, ok := cppTypes[typ]
	if !ok {
		return field{}, fmt.Errorf("field %q has unmapped type %q", name, typ)
	}
	return field{Name: name, GoName: goName(name), GoType: bound + goType, Offset: offset}, nil
}

// goName exports a C++ field name: batch_stride_a becomes BatchStrideA.
func goName(name string) string {
	var b strings.Builder
	for _, part := range strings.Split(name, "_") {
		if part == "" {
			continue
		}
		b.WriteString(strings.ToUpper(part[:1]))
		b.WriteString(part[1:])
	}
	return b.String()
}

func emitTypes(layouts []layout, revision string) []byte {
	var b bytes.Buffer
	b.WriteString("// Code generated by gen_abi.go. DO NOT EDIT.\n\n")
	fmt.Fprintf(&b, "// Source: MLX commit %s.\n\n", revision)
	b.WriteString("package metalmlx\n")
	for _, l := range layouts {
		fmt.Fprintf(&b, "\n// %s\n//\n", strings.ReplaceAll(l.Doc, "\n", "\n// "))
		fmt.Fprintf(&b, "// Measured from %s as %s: size %d, alignment %d.\n",
			l.Header, l.CPP, l.Size, l.Align)
		fmt.Fprintf(&b, "type %s struct {\n", l.Name)
		for _, f := range l.Fields {
			fmt.Fprintf(&b, "\t%s %s // %s, offset %d\n", f.GoName, f.GoType, f.Name, f.Offset)
		}
		b.WriteString("}\n")
	}
	return b.Bytes()
}

func emitTest(layouts []layout, revision string) []byte {
	var b bytes.Buffer
	b.WriteString("// Code generated by gen_abi.go. DO NOT EDIT.\n\n")
	fmt.Fprintf(&b, "// Source: MLX commit %s.\n\n", revision)
	b.WriteString("package metalmlx\n\nimport (\n\t\"testing\"\n\t\"unsafe\"\n)\n\n")
	b.WriteString("// TestGeneratedLayouts checks the Go mirrors against the sizes,\n")
	b.WriteString("// alignments and offsets clang measured in MLX's headers.\n")
	b.WriteString("func TestGeneratedLayouts(t *testing.T) {\n")
	for _, l := range layouts {
		fmt.Fprintf(&b, "\tt.Run(%q, func(t *testing.T) {\n", l.Name)
		fmt.Fprintf(&b, "\t\tvar v %s\n", l.Name)
		fmt.Fprintf(&b, "\t\tif got := unsafe.Sizeof(v); got != %d {\n", l.Size)
		fmt.Fprintf(&b, "\t\t\tt.Errorf(\"sizeof(%s) = %%d, want %d\", got)\n\t\t}\n", l.Name, l.Size)
		fmt.Fprintf(&b, "\t\tif got := unsafe.Alignof(v); got != %d {\n", l.Align)
		fmt.Fprintf(&b, "\t\t\tt.Errorf(\"alignof(%s) = %%d, want %d\", got)\n\t\t}\n", l.Name, l.Align)
		for _, f := range l.Fields {
			fmt.Fprintf(&b, "\t\tif got := unsafe.Offsetof(v.%s); got != %d {\n", f.GoName, f.Offset)
			fmt.Fprintf(&b, "\t\t\tt.Errorf(\"offsetof(%s.%s) = %%d, want %d\", got)\n\t\t}\n",
				l.Name, f.GoName, f.Offset)
		}
		b.WriteString("\t})\n")
	}
	b.WriteString("}\n")
	return b.Bytes()
}

func write(name string, src []byte) error {
	formatted, err := format.Source(src)
	if err != nil {
		return fmt.Errorf("format %s: %w\n%s", name, err, src)
	}
	if err := os.WriteFile(name, formatted, 0o644); err != nil {
		return err
	}
	fmt.Println("wrote", name)
	return nil
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "gen_abi:", err)
	os.Exit(1)
}
