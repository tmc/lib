package main

import (
	"bytes"
	"flag"
	"fmt"
	"go/format"
	"io"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"text/template"

	"github.com/tmc/lib/nvidia/x/internal/cudatool"
	"github.com/tmc/lib/nvidia/x/internal/precompile"
)

func main() {
	err := run(os.Args[1:], ".", os.Stdout, os.Stderr)
	if err == errStale {
		os.Exit(1)
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "cudaprecompile: %v\n", err)
		os.Exit(2)
	}
}

var errStale = fmt.Errorf("stale")

type listFlag []string

func (l *listFlag) String() string     { return strings.Join(*l, ",") }
func (l *listFlag) Set(s string) error { *l = append(*l, s); return nil }

type config struct {
	out, varName, pkg, arch, ptx, std, tag string
	include, defines, names                listFlag
	noLTO, check                           bool
	sources                                []string
}

func parse(args []string, stderr io.Writer) (*config, error) {
	var c config
	fs := flag.NewFlagSet("cudaprecompile", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.Usage = func() {
		fmt.Fprintf(stderr, "usage: cudaprecompile [flags] file.cu...\n")
		fs.PrintDefaults()
	}
	fs.StringVar(&c.out, "o", "", "output Go `file`")
	fs.StringVar(&c.varName, "var", "precompiled", "Module variable `name`")
	fs.StringVar(&c.pkg, "pkg", os.Getenv("GOPACKAGE"), "package `name`")
	fs.StringVar(&c.arch, "arch", "sm_80,sm_90,sm_100,sm_120", "machine code `targets`")
	fs.StringVar(&c.ptx, "ptx", "", "PTX `target`, or none")
	fs.StringVar(&c.std, "std", "", "C++ standard `version`")
	fs.Var(&c.include, "I", "include `dir`")
	fs.Var(&c.defines, "D", "define `name[=value]`")
	fs.Var(&c.names, "name", "name `expression`")
	fs.BoolVar(&c.noLTO, "nolto", false, "link without link-time optimization")
	fs.StringVar(&c.tag, "tag", "", "embed the image only with build `tag`")
	fs.BoolVar(&c.check, "check", false, "report whether the output is out of date")
	if err := fs.Parse(splitJoined(args)); err != nil {
		return nil, err
	}
	c.sources = fs.Args()
	if len(c.sources) == 0 {
		fs.Usage()
		return nil, fmt.Errorf("no sources")
	}
	if c.pkg == "" {
		return nil, fmt.Errorf("no package name: set -pkg or run from go generate")
	}
	if !token(c.varName) {
		return nil, fmt.Errorf("invalid -var %q", c.varName)
	}
	if c.out == "" {
		c.out = strings.TrimSuffix(c.sources[0], path.Ext(c.sources[0])) + "_precompiled.go"
	}
	return &c, nil
}

// splitJoined rewrites -DNAME and -Idir, as nvcc spells them, into the
// separate arguments the flag package expects.
func splitJoined(args []string) []string {
	var out []string
	for i, a := range args {
		if a == "--" || !strings.HasPrefix(a, "-") {
			return append(out, args[i:]...)
		}
		if len(a) > 2 && (a[:2] == "-D" || a[:2] == "-I") && a[2] != '=' {
			out = append(out, a[:2], a[2:])
			continue
		}
		out = append(out, a)
	}
	return out
}

// command returns a canonical command line for c, recorded in the
// generated file so that -check notices changed flags.
func (c *config) command() string {
	args := []string{"cudaprecompile", "-arch=" + c.arch}
	if c.ptx != "" {
		args = append(args, "-ptx="+c.ptx)
	}
	for _, d := range c.defines {
		args = append(args, "-D"+d)
	}
	for _, i := range c.include {
		args = append(args, "-I"+i)
	}
	if c.std != "" {
		args = append(args, "-std="+c.std)
	}
	for _, n := range c.names {
		args = append(args, "-name="+n)
	}
	if c.noLTO {
		args = append(args, "-nolto")
	}
	if c.tag != "" {
		args = append(args, "-tag="+c.tag)
	}
	if c.varName != "precompiled" {
		args = append(args, "-var="+c.varName)
	}
	return strings.Join(quote(append(args, c.sources...)), " ")
}

var identRE = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*$`)

func token(s string) bool { return identRE.MatchString(s) }

func (c *config) targets() (native []string, portable string, err error) {
	for _, a := range strings.Split(c.arch, ",") {
		a = strings.TrimSpace(a)
		if a == "" {
			continue
		}
		if !strings.HasPrefix(a, "sm_") || !slices.Contains(cudatool.KnownGPUArchitectures, a) {
			return nil, "", fmt.Errorf("unknown -arch %q", a)
		}
		native = append(native, a)
	}
	switch c.ptx {
	case "none":
	case "":
		low := 0
		for _, a := range native {
			n, err := strconv.Atoi(strings.TrimPrefix(a, "sm_"))
			if err == nil && (low == 0 || n < low) {
				low = n
			}
		}
		if low > 0 {
			portable = fmt.Sprintf("compute_%d", low)
		}
	default:
		if !strings.HasPrefix(c.ptx, "compute_") || !slices.Contains(cudatool.KnownGPUArchitectures, c.ptx) {
			return nil, "", fmt.Errorf("unknown -ptx %q", c.ptx)
		}
		portable = c.ptx
	}
	if len(native) == 0 && portable == "" {
		return nil, "", fmt.Errorf("no -arch or -ptx targets")
	}
	return native, portable, nil
}

func (c *config) options() []string {
	var opts []string
	for _, d := range c.defines {
		opts = append(opts, "-D"+d)
	}
	if c.std != "" {
		opts = append(opts, "-std="+c.std)
	}
	return opts
}

// local cleans a path given on the command line and requires it to stay
// within the package directory, since go:embed cannot reach beyond it.
func local(name string) (string, error) {
	p := filepath.ToSlash(filepath.Clean(name))
	if !filepath.IsLocal(p) {
		return "", fmt.Errorf("%s is outside the package directory", name)
	}
	return p, nil
}

func isHeader(name string) bool {
	switch path.Ext(name) {
	case ".h", ".cuh", ".hpp", ".inl":
		return true
	}
	return false
}

// headers returns the headers in the source directories and, recursively,
// in the include directories.
func headers(fsys fs.FS, sources, include []string) ([]string, error) {
	seen := make(map[string]bool)
	var list []string
	add := func(name string) {
		if !seen[name] {
			seen[name] = true
			list = append(list, name)
		}
	}
	dirs := make(map[string]bool)
	for _, s := range sources {
		dirs[path.Dir(s)] = true
	}
	for dir := range dirs {
		entries, err := fs.ReadDir(fsys, dir)
		if err != nil {
			return nil, err
		}
		for _, e := range entries {
			if !e.IsDir() && isHeader(e.Name()) {
				add(path.Join(dir, e.Name()))
			}
		}
	}
	for _, dir := range include {
		err := fs.WalkDir(fsys, dir, func(name string, d fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if !d.IsDir() && isHeader(name) {
				add(name)
			}
			return nil
		})
		if err != nil {
			return nil, err
		}
	}
	slices.Sort(list)
	return list, nil
}

func run(args []string, dir string, stdout, stderr io.Writer) error {
	c, err := parse(args, stderr)
	if err != nil {
		return err
	}
	native, portable, err := c.targets()
	if err != nil {
		return err
	}
	fsys := os.DirFS(dir)
	pc := &precompile.Config{Files: fsys, Options: c.options(), NoLTO: c.noLTO}
	for _, s := range c.sources {
		p, err := local(s)
		if err != nil {
			return err
		}
		pc.Sources = append(pc.Sources, p)
	}
	for _, i := range c.include {
		p, err := local(i)
		if err != nil {
			return err
		}
		pc.Include = append(pc.Include, p)
	}
	pc.Names = slices.Sorted(slices.Values(c.names))
	pc.Names = slices.Compact(pc.Names)
	if pc.Headers, err = headers(fsys, pc.Sources, pc.Include); err != nil {
		return err
	}
	sum, err := pc.Sum()
	if err != nil {
		return err
	}
	out, err := local(c.out)
	if err != nil {
		return err
	}
	stem := strings.TrimSuffix(out, ".go")
	g := &generated{
		Command: c.command(),
		Package: c.pkg,
		Var:     c.varName,
		Tag:     c.tag,
		Image:   path.Base(stem) + ".bin",
		Config:  pc,
		Sum:     sum,
	}
	if c.check {
		old, err := os.ReadFile(filepath.Join(dir, out))
		if err != nil || !bytes.Contains(old, []byte(g.header())) {
			fmt.Fprintf(stdout, "%s is out of date\n", out)
			return errStale
		}
		return nil
	}
	image, names, err := pc.Image(native, portable)
	if err != nil {
		return err
	}
	g.Names = names
	src, err := g.source(mainTemplate)
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(dir, stem+".bin"), image, 0o666); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(dir, out), src, 0o666); err != nil {
		return err
	}
	imageFile := filepath.Join(dir, stem+"_image.go")
	if c.tag == "" {
		if err := os.Remove(imageFile); err != nil && !os.IsNotExist(err) {
			return err
		}
		return nil
	}
	src, err = g.source(imageTemplate)
	if err != nil {
		return err
	}
	return os.WriteFile(imageFile, src, 0o666)
}

func quote(args []string) []string {
	q := make([]string, len(args))
	for i, a := range args {
		if a == "" || strings.ContainsAny(a, " \t\"'\\<>*?$") {
			a = strconv.Quote(a)
		}
		q[i] = a
	}
	return q
}

type generated struct {
	Command, Package, Var, Tag, Image, Sum string
	Config                                 *precompile.Config
	Names                                  map[string]string
}

// header identifies the inputs of a generated file. -check compares it.
func (g *generated) header() string {
	return fmt.Sprintf("// %s\n// %s\n", g.Command, g.Sum)
}

func (g *generated) Header() string { return g.header() }

// Embed lists the files to embed, which go:embed requires to be unquoted
// unless they contain spaces.
func (g *generated) Embed() string {
	files := append(slices.Clone(g.Config.Sources), g.Config.Headers...)
	return strings.Join(quote(files), " ")
}

func (g *generated) source(t *template.Template) ([]byte, error) {
	var buf bytes.Buffer
	if err := t.Execute(&buf, g); err != nil {
		return nil, err
	}
	src, err := format.Source(buf.Bytes())
	if err != nil {
		return nil, fmt.Errorf("format generated code: %w\n%s", err, buf.Bytes())
	}
	return src, nil
}

var funcs = template.FuncMap{"quote": strconv.Quote}

var mainTemplate = template.Must(template.New("main").Funcs(funcs).Parse(`// Code generated by cudaprecompile. DO NOT EDIT.

{{.Header}}
package {{.Package}}

import (
	"embed"

	"github.com/tmc/lib/nvidia/x/cudaprecompiled"
)

//go:embed {{.Embed}}
var {{.Var}}Files embed.FS
{{if not .Tag}}
//go:embed {{.Image}}
var {{.Var}}Image []byte
{{end}}
var {{.Var}} = &cudaprecompiled.Module{
	{{- if not .Tag}}
	Image: {{.Var}}Image,
	{{- end}}
	Sum: {{quote .Sum}},
	Files: {{.Var}}Files,
	Sources: []string{ {{- range .Config.Sources}}{{quote .}}, {{end -}} },
	{{- with .Config.Headers}}
	Headers: []string{ {{- range .}}{{quote .}}, {{end -}} },
	{{- end}}
	{{- with .Config.Include}}
	Include: []string{ {{- range .}}{{quote .}}, {{end -}} },
	{{- end}}
	{{- with .Config.Options}}
	Options: []string{ {{- range .}}{{quote .}}, {{end -}} },
	{{- end}}
	{{- if .Config.NoLTO}}
	NoLTO: true,
	{{- end}}
	{{- with .Names}}
	Names: map[string]string{
		{{- range $k, $v := .}}
		{{quote $k}}: {{quote $v}},
		{{- end}}
	},
	{{- end}}
}
`))

var imageTemplate = template.Must(template.New("image").Funcs(funcs).Parse(`// Code generated by cudaprecompile. DO NOT EDIT.

//go:build {{.Tag}}

package {{.Package}}

import _ "embed"

//go:embed {{.Image}}
var {{.Var}}Image []byte

func init() { {{.Var}}.Image = {{.Var}}Image }
`))
