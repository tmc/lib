package main

import (
	"errors"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
)

// config is what the flags ask for.
type config struct {
	file          string // the CUDA source, relative to dir
	dir           string // the package directory
	output        string // the Go file, relative to dir
	pkg           string
	typeName      string
	prefix        string
	includes      []string // relative to dir
	defines       []string
	std           string
	defaultDevice bool
	kernels       []string
	globals       []string
	macros        []string
	shared        []string
	arch          string
	libclang      string
	args          []string // the command line, for the generated header
}

func main() {
	log.SetFlags(0)
	log.SetPrefix("cudabind: ")
	cfg, err := newConfig(os.Args[1:])
	if errors.Is(err, flag.ErrHelp) {
		os.Exit(2)
	}
	if err != nil {
		log.Fatal(err)
	}
	if err := run(cfg); err != nil {
		log.Fatal(err)
	}
}

// newConfig returns the configuration the command line args ask for.
func newConfig(args []string) (*config, error) {
	cfg := &config{args: args}
	fs := flag.NewFlagSet("cudabind", flag.ContinueOnError)
	repeated := func(name, usage string, p *[]string) {
		fs.Func(name, usage, func(s string) error { *p = append(*p, s); return nil })
	}
	list := func(name, usage string, p *[]string) {
		fs.Func(name, usage, func(s string) error { *p = strings.Split(s, ","); return nil })
	}
	fs.StringVar(&cfg.typeName, "type", "", "Go type of the loaded module")
	fs.StringVar(&cfg.prefix, "prefix", "", "prefix of the Go names of C structs")
	fs.StringVar(&cfg.output, "o", "", "output `file` (default file.cu.gen.go)")
	repeated("I", "include `dir`, repeatable", &cfg.includes)
	repeated("D", "macro `name[=value]`, repeatable", &cfg.defines)
	fs.StringVar(&cfg.std, "std", "", "C++ standard, such as c++20")
	fs.BoolVar(&cfg.defaultDevice, "default-device", false, "compile unannotated functions as device functions")
	list("kernels", "comma-separated kernels to bind (default all)", &cfg.kernels)
	list("globals", "comma-separated variables to bind (default all)", &cfg.globals)
	list("macros", "comma-separated macros to bind (default all that are numbers)", &cfg.macros)
	list("shared", "comma-separated kernels that take dynamic shared memory", &cfg.shared)
	fs.StringVar(&cfg.arch, "arch", "sm_80", "architecture clang parses for")
	fs.StringVar(&cfg.libclang, "libclang", os.Getenv("CUDABIND_LIBCLANG"), "`path` of libclang")
	fs.Usage = func() {
		fmt.Fprintf(fs.Output(), "usage: cudabind -type name [flags] file.cu\n")
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		return nil, err
	}
	if fs.NArg() != 1 || cfg.typeName == "" {
		fs.Usage()
		return nil, flag.ErrHelp
	}
	cfg.file = fs.Arg(0)
	if cfg.output == "" {
		cfg.output = cfg.file + ".gen.go"
	}
	var err error
	if cfg.dir, err = filepath.Abs(filepath.Dir(cfg.output)); err != nil {
		return nil, err
	}
	cfg.output = filepath.Base(cfg.output)
	cfg.pkg = os.Getenv("GOPACKAGE")
	if cfg.pkg == "" {
		cfg.pkg = filepath.Base(cfg.dir)
	}
	return cfg, nil
}

// run parses cfg.file and writes the binding and its test.
func run(cfg *config) error {
	var err error
	if cfg.file, err = relative(cfg.dir, cfg.file); err != nil {
		return err
	}
	for i, dir := range cfg.includes {
		if cfg.includes[i], err = relative(cfg.dir, dir); err != nil {
			return err
		}
	}
	m, err := parse(cfg)
	if err != nil {
		return err
	}
	src, test, err := generate(cfg, m)
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(cfg.dir, cfg.output), src, 0o666); err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(cfg.dir, testName(cfg.output)), test, 0o666)
}

// relative returns path relative to dir, which must contain it, since the
// generated file embeds it.
func relative(dir, path string) (string, error) {
	abs, err := filepath.Abs(path)
	if err != nil {
		return "", err
	}
	rel, err := filepath.Rel(dir, abs)
	if err != nil || !filepath.IsLocal(rel) {
		return "", fmt.Errorf("%s is outside %s, so cannot be embedded", path, dir)
	}
	return filepath.ToSlash(rel), nil
}

// testName returns the name of the test file generated beside output.
func testName(output string) string {
	return strings.TrimSuffix(output, ".go") + "_test.go"
}
