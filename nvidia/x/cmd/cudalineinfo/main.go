// Cudalineinfo reports which program lines a compiled kernel came from.
//
// The input is one compilation unit: generated CUDA C++ annotated with #line
// directives that name the program's own files, or the JSON form of a
// [cudalineinfo.Unit], which also carries the function and callers of every
// line. The tool compiles the unit with NVRTC and disassembles it with
// nvdisasm, or reads artifacts already built, and prints how each line fared.
//
// Usage:
//
//	cudalineinfo -arch sm_121 kernel.cu                 instructions per line
//	cudalineinfo -arch sm_121 -format listing unit.json  every instruction under its line
//	cudalineinfo -arch sm_121 -format rollup unit.json   lines charged to their callers
//	cudalineinfo -arch sm_121 -diff -use_fast_math k.cu  opcodes a second build adds and drops
//	cudalineinfo -sass k.sass -ptx k.ptx k.cu            attribute artifacts without compiling
//	cudalineinfo -cubin k.cubin k.cu                     disassemble a CUBIN instead
//
// The formats digraph and dot print the unit as a call graph: program
// functions to the functions they called, down to the line that built each
// operation, the line to the kernel, and lines and kernel to the SASS
// subroutines their instructions call. The digraph form is what
// golang.org/x/tools/cmd/digraph reads, so
//
//	cudalineinfo -format digraph unit.json | digraph somepath main.main '$__internal_0_$__cuda_sm3x_div_rn_noftz_f32_slowpath'
//
// names the Go line that pulled the division slowpath into the kernel. The
// graph holds the paths that built the unit, not every path the program
// has; golang.org/x/tools/cmd/callgraph -format digraph prints those, and
// the two outputs concatenate, since the program's functions are named the
// same way in both.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudalineinfo"
	"github.com/tmc/lib/nvidia/x/cudasass"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

var (
	arch     = flag.String("arch", "", "physical architecture to compile for, such as sm_121")
	nvdisasm = flag.String("nvdisasm", "", "path to nvdisasm (default: search PATH)")
	nvrtc    = flag.String("nvrtc", "", "further NVRTC flags, space separated")
	sassFile = flag.String("sass", "", "nvdisasm output to attribute instead of compiling")
	ptxFile  = flag.String("ptx", "", "PTX to attribute with -sass or -cubin")
	cubin    = flag.String("cubin", "", "CUBIN to disassemble instead of compiling")
	format   = flag.String("format", "table", "output: table, listing, rollup, digraph, or dot")
	diff     = flag.String("diff", "", "NVRTC flags of a second build to compare with, such as -use_fast_math")
)

func usage() {
	fmt.Fprintln(os.Stderr, "usage: cudalineinfo [flags] unit.cu|unit.json")
	flag.PrintDefaults()
	os.Exit(2)
}

func main() {
	flag.Usage = usage
	flag.Parse()
	if flag.NArg() != 1 {
		usage()
	}
	if err := run(context.Background(), flag.Arg(0), os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, "cudalineinfo:", err)
		os.Exit(1)
	}
}

func run(ctx context.Context, path string, w io.Writer) error {
	unit, err := readUnit(path)
	if err != nil {
		return err
	}
	a, err := attribute(ctx, unit, nil)
	if err != nil {
		return err
	}
	if *diff != "" {
		if *sassFile != "" || *cubin != "" {
			return errors.New("-diff compiles a second build and needs -arch, not -sass or -cubin")
		}
		target, err := attribute(ctx, unit, strings.Fields(*diff))
		if err != nil {
			return fmt.Errorf("second build: %w", err)
		}
		return cudalineinfo.WriteDiff(w, cudalineinfo.Side{Attribution: a, Label: "default"}, cudalineinfo.Side{Attribution: target, Label: *diff})
	}
	switch *format {
	case "table":
		return cudalineinfo.WriteTable(w, a)
	case "listing":
		return cudalineinfo.WriteListing(w, a)
	case "rollup":
		return cudalineinfo.WriteRollup(w, a)
	case "digraph":
		return cudalineinfo.WriteDigraph(w, a)
	case "dot":
		return cudalineinfo.WriteDot(w, a)
	}
	return fmt.Errorf("unknown format %q", *format)
}

// readUnit reads a unit from its JSON form or from annotated source, which
// is named after the file.
func readUnit(path string) (cudalineinfo.Unit, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return cudalineinfo.Unit{}, err
	}
	if filepath.Ext(path) == ".json" {
		var u cudalineinfo.Unit
		if err := json.Unmarshal(data, &u); err != nil {
			return cudalineinfo.Unit{}, fmt.Errorf("decode %s: %w", path, err)
		}
		if u.Name == "" {
			return cudalineinfo.Unit{}, fmt.Errorf("%s: unit has no name", path)
		}
		return u, nil
	}
	name := strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))
	return cudalineinfo.ParseUnit(name, string(data)), nil
}

// attribute joins the unit to its artifacts: the SASS and PTX given, the
// CUBIN given after disassembling it, or a fresh compilation with the extra
// NVRTC flags.
func attribute(ctx context.Context, unit cudalineinfo.Unit, extra []string) (*cudalineinfo.Attribution, error) {
	switch {
	case *sassFile != "":
		sass, err := os.ReadFile(*sassFile)
		if err != nil {
			return nil, err
		}
		dis, err := cudasass.ParseString(string(sass))
		if err != nil {
			return nil, fmt.Errorf("parse %s: %w", *sassFile, err)
		}
		var ptx []byte
		if *ptxFile != "" {
			if ptx, err = os.ReadFile(*ptxFile); err != nil {
				return nil, err
			}
		}
		return cudalineinfo.Attribute(unit, string(ptx), dis), nil
	case *cubin != "":
		data, err := os.ReadFile(*cubin)
		if err != nil {
			return nil, err
		}
		out, err := cudasass.Disassembler{Path: *nvdisasm}.Disassemble(ctx, data, cudasass.Options{PrintCode: true, SourceLines: true})
		if err != nil {
			return nil, fmt.Errorf("disassemble %s: %w", *cubin, err)
		}
		dis, err := cudasass.ParseString(string(out.Stdout))
		if err != nil {
			return nil, fmt.Errorf("parse disassembly of %s: %w", *cubin, err)
		}
		var ptx []byte
		if *ptxFile != "" {
			if ptx, err = os.ReadFile(*ptxFile); err != nil {
				return nil, err
			}
		}
		return cudalineinfo.Attribute(unit, string(ptx), dis), nil
	}
	if *arch == "" {
		return nil, errors.New("compiling needs -arch; or give -sass or -cubin")
	}
	opts := cudalineinfo.Options{Arch: *arch, Nvdisasm: *nvdisasm}
	for _, f := range append(strings.Fields(*nvrtc), extra...) {
		opts.NVRTC = append(opts.NVRTC, nvrtcx.WithFlag(f))
	}
	return cudalineinfo.Compile(ctx, unit, opts)
}
