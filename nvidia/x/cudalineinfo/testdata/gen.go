//go:build ignore

// Gen regenerates the fixtures of this directory. The fused_gate and
// fused_layered units come from the source-wiring example, which builds its
// kernels with cudatensor and lowers them with cudafuse; taking them from
// that path rather than writing them by hand is what makes the fixtures carry
// the function and callers of every line as the compiler recorded them. The
// forward fixture is compiled from forward.cu, the hand-written CUDA the
// package's tests embed.
//
// Usage:
//
//	go run testdata/gen.go [-arch sm_121]
//
// It needs a GPU with NVRTC and an nvdisasm that understands the target. The
// example records absolute paths, so each unit's files are rewritten relative
// to the module root before anything is compiled: the checked-in artifacts
// then name the same files on every machine.
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudalineinfo"
	"github.com/tmc/lib/nvidia/x/cudasass"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// moduleRoot is the directory the rewritten paths are relative to, as seen
// from this program's working directory, which is the package directory.
const moduleRoot = "../.."

func main() {
	log.SetFlags(0)
	log.SetPrefix("gen: ")
	arch := flag.String("arch", "sm_121", "physical architecture to compile for")
	flag.Parse()
	if err := run(context.Background(), *arch); err != nil {
		log.Fatal(err)
	}
}

func run(ctx context.Context, arch string) error {
	dir, err := os.MkdirTemp("", "cudalineinfo-fixtures")
	if err != nil {
		return err
	}
	defer os.RemoveAll(dir)
	cmd := exec.CommandContext(ctx, "go", "run", "./x/examples/source-wiring", "-units", dir)
	cmd.Dir = moduleRoot
	cmd.Env = append(os.Environ(), "GOWORK=off")
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("run source-wiring: %w", err)
	}
	for _, name := range []string{"fused_gate", "fused_layered"} {
		unit, err := readUnit(filepath.Join(dir, name+".json"))
		if err != nil {
			return err
		}
		data, err := json.MarshalIndent(unit, "", "\t")
		if err != nil {
			return err
		}
		if err := os.WriteFile("testdata/"+name+".json", append(data, '\n'), 0o644); err != nil {
			return err
		}
		if err := os.WriteFile("testdata/"+name+".cu", []byte(unit.SourceWithLineDirectives()), 0o644); err != nil {
			return err
		}
		if err := write(ctx, unit, arch, "testdata/"+name, nil); err != nil {
			return err
		}
	}
	// The fast-math build of the first kernel is the other side of the diff
	// the package reports, and the example does not export it.
	unit, err := readUnit(filepath.Join(dir, "fused_gate.json"))
	if err != nil {
		return err
	}
	if err := write(ctx, unit, arch, "testdata/fused_gate-fast", []nvrtcx.Option{nvrtcx.WithFastMath()}); err != nil {
		return err
	}
	return forward(ctx, arch)
}

// forward compiles testdata/forward.cu, which is hand-written CUDA rather
// than a lowered unit: it has no line map, so its own lines are the program.
// The package's tests embed the same file, and disassembling it is what tells
// them the spellings a compiler really emits. Its kernels have C++ linkage,
// and matching a caller's geglu against the _Z5gegluPKfPfi that comes back is
// the case a hand-authored fixture is free to leave out.
func forward(ctx context.Context, arch string) error {
	src, err := os.ReadFile("testdata/forward.cu")
	if err != nil {
		return err
	}
	unit := cudalineinfo.Unit{Name: "forward", Source: string(src), Kernels: []string{"geglu", "rmsnorm"}}
	return write(ctx, unit, arch, "testdata/forward", nil)
}

// write compiles the unit and writes its PTX and disassembly under prefix.
//
// NVRTC resolves the relative names of the #line directives against the
// working directory, so the artifacts come back naming absolute paths of the
// machine that built them. Stripping that prefix again leaves the names the
// unit uses, which is what makes the fixtures readable anywhere.
func write(ctx context.Context, unit cudalineinfo.Unit, arch, prefix string, extra []nvrtcx.Option) error {
	a, err := cudalineinfo.Compile(ctx, unit, cudalineinfo.Options{Arch: arch, NVRTC: extra})
	if err != nil {
		return err
	}
	sass, err := cudasass.Disassembler{}.Disassemble(ctx, a.Module.CUBIN(), cudasass.Options{PrintCode: true, SourceLines: true})
	if err != nil {
		return fmt.Errorf("disassemble %s: %w", unit.Name, err)
	}
	cwd, err := os.Getwd()
	if err != nil {
		return err
	}
	strip := func(text []byte) []byte {
		return bytes.ReplaceAll(text, []byte(cwd+string(filepath.Separator)), nil)
	}
	for suffix, data := range map[string][]byte{
		".ptx":  []byte(a.PTX),
		".sass": sass.Stdout,
	} {
		if err := os.WriteFile(prefix+suffix, strip(data), 0o644); err != nil {
			return err
		}
	}
	return nil
}

// readUnit reads an exported unit and rewrites every path it names relative
// to the module root.
func readUnit(path string) (cudalineinfo.Unit, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return cudalineinfo.Unit{}, err
	}
	var u cudalineinfo.Unit
	if err := json.Unmarshal(data, &u); err != nil {
		return u, fmt.Errorf("decode %s: %w", path, err)
	}
	root, err := filepath.Abs(moduleRoot)
	if err != nil {
		return u, err
	}
	rel := func(file string) string {
		if r, err := filepath.Rel(root, file); err == nil && !strings.HasPrefix(r, "..") {
			return r
		}
		return file
	}
	for n, loc := range u.LineMap {
		loc.File = rel(loc.File)
		callers := make([]cudair.Frame, len(loc.Callers))
		for i, c := range loc.Callers {
			c.File = rel(c.File)
			callers[i] = c
		}
		loc.Callers = callers
		u.LineMap[n] = loc
	}
	return u, nil
}
