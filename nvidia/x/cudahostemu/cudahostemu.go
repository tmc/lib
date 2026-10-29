package cudahostemu

import (
	"bytes"
	"context"
	"embed"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// Header names, as written by WriteHeaders.
const (
	Header     = "cudahostemu.h" // the emulator
	SGLangMock = "sgl_mock.h"    // SGLang and tvm-ffi host mock; includes Header
	UMMA       = "umma.h"        // tcgen05 descriptor-level MMA and TMEM; include after Header
)

//go:embed cudahostemu.h sgl_mock.h umma.h
var headers embed.FS

// Source returns the text of the named header.
func Source(name string) ([]byte, error) {
	return headers.ReadFile(name)
}

// WriteHeaders writes every header into dir, which must exist.
func WriteHeaders(dir string) error {
	for _, name := range []string{Header, SGLangMock, UMMA} {
		b, err := headers.ReadFile(name)
		if err != nil {
			return err
		}
		if err := os.WriteFile(filepath.Join(dir, name), b, 0o644); err != nil {
			return err
		}
	}
	return nil
}

// ErrNoCompiler is returned when no C++20 compiler is found.
var ErrNoCompiler = errors.New("cudahostemu: no C++ compiler (set CXX or install clang++)")

// Compiler returns the C++ compiler to use: $CXX if set, else the first of
// clang++, g++ and c++ on PATH.
func Compiler() (string, error) {
	if cxx := os.Getenv("CXX"); cxx != "" {
		return exec.LookPath(cxx)
	}
	for _, name := range []string{"clang++", "g++", "c++"} {
		if p, err := exec.LookPath(name); err == nil {
			return p, nil
		}
	}
	return "", ErrNoCompiler
}

// A Buffer is one kernel buffer. Run binds it in main as a pointer named
// Name of type CType*, initialized with Data, and returns its final
// contents, len(Data) bytes.
type Buffer struct {
	Name  string
	CType string // element type, such as "float" or "uint16_t"
	Data  []byte
}

// Poison returns n bytes of 0xff, NaN in every float format the header
// handles. Passing outputs in poisoned shows an element a kernel never
// writes.
func Poison(n int) []byte { return bytes.Repeat([]byte{0xff}, n) }

// A Program is kernel source and the statement that launches it.
type Program struct {
	Source  string   // kernels and launchers, compiled after the headers
	Launch  string   // C++ statements run in main once buffers are bound
	Buffers []Buffer // in the order they are bound
	Flags   []string // extra compiler flags, such as -DEG_EMU_ZERO_SMEM
	Mock    bool     // also include sgl_mock.h
}

// Run compiles and runs p in dir, a scratch directory, and returns every
// buffer's final contents by name. Flags in $CUDAHOSTEMU_FLAGS, split on
// spaces, follow p.Flags, so that a caller can rebuild every program
// differently (-DEG_EMU_ZERO_SMEM, say) without editing it.
func Run(ctx context.Context, dir string, p Program) (map[string][]byte, error) {
	cxx, err := Compiler()
	if err != nil {
		return nil, err
	}
	if err := WriteHeaders(dir); err != nil {
		return nil, err
	}
	var main strings.Builder
	main.WriteString("#include \"" + Header + "\"\n")
	if p.Mock {
		main.WriteString("#include \"" + SGLangMock + "\"\n")
	}
	main.WriteString("#line 1 \"source\"\n")
	main.WriteString(p.Source)
	main.WriteString("\n#line 1 \"main\"\nint main() {\n")
	for _, b := range p.Buffers {
		in := filepath.Join(dir, b.Name+".bin")
		if err := os.WriteFile(in, b.Data, 0o644); err != nil {
			return nil, err
		}
		fmt.Fprintf(&main, "    auto cudahostemu_b_%s = eg_emu_read(%q);\n    auto* %s = reinterpret_cast<%s*>(cudahostemu_b_%s.data());\n",
			b.Name, in, b.Name, b.CType, b.Name)
	}
	fmt.Fprintf(&main, "    %s;\n", p.Launch)
	for _, b := range p.Buffers {
		fmt.Fprintf(&main, "    eg_emu_write(%q, %s, %d);\n", filepath.Join(dir, b.Name+".out"), b.Name, len(b.Data))
	}
	main.WriteString("    return 0;\n}\n")
	cc := filepath.Join(dir, "main.cc")
	if err := os.WriteFile(cc, []byte(main.String()), 0o644); err != nil {
		return nil, err
	}
	bin := filepath.Join(dir, "main")
	args := append([]string{"-std=c++20", "-O1", "-w", "-pthread", "-I", dir, "-o", bin}, p.Flags...)
	args = append(args, strings.Fields(os.Getenv("CUDAHOSTEMU_FLAGS"))...)
	args = append(args, cc)
	if out, err := exec.CommandContext(ctx, cxx, args...).CombinedOutput(); err != nil {
		return nil, fmt.Errorf("compile: %v\n%s", err, out)
	}
	var out bytes.Buffer
	cmd := exec.CommandContext(ctx, bin)
	cmd.Stdout, cmd.Stderr = &out, &out
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("run: %v\n%s", err, out.Bytes())
	}
	res := make(map[string][]byte, len(p.Buffers))
	for _, b := range p.Buffers {
		data, err := os.ReadFile(filepath.Join(dir, b.Name+".out"))
		if err != nil {
			return nil, err
		}
		res[b.Name] = data
	}
	return res, nil
}
