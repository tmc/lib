package cudasass

import (
	"bytes"
	"context"
	"debug/elf"
	"errors"
	"fmt"
	"os"
	"path/filepath"

	"github.com/tmc/lib/nvidia/x/internal/cudatool"
)

// Output contains command identity and diagnostics, including on command failure.
type Output = cudatool.Output

// Mode selects one mutually exclusive nvdisasm output form.
type Mode int

const (
	Text Mode = iota
	JSON
	ControlFlow
	BasicBlockFlow
	RegisterLife
)

// Options selects disassembly output. Line annotations require Text mode and
// corresponding debug sections in the input; absent sections yield no annotations.
type Options struct {
	Mode              Mode
	PrintCode         bool
	PrintHex          bool
	SeparateFunctions bool
	SourceLines       bool
	InlineLines       bool
	PTXLines          bool
}

// Disassembler invokes nvdisasm. The zero value searches PATH and uses the
// operating system's temporary directory. TempDir selects a scratch parent.
// Methods do not load code onto a GPU. Tool option support is version dependent.
type Disassembler struct{ Path, TempDir string }

func (d Disassembler) path() string {
	if d.Path != "" {
		return d.Path
	}
	return "nvdisasm"
}

// Version queries the selected executable; both output streams are retained.
func (d Disassembler) Version(ctx context.Context) (Output, error) {
	return cudatool.Run(ctx, d.path(), "", nil, "--version")
}

func (o Options) args() ([]string, error) {
	var args []string
	switch o.Mode {
	case Text:
	case JSON:
		args = append(args, "--emit-json")
	case ControlFlow:
		args = append(args, "--output-control-flow-graph")
	case BasicBlockFlow:
		args = append(args, "--output-control-flow-graph-with-basic-blocks")
	case RegisterLife:
		args = append(args, "--print-life-ranges")
	default:
		return nil, fmt.Errorf("unknown disassembly mode %d", o.Mode)
	}
	if o.Mode != Text && (o.SourceLines || o.InlineLines || o.PTXLines) {
		return nil, fmt.Errorf("line annotations require text mode")
	}
	if o.PrintCode {
		args = append(args, "--print-code")
	}
	if o.PrintHex {
		args = append(args, "--print-instruction-encoding")
	}
	if o.SeparateFunctions {
		args = append(args, "--separate-functions")
	}
	if o.InlineLines {
		args = append(args, "--print-line-info-inline")
	} else if o.SourceLines {
		args = append(args, "--print-line-info")
	}
	if o.PTXLines {
		args = append(args, "--print-line-info-ptx")
	}
	return args, nil
}

// Disassemble validates a CUDA ELF image and disassembles a private file copy.
// Scratch files are removed before return. Output remains available on failure.
func (d Disassembler) Disassemble(ctx context.Context, cubin []byte, opts Options) (out Output, err error) {
	args, err := opts.args()
	if err != nil {
		return Output{}, err
	}
	f, err := elf.NewFile(bytes.NewReader(cubin))
	if err != nil {
		return Output{}, fmt.Errorf("read cubin: %w", err)
	}
	defer f.Close()
	if f.Machine != elf.EM_CUDA {
		return Output{}, fmt.Errorf("cubin machine is %s, want CUDA", f.Machine)
	}
	dir, err := os.MkdirTemp(d.TempDir, "cudasass-")
	if err != nil {
		return Output{}, fmt.Errorf("create scratch: %w", err)
	}
	defer func() {
		if e := os.RemoveAll(dir); e != nil {
			err = errors.Join(err, fmt.Errorf("remove scratch %s: %w", dir, e))
		}
	}()
	path := filepath.Join(dir, "input.cubin")
	if err := os.WriteFile(path, cubin, 0600); err != nil {
		return Output{}, err
	}
	return cudatool.Run(ctx, d.path(), "", nil, append(args, path)...)
}

// Demangler selects cu++filt. Its zero value searches PATH.
type Demangler struct{ Path string }

func (d Demangler) path() string {
	if d.Path != "" {
		return d.Path
	}
	return "cu++filt"
}

// Version reports cu++filt's version. Unlike nvdisasm it uses the -v option.
func (d Demangler) Version(ctx context.Context) (Output, error) {
	return cudatool.Run(ctx, d.path(), "", nil, "-v")
}

// Demangle runs cu++filt on newline-delimited symbols through stdin, so symbols
// cannot be interpreted as command options. Unrecognized names pass through
// according to cu++filt's behavior; success does not prove every name was mangled.
func (d Demangler) Demangle(ctx context.Context, symbols []string) (Output, error) {
	var input bytes.Buffer
	for _, s := range symbols {
		if bytes.ContainsAny([]byte(s), "\x00\r\n") {
			return Output{}, fmt.Errorf("symbol contains a line separator or nul")
		}
		input.WriteString(s)
		input.WriteByte('\n')
	}
	return cudatool.Run(ctx, d.path(), "", &input)
}
