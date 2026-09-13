package cudasass

import (
	"bytes"
	"context"
	"debug/elf"
	"encoding/binary"
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
	Mode                       Mode
	LifeRangeMode              string // "count", "narrow", or "wide" for RegisterLife mode
	PrintCode                  bool
	PrintHex                   bool
	SeparateFunctions          bool
	PrintInstructionOffsetsCFG bool
	SourceLines                bool
	InlineLines                bool
	PTXLines                   bool
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

// ErrUnsupportedOption means the resolved nvdisasm does not accept an option the
// requested [Options] would emit, by name or by arity. [Disassembler.Disassemble]
// reports it instead of running a command line the tool would reject.
var ErrUnsupportedOption = cudatool.ErrUnsupportedOption

// ToolName is the executable [Disassembler] runs. It is also the name
// [cudatool.Capabilities] records option sets under.
const ToolName = "nvdisasm"

// flags returns the options the command line emits, in order, without running
// anything. It is the single account of what a command line contains, so the
// capability check and the executed arguments cannot describe different runs.
func (o Options) flags() ([]cudatool.Flag, error) {
	var flags []cudatool.Flag
	add := func(name string) { flags = append(flags, cudatool.Flag{Name: name}) }
	switch o.Mode {
	case Text:
	case JSON:
		add("--emit-json")
	case ControlFlow:
		add("--output-control-flow-graph")
	case BasicBlockFlow:
		add("--output-control-flow-graph-with-basic-blocks")
	case RegisterLife:
		add("--print-life-ranges")
	default:
		return nil, fmt.Errorf("unknown disassembly mode %d", o.Mode)
	}
	if o.LifeRangeMode != "" {
		if o.Mode != RegisterLife {
			return nil, fmt.Errorf("life range mode requires register life mode")
		}
		flags = append(flags, cudatool.Flag{Name: "--life-range-mode", Value: o.LifeRangeMode})
	}
	if o.Mode != Text && (o.SourceLines || o.InlineLines || o.PTXLines) {
		return nil, fmt.Errorf("line annotations require text mode")
	}
	if o.PrintInstructionOffsetsCFG {
		if o.Mode != ControlFlow && o.Mode != BasicBlockFlow {
			return nil, fmt.Errorf("instruction offsets cfg requires control flow mode")
		}
		add("--print-instr-offsets-cfg")
	}
	if o.PrintCode {
		add("--print-code")
	}
	if o.PrintHex {
		add("--print-instruction-encoding")
	}
	if o.SeparateFunctions {
		add("--separate-functions")
	}
	if o.InlineLines {
		add("--print-line-info-inline")
	} else if o.SourceLines {
		add("--print-line-info")
	}
	if o.PTXLines {
		add("--print-line-info-ptx")
	}
	return flags, nil
}

// Capabilities reports what the selected nvdisasm accepts. The answer comes
// from the option sets recorded for the toolkit the executable belongs to, or
// from its own --help for a toolkit newer than any recorded one.
func (d Disassembler) Capabilities(ctx context.Context) (cudatool.Caps, error) {
	return cudatool.Capabilities(ctx, ToolName, d.Path)
}

// ControlFlowGraph disassembles a CUDA ELF image into a Graphviz DOT control flow graph.
// If basicBlocks is true, basic blocks are generated instead of hyperblocks.
// Instruction offsets are included in the graph nodes.
func (d Disassembler) ControlFlowGraph(ctx context.Context, cubin []byte, basicBlocks bool) (Output, error) {
	mode := ControlFlow
	if basicBlocks {
		mode = BasicBlockFlow
	}
	return d.Disassemble(ctx, cubin, Options{
		Mode:                       mode,
		PrintInstructionOffsetsCFG: true,
	})
}

// checkCUBIN reports whether cubin identifies itself as a CUDA ELF image.
//
// It reads the header directly instead of calling [elf.NewFile], which rejects
// any e_version but EV_CURRENT. nvcc writes its own toolkit version into that
// field through CUDA 12: a 12.6 cubin carries 126 and a 13.4 cubin carries 1,
// measured with nvcc -cubin on the GB10 host, so elf.NewFile fails on every
// 12.x cubin with "mismatched ELF version" even though nvdisasm reads it.
// Nothing here needs the section table, only the identification bytes and
// e_machine, which sit at the same offsets in both ELF classes.
func checkCUBIN(cubin []byte) error {
	// Header sizes for the two classes, the minimum a complete header needs.
	const ehsize32, ehsize64 = 52, 64
	if len(cubin) < ehsize32 || string(cubin[:len(elf.ELFMAG)]) != elf.ELFMAG {
		return errors.New("not an ELF image")
	}
	switch class := elf.Class(cubin[elf.EI_CLASS]); class {
	case elf.ELFCLASS32:
	case elf.ELFCLASS64:
		if len(cubin) < ehsize64 {
			return errors.New("truncated 64-bit ELF header")
		}
	default:
		return fmt.Errorf("ELF class is %s, want 32-bit or 64-bit", class)
	}
	if data := elf.Data(cubin[elf.EI_DATA]); data != elf.ELFDATA2LSB {
		return fmt.Errorf("ELF data encoding is %s, want little-endian", data)
	}
	// e_machine follows the 16-byte identification and the 2-byte e_type.
	if m := elf.Machine(binary.LittleEndian.Uint16(cubin[18:])); m != elf.EM_CUDA {
		return fmt.Errorf("cubin machine is %s, want CUDA", m)
	}
	return nil
}

// Disassemble validates a CUDA ELF image and disassembles a private file copy.
// Scratch files are removed before return. Output remains available on failure.
func (d Disassembler) Disassemble(ctx context.Context, cubin []byte, opts Options) (out Output, err error) {
	flags, err := opts.flags()
	if err != nil {
		return Output{}, err
	}
	// A determinate answer about the tool is enforced; an indeterminate one
	// is not. When the executable does not report a version and its --help
	// cannot be read, nothing is known about what it accepts, and refusing to
	// run it would be a claim this package cannot make. The tool's own
	// diagnosis stands in that case.
	if caps, err := d.Capabilities(ctx); err == nil {
		if err := caps.Check(flags...); err != nil {
			return Output{}, err
		}
	}
	if err := checkCUBIN(cubin); err != nil {
		return Output{}, fmt.Errorf("read cubin: %w", err)
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
	return cudatool.Run(ctx, d.path(), "", nil, append(cudatool.Args(flags...), path)...)
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
