package tileirdisasm

import (
	"bytes"
	"context"
	"debug/elf"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/tmc/lib/nvidia/x/internal/cudatool"
)

// Option configures execution of the tileirdisasm disassembler.
type Option func(*config)

type config struct {
	disasmPath string
	debugInfo  bool
	tempDir    string
}

// Output retains the disassembler executable, arguments and both output streams.
type Output = cudatool.Output

// WithTempDir selects a scratch parent. Empty uses the operating system default.
func WithTempDir(dir string) Option {
	return func(c *config) { c.tempDir = dir }
}

// WithPath specifies an explicit path to the tileirdisasm executable.
func WithPath(path string) Option {
	return func(c *config) {
		c.disasmPath = path
	}
}

// WithPrintDebugInfo specifies whether to annotate disassembly with debug locations (-g).
func WithPrintDebugInfo(debug bool) Option {
	return func(c *config) {
		c.debugInfo = debug
	}
}

// ErrNotFound is returned when the tileirdisasm executable cannot be located in PATH or standard CUDA directories.
var ErrNotFound = errors.New("tileirdisasm: executable not found in PATH or standard CUDA directories")

// Disassemble disassembles Tile IR bytecode using tileirdisasm and parses it into a Module AST.
func Disassemble(ctx context.Context, bytecode []byte, opts ...Option) (*Module, error) {
	m, _, err := DisassembleWithOutput(ctx, bytecode, opts...)
	return m, err
}

// DisassembleWithOutput disassembles and parses bytecode while retaining raw
// output even when execution or parsing fails. Scratch files are removed before
// return; cleanup errors are joined with execution or parsing errors.
func DisassembleWithOutput(ctx context.Context, bytecode []byte, opts ...Option) (module *Module, output Output, err error) {
	if len(bytecode) == 0 {
		return nil, output, errors.New("tileirdisasm: empty bytecode input")
	}
	if err := ctx.Err(); err != nil {
		return nil, output, err
	}

	cfg := config{
		debugInfo: true, // Default to extracting debug info if present
	}
	for _, opt := range opts {
		opt(&cfg)
	}

	disasmBin := findTileirdisasm(cfg.disasmPath)
	if disasmBin == "" {
		return nil, output, ErrNotFound
	}

	tmpDir, err := os.MkdirTemp(cfg.tempDir, "tileirdisasm-*")
	if err != nil {
		return nil, output, fmt.Errorf("tileirdisasm: creating temp dir: %w", err)
	}
	defer func() {
		if cleanup := os.RemoveAll(tmpDir); cleanup != nil {
			err = errors.Join(err, fmt.Errorf("tileirdisasm: remove scratch: %w", cleanup))
		}
	}()

	inputPath := filepath.Join(tmpDir, "input.tilebc")
	if err := os.WriteFile(inputPath, bytecode, 0600); err != nil {
		return nil, output, fmt.Errorf("tileirdisasm: writing input: %w", err)
	}

	args := []string{}
	if cfg.debugInfo {
		args = append(args, "-g")
	}
	args = append(args, inputPath)

	output, err = cudatool.Run(ctx, disasmBin, "", nil, args...)
	if err != nil {
		return nil, output, err
	}
	module, err = Parse(bytes.NewReader(output.Stdout))
	return module, output, err
}

// DisassembleFile reads a Tile IR bytecode file and returns its parsed Module AST.
func DisassembleFile(ctx context.Context, path string, opts ...Option) (*Module, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("tileirdisasm: reading file %q: %w", path, err)
	}
	return Disassemble(ctx, data, opts...)
}

// ExtractFromELF searches an ELF image (such as a CUBIN) for embedded Tile IR bytecode.
func ExtractFromELF(elfBytes []byte) ([]byte, error) {
	ef, err := elf.NewFile(bytes.NewReader(elfBytes))
	if err != nil {
		return nil, fmt.Errorf("tileirdisasm: parsing ELF: %w", err)
	}
	defer ef.Close()

	candidates := []string{
		".nv.tileir",
		".nv_tile_ir",
		".tileir",
		".tilebc",
	}

	for _, name := range candidates {
		if sec := ef.Section(name); sec != nil {
			data, err := sec.Data()
			if err != nil {
				return nil, fmt.Errorf("tileirdisasm: reading section %s: %w", name, err)
			}
			return data, nil
		}
	}

	// Also check sections starting with .nv.tileir or containing TileIR
	for _, sec := range ef.Sections {
		if strings.Contains(sec.Name, "tileir") || strings.Contains(sec.Name, "tilebc") {
			data, err := sec.Data()
			if err != nil {
				continue
			}
			return data, nil
		}
	}

	return nil, errors.New("tileirdisasm: no Tile IR section found in ELF image")
}

func findTileirdisasm(custom string) string {
	if custom != "" {
		return custom
	}
	if p, err := exec.LookPath("tileirdisasm"); err == nil {
		return p
	}
	candidates := []string{
		"/usr/local/cuda-13.4/bin/tileirdisasm",
		"/usr/local/cuda-13.3/bin/tileirdisasm",
		"/usr/local/cuda-13.2/bin/tileirdisasm",
		"/usr/local/cuda-13.1/bin/tileirdisasm",
		"/usr/local/cuda/bin/tileirdisasm",
	}
	for _, c := range candidates {
		if fi, err := os.Stat(c); err == nil && !fi.IsDir() {
			return c
		}
	}
	return ""
}
