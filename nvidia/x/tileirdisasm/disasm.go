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
)

// Option configures execution of the tileirdisasm disassembler.
type Option func(*config)

type config struct {
	disasmPath string
	debugInfo  bool
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

// Disassemble disassembles Tile IR bytecode using tileirdisasm and parses it into a Module AST.
func Disassemble(ctx context.Context, bytecode []byte, opts ...Option) (*Module, error) {
	if len(bytecode) == 0 {
		return nil, errors.New("tileirdisasm: empty bytecode input")
	}

	cfg := config{
		debugInfo: true, // Default to extracting debug info if present
	}
	for _, opt := range opts {
		opt(&cfg)
	}

	disasmBin := findTileirdisasm(cfg.disasmPath)
	if disasmBin == "" {
		return nil, errors.New("tileirdisasm: executable not found in PATH or standard CUDA directories")
	}

	tmpDir, err := os.MkdirTemp("", "tileirdisasm-*")
	if err != nil {
		return nil, fmt.Errorf("tileirdisasm: creating temp dir: %w", err)
	}
	defer os.RemoveAll(tmpDir)

	inputPath := filepath.Join(tmpDir, "input.tilebc")
	if err := os.WriteFile(inputPath, bytecode, 0600); err != nil {
		return nil, fmt.Errorf("tileirdisasm: writing input: %w", err)
	}

	args := []string{}
	if cfg.debugInfo {
		args = append(args, "-g")
	}
	args = append(args, inputPath)

	cmd := exec.CommandContext(ctx, disasmBin, args...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("tileirdisasm execution failed: %w\n%s", err, stderr.String())
	}

	return Parse(bytes.NewReader(stdout.Bytes()))
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
