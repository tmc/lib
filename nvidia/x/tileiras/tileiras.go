package tileiras

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// Magic header identifying Tile IR bytecode.
var bytecodeMagic = []byte{0x7F, 'T', 'i', 'l', 'e', 'I', 'R', 0x00}

// ErrNotFound is returned when the tileiras executable cannot be located in PATH or standard CUDA directories.
var ErrNotFound = errors.New("tileiras: executable not found in PATH or standard CUDA directories")

// IsBytecode reports whether data starts with the Tile IR bytecode magic number.
func IsBytecode(data []byte) bool {
	return bytes.HasPrefix(data, bytecodeMagic)
}

// Option configures execution of the tileiras assembler.
type Option func(*config)

type config struct {
	gpuName         string
	optLevel        int
	debug           bool
	lineInfo        bool
	remarks         string
	remarksPassed   string
	remarksFailed   string
	remarksMissed   string
	remarksAnalysis string
	tileFatbin      bool
	tileirasPath    string
	fatbinaryPath   string
	extraFlags      []string
}

// WithGPUName specifies the target GPU architecture name, e.g. "sm_90", "sm_100", "sm_121".
func WithGPUName(name string) Option {
	return func(c *config) {
		c.gpuName = name
	}
}

// WithArch specifies the target architecture by major and minor numbers (e.g. 9, 0 -> "sm_90").
func WithArch(major, minor int) Option {
	return func(c *config) {
		c.gpuName = fmt.Sprintf("sm_%d%d", major, minor)
	}
}

// WithOptLevel specifies the optimization level (0 to 3). Default is 3.
func WithOptLevel(level int) Option {
	return func(c *config) {
		c.optLevel = level
	}
}

// WithDebug enables generation of device debug information (-g).
func WithDebug() Option {
	return func(c *config) {
		c.debug = true
	}
}

// WithLineInfo enables line number generation (--lineinfo).
func WithLineInfo() Option {
	return func(c *config) {
		c.lineInfo = true
	}
}

// WithRemarks enables compiler remarks for a category ("all", "memory", "tensorcore", "none").
func WithRemarks(category string) Option {
	return func(c *config) {
		c.remarks = category
	}
}

// WithRemarksPassed enables passed remarks for a category.
func WithRemarksPassed(category string) Option {
	return func(c *config) {
		c.remarksPassed = category
	}
}

// WithRemarksFailed enables failed remarks for a category.
func WithRemarksFailed(category string) Option {
	return func(c *config) {
		c.remarksFailed = category
	}
}

// WithRemarksMissed enables missed remarks for a category.
func WithRemarksMissed(category string) Option {
	return func(c *config) {
		c.remarksMissed = category
	}
}

// WithRemarksAnalysis enables analysis remarks for a category.
func WithRemarksAnalysis(category string) Option {
	return func(c *config) {
		c.remarksAnalysis = category
	}
}

// WithTileFatbin requests output packaged as a CUDA fatbinary (.fatbin) containing both
// the assembled ELF CUBIN and the Tile IR bitcode.
func WithTileFatbin() Option {
	return func(c *config) {
		c.tileFatbin = true
	}
}

// WithPath specifies an explicit path to the tileiras executable.
func WithPath(path string) Option {
	return func(c *config) {
		c.tileirasPath = path
	}
}

// WithFatbinaryPath specifies an explicit path to the fatbinary packaging executable.
func WithFatbinaryPath(path string) Option {
	return func(c *config) {
		c.fatbinaryPath = path
	}
}

// WithExtraFlags supplies additional command-line flags to tileiras.
func WithExtraFlags(flags ...string) Option {
	return func(c *config) {
		c.extraFlags = append(c.extraFlags, flags...)
	}
}

// Result encapsulates assembled binary outputs and parsed compiler remarks.
type Result struct {
	CUBIN   []byte   // Assembled device CUBIN ELF image
	Fatbin  []byte   // Packaged fatbinary image, if WithTileFatbin was requested
	Remarks []Remark // Compiler optimization remarks
	Log     string   // Raw compiler output (stdout/stderr)
}

// TMAFailures returns all remarks indicating TMA instruction-selection bottlenecks.
func (r *Result) TMAFailures() []Remark {
	if r == nil {
		return nil
	}
	var res []Remark
	for _, rmk := range r.Remarks {
		if rmk.IsTMAFailure() {
			res = append(res, rmk)
		}
	}
	return res
}

// TensorCoreFailures returns all remarks indicating an MMA operation failed to use Tensor Cores.
func (r *Result) TensorCoreFailures() []Remark {
	if r == nil {
		return nil
	}
	var res []Remark
	for _, rmk := range r.Remarks {
		if rmk.IsTensorCoreFailure() {
			res = append(res, rmk)
		}
	}
	return res
}

// Assemble compiles Tile IR input into executable device code.
func Assemble(ctx context.Context, input []byte, opts ...Option) (*Result, error) {
	if len(input) == 0 {
		return nil, errors.New("tileiras: empty input")
	}

	cfg := config{
		gpuName:  "sm_90",
		optLevel: 3,
	}
	for _, opt := range opts {
		opt(&cfg)
	}

	tileirasBin := findTileiras(cfg.tileirasPath)
	if tileirasBin == "" {
		return nil, ErrNotFound
	}

	// Prepare temp working directory
	tmpDir, err := os.MkdirTemp("", "tileiras-*")
	if err != nil {
		return nil, fmt.Errorf("tileiras: creating temp dir: %w", err)
	}
	defer os.RemoveAll(tmpDir)

	inputExt := ".tileir"
	if IsBytecode(input) {
		inputExt = ".tilebc"
	}
	inputPath := filepath.Join(tmpDir, "input"+inputExt)
	if err := os.WriteFile(inputPath, input, 0600); err != nil {
		return nil, fmt.Errorf("tileiras: writing input: %w", err)
	}

	outputPath := filepath.Join(tmpDir, "output.cubin")
	remarksPath := filepath.Join(tmpDir, "remarks.yaml")

	args := []string{
		fmt.Sprintf("--gpu-name=%s", cfg.gpuName),
		fmt.Sprintf("--opt-level=%d", cfg.optLevel),
		"-o", outputPath,
	}

	if cfg.debug {
		args = append(args, "-g")
	}
	if cfg.lineInfo {
		args = append(args, "--lineinfo")
	}

	wantRemarks := cfg.remarks != "" || cfg.remarksPassed != "" || cfg.remarksFailed != "" ||
		cfg.remarksMissed != "" || cfg.remarksAnalysis != ""

	if wantRemarks {
		if cfg.remarks != "" {
			args = append(args, fmt.Sprintf("--remarks=%s", cfg.remarks))
		}
		if cfg.remarksPassed != "" {
			args = append(args, fmt.Sprintf("--remarks-passed=%s", cfg.remarksPassed))
		}
		if cfg.remarksFailed != "" {
			args = append(args, fmt.Sprintf("--remarks-failed=%s", cfg.remarksFailed))
		}
		if cfg.remarksMissed != "" {
			args = append(args, fmt.Sprintf("--remarks-missed=%s", cfg.remarksMissed))
		}
		if cfg.remarksAnalysis != "" {
			args = append(args, fmt.Sprintf("--remarks-analysis=%s", cfg.remarksAnalysis))
		}
		args = append(args, "--remark-format=yaml", fmt.Sprintf("--remarks-output-file=%s", remarksPath))
	}

	args = append(args, cfg.extraFlags...)
	args = append(args, inputPath)

	cmd := exec.CommandContext(ctx, tileirasBin, args...)
	var outputBuf bytes.Buffer
	cmd.Stdout = &outputBuf
	cmd.Stderr = &outputBuf

	runErr := cmd.Run()
	logOutput := outputBuf.String()

	if runErr != nil {
		return nil, fmt.Errorf("tileiras assembly failed: %w\n%s", runErr, logOutput)
	}

	cubinBytes, err := os.ReadFile(outputPath)
	if err != nil {
		return nil, fmt.Errorf("tileiras: reading cubin output: %w", err)
	}

	res := &Result{
		CUBIN: cubinBytes,
		Log:   logOutput,
	}

	// Parse remarks YAML if produced
	if wantRemarks {
		if remarkData, err := os.ReadFile(remarksPath); err == nil && len(remarkData) > 0 {
			if remarks, err := ParseRemarksYAML(remarkData); err == nil {
				res.Remarks = remarks
			}
		}
	}

	// If fatbinary was requested, package via fatbinary tool
	if cfg.tileFatbin {
		fatbinBytes, err := buildFatbinary(ctx, cfg, tmpDir, cubinBytes, input)
		if err != nil {
			return nil, fmt.Errorf("tileiras: building fatbinary: %w", err)
		}
		res.Fatbin = fatbinBytes
	}

	return res, nil
}

func buildFatbinary(ctx context.Context, cfg config, tmpDir string, cubin, tilebc []byte) ([]byte, error) {
	fatbinTool := findFatbinary(cfg.fatbinaryPath)
	if fatbinTool == "" {
		return nil, errors.New("fatbinary tool not found")
	}

	cubinPath := filepath.Join(tmpDir, "image.cubin")
	tilebcPath := filepath.Join(tmpDir, "image.tilebc")
	fatbinPath := filepath.Join(tmpDir, "output.fatbin")

	if err := os.WriteFile(cubinPath, cubin, 0600); err != nil {
		return nil, err
	}
	if err := os.WriteFile(tilebcPath, tilebc, 0600); err != nil {
		return nil, err
	}

	// Extract numerical arch: e.g. "sm_90" -> "90"
	archNum := strings.TrimPrefix(cfg.gpuName, "sm_")

	args := []string{
		fmt.Sprintf("--create=%s", fatbinPath),
		"-64",
		fmt.Sprintf("--image3=kind=elf,sm=%s,file=%s", archNum, cubinPath),
		fmt.Sprintf("--image3=kind=tileir,sm=%s,file=%s", archNum, tilebcPath),
	}

	cmd := exec.CommandContext(ctx, fatbinTool, args...)
	var out bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &out

	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("%w: %s", err, out.String())
	}

	return os.ReadFile(fatbinPath)
}

func findTileiras(custom string) string {
	if custom != "" {
		return custom
	}
	if p, err := exec.LookPath("tileiras"); err == nil {
		return p
	}
	candidates := []string{
		"/usr/local/cuda-13.4/bin/tileiras",
		"/usr/local/cuda-13.3/bin/tileiras",
		"/usr/local/cuda-13.2/bin/tileiras",
		"/usr/local/cuda-13.1/bin/tileiras",
		"/usr/local/cuda/bin/tileiras",
	}
	for _, c := range candidates {
		if fi, err := os.Stat(c); err == nil && !fi.IsDir() {
			return c
		}
	}
	return ""
}

func findFatbinary(custom string) string {
	if custom != "" {
		return custom
	}
	if p, err := exec.LookPath("fatbinary"); err == nil {
		return p
	}
	candidates := []string{
		"/usr/local/cuda-13.4/bin/fatbinary",
		"/usr/local/cuda/bin/fatbinary",
	}
	for _, c := range candidates {
		if fi, err := os.Stat(c); err == nil && !fi.IsDir() {
			return c
		}
	}
	return ""
}
