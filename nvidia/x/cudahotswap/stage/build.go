package stage

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// Source is C or CUDA host glue defining int stage(struct stage_ctx *).
// Flags are compiler arguments; the builder owns input and output file names.
// Code and Flags must not be changed during Build.
type Source struct {
	Code  string
	Flags []string
	CUDA  bool
}

// Builder compiles libraries in an explicit retained directory. Its zero value
// is not usable. Compiler names cc or nvcc, or their explicit executable path.
type Builder struct {
	dir      string
	compiler string
}

var nextGeneration atomic.Uint64

// NewBuilder resolves compiler and creates dir. It does not load a library or
// perform CUDA work. Use a task-owned directory, not a shared temporary root.
func NewBuilder(dir, compiler string) (*Builder, error) {
	if dir == "" || compiler == "" {
		return nil, fmt.Errorf("create stage builder: directory and compiler are required")
	}
	dir, err := filepath.Abs(dir)
	if err != nil {
		return nil, fmt.Errorf("stage directory: %w", err)
	}
	compiler, err = exec.LookPath(compiler)
	if err != nil {
		return nil, fmt.Errorf("stage compiler: %w", err)
	}
	compiler, err = filepath.Abs(compiler)
	if err != nil {
		return nil, fmt.Errorf("stage compiler path: %w", err)
	}
	if err := os.MkdirAll(dir, 0700); err != nil {
		return nil, fmt.Errorf("create stage directory: %w", err)
	}
	return &Builder{dir: dir, compiler: compiler}, nil
}

// Artifact is one compiler result. Its source, generation file and digest are
// private so Load cannot accept a result with substituted identity metadata.
// Retain its directory until the library and all pins have closed.
type Artifact struct {
	path       string
	dir        string
	digest     [32]byte
	source     string
	flags      []string
	compiler   string
	generation uint64
}

// Path returns the unique generation library path.
func (a *Artifact) Path() string {
	if a == nil {
		return ""
	}
	return a.path
}

// Generation identifies this build within the process. Path also has a random
// generation-directory suffix, preventing reuse after a process restart.
func (a *Artifact) Generation() uint64 {
	if a == nil {
		return 0
	}
	return a.generation
}

// SourceDigest identifies the supplied stage body, independently of compiler
// output. Build does not claim identity for external headers or loaded libraries.
func (a *Artifact) SourceDigest() string {
	if a == nil {
		return ""
	}
	return fmt.Sprintf("%x", sha256.Sum256([]byte(a.source)))
}

// Build compiles the fixed pointed-to ABI and its compiler layout witness in
// the same translation unit as Source. It leaves source and output for receipts.
func (b *Builder) Build(ctx context.Context, source Source) (*Artifact, error) {
	if ctx == nil || b == nil || b.compiler == "" || source.Code == "" {
		return nil, fmt.Errorf("build stage: context, builder and code are required")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	ctx, cancel := context.WithTimeout(ctx, 2*time.Minute)
	defer cancel()
	if len(source.Code) > 1<<20 || len(source.Flags) > 64 {
		return nil, fmt.Errorf("build stage: source or flags exceed bounded size")
	}
	if runtime.GOOS != "linux" && runtime.GOOS != "darwin" {
		return nil, fmt.Errorf("build stage: platform %s is unsupported", runtime.GOOS)
	}
	flagBytes := 0
	for _, flag := range source.Flags {
		flagBytes += len(flag)
		if flagBytes > 16<<10 {
			return nil, fmt.Errorf("build stage: compiler flags exceed bounded size")
		}
		if flag == "" || !strings.HasPrefix(flag, "-") || flag == "-o" || strings.HasPrefix(flag, "-o=") || flag == "-c" || flag == "-E" || flag == "-S" {
			return nil, fmt.Errorf("build stage: unsupported compiler flag %q", flag)
		}
	}
	dir, err := os.MkdirTemp(b.dir, "generation-")
	if err != nil {
		return nil, fmt.Errorf("create stage generation: %w", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "stage_abi.h"), []byte(header), 0600); err != nil {
		return nil, fmt.Errorf("write stage header: %w", err)
	}
	ext := ".c"
	if source.CUDA {
		ext = ".cu"
	}
	input := filepath.Join(dir, "stage"+ext)
	body := "#include \"stage_abi.h\"\n" + source.Code + "\n" + witness
	if err := os.WriteFile(input, []byte(body), 0600); err != nil {
		return nil, fmt.Errorf("write stage source: %w", err)
	}
	generation, err := allocateGeneration()
	if err != nil {
		return nil, err
	}
	output := filepath.Join(dir, fmt.Sprintf("stage-%020d-%s.so", generation, filepath.Base(dir)))
	var args []string
	if source.CUDA {
		args = append(args, "-shared", "-Xcompiler=-fPIC", "-std=c++17")
	} else if runtime.GOOS == "darwin" {
		args = append(args, "-dynamiclib", "-fPIC", "-Werror", "-std=c11")
	} else {
		args = append(args, "-shared", "-fPIC", "-Werror", "-std=c11")
	}
	args = append(args, input)
	args = append(args, source.Flags...)
	args = append(args, "-o", output)
	cmd := exec.CommandContext(ctx, b.compiler, args...)
	cmd.Dir = dir
	configureCompiler(cmd)
	log := &compilerOutput{}
	cmd.Stdout, cmd.Stderr = log, log
	err = cmd.Run()
	err = errors.Join(err, finishCompiler(cmd))
	if writeErr := os.WriteFile(filepath.Join(dir, "compiler.log"), log.data, 0600); writeErr != nil {
		return nil, fmt.Errorf("retain compiler log: %w", writeErr)
	}
	if log.overflow {
		return nil, fmt.Errorf("compile stage in %s: compiler output exceeds %d bytes", dir, maxCompilerOutput)
	}
	if err != nil {
		return nil, fmt.Errorf("compile stage in %s: %w: %s", dir, err, strings.TrimSpace(string(log.data)))
	}
	image, err := readCompilerImage(output)
	if err != nil {
		return nil, fmt.Errorf("read stage output: %w", err)
	}
	if len(image) == 0 {
		return nil, fmt.Errorf("stage compiler produced an empty library")
	}
	return &Artifact{path: output, dir: dir, digest: sha256.Sum256(image), source: source.Code, flags: append([]string(nil), source.Flags...), compiler: b.compiler, generation: generation}, nil
}

const maxCompilerOutput = 64 << 10

// compilerOutput drains both pipes while retaining at most a bounded prefix.
type compilerOutput struct {
	mu       sync.Mutex
	data     []byte
	overflow bool
}

func (b *compilerOutput) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	n := len(p)
	remaining := maxCompilerOutput - len(b.data)
	if len(p) > remaining {
		b.overflow = true
		p = p[:remaining]
	}
	b.data = append(b.data, p...)
	return n, nil
}

func allocateGeneration() (uint64, error) {
	for {
		old := nextGeneration.Load()
		if old == ^uint64(0) {
			return 0, fmt.Errorf("stage generation counter exhausted")
		}
		if nextGeneration.CompareAndSwap(old, old+1) {
			return old + 1, nil
		}
	}
}
