package cudainject

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/tmc/lib/nvidia/internal/cudainject"
)

// BuildOptions selects explicit compiler and toolkit inputs. Compiler must be
// an executable path, not a shell command. Directory must not already exist;
// it retains the exact materialized sources, build log and resulting agent.
// Target is x86_64-linux or sbsa-linux; empty selects the host architecture.
// CUPTIInclude and CUPTILibrary select a separate CUPTI installation. Both must
// be absolute and supplied together. Empty uses the toolkit target directories.
// Library selection does not configure the runtime loader search path.
type BuildOptions struct {
	Compiler     string
	CUDAHome     string
	Directory    string
	Target       string
	CUPTIInclude string
	CUPTILibrary string
}

// Agent records the output and exact source inputs used by BuildAgent.
// A successful build is not proof that the target's driver supports the agent.
type Agent struct {
	Path      string
	Arguments []string
	Sources   map[string]string
}

// BuildAgent compiles the embedded, existing C agent without a source checkout.
// The context must have a deadline. Cancellation stops the compiler process;
// callers must use a trusted compiler that does not leave detached children.
// Failed builds retain their directory and log for inspection.
func BuildAgent(ctx context.Context, options BuildOptions) (*Agent, error) {
	if ctx == nil {
		return nil, errors.New("cudainject: nil build context")
	}
	if _, ok := ctx.Deadline(); !ok {
		return nil, errors.New("cudainject: build context requires deadline")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	for _, path := range []string{options.Compiler, options.CUDAHome, options.Directory} {
		if !filepath.IsAbs(path) {
			return nil, errors.New("cudainject: compiler, toolkit and directory must be absolute")
		}
	}
	if (options.CUPTIInclude == "") != (options.CUPTILibrary == "") {
		return nil, errors.New("cudainject: CUPTI include and library directories must be supplied together")
	}
	if options.CUPTIInclude != "" && (!filepath.IsAbs(options.CUPTIInclude) || !filepath.IsAbs(options.CUPTILibrary)) {
		return nil, errors.New("cudainject: CUPTI directories must be absolute")
	}
	target := options.Target
	if target == "" {
		switch runtime.GOARCH {
		case "amd64":
			target = "x86_64-linux"
		case "arm64":
			target = "sbsa-linux"
		default:
			return nil, errors.New("cudainject: unsupported agent architecture")
		}
	}
	if target != "x86_64-linux" && target != "sbsa-linux" {
		return nil, errors.New("cudainject: unsupported agent target")
	}
	if err := os.Mkdir(options.Directory, 0700); err != nil {
		return nil, fmt.Errorf("cudainject: create build directory: %w", err)
	}
	inputs, err := cudainject.Sources.ReadDir("native")
	if err != nil {
		return nil, err
	}
	agent := &Agent{Path: filepath.Join(options.Directory, "cupti_agent.so"), Sources: make(map[string]string)}
	for _, input := range inputs {
		data, err := cudainject.Sources.ReadFile("native/" + input.Name())
		if err != nil {
			return nil, err
		}
		hash := sha256.Sum256(data)
		agent.Sources[input.Name()] = hex.EncodeToString(hash[:])
		if err := os.WriteFile(filepath.Join(options.Directory, input.Name()), data, 0600); err != nil {
			return nil, err
		}
	}
	// Stable source identity replaces unavailable checkout revision metadata.
	names := make([]string, 0, len(agent.Sources))
	for name := range agent.Sources {
		names = append(names, name)
	}
	sort.Strings(names)
	var source strings.Builder
	for _, name := range names {
		fmt.Fprintf(&source, "%s %s\n", name, agent.Sources[name])
	}
	hash := sha256.Sum256([]byte(source.String()))
	revision := "sources-" + hex.EncodeToString(hash[:])
	agent.Arguments = []string{"-std=c11", "-O2", "-fPIC", "-shared", `-DCUDAPROF_AGENT_REVISION="` + revision + `"`,
		"-I" + filepath.Join(options.CUDAHome, "targets", target, "include"), "-I" + filepath.Join(options.CUDAHome, "include"), "-I" + options.Directory,
		filepath.Join(options.Directory, "cupti_agent.c"), "-o", agent.Path,
		"-L" + filepath.Join(options.CUDAHome, "targets", target, "lib"), "-L" + filepath.Join(options.CUDAHome, "lib64"), "-lcupti", "-lpthread"}
	if options.CUPTIInclude != "" {
		// Explicit CUPTI inputs precede toolkit search paths; they are never guessed.
		agent.Arguments = append([]string{"-I" + options.CUPTIInclude, "-L" + options.CUPTILibrary}, agent.Arguments...)
	}
	log, err := os.OpenFile(filepath.Join(options.Directory, "build.log"), os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		return nil, err
	}
	command := exec.CommandContext(ctx, options.Compiler, agent.Arguments...)
	command.Dir = options.Directory
	command.WaitDelay = time.Second
	output := &buildOutput{file: log, remaining: 16 << 20}
	command.Stdout, command.Stderr = output, output
	runErr := command.Run()
	closeErr := log.Close()
	if runErr != nil {
		return nil, fmt.Errorf("cudainject: compile agent: %w", runErr)
	}
	if closeErr != nil {
		return nil, closeErr
	}
	info, err := os.Stat(agent.Path)
	if err != nil {
		return nil, fmt.Errorf("cudainject: agent output: %w", err)
	}
	if !info.Mode().IsRegular() || info.Size() == 0 {
		return nil, errors.New("cudainject: empty or nonregular agent output")
	}
	return agent, nil
}

type buildOutput struct {
	mu        sync.Mutex
	file      *os.File
	remaining int
}

func (w *buildOutput) Write(p []byte) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if len(p) > w.remaining {
		return 0, errors.New("cudainject: compiler log exceeds 16 MiB")
	}
	n, err := w.file.Write(p)
	w.remaining -= n
	return n, err
}
