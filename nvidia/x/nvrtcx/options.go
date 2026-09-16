package nvrtcx

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/tmc/lib/nvidia/nvrtc"
)

// ToolkitIncludeDir reports a CUDA toolkit include directory on the host,
// or the empty string when none is found. It inspects CUDA_HOME, CUDA_PATH,
// standard installation paths like /usr/local/cuda and /opt/cuda, and
// Windows installation directories, verifying the presence of CUDA headers.
func ToolkitIncludeDir() string {
	var roots []string
	for _, env := range []string{"CUDA_HOME", "CUDA_PATH"} {
		if root := os.Getenv(env); root != "" {
			roots = append(roots, root)
		}
	}
	roots = append(roots, "/usr/local/cuda", "/opt/cuda")
	if pf := os.Getenv("ProgramFiles"); pf != "" {
		base := filepath.Join(pf, "NVIDIA GPU Computing Toolkit", "CUDA")
		if versions, err := filepath.Glob(filepath.Join(base, "v*")); err == nil {
			roots = append(roots, versions...)
		}
	}
	for _, root := range roots {
		candidates, _ := filepath.Glob(filepath.Join(root, "targets", "*", "include"))
		candidates = append(candidates, filepath.Join(root, "include"))
		for _, dir := range candidates {
			for _, header := range []string{"cuda_runtime.h", "cuda.h"} {
				if _, err := os.Stat(filepath.Join(dir, header)); err == nil {
					return dir
				}
			}
		}
	}
	return ""
}

// WithIncludePaths adds directories to the compiler's include search path (--include-path).
func WithIncludePaths(paths ...string) Option {
	dirs := append([]string(nil), paths...)
	return func(o *options) {
		for _, dir := range dirs {
			if dir == "" {
				o.err = fmt.Errorf("nvrtcx: empty include path")
				return
			}
			if strings.ContainsRune(dir, 0) {
				o.err = fmt.Errorf("nvrtcx: include path contains NUL")
				return
			}
			o.flags = append(o.flags, "--include-path="+dir)
		}
	}
}

// WithToolkitIncludes explicitly adds the CUDA toolkit include directory (--include-path).
// Compilation returns an error if no toolkit directory can be found on the host.
func WithToolkitIncludes() Option {
	return func(o *options) {
		o.toolkitIncludes = true
		o.noToolkitIncludes = false
	}
}

// WithNoToolkitIncludes disables automatic discovery and inclusion of the CUDA toolkit directory.
func WithNoToolkitIncludes() Option {
	return func(o *options) {
		o.noToolkitIncludes = true
		o.toolkitIncludes = false
	}
}

// WithDefines adds preprocessor macro definitions (-D).
// Each macro may be of the form "NAME" or "NAME=VALUE".
func WithDefines(macros ...string) Option {
	defs := append([]string(nil), macros...)
	return func(o *options) {
		for _, m := range defs {
			if m == "" {
				o.err = fmt.Errorf("nvrtcx: empty macro definition")
				return
			}
			if strings.ContainsRune(m, 0) {
				o.err = fmt.Errorf("nvrtcx: macro definition contains NUL")
				return
			}
			o.flags = append(o.flags, "-D"+m)
		}
	}
}

// WithUndefines undefines preprocessor macros (-U).
func WithUndefines(macros ...string) Option {
	defs := append([]string(nil), macros...)
	return func(o *options) {
		for _, m := range defs {
			if m == "" {
				o.err = fmt.Errorf("nvrtcx: empty macro name")
				return
			}
			if strings.ContainsRune(m, 0) {
				o.err = fmt.Errorf("nvrtcx: macro name contains NUL")
				return
			}
			o.flags = append(o.flags, "-U"+m)
		}
	}
}

// WithStandard configures the C++ language standard (-std).
// Supported values are "c++11", "c++14", "c++17", "c++20", "c++23",
// as well as shorthand versions "11", "14", "17", "20", "23".
func WithStandard(std string) Option {
	return func(o *options) {
		s := strings.ToLower(strings.TrimSpace(std))
		switch s {
		case "11", "c++11":
			s = "c++11"
		case "14", "c++14":
			s = "c++14"
		case "17", "c++17":
			s = "c++17"
		case "20", "c++20":
			s = "c++20"
		case "23", "c++23":
			s = "c++23"
		default:
			o.err = fmt.Errorf("nvrtcx: unsupported C++ standard %q", std)
			return
		}
		o.flags = append(o.flags, "-std="+s)
	}
}

// WithLineInfo generates source-level line information for debugging and
// profiling tools like cudaprof and NSight (--generate-line-info).
// Line info is enabled by default unless WithNoLineInfo or WithDeviceDebug is used.
func WithLineInfo() Option {
	return func(o *options) {
		o.lineInfo = true
		o.noLineInfo = false
	}
}

// WithNoLineInfo disables the default generation of source line information.
func WithNoLineInfo() Option {
	return func(o *options) {
		o.noLineInfo = true
		o.lineInfo = false
	}
}

// WithDeviceDebug enables device code debug generation (--device-debug).
// This turns off compiler optimizations and disables default line info injection.
func WithDeviceDebug() Option {
	return func(o *options) {
		o.deviceDebug = true
		for _, f := range o.flags {
			if f == "--device-debug" || f == "-G" {
				return
			}
		}
		o.flags = append(o.flags, "--device-debug")
	}
}

// WithFastMath enables fast math optimizations (--use_fast_math).
// This turns on -ftz=true, -prec-div=false, -prec-sqrt=false, and -fmad=true.
func WithFastMath() Option {
	return func(o *options) {
		for _, f := range o.flags {
			if f == "--use_fast_math" || f == "-use_fast_math" {
				return
			}
		}
		o.flags = append(o.flags, "--use_fast_math")
	}
}

// WithDefaultDevice treats functions without explicit __host__ or __device__
// execution space specifiers as __device__ functions (--device-as-default-execution-space).
func WithDefaultDevice() Option {
	return func(o *options) {
		for _, f := range o.flags {
			if f == "--device-as-default-execution-space" || f == "-default-device" {
				return
			}
		}
		o.flags = append(o.flags, "--device-as-default-execution-space")
	}
}

// WithDefaultExecutionSpace treats functions without explicit __host__ or __device__
// execution space specifiers as __device__ functions (--device-as-default-execution-space).
// It is an alias for WithDefaultDevice.
func WithDefaultExecutionSpace() Option {
	return WithDefaultDevice()
}

// WithRelocatableDeviceCode enables or disables relocatable device code (--relocatable-device-code).
func WithRelocatableDeviceCode(enable bool) Option {
	return func(o *options) {
		o.flags = append(o.flags, fmt.Sprintf("--relocatable-device-code=%t", enable))
	}
}

// WithOptimizationLevel sets the PTX optimization level (-O0, -O1, -O2, -O3).
func WithOptimizationLevel(level int) Option {
	return func(o *options) {
		if level < 0 || level > 3 {
			o.err = fmt.Errorf("nvrtcx: invalid optimization level %d (must be 0-3)", level)
			return
		}
		o.flags = append(o.flags, fmt.Sprintf("-O%d", level))
	}
}

// WithExtraDeviceVectorization enables aggressive device code vectorization (--extra-device-vectorization).
func WithExtraDeviceVectorization() Option {
	return func(o *options) {
		for _, f := range o.flags {
			if f == "--extra-device-vectorization" {
				return
			}
		}
		o.flags = append(o.flags, "--extra-device-vectorization")
	}
}

// WithDeviceVectorization enables aggressive device code vectorization (--extra-device-vectorization).
// It is an alias for WithExtraDeviceVectorization.
func WithDeviceVectorization() Option {
	return WithExtraDeviceVectorization()
}

// WithMaxRegisters sets the maximum register count per thread (--maxrregcount=N).
func WithMaxRegisters(n int) Option {
	return func(o *options) {
		if n <= 0 {
			o.err = fmt.Errorf("nvrtcx: invalid max registers %d", n)
			return
		}
		o.flags = append(o.flags, fmt.Sprintf("--maxrregcount=%d", n))
	}
}

// WithWarningsAsErrors treats all compiler warnings as errors (--warning-as-error).
func WithWarningsAsErrors() Option {
	return func(o *options) {
		for _, f := range o.flags {
			if f == "--warning-as-error" || f == "-Werror" {
				return
			}
		}
		o.flags = append(o.flags, "--warning-as-error")
	}
}

// WithDisableWarnings suppresses all compiler warnings (-w).
func WithDisableWarnings() Option {
	return func(o *options) {
		for _, f := range o.flags {
			if f == "-w" || f == "--disable-warnings" {
				return
			}
		}
		o.flags = append(o.flags, "-w")
	}
}

// WithContext associates a context.Context with compilation.
// If the context is cancelled during compilation and the NVRTC library
// supports flow callbacks, compilation is aborted early.
func WithContext(ctx context.Context) Option {
	return func(o *options) {
		if ctx == nil {
			o.err = fmt.Errorf("nvrtcx: nil context")
			return
		}
		o.ctx = ctx
	}
}

// DefaultBundledHeadersDir returns the default directory used for extracting bundled headers.
func DefaultBundledHeadersDir() string {
	base, err := os.UserCacheDir()
	if err != nil {
		base = os.TempDir()
	}
	return filepath.Join(base, "tmc-lib", "nvrtc-bundled-headers")
}

// InstallBundledHeaders extracts CUDA headers bundled with NVRTC to targetDir.
// If targetDir is empty, DefaultBundledHeadersDir() is used.
// If forceOverwrite is false, existing headers are preserved (nvrtc.InstallHeadersSkipIfExists).
func InstallBundledHeaders(targetDir string, forceOverwrite bool) (string, error) {
	if !nvrtc.HasInstallBundledHeaders() {
		return "", fmt.Errorf("nvrtcx: bundled headers not supported by loaded NVRTC library")
	}
	if targetDir == "" {
		targetDir = DefaultBundledHeadersDir()
	}
	if err := os.MkdirAll(targetDir, 0o755); err != nil {
		return "", fmt.Errorf("nvrtcx: create bundled headers directory: %w", err)
	}
	var flags uint32 = nvrtc.InstallHeadersSkipIfExists
	if forceOverwrite {
		flags = nvrtc.InstallHeadersForceOverwrite
	}
	if err := nvrtc.InstallBundledHeaders(targetDir, flags); err != nil {
		return "", fmt.Errorf("nvrtcx: install bundled headers: %w", err)
	}
	return targetDir, nil
}

// WithBundledHeaders enables extracting and using the CUDA headers bundled with NVRTC.
// If cacheDir is specified, it sets the directory where headers are extracted;
// otherwise DefaultBundledHeadersDir() is used.
func WithBundledHeaders(cacheDir ...string) Option {
	return func(o *options) {
		o.bundledHeaders = true
		if len(cacheDir) > 0 {
			if cacheDir[0] == "" {
				o.err = fmt.Errorf("nvrtcx: empty bundled headers directory")
				return
			}
			if strings.ContainsRune(cacheDir[0], 0) {
				o.err = fmt.Errorf("nvrtcx: bundled headers directory contains NUL")
				return
			}
			o.bundledHeadersDir = cacheDir[0]
		}
	}
}

// WithPCH enables precompiled header processing (-pch).
func WithPCH() Option {
	return func(o *options) {
		for _, f := range o.flags {
			if f == "-pch" || f == "--pch" {
				return
			}
		}
		o.flags = append(o.flags, "-pch")
	}
}

// WithCreatePCH instructs the compiler to create a precompiled header file (--create-pch=<file>).
func WithCreatePCH(file string) Option {
	return func(o *options) {
		if file == "" {
			o.err = fmt.Errorf("nvrtcx: empty PCH file")
			return
		}
		if strings.ContainsRune(file, 0) {
			o.err = fmt.Errorf("nvrtcx: PCH file contains NUL")
			return
		}
		o.flags = append(o.flags, "--create-pch="+file)
	}
}

// WithUsePCH instructs the compiler to use a precompiled header file (--use-pch=<file>).
func WithUsePCH(file string) Option {
	return func(o *options) {
		if file == "" {
			o.err = fmt.Errorf("nvrtcx: empty PCH file")
			return
		}
		if strings.ContainsRune(file, 0) {
			o.err = fmt.Errorf("nvrtcx: PCH file contains NUL")
			return
		}
		o.flags = append(o.flags, "--use-pch="+file)
	}
}

// WithFastCompile configures the fast compilation mode (--Ofast-compile=0|min|mid|max).
func WithFastCompile(level string) Option {
	return func(o *options) {
		s := strings.ToLower(strings.TrimSpace(level))
		switch s {
		case "0", "min", "mid", "max":
		default:
			o.err = fmt.Errorf("nvrtcx: invalid fast compile level %q (must be 0, min, mid, or max)", level)
			return
		}
		o.flags = append(o.flags, "--Ofast-compile="+s)
	}
}

// WithRandomSeed specifies a seed for compiler determinism (--frandom-seed=<seed>).
func WithRandomSeed(seed int) Option {
	return func(o *options) {
		o.flags = append(o.flags, fmt.Sprintf("--frandom-seed=%d", seed))
	}
}

// WithExtensibleWholeProgram enables extensible whole program compilation (--extensible-whole-program).
func WithExtensibleWholeProgram() Option {
	return func(o *options) {
		for _, f := range o.flags {
			if f == "--extensible-whole-program" || f == "-ewp" {
				return
			}
		}
		o.flags = append(o.flags, "--extensible-whole-program")
	}
}

// WithDeviceStackProtector enables or disables the device stack protector (--device-stack-protector).
func WithDeviceStackProtector(enable bool) Option {
	return func(o *options) {
		o.flags = append(o.flags, fmt.Sprintf("--device-stack-protector=%t", enable))
	}
}

// WithTimeTrace enables compiler phase time tracing to an output file (--time=<file>).
func WithTimeTrace(file string) Option {
	return func(o *options) {
		if file == "" {
			o.err = fmt.Errorf("nvrtcx: empty time trace file")
			return
		}
		if strings.ContainsRune(file, 0) {
			o.err = fmt.Errorf("nvrtcx: time trace file contains NUL")
			return
		}
		o.flags = append(o.flags, "--time="+file)
	}
}

func hasLineInfoFlag(flags []string) bool {
	for _, f := range flags {
		if f == "--generate-line-info" || f == "-lineinfo" || f == "-G" || f == "--device-debug" {
			return true
		}
	}
	return false
}

func hasIncludeFlag(flags []string) bool {
	for _, f := range flags {
		if strings.HasPrefix(f, "-I") || strings.HasPrefix(f, "--include-path") {
			return true
		}
	}
	return false
}

func hasFlag(flags []string, target string) bool {
	for _, f := range flags {
		if f == target {
			return true
		}
	}
	return false
}

// CompileFlags resolves the given options into a slice of NVRTC command-line flags.
// It applies line info and toolkit include defaults, and validates all options.
func CompileFlags(opts ...Option) ([]string, error) {
	cfg := options{
		lineInfo: true,
	}
	for _, opt := range opts {
		if opt == nil {
			return nil, fmt.Errorf("nvrtcx: nil option")
		}
		opt(&cfg)
	}
	if cfg.err != nil {
		return nil, cfg.err
	}
	if cfg.lineInfo && !cfg.noLineInfo && !cfg.deviceDebug && !hasLineInfoFlag(cfg.flags) {
		cfg.flags = append(cfg.flags, "--generate-line-info")
	}
	if cfg.toolkitIncludes && !cfg.noToolkitIncludes {
		dir := ToolkitIncludeDir()
		if dir != "" {
			flag := "--include-path=" + dir
			if !hasFlag(cfg.flags, flag) {
				cfg.flags = append(cfg.flags, flag)
			}
		} else {
			return nil, fmt.Errorf("nvrtcx: CUDA toolkit include directory not found")
		}
	}
	if cfg.bundledHeaders && !cfg.noToolkitIncludes {
		dir := cfg.bundledHeadersDir
		if dir == "" {
			dir = DefaultBundledHeadersDir()
		}
		flag := "--include-path=" + dir
		if !hasFlag(cfg.flags, flag) {
			cfg.flags = append(cfg.flags, flag)
		}
	}
	return append([]string(nil), cfg.flags...), nil
}
