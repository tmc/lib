package nvrtcx

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
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
