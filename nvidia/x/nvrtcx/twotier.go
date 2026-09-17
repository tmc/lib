package nvrtcx

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudakernel"
)

// IsOfflineFlag reports whether a flag is an offline nvcc or ptxas flag
// that should not be passed to nvrtcCompileProgram.
func IsOfflineFlag(flag string) bool {
	f := strings.TrimSpace(flag)
	if f == "" {
		return false
	}
	// Check -maxrregcount or --maxrregcount
	if f == "-maxrregcount" || strings.HasPrefix(f, "-maxrregcount=") ||
		f == "--maxrregcount" || strings.HasPrefix(f, "--maxrregcount=") {
		return true
	}
	// Check -Xptxas
	if f == "-Xptxas" || strings.HasPrefix(f, "-Xptxas=") || strings.HasPrefix(f, "-Xptxas,") {
		return true
	}
	// Check -O0, -O1, -O2, -O3, -O4
	if f == "-O0" || f == "-O1" || f == "-O2" || f == "-O3" || f == "-O4" || f == "-O" {
		return true
	}
	// Check -Xcudafe, -Xnvlink
	if strings.HasPrefix(f, "-Xcudafe") || strings.HasPrefix(f, "-Xnvlink") {
		return true
	}
	return false
}

// ValidateNVRTCFlags returns an error if any offline nvcc/ptxas flags are present.
func ValidateNVRTCFlags(flags []string) error {
	var bad []string
	for _, f := range flags {
		if IsOfflineFlag(f) {
			bad = append(bad, f)
		}
	}
	if len(bad) > 0 {
		return fmt.Errorf("nvrtcx: offline nvcc/ptxas flags not permitted in NVRTC compile phase: %s", strings.Join(bad, ", "))
	}
	return nil
}

// SeparateFlags separates compiler flags into source-level NVRTC options
// and offline flags, translating relevant offline options into JITOptions.
func SeparateFlags(flags []string) (sourceFlags []string, offlineFlags []string, jit cudakernel.JITOptions, err error) {
	for _, f := range flags {
		if IsOfflineFlag(f) {
			offlineFlags = append(offlineFlags, f)
			// Parse into JIT options if applicable
			if strings.HasPrefix(f, "-maxrregcount=") || strings.HasPrefix(f, "--maxrregcount=") {
				valStr := f[strings.IndexByte(f, '=')+1:]
				n, convErr := strconv.Atoi(valStr)
				if convErr == nil && n > 0 {
					jit.MaxRegisters = uint32(n)
				}
			} else if strings.HasPrefix(f, "-O") && len(f) == 3 && f[2] >= '0' && f[2] <= '4' {
				jit.OptLevel = uint32(f[2] - '0')
			}
		} else {
			sourceFlags = append(sourceFlags, f)
		}
	}
	return sourceFlags, offlineFlags, jit, nil
}

// StripOfflineFlags removes offline nvcc/ptxas flags from flags, returning
// only source-level flags and extracted JITOptions.
func StripOfflineFlags(flags []string) ([]string, cudakernel.JITOptions) {
	src, _, jit, _ := SeparateFlags(flags)
	return src, jit
}

// resolveExternalDependencies scans source for #include directives and attempts
// to resolve them against search paths, hashing their contents.
// If any external header cannot be resolved, allTracked is false.
func resolveExternalDependencies(src string, flags []string, inMemoryHeaders map[string]string) (deps []cudakernel.Dependency, allTracked bool) {
	allTracked = true
	var searchPaths []string
	for _, f := range flags {
		if strings.HasPrefix(f, "--include-path=") {
			searchPaths = append(searchPaths, strings.TrimPrefix(f, "--include-path="))
		} else if strings.HasPrefix(f, "-I") {
			searchPaths = append(searchPaths, strings.TrimPrefix(f, "-I"))
		}
	}

	scanner := bufio.NewScanner(strings.NewReader(src))
	seen := make(map[string]bool)

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if !strings.HasPrefix(line, "#include") {
			continue
		}
		rest := strings.TrimSpace(strings.TrimPrefix(line, "#include"))
		if len(rest) < 3 {
			continue
		}

		var headerName string
		if rest[0] == '"' {
			end := strings.IndexByte(rest[1:], '"')
			if end >= 0 {
				headerName = rest[1 : end+1]
			}
		} else if rest[0] == '<' {
			end := strings.IndexByte(rest[1:], '>')
			if end >= 0 {
				headerName = rest[1 : end+1]
			}
		}

		if headerName == "" || seen[headerName] {
			continue
		}
		seen[headerName] = true

		// In-memory headers are already tracked in cfg.headers
		if _, ok := inMemoryHeaders[headerName]; ok {
			continue
		}

		// Search include paths on disk
		var foundPath string
		for _, sp := range searchPaths {
			candidate := filepath.Join(sp, headerName)
			if fi, err := os.Stat(candidate); err == nil && !fi.IsDir() {
				foundPath = candidate
				break
			}
		}

		if foundPath != "" {
			data, err := os.ReadFile(foundPath)
			if err == nil {
				h := sha256.Sum256(data)
				deps = append(deps, cudakernel.Dependency{
					Path:    foundPath,
					Digest:  hex.EncodeToString(h[:]),
					Tracked: true,
				})
				continue
			}
		}

		// Unresolved external header: record explicit inability to validate
		deps = append(deps, cudakernel.Dependency{
			Path:     headerName,
			Tracked:  false,
			ErrorMsg: "unable to validate external header",
		})
		allTracked = false
	}

	return deps, allTracked
}

func computeDependencyKey(deps []cudakernel.Dependency) string {
	var sb strings.Builder
	for _, d := range deps {
		sb.WriteString(d.Path)
		sb.WriteByte(':')
		sb.WriteString(d.Digest)
		sb.WriteByte(';')
	}
	h := sha256.Sum256([]byte(sb.String()))
	return hex.EncodeToString(h[:])
}
