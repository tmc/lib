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

// resolveExternalDependencies walks the #include closure of src and hashes the
// contents of every header it can resolve, so that editing any of them changes
// the compilation cache identity. Headers reached through another header count:
// scanning only src leaves a transitive include un-hashed, and a cache keyed on
// an incomplete input set returns an artifact that no longer matches the source.
// If any header in the closure cannot be resolved, allTracked is false and the
// caller must not cache.
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

	// Breadth-first over include names so deps stays in a deterministic order:
	// computeDependencyKey hashes the sequence, not a set.
	queue := []string{src}
	seen := make(map[string]bool)

	for len(queue) > 0 {
		text := queue[0]
		queue = queue[1:]

		for _, headerName := range scanIncludes(text) {
			if seen[headerName] {
				continue
			}
			seen[headerName] = true

			// In-memory headers are hashed already as part of cfg.headers, but
			// what they include is not, so they still have to be walked.
			if content, ok := inMemoryHeaders[headerName]; ok {
				queue = append(queue, content)
				continue
			}

			var foundPath string
			for _, sp := range searchPaths {
				candidate := filepath.Join(sp, headerName)
				if fi, err := os.Stat(candidate); err == nil && !fi.IsDir() {
					foundPath = candidate
					break
				}
			}

			if foundPath != "" {
				if data, err := os.ReadFile(foundPath); err == nil {
					h := sha256.Sum256(data)
					deps = append(deps, cudakernel.Dependency{
						Path:    foundPath,
						Digest:  hex.EncodeToString(h[:]),
						Tracked: true,
					})
					queue = append(queue, string(data))
					continue
				}
			}

			// Unresolved external header: record explicit inability to validate.
			deps = append(deps, cudakernel.Dependency{
				Path:     headerName,
				Tracked:  false,
				ErrorMsg: "unable to validate external header",
			})
			allTracked = false
		}
	}

	return deps, allTracked
}

// scanIncludes returns the header names named by #include directives in src, in
// order of appearance. It is a line scan, not a preprocessor: it does not
// evaluate conditionals, so a header behind a false #if is still reported.
// Over-reporting costs a cache miss; under-reporting would cost a stale hit.
func scanIncludes(src string) []string {
	var names []string
	scanner := bufio.NewScanner(strings.NewReader(src))
	scanner.Buffer(make([]byte, 0, 64*1024), maxIncludeLineSize)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if !strings.HasPrefix(line, "#include") {
			continue
		}
		rest := strings.TrimSpace(strings.TrimPrefix(line, "#include"))
		if len(rest) < 3 {
			continue
		}
		var closer byte
		switch rest[0] {
		case '"':
			closer = '"'
		case '<':
			closer = '>'
		default:
			continue
		}
		if end := strings.IndexByte(rest[1:], closer); end > 0 {
			names = append(names, rest[1:end+1])
		}
	}
	return names
}

// maxIncludeLineSize bounds a single line during the include scan. Generated
// CUDA sources can carry lines far longer than bufio.Scanner allows by default,
// and a scan that stops early silently under-reports dependencies.
const maxIncludeLineSize = 1 << 20

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
