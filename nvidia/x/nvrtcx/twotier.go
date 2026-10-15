package nvrtcx

import (
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
	untracked := func(path, reason string) {
		deps = append(deps, cudakernel.Dependency{Path: path, ErrorMsg: reason})
		allTracked = false
	}
	var searchPaths []string
	addSearchPath := func(path string) {
		if parentTraversal(path) {
			untracked(path, "parent traversal in include search paths is not tracked")
			return
		}
		searchPaths = append(searchPaths, path)
	}
	for _, f := range flags {
		name, _, _ := strings.Cut(f, "=")
		switch name {
		case "--pre-include", "-include", "--pch", "-pch", "--use-pch", "-use-pch", "--create-pch", "-create-pch":
			untracked(f, "header injection or precompiled headers are not tracked")
		}
		if strings.HasPrefix(f, "--include-path=") {
			addSearchPath(strings.TrimPrefix(f, "--include-path="))
		} else if strings.HasPrefix(f, "-I=") {
			addSearchPath(strings.TrimPrefix(f, "-I="))
		} else if strings.HasPrefix(f, "-I") && len(f) > 2 {
			addSearchPath(strings.TrimPrefix(f, "-I"))
		} else if name == "--include-path" || f == "-I" {
			untracked(f, "separate include path arguments are not tracked")
		}
	}

	// Breadth-first over include names so deps stays in a deterministic order:
	// computeDependencyKey hashes the sequence, not a set.
	type source struct {
		text, path, dir string
	}
	queue := []source{{text: src, path: "<source>"}}
	seen := make(map[string]bool)

	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]

		includes, complete := dependencyIncludes(current.text)
		if !complete {
			untracked(current.path, "include preprocessing is outside the tracked literal subset")
		}
		for _, inc := range includes {
			headerName := inc.name
			if inc.quoted && current.dir == "" {
				untracked(current.path, "quoted include source origin is not available")
			}
			if filepath.IsAbs(headerName) {
				untracked(headerName, "absolute include paths are not tracked")
				continue
			}
			// Cleaning a/.. before following a can select the wrong file when
			// a is a symlink. Refuse this spelling before any path joining.
			if parentTraversal(headerName) {
				untracked(headerName, "parent traversal in include names is not tracked")
				continue
			}
			if inc.quoted && current.dir != "" {
				if relative := filepath.Join(current.dir, headerName); relative != headerName {
					if _, ok := inMemoryHeaders[relative]; ok {
						untracked(headerName, "relative in-memory header search is not tracked")
					}
				}
			}
			paths := searchPaths
			if inc.quoted && current.dir != "" {
				paths = append([]string{current.dir}, searchPaths...)
			}

			// In-memory headers are hashed already as part of cfg.headers, but
			// what they include is not, so they still have to be walked.
			if content, ok := inMemoryHeaders[headerName]; ok {
				for _, dir := range paths {
					if _, err := os.Stat(filepath.Join(dir, headerName)); err == nil {
						untracked(headerName, "in-memory and filesystem header search overlap")
						break
					}
				}
				key := "memory:" + headerName
				if !seen[key] {
					seen[key] = true
					queue = append(queue, source{text: content, path: key, dir: filepath.Dir(headerName)})
				}
				continue
			}

			var foundPath, physicalPath string
			for _, sp := range paths {
				candidate := filepath.Join(sp, headerName)
				fi, err := os.Lstat(candidate)
				if os.IsNotExist(err) {
					continue
				}
				if err != nil {
					untracked(candidate, "unable to inspect include path")
					break
				}
				if !fi.Mode().IsRegular() {
					// In particular, a symlink to a file leaves the compiler's
					// quoted-include origin ambiguous. Directory aliases are
					// allowed, and deduplicated by the physical file below.
					untracked(candidate, "include is not a regular file or is a file symlink")
					break
				}
				resolved, err := filepath.EvalSymlinks(candidate)
				if err == nil {
					resolved, err = filepath.Abs(resolved)
				}
				if err != nil {
					untracked(candidate, "unable to resolve physical include path")
					break
				}
				foundPath, physicalPath = candidate, resolved
				break
			}

			if foundPath != "" {
				key := "file:" + physicalPath
				if seen[key] {
					continue
				}
				seen[key] = true
				if data, err := os.ReadFile(physicalPath); err == nil {
					h := sha256.Sum256(data)
					deps = append(deps, cudakernel.Dependency{
						Path:    foundPath,
						Digest:  hex.EncodeToString(h[:]),
						Tracked: true,
					})
					queue = append(queue, source{text: string(data), path: foundPath, dir: filepath.Dir(foundPath)})
					continue
				}
			}

			// Unresolved external header: record explicit inability to validate.
			untracked(headerName, "unable to validate external header")
		}
	}

	return deps, allTracked
}

func parentTraversal(path string) bool {
	for _, elem := range strings.Split(filepath.ToSlash(path), "/") {
		if elem == ".." {
			return true
		}
	}
	return false
}

type dependencyInclude struct {
	name   string
	quoted bool
}

// dependencyIncludes recognizes literal includes without evaluating macros or
// conditional compilation. Unsupported preprocessing makes complete false, even
// if some literal includes were found. In particular, comments and line splicing
// can change directive boundaries, so their presence must not hide a dependency.
func dependencyIncludes(src string) (includes []dependencyInclude, complete bool) {
	complete = !strings.ContainsAny(src, "\\") && !strings.Contains(src, "/*") &&
		!strings.Contains(src, "%:") && !strings.Contains(src, "??")
	for line := range strings.Lines(src) {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "#") {
			continue
		}
		line = strings.TrimSpace(line[1:])
		end := 0
		for end < len(line) && (line[end] >= 'a' && line[end] <= 'z' || line[end] == '_') {
			end++
		}
		directive, rest := line[:end], strings.TrimSpace(line[end:])
		switch directive {
		case "define", "undef", "if", "ifdef", "ifndef", "elif", "else", "endif", "error", "warning":
			continue
		case "":
			if rest != "" && !strings.HasPrefix(rest, "//") {
				complete = false
			}
			continue
		case "include":
		default:
			complete = false
			continue
		}
		if len(rest) < 3 {
			complete = false
			continue
		}
		closer := byte('"')
		quoted := rest[0] == '"'
		if rest[0] == '<' {
			closer = '>'
		} else if !quoted {
			complete = false
			continue
		}
		end = strings.IndexByte(rest[1:], closer)
		if end <= 0 {
			complete = false
			continue
		}
		includes = append(includes, dependencyInclude{name: rest[1 : end+1], quoted: quoted})
		tail := strings.TrimSpace(rest[end+2:])
		if tail != "" && !strings.HasPrefix(tail, "//") {
			complete = false
		}
	}
	return includes, complete
}

// scanIncludes returns recognized literal header names, including duplicates.
// Dependency admission uses dependencyIncludes's completeness result as well.
func scanIncludes(src string) []string {
	var names []string
	includes, _ := dependencyIncludes(src)
	for _, inc := range includes {
		names = append(names, inc.name)
	}
	return names
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
