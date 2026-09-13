package cudatool

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
)

// ErrToolNotFound means no executable was found for a tool.
var ErrToolNotFound = errors.New("cuda tool not found")

// toolkitEnvVars are the environment variables that conventionally name a CUDA
// toolkit root. They are consulted before the default prefixes.
var toolkitEnvVars = []string{"CUDA_HOME", "CUDA_PATH", "CUDA_ROOT"}

// toolkitRoots are the install prefixes searched when no environment variable
// names one. /usr/local/cuda is first because a multi-version install points it
// at the version the machine is configured for.
var toolkitRoots = []string{
	"/usr/local/cuda",
	"/usr/local/cuda-13",
	"/usr/local/cuda-12",
	"/opt/cuda",
	"/usr/lib/cuda",
}

// toolkitSubdirs are the directories inside a toolkit root that hold
// executables, in preference order. bin holds the binary utilities and a
// launcher for compute-sanitizer; the compute-sanitizer directory holds that
// tool's real binary beside the libraries it needs.
var toolkitSubdirs = []string{"bin", "compute-sanitizer"}

// SearchPaths returns the executables [LookTool] tries for a tool before PATH,
// in order. The list is empty on GOOS values the CUDA toolkit does not ship
// for, leaving PATH as the only source.
//
// A toolkit install is preferred over PATH because a distribution package can
// leave an older build of a tool earlier on PATH than the toolkit's own. Such a
// binary is not merely out of date: it does not know the current devices, and
// one that shipped against different libraries fails to start at all rather
// than reporting that it is too old. Checking the toolkit first picks the build
// that matches the libraries beside it.
func SearchPaths(tool string) []string { return searchPaths(tool, runtime.GOOS, os.Getenv) }

// searchPaths is SearchPaths with the platform and environment injected, so the
// Linux ordering can be tested on any host.
func searchPaths(tool, goos string, getenv func(string) string) []string {
	if goos != "linux" || tool == "" {
		return nil
	}
	var roots []string
	for _, env := range toolkitEnvVars {
		if v := getenv(env); v != "" {
			roots = append(roots, v)
		}
	}
	roots = append(roots, toolkitRoots...)

	seen := make(map[string]bool, len(roots)*len(toolkitSubdirs))
	paths := make([]string, 0, len(roots)*len(toolkitSubdirs))
	for _, root := range roots {
		for _, sub := range toolkitSubdirs {
			p := filepath.Join(root, sub, tool)
			if seen[p] {
				continue
			}
			seen[p] = true
			paths = append(paths, p)
		}
	}
	return paths
}

// LookTool returns the executable to run for a CUDA tool.
//
// A non-empty path is resolved the way a shell would, including a bare name on
// PATH, and nothing else is tried: an explicit path is a decision, not a hint.
// An empty path tries the toolkit installs [SearchPaths] names for the tool and
// then the tool name on PATH. The error wraps [ErrToolNotFound].
func LookTool(tool, path string) (string, error) {
	if path != "" {
		found, err := exec.LookPath(path)
		if err != nil {
			return "", fmt.Errorf("%w: %v", ErrToolNotFound, err)
		}
		return found, nil
	}
	if tool == "" {
		return "", fmt.Errorf("%w: no tool named", ErrToolNotFound)
	}
	for _, candidate := range SearchPaths(tool) {
		if IsExecutableFile(candidate) {
			return candidate, nil
		}
	}
	found, err := exec.LookPath(tool)
	if err != nil {
		return "", fmt.Errorf("%w: %v", ErrToolNotFound, err)
	}
	return found, nil
}

// IsExecutableFile reports whether path is a regular file with an execute bit
// set. A directory of the same name, or a file that cannot be executed, is not
// a candidate.
func IsExecutableFile(path string) bool {
	fi, err := os.Stat(path)
	if err != nil || !fi.Mode().IsRegular() {
		return false
	}
	return fi.Mode().Perm()&0o111 != 0
}

// lookupCUDA returns an executable for a bare tool name, leaving a name that
// already carries a path separator to [exec.LookPath].
func lookupCUDA(name string) string {
	if filepath.IsAbs(name) || strings.ContainsRune(name, filepath.Separator) {
		return name
	}
	if found, err := LookTool(name, ""); err == nil {
		return found
	}
	return name
}
