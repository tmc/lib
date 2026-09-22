package main

import (
	"os"
	"path/filepath"
)

// The agent is C, links against the CUDA toolkit's CUPTI, and is built
// where a toolkit is present rather than by go build. The script installs
// it where go install puts binaries, which is where findAgent looks.
//
//go:generate sh ../../../internal/cudainject/native/build_agent.sh

// agentName is the injected CUPTI agent's file name, as
// internal/cudainject's generate directive builds it.
const agentName = "cupti_agent.so"

// findAgent looks for the injected agent beside the command and then where
// go install puts binaries, so an agent built by internal/cudainject and
// copied next to the rest of a toolchain is found without naming its path.
// It returns an empty string when no candidate exists, which leaves the
// capture to report that nothing was injected.
func findAgent(exeDir string, getenv func(string) string) string {
	dirs := []string{exeDir}
	if bin := getenv("GOBIN"); bin != "" {
		dirs = append(dirs, bin)
	}
	for _, p := range filepath.SplitList(getenv("GOPATH")) {
		if p != "" {
			dirs = append(dirs, filepath.Join(p, "bin"))
		}
	}
	// The default GOPATH is only a default: a GOPATH that is set has
	// already been added, and one that is not means the toolchain would
	// have installed into the home directory.
	if getenv("GOPATH") == "" {
		if home := getenv("HOME"); home != "" {
			dirs = append(dirs, filepath.Join(home, "go", "bin"))
		}
	}
	for _, dir := range dirs {
		if dir == "" {
			continue
		}
		path := filepath.Join(dir, agentName)
		if fi, err := os.Stat(path); err == nil && !fi.IsDir() {
			return path
		}
	}
	return ""
}
