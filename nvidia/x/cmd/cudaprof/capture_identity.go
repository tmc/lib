package main

import (
	"crypto/sha256"
	"debug/buildinfo"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// A capture names its command, but a command is a name, not a program: the
// same argv runs whatever binary sits at that path when it runs, and a
// rebuild between two arms leaves every recorded argument identical. The
// executable identity pins the bytes, and for a Go binary the commit it was
// built from, so two captures can be shown to have run the same program --
// or not -- without trusting whoever launched them.

// executableIdentity identifies one file the capture ran.
type executableIdentity struct {
	// Path is the resolved file, with symbolic links evaluated.
	Path   string `json:"path"`
	Size   int64  `json:"size,omitempty"`
	SHA256 string `json:"sha256,omitempty"`

	// Wrapper is the env(1) the command named first, when the target was
	// found by reading past it.
	Wrapper string `json:"wrapper,omitempty"`

	// Go is the build information of a Go binary, and absent for any other.
	Go *goBuildIdentity `json:"go,omitempty"`

	// Error says why the file could not be identified. The other fields
	// are then unset, never zero values standing in for a reading.
	Error string `json:"error,omitempty"`
}

// goBuildIdentity is what a Go binary says about its own build. The vcs
// settings are recorded verbatim and omitted when the build did not stamp
// them: a binary built with -buildvcs=false, or outside a repository, has no
// revision, and that absence is itself the answer.
type goBuildIdentity struct {
	GoVersion   string `json:"go_version"`
	MainModule  string `json:"main_module"`
	VCSRevision string `json:"vcs_revision,omitempty"`
	VCSTime     string `json:"vcs_time,omitempty"`
	VCSModified string `json:"vcs_modified,omitempty"`
}

// identifyTarget identifies the program command runs. path is the file
// exec resolved command[0] to, env the environment the command receives and
// dir the directory it runs in. A leading env(1) is read through, with its
// NAME=VALUE assignments applied to the PATH the program is then looked up
// in, as env itself would.
func identifyTarget(path string, command, env []string, dir string) *executableIdentity {
	// exec runs a relative path from the command's directory.
	if !filepath.IsAbs(path) {
		path = filepath.Join(dir, path)
	}
	if filepath.Base(path) != "env" {
		return identifyExecutable(path)
	}
	prog, err := envProgram(command[1:], env, dir)
	if err != nil {
		return &executableIdentity{Wrapper: path, Error: err.Error()}
	}
	id := identifyExecutable(prog)
	id.Wrapper = path
	return id
}

// envProgram returns the program env(1) runs for args, the arguments after
// "env". It understands -i, -u NAME and assignments; any other option leaves
// the program unresolved rather than guessed.
func envProgram(args, env []string, dir string) (string, error) {
	pathEnv := lookupEnv(env, "PATH")
	for i := 0; i < len(args); i++ {
		arg := args[i]
		switch {
		case arg == "-i" || arg == "-" || arg == "--ignore-environment":
			pathEnv = ""
		case arg == "-u":
			i++
		case strings.HasPrefix(arg, "--unset="):
		case arg == "--":
		case strings.HasPrefix(arg, "-"):
			return "", fmt.Errorf("env option %s is not understood", arg)
		case strings.Contains(arg, "="):
			if name, value, _ := strings.Cut(arg, "="); name == "PATH" {
				pathEnv = value
			}
		default:
			return lookPathIn(arg, pathEnv, dir)
		}
	}
	return "", errors.New("env names no program")
}

// lookPathIn finds name as a program run from dir with the given PATH would.
func lookPathIn(name, pathEnv, dir string) (string, error) {
	if strings.Contains(name, "/") {
		if !filepath.IsAbs(name) {
			name = filepath.Join(dir, name)
		}
		return name, nil
	}
	for _, d := range filepath.SplitList(pathEnv) {
		if d == "" {
			d = "."
		}
		p := filepath.Join(d, name)
		if !filepath.IsAbs(p) {
			p = filepath.Join(dir, p)
		}
		if fi, err := os.Stat(p); err == nil && fi.Mode().IsRegular() && fi.Mode()&0o111 != 0 {
			return p, nil
		}
	}
	return "", fmt.Errorf("%s not found in PATH", name)
}

// identifyExecutable hashes the file at path and reads its Go build
// information, if it has any.
func identifyExecutable(path string) *executableIdentity {
	abs, err := filepath.Abs(path)
	if err == nil {
		abs, err = filepath.EvalSymlinks(abs)
	}
	if err != nil {
		return &executableIdentity{Path: path, Error: err.Error()}
	}
	f, err := os.Open(abs)
	if err != nil {
		return &executableIdentity{Path: abs, Error: err.Error()}
	}
	defer f.Close()
	h := sha256.New()
	n, err := io.Copy(h, f)
	if err != nil {
		return &executableIdentity{Path: abs, Error: fmt.Sprintf("hash: %v", err)}
	}
	id := &executableIdentity{Path: abs, Size: n, SHA256: hex.EncodeToString(h.Sum(nil))}
	if info, err := buildinfo.Read(f); err == nil {
		g := &goBuildIdentity{GoVersion: info.GoVersion, MainModule: info.Main.Path}
		for _, s := range info.Settings {
			switch s.Key {
			case "vcs.revision":
				g.VCSRevision = s.Value
			case "vcs.time":
				g.VCSTime = s.Value
			case "vcs.modified":
				g.VCSModified = s.Value
			}
		}
		id.Go = g
	}
	return id
}
