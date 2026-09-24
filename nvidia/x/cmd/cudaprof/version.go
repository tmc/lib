package main

import (
	"errors"
	"fmt"
	"io"
	"os"
	"runtime"
	"runtime/debug"
)

// buildStamp identifies the cudaprof build that wrote a receipt.
// A field the binary does not record, such as the VCS fields of a binary
// built outside a repository or by go run, is "unknown".
type buildStamp struct {
	Module    string `json:"module"`
	Revision  string `json:"revision"`
	Time      string `json:"time"`
	Modified  string `json:"modified"`
	GoVersion string `json:"go_version"`
}

// currentBuild returns the build stamp of the running binary.
func currentBuild() buildStamp {
	info, _ := debug.ReadBuildInfo()
	return stampOf(info)
}

// stampOf returns the build stamp recorded in info, which may be nil.
func stampOf(info *debug.BuildInfo) buildStamp {
	b := buildStamp{
		Module:    "unknown",
		Revision:  "unknown",
		Time:      "unknown",
		Modified:  "unknown",
		GoVersion: runtime.Version(),
	}
	if info == nil {
		return b
	}
	if info.Main.Path != "" {
		b.Module = info.Main.Path
	}
	if info.GoVersion != "" {
		b.GoVersion = info.GoVersion
	}
	for _, s := range info.Settings {
		switch s.Key {
		case "vcs.revision":
			b.Revision = s.Value
		case "vcs.time":
			b.Time = s.Value
		case "vcs.modified":
			b.Modified = s.Value
		}
	}
	return b
}

// write writes b as one key=value pair per line.
func (b buildStamp) write(w io.Writer) error {
	_, err := fmt.Fprintf(w, "module=%s\nrevision=%s\ntime=%s\nmodified=%s\ngo=%s\n",
		b.Module, b.Revision, b.Time, b.Modified, b.GoVersion)
	return err
}

// runVersionCommand implements "cudaprof version".
func runVersionCommand(args []string) error {
	if len(args) != 1 {
		return errors.New(versionUsage)
	}
	return currentBuild().write(os.Stdout)
}
