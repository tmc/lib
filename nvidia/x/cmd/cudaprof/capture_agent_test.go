package main

import (
	"os"
	"path/filepath"
	"testing"
)

// TestFindAgent checks where the command looks for the injected agent when
// no path was given: beside itself first, because a toolchain installed
// together belongs together, and then where go install puts binaries.
func TestFindAgent(t *testing.T) {
	dir := t.TempDir()
	exeDir := filepath.Join(dir, "exe")
	gobin := filepath.Join(dir, "gobin")
	gopath := filepath.Join(dir, "gopath")
	home := filepath.Join(dir, "home")
	for _, d := range []string{exeDir, gobin, filepath.Join(gopath, "bin"), filepath.Join(home, "go", "bin")} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	write := func(dir string) string {
		path := filepath.Join(dir, agentName)
		if err := os.WriteFile(path, []byte("agent"), 0o644); err != nil {
			t.Fatal(err)
		}
		return path
	}

	env := func(m map[string]string) func(string) string {
		return func(k string) string { return m[k] }
	}
	for _, tt := range []struct {
		name  string
		build func() string // returns the path the search must find
		env   map[string]string
	}{{
		name:  "beside the command",
		build: func() string { return write(exeDir) },
		env:   map[string]string{"GOBIN": gobin, "HOME": home},
	}, {
		name:  "in GOBIN",
		build: func() string { return write(gobin) },
		env:   map[string]string{"GOBIN": gobin, "GOPATH": gopath, "HOME": home},
	}, {
		name:  "in GOPATH bin",
		build: func() string { return write(filepath.Join(gopath, "bin")) },
		env:   map[string]string{"GOPATH": gopath, "HOME": home},
	}, {
		name:  "in the default GOPATH bin",
		build: func() string { return write(filepath.Join(home, "go", "bin")) },
		env:   map[string]string{"HOME": home},
	}} {
		t.Run(tt.name, func(t *testing.T) {
			want := tt.build()
			defer os.Remove(want)
			if got := findAgent(exeDir, env(tt.env)); got != want {
				t.Errorf("findAgent = %q, want %q", got, want)
			}
		})
	}

	t.Run("nowhere", func(t *testing.T) {
		if got := findAgent(exeDir, env(map[string]string{"GOBIN": gobin, "HOME": home})); got != "" {
			t.Errorf("findAgent = %q, want no candidate", got)
		}
	})

	// A GOBIN naming a directory holding something else must not be taken
	// for the agent, and a GOPATH left unset must not send the search to
	// bin of the empty path.
	t.Run("a directory named like the agent", func(t *testing.T) {
		if err := os.MkdirAll(filepath.Join(gobin, agentName), 0o755); err != nil {
			t.Fatal(err)
		}
		defer os.Remove(filepath.Join(gobin, agentName))
		if got := findAgent(exeDir, env(map[string]string{"GOBIN": gobin, "HOME": home})); got != "" {
			t.Errorf("findAgent = %q, want no candidate", got)
		}
	})
}
