package main

import (
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"testing"

	"rsc.io/script"
	"rsc.io/script/scripttest"
)

// TestScript runs trainstat as a program on files and compares its output
// as text. The scripts also build benchstat, to check that trainstat's
// benchfmt output is benchstat's input, and run go test in a scratch module
// that uses the trainstat package from this module.
func TestScript(t *testing.T) {
	if testing.Short() {
		t.Skip("builds binaries")
	}
	bin := t.TempDir()
	for _, pkg := range []string{".", "golang.org/x/perf/cmd/benchstat"} {
		build := exec.Command("go", "build", "-o", bin, pkg)
		build.Env = append(os.Environ(), "GOWORK=off")
		if out, err := build.CombinedOutput(); err != nil {
			t.Fatalf("build %s: %v\n%s", pkg, err, out)
		}
	}
	fixtures, err := filepath.Abs("testdata")
	if err != nil {
		t.Fatal(err)
	}
	root, err := filepath.Abs("../../..")
	if err != nil {
		t.Fatal(err)
	}
	engine := &script.Engine{
		Cmds:  scripttest.DefaultCmds(),
		Conds: scripttest.DefaultConds(),
	}
	scripttest.Test(t, context.Background(), engine, scriptEnv(bin, fixtures, root), "testdata/script/*.txt")
}

func scriptEnv(bin, fixtures, root string) []string {
	// Keep HOME unchanged and use the caller's Go caches and offline policy.
	// Scripttest gives each script its own temporary directory.
	env := []string{
		"PATH=" + bin + string(os.PathListSeparator) + os.Getenv("PATH"),
		"HOME=" + os.Getenv("HOME"),
		"GOWORK=off",
		"TESTDATA=" + fixtures,
		"NVIDIA=" + root,
	}
	for _, key := range []string{"GOPATH", "GOMODCACHE", "GOCACHE", "GOPROXY", "GOSUMDB", "GOENV", "GOTOOLCHAIN", "GOROOT", "GOTMPDIR"} {
		if value, ok := os.LookupEnv(key); ok {
			env = append(env, key+"="+value)
		}
	}
	return env
}

func TestScriptGoEnvironment(t *testing.T) {
	for _, tc := range []struct {
		name     string
		modcache bool
	}{
		{"gopath", false},
		{"modcache", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			work := t.TempDir()
			gopath := filepath.Join(work, "go")
			modcache := filepath.Join(gopath, "pkg", "mod")
			if tc.modcache {
				modcache = filepath.Join(work, "modules")
			}
			t.Setenv("GOPATH", gopath)
			t.Setenv("GOMODCACHE", "")
			if tc.modcache {
				t.Setenv("GOMODCACHE", modcache)
			}
			buildcache := filepath.Join(work, "build")
			t.Setenv("GOCACHE", buildcache)
			t.Setenv("GOPROXY", "off")
			t.Setenv("GOSUMDB", "off")
			t.Setenv("GOENV", "off")
			t.Setenv("GOTOOLCHAIN", "local")

			cmd := exec.Command("go", "env", "-json", "GOPATH", "GOMODCACHE", "GOCACHE", "GOPROXY", "GOSUMDB", "GOENV", "GOTOOLCHAIN")
			cmd.Env = scriptEnv("", "", "")
			out, err := cmd.CombinedOutput()
			if err != nil {
				t.Fatalf("go env: %v\n%s", err, out)
			}
			var got map[string]string
			if err := json.Unmarshal(out, &got); err != nil {
				t.Fatal(err)
			}
			want := map[string]string{
				"GOPATH": gopath, "GOMODCACHE": modcache, "GOCACHE": buildcache,
				"GOPROXY": "off", "GOSUMDB": "off", "GOENV": "", "GOTOOLCHAIN": "local",
			}
			for key, value := range want {
				if got[key] != value {
					t.Errorf("go env %s = %q, want %q", key, got[key], value)
				}
			}
			// go env reports no configuration file when GOENV=off.
			if !slices.Contains(cmd.Env, "GOENV=off") {
				t.Error("script environment did not preserve GOENV=off")
			}
			for _, setting := range cmd.Env {
				if setting == "HOME="+os.Getenv("HOME") {
					return
				}
			}
			t.Error("script environment changed HOME")
		})
	}
}
