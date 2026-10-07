package cudainject_test

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudainject"
)

func TestInstalledModuleAgentBuild(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("Linux agent compiler fixture requires a Unix shell")
	}
	_, file, _, _ := runtime.Caller(0)
	module := filepath.Clean(filepath.Join(filepath.Dir(file), "../.."))
	root := t.TempDir()
	installed := filepath.Join(root, "installed")
	if b, err := exec.Command("cp", "-R", module, installed).CombinedOutput(); err != nil {
		t.Fatalf("module copy: %v %s", err, b)
	}
	consumer := filepath.Join(root, "consumer")
	if err := os.Mkdir(consumer, 0700); err != nil {
		t.Fatal(err)
	}
	mod := "module consumer.example/agent\n\ngo 1.27.0\n\nrequire github.com/tmc/lib/nvidia v0.0.0\nreplace github.com/tmc/lib/nvidia => " + installed + "\n"
	if err := os.WriteFile(filepath.Join(consumer, "go.mod"), []byte(mod), 0600); err != nil {
		t.Fatal(err)
	}
	if b, err := exec.Command("cp", filepath.Join(installed, "go.sum"), filepath.Join(consumer, "go.sum")).CombinedOutput(); err != nil {
		t.Fatalf("copy managed sum: %v %s", err, b)
	}
	source, err := os.ReadFile(filepath.Join(installed, "x/cudainject/testdata/consumer/main.go"))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(consumer, "main.go"), source, 0600); err != nil {
		t.Fatal(err)
	}
	binary := filepath.Join(root, "consumer-bin")
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	build := exec.CommandContext(ctx, "go", "build", "-mod=mod", "-trimpath", "-o", binary, ".")
	build.Dir = consumer
	build.Env = append(os.Environ(), "GOWORK=off", "GOTOOLCHAIN=local", "GOPROXY=off")
	if b, err := build.CombinedOutput(); err != nil {
		t.Fatalf("external build: %v %s", err, b)
	}
	// The executable must work after its entire module installation and source
	// consumer have gone. Neither trimpath nor an installed module offers a
	// runtime path back to the agent C inputs.
	if err := os.RemoveAll(installed); err != nil {
		t.Fatal(err)
	}
	if err := os.RemoveAll(consumer); err != nil {
		t.Fatal(err)
	}
	compiler := filepath.Join(root, "fixture-cc")
	script := `#!/bin/sh
set -eu
out=
src=
while [ "$#" -gt 0 ]; do
 case "$1" in
 -o) shift; out=$1 ;;
 */cupti_agent.c) src=$1 ;;
 esac
 shift
done
test -n "$src"
test -s "$src"
test -s "$(dirname "$src")/cupti_memory.h"
test -s "$(dirname "$src")/cupti_dependency.h"
grep 'static void \*flush_worker' "$src" >/dev/null
printf 'compiler fixture, not a native shared object' > "$out"
`
	if err := os.WriteFile(compiler, []byte(script), 0700); err != nil {
		t.Fatal(err)
	}
	run := exec.CommandContext(ctx, binary, compiler, root, filepath.Join(root, "build"))
	run.Dir = root
	if b, err := run.CombinedOutput(); err != nil || string(b) != "3\n" {
		t.Fatalf("installed consumer: %v %s", err, b)
	}
}

func TestBuildRefusesUnboundedAndInvalidOptions(t *testing.T) {
	if _, err := cudainject.BuildAgent(context.Background(), cudainject.BuildOptions{}); err == nil {
		t.Fatal("no deadline accepted")
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if _, err := cudainject.BuildAgent(ctx, cudainject.BuildOptions{Target: "unknown"}); err == nil {
		t.Fatal("invalid options accepted")
	}
}

func TestBuildMaterializesSources(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("Unix compiler fixture")
	}
	dir := t.TempDir()
	compiler := filepath.Join(dir, "cc")
	if err := os.WriteFile(compiler, []byte("#!/bin/sh\nset -eu\nwhile [ \"$1\" != -o ]; do shift; done\nshift\nprintf fixture > \"$1\"\n"), 0700); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	agent, err := cudainject.BuildAgent(ctx, cudainject.BuildOptions{Compiler: compiler, CUDAHome: dir, Directory: filepath.Join(dir, "build"), Target: "sbsa-linux"})
	if err != nil {
		t.Fatal(err)
	}
	if len(agent.Sources) != 3 {
		t.Fatalf("sources %v", agent.Sources)
	}
	for _, name := range []string{"cupti_agent.c", "cupti_memory.h", "cupti_dependency.h"} {
		if agent.Sources[name] == "" {
			t.Fatalf("missing source %s", name)
		}
		if b, err := os.ReadFile(filepath.Join(dir, "build", name)); err != nil || len(b) == 0 {
			t.Fatalf("source %s: %v", name, err)
		}
	}
	if _, err := cudainject.BuildAgent(ctx, cudainject.BuildOptions{Compiler: compiler, CUDAHome: dir, Directory: filepath.Join(dir, "build")}); err == nil {
		t.Fatal("existing build directory accepted")
	}
}

func TestBuildExplicitCUPTIDirectories(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("Unix compiler fixture")
	}
	root := t.TempDir()
	compiler := filepath.Join(root, "cc")
	capture := filepath.Join(root, "argv")
	script := "#!/bin/sh\nset -eu\nprintf '%s\\n' \"$@\" > '" + capture + "'\nwhile [ \"$1\" != -o ]; do shift; done\nshift\nprintf fixture > \"$1\"\n"
	if err := os.WriteFile(compiler, []byte(script), 0700); err != nil {
		t.Fatal(err)
	}
	for _, explicit := range []bool{false, true} {
		name := "default"
		if explicit {
			name = "explicit"
		}
		t.Run(name, func(t *testing.T) {
			o := cudainject.BuildOptions{Compiler: compiler, CUDAHome: root, Directory: filepath.Join(root, name), Target: "x86_64-linux"}
			if explicit {
				o.CUPTIInclude = filepath.Join(root, "separate include")
				o.CUPTILibrary = filepath.Join(root, "separate lib")
			}
			ctx, cancel := context.WithTimeout(context.Background(), time.Second)
			defer cancel()
			a, e := cudainject.BuildAgent(ctx, o)
			if e != nil {
				t.Fatal(e)
			}
			b, e := os.ReadFile(capture)
			if e != nil {
				t.Fatal(e)
			}
			if string(b) != strings.Join(a.Arguments, "\n")+"\n" {
				t.Fatalf("actual argv differs: %q", b)
			}
			if explicit {
				if len(a.Arguments) < 2 || a.Arguments[0] != "-I"+o.CUPTIInclude || a.Arguments[1] != "-L"+o.CUPTILibrary {
					t.Fatal(a.Arguments)
				}
			} else {
				for _, arg := range a.Arguments {
					if strings.Contains(arg, "separate") {
						t.Fatal(a.Arguments)
					}
				}
			}
			for _, want := range []string{"-I" + filepath.Join(root, "targets/x86_64-linux/include"), "-L" + filepath.Join(root, "targets/x86_64-linux/lib"), "-lcupti"} {
				found := false
				for _, arg := range a.Arguments {
					if arg == want {
						found = true
					}
				}
				if !found {
					t.Fatalf("missing %q: %v", want, a.Arguments)
				}
			}
		})
	}
}

func TestBuildCUPTIDirectoryRefusalsBeforeCompiler(t *testing.T) {
	root := t.TempDir()
	for _, pair := range [][2]string{{root, ""}, {"", root}, {"relative", root}, {root, "relative"}} {
		dir := filepath.Join(root, "must-not-exist")
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		_, err := cudainject.BuildAgent(ctx, cudainject.BuildOptions{Compiler: filepath.Join(root, "missing-compiler"), CUDAHome: root, Directory: dir, CUPTIInclude: pair[0], CUPTILibrary: pair[1]})
		cancel()
		if err == nil {
			t.Fatal("invalid pair accepted")
		}
		if _, e := os.Stat(dir); !os.IsNotExist(e) {
			t.Fatalf("side effect before refusal: %v", e)
		}
	}
}
