//go:build linux

package cudainject_test

import (
	"bufio"
	"context"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"testing"
	"time"

	"golang.org/x/sys/unix"
)

func TestPeriodicRuntimeCPUChild(t *testing.T) {
	if os.Getenv("CUDAINJECT_RUNTIME_CPU_CHILD") != "1" {
		return
	}
	for _, path := range strings.Split(os.Getenv("CUDAINJECT_RUNTIME_CPU_FILES"), "\n") {
		f, err := os.Open(path)
		if err != nil {
			t.Fatal(err)
		}
		data, err := unix.Mmap(int(f.Fd()), 0, 4096, unix.PROT_READ, unix.MAP_SHARED)
		if err != nil {
			t.Fatal(err)
		}
		defer unix.Munmap(data)
		defer f.Close()
	}
	fmt.Println("runtime-child-ready")
	var b [1]byte
	if n, err := os.Stdin.Read(b[:]); n != 1 || err != nil || b[0] != 'x' {
		t.Fatalf("release: %d %v", n, err)
	}
}

func TestPeriodicRuntimeActualCPUChild(t *testing.T) {
	for _, name := range []string{"mapped", "changed_body", "missing_mapping", "foreign_process"} {
		t.Run(name, func(t *testing.T) {
			dir, err := filepath.EvalSymlinks(t.TempDir())
			if err != nil {
				t.Fatal(err)
			}
			exe, err := os.Executable()
			if err != nil {
				t.Fatal(err)
			}
			exe, err = filepath.EvalSymlinks(exe)
			if err != nil {
				t.Fatal(err)
			}
			paths := []string{exe}
			for _, role := range []string{"agent", "driver", "cupti"} {
				path := filepath.Join(dir, role)
				if err := os.WriteFile(path, make([]byte, 4096), 0600); err != nil {
					t.Fatal(err)
				}
				paths = append(paths, path)
			}
			inputs, err := periodicRuntimeInputs(paths...)
			if err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithTimeout(context.Background(), 8*time.Second)
			defer cancel()
			cmd := exec.CommandContext(ctx, exe, "-test.run=^TestPeriodicRuntimeCPUChild$")
			cmd.WaitDelay = time.Second
			mapped := paths[1:]
			if name == "missing_mapping" {
				mapped = mapped[:2]
			}
			cmd.Env = append(os.Environ(), "CUDAINJECT_RUNTIME_CPU_CHILD=1", "CUDAINJECT_RUNTIME_CPU_FILES="+strings.Join(mapped, "\n"))
			in, err := cmd.StdinPipe()
			if err != nil {
				t.Fatal(err)
			}
			out, err := cmd.StdoutPipe()
			if err != nil {
				t.Fatal(err)
			}
			if err := cmd.Start(); err != nil {
				t.Fatal(err)
			}
			waited := false
			defer func() {
				if !waited {
					cancel()
					if err := cmd.Wait(); err == nil {
						t.Error("abandoned child unexpectedly succeeded")
					}
				}
			}()
			reader := bufio.NewReaderSize(out, 4096)
			line, err := reader.ReadString('\n')
			if err != nil || line != "runtime-child-ready\n" {
				t.Fatalf("readiness: %q %v", line, err)
			}
			if name == "changed_body" {
				body := make([]byte, 4096)
				body[0] = 1
				if err := os.WriteFile(paths[3], body, 0600); err != nil {
					t.Fatal(err)
				}
			}
			pid := cmd.Process.Pid
			if name == "foreign_process" {
				// Map the same files in the caller too, so refusal must come from
				// the actual parent relation rather than an absent mapping.
				for _, path := range paths[1:] {
					f, err := os.Open(path)
					if err != nil {
						t.Fatal(err)
					}
					data, err := unix.Mmap(int(f.Fd()), 0, 4096, unix.PROT_READ, unix.MAP_SHARED)
					closeErr := f.Close()
					if err != nil || closeErr != nil {
						t.Fatalf("caller map: %v %v", err, closeErr)
					}
					defer unix.Munmap(data)
				}
				pid = os.Getpid()
			}
			receipt, observedErr := observePeriodicRuntime(pid, inputs)
			if observedErr != nil {
				cancel()
			}
			releaseErr := releasePeriodicWorker(in, cancel, cmd.Wait)
			waited = true
			if cmd.ProcessState == nil {
				t.Fatal("worker was not waited")
			}
			if _, err := syscall.Wait4(cmd.Process.Pid, nil, syscall.WNOHANG, nil); err != syscall.ECHILD {
				t.Fatalf("worker reaping unproven: %v", err)
			}
			if name == "mapped" {
				if observedErr != nil || releaseErr != nil || receipt.Process.PID != cmd.Process.Pid || len(receipt.Files) != 4 || receipt.RawMaps == "" {
					t.Fatalf("observation: %+v %v release=%v", receipt, observedErr, releaseErr)
				}
			} else if observedErr == nil {
				t.Fatal("invalid runtime observation accepted")
			}
		})
	}
}

func TestPeriodicRuntimeDescriptorRefusals(t *testing.T) {
	for _, name := range []string{"missing", "symlink", "fifo", "duplicate"} {
		t.Run(name, func(t *testing.T) {
			dir := t.TempDir()
			path := filepath.Join(dir, "body")
			switch name {
			case "symlink":
				if err := os.Symlink("target", path); err != nil {
					t.Fatal(err)
				}
			case "fifo":
				if err := unix.Mkfifo(path, 0600); err != nil {
					t.Fatal(err)
				}
			case "duplicate":
				if err := os.WriteFile(path, []byte("body"), 0600); err != nil {
					t.Fatal(err)
				}
			}
			if _, err := periodicRuntimeInputs(path, path, path, path); err == nil {
				t.Fatal("invalid descriptor input accepted")
			}
		})
	}
}

func TestPeriodicRuntimeMappingRefusals(t *testing.T) {
	dir := t.TempDir()
	var inputs []periodicRuntimeFile
	var lines []string
	for i := 0; i < 4; i++ {
		path := filepath.Join(dir, fmt.Sprint(i))
		if err := os.WriteFile(path, []byte("body"), 0600); err != nil {
			t.Fatal(err)
		}
		f, err := periodicRuntimeBody(path)
		if err != nil {
			t.Fatal(err)
		}
		inputs = append(inputs, f)
		lines = append(lines, fmt.Sprintf("1000-2000 r--p 0 %x:%x %d %s\n", unix.Major(f.Device), unix.Minor(f.Device), f.Inode, path))
	}
	valid := strings.Join(lines, "")
	if err := periodicRuntimeMappings([]byte(valid), inputs); err != nil {
		t.Fatal(err)
	}
	bad := fmt.Sprintf("1000-2000 r--p 0 %x:%x %d %s\n", unix.Major(inputs[0].Device), unix.Minor(inputs[0].Device), inputs[0].Inode+1, inputs[0].Path)
	for _, test := range []struct{ name, raw string }{
		{"missing", strings.Join(lines[:3], "")},
		{"shadow_first", bad + valid},
		{"shadow_last", valid + bad},
		{"range", strings.Replace(valid, "1000-2000", "2000-1000", 1)},
		{"permission", strings.Replace(valid, "r--p", "bad!", 1)},
		{"extent", strings.Repeat(" ", (1<<20)+1)},
	} {
		t.Run(test.name, func(t *testing.T) {
			if err := periodicRuntimeMappings([]byte(test.raw), inputs); err == nil {
				t.Fatal("invalid selected mapping accepted")
			}
		})
	}
}

func TestPeriodicRuntimeNativeWiring(t *testing.T) {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("source identity unavailable")
	}
	f, err := parser.ParseFile(token.NewFileSet(), filepath.Join(filepath.Dir(file), "periodic_native_test.go"), nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	var body *ast.BlockStmt
	for _, declaration := range f.Decls {
		if fn, ok := declaration.(*ast.FuncDecl); ok && fn.Name.Name == "TestAgentPeriodicDrainNative" {
			body = fn.Body
		}
	}
	if body == nil {
		t.Fatal("native consumer missing")
	}
	observed, finished, joined := -1, -1, -1
	for i, statement := range body.List {
		if branch, ok := statement.(*ast.IfStmt); ok {
			id, found := branch.Cond.(*ast.Ident)
			if found && id.Name == "found" {
				ast.Inspect(branch.Body, func(n ast.Node) bool {
					if call, ok := n.(*ast.CallExpr); ok {
						if id, ok := call.Fun.(*ast.Ident); ok && id.Name == "observePeriodicRuntime" {
							if len(call.Args) == 2 {
								pid, p := call.Args[0].(*ast.Ident)
								inputs, q := call.Args[1].(*ast.Ident)
								if p && q && pid.Name == "workerPID" && inputs.Name == "runtimeInputs" {
									observed = i
								}
							}
						}
						if selector, ok := call.Fun.(*ast.SelectorExpr); ok && selector.Sel.Name == "Fatal" {
							t.Error("observation refuses before worker retirement")
						}
					}
					if _, ok := n.(*ast.ReturnStmt); ok {
						t.Error("observation returns before worker retirement")
					}
					return true
				})
			}
		}
		if assignment, ok := statement.(*ast.AssignStmt); ok {
			for _, rhs := range assignment.Rhs {
				call, ok := rhs.(*ast.CallExpr)
				if !ok {
					continue
				}
				if id, ok := call.Fun.(*ast.Ident); ok && id.Name == "finishPeriodicWorker" {
					finished = i
				}
				if selector, ok := call.Fun.(*ast.SelectorExpr); ok && selector.Sel.Name == "Join" {
					for _, arg := range call.Args {
						if id, ok := arg.(*ast.Ident); ok && id.Name == "runtimeErr" {
							joined = i
						}
					}
				}
			}
		}
	}
	if observed < 0 || finished != observed+1 || joined != finished+1 {
		t.Fatalf("observation/retirement/error order: %d/%d/%d", observed, finished, joined)
	}
}
