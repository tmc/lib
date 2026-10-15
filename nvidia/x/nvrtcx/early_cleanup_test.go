package nvrtcx

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"io"
	"reflect"
	"testing"
)

// Like nvrtc.Program.Destroy, Destroy has a pointer receiver and clears the
// handle on success. The callback records the value reached by a later Close.
type earlyCleanupProgram struct {
	handle  uintptr
	destroy func(uintptr) error
}

func (p *earlyCleanupProgram) Destroy() error {
	if p.handle == 0 {
		return errors.New("destroy zero program")
	}
	if err := p.destroy(p.handle); err != nil {
		return err
	}
	p.handle = 0
	return nil
}

func TestEarlyCleanupOwnership(t *testing.T) {
	for _, tt := range []struct {
		name string
		err  error
	}{
		{"name expression", errors.New("register failed")},
		{"flow callback", errors.New("callback failed")},
		{"initial cancellation", context.Canceled},
		{"retry cancellation", context.DeadlineExceeded},
	} {
		t.Run(tt.name, func(t *testing.T) {
			for _, fail := range []bool{false, true} {
				t.Run(fmt.Sprint(fail), func(t *testing.T) {
					destroyErr := errors.New("destroy failed")
					var handles []uintptr
					abort := func() (prog earlyCleanupProgram, err error) {
						prog = earlyCleanupProgram{17, func(handle uintptr) error {
							handles = append(handles, handle)
							if fail && len(handles) <= 2 {
								return destroyErr
							}
							return nil
						}}
						// The production setup branches return a zero named result.
						// A method value from prog would retain &prog, which is reset.
						pending := prog
						_, err = finishProgram(nil, fmt.Errorf("setup: %w", tt.err), pending.Destroy)
						return earlyCleanupProgram{}, err
					}
					prog, err := abort()
					if prog.handle != 0 || !errors.Is(err, tt.err) || errors.Is(err, destroyErr) != fail {
						t.Fatalf("program=%d error=%v", prog.handle, err)
					}
					var closer io.Closer
					if errors.As(err, &closer) != fail {
						t.Fatalf("cleanup owner presence differs: %v", err)
					}
					wantHandles := []uintptr{17}
					if fail {
						if err := closer.Close(); !errors.Is(err, destroyErr) {
							t.Fatalf("failed retry = %v", err)
						}
						if err := closer.Close(); err != nil {
							t.Fatal(err)
						}
						if err := closer.Close(); err != nil {
							t.Fatal(err)
						}
						wantHandles = []uintptr{17, 17, 17}
					}
					if !reflect.DeepEqual(handles, wantHandles) {
						t.Fatalf("destroyed handles = %v, want %v", handles, wantHandles)
					}
				})
			}
		})
	}
}

func TestEarlyCleanupRetryRetirement(t *testing.T) {
	compileErr, logErr := errors.New("heap exhausted"), errors.New("log failed")
	destroyErr := errors.New("destroy failed")
	for _, fail := range []bool{false, true} {
		t.Run(fmt.Sprint(fail), func(t *testing.T) {
			var handles []uintptr
			prog := earlyCleanupProgram{23, func(handle uintptr) error {
				handles = append(handles, handle)
				if fail && len(handles) == 1 {
					return destroyErr
				}
				return nil
			}}
			replacements := 0
			retry := func() error {
				pending := prog
				if _, err := finishProgram(nil, nil, pending.Destroy); err != nil {
					return errors.Join(compilerDiagnostic(compileErr, logErr, "old log"), err)
				}
				replacements++
				prog.handle = 29
				return nil
			}
			err := retry()
			if fail {
				for _, want := range []error{compileErr, logErr, destroyErr} {
					if !errors.Is(err, want) {
						t.Errorf("lost %v in %v", want, err)
					}
				}
				if replacements != 0 || prog.handle != 23 {
					t.Fatalf("replacement started: calls=%d program=%d", replacements, prog.handle)
				}
				var closer io.Closer
				if !errors.As(err, &closer) {
					t.Fatal("lost retired program cleanup owner")
				}
				// Even if the caller's variable changes, Close owns the old handle.
				prog.handle = 31
				if err := closer.Close(); err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(handles, []uintptr{23, 23}) {
					t.Fatalf("destroyed handles = %v, want [23 23]", handles)
				}
			} else if err != nil || replacements != 1 || prog.handle != 29 {
				t.Fatalf("successful retirement: error=%v replacements=%d program=%d", err, replacements, prog.handle)
			}
		})
	}
}

// The CPU transition tests above cannot execute NVRTC. Check the actual five
// production sites use that transition, with a stable copy before binding the
// pointer receiver, and that failed retirement returns before replacement.
func TestEarlyCleanupProductionSites(t *testing.T) {
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "nvrtcx.go", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	text := func(node ast.Node) string {
		var b bytes.Buffer
		if err := format.Node(&b, fset, node); err != nil {
			t.Fatal(err)
		}
		return b.String()
	}
	want := map[string]int{
		`_, err = finishProgram(nil, fmt.Errorf("nvrtcx: register %q: %w", expression, err), pending.Destroy)`:                                                1,
		`_, err = finishProgram(nil, fmt.Errorf("nvrtcx: set flow callback: %w", err), pending.Destroy)`:                                                      1,
		`return finishProgram(nil, fmt.Errorf("nvrtcx: compile: %w", cfg.ctx.Err()), pending.Destroy)`:                                                        2,
		"if _, err := finishProgram(nil, nil, pending.Destroy); err != nil {\n\treturn nil, errors.Join(compilerDiagnostic(compileErr, logErr, log), err)\n}": 1,
	}
	ast.Inspect(file, func(node ast.Node) bool {
		block, ok := node.(*ast.BlockStmt)
		if !ok {
			return true
		}
		for i, stmt := range block.List {
			s := text(stmt)
			if _, ok := want[s]; !ok {
				continue
			}
			want[s]--
			if i == 0 || text(block.List[i-1]) != "pending := prog" {
				t.Errorf("cleanup lacks stable program copy: %s", s)
			}
			if _, ok := stmt.(*ast.AssignStmt); ok {
				if i+1 == len(block.List) || text(block.List[i+1]) != `return 0, "", nil, nil, err` {
					t.Error("setup failure must return the retained cleanup error")
				}
			}
			if _, ok := stmt.(*ast.IfStmt); ok {
				if i+1 == len(block.List) || text(block.List[i+1]) != "prog, log, compileErr, logErr, err = compileOnce()" {
					t.Error("PCH replacement must follow guarded retirement")
				}
			}
		}
		return true
	})
	for site, remaining := range want {
		if remaining != 0 {
			t.Errorf("cleanup site count differs by %d: %s", remaining, site)
		}
	}
	// Pin the pointer-receiver assumption exercised by earlyCleanupProgram.
	program, err := parser.ParseFile(fset, "../../nvrtc/program.gen.go", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	for _, decl := range program.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if ok && fn.Name.Name == "Destroy" && fn.Recv != nil {
			if text(fn.Recv.List[0].Type) != "*Program" {
				t.Fatal("update cleanup receiver test for changed Program.Destroy receiver")
			}
			return
		}
	}
	t.Fatal("Program.Destroy declaration not found")
}
