//go:build !windows

package moonep

import (
	"bytes"
	"errors"
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"slices"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
)

func TestPlanAllocationFailureOwnership(t *testing.T) {
	planCleanupNoDriver(t)
	l := &Layout{Config: Config{R: 1, E: 4, B: 4, K: 2}, N: 4, NvS: 8}
	wantRequests := []uint{16, 32, 16, 64, 16, 96, 32, 16}
	for failAt := range 8 {
		for _, errorPointer := range []bool{false, true} {
			for _, failCleanup := range []bool{false, true} {
				name := fmt.Sprintf("slot%d/error-pointer=%t/cleanup-failure=%t", failAt, errorPointer, failCleanup)
				t.Run(name, func(t *testing.T) {
					allocErr := errors.New("allocation failed")
					firstErr, lastErr := errors.New("first free failed"), errors.New("last free failed")
					var requests []uint
					var allocated, freed []driver.DevicePtr
					failures := make(map[driver.DevicePtr]error)
					p, err := allocatePlan(l, func(bytes uint) (driver.DevicePtr, error) {
						i := len(requests)
						requests = append(requests, bytes)
						ptr := driver.DevicePtr((i + 1) * 0x1000)
						if i == failAt && !errorPointer {
							ptr = 0
						}
						if ptr != 0 {
							allocated = append(allocated, ptr)
						}
						if i == failAt {
							if failCleanup && len(allocated) > 0 {
								failures[allocated[0]] = firstErr
								if len(allocated) > 1 {
									failures[allocated[len(allocated)-1]] = lastErr
								}
							}
							return ptr, allocErr
						}
						return ptr, nil
					}, func(ptr driver.DevicePtr) error {
						freed = append(freed, ptr)
						return failures[ptr]
					})
					if !slices.Equal(requests, wantRequests[:failAt+1]) {
						t.Fatalf("allocation requests = %v, want %v", requests, wantRequests[:failAt+1])
					}
					if !slices.Equal(freed, allocated) {
						t.Fatalf("cleanup calls = %x, want every returned pointer once %x", freed, allocated)
					}
					if !errors.Is(err, allocErr) {
						t.Fatalf("allocation cause lost: %v", err)
					}
					for ptr, failure := range failures {
						if !errors.Is(err, failure) {
							t.Errorf("cleanup cause for %#x lost: %v", ptr, err)
						}
					}
					if len(failures) == 0 {
						if p != nil {
							t.Fatal("successful cleanup changed the nil-result allocation-error contract")
						}
						return
					}
					if p == nil {
						t.Fatal("failed constructor cleanup abandoned the plan owner")
					}
					if p.N != l.N || p.R != l.R || p.E != l.E || p.B != l.B || p.NvS != l.NvS || p.K != l.K {
						t.Fatal("partial plan changed layout identity")
					}
					var remaining []driver.DevicePtr
					for i, ptr := range planCleanupPointers(p) {
						original := driver.DevicePtr((i + 1) * 0x1000)
						if failures[original] != nil {
							if ptr != original {
								t.Errorf("failed slot %d = %#x, want retained %#x", i, ptr, original)
							}
							remaining = append(remaining, original)
						} else if ptr != 0 {
							t.Errorf("successful or unallocated slot %d still owns %#x", i, ptr)
						}
					}
					planCleanupRefusesWork(t, p)
					freed = nil
					if err := p.close(func(ptr driver.DevicePtr) error { freed = append(freed, ptr); return nil }); err != nil {
						t.Fatal(err)
					}
					if !slices.Equal(freed, remaining) {
						t.Errorf("retry = %x, want only failed owned slots %x", freed, remaining)
					}
					if err := p.close(func(ptr driver.DevicePtr) error {
						t.Errorf("constructor retry freed %#x twice", ptr)
						return nil
					}); err != nil {
						t.Fatal(err)
					}
					planCleanupRefusesWork(t, p)
				})
			}
		}
	}
}

func TestPlanAllocationSuccessOwnership(t *testing.T) {
	planCleanupNoDriver(t)
	l := &Layout{Config: Config{R: 1, E: 4, B: 4, K: 2}, N: 4, NvS: 8}
	var requests []uint
	var allocated []driver.DevicePtr
	p, err := allocatePlan(l, func(bytes uint) (driver.DevicePtr, error) {
		requests = append(requests, bytes)
		ptr := driver.DevicePtr(len(requests) * 0x1000)
		allocated = append(allocated, ptr)
		return ptr, nil
	}, func(ptr driver.DevicePtr) error {
		t.Errorf("successful construction freed %#x", ptr)
		return nil
	})
	if err != nil || p == nil || p.closed || !slices.Equal(planCleanupPointers(p), allocated) || !slices.Equal(requests, []uint{16, 32, 16, 64, 16, 96, 32, 16}) {
		t.Fatalf("successful allocation changed ownership: plan=%+v error=%v requests=%v", p, err, requests)
	}
	release, err := dedupAcquire(p, dedupSnapshot(p))
	if err != nil {
		t.Fatal(err)
	}
	release()
	if err := p.close(func(driver.DevicePtr) error { return nil }); err != nil {
		t.Fatal(err)
	}
}

func planAllocationFunction(t *testing.T, path, name string) *ast.FuncDecl {
	t.Helper()
	f, err := parser.ParseFile(token.NewFileSet(), path, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	for _, d := range f.Decls {
		if d, ok := d.(*ast.FuncDecl); ok && d.Name.Name == name {
			return d
		}
	}
	t.Fatalf("missing %s in %s", name, path)
	return nil
}

func planAllocationText(t *testing.T, n ast.Node) string {
	t.Helper()
	var b bytes.Buffer
	if err := format.Node(&b, token.NewFileSet(), n); err != nil {
		t.Fatal(err)
	}
	return b.String()
}

func TestPlanAllocationPublicDelegation(t *testing.T) {
	planCleanupNoDriver(t)
	f := planAllocationFunction(t, "plan.go", "AllocatePlan")
	if got := planAllocationText(t, f.Body); got != "{\n\treturn allocatePlan(l, driver.MemAllocBytes, driver.DevicePtr.Free)\n}" {
		t.Fatalf("public constructor no longer uses the host-controlled production transition: %s", got)
	}
	f = planAllocationFunction(t, "plan.go", "Close")
	if got := planAllocationText(t, f.Body); got != "{\n\treturn p.close(driver.DevicePtr.Free)\n}" {
		t.Fatalf("public Close no longer uses the host-controlled production transition: %s", got)
	}
}

func TestPlanAllocationCallerCustody(t *testing.T) {
	planCleanupNoDriver(t)
	for _, tt := range []struct {
		path, name, callee, errorName string
		cleanup                       string
	}{
		{"dedup_native_test.go", "dedupPopulatePlan", "devPlan", "e", "t.Cleanup"},
		{"gpu_test.go", "bufferChildMain", "AllocatePlan", "err", "defer"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			f := planAllocationFunction(t, tt.path, tt.name)
			found := false
			for i, stmt := range f.Body.List {
				assign, ok := stmt.(*ast.AssignStmt)
				if !ok || len(assign.Rhs) != 1 {
					continue
				}
				call, ok := assign.Rhs[0].(*ast.CallExpr)
				if !ok || planAllocationText(t, call.Fun) != tt.callee {
					continue
				}
				found = true
				if i+2 >= len(f.Body.List) {
					t.Fatal("missing constructor owner and error handling")
				}
				guard, ok := f.Body.List[i+1].(*ast.IfStmt)
				if !ok || planAllocationText(t, guard.Cond) != "p != nil" {
					t.Fatal("returned plan owner was not retained before the error check")
				}
				var closes, registrations int
				ast.Inspect(guard.Body, func(n ast.Node) bool {
					if s, ok := n.(*ast.SelectorExpr); ok && planAllocationText(t, s) == "p.Close" {
						closes++
					}
					if tt.cleanup == "defer" {
						if _, ok := n.(*ast.DeferStmt); ok {
							registrations++
						}
					} else if c, ok := n.(*ast.CallExpr); ok && planAllocationText(t, c.Fun) == tt.cleanup {
						registrations++
					}
					return true
				})
				if closes != 1 || registrations != 1 {
					t.Errorf("owner guard has %d Close references and %d cleanup registrations", closes, registrations)
				}
				check, ok := f.Body.List[i+2].(*ast.IfStmt)
				if !ok || planAllocationText(t, check.Cond) != tt.errorName+" != nil" {
					t.Fatal("expected the constructor error check after owner registration")
				}
			}
			if !found {
				t.Fatalf("actual %s call not found", tt.callee)
			}
		})
	}
}
