package main

import (
	"context"
	"encoding/json"
	"errors"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/nvml"
	"github.com/tmc/lib/nvidia/x/cudalock"
)

func TestActualLockAdmissionBeforeBackend(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "lock")
	h, err := cudalock.Acquire(context.Background(), cudalock.WithDir(dir), cudalock.WithLabel("CPU nvmlrecord admission"))
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := h.Release(); err != nil {
			t.Error(err)
		}
	}()
	t.Setenv("GPU_LOCKDIR", dir)
	t.Setenv("GPU_LOCK_HELD", strconv.Itoa(os.Getpid()))
	dead := exec.Command("/usr/bin/true")
	if err := dead.Start(); err != nil {
		t.Fatal(err)
	}
	deadPID := dead.Process.Pid
	if err := dead.Wait(); err != nil {
		t.Fatal(err)
	}
	deadDir := filepath.Join(t.TempDir(), "lock")
	if err := os.Mkdir(deadDir, 0700); err != nil {
		t.Fatal(err)
	}
	host, _ := os.Hostname()
	host = strings.SplitN(host, ".", 2)[0]
	for _, tt := range []struct {
		name, held, dir, host string
		pid                   int
		want                  bool
	}{
		{"missing", "", dir, "", 0, false},
		{"foreign-pid", strconv.Itoa(deadPID), dir, "", 0, false},
		{"dead", strconv.Itoa(deadPID), deadDir, host, deadPID, false},
		{"foreign-host", strconv.Itoa(os.Getpid()), deadDir, "foreign-host", os.Getpid(), false},
		{"current", strconv.Itoa(os.Getpid()), dir, "", 0, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("GPU_LOCK_HELD", tt.held)
			t.Setenv("GPU_LOCKDIR", tt.dir)
			if tt.pid != 0 {
				o := cudalock.Owner{PID: tt.pid, Host: tt.host, Epoch: time.Now().Unix(), At: time.Now(), Label: "CPU fixture"}
				if err := os.WriteFile(filepath.Join(tt.dir, "owner"), []byte(o.String()), 0600); err != nil {
					t.Fatal(err)
				}
			}
			calls := 0
			out := filepath.Join(t.TempDir(), "receipt.json")
			err := runWith(0, strings.Repeat("a", 64), out, func(_ uint32, _ string, pid int) (receipt, error) {
				calls++
				if pid != os.Getpid() {
					t.Fatal("unbound lock owner")
				}
				return receipt{LockPID: pid}, nil
			})
			if tt.want {
				if err != nil || calls != 1 {
					t.Fatalf("positive %v calls%d", err, calls)
				}
			} else if err == nil || calls != 0 {
				t.Fatalf("backend reached before lease validation: %v calls%d", err, calls)
			}
		})
	}
}

func TestGPMBranchPopulationAndRetirement(t *testing.T) {
	bad := errors.New("CPU prerequisite failed")
	for _, name := range []string{"positive", "unsupported", "support-error", "alloc1", "alloc2", "get1", "get2", "instance", "mig-disabled", "close1", "close2", "partial-owner"} {
		t.Run(name, func(t *testing.T) {
			r := receipt{}
			alloc, get, closed := 0, 0, 0
			q := gpmQueries{
				allocate: func() (nvml.GpmSample, error) {
					alloc++
					if name == "alloc"+strconv.Itoa(alloc) {
						return 0, bad
					}
					if name == "partial-owner" && alloc == 1 {
						return 17, bad
					}
					return nvml.GpmSample(alloc), nil
				},
				get: func(nvml.GpmSample) error {
					get++
					if name == "get"+strconv.Itoa(get) {
						return bad
					}
					return nil
				},
				metrics: func(nvml.GpmSample, nvml.GpmSample) ([]nvml.GpmMetric, error) { return nil, nil },
				instanceID: func() (uint32, error) {
					if name == "instance" {
						return 0, bad
					}
					return 9, nil
				},
				migGet: func(uint32, nvml.GpmSample) error { return nil },
				close: func(s *nvml.GpmSample) error {
					closed++
					if name == "close"+strconv.Itoa(int(*s)) {
						return bad
					}
					*s = 0
					return nil
				},
				pause: func() {},
			}
			var supportErr error
			if name == "support-error" {
				supportErr = bad
			}
			err := collectGPM(&r, name != "unsupported", supportErr, name != "mig-disabled", q)
			if strings.HasPrefix(name, "close") {
				if !errors.Is(err, bad) {
					t.Fatal("lost retirement failure")
				}
			} else if err != nil {
				t.Fatal(err)
			}
			counts := map[string]int{}
			for _, o := range r.Calls {
				counts[o.Method]++
				if o.Method != "GpmSample.Close" && o.Error == "" && o.Unmeasured == "" && o.Value == nil && o.Method != "GpmSample.Get" && o.Method != "GpmMetrics" && o.Method != "GpmMigSampleGet" {
					t.Fatalf("missing disposition %+v", o)
				}
			}
			for method, want := range map[string]int{"NewGpmSample": 2, "GpmSample.Get": 2, "GpmMetrics": 1, "GpuInstanceId": 1, "GpmMigSampleGet": 1} {
				if counts[method] != want {
					t.Fatalf("%s population%d want%d", method, counts[method], want)
				}
			}
			wantClose := 2
			if name == "unsupported" || name == "support-error" || name == "alloc1" {
				wantClose = 0
			}
			if name == "alloc2" || name == "partial-owner" {
				wantClose = 1
			}
			if closed != wantClose || counts["GpmSample.Close"] != wantClose {
				t.Fatalf("owners retired%d want%d", closed, wantClose)
			}
			for _, o := range r.Calls {
				if (name == "alloc1" || name == "alloc2" || name == "partial-owner" || name == "support-error" || name == "unsupported" || name == "get1" || name == "get2") && o.Method == "GpmMetrics" && o.Unmeasured == "" {
					t.Fatal("unmeasured metrics absent")
				}
				if name == "instance" && o.Method == "GpmMigSampleGet" && !strings.Contains(o.Unmeasured, bad.Error()) {
					t.Fatal("lost instance prerequisite error")
				}
			}
		})
	}
}

func TestFailureReceiptAfterCleanup(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "lock")
	h, err := cudalock.Acquire(context.Background(), cudalock.WithDir(dir))
	if err != nil {
		t.Fatal(err)
	}
	defer h.Release()
	t.Setenv("GPU_LOCKDIR", dir)
	t.Setenv("GPU_LOCK_HELD", strconv.Itoa(os.Getpid()))
	a, b := errors.New("sample close failed"), errors.New("shutdown failed")
	out := filepath.Join(t.TempDir(), "failure.json")
	err = runWith(0, strings.Repeat("a", 64), out, func(_ uint32, _ string, _ int) (receipt, error) {
		r := receipt{}
		e1 := cleanupObservation(&r, "GpmSample.Close", 1, func() error { return a })
		e2 := cleanupObservation(&r, "Shutdown", nil, func() error { return b })
		return r, errors.Join(e1, e2)
	})
	if !errors.Is(err, a) || !errors.Is(err, b) {
		t.Fatal("lost cleanup error", err)
	}
	body, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	var r receipt
	if err := json.Unmarshal(body, &r); err != nil {
		t.Fatal(err)
	}
	if r.Complete || len(r.Calls) != 2 || !strings.Contains(r.Failure, a.Error()) || !strings.Contains(r.Failure, b.Error()) {
		t.Fatalf("false success receipt %+v", r)
	}
	if err := writeReceipt(out, receipt{}); err == nil {
		t.Fatal("overwrote retained failure")
	}
}

func TestShutdownRetainedBeforePublicationWiring(t *testing.T) {
	f, err := parser.ParseFile(token.NewFileSet(), "main.go", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	var found bool
	for _, d := range f.Decls {
		fn, ok := d.(*ast.FuncDecl)
		if !ok || fn.Name.Name != "collect" {
			continue
		}
		for _, s := range fn.Body.List {
			deferStmt, ok := s.(*ast.DeferStmt)
			if !ok {
				continue
			}
			lit, ok := deferStmt.Call.Fun.(*ast.FuncLit)
			if !ok || len(lit.Body.List) != 1 {
				continue
			}
			a, ok := lit.Body.List[0].(*ast.AssignStmt)
			if !ok || len(a.Lhs) != 1 || len(a.Rhs) != 1 {
				continue
			}
			left, ok := a.Lhs[0].(*ast.Ident)
			if !ok || left.Name != "err" {
				continue
			}
			join, ok := a.Rhs[0].(*ast.CallExpr)
			if !ok || len(join.Args) != 2 {
				continue
			}
			selector, ok := join.Fun.(*ast.SelectorExpr)
			if !ok || selector.Sel.Name != "Join" {
				continue
			}
			pkg, ok := selector.X.(*ast.Ident)
			if !ok || pkg.Name != "errors" {
				continue
			}
			retained, ok := join.Args[0].(*ast.Ident)
			if !ok || retained.Name != "err" {
				continue
			}
			cleanup, ok := join.Args[1].(*ast.CallExpr)
			if !ok || len(cleanup.Args) != 4 {
				continue
			}
			name, ok := cleanup.Fun.(*ast.Ident)
			if !ok || name.Name != "cleanupObservation" {
				continue
			}
			shutdown, ok := cleanup.Args[3].(*ast.SelectorExpr)
			if !ok || shutdown.Sel.Name != "Shutdown" {
				continue
			}
			pkg, ok = shutdown.X.(*ast.Ident)
			if !ok || pkg.Name != "nvml" {
				continue
			}
			found = true
		}
	}
	if !found {
		t.Fatal("collect does not retain deferred actual NVML shutdown error")
	}
}

func TestEarlyPrerequisitePopulation(t *testing.T) {
	r := receipt{}
	completeUnmeasured(&r, errors.New("UUID prerequisite failed"))
	if len(r.Calls) != 18 {
		t.Fatalf("selected population %d", len(r.Calls))
	}
	for _, o := range r.Calls {
		if !strings.Contains(o.Unmeasured, "UUID prerequisite failed") || o.Value != nil || o.Error != "" {
			t.Fatalf("invented or unexplained result %+v", o)
		}
	}
	completeUnmeasured(&r, errors.New("another failure"))
	if len(r.Calls) != 18 {
		t.Fatal("duplicated existing dispositions")
	}
}
