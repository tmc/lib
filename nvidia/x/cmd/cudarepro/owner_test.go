package main

import (
	"errors"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"io"
	"path/filepath"
	"runtime"
	"slices"
	"testing"
)

type testOwner struct {
	calls *[]string
	err   error
}

func (o *testOwner) Error() string { return "owner pending" }
func (o *testOwner) Close() error {
	*o.calls = append(*o.calls, "owner")
	return o.err
}

func TestReleaseOwners(t *testing.T) {
	ownerErr, deviceErr, other := errors.New("owner close"), errors.New("device close"), errors.New("other")
	for _, tt := range []struct {
		name      string
		wrap      func(owner error) error // nil: cause carries no owner
		ownerErr  error
		deviceErr error
		calls     []string
		want      []error
		owned     bool
	}{
		{name: "no owner", calls: []string{"device"}},
		{name: "no owner device failure", deviceErr: deviceErr, calls: []string{"device"}, want: []error{deviceErr}},
		{name: "owner", wrap: func(o error) error { return o }, calls: []string{"owner", "device"}},
		{name: "wrapped owner", wrap: func(o error) error { return fmt.Errorf("replay: %w", o) }, calls: []string{"owner", "device"}},
		{name: "nested owner", wrap: func(o error) error { return fmt.Errorf("replay: %w", errors.Join(other, fmt.Errorf("stage: %w", o))) }, calls: []string{"owner", "device"}, want: []error{other}},
		{name: "owner failure keeps device", wrap: func(o error) error { return fmt.Errorf("replay: %w", o) }, ownerErr: ownerErr, calls: []string{"owner"}, want: []error{ownerErr}, owned: true},
		{name: "nested owner failure keeps device", wrap: func(o error) error { return errors.Join(other, o) }, ownerErr: ownerErr, calls: []string{"owner"}, want: []error{other, ownerErr}, owned: true},
		{name: "owner then device failure", wrap: func(o error) error { return o }, deviceErr: deviceErr, calls: []string{"owner", "device"}, want: []error{deviceErr}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var calls []string
			primary := errors.New("primary")
			owner := &testOwner{calls: &calls, err: tt.ownerErr}
			cause := primary
			if tt.wrap != nil {
				cause = tt.wrap(errors.Join(primary, owner))
			}
			err := releaseOwners(cause, func() error { calls = append(calls, "device"); return tt.deviceErr })
			if !slices.Equal(calls, tt.calls) {
				t.Fatalf("calls = %v, want %v", calls, tt.calls)
			}
			for _, want := range append([]error{primary}, tt.want...) {
				if !errors.Is(err, want) {
					t.Errorf("error %v lost %v", err, want)
				}
			}
			var got io.Closer
			if errors.As(err, &got) && tt.wrap == nil {
				t.Errorf("invented owner: %v", err)
			}
			if tt.owned {
				if got != owner {
					t.Fatalf("owner not retained: %v", err)
				}
				// The caller holds the explicit retry; releaseOwners does not loop.
				owner.err = nil
				if err := got.Close(); err != nil {
					t.Fatal(err)
				}
				if !slices.Equal(calls, []string{"owner", "owner"}) {
					t.Fatalf("calls after retry = %v", calls)
				}
			}
			if tt.wrap == nil && tt.deviceErr == nil && err != primary {
				t.Errorf("error = %v, want primary unchanged", err)
			}
		})
	}
}

func TestReleaseOwnersDependents(t *testing.T) {
	ownerErr, moduleErr := errors.New("owner close"), errors.New("module close")
	for _, tt := range []struct {
		name      string
		ownerErr  error
		moduleErr error
		calls     []string
		want      []error
	}{
		{name: "all released", calls: []string{"owner", "stream", "module", "device"}},
		{name: "dependent failure stops later dependents", moduleErr: moduleErr, calls: []string{"owner", "stream", "module"}, want: []error{moduleErr}},
		{name: "owner failure stops every dependent", ownerErr: ownerErr, calls: []string{"owner"}, want: []error{ownerErr}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var calls []string
			primary := errors.New("primary")
			owner := &testOwner{calls: &calls, err: tt.ownerErr}
			release := func(name string, err error) func() error {
				return func() error { calls = append(calls, name); return err }
			}
			err := releaseOwners(fmt.Errorf("replay: %w", errors.Join(primary, owner)),
				release("stream", nil), release("module", tt.moduleErr), release("device", nil))
			if !slices.Equal(calls, tt.calls) {
				t.Fatalf("calls = %v, want %v", calls, tt.calls)
			}
			for _, want := range append([]error{primary}, tt.want...) {
				if !errors.Is(err, want) {
					t.Errorf("error %v lost %v", err, want)
				}
			}
		})
	}
}

// TestReleaseOwnersSourceRoute binds run's device teardown to releaseOwners:
// a failed Replay returns releaseOwners(err, dev.Close), and dev.Close is
// deferred only after that return. It reads source and runs no device code.
func TestReleaseOwnersSourceRoute(t *testing.T) {
	_, filename, _, _ := runtime.Caller(0)
	fs := token.NewFileSet()
	f, err := parser.ParseFile(fs, filepath.Join(filepath.Dir(filename), "main.go"), nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	var run *ast.FuncDecl
	for _, d := range f.Decls {
		if fn, ok := d.(*ast.FuncDecl); ok && fn.Name.Name == "run" && fn.Recv == nil {
			run = fn
		}
	}
	if run == nil {
		t.Fatal("missing run")
	}
	var replays, releases, defers []token.Pos
	ast.Inspect(run.Body, func(n ast.Node) bool {
		switch n := n.(type) {
		case *ast.CallExpr:
			if sel, ok := n.Fun.(*ast.SelectorExpr); ok && sel.Sel.Name == "Replay" {
				replays = append(replays, n.Pos())
			}
		case *ast.ReturnStmt:
			if len(n.Results) == 1 && types.ExprString(n.Results[0]) == "releaseOwners(err, dev.Close)" {
				releases = append(releases, n.Pos())
			}
		case *ast.DeferStmt:
			if types.ExprString(n.Call) == "dev.Close()" {
				defers = append(defers, n.Pos())
			}
		}
		return true
	})
	if len(replays) != 1 || len(releases) != 1 || len(defers) != 1 {
		t.Fatalf("run has %d Replay calls, %d returns of releaseOwners(err, dev.Close), %d defers of dev.Close(); want 1 each", len(replays), len(releases), len(defers))
	}
	if !(replays[0] < releases[0] && releases[0] < defers[0]) {
		t.Errorf("run order: Replay at %v, releaseOwners return at %v, defer dev.Close at %v; want that order",
			fs.Position(replays[0]), fs.Position(releases[0]), fs.Position(defers[0]))
	}
	// The return must be the Replay error branch itself.
	found := false
	for i, stmt := range run.Body.List {
		if stmt.Pos() > replays[0] || replays[0] >= stmt.End() {
			continue
		}
		found = true
		var next *ast.IfStmt
		if i+1 < len(run.Body.List) {
			next, _ = run.Body.List[i+1].(*ast.IfStmt)
		}
		if next == nil || types.ExprString(next.Cond) != "err != nil" || len(next.Body.List) != 1 || next.Body.List[0].Pos() != releases[0] {
			t.Errorf("Replay at %v is not followed by if err != nil { return releaseOwners(err, dev.Close) }", fs.Position(replays[0]))
		}
	}
	if !found {
		t.Errorf("Replay at %v is not a statement of run's body", fs.Position(replays[0]))
	}
}
