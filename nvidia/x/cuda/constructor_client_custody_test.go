package cuda

import (
	"errors"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"sync"
	"testing"
)

// testOwners holds the handles a test acquires, in acquisition order, and
// closes them in reverse order at test cleanup. Each owner is a dependency of
// every owner added after it. A failed or unfinished close stops the sweep,
// so no earlier owner is released under it, and the unresolved chain stays
// reachable for the process lifetime. Only an explicit retry closes a failed
// owner again.
type testOwners struct {
	mu     sync.Mutex
	owners []*testOwner
}

// A testOwner closes one handle at most once successfully.
type testOwner struct {
	set   *testOwners
	name  string
	close func() error
	state testOwnerState // guarded by set.mu
	err   error          // guarded by set.mu
}

type testOwnerState int

const (
	ownerOpen testOwnerState = iota
	ownerClosing
	ownerClosed
	ownerFailed
)

// retainedTestOwners keeps unresolved owner chains reachable until exit.
var retainedTestOwners struct {
	sync.Mutex
	sets []*testOwners
}

// Export for package cuda_test.
var NewTestOwners = newTestOwners

type TestOwner = testOwner

func newTestOwners(tb testing.TB) *testOwners {
	o := new(testOwners)
	tb.Cleanup(func() { o.finish(tb.Errorf) })
	return o
}

// Add records close as the owner of a handle. Callers add a handle returned
// with an error too, before checking that error.
func (o *testOwners) Add(name string, close func() error) *testOwner {
	e := &testOwner{set: o, name: name, close: close}
	o.mu.Lock()
	o.owners = append(o.owners, e)
	o.mu.Unlock()
	return e
}

// Close closes the handle. After success it does nothing. After a failure it
// returns that failure without closing again. While another call is closing,
// it reports the close as unfinished.
func (e *testOwner) Close() error {
	o := e.set
	o.mu.Lock()
	switch e.state {
	case ownerClosed:
		o.mu.Unlock()
		return nil
	case ownerClosing:
		o.mu.Unlock()
		return fmt.Errorf("close %s: still in progress", e.name)
	case ownerFailed:
		err := e.err
		o.mu.Unlock()
		return err
	}
	e.state = ownerClosing
	o.mu.Unlock()

	err := e.close()

	o.mu.Lock()
	defer o.mu.Unlock()
	if err != nil {
		e.state, e.err = ownerFailed, fmt.Errorf("close %s: %w", e.name, err)
		return e.err
	}
	e.state = ownerClosed
	return nil
}

// closeAll closes the owners in reverse order and stops at the first owner
// that does not close.
func (o *testOwners) closeAll() error {
	o.mu.Lock()
	owners := slices.Clone(o.owners)
	o.mu.Unlock()
	for _, e := range slices.Backward(owners) {
		if err := e.Close(); err != nil {
			return err
		}
	}
	return nil
}

// retry clears recorded close failures and sweeps again. Owners that closed
// are not closed again, and no owner is closed before its dependents.
func (o *testOwners) retry() error {
	o.mu.Lock()
	for _, e := range o.owners {
		if e.state == ownerFailed {
			e.state, e.err = ownerOpen, nil
		}
	}
	o.mu.Unlock()
	return o.closeAll()
}

// unresolved returns the names of the owners that have not closed.
func (o *testOwners) unresolved() []string {
	o.mu.Lock()
	defer o.mu.Unlock()
	var names []string
	for _, e := range o.owners {
		if e.state != ownerClosed {
			names = append(names, e.name)
		}
	}
	return names
}

func (o *testOwners) finish(errorf func(format string, args ...any)) {
	err := o.closeAll()
	if err == nil {
		return
	}
	retainedTestOwners.Lock()
	retainedTestOwners.sets = append(retainedTestOwners.sets, o)
	retainedTestOwners.Unlock()
	errorf("cuda test cleanup: %v; retaining %s for the process lifetime", err, strings.Join(o.unresolved(), ", "))
}

// These controls exercise only the owner bookkeeping with injected close
// functions. They do not load a driver or create any CUDA handle.

type ownerLog struct {
	mu    sync.Mutex
	calls []string
}

func (l *ownerLog) closer(name string, err *error) func() error {
	return func() error {
		l.mu.Lock()
		l.calls = append(l.calls, name)
		l.mu.Unlock()
		return *err
	}
}

func (l *ownerLog) get() []string {
	l.mu.Lock()
	defer l.mu.Unlock()
	return slices.Clone(l.calls)
}

func TestTestOwnersCloseOrder(t *testing.T) {
	blocked := errors.New("blocked")
	for _, tt := range []struct {
		name           string
		fail           string
		wantCalls      []string
		wantUnresolved []string
	}{
		{name: "all close", wantCalls: []string{"event", "stream", "device"}},
		{name: "event fails", fail: "event", wantCalls: []string{"event"}, wantUnresolved: []string{"device", "stream", "event"}},
		{name: "stream fails", fail: "stream", wantCalls: []string{"event", "stream"}, wantUnresolved: []string{"device", "stream"}},
		{name: "device fails", fail: "device", wantCalls: []string{"event", "stream", "device"}, wantUnresolved: []string{"device"}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var log ownerLog
			errs := map[string]*error{"device": new(error), "stream": new(error), "event": new(error)}
			if tt.fail != "" {
				*errs[tt.fail] = blocked
			}
			o := new(testOwners)
			for _, name := range []string{"device", "stream", "event"} {
				o.Add(name, log.closer(name, errs[name]))
			}
			err := o.closeAll()
			if tt.fail == "" && err != nil {
				t.Fatalf("closeAll = %v", err)
			}
			if tt.fail != "" && (!errors.Is(err, blocked) || !strings.Contains(err.Error(), tt.fail)) {
				t.Fatalf("closeAll = %v, want %s failure", err, tt.fail)
			}
			if got := log.get(); !slices.Equal(got, tt.wantCalls) {
				t.Fatalf("calls = %v, want %v", got, tt.wantCalls)
			}
			if got := o.unresolved(); !slices.Equal(got, tt.wantUnresolved) {
				t.Fatalf("unresolved = %v, want %v", got, tt.wantUnresolved)
			}
			// A second sweep neither closes a closed owner again nor
			// retries a failed one.
			err2 := o.closeAll()
			if (err2 == nil) != (err == nil) || (err != nil && err2.Error() != err.Error()) {
				t.Fatalf("second closeAll = %v, want %v", err2, err)
			}
			if got := log.get(); !slices.Equal(got, tt.wantCalls) {
				t.Fatalf("second sweep calls = %v, want %v", got, tt.wantCalls)
			}
		})
	}
}

func TestTestOwnersExplicitClose(t *testing.T) {
	var log ownerLog
	blocked := errors.New("blocked")
	var deviceErr, streamErr error
	streamErr = blocked
	o := new(testOwners)
	o.Add("device", log.closer("device", &deviceErr))
	stream := o.Add("stream", log.closer("stream", &streamErr))
	if err := stream.Close(); !errors.Is(err, blocked) {
		t.Fatalf("stream Close = %v, want %v", err, blocked)
	}
	streamErr = nil
	if err := stream.Close(); !errors.Is(err, blocked) {
		t.Fatalf("second stream Close = %v, want recorded failure", err)
	}
	if err := o.closeAll(); !errors.Is(err, blocked) {
		t.Fatalf("closeAll = %v, want recorded failure", err)
	}
	if got, want := log.get(), []string{"stream"}; !slices.Equal(got, want) {
		t.Fatalf("calls = %v, want %v", got, want)
	}
	if err := o.retry(); err != nil {
		t.Fatalf("retry = %v", err)
	}
	if err := o.retry(); err != nil {
		t.Fatalf("second retry = %v", err)
	}
	if err := stream.Close(); err != nil {
		t.Fatalf("Close after retry = %v", err)
	}
	if got, want := log.get(), []string{"stream", "stream", "device"}; !slices.Equal(got, want) {
		t.Fatalf("calls = %v, want %v", got, want)
	}
	if got := o.unresolved(); len(got) != 0 {
		t.Fatalf("unresolved = %v", got)
	}
}

func TestTestOwnersRetryKeepsDependencies(t *testing.T) {
	var log ownerLog
	blocked := errors.New("blocked")
	deviceErr, streamErr, eventErr := new(error), new(error), new(error)
	*streamErr = blocked
	o := new(testOwners)
	o.Add("device", log.closer("device", deviceErr))
	o.Add("stream", log.closer("stream", streamErr))
	o.Add("event", log.closer("event", eventErr))
	if err := o.closeAll(); !errors.Is(err, blocked) {
		t.Fatalf("closeAll = %v", err)
	}
	// The stream still fails, so the retry must not reach the device.
	if err := o.retry(); !errors.Is(err, blocked) {
		t.Fatalf("retry = %v", err)
	}
	if got, want := log.get(), []string{"event", "stream", "stream"}; !slices.Equal(got, want) {
		t.Fatalf("calls = %v, want %v", got, want)
	}
	*streamErr = nil
	if err := o.retry(); err != nil {
		t.Fatalf("second retry = %v", err)
	}
	if got, want := log.get(), []string{"event", "stream", "stream", "stream", "device"}; !slices.Equal(got, want) {
		t.Fatalf("calls = %v, want %v", got, want)
	}
}

func TestTestOwnersCloseInProgress(t *testing.T) {
	var log ownerLog
	var deviceErr error
	entered, release := make(chan struct{}), make(chan struct{})
	o := new(testOwners)
	o.Add("device", log.closer("device", &deviceErr))
	stream := o.Add("stream", func() error {
		close(entered)
		<-release
		return nil
	})
	done := make(chan error, 1)
	go func() { done <- stream.Close() }()
	<-entered
	if err := o.closeAll(); err == nil || !strings.Contains(err.Error(), "in progress") {
		t.Fatalf("closeAll during close = %v, want in progress", err)
	}
	if got := log.get(); len(got) != 0 {
		t.Fatalf("device closed under an unfinished stream close: %v", got)
	}
	close(release)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if err := o.closeAll(); err != nil {
		t.Fatal(err)
	}
	if got, want := log.get(), []string{"device"}; !slices.Equal(got, want) {
		t.Fatalf("calls = %v, want %v", got, want)
	}
}

func TestTestOwnersFinish(t *testing.T) {
	blocked := errors.New("blocked")
	for _, tt := range []struct {
		name       string
		streamErr  error
		wantReport bool
	}{
		{name: "closed"},
		{name: "failed", streamErr: blocked, wantReport: true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var log ownerLog
			var deviceErr error
			o := new(testOwners)
			o.Add("device", log.closer("device", &deviceErr))
			o.Add("stream", log.closer("stream", &tt.streamErr))
			var reports []string
			o.finish(func(format string, args ...any) { reports = append(reports, fmt.Sprintf(format, args...)) })
			retainedTestOwners.Lock()
			retained := slices.Contains(retainedTestOwners.sets, o)
			retainedTestOwners.Unlock()
			if retained != tt.wantReport || (len(reports) == 1) != tt.wantReport || len(reports) > 1 {
				t.Fatalf("retained=%v reports=%q", retained, reports)
			}
			if tt.wantReport && (!strings.Contains(reports[0], blocked.Error()) || !strings.Contains(reports[0], "device, stream")) {
				t.Fatalf("report = %q", reports[0])
			}
			wantCalls := []string{"stream", "device"}
			if tt.wantReport {
				wantCalls = []string{"stream"}
			}
			if got := log.get(); !slices.Equal(got, wantCalls) {
				t.Fatalf("calls = %v, want %v", got, wantCalls)
			}
		})
	}
}

func TestTestOwnersCleanupRegistered(t *testing.T) {
	var log ownerLog
	var noErr error
	t.Run("owner", func(t *testing.T) {
		o := newTestOwners(t)
		o.Add("device", log.closer("device", &noErr))
		o.Add("stream", log.closer("stream", &noErr))
		if got := log.get(); len(got) != 0 {
			t.Fatalf("closed before cleanup: %v", got)
		}
	})
	if got, want := log.get(), []string{"stream", "device"}; !slices.Equal(got, want) {
		t.Fatalf("calls = %v, want %v", got, want)
	}
}

// TestClientConstructorCustodySource checks that each stream and event
// constructor call in this package's client tests and examples is followed
// directly by a nil check that registers the returned handle's Close, before
// the constructor error is examined, and that each test creating streams or
// events registers its device with the owners right after acquiring it.
// These tests take no GPU lease: they must not call cudatest.Require, whose
// own cleanup releases its lease outside the owner chain, or acquire a
// cudalock lease that the chain does not own.
func TestClientConstructorCustodySource(t *testing.T) {
	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("missing source location")
	}
	for _, tt := range []struct {
		file           string
		sites, devices int
	}{
		{"cuda_test.go", 4, 2},
		{"event_done_test.go", 2, 1},
		{"example_test.go", 2, 0},
		{"lifetime_linux_test.go", 8, 5},
	} {
		f, err := parser.ParseFile(token.NewFileSet(), filepath.Join(filepath.Dir(filename), tt.file), nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		sites, devices := 0, 0
		for _, d := range f.Decls {
			fn, ok := d.(*ast.FuncDecl)
			if !ok || fn.Body == nil {
				continue
			}
			n := constructorSites(t, tt.file, fn.Body)
			sites += n
			if n > 0 {
				devices += deviceSites(t, tt.file, fn)
			}
		}
		if sites != tt.sites || devices != tt.devices {
			t.Errorf("%s: found %d constructor and %d device sites, want %d and %d", tt.file, sites, devices, tt.sites, tt.devices)
		}
		for _, imp := range f.Imports {
			if strings.HasSuffix(imp.Path.Value, `/cudalock"`) {
				t.Errorf("%s: imports cudalock outside the owner chain", tt.file)
			}
		}
		ast.Inspect(f, func(n ast.Node) bool {
			if e, ok := n.(ast.Expr); ok && ownerSelector(e, "cudatest", "Require") {
				t.Errorf("%s: calls cudatest.Require, whose lease cleanup is outside the owner chain", tt.file)
			}
			return true
		})
	}
}

var clientConstructors = map[string]bool{"NewStream": true, "NewStreamWithFlags": true, "NewEvent": true, "NewEventWithFlags": true}

// constructorSites checks and counts the stream and event constructor calls
// in body.
func constructorSites(t *testing.T, file string, body *ast.BlockStmt) int {
	sites := 0
	ast.Inspect(body, func(n ast.Node) bool {
		block, ok := n.(*ast.BlockStmt)
		if !ok {
			return true
		}
		for i, stmt := range block.List {
			name, lhs := callAssign(stmt)
			if !clientConstructors[name] {
				continue
			}
			sites++
			if len(lhs) != 2 || lhs[0] == "_" {
				t.Errorf("%s: %s result is not captured", file, name)
				continue
			}
			if i+1 == len(block.List) || !nonNilCheck(block.List[i+1], lhs[0]) {
				t.Errorf("%s: %s %s is not registered before its error is checked", file, name, lhs[0])
			}
		}
		return true
	})
	return sites
}

// deviceSites checks and counts the device acquisitions in fn. Each must be
// followed, after its error check if it returns one, by
// "owners := NewTestOwners(t)" and a call registering dev.Close.
func deviceSites(t *testing.T, file string, fn *ast.FuncDecl) int {
	sites := 0
	list := fn.Body.List
	for i, stmt := range list {
		name, lhs := callAssign(stmt)
		if name != "RequireDevice" && name != "NewDevice" {
			continue
		}
		sites++
		j := i + 1
		if len(lhs) == 2 {
			j++ // the error check
		}
		if j+1 >= len(list) || !ownersInit(list[j]) || !registersClose(list[j+1], lhs[0]) {
			t.Errorf("%s: %s: device %s is not registered with the owners right after acquisition", file, fn.Name.Name, lhs[0])
		}
	}
	return sites
}

// callAssign returns the called function name and the assigned identifiers
// of stmt if it assigns the results of a single call.
func callAssign(stmt ast.Stmt) (name string, lhs []string) {
	assign, ok := stmt.(*ast.AssignStmt)
	if !ok || len(assign.Rhs) != 1 {
		return "", nil
	}
	call, ok := assign.Rhs[0].(*ast.CallExpr)
	if !ok {
		return "", nil
	}
	switch fun := call.Fun.(type) {
	case *ast.SelectorExpr:
		name = fun.Sel.Name
	case *ast.Ident:
		name = fun.Name
	}
	for _, e := range assign.Lhs {
		id, ok := e.(*ast.Ident)
		if !ok {
			return "", nil
		}
		lhs = append(lhs, id.Name)
	}
	return name, lhs
}

// ownersInit reports whether stmt is "owners := [cuda.]{N,n}ewTestOwners(t)".
func ownersInit(stmt ast.Stmt) bool {
	name, lhs := callAssign(stmt)
	return (name == "NewTestOwners" || name == "newTestOwners") && len(lhs) == 1 && lhs[0] == "owners"
}

// registersClose reports whether n contains "owners.Add(_, name.Close)".
func registersClose(n ast.Node, name string) bool {
	found := false
	ast.Inspect(n, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if ok && ownerSelector(call.Fun, "owners", "Add") && len(call.Args) == 2 && ownerSelector(call.Args[1], name, "Close") {
			found = true
		}
		return !found
	})
	return found
}

// deferredClose reports whether n contains a deferred call of name.Close,
// directly or inside a deferred function literal.
func deferredClose(n ast.Node, name string) bool {
	found := false
	ast.Inspect(n, func(n ast.Node) bool {
		d, ok := n.(*ast.DeferStmt)
		if !ok {
			return !found
		}
		ast.Inspect(d.Call, func(n ast.Node) bool {
			if call, ok := n.(*ast.CallExpr); ok && ownerSelector(call.Fun, name, "Close") {
				found = true
			}
			return !found
		})
		return !found
	})
	return found
}

// nonNilCheck reports whether stmt is "if name != nil { ... }" with a body
// that registers name.Close with the owners or defers it.
func nonNilCheck(stmt ast.Stmt, name string) bool {
	s, ok := stmt.(*ast.IfStmt)
	if !ok || s.Init != nil || s.Else != nil || len(s.Body.List) == 0 {
		return false
	}
	cond, ok := s.Cond.(*ast.BinaryExpr)
	if !ok || cond.Op != token.NEQ {
		return false
	}
	x, ok := cond.X.(*ast.Ident)
	if !ok || x.Name != name {
		return false
	}
	if y, ok := cond.Y.(*ast.Ident); !ok || y.Name != "nil" {
		return false
	}
	return registersClose(s.Body, name) || deferredClose(s.Body, name)
}
