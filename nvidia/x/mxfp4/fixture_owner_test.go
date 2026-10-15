package mxfp4

import (
	"errors"
	"go/ast"
	"go/parser"
	"go/token"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

const mxfp4NativeOptIn = "MXFP4_NATIVE"

var errMXFP4NativeOptInRequired = errors.New("mxfp4: native fixture opt-in required")

func nativeFixtureAdmission(optIn string, inherited func() error) error {
	if optIn != "1" {
		return errMXFP4NativeOptInRequired
	}
	if inherited == nil {
		return errors.New("mxfp4: native fixture requires inherited CUDA lease check")
	}
	return inherited()
}

func requireMXFP4Native(tb testing.TB) {
	tb.Helper()
	err := nativeFixtureAdmission(os.Getenv(mxfp4NativeOptIn), func() error {
		_, err := cudalock.Inherited(os.Getenv)
		return err
	})
	if errors.Is(err, errMXFP4NativeOptInRequired) {
		tb.Skip("set MXFP4_NATIVE=1 under an inherited canonical GPU lease")
	}
	if err != nil {
		tb.Fatalf("MXFP4 native fixture admission: %v", err)
	}
}

type fixtureOwnerSet struct {
	synchronize func() error
	engine      func() error
	errorOwner  func() error
	stream      func() error
	allocations []func() error
	device      func() error
}

func (o *fixtureOwnerSet) setEngine(engine *Engine, err error) {
	if engine != nil {
		o.engine = engine.Close
		return
	}
	var owner io.Closer
	if errors.As(err, &owner) {
		o.errorOwner = owner.Close
	}
}

func (o *fixtureOwnerSet) setStream(stream *cuda.Stream) {
	if stream != nil {
		o.stream = stream.Close
	}
}

func (o *fixtureOwnerSet) setDevice(dev *cuda.Device) {
	if dev != nil {
		o.device = dev.Close
	}
}

func (o *fixtureOwnerSet) setSynchronization(syncFn func() error) {
	o.synchronize = syncFn
}

func (o *fixtureOwnerSet) addAllocation(free func() error) {
	if free != nil {
		o.allocations = append(o.allocations, free)
	}
}

func retainFixtureAllocation[T any](o *fixtureOwnerSet, allocation cudamem.Slice[T]) {
	o.addAllocation(allocation.Free)
}

func (o *fixtureOwnerSet) close() error {
	if o.synchronize != nil {
		if err := o.synchronize(); err != nil {
			return err
		}
		o.synchronize = nil
	}
	if o.engine != nil {
		if err := o.engine(); err != nil {
			return err
		}
		o.engine = nil
	}
	if o.errorOwner != nil {
		if err := o.errorOwner(); err != nil {
			return err
		}
		o.errorOwner = nil
	}
	if o.stream != nil {
		if err := o.stream(); err != nil {
			return err
		}
		o.stream = nil
	}
	for len(o.allocations) > 0 {
		last := len(o.allocations) - 1
		if err := o.allocations[last](); err != nil {
			return err
		}
		o.allocations = o.allocations[:last]
	}
	if o.device != nil {
		if err := o.device(); err != nil {
			return err
		}
		o.device = nil
	}
	return nil
}

var failedFixtureOwners struct {
	sync.Mutex
	owners []*fixtureOwnerSet
}

func closeFixtureOwners(o *fixtureOwnerSet) error {
	if err := o.close(); err != nil {
		failedFixtureOwners.Lock()
		failedFixtureOwners.owners = append(failedFixtureOwners.owners, o)
		failedFixtureOwners.Unlock()
		return err
	}
	return nil
}

func registerFixtureCleanup(tb testing.TB, o *fixtureOwnerSet) {
	tb.Cleanup(func() {
		if err := closeFixtureOwners(o); err != nil {
			tb.Errorf("close MXFP4 fixture owners: %v", err)
		}
	})
}

func TestNativeFixtureAdmission(t *testing.T) {
	var calls int
	inheritedErr := errors.New("no inherited lease")
	if err := nativeFixtureAdmission("", func() error { calls++; return nil }); !errors.Is(err, errMXFP4NativeOptInRequired) || calls != 0 {
		t.Fatalf("without opt-in: error=%v inherited calls=%d", err, calls)
	}
	if err := nativeFixtureAdmission("1", func() error { calls++; return inheritedErr }); !errors.Is(err, inheritedErr) || calls != 1 {
		t.Fatalf("with failed lease: error=%v inherited calls=%d", err, calls)
	}
	if err := nativeFixtureAdmission("1", nil); err == nil {
		t.Fatal("opt-in without inherited lease checker was accepted")
	}
}

func TestFixtureOwnerCloseOrderAndRetry(t *testing.T) {
	var order []string
	syncCalls := 0
	engineErr := errors.New("engine close failed")
	engineFails := true
	owners := &fixtureOwnerSet{
		synchronize: func() error { syncCalls++; order = append(order, "synchronize"); return nil },
		engine: func() error {
			order = append(order, "engine")
			if engineFails {
				return engineErr
			}
			return nil
		},
		stream: func() error { order = append(order, "stream"); return nil },
		allocations: []func() error{
			func() error { order = append(order, "allocation-1"); return nil },
			func() error { order = append(order, "allocation-2"); return nil },
		},
		device: func() error { order = append(order, "device"); return nil },
	}

	if err := owners.close(); !errors.Is(err, engineErr) {
		t.Fatalf("first close error = %v, want %v", err, engineErr)
	}
	wantFirst := []string{"synchronize", "engine"}
	if strings.Join(order, ",") != strings.Join(wantFirst, ",") {
		t.Fatalf("first close order = %v, want %v", order, wantFirst)
	}
	if owners.engine == nil || owners.stream == nil || len(owners.allocations) != 2 || owners.device == nil {
		t.Fatal("failed engine close lost dependent owner custody")
	}

	engineFails = false
	order = nil
	if err := owners.close(); err != nil {
		t.Fatal(err)
	}
	want := []string{"engine", "stream", "allocation-2", "allocation-1", "device"}
	if strings.Join(order, ",") != strings.Join(want, ",") {
		t.Fatalf("retry close order = %v, want %v", order, want)
	}
	if syncCalls != 1 {
		t.Fatalf("completion callback ran %d times after succeeding", syncCalls)
	}
	if owners.engine != nil || owners.stream != nil || len(owners.allocations) != 0 || owners.device != nil {
		t.Fatal("successful close retained fixture owners")
	}
}

func TestFixtureOwnerCompletionSurvivesStreamRetirement(t *testing.T) {
	restoreErr := errors.New("context restoration after stream destruction")
	destroyedErr := errors.New("stream is destroyed")
	destroyed := false
	syncCalls, closeCalls, frees, devices := 0, 0, 0, 0
	owners := &fixtureOwnerSet{
		synchronize: func() error {
			syncCalls++
			if destroyed {
				return destroyedErr
			}
			return nil
		},
		stream: func() error {
			closeCalls++
			if !destroyed {
				destroyed = true
				return restoreErr
			}
			return nil
		},
		allocations: []func() error{func() error { frees++; return nil }},
		device:      func() error { devices++; return nil },
	}
	if err := owners.close(); !errors.Is(err, restoreErr) {
		t.Fatal(err)
	}
	if !destroyed || closeCalls != 1 || frees != 0 || devices != 0 || owners.stream == nil || len(owners.allocations) != 1 || owners.device == nil {
		t.Fatal("failed cleanup lost remaining ownership", destroyed, closeCalls, frees, devices)
	}
	if err := owners.close(); err != nil {
		t.Fatalf("completion had succeeded before stream retirement: %v (sync=%d close=%d free=%d device=%d)", err, syncCalls, closeCalls, frees, devices)
	}
	if syncCalls != 1 || closeCalls != 2 || frees != 1 || devices != 1 {
		t.Fatal("retry did not finish remaining cleanup", syncCalls, closeCalls, frees, devices)
	}
}

type fixtureClosableError struct {
	cause   error
	closeFn func() error
}

func (e *fixtureClosableError) Error() string { return e.cause.Error() }
func (e *fixtureClosableError) Unwrap() error { return e.cause }
func (e *fixtureClosableError) Close() error  { return e.closeFn() }

func TestFixtureOwnerRetainsClosableNewEngineError(t *testing.T) {
	primaryErr := errors.New("compile failed")
	cleanupErr := errors.New("compiler cleanup failed")
	closeFails := true
	closeCalls, deviceCloses := 0, 0
	owner := &fixtureClosableError{cause: errors.Join(primaryErr, cleanupErr), closeFn: func() error {
		closeCalls++
		if closeFails {
			return cleanupErr
		}
		return nil
	}}
	joinedErr := errors.Join(primaryErr, owner)
	var found io.Closer
	if !errors.Is(joinedErr, primaryErr) || !errors.As(joinedErr, &found) || found != owner {
		t.Fatal("test error did not preserve both compile and cleanup ownership")
	}
	owners := &fixtureOwnerSet{device: func() error { deviceCloses++; return nil }}
	owners.setEngine(nil, joinedErr)
	if owners.errorOwner == nil {
		t.Fatal("nil Engine result dropped its closable error owner")
	}
	if err := owners.close(); !errors.Is(err, cleanupErr) {
		t.Fatalf("first cleanup error = %v, want %v", err, cleanupErr)
	}
	if closeCalls != 1 || deviceCloses != 0 || owners.errorOwner == nil || owners.device == nil {
		t.Fatal("failed error-owner cleanup released dependent device", closeCalls, deviceCloses)
	}
	closeFails = false
	if err := owners.close(); err != nil {
		t.Fatal(err)
	}
	if closeCalls != 2 || deviceCloses != 1 || owners.errorOwner != nil || owners.device != nil {
		t.Fatal("error-owner retry did not finish cleanup", closeCalls, deviceCloses)
	}
}

func TestFixtureOwnerCloseStopsAfterSynchronizationFailure(t *testing.T) {
	syncErr := errors.New("completion failed")
	var order []string
	owners := &fixtureOwnerSet{
		synchronize: func() error { order = append(order, "synchronize"); return syncErr },
		engine:      func() error { order = append(order, "engine"); return nil },
		device:      func() error { order = append(order, "device"); return nil },
	}
	if err := closeFixtureOwners(owners); !errors.Is(err, syncErr) {
		t.Fatalf("closeFixtureOwners() = %v, want %v", err, syncErr)
	}
	if strings.Join(order, ",") != "synchronize" || owners.synchronize == nil || owners.engine == nil || owners.device == nil {
		t.Fatalf("order=%v synchronization retained=%t engine retained=%t device retained=%t", order, owners.synchronize != nil, owners.engine != nil, owners.device != nil)
	}

	failedFixtureOwners.Lock()
	defer failedFixtureOwners.Unlock()
	if len(failedFixtureOwners.owners) == 0 || failedFixtureOwners.owners[len(failedFixtureOwners.owners)-1] != owners {
		t.Fatal("failed cleanup did not retain terminal owner custody")
	}
	failedFixtureOwners.owners = failedFixtureOwners.owners[:len(failedFixtureOwners.owners)-1]
}

func TestFixtureOwnerCloseStopsAfterStreamFailure(t *testing.T) {
	streamErr := errors.New("stream close failed")
	var order []string
	owners := &fixtureOwnerSet{
		synchronize: func() error { order = append(order, "synchronize"); return nil },
		engine:      func() error { order = append(order, "engine"); return nil },
		stream:      func() error { order = append(order, "stream"); return streamErr },
		allocations: []func() error{func() error { order = append(order, "allocation"); return nil }},
		device:      func() error { order = append(order, "device"); return nil },
	}
	if err := owners.close(); !errors.Is(err, streamErr) {
		t.Fatalf("close() = %v, want %v", err, streamErr)
	}
	if strings.Join(order, ",") != "synchronize,engine,stream" {
		t.Fatalf("close order = %v", order)
	}
	if owners.stream == nil || len(owners.allocations) != 1 || owners.device == nil {
		t.Fatal("stream close failure lost dependent owner custody")
	}
}

func TestNativeFixtureSourceBindsInheritedCheck(t *testing.T) {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	source, err := os.ReadFile(file)
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := parser.ParseFile(token.NewFileSet(), "fixture_owner_test.go", source, 0)
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, decl := range parsed.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok || fn.Name.Name != "requireMXFP4Native" {
			continue
		}
		found = true
		if !callsInOrder(fn, "nativeFixtureAdmission", "cudalock.Inherited") {
			t.Fatal("native fixture gate does not verify the inherited cudalock")
		}
	}
	if !found {
		t.Fatal("requireMXFP4Native is missing")
	}
}

func TestNativeFixtureSourceBindsClosableEngineError(t *testing.T) {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	source, err := os.ReadFile(file)
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := parser.ParseFile(token.NewFileSet(), "fixture_owner_test.go", source, 0)
	if err != nil {
		t.Fatal(err)
	}
	for _, decl := range parsed.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok || fn.Name.Name != "setEngine" {
			continue
		}
		if !callsInOrder(fn, "errors.As") || !strings.Contains(string(source), "o.errorOwner = owner.Close") {
			t.Fatal("setEngine does not retain an error-only io.Closer")
		}
		return
	}
	t.Fatal("fixtureOwnerSet.setEngine is missing")
}

func TestNativeFixtureSourceBindsAdmissionAndOwners(t *testing.T) {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	source, err := os.ReadFile(filepath.Join(filepath.Dir(file), "mxfp4_test.go"))
	if err != nil {
		t.Fatal(err)
	}
	fset := token.NewFileSet()
	parsed, err := parser.ParseFile(fset, "mxfp4_test.go", source, 0)
	if err != nil {
		t.Fatal(err)
	}
	functions := map[string]*ast.FuncDecl{}
	for _, decl := range parsed.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if ok && fn.Body != nil {
			functions[fn.Name.Name] = fn
		}
	}
	for _, name := range []string{"TestCUDAMXFP4LinearGEMV", "BenchmarkMXFP4GEMV_Qwen38_Layer"} {
		fn := functions[name]
		if fn == nil || len(fn.Body.List) == 0 {
			t.Fatalf("missing fixture %s", name)
		}
		firstCall := firstCallName(fn.Body.List[0])
		if firstCall != "requireMXFP4Native" {
			t.Fatalf("%s first call = %q, want native admission before CUDA", name, firstCall)
		}
		if !callsInOrder(fn, "requireMXFP4Native", "driver.Init", "cuda.NewDevice", "owners.setDevice") {
			t.Fatalf("%s does not gate, acquire, and retain its device in order", name)
		}
		if !callBeforeNextErrorCheck(fn, "NewEngine", "owners.setEngine") {
			t.Fatalf("%s does not retain a returned engine before error handling", name)
		}
		if !callHasArguments(fn, "owners.setEngine", "engine", "err") {
			t.Fatalf("%s does not pass the error to engine owner retention", name)
		}
		if !callBeforeNextErrorCheck(fn, "cuda.NewDevice", "owners.setDevice") {
			t.Fatalf("%s does not retain a returned device before error handling", name)
		}
		if name == "TestCUDAMXFP4LinearGEMV" && !callsBeforeEveryErrorCheck(fn, "cudamem.Alloc", "retainFixtureAllocation") {
			t.Fatal("test fixture does not retain every allocated buffer before error handling")
		}
	}
	benchmark := functions["BenchmarkMXFP4GEMV_Qwen38_Layer"]
	if !callsBeforeEveryErrorCheck(benchmark, "cudamem.Alloc", "retainFixtureAllocation") {
		t.Fatal("benchmark does not retain every allocated buffer before error handling")
	}
	if !callBeforeNextErrorCheck(benchmark, "dev.NewStream", "owners.setStream") {
		t.Fatal("benchmark does not retain its stream before error handling")
	}
	if !hasCallAfter(benchmark, "engine.LinearGEMVOn", "stream.Synchronize") {
		t.Fatal("benchmark does not synchronize the stream used for GEMV")
	}
}

func firstCallName(stmt ast.Stmt) string {
	var name string
	ast.Inspect(stmt, func(node ast.Node) bool {
		if name != "" {
			return false
		}
		call, ok := node.(*ast.CallExpr)
		if ok {
			name = callName(call.Fun)
		}
		return name == ""
	})
	return name
}

func callsInOrder(fn *ast.FuncDecl, names ...string) bool {
	positions := make(map[string]token.Pos)
	ast.Inspect(fn.Body, func(node ast.Node) bool {
		call, ok := node.(*ast.CallExpr)
		if !ok {
			return true
		}
		name := callName(call.Fun)
		if positions[name] == token.NoPos {
			positions[name] = call.Pos()
		}
		return true
	})
	var previous token.Pos
	for _, name := range names {
		position := positions[name]
		if position == token.NoPos || (previous != token.NoPos && position <= previous) {
			return false
		}
		previous = position
	}
	return true
}

func callBeforeNextErrorCheck(fn *ast.FuncDecl, callName, retainedName string) bool {
	calls := callPositions(fn, callName)
	retainedPositions := callPositions(fn, retainedName)
	errorPositions := errorCheckPositions(fn)
	for _, callPos := range calls {
		var retainedPos, errorPos token.Pos
		for _, position := range retainedPositions {
			if position > callPos && (retainedPos == token.NoPos || position < retainedPos) {
				retainedPos = position
			}
		}
		for _, position := range errorPositions {
			if position > callPos && (errorPos == token.NoPos || position < errorPos) {
				errorPos = position
			}
		}
		if retainedPos == token.NoPos || errorPos == token.NoPos || retainedPos >= errorPos {
			return false
		}
	}
	return len(calls) > 0
}

func callsBeforeEveryErrorCheck(fn *ast.FuncDecl, callName, retainedName string) bool {
	return callBeforeNextErrorCheck(fn, callName, retainedName)
}

func callPositions(fn *ast.FuncDecl, name string) []token.Pos {
	var positions []token.Pos
	ast.Inspect(fn.Body, func(node ast.Node) bool {
		call, ok := node.(*ast.CallExpr)
		if ok && callName(call.Fun) == name {
			positions = append(positions, call.Pos())
		}
		return true
	})
	return positions
}

func callHasArguments(fn *ast.FuncDecl, name string, arguments ...string) bool {
	found := false
	ast.Inspect(fn.Body, func(node ast.Node) bool {
		call, ok := node.(*ast.CallExpr)
		if !ok || callName(call.Fun) != name || len(call.Args) != len(arguments) {
			return true
		}
		for i, argument := range call.Args {
			ident, ok := argument.(*ast.Ident)
			if !ok || ident.Name != arguments[i] {
				return true
			}
		}
		found = true
		return false
	})
	return found
}

func hasCallAfter(fn *ast.FuncDecl, first, second string) bool {
	firstPositions := callPositions(fn, first)
	secondPositions := callPositions(fn, second)
	for _, firstPos := range firstPositions {
		for _, secondPos := range secondPositions {
			if secondPos > firstPos {
				return true
			}
		}
	}
	return false
}

func errorCheckPositions(fn *ast.FuncDecl) []token.Pos {
	var positions []token.Pos
	ast.Inspect(fn.Body, func(node ast.Node) bool {
		stmt, ok := node.(*ast.IfStmt)
		if !ok {
			return true
		}
		cond, ok := stmt.Cond.(*ast.BinaryExpr)
		if !ok || cond.Op != token.NEQ {
			return true
		}
		left, leftOK := cond.X.(*ast.Ident)
		right, rightOK := cond.Y.(*ast.Ident)
		if leftOK && rightOK && ((left.Name == "err" && right.Name == "nil") || (left.Name == "nil" && right.Name == "err")) {
			positions = append(positions, stmt.Pos())
		}
		return true
	})
	return positions
}

func callName(expr ast.Expr) string {
	switch expr := expr.(type) {
	case *ast.Ident:
		return expr.Name
	case *ast.SelectorExpr:
		return callName(expr.X) + "." + expr.Sel.Name
	case *ast.IndexExpr:
		return callName(expr.X)
	case *ast.IndexListExpr:
		return callName(expr.X)
	default:
		return ""
	}
}
