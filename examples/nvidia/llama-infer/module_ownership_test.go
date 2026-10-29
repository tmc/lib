package main

import (
	"errors"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io"
	"maps"
	"os"
	"path"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
)

var (
	errFakeLookup = errors.New("fake lookup failed")
	errFakeUnload = errors.New("fake unload failed")
)

// fakeModule is a module handle with fake lookup and unload. Lookups fail
// from the failAt'th on (0 means never); the first unloadFails unloads fail.
// A successful unload does not clear the handle, so the caller must.
type fakeModule struct {
	failAt      int
	lookups     int
	names       []string
	unloadFails int
	unloads     int
	handles     []driver.Module // the handle each unload was given
}

func (f *fakeModule) lookup(name string) (driver.Function, error) {
	f.lookups++
	f.names = append(f.names, name)
	if f.failAt > 0 && f.lookups >= f.failAt {
		return 0, errFakeLookup
	}
	return driver.Function(f.lookups), nil
}

func (f *fakeModule) unload(m *driver.Module) error {
	f.unloads++
	f.handles = append(f.handles, *m)
	if f.unloadFails > 0 {
		f.unloadFails--
		return errFakeUnload
	}
	return nil
}

type lookupFunc func(driver.Module, func(string) (driver.Function, error), func(*driver.Module) error) (*GPUModules, error)

// moduleFailures returns how lookup departs from: on success, every kernel
// found and nothing unloaded; on a failed lookup, the module unloaded and
// the lookup error returned; and if that unload fails too, both errors and
// a GPUModules still owning the module, which a later Close releases once.
func moduleFailures(lookup lookupFunc) []string {
	const loaded = driver.Module(42)
	var bad []string

	f := &fakeModule{}
	gm, err := lookup(loaded, f.lookup, f.unload)
	wantNames := []string{
		"EmbeddingLookup",
		"RMSNorm",
		"LinearGEMV",
		"ApplyRoPE",
		"StoreKVCache",
		"SwiGLU",
		"AddResidual",
		"FusedRMSNormQKV",
		"FusedSwiGLUDown",
		"LinearGEMVBF16",
		"WeightConvertProbeBF16",
		"EmbeddingLookupBF16",
		"FusedRMSNormQKVBF16",
		"FusedSwiGLUDownBF16",
		"FusedRMSNormQKVRoPEStore",
		"FusedOutProjResidual",
		"FusedRMSNormGateUpSwiGLU",
		"FusedDownProjResidual",
		"FusedRMSNormQKVRoPEStoreBF16",
		"FusedOutProjResidualBF16",
		"FusedRMSNormGateUpSwiGLUBF16",
		"FusedDownProjResidualBF16",
		"ApplyRoPEScaled",
		"LinearGEMVBF16Wide",
	}
	wantGM := &GPUModules{
		mod:                     loaded,
		embedFn:                 1,
		normFn:                  2,
		gemvFn:                  3,
		ropeFn:                  4,
		storeKVFn:               5,
		swigluFn:                6,
		addResFn:                7,
		fusedQKVFn:              8,
		fusedSwiGLUFn:           9,
		gemvBF16Fn:              10,
		convertBF16Fn:           11,
		embedBF16Fn:             12,
		fusedQKVBF16Fn:          13,
		fusedSwiGLUBF16Fn:       14,
		fusedQKVRoPEStoreFn:     15,
		fusedOutProjResFn:       16,
		fusedGateUpSwiGLUFn:     17,
		fusedDownProjResFn:      18,
		fusedQKVRoPEStoreBF16Fn: 19,
		fusedOutProjResBF16Fn:   20,
		fusedGateUpSwiGLUBF16Fn: 21,
		fusedDownProjResBF16Fn:  22,
		ropeScaledFn:            23,
		gemvBF16WideFn:          24,
	}
	if err != nil || gm == nil || *gm != *wantGM || !slices.Equal(f.names, wantNames) || f.unloads != 0 {
		bad = append(bad, fmt.Sprintf("success: %+v, %v, names=%v, %d unloads", gm, err, f.names, f.unloads))
	}

	f = &fakeModule{failAt: 3}
	gm, err = lookup(loaded, f.lookup, f.unload)
	if gm != nil || !errors.Is(err, errFakeLookup) || errors.Is(err, errFakeUnload) || f.unloads != 1 {
		bad = append(bad, fmt.Sprintf("lookup fails: %v, %v, %d unloads", gm, err, f.unloads))
	}

	f = &fakeModule{failAt: 3, unloadFails: 1}
	gm, err = lookup(loaded, f.lookup, f.unload)
	if !errors.Is(err, errFakeLookup) || !errors.Is(err, errFakeUnload) {
		bad = append(bad, fmt.Sprintf("lookup and unload fail: error %v, want both", err))
	}
	if gm == nil || gm.mod != loaded {
		return append(bad, fmt.Sprintf("lookup and unload fail: %v, want the module kept", gm))
	}
	if err := gm.close(f.unload); err != nil || gm.mod != 0 {
		bad = append(bad, fmt.Sprintf("retry: %v, module %d", err, gm.mod))
	}
	if err := gm.close(f.unload); err != nil || f.unloads != 2 {
		bad = append(bad, fmt.Sprintf("repeat: %v, %d unloads, want 2", err, f.unloads))
	}
	if !slices.Equal(f.handles, []driver.Module{loaded, loaded}) {
		bad = append(bad, fmt.Sprintf("unloaded handles %v, want [%d %d]", f.handles, loaded, loaded))
	}
	return bad
}

func TestLookupModules(t *testing.T) {
	for _, f := range moduleFailures(lookupModules) {
		t.Error(f)
	}
	// Must-fail control: the behaviour this change replaced, which returned
	// nil whether or not the unload succeeded.
	old := func(loaded driver.Module, lookup func(string) (driver.Function, error), unload func(*driver.Module) error) (*GPUModules, error) {
		gm, err := lookupModules(loaded, lookup, unload)
		if err != nil {
			return nil, err
		}
		return gm, nil
	}
	if len(moduleFailures(old)) == 0 {
		t.Error("must-fail control: module dropped with a failed unload, not detected")
	}
}

func TestModulesCloseRetains(t *testing.T) {
	f := &fakeModule{unloadFails: 1}
	gm := &GPUModules{mod: 42}
	if err := gm.close(f.unload); !errors.Is(err, errFakeUnload) || gm.mod != 42 {
		t.Errorf("failed close: %v, module %d; want the error and the module kept", err, gm.mod)
	}
	if err := gm.close(f.unload); err != nil || gm.mod != 0 {
		t.Errorf("retry: %v, module %d", err, gm.mod)
	}
	if err := gm.close(f.unload); err != nil || f.unloads != 2 {
		t.Errorf("repeat: %v, %d unloads, want 2", err, f.unloads)
	}
	if !slices.Equal(f.handles, []driver.Module{42, 42}) {
		t.Errorf("unloaded handles %v, want [42 42]", f.handles)
	}
	var zero GPUModules
	if err := zero.Close(); err != nil {
		t.Errorf("zero modules: %v", err)
	}
}

// libX is the import path prefix of the nvidia/x packages.
const libX = "github.com/tmc/lib/nvidia/x/"

// ownerReturning names the constructors this package calls that can
// return an owner, to release, together with an error: a function of this
// package by its name, and one of another package by its import path, a
// dot and its name. The value reports whether the owner is a pointer, nil
// when nothing is held: Free on a nil *GPUScratchpad or *GPUModelWeights
// and Close on a nil *GPUModules panic, so its release must be guarded by
// "if owner != nil".
var ownerReturning = map[string]bool{
	"uploadModel": true, "uploadModelWith": true, "UploadModelToGPU": true,
	"NewGPUScratchpad": true, "newGPUScratchpadWith": true,
	"CompileGPUModules": true, "compileGPUModules": true,
	"NewDecodeGraph": true, "newDecodeGraphWith": true,
	"newEgraphDecoder":  true,
	libX + "cudakv.New": true, libX + "cudamem.Alloc": false,
}

// calleeName returns the name of the function call calls, as pkg.Name for
// a qualified name, ignoring type arguments.
func calleeName(call *ast.CallExpr) string {
	fun := call.Fun
	if ix, ok := fun.(*ast.IndexExpr); ok {
		fun = ix.X
	}
	switch f := fun.(type) {
	case *ast.Ident:
		return f.Name
	case *ast.SelectorExpr:
		if x, ok := f.X.(*ast.Ident); ok {
			return x.Name + "." + f.Sel.Name
		}
	}
	return ""
}

// fileImports returns the import path of each package f imports, by the
// name f refers to it by, and the paths f imports with a dot, whose names
// f uses unqualified.
func fileImports(f *ast.File) (names map[string]string, dot []string) {
	names = make(map[string]string)
	for _, imp := range f.Imports {
		p := strings.Trim(imp.Path.Value, "`\"")
		switch {
		case imp.Name == nil:
			names[path.Base(p)] = p
		case imp.Name.Name == ".":
			dot = append(dot, p)
		case imp.Name.Name != "_":
			names[imp.Name.Name] = p
		}
	}
	return names, dot
}

// refName returns the name e refers to, ignoring type arguments: an
// identifier's own name, or, for a selector on a package f imports, the
// package's import path, a dot and the selected name; otherwise "". It
// does not see an import name shadowed by a local declaration.
func refName(e ast.Expr, imports map[string]string) string {
	switch ix := e.(type) {
	case *ast.IndexExpr:
		e = ix.X
	case *ast.IndexListExpr:
		e = ix.X
	}
	switch e := e.(type) {
	case *ast.Ident:
		return e.Name
	case *ast.SelectorExpr:
		if x, ok := e.X.(*ast.Ident); ok {
			if p, ok := imports[x.Name]; ok {
				return p + "." + e.Sel.Name
			}
		}
	}
	return ""
}

// An ownerSite is where and how a reference to an ownerReturning
// constructor appears that releasesFirst does not inspect: the file, the
// function declaration enclosing it, the constructor, and its form, one
// of:
//
//   - "return": "return ctor(...)" in the body of a function that is
//     itself an ownerReturning constructor, whose callers are checked;
//   - "allocator for f": "func(...) (...) { return ctor(...) }" passed
//     to f, which then owns what the literal returns;
//   - "owner field": "if x, err = ctor(...); err != nil { return ..., e }"
//     in an ownerReturning constructor, where e is err or wraps it, and x
//     lies in an owner that a defer registered earlier in the
//     constructor's body releases first when err, its named error result,
//     is not nil (see storesIntoReleased);
//   - "value": the constructor used other than as the function of a call;
//   - "call": any other call.
type ownerSite struct{ file, fn, ctor, form string }

// ownerSites lists every reference to an ownerReturning constructor in the
// package that releasesFirst does not inspect, with how many there are at
// that site and why each is safe; test is the test of the code that takes
// over the owner, if one runs on the CPU.
var ownerSites = []struct {
	ownerSite
	n         int
	test, why string
}{
	{ownerSite{"gpu_kernels.go", "CompileGPUModules", "compileGPUModules", "return"}, 1, "",
		"hands the owner and error to the callers of CompileGPUModules, whose calls are checked"},
	{ownerSite{"gpu_model.go", "UploadModelToGPU", "uploadModel", "return"}, 1, "",
		"hands the owner and error to the callers of UploadModelToGPU, whose calls are checked"},
	{ownerSite{"gpu_model.go", "uploadModel", "uploadModelWith", "return"}, 1, "",
		"hands the owner and error to the callers of uploadModel, whose calls are checked"},
	{ownerSite{"gpu_model.go", "uploadModel", libX + "cudamem.Alloc", "allocator for uploadModelWith"}, 1, "TestUploadModelWith",
		"uploadModelWith stores what alloc returns in the weights its deferred free releases on error; " +
			"its fakes return only zero slices, so a nonzero slice returned with an error is not exercised"},
	{ownerSite{"gpu_model.go", "NewGPUScratchpad", "newGPUScratchpadWith", "return"}, 1, "",
		"hands the owner and error to the callers of NewGPUScratchpad, whose calls are checked"},
	{ownerSite{"gpu_model.go", "NewGPUScratchpad", libX + "cudamem.Alloc", "allocator for newGPUScratchpadWith"}, 2, "TestNewGPUScratchpadWith",
		"newGPUScratchpadWith stores what each allocator returns, with its error too, in the scratchpad " +
			"its deferred release frees on error"},
	{ownerSite{"decode_graph.go", "NewDecodeGraph", "newDecodeGraphWith", "return"}, 1, "",
		"hands the owner and error to the callers of NewDecodeGraph, whose calls are checked"},
	{ownerSite{"egraph_gpu.go", "newEgraphDecoder", libX + "cudamem.Alloc", "owner field"}, 25, "",
		"each buffer lies in eg, which the constructor's deferred eg.Close frees on error"},
	{ownerSite{"main.go", "runGPUInference", "NewDecodeGraph", "call"}, 1, "TestCloseDecodeGraph",
		"runGPUInference defers closeDecodeGraph(td, &dg) before calling NewDecodeGraph, which closes a non-nil dg when !td.held() and records its error in td"},
	{ownerSite{"decode_graph_test.go", "build", "newDecodeGraphWith", "call"}, 1, "TestDecodeGraphCloseAndSetupErrors",
		"called with fake graph operations, so nothing is allocated on a device"},
	{ownerSite{"weights_bf16.go", "uploadWeight", libX + "cudamem.Alloc", "allocator for uploadWeightWith"}, 2, "TestUploadWeightWith",
		"uploadWeightWith passes it to uploadAs, which returns a nonzero slice with its allocation error " +
			"(control W1 here, A1 in TestUploadAs)"},
	{ownerSite{"gpu_model_test.go", "TestUploadModelWith", "uploadModelWith", "value"}, 2, "TestUploadModelWith",
		"called with fake device operations, so nothing is allocated on a device"},
	{ownerSite{"gpu_model_test.go", "TestNewGPUScratchpadWith", "newGPUScratchpadWith", "value"}, 2, "TestNewGPUScratchpadWith",
		"called with fake device operations, so nothing is allocated on a device"},
}

// releasesFirst counts the references in src to an ownerReturning
// constructor, in any form, called or not, and checks each "owner, err :=
// ctor(...)" or "owner, err = ctor(...)" statement: owner and err must be
// named, and the next statement must register a release of owner (see
// registersRelease), so that an owner returned with an error is released
// before anything else runs. Every other reference must be listed for name
// in ownerSites, at its site, as often as it appears there.
func releasesFirst(name, src string) (n callCount, bad []string, err error) {
	n, sites, bad, err := ownerRefs(name, src)
	if err != nil {
		return callCount{}, nil, err
	}
	want := make(map[ownerSite]int)
	for _, s := range ownerSites {
		if s.file == name {
			want[s.ownerSite] += s.n
		}
	}
	keys := slices.Concat(slices.Collect(maps.Keys(sites)), slices.Collect(maps.Keys(want)))
	slices.SortFunc(keys, func(a, b ownerSite) int { return strings.Compare(fmt.Sprint(a), fmt.Sprint(b)) })
	for _, s := range slices.Compact(keys) {
		if sites[s] != want[s] {
			bad = append(bad, fmt.Sprintf("%s: %s: %d references to %s in form %q, %d listed in ownerSites", s.file, s.fn, sites[s], s.ctor, s.form, want[s]))
		}
	}
	return n, bad, nil
}

// ownerRefs counts the references in src to an ownerReturning constructor
// and inspects the calls in the form releasesFirst checks, returning the
// problems found with them, and the site of every other reference.
func ownerRefs(name, src string) (n callCount, sites map[ownerSite]int, bad []string, err error) {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, name, src, 0)
	if err != nil {
		return callCount{}, nil, nil, err
	}
	imports, dot := fileImports(f)
	for _, p := range dot {
		for ctor := range ownerReturning {
			if strings.HasPrefix(ctor, p+".") {
				bad = append(bad, fmt.Sprintf("%s: dot import of %s hides %s", name, p, ctor))
			}
		}
	}
	checked := make(map[*ast.CallExpr]bool)
	ast.Inspect(f, func(node ast.Node) bool {
		block, ok := node.(*ast.BlockStmt)
		if !ok {
			return true
		}
		for i, stmt := range block.List {
			as, ok := stmt.(*ast.AssignStmt)
			if !ok || len(as.Lhs) != 2 || len(as.Rhs) != 1 {
				continue
			}
			call, ok := as.Rhs[0].(*ast.CallExpr)
			if !ok {
				continue
			}
			ctor := refName(call.Fun, imports)
			nilable, ok := ownerReturning[ctor]
			if !ok {
				continue
			}
			n.checked++
			checked[call] = true
			pos := fset.Position(as.Pos())
			owner, _ := as.Lhs[0].(*ast.Ident)
			errv, _ := as.Lhs[1].(*ast.Ident)
			if owner == nil || owner.Name == "_" || errv == nil || errv.Name == "_" {
				bad = append(bad, fmt.Sprintf("%v: %s owner or error discarded", pos, ctor))
				continue
			}
			if i+1 >= len(block.List) || !registersRelease(block.List[i+1], owner.Name, nilable) {
				guard := ""
				if nilable {
					guard = fmt.Sprintf(" under \"if %s != nil\"", owner.Name)
				}
				bad = append(bad, fmt.Sprintf("%v: %s owner %s not released%s in the next statement", pos, ctor, owner.Name, guard))
			}
		}
		return true
	})

	sites = make(map[ownerSite]int)
	var stack []ast.Node // the ancestors of the node being visited
	ast.Inspect(f, func(node ast.Node) bool {
		if node == nil {
			stack = stack[:len(stack)-1]
			return true
		}
		var ctor string
		switch x := node.(type) {
		case *ast.SelectorExpr:
			ctor = refName(x, imports)
		case *ast.Ident:
			ctor = x.Name
			switch p := stack[len(stack)-1].(type) {
			case *ast.SelectorExpr:
				if p.Sel == x {
					ctor = "" // a field or method, not this package's function
				}
			case *ast.FuncDecl:
				if p.Name == x {
					ctor = "" // the declaration itself
				}
			}
		}
		if _, ok := ownerReturning[ctor]; !ok {
			stack = append(stack, node)
			return true
		}
		n.calls++
		if site, ok := classifyOwnerRef(name, ctor, node.(ast.Expr), stack, imports, checked); ok {
			sites[site]++
		}
		return false
	})
	return n, sites, bad, nil
}

// classifyOwnerRef returns the site of the reference ref to ctor, whose
// ancestors are stack, or false if releasesFirst checks it.
func classifyOwnerRef(file, ctor string, ref ast.Expr, stack []ast.Node, imports map[string]string, checked map[*ast.CallExpr]bool) (ownerSite, bool) {
	site := ownerSite{file: file, ctor: ctor, form: "value"}
	var fd *ast.FuncDecl
	for _, a := range stack {
		if d, ok := a.(*ast.FuncDecl); ok {
			fd = d
			site.fn = d.Name.Name
		}
	}
	parent := func(i int) ast.Node { // the ith ancestor up, from 1
		if i > len(stack) {
			return nil
		}
		return stack[len(stack)-i]
	}
	var e ast.Expr = ref
	up := 1
	switch ix := parent(up).(type) {
	case *ast.IndexExpr:
		if ix.X == e {
			e, up = ix, up+1
		}
	case *ast.IndexListExpr:
		if ix.X == e {
			e, up = ix, up+1
		}
	}
	call, ok := parent(up).(*ast.CallExpr)
	if !ok || call.Fun != e {
		return site, true
	}
	if checked[call] {
		return ownerSite{}, false
	}
	inOwner := false
	if fd != nil {
		_, inOwner = ownerReturning[fd.Name.Name]
	}
	site.form = "call"
	switch p := parent(up + 1).(type) {
	case *ast.ReturnStmt:
		if len(p.Results) != 1 {
			break
		}
		block, _ := parent(up + 2).(*ast.BlockStmt)
		if inOwner && block == fd.Body {
			site.form = "return"
			break
		}
		lit, _ := parent(up + 3).(*ast.FuncLit)
		outer, _ := parent(up + 4).(*ast.CallExpr)
		if block != nil && len(block.List) == 1 && lit != nil && outer != nil && slices.Contains(outer.Args, ast.Expr(lit)) {
			site.form = "allocator for " + refName(outer.Fun, imports)
		}
	case *ast.AssignStmt:
		is, _ := parent(up + 2).(*ast.IfStmt)
		if inOwner && is != nil && is.Init == p && storesIntoReleased(fd, p, is, stack) {
			site.form = "owner field"
		}
	}
	return site, true
}

// storesIntoReleased reports whether as, the initializer of is, which lies
// in fd, whose ancestors are stack, is "x, err = ctor(...)" with is's
// condition "err != nil" and is's body starting with a return of err
// (see returnsErr), where err is fd's named error result; a defer
// statement earlier in fd's body calls, with no arguments, a function
// literal with no parameters whose body is "if err != nil { ... }" with a
// release of an owner first in its body (see releaseOwner); and x lies in
// that owner (see ownerRoot).
//
// It does not check that err is not shadowed in a block enclosing as;
// the return of err sets the named result whichever err it returns.
func storesIntoReleased(fd *ast.FuncDecl, as *ast.AssignStmt, is *ast.IfStmt, stack []ast.Node) bool {
	if as.Tok != token.ASSIGN || len(as.Lhs) != 2 {
		return false
	}
	errv, ok := as.Lhs[1].(*ast.Ident)
	if !ok || !isNotNil(is.Cond, errv.Name) {
		return false
	}
	res := fd.Type.Results
	if res == nil || len(res.List) == 0 {
		return false
	}
	last := res.List[len(res.List)-1]
	if len(last.Names) == 0 || last.Names[len(last.Names)-1].Name != errv.Name {
		return false
	}
	if !returnsErr(is.Body, errv.Name, res.NumFields()) {
		return false
	}
	// top is the statement of fd's body that contains as.
	var top ast.Node
	for i, a := range stack {
		if a == fd.Body && i+1 < len(stack) {
			top = stack[i+1]
		}
	}
	if top == nil {
		return false
	}
	owner := ""
	for _, stmt := range fd.Body.List {
		if stmt == top {
			break
		}
		d, ok := stmt.(*ast.DeferStmt)
		if !ok {
			continue
		}
		lit, ok := d.Call.Fun.(*ast.FuncLit)
		if !ok || len(d.Call.Args) != 0 || lit.Type.Params.NumFields() != 0 || len(lit.Body.List) != 1 {
			continue
		}
		cond, ok := lit.Body.List[0].(*ast.IfStmt)
		if !ok || cond.Init != nil || cond.Else != nil || !isNotNil(cond.Cond, errv.Name) || len(cond.Body.List) == 0 {
			continue
		}
		if o := releaseOwner(cond.Body.List[0]); o != "" {
			owner = o
		}
	}
	return owner != "" && ownerRoot(as.Lhs[0], stack) == owner
}

// returnsErr reports whether body starts with a return of n results whose
// last is the variable named errName, or a call of fmt.Errorf or
// errors.Join with that variable among its arguments, so that the return
// sets a function's named error result to a non-nil error whenever that
// variable is not nil.
func returnsErr(body *ast.BlockStmt, errName string, n int) bool {
	if len(body.List) == 0 {
		return false
	}
	ret, ok := body.List[0].(*ast.ReturnStmt)
	if !ok || len(ret.Results) != n || n == 0 {
		return false
	}
	isErr := func(e ast.Expr) bool {
		id, ok := e.(*ast.Ident)
		return ok && id.Name == errName
	}
	switch e := ret.Results[n-1].(type) {
	case *ast.Ident:
		return isErr(e)
	case *ast.CallExpr:
		sel, ok := e.Fun.(*ast.SelectorExpr)
		if !ok {
			return false
		}
		pkg, ok := sel.X.(*ast.Ident)
		if ok && (pkg.Name == "fmt" && sel.Sel.Name == "Errorf" || pkg.Name == "errors" && sel.Sel.Name == "Join") {
			return slices.ContainsFunc(e.Args, isErr)
		}
	}
	return false
}

// releaseOwner returns v if stmt calls v.Free or v.Close, as defined by
// isRelease, whenever it runs: as an expression statement, as the value
// of an assignment, or in the initializer of an if statement. It returns
// "" for any other statement.
func releaseOwner(stmt ast.Stmt) string {
	var e ast.Expr
	switch s := stmt.(type) {
	case *ast.ExprStmt:
		e = s.X
	case *ast.AssignStmt:
		if len(s.Rhs) == 1 {
			e = s.Rhs[0]
		}
	case *ast.IfStmt:
		if init, ok := s.Init.(*ast.AssignStmt); ok && len(init.Rhs) == 1 {
			e = init.Rhs[0]
		}
	}
	c, ok := e.(*ast.CallExpr)
	if !ok {
		return ""
	}
	sel, ok := c.Fun.(*ast.SelectorExpr)
	if !ok {
		return ""
	}
	if id, ok := sel.X.(*ast.Ident); ok && isRelease(c, id.Name) {
		return id.Name
	}
	return ""
}

// isNotNil reports whether cond is "name != nil".
func isNotNil(cond ast.Expr, name string) bool {
	b, ok := cond.(*ast.BinaryExpr)
	if !ok || b.Op != token.NEQ {
		return false
	}
	x, xok := b.X.(*ast.Ident)
	y, yok := b.Y.(*ast.Ident)
	return xok && yok && x.Name == name && y.Name == "nil"
}

// ownerRoot returns the variable whose memory e, an assignable expression
// whose ancestors are stack, lies in: v for v, v.f, v[i], *v or any mix of
// them. For b.f, where b is the value of a range statement in stack over
// a composite literal of struct literals, it is the variable that field f
// of every element lies in, as &v.g does in v. It returns "" if there is
// none, or the elements disagree.
func ownerRoot(e ast.Expr, stack []ast.Node) string {
	switch x := e.(type) {
	case *ast.Ident:
		return x.Name
	case *ast.ParenExpr:
		return ownerRoot(x.X, stack)
	case *ast.StarExpr:
		return ownerRoot(x.X, stack)
	case *ast.IndexExpr:
		return ownerRoot(x.X, stack)
	case *ast.UnaryExpr:
		if x.Op == token.AND {
			return ownerRoot(x.X, stack)
		}
	case *ast.SelectorExpr:
		b, ok := x.X.(*ast.Ident)
		if !ok {
			return ownerRoot(x.X, stack)
		}
		for _, a := range slices.Backward(stack) {
			rs, ok := a.(*ast.RangeStmt)
			if !ok {
				continue
			}
			if v, ok := rs.Value.(*ast.Ident); !ok || v.Name != b.Name {
				continue
			}
			lit, ok := rs.X.(*ast.CompositeLit)
			if !ok {
				return ""
			}
			return fieldRoot(lit, x.Sel.Name, stack)
		}
		return b.Name
	}
	return ""
}

// fieldRoot returns the variable that field of every element of lit, a
// slice or array literal of struct literals, lies in, or "" if the
// elements disagree or a field is missing.
func fieldRoot(lit *ast.CompositeLit, field string, stack []ast.Node) string {
	pos := -1 // the field's position, for elements without keys
	if at, ok := lit.Type.(*ast.ArrayType); ok {
		if st, ok := at.Elt.(*ast.StructType); ok {
			i := 0
			for _, f := range st.Fields.List {
				for _, n := range f.Names {
					if n.Name == field {
						pos = i
					}
					i++
				}
			}
		}
	}
	root := ""
	for _, elt := range lit.Elts {
		el, ok := elt.(*ast.CompositeLit)
		if !ok {
			return ""
		}
		var v ast.Expr
		for i, fe := range el.Elts {
			if kv, ok := fe.(*ast.KeyValueExpr); ok {
				if k, ok := kv.Key.(*ast.Ident); ok && k.Name == field {
					v = kv.Value
				}
			} else if i == pos {
				v = fe
			}
		}
		if v == nil {
			return ""
		}
		r := ownerRoot(v, stack)
		if r == "" || root != "" && r != root {
			return ""
		}
		root = r
	}
	return root
}

// registersRelease reports whether stmt arranges for owner to be released
// on every path out of the function: a defer or t.Cleanup whose call
// releases it, or, if nilable, only "if owner != nil" whose body is
// such a statement.
func registersRelease(stmt ast.Stmt, owner string, nilable bool) bool {
	if nilable {
		s, ok := stmt.(*ast.IfStmt)
		if !ok || s.Init != nil || s.Else != nil || len(s.Body.List) != 1 {
			return false
		}
		cond, ok := s.Cond.(*ast.BinaryExpr)
		if !ok || cond.Op != token.NEQ {
			return false
		}
		x, xok := cond.X.(*ast.Ident)
		y, yok := cond.Y.(*ast.Ident)
		if !xok || !yok || x.Name != owner || y.Name != "nil" {
			return false
		}
		stmt = s.Body.List[0]
	}
	switch s := stmt.(type) {
	case *ast.DeferStmt:
		return callReleases(s.Call, owner)
	case *ast.ExprStmt:
		call, ok := s.X.(*ast.CallExpr)
		if !ok || len(call.Args) != 1 {
			return false
		}
		sel, ok := call.Fun.(*ast.SelectorExpr)
		if !ok || sel.Sel.Name != "Cleanup" {
			return false
		}
		lit, ok := call.Args[0].(*ast.FuncLit)
		return ok && bodyReleases(lit.Body, owner)
	}
	return false
}

// callReleases reports whether call is owner.Free() or owner.Close(), or a
// call of a function literal whose body releases owner.
func callReleases(call *ast.CallExpr, owner string) bool {
	if lit, ok := call.Fun.(*ast.FuncLit); ok {
		return bodyReleases(lit.Body, owner)
	}
	return isRelease(call, owner)
}

// bodyReleases reports whether a top-level statement of body calls
// owner.Free or owner.Close, directly or as the call in an if statement's
// initializer, as in "if err := owner.Free(); err != nil".
func bodyReleases(body *ast.BlockStmt, owner string) bool {
	for _, stmt := range body.List {
		switch s := stmt.(type) {
		case *ast.ExprStmt:
			if call, ok := s.X.(*ast.CallExpr); ok && isRelease(call, owner) {
				return true
			}
		case *ast.AssignStmt:
			if call, ok := s.Rhs[0].(*ast.CallExpr); ok && len(s.Rhs) == 1 && isRelease(call, owner) {
				return true
			}
		case *ast.IfStmt:
			if init, ok := s.Init.(*ast.AssignStmt); ok && len(init.Rhs) == 1 {
				if call, ok := init.Rhs[0].(*ast.CallExpr); ok && isRelease(call, owner) {
					return true
				}
			}
		}
	}
	return false
}

func isRelease(call *ast.CallExpr, owner string) bool {
	sel, ok := call.Fun.(*ast.SelectorExpr)
	if !ok || (sel.Sel.Name != "Free" && sel.Sel.Name != "Close") {
		return false
	}
	id, ok := sel.X.(*ast.Ident)
	return ok && id.Name == owner
}

// mentions reports whether n refers to the identifier name.
func mentions(n ast.Node, name string) bool {
	found := false
	ast.Inspect(n, func(n ast.Node) bool {
		if id, ok := n.(*ast.Ident); ok && id.Name == name {
			found = true
		}
		return !found
	})
	return found
}

// A callCount counts the calls a source check watches in one file: all of
// them, in any form, and those in the one form it inspects. Rewriting an
// inspected call into another form changes checked, not calls.
type callCount struct{ calls, checked int }

// scanDir runs check on every .go file in dir except those named in skip,
// whatever their build constraints, and returns the count check reports
// for each file, omitting files with no calls, and the problems it reports.
func scanDir(dir string, check func(name, src string) (callCount, []string, error), skip ...string) (map[string]callCount, []string, error) {
	paths, err := filepath.Glob(filepath.Join(dir, "*.go"))
	if err != nil {
		return nil, nil, err
	}
	counts := make(map[string]callCount)
	var bad []string
	for _, p := range paths {
		name := filepath.Base(p)
		if slices.Contains(skip, name) {
			continue
		}
		src, err := os.ReadFile(p)
		if err != nil {
			return nil, nil, err
		}
		n, b, err := check(name, string(src))
		if err != nil {
			return nil, nil, err
		}
		if n.calls > 0 {
			counts[name] = n
		}
		bad = append(bad, b...)
	}
	return counts, bad, nil
}

// countDiffs returns, sorted by file, each file whose counts in got differ
// from its counts in want, a missing file counting zero.
func countDiffs(got, want map[string]callCount) []string {
	names := slices.Concat(slices.Collect(maps.Keys(got)), slices.Collect(maps.Keys(want)))
	slices.Sort(names)
	var diffs []string
	for _, name := range slices.Compact(names) {
		if got[name] != want[name] {
			g, w := got[name], want[name]
			diffs = append(diffs, fmt.Sprintf("%s: %d calls, %d checked; want %d, %d", name, g.calls, g.checked, w.calls, w.checked))
		}
	}
	return diffs
}

// packageWith returns a temporary directory holding a copy of every .go
// file in the package and a file name with contents src.
func packageWith(t *testing.T, name, src string) string {
	t.Helper()
	dir := t.TempDir()
	paths, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range paths {
		b, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, p), b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(dir, name), []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	return dir
}

// isExtra reports whether a problem is in the file packageWith adds.
func isExtra(problem string) bool { return strings.HasPrefix(problem, "extra.go:") }

// rewriteScan returns what scanDir with check reports for a copy of the
// package in which the first from in file is replaced by to.
func rewriteScan(t *testing.T, check func(name, src string) (callCount, []string, error), file, from, to string, skip ...string) (map[string]callCount, []string) {
	t.Helper()
	b, err := os.ReadFile(file)
	if err != nil {
		t.Fatal(err)
	}
	src := strings.Replace(string(b), from, to, 1)
	if src == string(b) {
		t.Fatalf("control does not apply: %s has no %q", file, from)
	}
	got, bad, err := scanDir(packageWith(t, file, src), check, skip...)
	if err != nil {
		t.Fatal(err)
	}
	return got, bad
}

// rewriteDiffs returns the count differences from want that scanDir with
// check reports for a copy of the package in which the first from in file
// is replaced by to.
func rewriteDiffs(t *testing.T, check func(name, src string) (callCount, []string, error), want map[string]callCount, file, from, to string, skip ...string) []string {
	t.Helper()
	got, _ := rewriteScan(t, check, file, from, to, skip...)
	return countDiffs(got, want)
}

// controlImports imports, under the names the must-fail controls use, the
// packages whose functions the checks match.
const controlImports = `import (
	"github.com/tmc/lib/nvidia/x/cudaattn"
	"github.com/tmc/lib/nvidia/x/cudakv"
	"github.com/tmc/lib/nvidia/x/cudamem"
	mem "github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
	nv "github.com/tmc/lib/nvidia/x/nvrtcx"
	"regexp"
	other "example.com/other"
)
`

// controlSrc returns a file of package p, importing controlImports, with a
// function f whose body is body followed by "return nil".
func controlSrc(body string) string {
	return "package p\n" + controlImports + "func f() error {\n" + body + "\nreturn nil\n}\n"
}

// ownerCalls counts, by file, the references to an ownerReturning
// constructor in the package, called or not, and the calls releasesFirst
// inspects. A reference added in any file, in any form, or rewritten out of
// the inspected form, changes a count, and must be reviewed before the
// count is updated here; ownerSites lists each one not inspected.
var ownerCalls = map[string]callCount{
	"main.go":                   {6, 5},
	"fused_parity_cuda_test.go": {6, 6},
	"bf16_parity_cuda_test.go":  {6, 6},
	"fused_decode_cuda_test.go": {1, 1},
	"graph_parity_cuda_test.go": {12, 12},
	"profile_cuda_test.go":      {2, 2},
	"gpu_kernels.go":            {1, 0},
	"gpu_model.go":              {6, 0},
	"decode_graph.go":           {1, 0},
	"weights_bf16.go":           {2, 0},
	"gpu_model_test.go":         {4, 0},
	"decode_graph_test.go":      {1, 0},
	"egraph_gpu.go":             {25, 0},
}

// testFuncs returns the names of the functions declared in the package's
// _test.go files.
func testFuncs(t *testing.T) map[string]bool {
	t.Helper()
	paths, err := filepath.Glob("*_test.go")
	if err != nil {
		t.Fatal(err)
	}
	names := make(map[string]bool)
	fset := token.NewFileSet()
	for _, p := range paths {
		f, err := parser.ParseFile(fset, p, nil, parser.SkipObjectResolution)
		if err != nil {
			t.Fatal(err)
		}
		for _, d := range f.Decls {
			if fd, ok := d.(*ast.FuncDecl); ok && fd.Recv == nil {
				names[fd.Name.Name] = true
			}
		}
	}
	return names
}

// TestCallersReleaseReturnedOwners checks every reference to a constructor
// that can return an owner with an error, in every .go file of the
// package: a call in the inspected form releases the owner first, and any
// other reference is listed in ownerSites. That a listed site is safe is
// checked in the source for the "return" and "owner field" forms only;
// for the others it rests on the test ownerSites names, or on its reason.
func TestCallersReleaseReturnedOwners(t *testing.T) {
	got, bad, err := scanDir(".", releasesFirst)
	if err != nil {
		t.Fatal(err)
	}
	for _, d := range countDiffs(got, ownerCalls) {
		t.Error(d)
	}
	for _, b := range bad {
		t.Error(b)
	}
	tests := testFuncs(t)
	for _, s := range ownerSites {
		if _, err := os.Stat(s.file); err != nil {
			t.Errorf("ownerSites lists %s: %v", s.file, err)
		}
		if s.test != "" && !tests[s.test] {
			t.Errorf("ownerSites names %s for %v, which is not declared", s.test, s.ownerSite)
		}
	}
	if tests["TestNoSuchTest"] || !tests["TestUploadModelWith"] {
		t.Error("must-fail control: testFuncs does not report the declared tests")
	}

	// Must-fail controls: a reference in a file the count does not list is
	// caught, by its count whatever its form, and by the check too unless it
	// releases its owner first, since ownerSites lists nothing there; and a
	// checked call in a listed file rewritten into a form the check does
	// not inspect is caught by its file's count.
	for _, tt := range []struct {
		name, body string
		bad        bool
	}{
		{"released", "kv, err := cudakv.New(dev, 1, 1, 1, 1)\nif kv != nil {\ndefer kv.Close()\n}\nreturn err", false},
		{"error first", "kv, err := cudakv.New(dev, 1, 1, 1, 1)\nif err != nil {\nreturn err\n}\nreturn kv.Close()", true},
		{"returned", "return NewGPUScratchpad(dev, cfg)", true},
		{"function value", "alloc := cudamem.Alloc[float32]\n_ = alloc\nreturn nil", true},
	} {
		dir := packageWith(t, "extra.go", "package main\n"+controlImports+"func f() error {\n"+tt.body+"\n}\n")
		got, bad, err := scanDir(dir, releasesFirst)
		if err != nil {
			t.Fatalf("unlisted file, %s: %v", tt.name, err)
		}
		if d := countDiffs(got, ownerCalls); !slices.ContainsFunc(d, isExtra) {
			t.Errorf("unlisted file, %s: count differences %q, want extra.go's", tt.name, d)
		}
		if flagged := slices.ContainsFunc(bad, isExtra); flagged != tt.bad {
			t.Errorf("unlisted file, %s: %q, want extra.go flagged %v", tt.name, bad, tt.bad)
		}
	}
	d := rewriteDiffs(t, releasesFirst, ownerCalls, "main.go", "scratch, err := NewGPUScratchpad(", "var scratch, err = NewGPUScratchpad(")
	if !slices.ContainsFunc(d, func(s string) bool { return strings.HasPrefix(s, "main.go:") }) {
		t.Errorf("checked call rewritten as a var declaration: count differences %q, want main.go's", d)
	}

	// Must-fail controls on the listed sites: each edit of the real source
	// moves a reference out of its listed site, or empties one, and is
	// caught in its file.
	for _, tt := range []struct{ name, file, from, to string }{
		{"scratch allocators handed to another function", "gpu_model.go", "return newGPUScratchpadWith(cfg,", "return newGPUScratchpadWith2(cfg,"},
		{"delegation moved out of the constructor", "gpu_kernels.go", "return compileGPUModules(dev, ", "return compileGPUModulesAgain(dev, "},
		{"allocator handed to another function", "weights_bf16.go", "return uploadWeightWith(host, dtype,", "return uploadWeightWith2(host, dtype,"},
	} {
		_, bad := rewriteScan(t, releasesFirst, tt.file, tt.from, tt.to)
		if !slices.ContainsFunc(bad, func(s string) bool { return strings.HasPrefix(s, tt.file+":") }) {
			t.Errorf("%s: %q, want %s flagged", tt.name, bad, tt.file)
		}
	}

	for _, tt := range []struct {
		name, body string
		bad        bool
	}{
		{"positive control: release registered first, any error name", `
		kv, e := cudakv.New(dev, 1, 1, 1, 1)
		if kv != nil {
			defer kv.Close()
		}
		if e != nil {
			return e
		}`, false},
		{"positive control: deferred literal reporting the error", `
		sp, err := NewGPUScratchpad(dev, cfg)
		if sp != nil {
			defer func() {
				if err := sp.Free(); err != nil {
					log(err)
				}
			}()
		}
		if err != nil {
			return err
		}`, false},
		{"positive control: t.Cleanup reporting the error", `
		s, err := cudamem.Alloc[int32](dev, 1)
		t.Cleanup(func() {
			if err := s.Free(); err != nil {
				t.Error(err)
			}
		})
		if err != nil {
			t.Fatal(err)
		}`, false},
		{"positive control: aliased import", `
		s, err := mem.Alloc[int32](dev, 1)
		t.Cleanup(func() { s.Free() })
		if err != nil {
			t.Fatal(err)
		}`, false},
		{"aliased import, error first", `
		s, err := mem.Alloc[int32](dev, 1)
		if err != nil {
			return err
		}
		defer s.Free()`, true},
		{"error examined first", `
		kv, err := cudakv.New(dev, 1, 1, 1, 1)
		if err != nil {
			return err
		}
		defer kv.Close()`, true},
		{"generic allocation, error first", `
		s, err := cudamem.Alloc[int32](dev, 1)
		if err != nil {
			return err
		}
		defer s.Free()`, true},
		{"releases another owner", `
		gm, err := CompileGPUModules(dev)
		defer kv.Close()
		if err != nil {
			return err
		}`, true},
		{"guard releases another owner", `
		sp, err := NewGPUScratchpad(dev, cfg)
		if sp != nil {
			defer gw.Free()
		}`, true},
		{"guard without a release", `
		sp, err := NewGPUScratchpad(dev, cfg)
		if sp != nil {
			_ = sp
		}`, true},
		{"owner discarded", `
		_, err := NewGPUScratchpad(dev, cfg)
		return err`, true},
		{"error discarded", `
		sp, _ := NewGPUScratchpad(dev, cfg)
		if sp != nil {
			defer sp.Free()
		}`, true},
		{"assignment, not declaration", `
		gw, err = uploadModel(dev, w, cfg, dtype)
		return err`, true},
		{"nilable owner released unguarded", `
		sp, err := NewGPUScratchpad(dev, cfg)
		defer func() { _ = sp.Free() }()
		if err != nil {
			return err
		}`, true},
		{"statement before the release", `
		kv, err := cudakv.New(dev, 1, 1, 1, 1)
		if done {
			return nil
		}
		if kv != nil {
			defer kv.Close()
		}`, true},
		{"guarded release not deferred", `
		sp, err := NewGPUScratchpad(dev, cfg)
		if sp != nil {
			sp.Free()
		}`, true},
		{"release only in the else branch", `
		kv, err := cudakv.New(dev, 1, 1, 1, 1)
		if err != nil {
			return err
		} else {
			defer kv.Close()
		}`, true},
		{"release only on success", `
		kv, err := cudakv.New(dev, 1, 1, 1, 1)
		if err == nil {
			defer kv.Close()
		}`, true},
		{"release in a closure never called", `
		kv, err := cudakv.New(dev, 1, 1, 1, 1)
		release := func() { kv.Close() }
		if err != nil {
			return err
		}
		release()`, true},
		{"method value, not a call", `
		sp, err := NewGPUScratchpad(dev, cfg)
		if sp != nil {
			defer func() { _ = sp.Free }()
		}`, true},
		{"release under if false", `
		kv, err := cudakv.New(dev, 1, 1, 1, 1)
		if kv != nil {
			if false {
				defer kv.Close()
			}
		}`, true},
	} {
		n, bad, err := releasesFirst("control.go", controlSrc(tt.body))
		if err != nil || n != (callCount{1, 1}) || (len(bad) > 0) != tt.bad {
			t.Errorf("%s: %+v calls, %q, %v; want 1 checked, flagged %v", tt.name, n, bad, err, tt.bad)
		}
	}
	// Other references count, but are not inspected, and are flagged unless
	// ownerSites lists them, which it does for no control.go; names that
	// only look like a constructor do not count.
	for _, tt := range []struct {
		body string
		want callCount
	}{
		{"var sp, err = NewGPUScratchpad(dev, cfg)", callCount{1, 0}},
		{"if sp, err = NewGPUScratchpad(dev, cfg); err != nil {\nreturn err\n}", callCount{1, 0}},
		{"return NewGPUScratchpad(dev, cfg)", callCount{1, 0}},
		{"alloc := cudamem.Alloc[float32]", callCount{1, 0}},
		{"alloc := mem.Alloc[float32]", callCount{1, 0}},
		{"newKV := cudakv.New\nkv, err := newKV(dev, 1, 1, 1, 1)\nif kv != nil {\ndefer kv.Close()\n}", callCount{1, 0}},
		{"upload := uploadModelWith", callCount{1, 0}},
		{"use(NewGPUScratchpad)", callCount{1, 0}},
		{"s, err := other.Alloc(dev, 1)\nx.uploadModel(dev)\n_ = w.NewGPUScratchpad", callCount{0, 0}},
	} {
		n, bad, err := releasesFirst("control.go", controlSrc(tt.body))
		if err != nil || n != tt.want || (len(bad) > 0) != (tt.want.calls > 0) {
			t.Errorf("%q: %+v calls, %q, %v; want %+v, flagged %v", tt.body, n, bad, err, tt.want, tt.want.calls > 0)
		}
	}
	src := "package p\nimport . \"github.com/tmc/lib/nvidia/x/cudamem\"\nfunc f() {\n_, _ = Alloc[float32](dev, 1)\n}\n"
	if _, bad, err := releasesFirst("control.go", src); err != nil || len(bad) == 0 {
		t.Errorf("dot import: %q, %v; want flagged", bad, err)
	}
}

// TestOwnerSiteForms checks the form ownerRefs gives references that
// releasesFirst does not inspect, with must-fail controls for the forms
// it accepts as safe in the source, "return" and "owner field".
func TestOwnerSiteForms(t *testing.T) {
	const (
		head = `func NewGPUScratchpad(dev *cuda.Device, cfg *ModelConfig) (res *GPUScratchpad, err error) {
	sp := &GPUScratchpad{}
`
		release = `	defer func() {
		if err != nil {
			if ferr := sp.Free(); ferr != nil {
				err = errors.Join(err, ferr)
				res = sp
			}
		}
	}()
`
		allocs = `	for _, b := range []struct {
		dst *cudamem.Slice[float32]
		n   int
	}{
		{&sp.X, 1},
		{dst: &sp.Q, n: 2},
	} {
		if *b.dst, err = cudamem.Alloc[float32](dev, b.n); err != nil {
			return nil, err
		}
	}
	if sp.TokenIn, err = cudamem.Alloc[int32](dev, 1); err != nil {
		return nil, err
	}
`
		tail    = "\treturn sp, nil\n}"
		scratch = head + release + allocs + tail
	)
	field := func(fn string) map[ownerSite]int {
		return map[ownerSite]int{{"control.go", fn, libX + "cudamem.Alloc", "owner field"}: 2}
	}
	call := func(fn string, n int) map[ownerSite]int {
		return map[ownerSite]int{{"control.go", fn, libX + "cudamem.Alloc", "call"}: n}
	}
	// oneField is scratch's sites with the token buffer's call not
	// accepted as an owner field.
	oneField := map[ownerSite]int{
		{"control.go", "NewGPUScratchpad", libX + "cudamem.Alloc", "owner field"}: 1,
		{"control.go", "NewGPUScratchpad", libX + "cudamem.Alloc", "call"}:        1,
	}
	// wrapFree puts the deferred release inside the block opened by open.
	wrapFree := func(open string) string {
		const free = "\t\t\tif ferr := sp.Free(); ferr != nil {\n\t\t\t\terr = errors.Join(err, ferr)\n\t\t\t\tres = sp\n\t\t\t}\n"
		return strings.Replace(scratch, free, "\t\t\t"+open+" {\n"+free+"\t\t\t}\n", 1)
	}
	tokenReturn := "dev, 1); err != nil {\n\t\treturn nil, err\n"
	for _, tt := range []struct {
		name, src string
		want      map[ownerSite]int
	}{
		{"positive control: owner fields", scratch, field("NewGPUScratchpad")},
		{"not an owner constructor", strings.Replace(scratch, "func NewGPUScratchpad(", "func newScratch(", 1), call("newScratch", 2)},
		{"defer registered after", head + allocs + release + tail, call("NewGPUScratchpad", 2)},
		{"defer releases another owner", strings.Replace(scratch, "sp.Free()", "gw.Free()", 1), call("NewGPUScratchpad", 2)},
		{"defer releases on success", strings.Replace(scratch, "\t\tif err != nil {\n\t\t\tif ferr", "\t\tif err == nil {\n\t\t\tif ferr", 1), call("NewGPUScratchpad", 2)},
		{"error result not named err", strings.Replace(scratch, "res *GPUScratchpad, err error", "res *GPUScratchpad, e error", 1), call("NewGPUScratchpad", 2)},
		{"condition on success", strings.Replace(scratch, "dev, 1); err != nil", "dev, 1); err == nil", 1), map[ownerSite]int{
			{"control.go", "NewGPUScratchpad", libX + "cudamem.Alloc", "owner field"}: 1,
			{"control.go", "NewGPUScratchpad", libX + "cudamem.Alloc", "call"}:        1,
		}},
		{"stored outside the owner", strings.Replace(scratch, "if sp.TokenIn, err", "if tok, err", 1), map[ownerSite]int{
			{"control.go", "NewGPUScratchpad", libX + "cudamem.Alloc", "owner field"}: 1,
			{"control.go", "NewGPUScratchpad", libX + "cudamem.Alloc", "call"}:        1,
		}},
		{"positional element outside the owner", strings.Replace(scratch, "{&sp.X, 1}", "{&other.X, 1}", 1), map[ownerSite]int{
			{"control.go", "NewGPUScratchpad", libX + "cudamem.Alloc", "owner field"}: 1,
			{"control.go", "NewGPUScratchpad", libX + "cudamem.Alloc", "call"}:        1,
		}},
		{"keyed element outside the owner", strings.Replace(scratch, "{dst: &sp.Q, n: 2}", "{dst: &other.Q, n: 2}", 1), map[ownerSite]int{
			{"control.go", "NewGPUScratchpad", libX + "cudamem.Alloc", "owner field"}: 1,
			{"control.go", "NewGPUScratchpad", libX + "cudamem.Alloc", "call"}:        1,
		}},
		{"positive control: wrapped error returned", strings.Replace(scratch, tokenReturn, "dev, 1); err != nil {\n\t\treturn nil, fmt.Errorf(\"alloc: %w\", err)\n", 1), field("NewGPUScratchpad")},
		{"error swallowed", strings.Replace(scratch, tokenReturn, "dev, 1); err != nil {\n\t\treturn nil, nil\n", 1), oneField},
		{"error cleared before the return", strings.Replace(scratch, tokenReturn, "dev, 1); err != nil {\n\t\terr = nil\n\t\treturn nil, err\n", 1), oneField},
		{"error passed through another function", strings.Replace(scratch, tokenReturn, "dev, 1); err != nil {\n\t\treturn nil, ignore(err)\n", 1), oneField},
		{"defer literal shadows err", strings.Replace(strings.Replace(scratch, "\tdefer func() {", "\tdefer func(err error) {", 1), "\t}()\n", "\t}(nil)\n", 1), call("NewGPUScratchpad", 2)},
		{"release under if false", wrapFree("if false"), call("NewGPUScratchpad", 2)},
		{"release in an uncalled closure", wrapFree("_ = func()"), call("NewGPUScratchpad", 2)},
		{"release after a return", strings.Replace(scratch, "\t\tif err != nil {\n\t\t\tif ferr", "\t\tif err != nil {\n\t\t\treturn\n\t\t\tif ferr", 1), call("NewGPUScratchpad", 2)},
		{"positive control: returned by a constructor", "func UploadModelToGPU() (*GPUModelWeights, error) {\nreturn uploadModel(dev, w, cfg, weightF32)\n}",
			map[ownerSite]int{{"control.go", "UploadModelToGPU", "uploadModel", "return"}: 1}},
		{"returned by another function", "func upload() (*GPUModelWeights, error) {\nreturn uploadModel(dev, w, cfg, weightF32)\n}",
			map[ownerSite]int{{"control.go", "upload", "uploadModel", "call"}: 1}},
		{"returned from a nested block", "func UploadModelToGPU() (*GPUModelWeights, error) {\nif ok {\nreturn uploadModel(dev, w, cfg, weightF32)\n}\nreturn nil, nil\n}",
			map[ownerSite]int{{"control.go", "UploadModelToGPU", "uploadModel", "call"}: 1}},
		{"positive control: allocator", "func g() {\ntake(func(n int) (cudamem.Slice[float32], error) { return cudamem.Alloc[float32](dev, n) })\n}",
			map[ownerSite]int{{"control.go", "g", libX + "cudamem.Alloc", "allocator for take"}: 1}},
		{"allocator with another statement", "func g() {\ntake(func(n int) (cudamem.Slice[float32], error) { n++; return cudamem.Alloc[float32](dev, n) })\n}",
			call("g", 1)},
		{"allocator kept, not passed", "func g() {\nalloc := func(n int) (cudamem.Slice[float32], error) { return cudamem.Alloc[float32](dev, n) }\n_ = alloc\n}",
			call("g", 1)},
		{"function value", "func g() {\ntake(cudamem.Alloc[float32])\n}",
			map[ownerSite]int{{"control.go", "g", libX + "cudamem.Alloc", "value"}: 1}},
	} {
		if tt.src == scratch && tt.name != "positive control: owner fields" {
			t.Fatalf("%s: control does not apply", tt.name)
		}
		_, sites, _, err := ownerRefs("control.go", "package p\n"+controlImports+tt.src+"\n")
		if err != nil || !maps.Equal(sites, tt.want) {
			t.Errorf("%s: %v, %v; want %v", tt.name, sites, err, tt.want)
		}
	}
}

// heldCleanupError is the error a failed load returns while still holding
// a module, like nvrtcx's: Close retries the unload, failing the first
// fails times.
type heldCleanupError struct {
	fails  int
	closes int
}

func (e *heldCleanupError) Error() string { return "load failed, module retained" }

func (e *heldCleanupError) Close() error {
	e.closes++
	if e.fails > 0 {
		e.fails--
		return errFakeUnload
	}
	return nil
}

// releaseErrorFailures returns how release departs from: nothing closed for a
// nil or unrelated error, which is returned as is; a retained module
// closed once through a wrapped error, and the error returned as is; and
// if that close fails, the original error still carrying its Closer,
// joined with the close error.
func releaseErrorFailures(release func(error) error) []string {
	var bad []string
	if err := release(nil); err != nil {
		bad = append(bad, fmt.Sprintf("nil: %v", err))
	}
	if err := release(errFakeLookup); err != errFakeLookup {
		bad = append(bad, fmt.Sprintf("unrelated: %v", err))
	}

	held := &heldCleanupError{}
	wrapped := fmt.Errorf("load: %w", held)
	if err := release(wrapped); err != wrapped || held.closes != 1 {
		bad = append(bad, fmt.Sprintf("close succeeds: %v, %d closes", err, held.closes))
	}

	held = &heldCleanupError{fails: 1}
	wrapped = fmt.Errorf("load: %w", held)
	err := release(wrapped)
	var c *heldCleanupError
	if !errors.Is(err, wrapped) || !errors.As(err, &c) || c != held {
		bad = append(bad, fmt.Sprintf("close fails: %v loses the error carrying the module", err))
	}
	if !errors.Is(err, errFakeUnload) || !strings.Contains(fmt.Sprint(err), "release resources held by failed call") {
		bad = append(bad, fmt.Sprintf("close fails: %v does not report the failed release", err))
	}
	if held.closes != 1 {
		bad = append(bad, fmt.Sprintf("close fails: %d closes, want 1", held.closes))
	}
	return bad
}

func TestReleaseErrorOwner(t *testing.T) {
	for _, f := range releaseErrorFailures(releaseErrorOwner) {
		t.Error(f)
	}
	for _, m := range []struct {
		name    string
		release func(error) error
	}{
		{"L1 nothing closed", func(err error) error { return err }},
		{"L2 replaced by the close error", func(err error) error {
			var c io.Closer
			if errors.As(err, &c) {
				if cerr := c.Close(); cerr != nil {
					return cerr
				}
			}
			return err
		}},
		{"L3 close error dropped", func(err error) error {
			var c io.Closer
			if errors.As(err, &c) {
				c.Close()
			}
			return err
		}},
	} {
		if len(releaseErrorFailures(m.release)) == 0 {
			t.Errorf("must-fail control %s: not detected", m.name)
		}
	}
}

// loadCalls counts, by file, the references to a compile or load entry
// point in the package (see isLoad), called or not, and the calls
// callErrorsReleased inspects, except gen.go's, which
// TestGenReleasesCompileError counts and checks: gen.go builds alone and
// cannot call releaseErrorOwner.
var loadCalls = map[string]callCount{
	"main.go":                         {5, 5},
	"fused_parity_cuda_test.go":       {4, 4},
	"bf16_parity_cuda_test.go":        {4, 4},
	"context_experiment_cuda_test.go": {2, 2},
	"gpu_kernels.go":                  {2, 2},
	"egraph_gpu.go":                   {2, 2},
	"egraph_plan_tcgen05_test.go":     {1, 1},
	"tp.go":                           {2, 2},
}

// TestCallErrorsReleased checks that every module compile and load in the
// package passes its error through releaseErrorOwner.
func TestCallErrorsReleased(t *testing.T) {
	got, bad, err := scanDir(".", callErrorsReleased, "gen.go")
	if err != nil {
		t.Fatal(err)
	}
	for _, d := range countDiffs(got, loadCalls) {
		t.Error(d)
	}
	for _, b := range bad {
		t.Error(b)
	}

	// Must-fail controls: gen.go is skipped by name only; a reference in a
	// file the count does not list is caught, by its count whatever its form
	// and by the check too if it returns the error unreleased; and a checked
	// load in a listed file rewritten into a form the check does not
	// inspect is caught by its file's count.
	if _, bad, err := scanDir(".", callErrorsReleased); err != nil || len(bad) == 0 {
		t.Errorf("gen.go not skipped: %q, %v; want its compile flagged", bad, err)
	}
	for _, tt := range []struct {
		name, body string
		bad        bool
	}{
		{"released", "m, err := c.Load(dev)\nif err != nil {\nreturn releaseErrorOwner(err)\n}\n_ = m\nreturn nil", false},
		{"error returned unreleased", "m, err := c.Load(dev)\nif err != nil {\nreturn err\n}\n_ = m\nreturn nil", true},
		{"returned", "return c.Load(dev)", false},
		{"function value", "compile := nvrtcx.CompileContext\n_ = compile\nreturn nil", false},
	} {
		dir := packageWith(t, "extra.go", "package main\n"+controlImports+"func f() error {\n"+tt.body+"\n}\n")
		got, bad, err := scanDir(dir, callErrorsReleased, "gen.go")
		if err != nil {
			t.Fatalf("unlisted file, %s: %v", tt.name, err)
		}
		if d := countDiffs(got, loadCalls); !slices.ContainsFunc(d, isExtra) {
			t.Errorf("unlisted file, %s: count differences %q, want extra.go's", tt.name, d)
		}
		if flagged := slices.ContainsFunc(bad, isExtra); flagged != tt.bad {
			t.Errorf("unlisted file, %s: %q, want extra.go flagged %v", tt.name, bad, tt.bad)
		}
	}
	d := rewriteDiffs(t, callErrorsReleased, loadCalls, "main.go", "attnMod, err := attnModCompiled.Load(", "var attnMod, err = attnModCompiled.Load(", "gen.go")
	if !slices.ContainsFunc(d, func(s string) bool { return strings.HasPrefix(s, "main.go:") }) {
		t.Errorf("checked load rewritten as a var declaration: count differences %q, want main.go's", d)
	}

	for _, tt := range []struct {
		name, body string
		bad        bool
	}{
		{"positive control", "if err != nil {\nreturn releaseErrorOwner(err)\n}", false},
		{"error returned unreleased", "if err != nil {\nreturn err\n}", true},
		{"helper given another error", "if err != nil {\nreleaseErrorOwner(nil)\nreturn err\n}", true},
		{"helper only on success", "if err == nil {\nreleaseErrorOwner(err)\n}", true},
	} {
		n, bad, err := callErrorsReleased("control.go", controlSrc("m, err := c.Load(dev)\n"+tt.body))
		if err != nil || n != (callCount{1, 1}) || (len(bad) > 0) != tt.bad {
			t.Errorf("%s: %+v loads, %q, %v; want 1 checked, flagged %v", tt.name, n, bad, err, tt.bad)
		}
	}
	// Every entry point, under any import name, is inspected; a function
	// of another package is not an entry point, whatever its name.
	const unreleased = "\nif err != nil {\nreturn err\n}"
	for _, tt := range []struct {
		call string
		want callCount
	}{
		{"nvrtcx.CompileContext(ctx, src)", callCount{1, 1}},
		{"nvrtcx.SpliceAndCompile(dev, tmpl, nil)", callCount{1, 1}},
		{"nv.Compile(src)", callCount{1, 1}},
		{"cudaattn.CompilePaged(8, 0)", callCount{1, 1}},
		{"cudaattn.LoadIndexedAttention(dev, c, a)", callCount{1, 1}},
		{"c.LoadModule(dev)", callCount{1, 1}},
		{"c.LoadOutput(dev, kind)", callCount{1, 1}},
		{"c.Bind(dev, &k)", callCount{1, 1}},
		{"nvrtcx.CompileFlags()", callCount{0, 0}},
		{"nvrtcx.WithFastCompile(\"0\")", callCount{0, 0}},
		{"regexp.Compile(src)", callCount{0, 0}},
		{"other.Load(dev)", callCount{0, 0}},
	} {
		n, bad, err := callErrorsReleased("control.go", controlSrc("m, err := "+tt.call+unreleased))
		if err != nil || n != tt.want || (len(bad) > 0) != (tt.want.checked > 0) {
			t.Errorf("%s: %+v loads, %q, %v; want %+v, flagged %v", tt.call, n, bad, err, tt.want, tt.want.checked > 0)
		}
	}
	// Forms the check does not inspect, including references not called,
	// count as calls but not as checked.
	for _, body := range []string{
		"var m, err = c.Load(dev)",
		"if m, err = c.Load(dev); err != nil {\nreturn err\n}",
		"return c.Load(dev)",
		"load := c.Load",
		"compile := cudaattn.Compile",
		"use(nv.CompileContext)",
	} {
		if n, _, err := callErrorsReleased("control.go", controlSrc(body)); err != nil || n != (callCount{1, 0}) {
			t.Errorf("%q: %+v loads, %v; want 1 call, 0 checked", body, n, err)
		}
	}
	src := "package p\nimport . \"github.com/tmc/lib/nvidia/x/nvrtcx\"\nfunc f() {\n_, _ = Compile(src)\n}\n"
	if _, bad, err := callErrorsReleased("control.go", src); err != nil || len(bad) == 0 {
		t.Errorf("dot import: %q, %v; want flagged", bad, err)
	}
}

// libRoot is the directory of the nvidia module's x packages, which the
// examples module replaces with this tree's copy.
const libRoot = "../../../nvidia/x"

// loadEntries returns, sorted, the exported functions and methods
// declared in the non-test .go files of each loadPackages package under
// root, as pkg.Name or pkg.(Recv).Name: those match reports, and the rest
// whose names contain Compile, Load or Bind.
func loadEntries(root string, match func(string) bool) (matched, near []string, err error) {
	fset := token.NewFileSet()
	for p := range loadPackages {
		pkg := path.Base(p)
		paths, err := filepath.Glob(filepath.Join(root, pkg, "*.go"))
		if err != nil {
			return nil, nil, err
		}
		if len(paths) == 0 {
			return nil, nil, fmt.Errorf("no .go files in %s", filepath.Join(root, pkg))
		}
		for _, file := range paths {
			if strings.HasSuffix(file, "_test.go") {
				continue
			}
			f, err := parser.ParseFile(fset, file, nil, parser.SkipObjectResolution)
			if err != nil {
				return nil, nil, err
			}
			for _, d := range f.Decls {
				fd, ok := d.(*ast.FuncDecl)
				if !ok || !fd.Name.IsExported() {
					continue
				}
				name := pkg + "." + fd.Name.Name
				if fd.Recv != nil && len(fd.Recv.List) == 1 {
					name = fmt.Sprintf("%s.(%s).%s", pkg, recvString(fd.Recv.List[0].Type), fd.Name.Name)
				}
				switch {
				case match(fd.Name.Name):
					matched = append(matched, name)
				case strings.Contains(fd.Name.Name, "Compile"), strings.Contains(fd.Name.Name, "Load"), strings.Contains(fd.Name.Name, "Bind"):
					near = append(near, name)
				}
			}
		}
	}
	slices.Sort(matched)
	slices.Sort(near)
	return matched, near, nil
}

// recvString returns a method receiver's type, such as *T, without type
// parameters.
func recvString(e ast.Expr) string {
	switch x := e.(type) {
	case *ast.StarExpr:
		return "*" + recvString(x.X)
	case *ast.IndexExpr:
		return recvString(x.X)
	case *ast.IndexListExpr:
		return recvString(x.X)
	case *ast.Ident:
		return x.Name
	}
	return "?"
}

// loadEntryPoints are the entry points isLoadName matches in the library
// source, and loadNearMisses those whose names resemble one but compile
// and load nothing: CompileFlags resolves options to flags,
// ValidateJITFusionCompilerVersion checks a version number, and
// WithFastCompile returns an Option.
var (
	loadEntryPoints = []string{
		"cudaattn.Compile",
		"cudaattn.CompileIndexedArtifact",
		"cudaattn.CompileIndexedAttention",
		"cudaattn.CompileKDA",
		"cudaattn.CompilePaged",
		"cudaattn.LoadIndexedAttention",
		"cudasample.Compile",
		"nvrtcx.(*CompiledModule).Bind",
		"nvrtcx.(*CompiledModule).Load",
		"nvrtcx.(*CompiledModule).LoadModule",
		"nvrtcx.(*CompiledModule).LoadOutput",
		"nvrtcx.(*CompiledModule).LoadOutputModule",
		"nvrtcx.(*CompiledModule).LoadOutputWithLog",
		"nvrtcx.(*CompiledModule).LoadOwnedOutput",
		"nvrtcx.Compile",
		"nvrtcx.CompileContext",
		"nvrtcx.SpliceAndCompile",
	}
	loadNearMisses = []string{
		"nvrtcx.(*CompiledModule).CompilerOptions",
		"nvrtcx.CompileFlags",
		"nvrtcx.ValidateJITFusionCompilerVersion",
		"nvrtcx.WithFastCompile",
	}
)

// TestLoadEntryPoints checks isLoadName against the library source: it
// must match exactly loadEntryPoints, and every other exported name with
// Compile, Load or Bind in it must be a reviewed near miss. An entry point
// added to the library under another name escapes it.
func TestLoadEntryPoints(t *testing.T) {
	matched, near, err := loadEntries(libRoot, isLoadName)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(matched, loadEntryPoints) {
		t.Errorf("isLoadName matches %q in %s, want %q", matched, libRoot, loadEntryPoints)
	}
	if !slices.Equal(near, loadNearMisses) {
		t.Errorf("unmatched names like an entry point %q in %s, want %q", near, libRoot, loadNearMisses)
	}

	// Must-fail controls: the rule this replaced, matching only Load and
	// Compile; and an entry point the library adds.
	old := func(name string) bool { return name == "Load" || name == "Compile" }
	if matched, _, err := loadEntries(libRoot, old); err != nil || slices.Equal(matched, loadEntryPoints) {
		t.Errorf("must-fail control, old rule: %q, %v; want it to miss entry points", matched, err)
	}
	root := t.TempDir()
	for p := range loadPackages {
		dir := filepath.Join(root, path.Base(p))
		if err := os.Mkdir(dir, 0o755); err != nil {
			t.Fatal(err)
		}
		src := "package " + path.Base(p) + "\nfunc Compile() {}\n"
		if path.Base(p) == "nvrtcx" {
			src += "func BuildAndLoad() {}\nfunc LoadAll() {}\n"
		}
		if err := os.WriteFile(filepath.Join(dir, "x.go"), []byte(src), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	matched, near, err = loadEntries(root, isLoadName)
	if err != nil || !slices.Contains(matched, "nvrtcx.LoadAll") || !slices.Equal(near, []string{"nvrtcx.BuildAndLoad"}) {
		t.Errorf("must-fail control, added entry points: %q, %q, %v", matched, near, err)
	}
}

// loadPackages are the import paths of the packages whose compile and load
// entry points, named per isLoadName, return an error that may hold a
// module or program to release (see TestLoadEntryPoints).
var loadPackages = map[string]bool{
	libX + "nvrtcx":     true,
	libX + "cudaattn":   true,
	libX + "cudasample": true,
}

// isLoadName reports whether name is that of a compile or load entry
// point: Compile or Load followed by anything, Bind, or SpliceAndCompile,
// but not CompileFlags or CompilerOptions, which only resolve or report options.
func isLoadName(name string) bool {
	switch name {
	case "CompileFlags", "CompilerOptions":
		return false
	case "Bind", "SpliceAndCompile":
		return true
	}
	return strings.HasPrefix(name, "Compile") || strings.HasPrefix(name, "Load")
}

// isLoad reports whether sel refers to a compile or load entry point: a
// function of a loadPackages package named per isLoadName, or a method so
// named on any receiver not a package, since its type is not resolved. A
// function of any other package is not one, whatever its name.
func isLoad(sel *ast.SelectorExpr, imports map[string]string) bool {
	if x, ok := sel.X.(*ast.Ident); ok {
		if p, ok := imports[x.Name]; ok {
			return loadPackages[p] && isLoadName(sel.Sel.Name)
		}
	}
	return isLoadName(sel.Sel.Name)
}

// loadRefs returns the selectors in f that refer to a compile or load
// entry point, called or not.
func loadRefs(f *ast.File, imports map[string]string) []*ast.SelectorExpr {
	var refs []*ast.SelectorExpr
	ast.Inspect(f, func(n ast.Node) bool {
		if sel, ok := n.(*ast.SelectorExpr); ok && isLoad(sel, imports) {
			refs = append(refs, sel)
		}
		return true
	})
	return refs
}

// callErrorsReleased counts the references in src to a compile or load
// entry point, in any form, called or not, and checks each "x, err :=
// y.Load(...)" or "x, err = y.Load(...)": the next statement must be "if
// err != nil" whose body calls releaseErrorOwner(err). References in other
// forms are counted but not checked. A dot import of a loadPackages
// package, whose entry points would be unqualified, is a problem.
func callErrorsReleased(name, src string) (n callCount, bad []string, err error) {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, name, src, 0)
	if err != nil {
		return callCount{}, nil, err
	}
	imports, dot := fileImports(f)
	for _, p := range dot {
		if loadPackages[p] {
			bad = append(bad, fmt.Sprintf("%s: dot import of %s hides its entry points", name, p))
		}
	}
	n.calls = len(loadRefs(f, imports))
	ast.Inspect(f, func(node ast.Node) bool {
		block, ok := node.(*ast.BlockStmt)
		if !ok {
			return true
		}
		for i, stmt := range block.List {
			as, ok := stmt.(*ast.AssignStmt)
			if !ok || len(as.Lhs) != 2 || len(as.Rhs) != 1 {
				continue
			}
			call, ok := as.Rhs[0].(*ast.CallExpr)
			if !ok {
				continue
			}
			if sel, ok := call.Fun.(*ast.SelectorExpr); !ok || !isLoad(sel, imports) {
				continue
			}
			n.checked++
			errv, _ := as.Lhs[1].(*ast.Ident)
			ok = errv != nil && i+1 < len(block.List) && releasesCallError(block.List[i+1], errv.Name)
			if !ok {
				bad = append(bad, fmt.Sprintf("%v: call error not passed to releaseErrorOwner", fset.Position(as.Pos())))
			}
		}
		return true
	})
	return n, bad, nil
}

// releasesCallError reports whether stmt is "if errv != nil { ... }" whose
// body calls releaseErrorOwner(errv).
func releasesCallError(stmt ast.Stmt, errv string) bool {
	is, ok := stmt.(*ast.IfStmt)
	if !ok || is.Init != nil {
		return false
	}
	cond, ok := is.Cond.(*ast.BinaryExpr)
	if !ok || cond.Op != token.NEQ {
		return false
	}
	x, xok := cond.X.(*ast.Ident)
	y, yok := cond.Y.(*ast.Ident)
	if !xok || !yok || x.Name != errv || y.Name != "nil" {
		return false
	}
	found := false
	ast.Inspect(is.Body, func(n ast.Node) bool {
		if c, isCall := n.(*ast.CallExpr); isCall && calleeName(c) == "releaseErrorOwner" && len(c.Args) == 1 {
			if id, isIdent := c.Args[0].(*ast.Ident); isIdent && id.Name == errv {
				found = true
			}
		}
		return !found
	})
	return found
}

// TestGenReleasesCompileError checks gen.go, which builds alone with
// "go run gen.go" and so cannot call releaseErrorOwner: its Compile error
// branch must find an io.Closer with errors.As, close it, and join a
// failure to the error.
func TestGenReleasesCompileError(t *testing.T) {
	src, err := os.ReadFile("gen.go")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(string(src), "//go:build ignore\n") {
		t.Fatal("gen.go is no longer a standalone build; this check assumes it is")
	}
	if bad, err := genReleaseFailures(string(src)); err != nil || len(bad) > 0 {
		t.Errorf("gen.go: %q, %v", bad, err)
	}

	const ok = `if err != nil {
		var c io.Closer
		if errors.As(err, &c) {
			if cerr := c.Close(); cerr != nil {
				err = errors.Join(err, cerr)
			}
		}
		return err
	}`
	for _, tt := range []struct {
		name, branch string
		bad          bool
	}{
		{"positive control", ok, false},
		{"branch taken on success", strings.Replace(ok, "err != nil {", "err == nil {", 1), true},
		{"closes another value", strings.Replace(ok, "c.Close()", "os.Stdout.Close()", 1), true},
		{"As on another error", strings.Replace(ok, "errors.As(err, &c)", "errors.As(other, &c)", 1), true},
		{"retry error dropped", strings.Replace(ok, "err = errors.Join(err, cerr)", "_ = cerr", 1), true},
		{"calls the program's helper", "if err != nil {\nreturn releaseErrorOwner(err)\n}", true},
		{"second compile, if initializer", ok + "\nif _, err := nvrtcx.Compile(src); err != nil {\nreturn err\n}", true},
		{"compile returned", ok + "\nreturn nvrtcx.Compile(src)", true},
		{"load, var declaration", ok + "\nvar m, lerr = mod.Load(dev)", true},
		{"entry point as a value", ok + "\ncompile := nvrtcx.CompileContext", true},
	} {
		src := "package p\nimport \"github.com/tmc/lib/nvidia/x/nvrtcx\"\nfunc f() error {\nmod, err := nvrtcx.Compile(src)\n" + tt.branch + "\n_ = mod\nreturn nil\n}\n"
		bad, err := genReleaseFailures(src)
		if err != nil || (len(bad) > 0) != tt.bad {
			t.Errorf("%s: %q, %v; want flagged %v", tt.name, bad, err, tt.bad)
		}
	}
}

// genReleaseFailures returns how src departs from: no use of
// releaseErrorOwner, no dot import of a loadPackages package, and exactly
// one reference to a compile or load entry point (see isLoad), called or
// not, which is "x, err := nvrtcx.Compile(...)" followed by "if err != nil"
// whose body passes err to errors.As with &c, calls c.Close(), and joins
// the Close error into err.
func genReleaseFailures(src string) ([]string, error) {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "gen.go", src, 0)
	if err != nil {
		return nil, err
	}
	var bad []string
	if mentions(f, "releaseErrorOwner") {
		bad = append(bad, "calls releaseErrorOwner, which its standalone build cannot see")
	}
	imports, dot := fileImports(f)
	for _, p := range dot {
		if loadPackages[p] {
			bad = append(bad, fmt.Sprintf("dot import of %s hides its entry points", p))
		}
	}
	calls, compiles, released := len(loadRefs(f, imports)), 0, 0
	ast.Inspect(f, func(n ast.Node) bool {
		block, ok := n.(*ast.BlockStmt)
		if !ok {
			return true
		}
		for i, stmt := range block.List {
			as, ok := stmt.(*ast.AssignStmt)
			if !ok || len(as.Lhs) != 2 || len(as.Rhs) != 1 {
				continue
			}
			call, ok := as.Rhs[0].(*ast.CallExpr)
			if !ok || refName(call.Fun, imports) != libX+"nvrtcx.Compile" {
				continue
			}
			compiles++
			errv, _ := as.Lhs[1].(*ast.Ident)
			if errv != nil && i+1 < len(block.List) && genBranchReleases(block.List[i+1], errv.Name) {
				released++
			}
		}
		return true
	})
	if calls != 1 || compiles != 1 || released != 1 {
		bad = append(bad, fmt.Sprintf("%d loads and compiles, %d checked compiles, %d release their error; want 1, 1 and 1", calls, compiles, released))
	}
	return bad, nil
}

// genBranchReleases reports whether stmt is "if errv != nil" whose body
// calls errors.As(errv, &c), calls c.Close(), and calls errors.Join with
// errv.
func genBranchReleases(stmt ast.Stmt, errv string) bool {
	is, ok := stmt.(*ast.IfStmt)
	if !ok || is.Init != nil {
		return false
	}
	cond, ok := is.Cond.(*ast.BinaryExpr)
	if !ok || cond.Op != token.NEQ {
		return false
	}
	if x, ok := cond.X.(*ast.Ident); !ok || x.Name != errv {
		return false
	}
	if y, ok := cond.Y.(*ast.Ident); !ok || y.Name != "nil" {
		return false
	}
	target := ""
	ast.Inspect(is.Body, func(n ast.Node) bool {
		c, ok := n.(*ast.CallExpr)
		if !ok || calleeName(c) != "errors.As" || len(c.Args) != 2 {
			return true
		}
		e, eok := c.Args[0].(*ast.Ident)
		u, uok := c.Args[1].(*ast.UnaryExpr)
		if eok && uok && e.Name == errv && u.Op == token.AND {
			if id, ok := u.X.(*ast.Ident); ok {
				target = id.Name
			}
		}
		return true
	})
	if target == "" {
		return false
	}
	closes, joins := false, false
	ast.Inspect(is.Body, func(n ast.Node) bool {
		c, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		if isRelease(c, target) {
			closes = true
		}
		if calleeName(c) == "errors.Join" && len(c.Args) > 0 && mentions(c.Args[0], errv) {
			joins = true
		}
		return true
	})
	return closes && joins
}
