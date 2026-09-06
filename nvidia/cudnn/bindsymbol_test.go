package cudnn

import (
	"reflect"
	"runtime"
	"testing"

	"github.com/ebitengine/purego"
)

// systemSymbol returns the address of a symbol that is certain to exist in the
// running process, for use as a binding target. The symbol is never called; it
// only has to be a valid non-zero code address, because purego decides whether
// it can bind a signature before it ever looks at the address.
func systemSymbol(t *testing.T) uintptr {
	t.Helper()
	var name string
	switch runtime.GOOS {
	case "darwin":
		name = "libSystem.B.dylib"
	case "linux":
		name = "libc.so.6"
	default:
		t.Skipf("no system library to bind against on %s", runtime.GOOS)
	}
	lib, err := purego.Dlopen(name, purego.RTLD_LAZY|purego.RTLD_GLOBAL)
	if err != nil || lib == 0 {
		t.Skipf("dlopen %s: %v", name, err)
	}
	sym, err := purego.Dlsym(lib, "abs")
	if err != nil || sym == 0 {
		t.Skipf("dlsym abs: %v", err)
	}
	return sym
}

// registerPanics reports whether purego.RegisterFunc panics when asked to bind
// a fresh function pointer of type ty. This is the unguarded behaviour that
// register used to have.
func registerPanics(ty reflect.Type, sym uintptr) (panicked bool) {
	defer func() {
		if recover() != nil {
			panicked = true
		}
	}()
	purego.RegisterFunc(reflect.New(ty).Interface(), sym)
	return false
}

// TestBindSymbolContainsRegisterFuncPanics is the regression test for the
// cudnnNormalizationBackward crash: purego cannot build a trampoline for a
// 33-argument signature on a 64-bit target (its limit is 32; on ppc64le it is
// 15), and the unguarded purego.RegisterFunc call that register used to make
// panicked during package initialization, killing the process on any host
// where libcudnn resolved the symbol.
//
// The test asserts the equivalence directly rather than hardcoding a
// threshold, because the threshold is architecture-dependent.
func TestBindSymbolContainsRegisterFuncPanics(t *testing.T) {
	sym := systemSymbol(t)

	uintptrType := reflect.TypeOf(uintptr(0))
	sawPanic := false
	for n := 1; n <= 40; n++ {
		in := make([]reflect.Type, n)
		for i := range in {
			in[i] = uintptrType
		}
		ty := reflect.FuncOf(in, []reflect.Type{uintptrType}, false)

		want := !registerPanics(ty, sym)
		if !want {
			sawPanic = true
		}
		// bindSymbol must never panic, whatever purego does.
		got := bindSymbol(reflect.New(ty).Interface(), sym)
		if got != want {
			t.Errorf("%d args: bindSymbol = %v, purego.RegisterFunc succeeds = %v", n, got, want)
		}
	}
	if !sawPanic {
		t.Fatalf("purego bound every arity up to 40 on %s/%s; the test proves nothing", runtime.GOOS, runtime.GOARCH)
	}
}

// TestBindNormalizationBackward exercises the exact signature that caused the
// crash. cudnnNormalizationBackward takes 33 parameters, one over purego's
// 64-bit ceiling, so it is the only entry point in the module that cannot be
// bound on any supported architecture.
func TestBindNormalizationBackward(t *testing.T) {
	sym := systemSymbol(t)

	ty := reflect.TypeOf(fnnormalizationBackward)
	if got := ty.NumIn(); got != 33 {
		t.Fatalf("cudnnNormalizationBackward has %d parameters, want 33", got)
	}
	if !registerPanics(ty, sym) {
		t.Skipf("purego binds 33 arguments on %s/%s", runtime.GOOS, runtime.GOARCH)
	}
	// The unguarded call panics here; bindSymbol must report failure instead.
	if bindSymbol(reflect.New(ty).Interface(), sym) {
		t.Error("bindSymbol reported success for a signature purego cannot bind")
	}
}

// TestRegisterLeavesUnbindableSymbolUnbound checks the contract register
// documents: an unbindable symbol is left out of the presence map, so calling
// it reports *SymbolError rather than crashing.
func TestRegisterLeavesUnbindableSymbolUnbound(t *testing.T) {
	if Has("cudnnNormalizationBackward") {
		t.Skip("libcudnn is loaded and bound the symbol")
	}
	if fnnormalizationBackward != nil {
		t.Fatal("cudnnNormalizationBackward is bound but not recorded as present")
	}
	if err := checkLibSym("cudnnNormalizationBackward"); err == nil {
		t.Error("checkLibSym returned nil for an unbound symbol")
	}
}
