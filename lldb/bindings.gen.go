// Package lldb provides low-level Go bindings for the liblldb bridge.
//
// The bridge is compiled against one liblldb installation. Handles own C++ SB API values and must be released with their matching Delete function. Call LLDBSBDebuggerDestroy before deleting a debugger created with LLDBSBDebuggerCreateWithInitFiles.
package lldb

import (
	"fmt"
	"github.com/ebitengine/purego"
	"os"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"unsafe"
)

// packageName prefixes errors reported by this package.
const packageName = "lldb"

var (
	loadMu        sync.Mutex
	loadAttempted bool
	lib           uintptr
	libErr        error
	libPath       string

	// libReady is the fast path: once the library is open and every symbol
	// bound, calls skip loadMu entirely.
	libReady atomic.Bool

	// present records which symbols resolved. It is written under loadMu
	// before libReady is set, and only read afterwards.
	present = make(map[string]bool)

	bindingsDisabled = os.Getenv("CODEGEN_SKIP_BINDINGS") != ""
)

// SymbolError reports that a symbol is absent from the loaded library, which
// happens when the installed version predates it.
type SymbolError struct {
	Symbol string
}

func (e *SymbolError) Error() string {
	return packageName + ": symbol " + e.Symbol + " unavailable"
}

// Has reports whether a symbol resolved in the loaded library, loading it
// first if necessary. It reports false when the library cannot be loaded.
// Use it to gate on capabilities before calling into them.
func Has(symbol string) bool {
	if err := ensureLoaded(); err != nil {
		return false
	}
	return present[symbol]
}

// bind resolves one symbol, leaving its pointer nil when the library does not
// export it. Registering blindly would panic inside purego and take the whole
// package down at load time, so a library that is merely older than these
// bindings must instead fail at the call to the one symbol it lacks.
func bind(fptr any, symbol string) {
	if _, err := purego.Dlsym(lib, symbol); err != nil {
		return
	}
	present[symbol] = true
	purego.RegisterLibFunc(fptr, lib, symbol)
}

// checkSym reports whether a symbol can be called: the library loads and the
// symbol resolved.
func checkSym(symbol string) error {
	if err := ensureLoaded(); err != nil {
		return err
	}
	if !present[symbol] {
		return &SymbolError{Symbol: symbol}
	}
	return nil
}

// LibraryPathEnv names the environment variable that overrides where the
// bridge library is loaded from.
const LibraryPathEnv = "LIBLLDB_BRIDGE_PATH"

// Load opens the bridge library from path, or from the default search list
// when path is empty, and binds every symbol.
//
// Loading is otherwise lazy: the first call into this package loads the
// library. Call Load first only to choose the library or to surface a load
// failure at a point of your choosing.
//
// Load is idempotent. Calling it again with the same path, or with an empty
// path, returns nil; calling it with a different path after a successful load
// returns an error rather than swapping the library underneath live handles.
// After a failed load, Load retries.
func Load(path string) error {
	loadMu.Lock()
	defer loadMu.Unlock()
	if lib != 0 {
		if path != "" && path != libPath {
			return fmt.Errorf("%s: already loaded from %q, cannot load %q", packageName, libPath, path)
		}
		return nil
	}
	loadAttempted = false
	return loadLocked(path)
}

// Loaded reports whether the library is open. It never triggers a load.
func Loaded() bool { return libReady.Load() }

// Path reports where the library was loaded from, or "" if it is not loaded.
func Path() string {
	loadMu.Lock()
	defer loadMu.Unlock()
	return libPath
}

// Close unloads the library and clears every bound function pointer.
//
// The pointers are package-level state, so Close must not run concurrently
// with any other call into this package. It exists for test teardown, not for
// reclaiming a library still in use.
func Close() error {
	loadMu.Lock()
	defer loadMu.Unlock()
	if lib == 0 {
		return nil
	}
	err := purego.Dlclose(lib)
	lib, libErr, libPath, loadAttempted = 0, nil, "", false
	libReady.Store(false)
	present = make(map[string]bool)
	clearFuncs()
	if err != nil {
		return fmt.Errorf("%s: close: %w", packageName, err)
	}
	return nil
}

// ensureLoaded opens the library on first use. Wrappers that can report an
// error call it and return what it gives them.
func ensureLoaded() error {
	if libReady.Load() {
		return nil
	}
	loadMu.Lock()
	defer loadMu.Unlock()
	return loadLocked("")
}

// mustSym is checkSym for wrappers with no error to return: the raw binding
// layer mirrors the C ABI, which has no channel for a load failure. It panics
// with an error naming the library or the symbol, rather than calling through
// a nil pointer with no explanation. The idiomatic layer checks first and
// never reaches this.
func mustSym(symbol string) {
	if err := checkSym(symbol); err != nil {
		panic(err)
	}
}

// loadLocked performs the one-time load; callers must hold loadMu.
func loadLocked(path string) error {
	if loadAttempted {
		return libErr
	}
	loadAttempted = true
	if bindingsDisabled {
		libErr = fmt.Errorf("%s: bindings disabled by CODEGEN_SKIP_BINDINGS", packageName)
		return libErr
	}
	names := []string{path}
	if path == "" {
		if env := os.Getenv(LibraryPathEnv); env != "" {
			names = []string{env}
		} else {
			names = libraryNames()
		}
	}
	var firstErr error
	for _, name := range names {
		h, err := purego.Dlopen(name, purego.RTLD_NOW|purego.RTLD_GLOBAL)
		if err == nil {
			lib, libPath = h, name
			registerFuncs()
			libReady.Store(true)
			return nil
		}
		if firstErr == nil {
			firstErr = err
		}
	}
	libErr = fmt.Errorf("%s: open %s: %w", packageName, "liblldb_bridge", firstErr)
	return libErr
}

// libraryNames lists the platform's shared-library spellings to try.
func libraryNames() []string {
	const base = "liblldb_bridge"
	switch runtime.GOOS {
	case "darwin":
		return []string{base + ".dylib", base + ".so"}
	case "windows":
		return []string{strings.TrimPrefix(base, "lib") + ".dll", base + ".dll"}
	default:
		return []string{base + ".so"}
	}
}

type LLDBSBBreakpointRef = uintptr

type LLDBSBDebuggerRef = uintptr

type LLDBSBErrorRef = uintptr

type LLDBSBFrameRef = uintptr

type LLDBSBProcessRef = uintptr

type LLDBSBTargetRef = uintptr

type LLDBSBThreadRef = uintptr

type LLDBSBValueRef = uintptr

var LLDBLastError_fn func() string

// LLDBLastError Returns the message of the exception thrown by the most recent bridge call on this thread, or NULL.
func LLDBLastError() string {
	mustSym("LLDBLastError")
	return LLDBLastError_fn()
}

var LLDBSBErrorNew_fn func() LLDBSBErrorRef

func LLDBSBErrorNew() LLDBSBErrorRef {
	mustSym("LLDBSBErrorNew")
	return LLDBSBErrorNew_fn()
}

var LLDBSBErrorDelete_fn func(LLDBSBErrorRef)

func LLDBSBErrorDelete(self LLDBSBErrorRef) {
	mustSym("LLDBSBErrorDelete")
	LLDBSBErrorDelete_fn(self)
}

var LLDBSBErrorClear_fn func(LLDBSBErrorRef)

func LLDBSBErrorClear(self LLDBSBErrorRef) {
	mustSym("LLDBSBErrorClear")
	LLDBSBErrorClear_fn(self)
}

var LLDBSBErrorFail_fn func(LLDBSBErrorRef) bool

func LLDBSBErrorFail(self LLDBSBErrorRef) bool {
	mustSym("LLDBSBErrorFail")
	return LLDBSBErrorFail_fn(self)
}

var LLDBSBErrorSuccess_fn func(LLDBSBErrorRef) bool

func LLDBSBErrorSuccess(self LLDBSBErrorRef) bool {
	mustSym("LLDBSBErrorSuccess")
	return LLDBSBErrorSuccess_fn(self)
}

var LLDBSBErrorIsValid_fn func(LLDBSBErrorRef) bool

func LLDBSBErrorIsValid(self LLDBSBErrorRef) bool {
	mustSym("LLDBSBErrorIsValid")
	return LLDBSBErrorIsValid_fn(self)
}

var LLDBSBErrorGetError_fn func(LLDBSBErrorRef) uint32

func LLDBSBErrorGetError(self LLDBSBErrorRef) uint32 {
	mustSym("LLDBSBErrorGetError")
	return LLDBSBErrorGetError_fn(self)
}

var LLDBSBErrorGetCString_fn func(LLDBSBErrorRef) string

func LLDBSBErrorGetCString(self LLDBSBErrorRef) string {
	mustSym("LLDBSBErrorGetCString")
	return LLDBSBErrorGetCString_fn(self)
}

var LLDBSBDebuggerDelete_fn func(LLDBSBDebuggerRef)

func LLDBSBDebuggerDelete(self LLDBSBDebuggerRef) {
	mustSym("LLDBSBDebuggerDelete")
	LLDBSBDebuggerDelete_fn(self)
}

var LLDBSBDebuggerInitialize_fn func()

func LLDBSBDebuggerInitialize() {
	mustSym("LLDBSBDebuggerInitialize")
	LLDBSBDebuggerInitialize_fn()
}

var LLDBSBDebuggerTerminate_fn func()

func LLDBSBDebuggerTerminate() {
	mustSym("LLDBSBDebuggerTerminate")
	LLDBSBDebuggerTerminate_fn()
}

var LLDBSBDebuggerCreateWithInitFiles_fn func(bool) LLDBSBDebuggerRef

func LLDBSBDebuggerCreateWithInitFiles(sourceInitFiles bool) LLDBSBDebuggerRef {
	mustSym("LLDBSBDebuggerCreateWithInitFiles")
	return LLDBSBDebuggerCreateWithInitFiles_fn(sourceInitFiles)
}

var LLDBSBDebuggerDestroy_fn func(LLDBSBDebuggerRef)

func LLDBSBDebuggerDestroy(debugger LLDBSBDebuggerRef) {
	mustSym("LLDBSBDebuggerDestroy")
	LLDBSBDebuggerDestroy_fn(debugger)
}

var LLDBSBDebuggerIsValid_fn func(LLDBSBDebuggerRef) bool

func LLDBSBDebuggerIsValid(self LLDBSBDebuggerRef) bool {
	mustSym("LLDBSBDebuggerIsValid")
	return LLDBSBDebuggerIsValid_fn(self)
}

var LLDBSBDebuggerClear_fn func(LLDBSBDebuggerRef)

func LLDBSBDebuggerClear(self LLDBSBDebuggerRef) {
	mustSym("LLDBSBDebuggerClear")
	LLDBSBDebuggerClear_fn(self)
}

var LLDBSBDebuggerSetAsync_fn func(LLDBSBDebuggerRef, bool)

func LLDBSBDebuggerSetAsync(self LLDBSBDebuggerRef, enabled bool) {
	mustSym("LLDBSBDebuggerSetAsync")
	LLDBSBDebuggerSetAsync_fn(self, enabled)
}

var LLDBSBDebuggerGetAsync_fn func(LLDBSBDebuggerRef) bool

func LLDBSBDebuggerGetAsync(self LLDBSBDebuggerRef) bool {
	mustSym("LLDBSBDebuggerGetAsync")
	return LLDBSBDebuggerGetAsync_fn(self)
}

var LLDBSBDebuggerSkipLLDBInitFiles_fn func(LLDBSBDebuggerRef, bool)

func LLDBSBDebuggerSkipLLDBInitFiles(self LLDBSBDebuggerRef, skip bool) {
	mustSym("LLDBSBDebuggerSkipLLDBInitFiles")
	LLDBSBDebuggerSkipLLDBInitFiles_fn(self, skip)
}

var LLDBSBDebuggerSkipAppInitFiles_fn func(LLDBSBDebuggerRef, bool)

func LLDBSBDebuggerSkipAppInitFiles(self LLDBSBDebuggerRef, skip bool) {
	mustSym("LLDBSBDebuggerSkipAppInitFiles")
	LLDBSBDebuggerSkipAppInitFiles_fn(self, skip)
}

var LLDBSBDebuggerHandleCommand_fn func(LLDBSBDebuggerRef, string)

func LLDBSBDebuggerHandleCommand(self LLDBSBDebuggerRef, command string) {
	mustSym("LLDBSBDebuggerHandleCommand")
	LLDBSBDebuggerHandleCommand_fn(self, command)
}

var LLDBSBDebuggerCreateTargetSimple_fn func(LLDBSBDebuggerRef, string) LLDBSBTargetRef

func LLDBSBDebuggerCreateTargetSimple(self LLDBSBDebuggerRef, filename string) LLDBSBTargetRef {
	mustSym("LLDBSBDebuggerCreateTargetSimple")
	return LLDBSBDebuggerCreateTargetSimple_fn(self, filename)
}

var LLDBSBDebuggerGetSelectedTarget_fn func(LLDBSBDebuggerRef) LLDBSBTargetRef

func LLDBSBDebuggerGetSelectedTarget(self LLDBSBDebuggerRef) LLDBSBTargetRef {
	mustSym("LLDBSBDebuggerGetSelectedTarget")
	return LLDBSBDebuggerGetSelectedTarget_fn(self)
}

var LLDBSBDebuggerGetNumTargets_fn func(LLDBSBDebuggerRef) uint32

func LLDBSBDebuggerGetNumTargets(self LLDBSBDebuggerRef) uint32 {
	mustSym("LLDBSBDebuggerGetNumTargets")
	return LLDBSBDebuggerGetNumTargets_fn(self)
}

var LLDBSBDebuggerGetVersionString_fn func() string

func LLDBSBDebuggerGetVersionString() string {
	mustSym("LLDBSBDebuggerGetVersionString")
	return LLDBSBDebuggerGetVersionString_fn()
}

var LLDBSBTargetDelete_fn func(LLDBSBTargetRef)

func LLDBSBTargetDelete(self LLDBSBTargetRef) {
	mustSym("LLDBSBTargetDelete")
	LLDBSBTargetDelete_fn(self)
}

var LLDBSBTargetIsValid_fn func(LLDBSBTargetRef) bool

func LLDBSBTargetIsValid(self LLDBSBTargetRef) bool {
	mustSym("LLDBSBTargetIsValid")
	return LLDBSBTargetIsValid_fn(self)
}

var LLDBSBTargetGetProcess_fn func(LLDBSBTargetRef) LLDBSBProcessRef

func LLDBSBTargetGetProcess(self LLDBSBTargetRef) LLDBSBProcessRef {
	mustSym("LLDBSBTargetGetProcess")
	return LLDBSBTargetGetProcess_fn(self)
}

var LLDBSBTargetGetNumModules_fn func(LLDBSBTargetRef) uint32

func LLDBSBTargetGetNumModules(self LLDBSBTargetRef) uint32 {
	mustSym("LLDBSBTargetGetNumModules")
	return LLDBSBTargetGetNumModules_fn(self)
}

var LLDBSBTargetDeleteAllBreakpoints_fn func(LLDBSBTargetRef)

func LLDBSBTargetDeleteAllBreakpoints(self LLDBSBTargetRef) {
	mustSym("LLDBSBTargetDeleteAllBreakpoints")
	LLDBSBTargetDeleteAllBreakpoints_fn(self)
}

var LLDBSBTargetBreakpointCreateByNameSimple_fn func(LLDBSBTargetRef, string) LLDBSBBreakpointRef

func LLDBSBTargetBreakpointCreateByNameSimple(self LLDBSBTargetRef, symbol string) LLDBSBBreakpointRef {
	mustSym("LLDBSBTargetBreakpointCreateByNameSimple")
	return LLDBSBTargetBreakpointCreateByNameSimple_fn(self, symbol)
}

var LLDBSBTargetBreakpointCreateByLocationSimple_fn func(LLDBSBTargetRef, string, uint32) LLDBSBBreakpointRef

func LLDBSBTargetBreakpointCreateByLocationSimple(self LLDBSBTargetRef, file string, line uint32) LLDBSBBreakpointRef {
	mustSym("LLDBSBTargetBreakpointCreateByLocationSimple")
	return LLDBSBTargetBreakpointCreateByLocationSimple_fn(self, file, line)
}

var LLDBSBProcessDelete_fn func(LLDBSBProcessRef)

func LLDBSBProcessDelete(self LLDBSBProcessRef) {
	mustSym("LLDBSBProcessDelete")
	LLDBSBProcessDelete_fn(self)
}

var LLDBSBProcessIsValid_fn func(LLDBSBProcessRef) bool

func LLDBSBProcessIsValid(self LLDBSBProcessRef) bool {
	mustSym("LLDBSBProcessIsValid")
	return LLDBSBProcessIsValid_fn(self)
}

var LLDBSBProcessGetProcessID_fn func(LLDBSBProcessRef) uint64

func LLDBSBProcessGetProcessID(self LLDBSBProcessRef) uint64 {
	mustSym("LLDBSBProcessGetProcessID")
	return LLDBSBProcessGetProcessID_fn(self)
}

var LLDBSBProcessGetState_fn func(LLDBSBProcessRef) int32

func LLDBSBProcessGetState(self LLDBSBProcessRef) int32 {
	mustSym("LLDBSBProcessGetState")
	return LLDBSBProcessGetState_fn(self)
}

var LLDBSBProcessGetNumThreads_fn func(LLDBSBProcessRef) uint32

func LLDBSBProcessGetNumThreads(self LLDBSBProcessRef) uint32 {
	mustSym("LLDBSBProcessGetNumThreads")
	return LLDBSBProcessGetNumThreads_fn(self)
}

var LLDBSBProcessGetThreadAtIndex_fn func(LLDBSBProcessRef, uint32) LLDBSBThreadRef

func LLDBSBProcessGetThreadAtIndex(self LLDBSBProcessRef, index uint32) LLDBSBThreadRef {
	mustSym("LLDBSBProcessGetThreadAtIndex")
	return LLDBSBProcessGetThreadAtIndex_fn(self, index)
}

var LLDBSBProcessGetSelectedThread_fn func(LLDBSBProcessRef) LLDBSBThreadRef

func LLDBSBProcessGetSelectedThread(self LLDBSBProcessRef) LLDBSBThreadRef {
	mustSym("LLDBSBProcessGetSelectedThread")
	return LLDBSBProcessGetSelectedThread_fn(self)
}

var LLDBSBProcessSetSelectedThreadByID_fn func(LLDBSBProcessRef, uint64) bool

func LLDBSBProcessSetSelectedThreadByID(self LLDBSBProcessRef, threadID uint64) bool {
	mustSym("LLDBSBProcessSetSelectedThreadByID")
	return LLDBSBProcessSetSelectedThreadByID_fn(self, threadID)
}

var LLDBSBProcessContinue_fn func(LLDBSBProcessRef) LLDBSBErrorRef

func LLDBSBProcessContinue(self LLDBSBProcessRef) LLDBSBErrorRef {
	mustSym("LLDBSBProcessContinue")
	return LLDBSBProcessContinue_fn(self)
}

var LLDBSBProcessStop_fn func(LLDBSBProcessRef) LLDBSBErrorRef

func LLDBSBProcessStop(self LLDBSBProcessRef) LLDBSBErrorRef {
	mustSym("LLDBSBProcessStop")
	return LLDBSBProcessStop_fn(self)
}

var LLDBSBProcessKill_fn func(LLDBSBProcessRef) LLDBSBErrorRef

func LLDBSBProcessKill(self LLDBSBProcessRef) LLDBSBErrorRef {
	mustSym("LLDBSBProcessKill")
	return LLDBSBProcessKill_fn(self)
}

var LLDBSBProcessDetachSimple_fn func(LLDBSBProcessRef) LLDBSBErrorRef

func LLDBSBProcessDetachSimple(self LLDBSBProcessRef) LLDBSBErrorRef {
	mustSym("LLDBSBProcessDetachSimple")
	return LLDBSBProcessDetachSimple_fn(self)
}

var LLDBSBProcessSignal_fn func(LLDBSBProcessRef, int32) LLDBSBErrorRef

func LLDBSBProcessSignal(self LLDBSBProcessRef, signal int32) LLDBSBErrorRef {
	mustSym("LLDBSBProcessSignal")
	return LLDBSBProcessSignal_fn(self, signal)
}

var LLDBSBProcessPutSTDIN_fn func(LLDBSBProcessRef, uintptr, uint) uint

func LLDBSBProcessPutSTDIN(self LLDBSBProcessRef, data []byte) uint {
	var dataPtr uintptr
	if len(data) != 0 {
		dataPtr = uintptr(unsafe.Pointer(&data[0]))
	}
	mustSym("LLDBSBProcessPutSTDIN")
	ret := LLDBSBProcessPutSTDIN_fn(self, dataPtr, uint(len(data)))
	runtime.KeepAlive(data)
	return ret
}

var LLDBSBProcessGetSTDOUT_fn func(LLDBSBProcessRef, uintptr, uint) uint

func LLDBSBProcessGetSTDOUT(self LLDBSBProcessRef, data []byte) uint {
	var dataPtr uintptr
	if len(data) != 0 {
		dataPtr = uintptr(unsafe.Pointer(&data[0]))
	}
	mustSym("LLDBSBProcessGetSTDOUT")
	ret := LLDBSBProcessGetSTDOUT_fn(self, dataPtr, uint(len(data)))
	runtime.KeepAlive(data)
	return ret
}

var LLDBSBProcessGetSTDERR_fn func(LLDBSBProcessRef, uintptr, uint) uint

func LLDBSBProcessGetSTDERR(self LLDBSBProcessRef, data []byte) uint {
	var dataPtr uintptr
	if len(data) != 0 {
		dataPtr = uintptr(unsafe.Pointer(&data[0]))
	}
	mustSym("LLDBSBProcessGetSTDERR")
	ret := LLDBSBProcessGetSTDERR_fn(self, dataPtr, uint(len(data)))
	runtime.KeepAlive(data)
	return ret
}

var LLDBSBProcessReadMemory_fn func(LLDBSBProcessRef, uint64, uintptr, uint, LLDBSBErrorRef) uint

func LLDBSBProcessReadMemory(self LLDBSBProcessRef, address uint64, data []byte, error LLDBSBErrorRef) uint {
	var dataPtr uintptr
	if len(data) != 0 {
		dataPtr = uintptr(unsafe.Pointer(&data[0]))
	}
	mustSym("LLDBSBProcessReadMemory")
	ret := LLDBSBProcessReadMemory_fn(self, address, dataPtr, uint(len(data)), error)
	runtime.KeepAlive(data)
	return ret
}

var LLDBSBProcessWriteMemory_fn func(LLDBSBProcessRef, uint64, uintptr, uint, LLDBSBErrorRef) uint

func LLDBSBProcessWriteMemory(self LLDBSBProcessRef, address uint64, data []byte, error LLDBSBErrorRef) uint {
	var dataPtr uintptr
	if len(data) != 0 {
		dataPtr = uintptr(unsafe.Pointer(&data[0]))
	}
	mustSym("LLDBSBProcessWriteMemory")
	ret := LLDBSBProcessWriteMemory_fn(self, address, dataPtr, uint(len(data)), error)
	runtime.KeepAlive(data)
	return ret
}

var LLDBSBThreadDelete_fn func(LLDBSBThreadRef)

func LLDBSBThreadDelete(self LLDBSBThreadRef) {
	mustSym("LLDBSBThreadDelete")
	LLDBSBThreadDelete_fn(self)
}

var LLDBSBThreadIsValid_fn func(LLDBSBThreadRef) bool

func LLDBSBThreadIsValid(self LLDBSBThreadRef) bool {
	mustSym("LLDBSBThreadIsValid")
	return LLDBSBThreadIsValid_fn(self)
}

var LLDBSBThreadGetThreadID_fn func(LLDBSBThreadRef) uint64

func LLDBSBThreadGetThreadID(self LLDBSBThreadRef) uint64 {
	mustSym("LLDBSBThreadGetThreadID")
	return LLDBSBThreadGetThreadID_fn(self)
}

var LLDBSBThreadGetIndexID_fn func(LLDBSBThreadRef) uint32

func LLDBSBThreadGetIndexID(self LLDBSBThreadRef) uint32 {
	mustSym("LLDBSBThreadGetIndexID")
	return LLDBSBThreadGetIndexID_fn(self)
}

var LLDBSBThreadGetNumFrames_fn func(LLDBSBThreadRef) uint32

func LLDBSBThreadGetNumFrames(self LLDBSBThreadRef) uint32 {
	mustSym("LLDBSBThreadGetNumFrames")
	return LLDBSBThreadGetNumFrames_fn(self)
}

var LLDBSBThreadGetFrameAtIndex_fn func(LLDBSBThreadRef, uint32) LLDBSBFrameRef

func LLDBSBThreadGetFrameAtIndex(self LLDBSBThreadRef, index uint32) LLDBSBFrameRef {
	mustSym("LLDBSBThreadGetFrameAtIndex")
	return LLDBSBThreadGetFrameAtIndex_fn(self, index)
}

var LLDBSBThreadGetStopReason_fn func(LLDBSBThreadRef) int32

func LLDBSBThreadGetStopReason(self LLDBSBThreadRef) int32 {
	mustSym("LLDBSBThreadGetStopReason")
	return LLDBSBThreadGetStopReason_fn(self)
}

var LLDBSBFrameDelete_fn func(LLDBSBFrameRef)

func LLDBSBFrameDelete(self LLDBSBFrameRef) {
	mustSym("LLDBSBFrameDelete")
	LLDBSBFrameDelete_fn(self)
}

var LLDBSBFrameIsValid_fn func(LLDBSBFrameRef) bool

func LLDBSBFrameIsValid(self LLDBSBFrameRef) bool {
	mustSym("LLDBSBFrameIsValid")
	return LLDBSBFrameIsValid_fn(self)
}

var LLDBSBFrameGetFunctionName_fn func(LLDBSBFrameRef) string

func LLDBSBFrameGetFunctionName(self LLDBSBFrameRef) string {
	mustSym("LLDBSBFrameGetFunctionName")
	return LLDBSBFrameGetFunctionName_fn(self)
}

var LLDBSBFrameGetDisplayFunctionName_fn func(LLDBSBFrameRef) string

func LLDBSBFrameGetDisplayFunctionName(self LLDBSBFrameRef) string {
	mustSym("LLDBSBFrameGetDisplayFunctionName")
	return LLDBSBFrameGetDisplayFunctionName_fn(self)
}

var LLDBSBFrameGetPC_fn func(LLDBSBFrameRef) uint64

func LLDBSBFrameGetPC(self LLDBSBFrameRef) uint64 {
	mustSym("LLDBSBFrameGetPC")
	return LLDBSBFrameGetPC_fn(self)
}

var LLDBSBFrameEvaluateExpressionSimple_fn func(LLDBSBFrameRef, string) LLDBSBValueRef

func LLDBSBFrameEvaluateExpressionSimple(self LLDBSBFrameRef, expression string) LLDBSBValueRef {
	mustSym("LLDBSBFrameEvaluateExpressionSimple")
	return LLDBSBFrameEvaluateExpressionSimple_fn(self, expression)
}

var LLDBSBValueDelete_fn func(LLDBSBValueRef)

func LLDBSBValueDelete(self LLDBSBValueRef) {
	mustSym("LLDBSBValueDelete")
	LLDBSBValueDelete_fn(self)
}

var LLDBSBValueIsValid_fn func(LLDBSBValueRef) bool

func LLDBSBValueIsValid(self LLDBSBValueRef) bool {
	mustSym("LLDBSBValueIsValid")
	return LLDBSBValueIsValid_fn(self)
}

var LLDBSBValueGetName_fn func(LLDBSBValueRef) string

func LLDBSBValueGetName(self LLDBSBValueRef) string {
	mustSym("LLDBSBValueGetName")
	return LLDBSBValueGetName_fn(self)
}

var LLDBSBValueGetValue_fn func(LLDBSBValueRef) string

func LLDBSBValueGetValue(self LLDBSBValueRef) string {
	mustSym("LLDBSBValueGetValue")
	return LLDBSBValueGetValue_fn(self)
}

var LLDBSBValueGetSummary_fn func(LLDBSBValueRef) string

func LLDBSBValueGetSummary(self LLDBSBValueRef) string {
	mustSym("LLDBSBValueGetSummary")
	return LLDBSBValueGetSummary_fn(self)
}

var LLDBSBValueGetNumChildren_fn func(LLDBSBValueRef) uint32

func LLDBSBValueGetNumChildren(self LLDBSBValueRef) uint32 {
	mustSym("LLDBSBValueGetNumChildren")
	return LLDBSBValueGetNumChildren_fn(self)
}

var LLDBSBValueGetChildAtIndex_fn func(LLDBSBValueRef, uint32) LLDBSBValueRef

func LLDBSBValueGetChildAtIndex(self LLDBSBValueRef, index uint32) LLDBSBValueRef {
	mustSym("LLDBSBValueGetChildAtIndex")
	return LLDBSBValueGetChildAtIndex_fn(self, index)
}

var LLDBSBBreakpointDelete_fn func(LLDBSBBreakpointRef)

func LLDBSBBreakpointDelete(self LLDBSBBreakpointRef) {
	mustSym("LLDBSBBreakpointDelete")
	LLDBSBBreakpointDelete_fn(self)
}

var LLDBSBBreakpointIsValid_fn func(LLDBSBBreakpointRef) bool

func LLDBSBBreakpointIsValid(self LLDBSBBreakpointRef) bool {
	mustSym("LLDBSBBreakpointIsValid")
	return LLDBSBBreakpointIsValid_fn(self)
}

var LLDBSBBreakpointGetID_fn func(LLDBSBBreakpointRef) uint32

func LLDBSBBreakpointGetID(self LLDBSBBreakpointRef) uint32 {
	mustSym("LLDBSBBreakpointGetID")
	return LLDBSBBreakpointGetID_fn(self)
}

var LLDBSBBreakpointSetEnabled_fn func(LLDBSBBreakpointRef, bool)

func LLDBSBBreakpointSetEnabled(self LLDBSBBreakpointRef, enabled bool) {
	mustSym("LLDBSBBreakpointSetEnabled")
	LLDBSBBreakpointSetEnabled_fn(self, enabled)
}

var LLDBSBBreakpointIsEnabled_fn func(LLDBSBBreakpointRef) bool

func LLDBSBBreakpointIsEnabled(self LLDBSBBreakpointRef) bool {
	mustSym("LLDBSBBreakpointIsEnabled")
	return LLDBSBBreakpointIsEnabled_fn(self)
}

// registerFuncs binds every symbol. It runs once, under loadMu, before
// libReady is set.
func registerFuncs() {
	bind(&LLDBLastError_fn, "LLDBLastError")
	bind(&LLDBSBErrorNew_fn, "LLDBSBErrorNew")
	bind(&LLDBSBErrorDelete_fn, "LLDBSBErrorDelete")
	bind(&LLDBSBErrorClear_fn, "LLDBSBErrorClear")
	bind(&LLDBSBErrorFail_fn, "LLDBSBErrorFail")
	bind(&LLDBSBErrorSuccess_fn, "LLDBSBErrorSuccess")
	bind(&LLDBSBErrorIsValid_fn, "LLDBSBErrorIsValid")
	bind(&LLDBSBErrorGetError_fn, "LLDBSBErrorGetError")
	bind(&LLDBSBErrorGetCString_fn, "LLDBSBErrorGetCString")
	bind(&LLDBSBDebuggerDelete_fn, "LLDBSBDebuggerDelete")
	bind(&LLDBSBDebuggerInitialize_fn, "LLDBSBDebuggerInitialize")
	bind(&LLDBSBDebuggerTerminate_fn, "LLDBSBDebuggerTerminate")
	bind(&LLDBSBDebuggerCreateWithInitFiles_fn, "LLDBSBDebuggerCreateWithInitFiles")
	bind(&LLDBSBDebuggerDestroy_fn, "LLDBSBDebuggerDestroy")
	bind(&LLDBSBDebuggerIsValid_fn, "LLDBSBDebuggerIsValid")
	bind(&LLDBSBDebuggerClear_fn, "LLDBSBDebuggerClear")
	bind(&LLDBSBDebuggerSetAsync_fn, "LLDBSBDebuggerSetAsync")
	bind(&LLDBSBDebuggerGetAsync_fn, "LLDBSBDebuggerGetAsync")
	bind(&LLDBSBDebuggerSkipLLDBInitFiles_fn, "LLDBSBDebuggerSkipLLDBInitFiles")
	bind(&LLDBSBDebuggerSkipAppInitFiles_fn, "LLDBSBDebuggerSkipAppInitFiles")
	bind(&LLDBSBDebuggerHandleCommand_fn, "LLDBSBDebuggerHandleCommand")
	bind(&LLDBSBDebuggerCreateTargetSimple_fn, "LLDBSBDebuggerCreateTargetSimple")
	bind(&LLDBSBDebuggerGetSelectedTarget_fn, "LLDBSBDebuggerGetSelectedTarget")
	bind(&LLDBSBDebuggerGetNumTargets_fn, "LLDBSBDebuggerGetNumTargets")
	bind(&LLDBSBDebuggerGetVersionString_fn, "LLDBSBDebuggerGetVersionString")
	bind(&LLDBSBTargetDelete_fn, "LLDBSBTargetDelete")
	bind(&LLDBSBTargetIsValid_fn, "LLDBSBTargetIsValid")
	bind(&LLDBSBTargetGetProcess_fn, "LLDBSBTargetGetProcess")
	bind(&LLDBSBTargetGetNumModules_fn, "LLDBSBTargetGetNumModules")
	bind(&LLDBSBTargetDeleteAllBreakpoints_fn, "LLDBSBTargetDeleteAllBreakpoints")
	bind(&LLDBSBTargetBreakpointCreateByNameSimple_fn, "LLDBSBTargetBreakpointCreateByNameSimple")
	bind(&LLDBSBTargetBreakpointCreateByLocationSimple_fn, "LLDBSBTargetBreakpointCreateByLocationSimple")
	bind(&LLDBSBProcessDelete_fn, "LLDBSBProcessDelete")
	bind(&LLDBSBProcessIsValid_fn, "LLDBSBProcessIsValid")
	bind(&LLDBSBProcessGetProcessID_fn, "LLDBSBProcessGetProcessID")
	bind(&LLDBSBProcessGetState_fn, "LLDBSBProcessGetState")
	bind(&LLDBSBProcessGetNumThreads_fn, "LLDBSBProcessGetNumThreads")
	bind(&LLDBSBProcessGetThreadAtIndex_fn, "LLDBSBProcessGetThreadAtIndex")
	bind(&LLDBSBProcessGetSelectedThread_fn, "LLDBSBProcessGetSelectedThread")
	bind(&LLDBSBProcessSetSelectedThreadByID_fn, "LLDBSBProcessSetSelectedThreadByID")
	bind(&LLDBSBProcessContinue_fn, "LLDBSBProcessContinue")
	bind(&LLDBSBProcessStop_fn, "LLDBSBProcessStop")
	bind(&LLDBSBProcessKill_fn, "LLDBSBProcessKill")
	bind(&LLDBSBProcessDetachSimple_fn, "LLDBSBProcessDetachSimple")
	bind(&LLDBSBProcessSignal_fn, "LLDBSBProcessSignal")
	bind(&LLDBSBProcessPutSTDIN_fn, "LLDBSBProcessPutSTDIN")
	bind(&LLDBSBProcessGetSTDOUT_fn, "LLDBSBProcessGetSTDOUT")
	bind(&LLDBSBProcessGetSTDERR_fn, "LLDBSBProcessGetSTDERR")
	bind(&LLDBSBProcessReadMemory_fn, "LLDBSBProcessReadMemory")
	bind(&LLDBSBProcessWriteMemory_fn, "LLDBSBProcessWriteMemory")
	bind(&LLDBSBThreadDelete_fn, "LLDBSBThreadDelete")
	bind(&LLDBSBThreadIsValid_fn, "LLDBSBThreadIsValid")
	bind(&LLDBSBThreadGetThreadID_fn, "LLDBSBThreadGetThreadID")
	bind(&LLDBSBThreadGetIndexID_fn, "LLDBSBThreadGetIndexID")
	bind(&LLDBSBThreadGetNumFrames_fn, "LLDBSBThreadGetNumFrames")
	bind(&LLDBSBThreadGetFrameAtIndex_fn, "LLDBSBThreadGetFrameAtIndex")
	bind(&LLDBSBThreadGetStopReason_fn, "LLDBSBThreadGetStopReason")
	bind(&LLDBSBFrameDelete_fn, "LLDBSBFrameDelete")
	bind(&LLDBSBFrameIsValid_fn, "LLDBSBFrameIsValid")
	bind(&LLDBSBFrameGetFunctionName_fn, "LLDBSBFrameGetFunctionName")
	bind(&LLDBSBFrameGetDisplayFunctionName_fn, "LLDBSBFrameGetDisplayFunctionName")
	bind(&LLDBSBFrameGetPC_fn, "LLDBSBFrameGetPC")
	bind(&LLDBSBFrameEvaluateExpressionSimple_fn, "LLDBSBFrameEvaluateExpressionSimple")
	bind(&LLDBSBValueDelete_fn, "LLDBSBValueDelete")
	bind(&LLDBSBValueIsValid_fn, "LLDBSBValueIsValid")
	bind(&LLDBSBValueGetName_fn, "LLDBSBValueGetName")
	bind(&LLDBSBValueGetValue_fn, "LLDBSBValueGetValue")
	bind(&LLDBSBValueGetSummary_fn, "LLDBSBValueGetSummary")
	bind(&LLDBSBValueGetNumChildren_fn, "LLDBSBValueGetNumChildren")
	bind(&LLDBSBValueGetChildAtIndex_fn, "LLDBSBValueGetChildAtIndex")
	bind(&LLDBSBBreakpointDelete_fn, "LLDBSBBreakpointDelete")
	bind(&LLDBSBBreakpointIsValid_fn, "LLDBSBBreakpointIsValid")
	bind(&LLDBSBBreakpointGetID_fn, "LLDBSBBreakpointGetID")
	bind(&LLDBSBBreakpointSetEnabled_fn, "LLDBSBBreakpointSetEnabled")
	bind(&LLDBSBBreakpointIsEnabled_fn, "LLDBSBBreakpointIsEnabled")
}

// clearFuncs drops every bound pointer so a closed library leaves no
// dangling call targets.
func clearFuncs() {
	LLDBLastError_fn = nil
	LLDBSBErrorNew_fn = nil
	LLDBSBErrorDelete_fn = nil
	LLDBSBErrorClear_fn = nil
	LLDBSBErrorFail_fn = nil
	LLDBSBErrorSuccess_fn = nil
	LLDBSBErrorIsValid_fn = nil
	LLDBSBErrorGetError_fn = nil
	LLDBSBErrorGetCString_fn = nil
	LLDBSBDebuggerDelete_fn = nil
	LLDBSBDebuggerInitialize_fn = nil
	LLDBSBDebuggerTerminate_fn = nil
	LLDBSBDebuggerCreateWithInitFiles_fn = nil
	LLDBSBDebuggerDestroy_fn = nil
	LLDBSBDebuggerIsValid_fn = nil
	LLDBSBDebuggerClear_fn = nil
	LLDBSBDebuggerSetAsync_fn = nil
	LLDBSBDebuggerGetAsync_fn = nil
	LLDBSBDebuggerSkipLLDBInitFiles_fn = nil
	LLDBSBDebuggerSkipAppInitFiles_fn = nil
	LLDBSBDebuggerHandleCommand_fn = nil
	LLDBSBDebuggerCreateTargetSimple_fn = nil
	LLDBSBDebuggerGetSelectedTarget_fn = nil
	LLDBSBDebuggerGetNumTargets_fn = nil
	LLDBSBDebuggerGetVersionString_fn = nil
	LLDBSBTargetDelete_fn = nil
	LLDBSBTargetIsValid_fn = nil
	LLDBSBTargetGetProcess_fn = nil
	LLDBSBTargetGetNumModules_fn = nil
	LLDBSBTargetDeleteAllBreakpoints_fn = nil
	LLDBSBTargetBreakpointCreateByNameSimple_fn = nil
	LLDBSBTargetBreakpointCreateByLocationSimple_fn = nil
	LLDBSBProcessDelete_fn = nil
	LLDBSBProcessIsValid_fn = nil
	LLDBSBProcessGetProcessID_fn = nil
	LLDBSBProcessGetState_fn = nil
	LLDBSBProcessGetNumThreads_fn = nil
	LLDBSBProcessGetThreadAtIndex_fn = nil
	LLDBSBProcessGetSelectedThread_fn = nil
	LLDBSBProcessSetSelectedThreadByID_fn = nil
	LLDBSBProcessContinue_fn = nil
	LLDBSBProcessStop_fn = nil
	LLDBSBProcessKill_fn = nil
	LLDBSBProcessDetachSimple_fn = nil
	LLDBSBProcessSignal_fn = nil
	LLDBSBProcessPutSTDIN_fn = nil
	LLDBSBProcessGetSTDOUT_fn = nil
	LLDBSBProcessGetSTDERR_fn = nil
	LLDBSBProcessReadMemory_fn = nil
	LLDBSBProcessWriteMemory_fn = nil
	LLDBSBThreadDelete_fn = nil
	LLDBSBThreadIsValid_fn = nil
	LLDBSBThreadGetThreadID_fn = nil
	LLDBSBThreadGetIndexID_fn = nil
	LLDBSBThreadGetNumFrames_fn = nil
	LLDBSBThreadGetFrameAtIndex_fn = nil
	LLDBSBThreadGetStopReason_fn = nil
	LLDBSBFrameDelete_fn = nil
	LLDBSBFrameIsValid_fn = nil
	LLDBSBFrameGetFunctionName_fn = nil
	LLDBSBFrameGetDisplayFunctionName_fn = nil
	LLDBSBFrameGetPC_fn = nil
	LLDBSBFrameEvaluateExpressionSimple_fn = nil
	LLDBSBValueDelete_fn = nil
	LLDBSBValueIsValid_fn = nil
	LLDBSBValueGetName_fn = nil
	LLDBSBValueGetValue_fn = nil
	LLDBSBValueGetSummary_fn = nil
	LLDBSBValueGetNumChildren_fn = nil
	LLDBSBValueGetChildAtIndex_fn = nil
	LLDBSBBreakpointDelete_fn = nil
	LLDBSBBreakpointIsValid_fn = nil
	LLDBSBBreakpointGetID_fn = nil
	LLDBSBBreakpointSetEnabled_fn = nil
	LLDBSBBreakpointIsEnabled_fn = nil
}
