//go:build linux
// +build linux

package cudss

import (
	"sync"

	"github.com/ebitengine/purego"
)

// A LogFunc receives one cuDSS log record. level is the vendor's log level,
// function is the cuDSS entry point that produced the record, and message is
// the text.
//
// LogFunc is called from a thread owned by libcudss, possibly while another
// goroutine is inside an unrelated cuDSS call. It must not call back into this
// package, and it must not block: the library thread is stopped for its
// duration.
type LogFunc func(level int, function, message string)

var (
	logMu sync.Mutex

	// logFn is the installed callback, read by logCallback on a library
	// thread and written by SetLogFunc.
	logFn LogFunc

	// logTrampoline is the C function pointer handed to
	// cudssLoggerSetCallback. purego allocates it once and never frees it, so
	// it is created at most once per process and reused. Handing the library
	// a new trampoline for every SetLogFunc call would exhaust purego's
	// fixed callback table.
	logTrampoline uintptr
)

// SetLogFunc installs fn as the destination for cuDSS log records and reports
// whether the library accepted it.
//
// The first call registers a C trampoline with the library. That trampoline
// stays registered: SetLogFunc(nil) stops delivery to Go without unregistering
// it, because handing libcudss a null callback pointer is not a documented way
// to turn logging off. [LoggerForceDisable] is the vendor's own off switch,
// and [LoggerSetLevel] and [LoggerSetMask] select what is produced.
//
// fn is retained until it is replaced. Neither fn nor anything it closes over
// may be assumed collectable while it is installed.
func SetLogFunc(fn LogFunc) error {
	logMu.Lock()
	defer logMu.Unlock()
	if logTrampoline == 0 {
		// Register with the library before publishing the trampoline, so a
		// failure leaves nothing installed.
		tramp := purego.NewCallback(logCallback)
		if err := LoggerSetCallback(tramp); err != nil {
			return err
		}
		logTrampoline = tramp
	}
	logFn = fn
	return nil
}

// logCallback is the C entry point for cuDSS log records.
//
// The two string arguments point into memory libcudss owns and reuses; they
// are copied here and are not valid after this function returns. They are
// typed *byte rather than uintptr so that no uintptr is ever converted back
// into a pointer, which is the conversion the collector cannot follow.
func logCallback(level int32, function *byte, message *byte) {
	logMu.Lock()
	fn := logFn
	logMu.Unlock()
	if fn == nil {
		return
	}
	fn(int(level), goString(function), goString(message))
}
