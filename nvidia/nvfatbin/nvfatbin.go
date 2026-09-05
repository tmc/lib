package nvfatbin

import (
	"runtime"
	"unsafe"
)

// AddPTXString adds PTX source text to the fatbinary.
//
// It exists because nvFatbinAddPTX's size is documented to include the
// terminating NUL, and a Go string has none. Passing len(code) where the
// library expects len(code)+1 is not an error the library reports: the header
// says it appends the NUL itself, copying the code to do so, so the mistake
// costs a copy and silently changes what was packaged. This function appends
// the NUL and passes the inclusive length, which is the only spelling of the
// call that is right by construction.
//
// arch is the numeric architecture, the XX of sm_XX, lto_XX or compute_XX --
// "90", not "sm_90". identifier names the entry for tools like cuobjdump.
// options are the ptxas options applied during JIT compilation, as one
// command-line string; pass "" for none.
//
// Use [Handle.AddPTX] directly for PTX that is already NUL-terminated in
// memory the caller owns, or that is not a Go string.
func (h Handle) AddPTXString(code, arch, identifier, options string) error {
	// The NUL is appended by copying into a slice rather than relying on Go's
	// strings being NUL-terminated in practice: they are not.
	buf := make([]byte, len(code)+1)
	copy(buf, code)
	err := h.AddPTX(unsafe.Pointer(unsafe.SliceData(buf)), uint(len(buf)), arch, identifier, options)
	// buf must outlive the call. unsafe.Pointer(unsafe.SliceData(buf)) is not a
	// reference the collector tracks past the argument evaluation.
	runtime.KeepAlive(buf)
	return err
}
