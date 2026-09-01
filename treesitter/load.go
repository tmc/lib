package treesitter

import (
	"strings"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"

	"github.com/tmc/lib/treesitter/internal/ts"
)

// LanguageVersion is the latest grammar ABI version this library
// understands, and MinCompatibleLanguageVersion the earliest.
const (
	LanguageVersion              = 15
	MinCompatibleLanguageVersion = 13
)

// Load opens libtree-sitter from path, or from the default search list when
// path is empty. Calling Load is optional: the first operation that needs
// the library loads it from the default search list.
func Load(path string) error {
	if err := ts.Load(path); err != nil {
		return err
	}
	return loadLibc()
}

// Loaded reports whether libtree-sitter is loaded.
func Loaded() bool {
	return ts.Loaded()
}

// LibraryPath returns the path libtree-sitter was opened from, or "" if it
// is not loaded.
func LibraryPath() string {
	return ts.Path()
}

// load loads the library if needed and returns any load error.
func load() error {
	return Load("")
}

var (
	libcOnce sync.Once
	libcErr  error
	cFree    func(uintptr)
)

// loadLibc binds free from the C library so that strings and arrays
// tree-sitter allocates for the caller can be released.
func loadLibc() error {
	libcOnce.Do(func() {
		sym, err := purego.Dlsym(purego.RTLD_DEFAULT, "free")
		if err != nil {
			libcErr = err
			return
		}
		purego.RegisterFunc(&cFree, sym)
	})
	return libcErr
}

// cptr converts an address returned by C into a pointer. The memory is not
// Go memory, which is what the plain conversion vet flags would imply.
func cptr(p uintptr) unsafe.Pointer {
	return *(*unsafe.Pointer)(unsafe.Pointer(&p))
}

// goString copies a NUL-terminated C string.
func goString(p uintptr) string {
	if p == 0 {
		return ""
	}
	n := 0
	for *(*byte)(cptr(p + uintptr(n))) != 0 {
		n++
	}
	return strings.Clone(unsafe.String((*byte)(cptr(p)), n))
}

// goStringN copies n bytes of C memory.
func goStringN(p uintptr, n uint32) string {
	if p == 0 || n == 0 {
		return ""
	}
	return strings.Clone(unsafe.String((*byte)(cptr(p)), int(n)))
}

// bytesPtr returns a pointer to the first byte of b, or nil when b is
// empty. tree-sitter reads the memory only for the duration of the call.
func bytesPtr(b []byte) *byte {
	if len(b) == 0 {
		return nil
	}
	return unsafe.SliceData(b)
}

// stringPtr returns a pointer to the bytes of s, or nil when s is empty.
func stringPtr(s string) *byte {
	if s == "" {
		return nil
	}
	return unsafe.StringData(s)
}
