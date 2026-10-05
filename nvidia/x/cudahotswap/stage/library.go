package stage

import (
	"crypto/sha256"
	"errors"
	"fmt"
	"os"
	"reflect"
	"runtime"
	"sync"
	"unsafe"
)

// ErrPinned means a library still has retained calls or submitted device work.
// Release the pins only after actual completion, then retry Close if necessary.
var ErrPinned = errors.New("stage library is pinned")

// Library owns one loaded host library. Close refuses new pins; existing pins
// remain usable until released. Failed unload retains its handle for retry.
// Its zero value is closed. A Library must not be copied after use.
type Library struct {
	mu       sync.Mutex
	handle   uintptr
	entry    func(*Context) int32
	ready    bool
	closing  bool
	pins     uint64
	close    func(uintptr) error
	artifact *Artifact
}

// Load verifies output identity and compiler-evaluated ABI before admitting
// calls. A nonnil result with an error retains an unload owner; Close it before
// removing the artifact. Loading runs host library constructors, not a sandbox.
func Load(artifact *Artifact) (*Library, error) {
	if artifact == nil || artifact.path == "" || artifact.generation == 0 {
		return nil, fmt.Errorf("load stage: invalid artifact")
	}
	info, err := os.Lstat(artifact.path)
	if err != nil || info == nil || !info.Mode().IsRegular() {
		return nil, fmt.Errorf("load stage: artifact must be a regular file")
	}
	image, err := readCompilerImage(artifact.path)
	if err != nil {
		return nil, fmt.Errorf("read stage library: %w", err)
	}
	if sha256.Sum256(image) != artifact.digest {
		return nil, fmt.Errorf("load stage: compiler output changed")
	}
	handle, err := openLibrary(artifact.path)
	if err != nil {
		return nil, fmt.Errorf("open stage library: %w", err)
	}
	l := &Library{handle: handle, close: closeLibrary, artifact: artifact}
	fail := func(err error) (*Library, error) { return l, errors.Join(err, l.Close()) }
	// The second identity check refuses a changed path during load; callers
	// must still protect the owned artifact from concurrent modification.
	image, err = readCompilerImage(artifact.path)
	if err != nil || sha256.Sum256(image) != artifact.digest {
		return fail(fmt.Errorf("load stage: compiler output changed during load"))
	}
	if err := bindEntry(handle, "stage", &l.entry); err != nil {
		return fail(err)
	}
	var abi func(*abiValues) int32
	if err := bindEntry(handle, "stage_abi", &abi); err != nil {
		return fail(err)
	}
	want := expectedABI()
	got := abiValues{Count: uint64(len(want))}
	if result := abi(&got); result != 0 {
		return fail(fmt.Errorf("stage ABI witness returned %d", result))
	}
	for i, w := range want {
		if got.Values[i] != w {
			return fail(fmt.Errorf("stage ABI field %d has %d; Go requires %d", i, got.Values[i], w))
		}
	}
	l.ready = true
	return l, nil
}

// Pin retains the loaded entry until Release. It performs no CUDA work.
func (l *Library) Pin() (*Pin, error) {
	if l == nil {
		return nil, fmt.Errorf("pin stage: nil library")
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.handle == 0 || !l.ready || l.closing {
		return nil, fmt.Errorf("pin stage: library is unavailable")
	}
	if l.pins == ^uint64(0) {
		return nil, fmt.Errorf("pin stage: pin counter exhausted")
	}
	l.pins++
	return &Pin{library: l}, nil
}

// Pin retains a host stage generation. It does not itself retain the device
// allocations, stream or CUDA function pins supplied in Context.
type Pin struct {
	mu      sync.Mutex
	library *Library
}

// Call invokes int stage(stage_ctx*) synchronously on the host. Success means
// only that host submission returned zero; it does not establish GPU completion.
// Keep this pin and every Context dependency until actual stream completion.
func (p *Pin) Call(ctx *Context) error {
	if p == nil {
		return fmt.Errorf("call stage: nil pin")
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.library == nil {
		return fmt.Errorf("call stage: released pin")
	}
	if ctx == nil || ctx.Version != abiVersion || ctx.Size != uint32(unsafe.Sizeof(Context{})) || ctx.BufferCount > 64 || ctx.FunctionCount > 64 || ctx.BufferCount != 0 && ctx.Buffers == nil || ctx.FunctionCount != 0 && ctx.Functions == nil {
		return fmt.Errorf("call stage: invalid bounded context")
	}
	result := p.library.entry(ctx)
	runtime.KeepAlive(ctx)
	if result != 0 {
		return fmt.Errorf("stage entry returned %d", result)
	}
	return nil
}

// NewContext creates the fixed versioned descriptor. buffers/functions and
// their names remain caller-owned; their addresses must not move during Call.
func NewContext(buffers []Buffer, stream uintptr, functions []Function) (*Context, error) {
	if len(buffers) > 64 || len(functions) > 64 {
		return nil, fmt.Errorf("stage context exceeds 64 named buffers or functions")
	}
	return &Context{Version: abiVersion, Size: uint32(unsafe.Sizeof(Context{})), Buffers: unsafe.SliceData(buffers), BufferCount: uint64(len(buffers)), Stream: stream, Functions: unsafe.SliceData(functions), FunctionCount: uint64(len(functions))}, nil
}

// Release drops the library pin. It must follow actual completion even when
// Call failed. An unload failure leaves the Library available for Close retry.
func (p *Pin) Release() error {
	if p == nil {
		return nil
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	l := p.library
	if l == nil {
		return nil
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	l.pins--
	p.library = nil
	if l.closing && l.pins == 0 {
		return l.closeLocked()
	}
	return nil
}

// Close refuses new pins and unloads only after all pins are released. It does
// not delete retained compiler artifacts. An error permits an explicit retry.
func (l *Library) Close() error {
	if l == nil {
		return nil
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	l.closing = true
	return l.closeLocked()
}

func (l *Library) closeLocked() error {
	if l.handle == 0 {
		return nil
	}
	if l.pins != 0 {
		return ErrPinned
	}
	if err := l.close(l.handle); err != nil {
		return fmt.Errorf("unload stage library: %w", err)
	}
	l.handle, l.entry, l.ready = 0, nil, false
	return nil
}

func entryType(pointer any) (reflect.Type, error) {
	t := reflect.TypeOf(pointer)
	if t == nil || t.Kind() != reflect.Pointer {
		return nil, fmt.Errorf("bind stage entry: expected function pointer")
	}
	t = t.Elem()
	return t, checkEntrySignature(t)
}
