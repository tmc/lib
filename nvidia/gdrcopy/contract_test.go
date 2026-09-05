//go:build linux
// +build linux

package gdrcopy

import (
	"errors"
	"strings"
	"syscall"
	"testing"
)

// These tests check the parts of the contract this package enforces itself:
// the release order gdrapi.h requires, the argument checks that stand in for
// libgdrapi's missing ones, and what every entry point does when the library
// is not there. None of them calls into C, so all of them run on any Linux
// machine with no GPU, no gdrdrv and no libgdrapi installed.

// seedPin installs a pinning record as if PinBuffer, and optionally Map, had
// created it. The bookkeeping is what the release-order checks read, and it is
// the only way to reach them without a library.
func seedPin(t *testing.T, g Handle, mh MemHandle, va, size uintptr, mapped bool) {
	t.Helper()
	stateMu.Lock()
	pins[mh] = &pinState{g: g, mapped: mapped, va: va, size: size}
	stateMu.Unlock()
	t.Cleanup(func() {
		stateMu.Lock()
		delete(pins, mh)
		stateMu.Unlock()
	})
}

// TestUnpinRejectsMappedHandle checks the rule gdrapi.h states and does not
// enforce: gdr_unmap comes before gdr_unpin_buffer. Unpinning first leaves the
// process holding an mmap of an aperture the driver has released.
func TestUnpinRejectsMappedHandle(t *testing.T) {
	const g, mh = Handle(1), MemHandle(2)
	seedPin(t, g, mh, 0x1000, 0x100, true)
	if err := g.UnpinBuffer(mh); !errors.Is(err, ErrMapped) {
		t.Errorf("UnpinBuffer on a mapped handle = %v, want ErrMapped", err)
	}
}

// TestCloseRejectsMappedHandle checks the same rule for Close. gdr_close unpins
// what is left but does not unmap it.
func TestCloseRejectsMappedHandle(t *testing.T) {
	const g, mh = Handle(3), MemHandle(4)
	seedPin(t, g, mh, 0x1000, 0x100, true)
	if err := g.Close(); !errors.Is(err, ErrMapped) {
		t.Errorf("Close with a live mapping = %v, want ErrMapped", err)
	}
}

// TestCloseIgnoresOtherHandlesMappings checks that one handle's mapping does
// not block a different handle's close.
func TestCloseIgnoresOtherHandlesMappings(t *testing.T) {
	seedPin(t, Handle(5), MemHandle(6), 0x1000, 0x100, true)
	if err := Handle(7).Close(); errors.Is(err, ErrMapped) {
		t.Error("Close was blocked by a mapping belonging to a different handle")
	}
}

// TestUnmapRequiresTheMappedRange checks that Unmap refuses an address or
// length other than the one Map produced. gdr_unmap rounds the length up to a
// host page and hands it to munmap, so a wrong length unmaps memory the caller
// never mapped and libgdrapi reports success.
func TestUnmapRequiresTheMappedRange(t *testing.T) {
	const g, mh = Handle(8), MemHandle(9)
	const va, size = uintptr(0x10000), uintptr(0x1000)
	seedPin(t, g, mh, va, size, true)
	for _, tt := range []struct {
		name     string
		va, size uintptr
	}{
		{"address inside the mapping", va + 0x10, size},
		{"address outside the mapping", va - 0x1000, size},
		{"length longer than the mapping", va, size * 2},
		{"length shorter than the mapping", va, size / 2},
	} {
		if err := g.Unmap(mh, tt.va, tt.size); !errors.Is(err, ErrUnmapRange) {
			t.Errorf("Unmap with %s = %v, want ErrUnmapRange", tt.name, err)
		}
	}
}

// TestUnmapOfUnmappedHandle checks that Unmap on a handle with no mapping is
// reported rather than passed to libgdrapi, which would munmap an address the
// caller invented.
func TestUnmapOfUnmappedHandle(t *testing.T) {
	const g, mh = Handle(10), MemHandle(11)
	seedPin(t, g, mh, 0, 0, false)
	if err := g.Unmap(mh, 0x2000, 0x100); !errors.Is(err, ErrNotMapped) {
		t.Errorf("Unmap on an unmapped handle = %v, want ErrNotMapped", err)
	}
	if err := g.Unmap(MemHandle(12), 0x2000, 0x100); !errors.Is(err, ErrNotMapped) {
		t.Errorf("Unmap on an unknown handle = %v, want ErrNotMapped", err)
	}
}

// TestCopyRange checks that a copy outside the mapping is refused. The copy
// entry points take no gdr_t and check only that the handle is mapped; the
// range is the caller's responsibility, and a wrong one writes over whatever
// the aperture happens to sit next to.
func TestCopyRange(t *testing.T) {
	const mh = MemHandle(13)
	const va, size = uintptr(0x20000), uintptr(0x1000)
	seedPin(t, Handle(14), mh, va, size, true)
	buf := make([]byte, 64)
	for _, tt := range []struct {
		name string
		at   uintptr
		n    int
	}{
		{"before the mapping", va - 1, len(buf)},
		{"after the mapping", va + size, len(buf)},
		{"straddling the end", va + size - 8, len(buf)},
		{"far away", 0, len(buf)},
	} {
		if err := mh.CopyToMapping(tt.at, buf[:tt.n]); !errors.Is(err, ErrCopyRange) {
			t.Errorf("CopyToMapping %s = %v, want ErrCopyRange", tt.name, err)
		}
		if err := mh.CopyFromMapping(buf[:tt.n], tt.at); !errors.Is(err, ErrCopyRange) {
			t.Errorf("CopyFromMapping %s = %v, want ErrCopyRange", tt.name, err)
		}
	}
}

// TestCopyRequiresMapping checks that a copy through a pinned but unmapped
// handle is refused before it reaches C.
func TestCopyRequiresMapping(t *testing.T) {
	const mh = MemHandle(15)
	seedPin(t, Handle(16), mh, 0, 0, false)
	if err := mh.CopyToMapping(0x1000, make([]byte, 8)); !errors.Is(err, ErrNotMapped) {
		t.Errorf("CopyToMapping without a mapping = %v, want ErrNotMapped", err)
	}
}

// TestZeroHandles checks that a zero Handle or MemHandle is reported instead of
// passed to libgdrapi, which dereferences both without checking.
func TestZeroHandles(t *testing.T) {
	for _, tt := range []struct {
		name string
		call func() error
		want error
	}{
		{"Close", func() error { return Handle(0).Close() }, ErrHandle},
		{"PinBuffer", func() error { _, err := Handle(0).PinBuffer(0, 0, 0, 0); return err }, ErrHandle},
		{"PinBufferV2", func() error { _, err := Handle(0).PinBufferV2(0, 0, 0); return err }, ErrHandle},
		{"UnpinBuffer", func() error { return Handle(0).UnpinBuffer(1) }, ErrHandle},
		{"Map", func() error { _, err := Handle(0).Map(1, 0); return err }, ErrHandle},
		{"MapV2", func() error { _, err := Handle(0).MapV2(1, 0, 0); return err }, ErrHandle},
		{"Unmap", func() error { return Handle(0).Unmap(1, 0, 0) }, ErrHandle},
		{"GetInfoV2", func() error { _, err := Handle(0).GetInfoV2(1); return err }, ErrHandle},
		{"GetCallbackFlag", func() error { _, err := Handle(0).GetCallbackFlag(1); return err }, ErrHandle},
		{"GetAttribute", func() error { _, err := Handle(0).GetAttribute(0); return err }, ErrHandle},
		{"DriverGetVersion", func() error { _, _, err := Handle(0).DriverGetVersion(); return err }, ErrHandle},
		{"UnpinBuffer(0)", func() error { return Handle(1).UnpinBuffer(0) }, ErrMemHandle},
		{"Map(0)", func() error { _, err := Handle(1).Map(0, 0); return err }, ErrMemHandle},
		{"Unmap(0)", func() error { return Handle(1).Unmap(0, 0, 0) }, ErrMemHandle},
		{"GetInfoV2(0)", func() error { _, err := Handle(1).GetInfoV2(0); return err }, ErrMemHandle},
		{"CopyToMapping", func() error { return MemHandle(0).CopyToMapping(0, nil) }, ErrMemHandle},
		{"CopyFromMapping", func() error { return MemHandle(0).CopyFromMapping(nil, 0) }, ErrMemHandle},
	} {
		if err := tt.call(); !errors.Is(err, tt.want) {
			t.Errorf("%s with a zero handle = %v, want %v", tt.name, err, tt.want)
		}
	}
}

// TestUnavailableLibrary checks that every entry point reports an error when
// libgdrapi cannot be loaded, rather than panicking on a nil function pointer
// or reporting the zero return value as success. gdrcopy returns 0 for success,
// so a binding that guarded its calls by returning the zero value would claim
// every call succeeded on a machine with no library at all.
func TestUnavailableLibrary(t *testing.T) {
	// Close first: if an earlier test loaded the library, the skip switch is
	// only consulted on the next load attempt.
	if err := Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	t.Setenv("CODEGEN_SKIP_BINDINGS", "1")
	t.Cleanup(func() { _ = Close() })

	const mh = MemHandle(17)
	seedPin(t, Handle(18), mh, 0x30000, 0x1000, true)

	for _, tt := range []struct {
		name string
		call func() error
	}{
		{"Open", func() error { _, err := Open(); return err }},
		{"Close", func() error { return Handle(1).Close() }},
		{"PinBuffer", func() error { _, err := Handle(1).PinBuffer(0x1000, 0x1000, 0, 0); return err }},
		{"PinBufferV2", func() error { _, err := Handle(1).PinBufferV2(0x1000, 0x1000, PinFlagDefault); return err }},
		{"UnpinBuffer", func() error { return Handle(1).UnpinBuffer(MemHandle(19)) }},
		{"Map", func() error { _, err := Handle(1).Map(MemHandle(19), 0x1000); return err }},
		{"MapV2", func() error { _, err := Handle(1).MapV2(MemHandle(19), 0x1000, MapFlagDefault); return err }},
		{"Unmap", func() error { return Handle(18).Unmap(mh, 0x30000, 0x1000) }},
		{"GetInfoV2", func() error { _, err := Handle(1).GetInfoV2(MemHandle(19)); return err }},
		{"GetCallbackFlag", func() error { _, err := Handle(1).GetCallbackFlag(MemHandle(19)); return err }},
		{"GetAttribute", func() error { _, err := Handle(1).GetAttribute(AttrUsePersistentMapping); return err }},
		{"DriverGetVersion", func() error { _, _, err := Handle(1).DriverGetVersion(); return err }},
		{"RuntimeGetVersion", func() error { _, _, err := RuntimeGetVersion(); return err }},
		{"MappingTypeString", func() error { _, err := MappingTypeString(MappingTypeWc); return err }},
		{"CopyToMapping", func() error { return mh.CopyToMapping(0x30000, make([]byte, 8)) }},
		{"CopyFromMapping", func() error { return mh.CopyFromMapping(make([]byte, 8), 0x30000) }},
	} {
		if err := tt.call(); err == nil {
			t.Errorf("%s reported success with no library loaded", tt.name)
		}
	}
	if Loaded() {
		t.Error("Loaded reported true after every load attempt failed")
	}
}

// TestErrorIsErrno checks that a libgdrapi return code can be compared against
// the POSIX error it is. gdrcopy has no status enum, so this is how a caller
// distinguishes, say, an unsupported flag from a missing device.
func TestErrorIsErrno(t *testing.T) {
	err := toError("gdr_pin_buffer_v2", int32(syscall.EINVAL))
	if err == nil {
		t.Fatal("toError returned nil for a nonzero return code")
	}
	if !errors.Is(err, syscall.EINVAL) {
		t.Errorf("errors.Is(%v, EINVAL) = false", err)
	}
	var gdrErr *Error
	if !errors.As(err, &gdrErr) {
		t.Fatalf("error is %T, want *Error", err)
	}
	if gdrErr.Op != "gdr_pin_buffer_v2" {
		t.Errorf("Error.Op = %q, want the C entry point", gdrErr.Op)
	}
	if !strings.Contains(err.Error(), "gdr_pin_buffer_v2") {
		t.Errorf("error message %q does not name the call", err.Error())
	}
	if toError("gdr_close", 0) != nil {
		t.Error("toError reported an error for return code 0, which is success")
	}
}

// TestInfoOffset checks the arithmetic a caller needs after a pin: the kernel
// module aligns the pinning down to a GPU page, so the data pinned is at an
// offset inside the mapping rather than at its base.
func TestInfoOffset(t *testing.T) {
	const base = uintptr(0x7f0000000000)
	info := Info{Va: uint64(base), MappedSize: uint64(2 * GPUPageSize)}
	for _, tt := range []struct {
		name string
		addr uintptr
		want uintptr
		ok   bool
	}{
		{"the base itself", base, 0, true},
		{"inside the first page", base + 0x40, 0x40, true},
		{"the last byte", base + 2*GPUPageSize - 1, 2*GPUPageSize - 1, true},
		{"one before the base", base - 1, 0, false},
		{"one past the end", base + 2*GPUPageSize, 0, false},
	} {
		got, ok := info.Offset(tt.addr)
		if ok != tt.ok || got != tt.want {
			t.Errorf("Offset(%s) = (%d, %v), want (%d, %v)", tt.name, got, ok, tt.want, tt.ok)
		}
	}
}

// TestPageAlignment checks GPUPageMask against the alignment the kernel module
// performs, using the worked example from gdrapi.h's own warning: an address
// partway into a page is pinned from the start of that page.
func TestPageAlignment(t *testing.T) {
	const addr = uintptr(0x7f0000012345)
	base := addr & GPUPageMask
	if base > addr || addr-base >= GPUPageSize {
		t.Fatalf("aligning %#x down gave %#x, which is not the start of its GPU page", addr, base)
	}
	if base&(GPUPageSize-1) != 0 {
		t.Errorf("aligned base %#x is not a multiple of the GPU page size", base)
	}
	info := Info{Va: uint64(base), MappedSize: uint64(GPUPageSize)}
	off, ok := info.Offset(addr)
	if !ok {
		t.Fatal("the pinned address is reported outside its own pinning")
	}
	if off != addr-base {
		t.Errorf("Offset = %#x, want %#x", off, addr-base)
	}
}
