package cudamem

import (
	"bytes"
	"errors"
	"math"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
)

func TestRoundUpPageAligned(t *testing.T) {
	tests := []struct {
		name    string
		in      int64
		want    int64
		wantErr bool
	}{
		{name: "neg", in: -4096, wantErr: true},
		{name: "zero", in: 0, wantErr: true},
		{name: "one", in: 1, want: 4096},
		{name: "page_minus_one", in: 4095, want: 4096},
		{name: "exact_one_page", in: 4096, want: 4096},
		{name: "page_plus_one", in: 4097, want: 8192},
		{name: "two_pages_exact", in: 8192, want: 8192},
		{name: "large", in: 100_000, want: 102_400},
		{name: "overflow", in: math.MaxInt64, wantErr: true},
		{name: "overflow_boundary", in: math.MaxInt64 - 4094, wantErr: true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := RoundUpPageAligned(tc.in)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("RoundUpPageAligned(%d) = %d, want error", tc.in, got)
				}
				return
			}
			if err != nil {
				t.Fatalf("RoundUpPageAligned(%d) unexpected error: %v", tc.in, err)
			}
			if got != tc.want {
				t.Fatalf("RoundUpPageAligned(%d) = %d, want %d", tc.in, got, tc.want)
			}
		})
	}

	// Exhaustive oracle check over 1..10000 against independent remainder arithmetic.
	checked := 0
	for n := int64(1); n <= 10000; n++ {
		got, err := RoundUpPageAligned(n)
		if err != nil {
			t.Fatalf("RoundUpPageAligned(%d): %v", n, err)
		}
		rem := n % 4096
		want := n
		if rem != 0 {
			want = n + (4096 - rem)
		}
		if got != want || got%4096 != 0 || got < n || got-n >= 4096 {
			t.Fatalf("RoundUpPageAligned(%d) = %d, want %d", n, got, want)
		}
		checked++
	}
	if checked < 1000 {
		t.Fatalf("checked %d cases, want >= 1000", checked)
	}
}

func TestIPCHandleHexRoundTrip(t *testing.T) {
	var zero IPCHandle
	if !zero.IsZero() {
		t.Fatal("zero IPCHandle.IsZero() = false, want true")
	}
	if len(zero.Bytes()) != 64 {
		t.Fatalf("len(zero.Bytes()) = %d, want 64", len(zero.Bytes()))
	}

	var h IPCHandle
	for i := range h {
		h[i] = byte(i*3 + 7)
	}
	if h.IsZero() {
		t.Fatal("non-zero IPCHandle.IsZero() = true, want false")
	}

	// Mutating the slice returned by Bytes() must not mutate h.
	b := h.Bytes()
	b[0] ^= 0xff
	if h[0] == b[0] {
		t.Fatal("IPCHandle.Bytes() returned an aliased slice")
	}

	hexStr := h.Hex()
	if len(hexStr) != 128 {
		t.Fatalf("len(h.Hex()) = %d, want 128", len(hexStr))
	}

	parsed, err := ParseIPCHandleHex(hexStr)
	if err != nil {
		t.Fatalf("ParseIPCHandleHex: %v", err)
	}
	if parsed != h || !bytes.Equal(parsed.Bytes(), h.Bytes()) {
		t.Fatalf("parsed handle mismatch: got %x, want %x", parsed, h)
	}

	// Also support 0x prefix and surrounding whitespace.
	parsedPrefixed, err := ParseIPCHandleHex("  0x" + strings.ToUpper(hexStr) + "\n")
	if err != nil {
		t.Fatalf("ParseIPCHandleHex(0x upper): %v", err)
	}
	if parsedPrefixed != h {
		t.Fatalf("parsedPrefixed = %x, want %x", parsedPrefixed, h)
	}

	// Reject malformed inputs.
	for _, bad := range []string{
		"",
		"0x1234",
		strings.Repeat("ab", 63),
		strings.Repeat("ab", 65),
		strings.Repeat("zz", 64),
	} {
		if _, err := ParseIPCHandleHex(bad); err == nil {
			t.Fatalf("ParseIPCHandleHex(%q) succeeded, want error", bad)
		}
	}
}

func TestIPCDriverOperations(t *testing.T) {
	origAlloc := memAllocPageAlignedFn
	origGet := ipcGetMemHandleFn
	origOpen := ipcOpenMemHandleFn
	origClose := ipcCloseMemHandleFn
	t.Cleanup(func() {
		memAllocPageAlignedFn = origAlloc
		ipcGetMemHandleFn = origGet
		ipcOpenMemHandleFn = origOpen
		ipcCloseMemHandleFn = origClose
	})

	var lastAllocSize int64
	memAllocPageAlignedFn = func(alignedBytes int64) (driver.DevicePtr, error) {
		lastAllocSize = alignedBytes
		return driver.DevicePtr(0xdeadbeef0000), nil
	}

	var exportedPtr driver.DevicePtr
	ipcGetMemHandleFn = func(ptr driver.DevicePtr) (driver.IpcMemHandle, error) {
		exportedPtr = ptr
		var raw driver.IpcMemHandle
		for i := range raw.Reserved {
			raw.Reserved[i] = byte(i + 1)
		}
		return raw, nil
	}

	var openedHandle driver.IpcMemHandle
	var openedFlags driver.IpcmemFlags
	ipcOpenMemHandleFn = func(raw driver.IpcMemHandle, flags driver.IpcmemFlags) (driver.DevicePtr, error) {
		openedHandle = raw
		openedFlags = flags
		return driver.DevicePtr(0xcafebabe0000), nil
	}

	var closedPtr driver.DevicePtr
	ipcCloseMemHandleFn = func(ptr driver.DevicePtr) error {
		closedPtr = ptr
		return nil
	}

	// Validation errors.
	if _, err := AllocPageAligned(0); err == nil {
		t.Fatal("AllocPageAligned(0) succeeded, want error")
	}
	if _, err := ExportIPC(0); err == nil {
		t.Fatal("ExportIPC(0) succeeded, want error")
	}
	if _, err := OpenIPC(IPCHandle{}); err == nil {
		t.Fatal("OpenIPC(zero) succeeded, want error")
	}
	if err := CloseIPC(0); err == nil {
		t.Fatal("CloseIPC(0) succeeded, want error")
	}

	// Positive flow: AllocPageAligned -> ExportIPC -> OpenIPC -> CloseIPC.
	alloc, err := AllocPageAligned(5000)
	if err != nil {
		t.Fatalf("AllocPageAligned(5000): %v", err)
	}
	if alloc.Ptr != 0xdeadbeef0000 || alloc.RequestedBytes != 5000 || alloc.AlignedBytes != 8192 || lastAllocSize != 8192 {
		t.Fatalf("AllocPageAligned(5000) = %+v (lastAllocSize=%d), want AlignedBytes=8192", alloc, lastAllocSize)
	}

	h, err := ExportIPC(alloc.Ptr)
	if err != nil {
		t.Fatalf("ExportIPC: %v", err)
	}
	if exportedPtr != alloc.Ptr || h.IsZero() || h[0] != 1 || h[63] != 64 {
		t.Fatalf("ExportIPC returned handle %x for ptr %#x", h, exportedPtr)
	}

	peerPtr, err := OpenIPC(h)
	if err != nil {
		t.Fatalf("OpenIPC: %v", err)
	}
	if peerPtr != 0xcafebabe0000 || openedFlags != driver.IpcMemLazyEnablePeerAccess || openedHandle.Reserved != [64]byte(h) {
		t.Fatalf("OpenIPC = (ptr=%#x, flags=%v)", peerPtr, openedFlags)
	}

	if err := CloseIPC(peerPtr); err != nil {
		t.Fatalf("CloseIPC: %v", err)
	}
	if closedPtr != peerPtr {
		t.Fatalf("CloseIPC closed %#x, want %#x", closedPtr, peerPtr)
	}

	// Verify driver errors are wrapped cleanly.
	injectedErr := errors.New("driver failure")
	memAllocPageAlignedFn = func(int64) (driver.DevicePtr, error) { return 0, injectedErr }
	if _, err := AllocPageAligned(4096); !errors.Is(err, injectedErr) {
		t.Fatalf("AllocPageAligned error = %v, want wrapping %v", err, injectedErr)
	}
}
