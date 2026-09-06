package cudnn

import (
	"errors"
	"testing"
	"unsafe"
)

func TestDirectReturns(t *testing.T) {
	oldReady, oldPresent := ready.Load(), present
	oldVersion, oldMax, oldRuntime, oldLast := fnGetVersion, fnGetMaxDeviceVersion, fnGetCudartVersion, fnGetLastErrorString
	defer func() {
		ready.Store(oldReady)
		present = oldPresent
		fnGetVersion, fnGetMaxDeviceVersion, fnGetCudartVersion, fnGetLastErrorString = oldVersion, oldMax, oldRuntime, oldLast
	}()
	ready.Store(true)
	present = map[string]bool{}
	fnGetVersion, fnGetMaxDeviceVersion, fnGetCudartVersion, fnGetLastErrorString = nil, nil, nil, nil
	for _, tt := range []struct {
		name string
		call func() error
	}{
		{"cudnnGetVersion", func() error { _, e := GetVersion(); return e }},
		{"cudnnGetMaxDeviceVersion", func() error { _, e := GetMaxDeviceVersion(); return e }},
		{"cudnnGetCudartVersion", func() error { _, e := GetCudartVersion(); return e }},
		{"cudnnGetLastErrorString", func() error { return GetLastErrorString(nil, 0) }},
	} {
		err := tt.call()
		var absent *SymbolError
		if !errors.As(err, &absent) || absent.Symbol != tt.name {
			t.Errorf("%s: %v, want named SymbolError", tt.name, err)
		}
		present[tt.name] = true
	}
	// Preserve the full size_t width rather than converting it to a status or int32.
	want := ^uint(0) - 17
	fnGetVersion = func() uint { return want }
	fnGetMaxDeviceVersion = func() uint { return want - 1 }
	fnGetCudartVersion = func() uint { return want - 2 }
	for i, f := range []func() (uint, error){GetVersion, GetMaxDeviceVersion, GetCudartVersion} {
		if v, e := f(); e != nil || v != want-uint(i) {
			t.Errorf("getter %d = %d,%v", i, v, e)
		}
	}
	var buf [6]byte
	buf[0], buf[5] = 0xa5, 0x5a
	called := false
	fnGetLastErrorString = func(p *byte, n uint) {
		called = true
		if n != 4 {
			t.Fatalf("buffer capacity=%d", n)
		}
		copy(unsafe.Slice(p, int(n)), []byte{'o', 'k', 0, 0})
	}
	if err := GetLastErrorString(&buf[1], 4); err != nil {
		t.Fatal(err)
	}
	if !called || buf != [6]byte{0xa5, 'o', 'k', 0, 0, 0x5a} {
		t.Fatalf("buffer call=%v bytes=%v", called, buf)
	}
}
