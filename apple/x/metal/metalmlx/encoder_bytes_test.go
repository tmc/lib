package metalmlx

import (
	"strings"
	"testing"
)

// TestValueBytesRejectsWordSized checks that the argument packers refuse Go's
// int and uint, whose 64-bit width does not match MSL's 32-bit int and uint.
func TestValueBytesRejectsWordSized(t *testing.T) {
	tests := []struct {
		name string
		pack func() error
	}{
		{"int", func() error { i := 1; _, err := ValueBytes(&i); return err }},
		{"uint", func() error { u := uint(1); _, err := ValueBytes(&u); return err }},
		{"uintptr", func() error { p := uintptr(1); _, err := ValueBytes(&p); return err }},
		{"int slice", func() error { _, err := SliceBytes([]int{1}); return err }},
		{"uint slice", func() error { _, err := SliceBytes([]uint{1}); return err }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.pack()
			if err == nil {
				t.Fatal("packing succeeded, want error")
			}
			if !strings.Contains(err.Error(), "fixed-width") {
				t.Errorf("error %q does not explain the fix", err)
			}
		})
	}
}

// TestValueBytesFixedWidth checks the widths the packers do accept.
func TestValueBytesFixedWidth(t *testing.T) {
	tests := []struct {
		name string
		pack func() ([]byte, error)
		want int
	}{
		{"uint32", func() ([]byte, error) { return ValueBytes(new(uint32)) }, 4},
		{"float32", func() ([]byte, error) { return ValueBytes(new(float32)) }, 4},
		{"int64", func() ([]byte, error) { return ValueBytes(new(int64)) }, 8},
		{"struct", func() ([]byte, error) { return ValueBytes(&struct{ A, B uint32 }{}) }, 8},
		{"empty slice", func() ([]byte, error) { return SliceBytes([]uint32(nil)) }, 0},
		{"slice", func() ([]byte, error) { return SliceBytes([]uint32{1, 2, 3}) }, 12},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tt.pack()
			if err != nil {
				t.Fatal(err)
			}
			if len(got) != tt.want {
				t.Errorf("got %d bytes, want %d", len(got), tt.want)
			}
		})
	}
}

func TestValueBytesRejectsNil(t *testing.T) {
	var p *uint32
	if _, err := ValueBytes(p); err == nil {
		t.Fatal("ValueBytes(nil) succeeded")
	}
}
