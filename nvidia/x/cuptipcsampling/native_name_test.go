package cuptipcsampling

import (
	"runtime"
	"testing"
	"unsafe"
)

func TestReadSamplingName(t *testing.T) {
	for _, tt := range []struct {
		name      string
		data      []byte
		limit     int
		want      string
		wantError bool
	}{
		{"terminated", []byte("kernel\x00"), 7, "kernel", false},
		{"last byte", []byte("abc\x00"), 4, "abc", false},
		{"bound", []byte("abc\x00"), 3, "", true},
		{"empty", []byte{0}, 1, "", false},
		{"zero limit", []byte{0}, 0, "", true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got, err := readSamplingName(uintptr(unsafe.Pointer(&tt.data[0])), tt.limit)
			runtime.KeepAlive(tt.data)
			if (err != nil) != tt.wantError || got != tt.want {
				t.Fatalf("got=%q err=%v", got, err)
			}
			if len(tt.data) > 1 {
				tt.data[0] = 'X'
				if got != tt.want {
					t.Fatal("borrowed native string")
				}
			}
		})
	}
	for _, ptr := range []uintptr{0, ^uintptr(0)} {
		if _, err := readSamplingName(ptr, 2); err == nil {
			t.Fatal("accepted invalid address bounds")
		}
	}
}
