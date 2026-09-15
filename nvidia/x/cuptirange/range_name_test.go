package cuptirange

import (
	"bytes"
	"runtime"
	"strings"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/cupti"
)

func TestRangeName(t *testing.T) {
	for _, tt := range []struct {
		name      string
		data      []byte
		want, err string
	}{
		{"nil", nil, "", "missing"},
		{"empty", []byte{0}, "", "empty"},
		{"terminated", []byte("kernel\x00ignored"), "kernel", ""},
		{"terminator at bound", append(bytes.Repeat([]byte{'x'}, (1<<20)-1), 0), strings.Repeat("x", (1<<20)-1), ""},
		{"unterminated", bytes.Repeat([]byte{'x'}, 1<<20), "", "exceeds limit"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			// Emulate the native char* output slot without changing generated ABI.
			var params cupti.RangeProfilerCounterDataGetRangeInfoParams
			var pin runtime.Pinner
			defer pin.Unpin()
			if len(tt.data) > 0 {
				pin.Pin(&tt.data[0])
				params.RangeName = uintptr(unsafe.Pointer(&tt.data[0]))
			}
			got, err := rangeName(*(**byte)(unsafe.Pointer(&params.RangeName)))
			runtime.KeepAlive(tt.data)
			if tt.err != "" {
				if err == nil || !strings.Contains(err.Error(), tt.err) {
					t.Fatalf("error=%v want=%s", err, tt.err)
				}
				return
			}
			if err != nil || got != tt.want {
				t.Fatalf("length=%d error=%v want length=%d", len(got), err, len(tt.want))
			}
		})
	}
}
