package openxla

import (
	"strings"
	"testing"

	"github.com/ebitengine/purego"
)

func TestBufferWrapperIsOnCPU(t *testing.T) {
	raw := &Buffer{}
	api := new(Api)
	b := &BufferWrapper{api: api, buffer: raw}
	for _, want := range []bool{false, true} {
		api.Buffer_IsOnCpu = purego.NewCallback(func(a *Buffer_IsOnCpu_Args) *ErrorHandle {
			if a.Struct_size != PJRT_Buffer_IsOnCpu_Args_STRUCT_SIZE || a.Buffer != raw {
				t.Fatal("invalid IsOnCpu arguments")
			}
			a.Is_on_cpu = want
			return nil
		})
		got, err := b.IsOnCPU()
		if err != nil || got != want {
			t.Fatalf("IsOnCPU() = %v, %v; want %v, nil", got, err, want)
		}
	}
	api.Buffer_IsOnCpu = 0
	if _, err := b.IsOnCPU(); err == nil || !strings.Contains(err.Error(), "not available") {
		t.Fatalf("missing slot error = %v", err)
	}
	if _, err := (&BufferWrapper{}).IsOnCPU(); err == nil || !strings.Contains(err.Error(), "closed") {
		t.Fatalf("closed error = %v", err)
	}
}
