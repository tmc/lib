package openxla

import (
	"testing"

	"github.com/ebitengine/purego"
)

func TestBufferWrapperWaitReady(t *testing.T) {
	api := new(Api)
	b := &BufferWrapper{api: api, buffer: &Buffer{}}
	event := &Event{}
	awaited, destroyed := 0, 0
	api.Buffer_ReadyEvent = purego.NewCallback(func(a *Buffer_ReadyEvent_Args) *ErrorHandle {
		if a.Struct_size != PJRT_Buffer_ReadyEvent_Args_STRUCT_SIZE || a.Buffer != b.buffer {
			t.Fatal("bad ready args")
		}
		a.Event = event
		return nil
	})
	api.Event_Await = purego.NewCallback(func(a *Event_Await_Args) *ErrorHandle {
		if a.Event != event {
			t.Fatal("bad event")
		}
		awaited++
		return nil
	})
	api.Event_Destroy = purego.NewCallback(func(a *Event_Destroy_Args) *ErrorHandle { destroyed++; return nil })
	if err := b.WaitReady(); err != nil {
		t.Fatal(err)
	}
	if awaited != 1 || destroyed != 1 {
		t.Fatalf("awaited=%d destroyed=%d", awaited, destroyed)
	}
	api.Buffer_ReadyEvent = 0
	if err := b.WaitReady(); err == nil {
		t.Fatal("missing slot succeeded")
	}
	api.Buffer_ReadyEvent = purego.NewCallback(func(*Buffer_ReadyEvent_Args) *ErrorHandle { return nil })
	if err := b.WaitReady(); err == nil {
		t.Fatal("nil event succeeded")
	}
}
