package openxla

import (
	"github.com/ebitengine/purego"
	"strings"
	"testing"
	"unsafe"
)

func TestWaitReadyPreservesErrors(t *testing.T) {
	for _, tt := range []struct {
		name           string
		await, destroy bool
	}{
		{"success", false, false}, {"await", true, false}, {"destroy", false, true}, {"both", true, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var message []byte
			freed, events := 0, 0
			api := &Api{}
			api.Error_Message = purego.NewCallback(func(a *Error_Message_Args) { a.Message = &message[0]; a.Message_size = uintptr(len(message)) })
			api.Error_Destroy = purego.NewCallback(func(*Error_Destroy_Args) { freed++ })
			api.Event_Await = purego.NewCallback(func(*Event_Await_Args) *ErrorHandle {
				if tt.await {
					message = []byte("await failure")
					return &ErrorHandle{}
				}
				return nil
			})
			api.Event_Destroy = purego.NewCallback(func(*Event_Destroy_Args) *ErrorHandle {
				events++
				if tt.destroy {
					message = []byte("destroy failure")
					return &ErrorHandle{}
				}
				return nil
			})
			ready := 0
			api.Buffer_ReadyEvent = purego.NewCallback(func(a *Buffer_ReadyEvent_Args) *ErrorHandle {
				ready++
				a.Event = &Event{}
				return nil
			})
			b := &BufferWrapper{api: api, buffer: &Buffer{}}
			err := b.WaitReady()
			if ready != 1 {
				t.Fatalf("ready calls = %d", ready)
			}
			wantFreed := 0
			if tt.await {
				wantFreed++
				if err == nil || !strings.Contains(err.Error(), "await failure") {
					t.Fatalf("lost await error: %v", err)
				}
			}
			if tt.destroy {
				wantFreed++
				if err == nil || !strings.Contains(err.Error(), "destroy failure") {
					t.Fatalf("lost destroy error: %v", err)
				}
			}
			if wantFreed == 0 && err != nil {
				t.Fatal(err)
			}
			if events != 1 || freed != wantFreed {
				t.Fatalf("events destroyed=%d errors destroyed=%d want=%d", events, freed, wantFreed)
			}
		})
	}
}

func TestSharedBufferPluginErrors(t *testing.T) {
	for _, name := range []string{"cpu", "ready", "view"} {
		t.Run(name, func(t *testing.T) {
			api := new(Api)
			message := []byte("producer failed")
			freed, dispatched := 0, 0
			api.Error_Message = purego.NewCallback(func(a *Error_Message_Args) { a.Message = &message[0]; a.Message_size = uintptr(len(message)) })
			api.Error_Destroy = purego.NewCallback(func(*Error_Destroy_Args) { freed++ })
			api.Buffer_Destroy = purego.NewCallback(func(*Buffer_Destroy_Args) *ErrorHandle { t.Error("unexpected buffer destruction"); return nil })
			api.Event_Await = purego.NewCallback(func(*Event_Await_Args) *ErrorHandle { t.Error("unexpected await"); return nil })
			api.Event_Destroy = purego.NewCallback(func(*Event_Destroy_Args) *ErrorHandle { t.Error("unexpected event destruction"); return nil })
			api.Buffer_IsOnCpu = purego.NewCallback(func(*Buffer_IsOnCpu_Args) *ErrorHandle { dispatched++; return &ErrorHandle{} })
			api.Buffer_ReadyEvent = purego.NewCallback(func(*Buffer_ReadyEvent_Args) *ErrorHandle { dispatched++; return &ErrorHandle{} })
			api.Client_CreateViewOfDeviceBuffer = purego.NewCallback(func(*Client_CreateViewOfDeviceBuffer_Args) *ErrorHandle { dispatched++; return &ErrorHandle{} })
			b := &BufferWrapper{api: api, buffer: &Buffer{}}
			var err error
			switch name {
			case "cpu":
				var got bool
				got, err = b.IsOnCPU()
				if got {
					t.Error("failed query returned true")
				}
			case "ready":
				err = b.WaitReady()
			case "view":
				c := &ClientWrapper{api: api, client: &Client{}}
				var data byte
				var out *BufferWrapper
				out, err = c.CreateViewOfDeviceBuffer(unsafe.Pointer(&data), Buffer_Type_U8, nil, nil, &Device{}, 0, nil)
				if out != nil {
					t.Error("failed view returned buffer")
				}
			}
			if err == nil || !strings.Contains(err.Error(), string(message)) {
				t.Fatalf("error = %v", err)
			}
			if freed != 1 || dispatched != 1 {
				t.Fatalf("freed=%d dispatched=%d", freed, dispatched)
			}
		})
	}
}

func TestWaitReadyPreflight(t *testing.T) {
	for _, name := range []string{"ready", "await", "destroy", "closed", "nil"} {
		t.Run(name, func(t *testing.T) {
			api := new(Api)
			api.Buffer_ReadyEvent = purego.NewCallback(func(*Buffer_ReadyEvent_Args) *ErrorHandle { t.Error("unexpected ready dispatch"); return nil })
			api.Event_Await = purego.NewCallback(func(*Event_Await_Args) *ErrorHandle { t.Error("unexpected await"); return nil })
			api.Event_Destroy = purego.NewCallback(func(*Event_Destroy_Args) *ErrorHandle { t.Error("unexpected destroy"); return nil })
			b := &BufferWrapper{api: api, buffer: &Buffer{}}
			switch name {
			case "ready":
				api.Buffer_ReadyEvent = 0
			case "await":
				api.Event_Await = 0
			case "destroy":
				api.Event_Destroy = 0
			case "closed":
				b.buffer = nil
			case "nil":
				b = nil
			}
			if err := b.WaitReady(); err == nil {
				t.Fatal("preflight succeeded")
			}
		})
	}
}
