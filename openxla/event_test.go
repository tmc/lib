package openxla

import (
	"strings"
	"testing"
	"unsafe"

	"github.com/ebitengine/purego"
)

func TestAwaitEventPreservesErrors(t *testing.T) {
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
			err := awaitEvent(api, &Event{})
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

func TestExecuteDrainsDevicesOnFailure(t *testing.T) {
	api := &Api{}
	awaited, events, buffers, freed := 0, 0, 0, 0
	var message []byte
	api.Error_Message = purego.NewCallback(func(a *Error_Message_Args) { a.Message = &message[0]; a.Message_size = uintptr(len(message)) })
	api.Error_Destroy = purego.NewCallback(func(*Error_Destroy_Args) { freed++ })
	api.LoadedExecutable_GetExecutable = purego.NewCallback(func(a *LoadedExecutable_GetExecutable_Args) *ErrorHandle { a.Executable = &Executable{}; return nil })
	api.Executable_NumOutputs = purego.NewCallback(func(a *Executable_NumOutputs_Args) *ErrorHandle { a.Num_outputs = 1; return nil })
	api.Executable_Destroy = purego.NewCallback(func(*Executable_Destroy_Args) *ErrorHandle { return nil })
	api.LoadedExecutable_Execute = purego.NewCallback(func(a *LoadedExecutable_Execute_Args) *ErrorHandle {
		out := unsafe.Slice((***Buffer)(a.Output_lists), a.Num_devices)
		events := unsafe.Slice((**Event)(a.Device_complete_events), a.Num_devices)
		for i := range out {
			*out[i] = &Buffer{}
			events[i] = &Event{}
		}
		return nil
	})
	api.Event_Await = purego.NewCallback(func(*Event_Await_Args) *ErrorHandle {
		awaited++
		if awaited == 1 {
			message = []byte("first device failed")
			return &ErrorHandle{}
		}
		return nil
	})
	api.Event_Destroy = purego.NewCallback(func(*Event_Destroy_Args) *ErrorHandle {
		events++
		if events == 2 {
			message = []byte("second event release failed")
			return &ErrorHandle{}
		}
		return nil
	})
	api.Buffer_Destroy = purego.NewCallback(func(*Buffer_Destroy_Args) *ErrorHandle { buffers++; return nil })
	e := &ExecutableWrapper{api: api, executable: &LoadedExecutable{}}
	out, err := e.Execute([][]*BufferWrapper{nil, nil})
	if out != nil || err == nil || !strings.Contains(err.Error(), "first device failed") || !strings.Contains(err.Error(), "second event release failed") {
		t.Fatalf("outputs=%v err=%v", out, err)
	}
	if awaited != 2 || events != 2 || buffers != 2 || freed != 2 {
		t.Fatalf("awaited=%d events=%d buffers=%d errors=%d", awaited, events, buffers, freed)
	}
}
