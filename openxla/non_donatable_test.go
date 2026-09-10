package openxla

import (
	"slices"
	"testing"
	"unsafe"

	"github.com/ebitengine/purego"
)

func protectedExecutable(t *testing.T, check func(*LoadedExecutable_Execute_Args)) *ExecutableWrapper {
	t.Helper()
	api := new(Api)
	api.LoadedExecutable_GetExecutable = purego.NewCallback(func(a *LoadedExecutable_GetExecutable_Args) *ErrorHandle { a.Executable = &Executable{}; return nil })
	api.Executable_NumOutputs = purego.NewCallback(func(a *Executable_NumOutputs_Args) *ErrorHandle { a.Num_outputs = 0; return nil })
	api.Executable_Destroy = purego.NewCallback(func(*Executable_Destroy_Args) *ErrorHandle { return nil })
	api.Event_Await = purego.NewCallback(func(*Event_Await_Args) *ErrorHandle { return nil })
	api.Event_Destroy = purego.NewCallback(func(*Event_Destroy_Args) *ErrorHandle { return nil })
	api.LoadedExecutable_Execute = purego.NewCallback(func(a *LoadedExecutable_Execute_Args) *ErrorHandle {
		check(a)
		for i := range unsafe.Slice((**Event)(a.Device_complete_events), a.Num_devices) {
			unsafe.Slice((**Event)(a.Device_complete_events), a.Num_devices)[i] = &Event{}
		}
		return nil
	})
	return &ExecutableWrapper{api: api, executable: &LoadedExecutable{}}
}

func TestExecuteNonDonatableInputs(t *testing.T) {
	for _, indices := range [][]int64{nil, {0}, {1, 0}} {
		t.Run(string(rune('0'+len(indices))), func(t *testing.T) {
			called := 0
			e := protectedExecutable(t, func(a *LoadedExecutable_Execute_Args) {
				called++
				if a.Options.Struct_size != PJRT_ExecuteOptions_STRUCT_SIZE {
					t.Error("incorrect options ABI size")
				}
				got := unsafe.Slice(a.Options.Non_donatable_input_indices, a.Options.Num_non_donatable_input_indices)
				if !slices.Equal(got, indices) || a.Num_devices != 2 || a.Num_args != 2 {
					t.Fatalf("protected=%v devices=%d args=%d", got, a.Num_devices, a.Num_args)
				}
				if len(indices) != 0 && &got[0] == &indices[0] {
					t.Error("native options alias caller index slice")
				}
			})
			b := &BufferWrapper{api: e.api, buffer: &Buffer{}}
			if _, err := e.ExecuteWithNonDonatableInputs([][]*BufferWrapper{{b, b}, {b, b}}, indices); err != nil {
				t.Fatal(err)
			}
			if called != 1 {
				t.Fatalf("execute calls=%d", called)
			}
		})
	}
}

func TestExecuteNonDonatablePreflight(t *testing.T) {
	for _, indices := range [][]int64{{-1}, {1}} {
		e := protectedExecutable(t, func(*LoadedExecutable_Execute_Args) { t.Error("invalid indices dispatched") })
		e.api.LoadedExecutable_GetExecutable = purego.NewCallback(func(*LoadedExecutable_GetExecutable_Args) *ErrorHandle {
			t.Error("invalid indices queried native executable")
			return nil
		})
		b := &BufferWrapper{api: e.api, buffer: &Buffer{}}
		if _, err := e.ExecuteWithNonDonatableInputs([][]*BufferWrapper{{b}}, indices); err == nil {
			t.Fatal("invalid indices succeeded")
		}
		if _, err := e.ExecuteWithNonDonatableInputs(nil, indices); err == nil {
			t.Fatal("indices without devices succeeded")
		}
	}
	var e *ExecutableWrapper
	if _, err := e.ExecuteWithNonDonatableInputs(nil, nil); err == nil {
		t.Fatal("nil executable succeeded")
	}
}
