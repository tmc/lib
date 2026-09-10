package openxla

import (
	"strings"
	"testing"
	"unsafe"

	"github.com/ebitengine/purego"
)

func TestBufferFromHostMalformedSuccessGuard(t *testing.T) {
	for _, wantBuffer := range []bool{false, true} {
		t.Run(map[bool]string{false: "nil", true: "valid"}[wantBuffer], func(t *testing.T) {
			awaited, destroyed := false, false
			api := &Api{}
			api.Buffer_Destroy = purego.NewCallback(func(*Buffer_Destroy_Args) *Error { return nil })
			api.Event_Await = purego.NewCallback(func(*Event_Await_Args) *Error { awaited = true; return nil })
			api.Event_Destroy = purego.NewCallback(func(*Event_Destroy_Args) *Error { destroyed = true; return nil })
			api.Client_BufferFromHostBuffer = purego.NewCallback(func(a *Client_BufferFromHostBuffer_Args) *Error {
				a.Done_with_host_buffer = &Event{}
				if wantBuffer {
					a.Buffer = &Buffer{}
				}
				return nil
			})

			var value int32
			buf, err := (&ClientWrapper{api: api, client: &Client{}}).BufferFromHost(
				unsafe.Pointer(&value), BufferTypeS32, []int64{1}, &Device{})
			if wantBuffer {
				if err != nil || buf == nil {
					t.Fatalf("valid output: buf=%v err=%v", buf, err)
				}
			} else {
				if err == nil || buf != nil || !strings.Contains(err.Error(), "returned nil buffer") {
					t.Fatalf("malformed output: buf=%v err=%v", buf, err)
				}
			}
			if !awaited || !destroyed {
				t.Fatalf("event cleanup: awaited=%v destroyed=%v", awaited, destroyed)
			}
		})
	}
}

func TestCompileMalformedSuccessGuard(t *testing.T) {
	for _, wantExecutable := range []bool{false, true} {
		t.Run(map[bool]string{false: "nil", true: "valid"}[wantExecutable], func(t *testing.T) {
			api := &Api{}
			api.LoadedExecutable_Destroy = purego.NewCallback(func(*LoadedExecutable_Destroy_Args) *Error { return nil })
			api.Client_Compile = purego.NewCallback(func(a *Client_Compile_Args) *Error {
				if wantExecutable {
					a.Executable = &LoadedExecutable{}
				}
				return nil
			})

			executable, err := (&ClientWrapper{api: api, client: &Client{}}).Compile([]byte("module"), "mlir", nil)
			if wantExecutable {
				if err != nil || executable == nil {
					t.Fatalf("valid output: executable=%v err=%v", executable, err)
				}
			} else if err == nil || executable != nil || !strings.Contains(err.Error(), "returned nil executable") {
				t.Fatalf("malformed output: executable=%v err=%v", executable, err)
			}
		})
	}
}
