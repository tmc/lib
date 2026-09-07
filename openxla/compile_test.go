package openxla

import (
	"bytes"
	"testing"
	"unsafe"

	"github.com/ebitengine/purego"
)

func TestCompileDefaults(t *testing.T) {
	for _, tt := range []struct {
		name          string
		options, want []byte
	}{
		{"nil", nil, []byte{26, 4, 32, 1, 40, 1}},
		{"empty", []byte{}, []byte{26, 4, 32, 1, 40, 1}},
		{"explicit", []byte{26, 4, 32, 2, 40, 3}, []byte{26, 4, 32, 2, 40, 3}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			called := false
			api := &Api{}
			api.LoadedExecutable_Destroy = purego.NewCallback(func(*LoadedExecutable_Destroy_Args) *Error { return nil })
			api.Client_Compile = purego.NewCallback(func(a *Client_Compile_Args) *Error {
				called = true
				got := unsafe.Slice(a.Compile_options, a.Compile_options_size)
				if !bytes.Equal(got, tt.want) {
					t.Errorf("compile options=%v want=%v", got, tt.want)
				}
				a.Executable = &LoadedExecutable{}
				return nil
			})
			c := &ClientWrapper{api: api, client: &Client{}}
			e, err := c.Compile([]byte("unused fake program"), "mlir", tt.options)
			if err != nil {
				t.Fatal(err)
			}
			if err := e.Destroy(); err != nil {
				t.Fatal(err)
			}
			if !called {
				t.Fatal("compile did not dispatch")
			}
		})
	}
}
